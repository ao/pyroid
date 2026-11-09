//! `pyroid.nemo`: NeMo Curator 1.4.0's heuristic filter scores, string modifiers and fastText stage helpers.
//!
//! Each score function is one `DocumentFilter.score_document`, run detached on the shared
//! pool. A str gives a Python float or int; a list, tuple or Arrow column gives a numpy
//! float64 or int64 array. Where NeMo raises for a document, the first such document (in
//! order) raises the same exception type, with a note naming its index; an Arrow null
//! raises ValueError (NeMo needs a str for every document).

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{
    PyImportError, PyIndexError, PyNotImplementedError, PyUnicodeDecodeError, PyValueError,
    PyZeroDivisionError,
};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyList, PyString, PyTuple};
use pyroid_core::fasttext::PredictError;
use pyroid_core::json_repair::dumps::float_repr;
use pyroid_core::nemo::fasttext as nft;
use pyroid_core::nemo::filters as f;
use pyroid_core::nemo::pyhash::{self, HashKey};
use pyroid_core::nemo::repetition::{self as r, TopNGram};
use pyroid_core::nemo::text::{ScoreError, Splitter};

use crate::batch::{DocCost, map_values_cost};
use crate::errors::with_note;
use crate::fasttext::{Model, predict_error};
use crate::input::{TextInput, ThreadsArg, texts_and_threads};

static NUMPY: PyOnceLock<()> = PyOnceLock::new();

/// rust-numpy needs numpy at the first array: fail with a clear ImportError instead.
fn require_numpy(py: Python<'_>) -> PyResult<()> {
    NUMPY
        .get_or_try_init(py, || {
            py.import("numpy").map(|_| ()).map_err(|e| {
                let err = PyImportError::new_err(
                    "pyroid.nemo returns numpy arrays for batches: pip install numpy",
                );
                err.set_cause(py, Some(e));
                err
            })
        })
        .copied()
}

fn score_error(e: ScoreError) -> PyErr {
    match e {
        ScoreError::ZeroDivision => PyZeroDivisionError::new_err(e.message()),
        ScoreError::EmptyMax => PyValueError::new_err(e.message()),
    }
}

/// `err` raised for document `i`: a batch's error says which document it was.
fn at_index(py: Python<'_>, input: &TextInput, i: usize, err: PyErr) -> PyErr {
    if matches!(input, TextInput::Str(_)) {
        err
    } else {
        with_note(
            py,
            err,
            format!("pyroid: raised for the document at index {i}"),
        )
    }
}

fn null_error(i: usize) -> PyErr {
    PyValueError::new_err(format!(
        "pyroid.nemo: the document at index {i} is null; NeMo Curator's filters need a str for \
         every document"
    ))
}

/// Every document's value in order, or the exception of the first one that has none (an
/// Arrow null, or the error NeMo raises for it).
fn values<T, E>(
    py: Python<'_>,
    input: &TextInput,
    results: Vec<Option<Result<T, E>>>,
    err: impl Fn(E) -> PyErr,
) -> PyResult<Vec<T>> {
    let mut out = Vec::with_capacity(results.len());
    for (i, r) in results.into_iter().enumerate() {
        match r {
            Some(Ok(v)) => out.push(v),
            Some(Err(e)) => return Err(at_index(py, input, i, err(e))),
            None => return Err(null_error(i)),
        }
    }
    Ok(out)
}

/// A str's value as a Python scalar, a batch's as a 1-D numpy array.
fn output<'py, T>(py: Python<'py>, input: &TextInput, vals: Vec<T>) -> PyResult<Bound<'py, PyAny>>
where
    T: numpy::Element + IntoPyObject<'py>,
{
    if matches!(input, TextInput::Str(_)) {
        let v = vals.into_iter().next().expect("one value for one str");
        return v.into_bound_py_any(py);
    }
    require_numpy(py)?;
    Ok(PyArray1::from_vec(py, vals).into_any())
}

/// The documents of a call, numpy checked for a batch.
fn docs(
    py: Python<'_>,
    texts: &Bound<'_, PyAny>,
    threads: Option<ThreadsArg>,
) -> PyResult<(TextInput, Option<crate::input::Threads>)> {
    let (input, threads) = texts_and_threads(texts, threads, None)?;
    if !matches!(input, TextInput::Str(_)) {
        require_numpy(py)?;
    }
    Ok((input, threads))
}

/// Run a score kernel over `texts` and build the result.
fn score<'py, T, F>(
    py: Python<'py>,
    texts: &Bound<'py, PyAny>,
    threads: Option<ThreadsArg>,
    kernel: F,
) -> PyResult<Bound<'py, PyAny>>
where
    T: Send + numpy::Element + IntoPyObject<'py>,
    F: Fn(&str) -> Result<T, ScoreError> + Sync,
{
    let (input, threads) = docs(py, texts, threads)?;
    let results = map_values_cost(py, &input, threads, DocCost::Light, kernel)?;
    let vals = values(py, &input, results, score_error)?;
    output(py, &input, vals)
}

/// [`score`] for a kernel that never raises.
fn score_ok<'py, T, F>(
    py: Python<'py>,
    texts: &Bound<'py, PyAny>,
    threads: Option<ThreadsArg>,
    kernel: F,
) -> PyResult<Bound<'py, PyAny>>
where
    T: Send + numpy::Element + IntoPyObject<'py>,
    F: Fn(&str) -> T + Sync,
{
    score(py, texts, threads, |s| Ok(kernel(s)))
}

/// NeMo's `get_word_splitter(lang)`.
fn splitter(lang: &str) -> PyResult<Splitter> {
    Splitter::for_lang(lang).ok_or_else(|| {
        PyNotImplementedError::new_err(format!(
            "lang={lang:?}: pyroid.nemo does not port NeMo Curator's Japanese word splitter \
             (MeCab); use NeMo Curator's own filter"
        ))
    })
}

/// A count as numpy's int64 (document counts are far below 2**63).
fn int(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

macro_rules! char_score {
    ($(#[$doc:meta])* $name:ident, $kernel:path) => {
        $(#[$doc])*
        #[pyfunction]
        #[pyo3(signature = (texts, /, *, threads = None))]
        fn $name<'py>(
            py: Python<'py>,
            texts: &Bound<'py, PyAny>,
            threads: Option<ThreadsArg>,
        ) -> PyResult<Bound<'py, PyAny>> {
            score_ok(py, texts, threads, $kernel)
        }
    };
}

macro_rules! line_score {
    ($(#[$doc:meta])* $name:ident, $kernel:path) => {
        $(#[$doc])*
        #[pyfunction]
        #[pyo3(signature = (texts, /, *, threads = None))]
        fn $name<'py>(
            py: Python<'py>,
            texts: &Bound<'py, PyAny>,
            threads: Option<ThreadsArg>,
        ) -> PyResult<Bound<'py, PyAny>> {
            score(py, texts, threads, $kernel)
        }
    };
}

macro_rules! word_score {
    ($(#[$doc:meta])* $name:ident, $kernel:path) => {
        $(#[$doc])*
        #[pyfunction]
        #[pyo3(signature = (texts, /, *, lang = "en", threads = None))]
        fn $name<'py>(
            py: Python<'py>,
            texts: &Bound<'py, PyAny>,
            lang: &str,
            threads: Option<ThreadsArg>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let sp = splitter(lang)?;
            score(py, texts, threads, |s| $kernel(s, sp))
        }
    };
}

char_score!(
    /// ``NonAlphaNumericFilter``: the share of characters outside ``[a-zA-Z0-9\n?!,.]``.
    non_alpha_numeric,
    f::non_alpha_numeric
);
char_score!(
    /// ``NumbersFilter``: the share of characters that are ASCII digits.
    numbers,
    f::numbers
);
char_score!(
    /// ``UrlsFilter``: the share of characters in URLs (NeMo's default URL pattern).
    urls,
    f::urls
);
char_score!(
    /// ``WhiteSpaceFilter``: the share of characters in ``\t \n \r \b`` and space.
    white_space,
    f::white_space
);
char_score!(
    /// ``ParenthesesFilter``: the share of characters in ``{ } ⟨ ⟩ [ ] ( )``.
    parentheses,
    f::parentheses
);
char_score!(
    /// ``BoilerPlateStringFilter``: the share of paragraphs with a policy string; 1.0 with
    /// "lorem ipsum".
    boilerplate_string,
    f::boilerplate_string
);
char_score!(
    /// ``RepeatedParagraphsFilter``: distinct paragraphs over paragraphs.
    repeated_paragraphs,
    r::repeated_paragraphs
);
line_score!(
    /// ``RepeatedLinesFilter``: distinct lines over lines.
    repeated_lines,
    r::repeated_lines
);
line_score!(
    /// ``RepeatedLinesByCharFilter``: characters of distinct lines over all line characters.
    repeated_lines_by_char,
    r::repeated_lines_by_char
);
line_score!(
    /// ``RepeatedParagraphsByCharFilter``: the same over paragraphs.
    repeated_paragraphs_by_char,
    r::repeated_paragraphs_by_char
);
line_score!(
    /// ``PunctuationFilter``: the share of lines that do not end with ``. ? ! " '``.
    punctuation,
    f::punctuation
);
line_score!(
    /// ``EllipsisFilter``: the share of lines that end with an ellipsis mark.
    ellipsis,
    f::ellipsis
);
line_score!(
    /// ``BulletsFilter``: the share of lines that start with a bullet.
    bullets,
    f::bullets
);
word_score!(
    /// ``SymbolsToWordsFilter``: the share of words that are an ellipsis or mostly ``#``.
    symbols_to_words,
    f::symbols_to_words
);
word_score!(
    /// ``WordsWithoutAlphabetsFilter``: the share of words with a ``[[:alpha:]]`` character.
    words_without_alphabets,
    f::words_without_alphabets
);
word_score!(
    /// ``MeanWordLengthFilter``: the mean word length.
    mean_word_length,
    f::mean_word_length
);

/// ``WordCountFilter``: the number of words.
#[pyfunction]
#[pyo3(signature = (texts, /, *, lang = "en", threads = None))]
fn word_count<'py>(
    py: Python<'py>,
    texts: &Bound<'py, PyAny>,
    lang: &str,
    threads: Option<ThreadsArg>,
) -> PyResult<Bound<'py, PyAny>> {
    let sp = splitter(lang)?;
    score_ok(py, texts, threads, |s| int(f::word_count(s, sp)))
}

/// ``LongWordFilter``: the length of the longest word.
#[pyfunction]
#[pyo3(signature = (texts, /, *, lang = "en", threads = None))]
fn long_word<'py>(
    py: Python<'py>,
    texts: &Bound<'py, PyAny>,
    lang: &str,
    threads: Option<ThreadsArg>,
) -> PyResult<Bound<'py, PyAny>> {
    let sp = splitter(lang)?;
    score(py, texts, threads, |s| f::long_word(s, sp).map(int))
}

/// ``CommonEnglishWordsFilter``: the number of common English words (see the stub).
#[pyfunction]
#[pyo3(signature = (texts, /, *, min_num_common_words = 2, stop_at_false = true, threads = None))]
fn common_english_words<'py>(
    py: Python<'py>,
    texts: &Bound<'py, PyAny>,
    min_num_common_words: i64,
    stop_at_false: bool,
    threads: Option<ThreadsArg>,
) -> PyResult<Bound<'py, PyAny>> {
    score_ok(py, texts, threads, |s| {
        let c = f::common_english_words(s, min_num_common_words, stop_at_false);
        i64::try_from(c).unwrap_or(i64::MAX)
    })
}

/// ``PornographicUrlsFilter``: 1 if a URL contains "porn", else 0.
#[pyfunction]
#[pyo3(signature = (texts, /, *, threads = None))]
fn pornographic_urls<'py>(
    py: Python<'py>,
    texts: &Bound<'py, PyAny>,
    threads: Option<ThreadsArg>,
) -> PyResult<Bound<'py, PyAny>> {
    score_ok(py, texts, threads, |s| i64::from(f::pornographic_urls(s)))
}

/// ``SubstringFilter``: 1 if ``substring`` is at ``position`` of the text, else 0.
#[pyfunction(name = "substring")]
#[pyo3(signature = (texts, /, substring, position, *, threads = None))]
fn substring_score<'py>(
    py: Python<'py>,
    texts: &Bound<'py, PyAny>,
    substring: PyBackedStr,
    position: &Bound<'py, PyAny>,
    threads: Option<ThreadsArg>,
) -> PyResult<Bound<'py, PyAny>> {
    // NeMo checks the position when the filter is built, before any document.
    let pos = position
        .extract::<PyBackedStr>()
        .ok()
        .and_then(|p| f::Position::parse(&p));
    let Some(pos) = pos else {
        return Err(PyValueError::new_err(format!(
            "Invalid position: {}. Must be one of: prefix, suffix, any.",
            position.str()?
        )));
    };
    score_ok(py, texts, threads, |s| {
        i64::from(f::substring(s, &substring, pos))
    })
}

/// ``RepeatingDuplicateNGramsFilter(n)``: the share of characters in repeated n-grams.
#[pyfunction]
#[pyo3(signature = (texts, /, n = 2, *, lang = "en", threads = None))]
fn repeating_duplicate_ngrams<'py>(
    py: Python<'py>,
    texts: &Bound<'py, PyAny>,
    n: i64,
    lang: &str,
    threads: Option<ThreadsArg>,
) -> PyResult<Bound<'py, PyAny>> {
    let sp = splitter(lang)?;
    score_ok(py, texts, threads, |s| {
        r::repeating_duplicate_ngrams(s, n, sp)
    })
}

/// Document `i`'s text (`None` for an Arrow null or invalid UTF-8, which never get here).
fn doc_text(input: &TextInput, i: usize) -> Option<&str> {
    match input {
        TextInput::Str(s) => Some(s),
        TextInput::List(d) | TextInput::Tuple(d) => d.get(i).map(|s| &**s),
        TextInput::Arrow(strings) => {
            let mut row = i;
            for chunk in &strings.chunks {
                if row < chunk.len() {
                    return chunk.get(row).and_then(Result::ok);
                }
                row -= chunk.len();
            }
            None
        }
    }
}

/// Document `i` as a str object: the caller's own object for str, list and tuple input.
fn doc_object<'py>(py: Python<'py>, input: &TextInput, i: usize) -> Bound<'py, PyString> {
    match input {
        TextInput::Str(s) => s.into_pyobject(py).unwrap_or_else(|e| match e {}),
        TextInput::List(d) | TextInput::Tuple(d) => {
            (&d[i]).into_pyobject(py).unwrap_or_else(|e| match e {})
        }
        TextInput::Arrow(_) => PyString::new(py, doc_text(input, i).unwrap_or_default()),
    }
}

/// ``RepeatingTopNGramsFilter(n)``, native half: the scores (NaN for tied documents) and,
/// per tied document, ``(index, text, starts, words)``. ``starts`` holds the first word
/// index of each tied n-gram (``None``: every n-gram is tied); ``words`` is the document's
/// words for ``lang="zh"`` (``None``: ``text.strip().split()``). ``pyroid.nemo`` picks
/// among the tied n-grams with NeMo's own ``set`` order.
///
/// With ``hash_key`` (the first 16 bytes of ``_Py_HashSecret``, which ``pyroid.nemo`` has
/// checked against this interpreter), ties are resolved here, in ``set`` order as CPython
/// builds it, and no document comes back tied.
#[pyfunction(name = "_repeating_top_ngrams")]
#[pyo3(signature = (texts, /, n = 2, *, lang = "en", threads = None, hash_key = None))]
fn repeating_top_ngrams_native<'py>(
    py: Python<'py>,
    texts: &Bound<'py, PyAny>,
    n: i64,
    lang: &str,
    threads: Option<ThreadsArg>,
    hash_key: Option<&[u8]>,
) -> PyResult<Bound<'py, PyTuple>> {
    let sp = splitter(lang)?;
    let key = hash_key.map(parse_key).transpose()?;
    let (input, threads) = docs(py, texts, threads)?;
    let results = map_values_cost(py, &input, threads, DocCost::Light, |s| {
        r::repeating_top_ngrams(s, n, sp, key)
    })?;
    let tops = values(py, &input, results, score_error)?;
    let mut scores = Vec::with_capacity(tops.len());
    let mut ties = Vec::new();
    for (i, top) in tops.into_iter().enumerate() {
        let tie = match top {
            TopNGram::Score(v) => {
                scores.push(v);
                continue;
            }
            TopNGram::Tied(tie) => tie,
        };
        scores.push(f64::NAN);
        let starts = match tie.starts {
            Some(s) => PyList::new(py, s)?.into_any(),
            None => py.None().into_bound(py),
        };
        let words = match sp {
            Splitter::Whitespace => py.None().into_bound(py),
            Splitter::Jieba => {
                let text = doc_text(&input, i).unwrap_or_default();
                PyList::new(py, sp.doc_words(text))?.into_any()
            }
        };
        let entry = (i, doc_object(py, &input, i), starts, words);
        ties.push(entry.into_pyobject(py)?);
    }
    let scores = output(py, &input, scores)?;
    PyTuple::new(py, [scores, PyList::new(py, ties)?.into_any()])
}

fn parse_key(secret: &[u8]) -> PyResult<HashKey> {
    HashKey::from_secret(secret)
        .ok_or_else(|| PyValueError::new_err("hash_key must hold at least 16 bytes"))
}

/// ``hash(s)`` for each str, from CPython's hash key (``pyroid.nemo`` compares these with
/// ``hash()`` before it uses the key).
#[pyfunction(name = "_str_hashes")]
fn str_hashes(strings: Vec<PyBackedStr>, hash_key: &[u8]) -> PyResult<Vec<i64>> {
    let key = parse_key(hash_key)?;
    let mut buf = Vec::new();
    Ok(strings
        .iter()
        .map(|s| pyhash::str_hash(key, s, &mut buf))
        .collect())
}

/// ``list(set(ngrams))`` for ``ngrams = list(zip(*[words[i:] for i in range(n)]))``, as the
/// index of each distinct n-gram's first occurrence, from CPython's hash key: the code that
/// resolves ``_repeating_top_ngrams``' ties.
#[pyfunction(name = "_ngram_set_order")]
fn ngram_set_order(words: Vec<PyBackedStr>, n: usize, hash_key: &[u8]) -> PyResult<Vec<usize>> {
    let key = parse_key(hash_key)?;
    let words: Vec<&str> = words.iter().map(|w| &**w).collect();
    Ok(r::ngram_set_order(key, &words, n))
}

/// ``list(set(items))`` for items with these ``hashes`` that are equal when their ``keys``
/// are, as the index of each distinct item's first occurrence.
#[pyfunction(name = "_set_order")]
fn set_order(hashes: Vec<i64>, keys: Vec<i64>) -> PyResult<Vec<usize>> {
    if hashes.len() != keys.len() || u32::try_from(hashes.len()).is_err() {
        return Err(PyValueError::new_err(
            "hashes and keys must have the same length, below 2**32",
        ));
    }
    Ok(pyhash::set_order(&hashes, |a, b| keys[a] == keys[b]))
}

// ---- fastText stages ---------------------------------------------------------------------

/// Why a fastText stage has no score for a document.
#[derive(Clone, Copy, Debug)]
enum FtError {
    Predict(PredictError),
    /// No label predicted: NeMo's `score[0][0]` raises IndexError.
    NoLabel,
}

fn ft_error(e: FtError) -> PyErr {
    match e {
        FtError::Predict(e) => predict_error(e),
        FtError::NoLabel => {
            PyIndexError::new_err("index 0 is out of bounds for axis 0 with size 0")
        }
    }
}

/// The top-1 prediction for `line` (`model.predict([line], k=1)`): label id and probability.
fn top1(model: &pyroid_core::fasttext::Model, line: &str) -> Result<(usize, f32), FtError> {
    let preds = model.predict(line, 1, 0.0).map_err(FtError::Predict)?;
    preds
        .first()
        .map(|p| (p.label, p.prob))
        .ok_or(FtError::NoLabel)
}

/// fastText decodes labels strictly: the UnicodeDecodeError for a label that is not UTF-8.
fn label_decode_error(py: Python<'_>, bytes: &[u8], err: std::str::Utf8Error) -> PyErr {
    PyUnicodeDecodeError::new_utf8(py, bytes, err)
        .map_or_else(|e| e, |e| PyErr::from_value(e.into_any()))
}

/// ``FastTextLangId``'s score: ``(probability, language)`` per document.
#[pyfunction]
#[pyo3(signature = (model, texts, /, *, threads = None))]
fn fasttext_lang_id<'py>(
    py: Python<'py>,
    model: &Bound<'py, Model>,
    texts: &Bound<'py, PyAny>,
    threads: Option<ThreadsArg>,
) -> PyResult<Bound<'py, PyTuple>> {
    let (input, threads) = docs(py, texts, threads)?;
    let m = &model.get().inner;
    let results = map_values_cost(py, &input, threads, DocCost::Heavy, |s| {
        top1(m, &nft::lang_id_line(s))
    })?;
    let preds = values(py, &input, results, ft_error)?;
    // One str per label, made on first use.
    let mut codes: Vec<Option<Bound<'py, PyString>>> = vec![None; m.nlabels()];
    let mut langs = Vec::with_capacity(preds.len());
    for (i, &(id, _)) in preds.iter().enumerate() {
        if codes[id].is_none() {
            let bytes = m.label(id);
            let label = std::str::from_utf8(bytes)
                .map_err(|err| at_index(py, &input, i, label_decode_error(py, bytes, err)))?;
            codes[id] = Some(PyString::new(py, nft::lang_code(label)));
        }
        langs.push(codes[id].clone().expect("set above"));
    }
    let scores: Vec<f64> = preds.iter().map(|&(_, p)| f64::from(p)).collect();
    if matches!(input, TextInput::Str(_)) {
        let lang = langs.pop().expect("one document");
        return PyTuple::new(py, [scores[0].into_bound_py_any(py)?, lang.into_any()]);
    }
    let scores = output(py, &input, scores)?;
    PyTuple::new(py, [scores, PyList::new(py, langs)?.into_any()])
}

/// ``FastTextQualityFilter``'s score: ``p`` if the top label is ``label``, else ``1 - p``.
#[pyfunction]
#[pyo3(signature = (model, texts, /, label = "__label__hq", *, threads = None))]
fn fasttext_quality<'py>(
    py: Python<'py>,
    model: &Bound<'py, Model>,
    texts: &Bound<'py, PyAny>,
    label: &str,
    threads: Option<ThreadsArg>,
) -> PyResult<Bound<'py, PyAny>> {
    let (input, threads) = docs(py, texts, threads)?;
    let m = &model.get().inner;
    let results = map_values_cost(py, &input, threads, DocCost::Heavy, |s| {
        top1(m, &nft::quality_line(s))
    })?;
    let preds = values(py, &input, results, ft_error)?;
    let mut scores = Vec::with_capacity(preds.len());
    for (i, &(id, p)) in preds.iter().enumerate() {
        let bytes = m.label(id);
        if let Err(err) = std::str::from_utf8(bytes) {
            return Err(at_index(py, &input, i, label_decode_error(py, bytes, err)));
        }
        let p = f64::from(p);
        scores.push(if bytes == label.as_bytes() {
            p
        } else {
            1.0 - p
        });
    }
    output(py, &input, scores)
}

/// ``repr(s)`` for a language code: plain printable ASCII without quotes or backslashes
/// directly, anything else through Python's ``repr``.
fn push_str_repr(out: &mut String, s: &Bound<'_, PyString>) -> PyResult<()> {
    if let Ok(t) = s.to_str() {
        if t.bytes()
            .all(|b| (0x20..0x7F).contains(&b) && b != b'\'' && b != b'\\')
        {
            out.push('\'');
            out.push_str(t);
            out.push('\'');
            return Ok(());
        }
    }
    out.push_str(&s.repr()?.to_cow()?);
    Ok(())
}

/// ``FastTextLangId.score_document``'s strings: ``str([score, lang])`` per document.
#[pyfunction]
fn lang_id_strings<'py>(
    py: Python<'py>,
    scores: &Bound<'py, PyAny>,
    langs: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let array = if require_numpy(py).is_ok() {
        scores.extract::<PyReadonlyArray1<'py, f64>>().ok()
    } else {
        None
    };
    let scores: Vec<f64> = match array {
        Some(a) => a.as_array().to_vec(),
        None => scores.extract()?,
    };
    let langs: Vec<Bound<'py, PyString>> = langs.extract()?;
    if scores.len() != langs.len() {
        return Err(PyValueError::new_err(format!(
            "scores and langs differ in length ({} and {})",
            scores.len(),
            langs.len()
        )));
    }
    let mut out = Vec::with_capacity(scores.len());
    let mut buf = String::new();
    for (&score, lang) in scores.iter().zip(&langs) {
        buf.clear();
        buf.push('[');
        float_repr(score, &mut buf);
        buf.push_str(", ");
        push_str_repr(&mut buf, lang)?;
        buf.push(']');
        out.push(PyString::new(py, &buf));
    }
    PyList::new(py, out)
}

fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(non_alpha_numeric, m)?)?;
    m.add_function(wrap_pyfunction!(symbols_to_words, m)?)?;
    m.add_function(wrap_pyfunction!(numbers, m)?)?;
    m.add_function(wrap_pyfunction!(urls, m)?)?;
    m.add_function(wrap_pyfunction!(white_space, m)?)?;
    m.add_function(wrap_pyfunction!(parentheses, m)?)?;
    m.add_function(wrap_pyfunction!(boilerplate_string, m)?)?;
    m.add_function(wrap_pyfunction!(repeated_lines, m)?)?;
    m.add_function(wrap_pyfunction!(repeated_paragraphs, m)?)?;
    m.add_function(wrap_pyfunction!(repeated_lines_by_char, m)?)?;
    m.add_function(wrap_pyfunction!(repeated_paragraphs_by_char, m)?)?;
    m.add_function(wrap_pyfunction!(word_count, m)?)?;
    m.add_function(wrap_pyfunction!(punctuation, m)?)?;
    m.add_function(wrap_pyfunction!(words_without_alphabets, m)?)?;
    m.add_function(wrap_pyfunction!(common_english_words, m)?)?;
    m.add_function(wrap_pyfunction!(mean_word_length, m)?)?;
    m.add_function(wrap_pyfunction!(long_word, m)?)?;
    m.add_function(wrap_pyfunction!(ellipsis, m)?)?;
    m.add_function(wrap_pyfunction!(repeating_top_ngrams_native, m)?)?;
    m.add_function(wrap_pyfunction!(str_hashes, m)?)?;
    m.add_function(wrap_pyfunction!(ngram_set_order, m)?)?;
    m.add_function(wrap_pyfunction!(set_order, m)?)?;
    m.add_function(wrap_pyfunction!(repeating_duplicate_ngrams, m)?)?;
    m.add_function(wrap_pyfunction!(bullets, m)?)?;
    m.add_function(wrap_pyfunction!(pornographic_urls, m)?)?;
    m.add_function(wrap_pyfunction!(substring_score, m)?)?;
    m.add_function(wrap_pyfunction!(fasttext_lang_id, m)?)?;
    m.add_function(wrap_pyfunction!(fasttext_quality, m)?)?;
    m.add_function(wrap_pyfunction!(lang_id_strings, m)?)?;
    Ok(())
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pyroid.nemo")?;
    m.gil_used(false)?;
    // Filter scores and fastText helpers.
    add(&m)?;
    // String modifiers:
    crate::nemo_modifiers::add(&m)?;
    parent.add_submodule(&m)
}
