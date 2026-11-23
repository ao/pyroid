//! `pyroid._pyroid.nltk`: the native half of `pyroid.nltk` (NLTK 3.10.4's
//! Punkt and Treebank tokenizers, `pyroid_core::nltk`).
//!
//! `_Punkt` is one `PunktSentenceTokenizer`'s parameters, built by the Python
//! package from a `punkt_tab` model or a `PunktParameters`; it is immutable,
//! and the package builds a new one when the parameters change. The Python
//! package adds NLTK's API on top: the parameter containers, model loading,
//! generators, and the delegation to NLTK.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple};
use pyroid_core::nltk::UnicodeVersion;
use pyroid_core::nltk::punkt::{
    self, Decision, LangVars, OrthoHeuristic, Params, PunktError, Raised,
};
use pyroid_core::nltk::regexp::{self, RegexpTokenizer};
use pyroid_core::nltk::treebank::{self, AlignError, Flavor};
use rustc_hash::FxHashMap;

use crate::batch::{DocCost, map_values_cost};
use crate::input::{TextInput, Threads};

fn version(v: &str) -> PyResult<UnicodeVersion> {
    UnicodeVersion::from_unidata_version(v).ok_or_else(|| {
        PyValueError::new_err(format!(
            "pyroid.nltk does not support Unicode {v} (this CPython's unicodedata version)"
        ))
    })
}

fn flavor(nltk: bool) -> Flavor {
    if nltk { Flavor::Nltk } else { Flavor::Treebank }
}

/// The Python strs made in this call, by text: tokens repeat, and a cache hit
/// is cheaper than decoding the UTF-8 again.
struct Strs<'py, 'd> {
    py: Python<'py>,
    cache: FxHashMap<&'d str, Bound<'py, PyString>>,
}

impl<'py, 'd> Strs<'py, 'd> {
    fn new(py: Python<'py>) -> Self {
        Self {
            py,
            cache: FxHashMap::default(),
        }
    }

    fn get(&mut self, s: &'d str) -> Bound<'py, PyAny> {
        if s.len() > 24 {
            return PyString::new(self.py, s).into_any();
        }
        if self.cache.len() >= 1 << 16 {
            self.cache.clear();
        }
        self.cache
            .entry(s)
            .or_insert_with(|| PyString::new(self.py, s))
            .clone()
            .into_any()
    }
}

/// A str for a str input, else a list with one entry per document (`None`
/// for Arrow nulls).
fn per_doc<'py, 'd, R>(
    py: Python<'py>,
    x: &TextInput,
    results: &'d [Option<R>],
    mut one: impl FnMut(usize, &'d R) -> PyResult<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if matches!(x, TextInput::Str(_)) {
        let r = results
            .first()
            .and_then(Option::as_ref)
            .expect("one result");
        return one(0, r);
    }
    let out = results
        .iter()
        .enumerate()
        .map(|(i, r)| match r {
            Some(r) => one(i, r),
            None => Ok(py.None().into_bound(py)),
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, out)?.into_any())
}

/// Each document's text (`None` for an Arrow null), in input order.
fn docs(x: &TextInput) -> Vec<Option<&str>> {
    match x {
        TextInput::Str(s) => vec![Some(&**s)],
        TextInput::List(d) | TextInput::Tuple(d) => d.iter().map(|s| Some(&**s)).collect(),
        TextInput::Arrow(strings) => strings
            .chunks
            .iter()
            .flat_map(|c| (0..c.len()).map(move |i| c.get(i).and_then(Result::ok)))
            .collect(),
    }
}

/// Tokens as one buffer and their byte ranges in it.
struct Tokens {
    buf: String,
    spans: Vec<(u32, u32)>,
}

impl Tokens {
    fn new() -> Self {
        Self {
            buf: String::new(),
            spans: Vec::new(),
        }
    }

    fn push(&mut self, tok: &str) {
        let start = self.buf.len();
        self.buf.push_str(tok);
        #[allow(clippy::cast_possible_truncation)] // documents are < 4 GiB (Arrow string offsets)
        self.spans.push((start as u32, self.buf.len() as u32));
    }

    fn list<'py, 'd>(&'d self, strs: &mut Strs<'py, 'd>) -> PyResult<Bound<'py, PyAny>> {
        let items: Vec<_> = self
            .spans
            .iter()
            .map(|&(a, b)| strs.get(&self.buf[a as usize..b as usize]))
            .collect();
        Ok(PyList::new(strs.py, items)?.into_any())
    }
}

fn span_list<'py>(py: Python<'py>, spans: &[(usize, usize)]) -> PyResult<Bound<'py, PyAny>> {
    let items = spans
        .iter()
        .map(|&(a, b)| PyTuple::new(py, [a, b]))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, items)?.into_any())
}

/// Byte spans to Python (code point) offsets.
fn char_spans(text: &str, spans: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    if text.is_ascii() {
        return spans;
    }
    let mut flat: Vec<usize> = spans.iter().flat_map(|&(a, b)| [a, b]).collect();
    punkt::char_offsets(text, &mut flat);
    flat.chunks_exact(2).map(|p| (p[0], p[1])).collect()
}

/// One `PunktSentenceTokenizer`'s parameters and language variables.
#[pyclass(frozen, module = "pyroid.nltk", name = "_Punkt")]
#[derive(Debug)]
pub struct Punkt {
    inner: punkt::Punkt,
    version: UnicodeVersion,
}

#[pymethods]
impl Punkt {
    #[new]
    fn new(
        abbrev_types: Vec<String>,
        collocations: Vec<(String, String)>,
        sent_starters: Vec<String>,
        ortho_context: Vec<(String, i64)>,
        sent_end_chars: &str,
        unicode_version: &str,
    ) -> PyResult<Self> {
        let v = version(unicode_version)?;
        let params = Params::new(abbrev_types, collocations, sent_starters, ortho_context);
        let lang = LangVars::new(sent_end_chars.chars().collect());
        Ok(Self {
            inner: punkt::Punkt::new(params, lang, v),
            version: v,
        })
    }

    /// The sentences of each document (`span_tokenize` slices).
    #[pyo3(signature = (text, /, realign = true, *, threads = None))]
    fn sentences<'py>(
        &self,
        py: Python<'py>,
        text: TextInput,
        realign: bool,
        threads: Option<Threads>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let results = map_values_cost(py, &text, threads, DocCost::Light, |s| {
            self.inner.span_tokenize(s, realign)
        })?;
        let texts = docs(&text);
        let mut strs = Strs::new(py);
        per_doc(py, &text, &results, |i, spans| {
            let doc = texts[i].unwrap_or_default();
            // text[s:e] in Python: empty when the span is empty or inverted.
            let items: Vec<_> = spans
                .iter()
                .map(|&(a, b)| strs.get(if a < b { &doc[a..b] } else { "" }))
                .collect();
            Ok(PyList::new(py, items)?.into_any())
        })
    }

    /// `span_tokenize` of each document, in code point offsets.
    #[pyo3(signature = (text, /, realign = true, *, threads = None))]
    fn spans<'py>(
        &self,
        py: Python<'py>,
        text: TextInput,
        realign: bool,
        threads: Option<Threads>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let results = map_values_cost(py, &text, threads, DocCost::Light, |s| {
            char_spans(s, self.inner.span_tokenize(s, realign))
        })?;
        per_doc(py, &text, &results, |_, spans| span_list(py, spans))
    }

    /// `text_contains_sentbreak(text)`.
    fn contains_sentbreak(&self, py: Python<'_>, text: &str) -> bool {
        py.detach(|| self.inner.text_contains_sentbreak(text))
    }

    /// `sentences_from_tokens(tokens)`: the sentence lengths (in tokens), and
    /// the exception NLTK's generator raises after them, if any.
    fn sentences_from_tokens(&self, tokens: Vec<String>) -> (Vec<usize>, Option<Exc>) {
        split(self.inner.sentences_from_tokens(&tokens))
    }

    /// `sentences_from_text_legacy(text)`: the sentences, and the exception
    /// NLTK's generator raises after them, if any.
    fn legacy(&self, py: Python<'_>, text: &str) -> (Vec<String>, Option<Exc>) {
        split(py.detach(|| self.inner.sentences_from_text_legacy(text)))
    }

    /// `debug_decisions(text)` as tuples (the package makes NLTK's dicts), and
    /// the exception NLTK's generator raises after them, if any.
    fn debug_decisions<'py>(
        &self,
        py: Python<'py>,
        text: &str,
    ) -> PyResult<(Bound<'py, PyList>, Option<Exc>)> {
        let (decisions, exc) = split(py.detach(|| self.inner.debug_decisions(text)));
        let mut offsets: Vec<usize> = decisions.iter().map(|d| d.period_index).collect();
        punkt::char_offsets(text, &mut offsets);
        let items = decisions
            .iter()
            .zip(offsets)
            .map(|(d, at)| decision(py, d, at))
            .collect::<PyResult<Vec<_>>>()?;
        Ok((PyList::new(py, items)?, exc))
    }

    /// `word_tokenize(text)`: the sentences, then `NLTKWordTokenizer` on each.
    #[pyo3(signature = (text, /, *, threads = None))]
    fn word_tokenize<'py>(
        &self,
        py: Python<'py>,
        text: TextInput,
        threads: Option<Threads>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = self.version;
        let results = map_values_cost(py, &text, threads, DocCost::Light, |s| {
            let mut out = Tokens::new();
            for sent in self.inner.tokenize(s, true) {
                let text = treebank::tokenize_text(sent, Flavor::Nltk, false, v);
                for tok in treebank::tokens(&text) {
                    out.push(tok);
                }
            }
            out
        })?;
        let mut strs = Strs::new(py);
        per_doc(py, &text, &results, |_, t| t.list(&mut strs))
    }
}

/// An exception NLTK raises: its type's name and its message.
type Exc = (&'static str, &'static str);

fn exc(e: PunktError) -> Exc {
    (e.python_type(), e.message())
}

fn split<T>(r: Result<Vec<T>, Raised<T>>) -> (Vec<T>, Option<Exc>) {
    match r {
        Ok(v) => (v, None),
        Err(Raised { yielded, error }) => (yielded, Some(exc(error))),
    }
}

fn decision<'py>(py: Python<'py>, d: &Decision, at: usize) -> PyResult<Bound<'py, PyAny>> {
    let ortho = match d.type2_ortho_heuristic {
        OrthoHeuristic::Yes => true.into_pyobject(py)?.to_owned().into_any(),
        OrthoHeuristic::No => false.into_pyobject(py)?.to_owned().into_any(),
        OrthoHeuristic::Unknown => PyString::new(py, "unknown").into_any(),
    };
    let items: [Bound<'py, PyAny>; 12] = [
        at.into_pyobject(py)?.into_any(),
        PyString::new(py, &d.text).into_any(),
        PyString::new(py, &d.type1).into_any(),
        PyString::new(py, &d.type2).into_any(),
        d.type1_in_abbrs.into_pyobject(py)?.to_owned().into_any(),
        d.type1_is_initial.into_pyobject(py)?.to_owned().into_any(),
        d.type2_is_sent_starter
            .into_pyobject(py)?
            .to_owned()
            .into_any(),
        ortho,
        d.type2_ortho_contexts.into_pyobject(py)?.into_any(),
        d.collocation.into_pyobject(py)?.to_owned().into_any(),
        PyString::new(py, d.reason.as_str()).into_any(),
        d.break_decision.into_pyobject(py)?.into_any(),
    ];
    Ok(PyTuple::new(py, items)?.into_any())
}

/// The documents of `text` as strs (`None` for Arrow nulls), for the calls the
/// package hands to NLTK one document at a time.
#[pyfunction]
fn texts(text: TextInput) -> Vec<Option<String>> {
    docs(&text)
        .into_iter()
        .map(|d| d.map(str::to_owned))
        .collect()
}

/// `TreebankWordTokenizer().tokenize` (`nltk=False`) or `NLTKWordTokenizer().tokenize`.
#[pyfunction]
#[pyo3(name = "treebank", signature = (text, nltk, convert_parentheses, unicode_version, /, *, threads = None))]
fn treebank_tokenize<'py>(
    py: Python<'py>,
    text: TextInput,
    nltk: bool,
    convert_parentheses: bool,
    unicode_version: &str,
    threads: Option<Threads>,
) -> PyResult<Bound<'py, PyAny>> {
    let v = version(unicode_version)?;
    let f = flavor(nltk);
    let results = map_values_cost(py, &text, threads, DocCost::Light, |s| {
        treebank::tokenize_text(s, f, convert_parentheses, v)
    })?;
    let mut strs = Strs::new(py);
    per_doc(py, &text, &results, |_, tokenized: &String| {
        let items: Vec<_> = treebank::tokens(tokenized).map(|t| strs.get(t)).collect();
        Ok(PyList::new(py, items)?.into_any())
    })
}

/// `span_tokenize(text)` of either Treebank flavor, in code point offsets.
#[pyfunction]
fn treebank_spans(
    py: Python<'_>,
    text: &str,
    nltk: bool,
    unicode_version: &str,
) -> PyResult<Vec<(usize, usize)>> {
    let v = version(unicode_version)?;
    py.detach(|| treebank::span_tokenize(text, flavor(nltk), v))
        .map_err(align_error)
}

/// `TreebankWordDetokenizer().detokenize(tokens, convert_parentheses)`.
#[pyfunction]
fn detokenize(tokens: Vec<String>, convert_parentheses: bool) -> String {
    let toks: Vec<&str> = tokens.iter().map(String::as_str).collect();
    treebank::detokenize(&toks, convert_parentheses)
}

/// `nltk.tokenize.util.align_tokens(tokens, sentence)`.
#[pyfunction]
fn align_tokens(tokens: Vec<String>, sentence: &str) -> PyResult<Vec<(usize, usize)>> {
    let toks: Vec<&str> = tokens.iter().map(String::as_str).collect();
    treebank::align_tokens(&toks, sentence).map_err(align_error)
}

fn align_error(e: AlignError) -> PyErr {
    match e {
        // span_tokenize's next() on its exhausted quote iterator, inside a list comprehension.
        AlignError::QuotesExhausted => PyRuntimeError::new_err("generator raised StopIteration"),
        e @ AlignError::NotFound { .. } => PyValueError::new_err(e.to_string()),
    }
}

fn regexp_tokenizer(
    pattern: &str,
    gaps: bool,
    discard_empty: bool,
    flags: u32,
) -> PyResult<RegexpTokenizer> {
    RegexpTokenizer::from_regex(pattern, gaps, discard_empty, flags).ok_or_else(|| {
        PyValueError::new_err(format!(
            "pattern {pattern:?} with flags {flags} is not native"
        ))
    })
}

/// Whether `RegexpTokenizer(pattern, flags=flags)` runs natively.
#[pyfunction]
fn regexp_native(pattern: &str, flags: u32) -> bool {
    regexp::Pattern::from_regex(pattern, flags).is_some()
}

/// `RegexpTokenizer(pattern, gaps, discard_empty, flags).tokenize(text)` for a
/// native pattern (`regexp_native`).
#[pyfunction]
#[pyo3(signature = (text, pattern, gaps, discard_empty, flags, /, *, threads = None))]
fn regexp_tokenize<'py>(
    py: Python<'py>,
    text: TextInput,
    pattern: &str,
    gaps: bool,
    discard_empty: bool,
    flags: u32,
    threads: Option<Threads>,
) -> PyResult<Bound<'py, PyAny>> {
    let tok = regexp_tokenizer(pattern, gaps, discard_empty, flags)?;
    let results = map_values_cost(py, &text, threads, DocCost::Light, |s| {
        let mut out = Tokens::new();
        for t in tok.tokenize(s) {
            out.push(t);
        }
        out
    })?;
    let mut strs = Strs::new(py);
    per_doc(py, &text, &results, |_, t| t.list(&mut strs))
}

/// `RegexpTokenizer(...).span_tokenize(text)` for a native pattern, in code
/// point offsets.
#[pyfunction]
fn regexp_spans(
    py: Python<'_>,
    text: &str,
    pattern: &str,
    gaps: bool,
    discard_empty: bool,
    flags: u32,
) -> PyResult<Vec<(usize, usize)>> {
    let tok = regexp_tokenizer(pattern, gaps, discard_empty, flags)?;
    Ok(py.detach(|| tok.span_tokenize(text)))
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pyroid.nltk")?;
    m.gil_used(false)?;
    m.add_class::<Punkt>()?;
    m.add_function(wrap_pyfunction!(texts, &m)?)?;
    m.add_function(wrap_pyfunction!(regexp_native, &m)?)?;
    m.add_function(wrap_pyfunction!(regexp_tokenize, &m)?)?;
    m.add_function(wrap_pyfunction!(regexp_spans, &m)?)?;
    m.add_function(wrap_pyfunction!(treebank_tokenize, &m)?)?;
    m.add_function(wrap_pyfunction!(treebank_spans, &m)?)?;
    m.add_function(wrap_pyfunction!(detokenize, &m)?)?;
    m.add_function(wrap_pyfunction!(align_tokens, &m)?)?;
    parent.add_submodule(&m)
}
