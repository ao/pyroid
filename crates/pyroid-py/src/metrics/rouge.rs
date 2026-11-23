// Derived from rouge-score 0.1.2 (Apache-2.0), © The rouge_score Authors; modified.
//! Bindings for `pyroid_core::metrics::rouge`: the native half of
//! `pyroid.metrics.rouge_scorer` and `pyroid.metrics.tokenizers`.
//!
//! The Python modules own rouge-score's classes, argument handling and every
//! path that needs the interpreter (custom tokenizers, nltk sentence
//! splitting, delegation); these functions only run the Rust kernels and
//! build rouge-score's result objects (`dict[str, Score]`).
//!
//! - `rouge_tokenize(text, use_stemmer, lowered=False)`: `DefaultTokenizer.tokenize`.
//! - `rouge_stem(word)`: nltk's Porter `stem` for an ASCII word (tests only).
//! - `rouge_score(types, target, prediction, use_stemmer, target_sentences=None,
//!   prediction_sentences=None)`: `RougeScorer.score` with the default tokenizer.
//! - `rouge_score_batch(...)`: the same over many pairs, on the shared pool
//!   (`None` for a pair with an Arrow null).
//! - `rouge_score_tokens(jobs, *, threads=None)`: scores from token lists.
//!
//! The scoring functions return `None` for `rouge_types` they can't parse
//! natively (not an exact list/tuple of exact strs); the Python side then
//! runs rouge-score's own logic.

use pyo3::exceptions::{PyUnicodeEncodeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyFloat, PyList, PyString, PyTuple};
use pyroid_core::metrics::rouge::{self as kernel, RougeType, Score, Scorer, TypeError};
use pyroid_core::par;

use super::batch_arg;
use crate::batch::{BatchError, DocCost, detach_one, rows};
use crate::errors::{input_error, type_name};
use crate::input::{TextInput, Threads, ThreadsArg, text_arg};

static SCORE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static TUPLE_NEW: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// Builds `pyroid.metrics.scoring.Score` objects. `tuple.__new__(Score,
/// values)` is exactly what the namedtuple's own `__new__` does, without its
/// Python frame.
struct ScoreBuilder<'py> {
    cls: &'py Bound<'py, PyAny>,
    new: &'py Bound<'py, PyAny>,
}

impl<'py> ScoreBuilder<'py> {
    fn get(py: Python<'py>) -> PyResult<Self> {
        let cls = SCORE.get_or_try_init(py, || -> PyResult<_> {
            Ok(py
                .import("pyroid.metrics.scoring")?
                .getattr("Score")?
                .unbind())
        })?;
        let new = TUPLE_NEW.get_or_try_init(py, || -> PyResult<_> {
            Ok(py.get_type::<PyTuple>().getattr("__new__")?.unbind())
        })?;
        Ok(Self {
            cls: cls.bind(py),
            new: new.bind(py),
        })
    }

    fn build(&self, score: Score) -> PyResult<Bound<'py, PyAny>> {
        let py = self.cls.py();
        let values = match score {
            // rouge-score's `Score(precision=0, recall=0, fmeasure=0)`: ints.
            Score::IntZero => PyTuple::new(py, [0, 0, 0])?,
            Score::Float {
                precision,
                recall,
                fmeasure,
            } => PyTuple::new(
                py,
                [
                    PyFloat::new(py, precision),
                    PyFloat::new(py, recall),
                    PyFloat::new(py, fmeasure),
                ],
            )?,
        };
        self.new.call1((self.cls, values))
    }

    /// `{key: Score}` in `keys` order; a repeated key keeps its first
    /// position and its last value, as rouge-score's `result[t] = scores`.
    fn dict(&self, keys: &[Bound<'py, PyAny>], scores: &[Score]) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(self.cls.py());
        for (k, s) in keys.iter().zip(scores) {
            d.set_item(k, self.build(*s)?)?;
        }
        Ok(d)
    }
}

/// Parsed `rouge_types` and the original objects (the result dict's keys).
type Types<'py> = (Vec<RougeType>, Vec<Bound<'py, PyAny>>);

/// `rouge_types` as rouge-score's `score` reads them: `None` if they are not
/// an exact list/tuple of exact strs (the caller then runs rouge-score's own
/// loop in Python). The first invalid type raises rouge-score's `ValueError`.
fn parse_types<'py>(types: &Bound<'py, PyAny>) -> PyResult<Option<Types<'py>>> {
    let items: Vec<Bound<'py, PyAny>> = if let Ok(l) = types.cast_exact::<PyList>() {
        l.iter().collect()
    } else if let Ok(t) = types.cast_exact::<PyTuple>() {
        t.iter().collect()
    } else {
        return Ok(None);
    };
    let mut kinds = Vec::with_capacity(items.len());
    for item in &items {
        let Ok(s) = item.cast_exact::<PyString>() else {
            return Ok(None);
        };
        // A lone surrogate (an invalid type either way): let Python format it.
        let Ok(s) = s.to_str() else { return Ok(None) };
        kinds.push(match RougeType::parse(s) {
            Ok(k) => k,
            Err(TypeError::NonPositive) => {
                return Err(PyValueError::new_err(format!(
                    "rougen requires positive n: {s}"
                )));
            }
            Err(TypeError::Invalid) => {
                return Err(PyValueError::new_err(format!("Invalid rouge type: {s}")));
            }
        });
    }
    Ok(Some((kinds, items)))
}

/// One `str` argument as UTF-8, with pyroid's usual note on lone surrogates.
fn one_text(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<PyBackedStr> {
    if !obj.is_instance_of::<PyString>() {
        return Err(input_error(
            obj.py(),
            format!("{name} must be a str, got {}", type_name(obj)),
        ));
    }
    match text_arg(obj, None, name)? {
        TextInput::Str(s) => Ok(s),
        _ => unreachable!("a str extracts as TextInput::Str"),
    }
}

/// `DefaultTokenizer(use_stemmer).tokenize(text)` for a str. `lowered`: the
/// caller already applied Python's `lower()` (rouge-score's path for `bytes`
/// and other non-str input), so no case mapping is done here.
#[pyfunction]
#[pyo3(signature = (text, use_stemmer, lowered = false))]
fn rouge_tokenize(
    text: &Bound<'_, PyAny>,
    use_stemmer: bool,
    lowered: bool,
) -> PyResult<Vec<String>> {
    let py = text.py();
    let s = one_text(text, "text")?;
    Ok(if detach_one(py, s.len(), DocCost::Light) {
        py.detach(|| kernel::tokenize::tokenize(&s, use_stemmer, !lowered))
    } else {
        kernel::tokenize::tokenize(&s, use_stemmer, !lowered)
    })
}

/// nltk's `PorterStemmer().stem(word)` for an ASCII `word` (for tests and
/// parity checks; ROUGE stems inside the tokenizer).
#[pyfunction]
fn rouge_stem(word: &str) -> PyResult<String> {
    if !word.is_ascii() {
        return Err(PyValueError::new_err("rouge_stem: word must be ASCII"));
    }
    let stemmed = kernel::porter::stem(word.as_bytes());
    Ok(String::from_utf8_lossy(&stemmed).into_owned())
}

fn score_one<S: AsRef<str>>(
    scorer: &Scorer,
    target: &str,
    prediction: &str,
    sentences: Option<(&[S], &[S])>,
) -> Vec<Score> {
    match sentences {
        Some((t, p)) => scorer.score_with_sentences(target, prediction, t, p),
        None => scorer.score(target, prediction),
    }
}

/// `RougeScorer(rouge_types, use_stemmer).score(target, prediction)` with the
/// default tokenizer; `None` if `rouge_types` need rouge-score's Python loop.
/// The sentences are rougeLsum's when `split_summaries=True` (split by nltk in
/// Python); both or neither.
#[pyfunction]
#[pyo3(signature = (rouge_types, target, prediction, use_stemmer, target_sentences = None, prediction_sentences = None))]
fn rouge_score<'py>(
    py: Python<'py>,
    rouge_types: &Bound<'py, PyAny>,
    target: &Bound<'py, PyAny>,
    prediction: &Bound<'py, PyAny>,
    use_stemmer: bool,
    target_sentences: Option<Vec<PyBackedStr>>,
    prediction_sentences: Option<Vec<PyBackedStr>>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let Some((types, keys)) = parse_types(rouge_types)? else {
        return Ok(None);
    };
    let builder = ScoreBuilder::get(py)?;
    let target = one_text(target, "target")?;
    let prediction = one_text(prediction, "prediction")?;
    let sentences = match (&target_sentences, &prediction_sentences) {
        (Some(t), Some(p)) => Some((t.as_slice(), p.as_slice())),
        _ => None,
    };
    let scorer = Scorer::new(types, use_stemmer);
    let run = || score_one(&scorer, &target, &prediction, sentences);
    let results = if detach_one(py, target.len() + prediction.len(), DocCost::Light) {
        py.detach(run)
    } else {
        run()
    };
    Ok(Some(builder.dict(&keys, &results)?))
}

fn batch_len(input: &TextInput) -> usize {
    match input {
        TextInput::Str(_) => 1,
        TextInput::List(d) | TextInput::Tuple(d) => d.len(),
        TextInput::Arrow(a) => a.len(),
    }
}

/// `[RougeScorer(...).score(t, p) for t, p in zip(targets, predictions)]`
/// with the default tokenizer, on the shared pool; `None` if `rouge_types`
/// need rouge-score's Python loop. A pair with an Arrow null on either side
/// gives `None`.
#[pyfunction]
#[pyo3(signature = (rouge_types, targets, predictions, use_stemmer, target_sentences = None, prediction_sentences = None, *, threads = None))]
#[allow(clippy::too_many_arguments)]
fn rouge_score_batch<'py>(
    py: Python<'py>,
    rouge_types: &Bound<'py, PyAny>,
    targets: &Bound<'py, PyAny>,
    predictions: &Bound<'py, PyAny>,
    use_stemmer: bool,
    target_sentences: Option<Vec<Vec<PyBackedStr>>>,
    prediction_sentences: Option<Vec<Vec<PyBackedStr>>>,
    threads: Option<ThreadsArg>,
) -> PyResult<Option<Bound<'py, PyList>>> {
    let Some((types, keys)) = parse_types(rouge_types)? else {
        return Ok(None);
    };
    let builder = ScoreBuilder::get(py)?;
    let for_texts = ThreadsArg::for_texts(threads.as_ref());
    let targets = batch_arg(targets, for_texts, "targets")?;
    let predictions = batch_arg(predictions, for_texts, "predictions")?;
    let t = crate::input::threads(ThreadsArg::resolve(threads)?);
    let n = batch_len(&targets);
    if batch_len(&predictions) != n {
        return Err(PyValueError::new_err(format!(
            "score_batch: targets and predictions must have the same length, got {n} and {}",
            batch_len(&predictions)
        )));
    }
    let sentences = match (&target_sentences, &prediction_sentences) {
        (Some(ts), Some(ps)) => {
            if ts.len() != n || ps.len() != n {
                return Err(PyValueError::new_err(
                    "score_batch: one sentence list per pair is needed",
                ));
            }
            Some((ts, ps))
        }
        _ => None,
    };
    let scorer = Scorer::new(types, use_stemmer);
    let results = py
        .detach(|| -> Result<Vec<Option<Vec<Score>>>, BatchError> {
            let tr = rows(&targets)?;
            let pr = rows(&predictions)?;
            Ok(par::map(n, t, |i| {
                let (target, prediction) = (tr[i]?, pr[i]?);
                let s = sentences.map(|(ts, ps)| (ts[i].as_slice(), ps[i].as_slice()));
                Some(score_one(&scorer, target, prediction, s))
            })?)
        })
        .map_err(|e| e.into_pyerr(py))?;
    let dicts = results
        .iter()
        .map(|s| match s {
            Some(s) => Ok(builder.dict(&keys, s)?.into_any()),
            None => Ok(py.None().into_bound(py)),
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(Some(PyList::new(py, dicts)?))
}

/// A token list (exact list or tuple of str instances) as UTF-8; `None` if
/// it is anything else or holds a lone surrogate (rouge-score's own code
/// then scores it).
fn token_list(obj: &Bound<'_, PyAny>) -> PyResult<Option<Vec<PyBackedStr>>> {
    let items: Vec<Bound<'_, PyAny>> = if let Ok(l) = obj.cast_exact::<PyList>() {
        l.iter().collect()
    } else if let Ok(t) = obj.cast_exact::<PyTuple>() {
        t.iter().collect()
    } else {
        return Ok(None);
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let Ok(s) = item.cast_into::<PyString>() else {
            return Ok(None);
        };
        match PyBackedStr::try_from(s) {
            Ok(s) => out.push(s),
            Err(e) if e.is_instance_of::<PyUnicodeEncodeError>(obj.py()) => return Ok(None),
            Err(e) => return Err(e),
        }
    }
    Ok(Some(out))
}

/// A list of token lists; `None` as for [`token_list`].
fn sentence_lists(obj: &Bound<'_, PyAny>) -> PyResult<Option<Vec<Vec<PyBackedStr>>>> {
    let items: Vec<Bound<'_, PyAny>> = if let Ok(l) = obj.cast_exact::<PyList>() {
        l.iter().collect()
    } else if let Ok(t) = obj.cast_exact::<PyTuple>() {
        t.iter().collect()
    } else {
        return Ok(None);
    };
    let mut out = Vec::with_capacity(items.len());
    for item in &items {
        let Some(tokens) = token_list(item)? else {
            return Ok(None);
        };
        out.push(tokens);
    }
    Ok(Some(out))
}

enum Job {
    N(usize, Vec<PyBackedStr>, Vec<PyBackedStr>),
    L(Vec<PyBackedStr>, Vec<PyBackedStr>),
    LSum(Vec<Vec<PyBackedStr>>, Vec<Vec<PyBackedStr>>),
    /// Not token lists of strs: the Python side delegates it.
    Unsupported,
}

impl Job {
    fn run(&self) -> Option<Score> {
        Some(match self {
            Self::N(n, t, p) => kernel::rouge_n_tokens(t, p, *n),
            Self::L(t, p) => kernel::rouge_l_tokens(t, p),
            Self::LSum(t, p) => kernel::rouge_lsum_tokens(t, p),
            Self::Unsupported => return None,
        })
    }
}

/// Scores from token lists, for custom tokenizers: each job is
/// `(kind, target, prediction)` with `kind` = n >= 1 (rouge-n, token lists),
/// 0 (rougeL, token lists) or -1 (rougeLsum, lists of token lists). Returns a
/// `Score` per job, or `None` where the tokens are not lists/tuples of strs
/// (or hold lone surrogates).
#[pyfunction]
#[pyo3(signature = (jobs, *, threads = None))]
fn rouge_score_tokens<'py>(
    py: Python<'py>,
    jobs: Vec<(i64, Bound<'py, PyAny>, Bound<'py, PyAny>)>,
    threads: Option<Threads>,
) -> PyResult<Bound<'py, PyList>> {
    let builder = ScoreBuilder::get(py)?;
    let mut work = Vec::with_capacity(jobs.len());
    for (kind, t, p) in &jobs {
        work.push(match *kind {
            -1 => match (sentence_lists(t)?, sentence_lists(p)?) {
                (Some(t), Some(p)) => Job::LSum(t, p),
                _ => Job::Unsupported,
            },
            0.. => match (token_list(t)?, token_list(p)?) {
                (Some(t), Some(p)) if *kind == 0 => Job::L(t, p),
                (Some(t), Some(p)) => Job::N(usize::try_from(*kind).unwrap_or(usize::MAX), t, p),
                _ => Job::Unsupported,
            },
            _ => return Err(PyValueError::new_err(format!("bad job kind {kind}"))),
        });
    }
    let t = crate::input::threads(threads);
    let scores = py
        .detach(|| par::map(work.len(), t, |i| work[i].run()))
        .map_err(|e| BatchError::from(e).into_pyerr(py))?;
    let out = scores
        .into_iter()
        .map(|s| match s {
            Some(s) => builder.build(s),
            None => Ok(py.None().into_bound(py)),
        })
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, out)
}

/// Add this half's functions and classes to the shared `pyroid.metrics` native module.
pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rouge_tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(rouge_stem, m)?)?;
    m.add_function(wrap_pyfunction!(rouge_score, m)?)?;
    m.add_function(wrap_pyfunction!(rouge_score_batch, m)?)?;
    m.add_function(wrap_pyfunction!(rouge_score_tokens, m)?)?;
    Ok(())
}
