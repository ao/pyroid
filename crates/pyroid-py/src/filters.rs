// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace; modified.
//! `pyroid._pyroid.filters`: native entry points behind `pyroid.filters`.
//!
//! The typed Python module (`python/pyroid/filters.py`) owns datatrove's
//! signatures, defaults and value checks, and builds one [`NativePipeline`]
//! per distinct configuration. This module runs it over a batch: detached,
//! on the shared pool, one pass per document with a shared tokenization.
//!
//! For languages pyroid does not tokenize (and `python_tokenizer=True`), the
//! shim tokenizes with datatrove's own tokenizer in Python and hands the words
//! to [`_run_list`], which runs the same filter math on them
//! ([`Pipeline::run_with_words`]); [`_to_list`] and [`_pack`] move documents
//! between the caller's container kind and the Python lists that path works on.

use std::borrow::Cow;
use std::sync::{Mutex, PoisonError};

use pyo3::exceptions::{PyNotImplementedError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyBool, PyDict, PyInt, PyList, PyNone, PyString, PyTuple};
use pyroid_core::columnar::{BoolBuffers, StringBuffers};
use pyroid_core::filters::{
    self, C4Quality, FilterError, FineWebQuality, GopherQuality, GopherRepetition, Outcome,
    Pipeline, Step, WordTokenizer,
};
use pyroid_core::ftfy::UnicodeVersion;
use pyroid_core::par::{self, PoolError};

use crate::arrow::{Chunk, Column, ColumnType, StrChunk, StrType, build_error};
use crate::batch::{BatchError, DocCost, detach_one};
use crate::errors::{input_error, type_name, with_note};
use crate::input::{TextInput, ThreadsArg, texts_and_threads};

/// The stand-in tokenizer of the steps whose words come from Python
/// (datatrove's own tokenizer, run by the shim). Asking it for words is the
/// signal ([`FilterError::NeedsTokenizer`]) that a document reached a check
/// that needs them; it only supplies the Unicode version otherwise.
#[derive(Debug)]
struct PythonWords(UnicodeVersion);

impl WordTokenizer for PythonWords {
    fn unicode_version(&self) -> UnicodeVersion {
        self.0
    }

    fn words<'a>(&self, _text: &'a str, _out: &mut Vec<&'a str>) -> Result<(), FilterError> {
        Err(FilterError::NeedsTokenizer)
    }

    fn sentence_ends(&self, _text: &str, _out: &mut Vec<usize>) -> Result<(), FilterError> {
        Err(FilterError::NeedsTokenizer)
    }
}

/// The [`PythonWords`] of `v` (one per Unicode version, made on first use).
fn python_words(v: UnicodeVersion) -> &'static dyn WordTokenizer {
    static MADE: Mutex<Vec<&'static PythonWords>> = Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(t) = made.iter().find(|t| t.0 == v) {
        return *t;
    }
    let t: &'static PythonWords = Box::leak(Box::new(PythonWords(v)));
    made.push(t);
    t
}

/// datatrove's `load_word_tokenizer` error for a code it does not know.
fn unknown_language(language: &str) -> PyErr {
    PyValueError::new_err(format!(
        "Language '{language}' doesn't have a tokenizer assigned. Pass in a WordTokenizer \
         directly or update tokenizer_assignment.csv"
    ))
}

/// A validated filter chain, built once per distinct config by the shim.
#[pyclass(frozen, module = "pyroid.filters", name = "_Pipeline")]
#[derive(Debug)]
pub struct NativePipeline {
    pipeline: Pipeline,
}

fn opt_f64(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<f64>> {
    match d.get_item(key)? {
        None => Err(PyValueError::new_err(format!("missing {key}"))),
        Some(v) if v.is_none() => Ok(None),
        Some(v) => Ok(Some(v.extract()?)),
    }
}

fn item<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing {key}")))
}

fn get_bool(d: &Bound<'_, PyDict>, key: &str) -> PyResult<bool> {
    item(d, key)?.extract()
}

fn get_f64(d: &Bound<'_, PyDict>, key: &str) -> PyResult<f64> {
    item(d, key)?.extract()
}

fn get_strings(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<String>> {
    item(d, key)?.extract()
}

fn get_ngrams(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<(usize, f64)>> {
    item(d, key)?.extract()
}

fn step_from(kind: &str, p: &Bound<'_, PyDict>) -> PyResult<Step> {
    Ok(match kind {
        "gopher_quality" => Step::GopherQuality(GopherQuality {
            min_doc_words: opt_f64(p, "min_doc_words")?,
            max_doc_words: opt_f64(p, "max_doc_words")?,
            min_avg_word_length: opt_f64(p, "min_avg_word_length")?,
            max_avg_word_length: opt_f64(p, "max_avg_word_length")?,
            max_symbol_word_ratio: opt_f64(p, "max_symbol_word_ratio")?,
            max_bullet_lines_ratio: opt_f64(p, "max_bullet_lines_ratio")?,
            max_ellipsis_lines_ratio: opt_f64(p, "max_ellipsis_lines_ratio")?,
            max_non_alpha_words_ratio: opt_f64(p, "max_non_alpha_words_ratio")?,
            min_stop_words: opt_f64(p, "min_stop_words")?,
            stop_words: get_strings(p, "stop_words")?,
        }),
        "gopher_repetition" => Step::GopherRepetition(GopherRepetition {
            dup_line_frac: opt_f64(p, "dup_line_frac")?,
            dup_para_frac: opt_f64(p, "dup_para_frac")?,
            dup_line_char_frac: opt_f64(p, "dup_line_char_frac")?,
            dup_para_char_frac: opt_f64(p, "dup_para_char_frac")?,
            top_n_grams: get_ngrams(p, "top_n_grams")?,
            dup_n_grams: get_ngrams(p, "dup_n_grams")?,
        }),
        "c4_quality" => Step::C4Quality(C4Quality {
            split_paragraph: get_bool(p, "split_paragraph")?,
            remove_citations: get_bool(p, "remove_citations")?,
            filter_no_terminal_punct: get_bool(p, "filter_no_terminal_punct")?,
            min_num_sentences: get_f64(p, "min_num_sentences")?,
            min_words_per_line: get_f64(p, "min_words_per_line")?,
            max_word_length: get_f64(p, "max_word_length")?,
            filter_lorem_ipsum: get_bool(p, "filter_lorem_ipsum")?,
            filter_javascript: get_bool(p, "filter_javascript")?,
            filter_curly_bracket: get_bool(p, "filter_curly_bracket")?,
            filter_policy: get_bool(p, "filter_policy")?,
        }),
        "fineweb_quality" => Step::FineWebQuality(FineWebQuality {
            line_punct_thr: get_f64(p, "line_punct_thr")?,
            line_punct_exclude_zero: get_bool(p, "line_punct_exclude_zero")?,
            stop_chars: item(p, "stop_chars").and_then(|v| v.extract())?,
            short_line_thr: get_f64(p, "short_line_thr")?,
            short_line_length: get_f64(p, "short_line_length")?,
            char_duplicates_ratio: get_f64(p, "char_duplicates_ratio")?,
            new_line_ratio: get_f64(p, "new_line_ratio")?,
        }),
        other => return Err(PyValueError::new_err(format!("unknown filter {other:?}"))),
    })
}

fn version_arg(unicode_version: &str) -> PyResult<UnicodeVersion> {
    UnicodeVersion::from_unidata_version(unicode_version)
        .ok_or_else(|| PyValueError::new_err(format!("unicode_version={unicode_version:?}")))
}

#[pymethods]
impl NativePipeline {
    /// `steps`: `(kind, language, params)` per filter, already validated by
    /// `pyroid.filters`. `language=None` marks a step whose words come from
    /// Python ([`_run_list`]'s `words`); C4 steps can't be such steps.
    #[new]
    #[pyo3(signature = (steps, unicode_version))]
    fn new(
        steps: Vec<(String, Option<String>, Bound<'_, PyDict>)>,
        unicode_version: &str,
    ) -> PyResult<Self> {
        let version = version_arg(unicode_version)?;
        let mut out = Vec::with_capacity(steps.len());
        for (kind, language, params) in &steps {
            let tok: &'static dyn WordTokenizer = match language {
                None if kind == "c4_quality" => {
                    return Err(PyValueError::new_err(
                        "a C4 step needs a native tokenizer (it splits sentences itself)",
                    ));
                }
                None => python_words(version),
                Some(language) => {
                    filters::tokenizer_for(language, version).ok_or_else(
                        || match filters::backend(language) {
                            None => unknown_language(language),
                            Some(b) => PyNotImplementedError::new_err(format!(
                                "pyroid has no native word tokenizer for language {language:?} \
                             (datatrove's {}({:?}))",
                                b.class, b.arg
                            )),
                        },
                    )?
                }
            };
            out.push((step_from(kind, params)?, tok));
        }
        Ok(Self {
            pipeline: Pipeline { steps: out },
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "<pyroid.filters native pipeline of {} steps>",
            self.pipeline.steps.len()
        )
    }
}

/// The datatrove language codes the English tokenizer serves.
#[pyfunction]
fn english_languages() -> Vec<&'static str> {
    filters::english_languages().to_vec()
}

/// Every datatrove language code: `(code, tokenizer class, its argument,
/// tokenized natively by pyroid)`, sorted by code.
#[pyfunction]
fn languages(
    unicode_version: &str,
) -> PyResult<Vec<(&'static str, &'static str, &'static str, bool)>> {
    let v = version_arg(unicode_version)?;
    Ok(filters::languages()
        .map(|(code, b)| {
            (
                code,
                b.class,
                b.arg,
                filters::tokenizer_for(code, v).is_some(),
            )
        })
        .collect())
}

/// datatrove's tokenizer for one code: `(class, argument, native)`, or
/// `None` if datatrove does not know the code.
#[pyfunction]
fn backend(
    language: &str,
    unicode_version: &str,
) -> PyResult<Option<(&'static str, &'static str, bool)>> {
    let v = version_arg(unicode_version)?;
    Ok(filters::backend(language).map(|b| {
        (
            b.class,
            b.arg,
            filters::tokenizer_for(language, v).is_some(),
        )
    }))
}

/// Why a run failed inside the detached region.
#[derive(Debug)]
enum RunError {
    Batch(BatchError),
    Filter { at: String, err: FilterError },
}

impl From<PoolError> for RunError {
    fn from(e: PoolError) -> Self {
        Self::Batch(BatchError::from(e))
    }
}

impl RunError {
    fn into_pyerr(self, py: Python<'_>) -> PyErr {
        match self {
            Self::Batch(e) => e.into_pyerr(py),
            Self::Filter { at, err } => {
                let base = match err {
                    FilterError::ZeroDivision => PyZeroDivisionError::new_err(err.to_string()),
                    FilterError::TextTooLong(_)
                    | FilterError::SubstringNotFound
                    | FilterError::NeedsTokenizer => PyValueError::new_err(err.to_string()),
                };
                with_note(py, base, format!("pyroid.filters: raised by {at}"))
            }
        }
    }
}

/// What to return.
#[derive(Clone, Copy, PartialEq, Eq)]
enum What {
    Keep,
    Reason,
    Text,
    All,
}

impl What {
    fn parse(s: &str) -> PyResult<Self> {
        Ok(match s {
            "keep" => Self::Keep,
            "reason" => Self::Reason,
            "text" => Self::Text,
            "all" => Self::All,
            _ => return Err(PyValueError::new_err(format!("what={s:?}"))),
        })
    }
}

fn run_chunk<'c>(
    p: &Pipeline,
    chunk: &'c StrChunk,
    c: usize,
    threads: par::Threads,
) -> Result<Vec<Option<Outcome<'c>>>, RunError> {
    par::try_map(chunk.len(), threads, |row| match chunk.get(row) {
        None => Ok(None),
        Some(Ok(s)) => p.run(s).map(Some).map_err(|err| RunError::Filter {
            at: format!("row {row} of chunk {c} of the Arrow input"),
            err,
        }),
        Some(Err(err)) => Err(RunError::Batch(BatchError::BadUtf8 {
            chunk: c,
            row,
            bytes: chunk.bytes(row).unwrap_or_default().to_vec(),
            err,
        })),
    })
}

/// The Python str for an outcome's text: the original object if unchanged.
#[allow(clippy::ptr_arg)] // `Borrowed` (unchanged) vs `Owned` is the signal
fn text_obj<'py>(
    py: Python<'py>,
    original: &PyBackedStr,
    out: &Cow<'_, str>,
) -> Bound<'py, PyString> {
    match out {
        Cow::Borrowed(_) => original.into_pyobject(py).unwrap_or_else(|e| match e {}),
        Cow::Owned(s) => PyString::new(py, s),
    }
}

fn reason_obj<'py>(py: Python<'py>, o: &Outcome<'_>) -> Bound<'py, PyAny> {
    match o.reason {
        Some(r) => PyString::new(py, &r.as_str()).into_any(),
        None => PyNone::get(py).to_owned().into_any(),
    }
}

fn seq<'py>(
    py: Python<'py>,
    tuple: bool,
    items: impl ExactSizeIterator<Item = Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(if tuple {
        PyTuple::new(py, items)?.into_any()
    } else {
        PyList::new(py, items)?.into_any()
    })
}

fn pack<'py>(
    py: Python<'py>,
    what: What,
    keep: impl FnOnce() -> PyResult<Bound<'py, PyAny>>,
    reason: impl FnOnce() -> PyResult<Bound<'py, PyAny>>,
    text: impl FnOnce() -> PyResult<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(match what {
        What::Keep => keep()?,
        What::Reason => reason()?,
        What::Text => text()?,
        What::All => PyTuple::new(py, [keep()?, reason()?, text()?])?.into_any(),
    })
}

/// Run `pipeline` over `x`; return `what` (`"keep"`, `"reason"`, `"text"`
/// or `"all"` = `(keep, reason, text)`) in the caller's container kind.
#[pyfunction]
#[pyo3(signature = (texts, pipeline, what, /, *, threads = None))]
fn run<'py>(
    py: Python<'py>,
    texts: &Bound<'py, PyAny>,
    pipeline: &NativePipeline,
    what: &str,
    threads: Option<ThreadsArg>,
) -> PyResult<Bound<'py, PyAny>> {
    let (texts, threads) = texts_and_threads(texts, threads, None)?;
    let what = What::parse(what)?;
    let n_threads = crate::input::threads(threads);
    let p = &pipeline.pipeline;
    match &texts {
        TextInput::Str(s) => {
            let res = if detach_one(py, s.len(), DocCost::Heavy) {
                py.detach(|| p.run(s))
            } else {
                p.run(s)
            };
            let o = res.map_err(|err| {
                RunError::Filter {
                    at: "the str".to_owned(),
                    err,
                }
                .into_pyerr(py)
            })?;
            pack(
                py,
                what,
                || Ok(PyBool::new(py, o.reason.is_none()).to_owned().into_any()),
                || Ok(reason_obj(py, &o)),
                || Ok(text_obj(py, s, &o.text).into_any()),
            )
        }
        TextInput::List(docs) | TextInput::Tuple(docs) => {
            let tuple = matches!(texts, TextInput::Tuple(_));
            let outs = py
                .detach(|| {
                    par::try_map(docs.len(), n_threads, |i| {
                        p.run(&docs[i]).map_err(|err| RunError::Filter {
                            at: format!("the str at index {i}"),
                            err,
                        })
                    })
                })
                .map_err(|e| e.into_pyerr(py))?;
            pack(
                py,
                what,
                || {
                    seq(
                        py,
                        tuple,
                        outs.iter()
                            .map(|o| PyBool::new(py, o.reason.is_none()).to_owned().into_any()),
                    )
                },
                || seq(py, tuple, outs.iter().map(|o| reason_obj(py, o))),
                || {
                    seq(
                        py,
                        tuple,
                        docs.iter()
                            .zip(&outs)
                            .map(|(d, o)| text_obj(py, d, &o.text).into_any()),
                    )
                },
            )
        }
        TextInput::Arrow(strings) => run_arrow(py, strings, p, what, n_threads),
    }
}

fn run_arrow<'py>(
    py: Python<'py>,
    strings: &crate::arrow::ArrowStrings,
    p: &Pipeline,
    what: What,
    n_threads: par::Threads,
) -> PyResult<Bound<'py, PyAny>> {
    let large = strings.field.ty == StrType::LargeUtf8;
    let name = strings.field.name.as_str();
    let chunks = py
        .detach(|| {
            strings
                .chunks
                .iter()
                .enumerate()
                .map(|(c, chunk)| run_chunk(p, chunk, c, n_threads))
                .collect::<Result<Vec<_>, RunError>>()
        })
        .map_err(|e| e.into_pyerr(py))?;
    let keep = || -> PyResult<Bound<'py, PyAny>> {
        let out = chunks
            .iter()
            .map(|rows| {
                let mask: Vec<Option<bool>> = rows
                    .iter()
                    .map(|o| o.as_ref().map(|o| o.reason.is_none()))
                    .collect();
                Chunk::Bool(BoolBuffers::build(&mask))
            })
            .collect();
        Ok(Bound::new(py, Column::new(ColumnType::Bool, name, out))?.into_any())
    };
    let reason = || -> PyResult<Bound<'py, PyAny>> {
        let mut out = Vec::new();
        for rows in &chunks {
            let vals: Vec<Option<Cow<'static, str>>> = rows
                .iter()
                .map(|o| o.as_ref().and_then(|o| o.reason.map(|r| r.as_str())))
                .collect();
            let built = StringBuffers::build_split(&vals, n_threads).map_err(build_error)?;
            out.extend(built.into_iter().map(Chunk::Utf8));
        }
        Ok(Bound::new(py, Column::new(ColumnType::Utf8, name, out))?.into_any())
    };
    let text = || -> PyResult<Bound<'py, PyAny>> {
        let mut out = Vec::new();
        for rows in &chunks {
            let vals: Vec<Option<&str>> =
                rows.iter().map(|o| o.as_ref().map(|o| &*o.text)).collect();
            if large {
                out.push(Chunk::LargeUtf8(
                    StringBuffers::build(&vals, n_threads).map_err(build_error)?,
                ));
            } else {
                let built = StringBuffers::build_split(&vals, n_threads).map_err(build_error)?;
                out.extend(built.into_iter().map(Chunk::Utf8));
            }
        }
        let ty = if large {
            ColumnType::LargeUtf8
        } else {
            ColumnType::Utf8
        };
        Ok(Bound::new(py, Column::new(ty, name, out))?.into_any())
    };
    pack(py, what, keep, reason, text)
}

// ---- the Python-tokenizer path -------------------------------------------------

/// [`_run_list`]'s status for a document that reached a check needing words
/// it was not given (besides kept, `None`, and dropped, the reason `str`).
const NEEDS_WORDS: u8 = 0;

/// One document's words: a list or tuple of str.
fn word_list(obj: &Bound<'_, PyAny>, i: usize) -> PyResult<Option<Vec<PyBackedStr>>> {
    let py = obj.py();
    if obj.is_none() {
        return Ok(None);
    }
    let items: Vec<Bound<'_, PyAny>> = if let Ok(l) = obj.cast::<PyList>() {
        l.iter().collect()
    } else if let Ok(t) = obj.cast::<PyTuple>() {
        t.iter().collect()
    } else {
        return Err(input_error(
            py,
            format!(
                "the word tokenizer returned {} for document {i}; expected a list of str",
                type_name(obj)
            ),
        ));
    };
    items
        .iter()
        .map(|w| {
            let Ok(s) = w.cast::<PyString>() else {
                return Err(input_error(
                    py,
                    format!(
                        "the word tokenizer returned a {} word for document {i}; expected str",
                        type_name(w)
                    ),
                ));
            };
            PyBackedStr::try_from(s.clone())
                .map_err(|e| with_note(py, e, format!("pyroid.filters: words of document {i}")))
        })
        .collect::<PyResult<Vec<_>>>()
        .map(Some)
}

/// Run `pipeline` over `texts` (a list of str), on the shared pool.
///
/// Without `words`, the Python-word steps (`language=None`) report the
/// documents that reach a check needing words (status `0`); the others are
/// done. With `words` (per document a list of str, or `None`), those steps use
/// them ([`Pipeline::run_with_words`]). Returns `(statuses, texts)`: per
/// document `None` (kept), the drop reason (`str`) or `0`; and the text
/// afterwards, `None` when unchanged. `index` gives the documents' indices in
/// the caller's batch, for error notes.
#[pyfunction]
#[pyo3(signature = (texts, pipeline, words = None, index = None, /, *, threads = None))]
fn _run_list<'py>(
    py: Python<'py>,
    texts: &Bound<'py, PyAny>,
    pipeline: &NativePipeline,
    words: Option<&Bound<'py, PyList>>,
    index: Option<Vec<usize>>,
    threads: Option<ThreadsArg>,
) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
    let (texts, threads) = texts_and_threads(texts, threads, None)?;
    let TextInput::List(docs) = &texts else {
        return Err(input_error(py, "expected a list of str"));
    };
    let name = |i: usize| {
        index
            .as_ref()
            .and_then(|ix| ix.get(i).copied())
            .unwrap_or(i)
    };
    let words: Option<Vec<Option<Vec<PyBackedStr>>>> = words
        .map(|w| {
            if w.len() != docs.len() {
                return Err(PyValueError::new_err(format!(
                    "{} word lists for {} documents",
                    w.len(),
                    docs.len()
                )));
            }
            w.iter()
                .enumerate()
                .map(|(i, obj)| word_list(&obj, name(i)))
                .collect()
        })
        .transpose()?;
    let p = &pipeline.pipeline;
    let n_threads = crate::input::threads(threads);
    // `None`: needs words.
    let outcomes: Vec<Option<Outcome<'_>>> = py
        .detach(|| {
            par::try_map(docs.len(), n_threads, |i| {
                let text: &str = &docs[i];
                let res = match words.as_ref().and_then(|w| w[i].as_ref()) {
                    None => p.run(text),
                    Some(w) => p.run_with_words(text, w.iter().map(|s| &**s).collect()),
                };
                match res {
                    Ok(o) => Ok(Some(o)),
                    Err(FilterError::NeedsTokenizer) => Ok(None),
                    Err(err) => Err(RunError::Filter {
                        at: format!("document {}", name(i)),
                        err,
                    }),
                }
            })
        })
        .map_err(|e| e.into_pyerr(py))?;
    let codes = PyList::new(
        py,
        outcomes.iter().map(|o| match o {
            Some(o) => reason_obj(py, o),
            None => PyInt::new(py, NEEDS_WORDS).into_any(),
        }),
    )?;
    let new_texts = PyList::new(
        py,
        outcomes.iter().map(|o| match o {
            Some(Outcome {
                text: Cow::Owned(t),
                ..
            }) => PyString::new(py, t).into_any(),
            _ => PyNone::get(py).to_owned().into_any(),
        }),
    )?;
    Ok((codes, new_texts))
}

/// The container kind of a call's documents, for [`_pack`].
#[derive(Debug)]
enum ShapeKind {
    Str,
    List,
    Tuple,
    Arrow {
        name: String,
        large: bool,
        lens: Vec<usize>,
    },
}

/// What [`_to_list`] read the documents from (opaque to Python).
#[pyclass(frozen, module = "pyroid.filters", name = "_Shape")]
#[derive(Debug)]
pub struct Shape(ShapeKind);

#[pymethods]
impl Shape {
    fn __repr__(&self) -> String {
        format!("<pyroid.filters shape {:?}>", self.0)
    }
}

/// The documents of `texts` as a list (the original str objects; Arrow rows
/// decoded, nulls as `None`) and their [`Shape`].
#[pyfunction]
#[pyo3(signature = (texts, /, *, threads = None))]
fn _to_list<'py>(
    py: Python<'py>,
    texts: &Bound<'py, PyAny>,
    threads: Option<ThreadsArg>,
) -> PyResult<(Bound<'py, PyList>, Shape)> {
    let (texts, _) = texts_and_threads(texts, threads, None)?;
    Ok(match &texts {
        TextInput::Str(s) => (PyList::new(py, [s])?, Shape(ShapeKind::Str)),
        TextInput::List(d) => (PyList::new(py, d)?, Shape(ShapeKind::List)),
        TextInput::Tuple(d) => (PyList::new(py, d)?, Shape(ShapeKind::Tuple)),
        TextInput::Arrow(strings) => {
            let mut out = Vec::with_capacity(strings.len());
            for (c, chunk) in strings.chunks.iter().enumerate() {
                for row in 0..chunk.len() {
                    out.push(match chunk.get(row) {
                        None => PyNone::get(py).to_owned().into_any(),
                        Some(Ok(s)) => PyString::new(py, s).into_any(),
                        Some(Err(err)) => {
                            return Err(BatchError::BadUtf8 {
                                chunk: c,
                                row,
                                bytes: chunk.bytes(row).unwrap_or_default().to_vec(),
                                err,
                            }
                            .into_pyerr(py));
                        }
                    });
                }
            }
            let shape = ShapeKind::Arrow {
                name: strings.field.name.clone(),
                large: strings.field.ty == StrType::LargeUtf8,
                lens: strings.chunks.iter().map(StrChunk::len).collect(),
            };
            (PyList::new(py, out)?, Shape(shape))
        }
    })
}

fn opt_strs(list: &Bound<'_, PyList>) -> PyResult<Vec<Option<PyBackedStr>>> {
    list.iter()
        .map(|v| {
            if v.is_none() {
                Ok(None)
            } else {
                v.extract::<PyBackedStr>().map(Some)
            }
        })
        .collect()
}

/// Per-document results (lists, one entry per document, as [`_to_list`]
/// gave them) in the caller's kind: `keep` (bool, `None` for Arrow nulls),
/// `reason` (str or `None`) and `text` (str or `None`), packed as
/// [`run`] packs them for `what`.
#[pyfunction]
#[pyo3(signature = (shape, what, keep, reason, text, /, *, threads = None))]
fn _pack<'py>(
    py: Python<'py>,
    shape: &Shape,
    what: &str,
    keep: &Bound<'py, PyList>,
    reason: &Bound<'py, PyList>,
    text: &Bound<'py, PyList>,
    threads: Option<crate::input::Threads>,
) -> PyResult<Bound<'py, PyAny>> {
    let what = What::parse(what)?;
    let n_threads = crate::input::threads(threads);
    let n = keep.len();
    if reason.len() != n || text.len() != n {
        return Err(PyValueError::new_err(
            "keep, reason and text differ in length",
        ));
    }
    match &shape.0 {
        ShapeKind::Str => {
            if n != 1 {
                return Err(PyValueError::new_err("a str has one document"));
            }
            pack(
                py,
                what,
                || keep.get_item(0),
                || reason.get_item(0),
                || text.get_item(0),
            )
        }
        ShapeKind::List => pack(
            py,
            what,
            || Ok(PyList::new(py, keep.iter())?.into_any()),
            || Ok(PyList::new(py, reason.iter())?.into_any()),
            || Ok(PyList::new(py, text.iter())?.into_any()),
        ),
        ShapeKind::Tuple => pack(
            py,
            what,
            || Ok(PyTuple::new(py, keep.iter())?.into_any()),
            || Ok(PyTuple::new(py, reason.iter())?.into_any()),
            || Ok(PyTuple::new(py, text.iter())?.into_any()),
        ),
        ShapeKind::Arrow { name, large, lens } => {
            if lens.iter().sum::<usize>() != n {
                return Err(PyValueError::new_err(
                    "result length differs from the input's",
                ));
            }
            let ranges = || {
                lens.iter().scan(0usize, |start, &len| {
                    let r = *start..*start + len;
                    *start += len;
                    Some(r)
                })
            };
            let keep_col = || -> PyResult<Bound<'py, PyAny>> {
                let mask: Vec<Option<bool>> = keep
                    .iter()
                    .map(|v| {
                        if v.is_none() {
                            Ok(None)
                        } else {
                            v.extract().map(Some)
                        }
                    })
                    .collect::<PyResult<_>>()?;
                let chunks = ranges()
                    .map(|r| Chunk::Bool(BoolBuffers::build(&mask[r])))
                    .collect();
                Ok(Bound::new(py, Column::new(ColumnType::Bool, name, chunks))?.into_any())
            };
            let str_col = |list: &Bound<'py, PyList>, large: bool| -> PyResult<Bound<'py, PyAny>> {
                let vals = opt_strs(list)?;
                let mut chunks = Vec::new();
                for r in ranges() {
                    if large {
                        chunks.push(Chunk::LargeUtf8(
                            StringBuffers::build(&vals[r], n_threads).map_err(build_error)?,
                        ));
                    } else {
                        let built =
                            StringBuffers::build_split(&vals[r], n_threads).map_err(build_error)?;
                        chunks.extend(built.into_iter().map(Chunk::Utf8));
                    }
                }
                let ty = if large {
                    ColumnType::LargeUtf8
                } else {
                    ColumnType::Utf8
                };
                Ok(Bound::new(py, Column::new(ty, name, chunks))?.into_any())
            };
            pack(
                py,
                what,
                keep_col,
                || str_col(reason, false),
                || str_col(text, *large),
            )
        }
    }
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pyroid.filters")?;
    m.gil_used(false)?;
    m.add_class::<NativePipeline>()?;
    m.add_class::<Shape>()?;
    m.add_function(wrap_pyfunction!(run, &m)?)?;
    m.add_function(wrap_pyfunction!(english_languages, &m)?)?;
    m.add_function(wrap_pyfunction!(languages, &m)?)?;
    m.add_function(wrap_pyfunction!(backend, &m)?)?;
    m.add_function(wrap_pyfunction!(_run_list, &m)?)?;
    m.add_function(wrap_pyfunction!(_to_list, &m)?)?;
    m.add_function(wrap_pyfunction!(_pack, &m)?)?;
    register_more(&m)?;
    parent.add_submodule(&m)
}

// ===========================================================================
// More datatrove 0.10.1 steps: one inline module per step, each registering
// its classes and functions on `pyroid._pyroid.filters`.
// ===========================================================================

fn register_more(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pii::register(m)?;
    badwords::register(m)?;
    url::register(m)?;
    symbol_lines::register(m)?;
    unigram::register(m)?;
    Ok(())
}

// ---- PIIFormatter ----
mod pii {
    use pyo3::prelude::*;

    #[allow(clippy::unnecessary_wraps)] // placeholder
    pub(super) fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        Ok(())
    }
}
// ---- end PIIFormatter ----

// ---- C4BadWordsFilter ----
mod badwords {
    use pyo3::prelude::*;

    #[allow(clippy::unnecessary_wraps)] // placeholder
    pub(super) fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        Ok(())
    }
}
// ---- end C4BadWordsFilter ----

// ---- URLFilter ----
mod url {
    use pyo3::prelude::*;

    #[allow(clippy::unnecessary_wraps)] // placeholder
    pub(super) fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        Ok(())
    }
}
// ---- end URLFilter ----

// ---- SymbolLinesFormatter ----
mod symbol_lines {
    use pyo3::prelude::*;

    #[allow(clippy::unnecessary_wraps)] // placeholder
    pub(super) fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        Ok(())
    }
}
// ---- end SymbolLinesFormatter ----

// ---- UnigramLogProbFilter ----
mod unigram {
    use pyo3::prelude::*;

    #[allow(clippy::unnecessary_wraps)] // placeholder
    pub(super) fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        Ok(())
    }
}
// ---- end UnigramLogProbFilter ----
