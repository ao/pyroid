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
        py: Python<'_>,
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
                    crate::segmenters::prepare(py, language)?;
                    filters::tokenizer_for(language, version).ok_or_else(
                        || match filters::backend(language) {
                            None => unknown_language(language),
                            Some(b) => PyNotImplementedError::new_err(format!(
                                "pyroid has no native word tokenizer for language {language:?} \
                             (datatrove's {}({:?})){}",
                                b.class,
                                b.arg,
                                crate::segmenters::hint(language)
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
    py: Python<'_>,
    unicode_version: &str,
) -> PyResult<Vec<(&'static str, &'static str, &'static str, bool)>> {
    let v = version_arg(unicode_version)?;
    let mut native = crate::segmenters::Native::new();
    filters::languages()
        .map(|(code, b)| Ok((code, b.class, b.arg, native.get(py, code, v)?)))
        .collect()
}

/// datatrove's tokenizer for one code: `(class, argument, native)`, or
/// `None` if datatrove does not know the code.
#[pyfunction]
fn backend(
    py: Python<'_>,
    language: &str,
    unicode_version: &str,
) -> PyResult<Option<(&'static str, &'static str, bool)>> {
    let v = version_arg(unicode_version)?;
    let native = crate::segmenters::Native::new().get(py, language, v)?;
    Ok(filters::backend(language).map(|b| (b.class, b.arg, native)))
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
                    | FilterError::NeedsTokenizer
                    | FilterError::Tokenizer(_) => PyValueError::new_err(err.to_string()),
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
    Ok(())
}

// ---- PIIFormatter ----
mod pii {
    //! `PIIFormatter`: the matches are found and replaced detached, in
    //! parallel; between the passes the calling thread numbers the
    //! replacements in document order (datatrove's counters carry over from
    //! one document to the next) and asks Python's `ipaddress` which
    //! addresses are global, once per distinct address.

    use std::borrow::Cow;
    use std::collections::HashMap;
    use std::ops::Range;

    use pyo3::exceptions::{PyIndexError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::{PyBool, PyList, PyNone, PyString, PyTuple};
    use pyroid_core::columnar::{BoolBuffers, StringBuffers};
    use pyroid_core::filters::pii;
    use pyroid_core::ftfy::UnicodeVersion;
    use pyroid_core::par;

    use crate::arrow::{Chunk, Column, ColumnType, StrType, build_error};
    use crate::batch::BatchError;
    use crate::input::{TextInput, ThreadsArg, texts_and_threads};

    /// Every document of `input` in order (Arrow nulls are `None`), checked
    /// to be UTF-8.
    pub(super) fn flat_docs(
        input: &TextInput,
        threads: par::Threads,
    ) -> Result<Vec<Option<&str>>, BatchError> {
        Ok(match input {
            TextInput::Str(s) => vec![Some(&**s)],
            TextInput::List(d) | TextInput::Tuple(d) => d.iter().map(|s| Some(&**s)).collect(),
            TextInput::Arrow(strings) => {
                let mut out = Vec::with_capacity(strings.len());
                for (c, chunk) in strings.chunks.iter().enumerate() {
                    out.extend(par::try_map(chunk.len(), threads, |row| {
                        chunk
                            .get(row)
                            .transpose()
                            .map_err(|err| BatchError::BadUtf8 {
                                chunk: c,
                                row,
                                bytes: chunk.bytes(row).unwrap_or_default().to_vec(),
                                err,
                            })
                    })?);
                }
                out
            }
        })
    }

    /// The transform result in the caller's kind: `outs[i]` for document
    /// `i`, `Cow::Borrowed` meaning unchanged (the original object comes
    /// back), `None` for an Arrow null.
    pub(super) fn pack_texts<'py>(
        py: Python<'py>,
        input: &TextInput,
        outs: &[Option<Cow<'_, str>>],
        threads: par::Threads,
    ) -> PyResult<Bound<'py, PyAny>> {
        let obj = |orig: &pyo3::pybacked::PyBackedStr, out: &Option<Cow<'_, str>>| match out {
            Some(Cow::Owned(s)) => PyString::new(py, s).into_any(),
            _ => orig
                .into_pyobject(py)
                .unwrap_or_else(|e| match e {})
                .into_any(),
        };
        Ok(match input {
            TextInput::Str(s) => obj(s, &outs[0]),
            TextInput::List(d) => {
                PyList::new(py, d.iter().zip(outs).map(|(s, o)| obj(s, o)))?.into_any()
            }
            TextInput::Tuple(d) => {
                PyTuple::new(py, d.iter().zip(outs).map(|(s, o)| obj(s, o)))?.into_any()
            }
            TextInput::Arrow(strings) => {
                let large = strings.field.ty == StrType::LargeUtf8;
                let mut chunks = Vec::new();
                let mut rest = outs;
                for chunk in &strings.chunks {
                    let (rows, tail) = rest.split_at(chunk.len());
                    rest = tail;
                    if large {
                        chunks.push(Chunk::LargeUtf8(
                            StringBuffers::build(rows, threads).map_err(build_error)?,
                        ));
                    } else {
                        let built =
                            StringBuffers::build_split(rows, threads).map_err(build_error)?;
                        chunks.extend(built.into_iter().map(Chunk::Utf8));
                    }
                }
                let ty = if large {
                    ColumnType::LargeUtf8
                } else {
                    ColumnType::Utf8
                };
                Bound::new(py, Column::new(ty, &strings.field.name, chunks))?.into_any()
            }
        })
    }

    /// Per-document values in the caller's kind: `item` builds one value for
    /// a str/list/tuple; Arrow gets a column built by `chunk` per input chunk
    /// (`vals` holds every row, nulls included).
    fn pack_values<'py, T>(
        py: Python<'py>,
        input: &TextInput,
        vals: &[Option<T>],
        item: impl Fn(&Option<T>) -> Bound<'py, PyAny>,
        ty: ColumnType,
        chunk: impl Fn(&[Option<T>], &mut Vec<Chunk>) -> PyResult<()>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Ok(match input {
            TextInput::Str(_) => item(&vals[0]),
            TextInput::List(_) => PyList::new(py, vals.iter().map(&item))?.into_any(),
            TextInput::Tuple(_) => PyTuple::new(py, vals.iter().map(&item))?.into_any(),
            TextInput::Arrow(strings) => {
                let mut chunks = Vec::new();
                let mut rest = vals;
                for c in &strings.chunks {
                    let (rows, tail) = rest.split_at(c.len());
                    rest = tail;
                    chunk(rows, &mut chunks)?;
                }
                Bound::new(py, Column::new(ty, &strings.field.name, chunks))?.into_any()
            }
        })
    }

    /// A mask in the caller's kind (`None`: an Arrow null).
    pub(super) fn pack_bools<'py>(
        py: Python<'py>,
        input: &TextInput,
        vals: &[Option<bool>],
    ) -> PyResult<Bound<'py, PyAny>> {
        pack_values(
            py,
            input,
            vals,
            |v| PyBool::new(py, v.unwrap_or(false)).to_owned().into_any(),
            ColumnType::Bool,
            |rows, out| {
                out.push(Chunk::Bool(BoolBuffers::build(rows)));
                Ok(())
            },
        )
    }

    /// `str | None` per document in the caller's kind (Arrow: a `string`
    /// column, null for `None` and for null inputs).
    pub(super) fn pack_reasons<'py>(
        py: Python<'py>,
        input: &TextInput,
        vals: &[Option<&'static str>],
        threads: par::Threads,
    ) -> PyResult<Bound<'py, PyAny>> {
        pack_values(
            py,
            input,
            vals,
            |v| match v {
                Some(s) => PyString::new(py, s).into_any(),
                None => PyNone::get(py).to_owned().into_any(),
            },
            ColumnType::Utf8,
            |rows, out| {
                let built = StringBuffers::build_split(rows, threads).map_err(build_error)?;
                out.extend(built.into_iter().map(Chunk::Utf8));
                Ok(())
            },
        )
    }

    /// Each document's start index into a replacement tuple of `len` items,
    /// given how many replacements each makes; and the index after the last.
    fn starts(counts: &[usize], start: usize, len: usize) -> PyResult<(Vec<usize>, usize)> {
        let mut out = Vec::with_capacity(counts.len());
        let mut cur = start;
        for &n in counts {
            out.push(cur);
            if n > 0 {
                if len == 0 {
                    return Err(PyIndexError::new_err("tuple index out of range"));
                }
                cur = (cur + n % len) % len;
            }
        }
        Ok((out, cur))
    }

    pub(super) fn pool_err(py: Python<'_>, e: par::PoolError) -> PyErr {
        BatchError::from(e).into_pyerr(py)
    }

    /// A `PIIFormatter` configuration, built once by `pyroid.filters`.
    #[pyclass(frozen, module = "pyroid.filters", name = "_PII")]
    #[derive(Debug)]
    pub(super) struct NativePii {
        remove_emails: bool,
        remove_ips: bool,
        email: Vec<String>,
        ip: Vec<String>,
        version: UnicodeVersion,
    }

    #[pymethods]
    impl NativePii {
        #[new]
        fn new(
            remove_emails: bool,
            remove_ips: bool,
            email_replacement: Vec<String>,
            ip_replacement: Vec<String>,
            unicode_version: &str,
        ) -> PyResult<Self> {
            let version =
                UnicodeVersion::from_unidata_version(unicode_version).ok_or_else(|| {
                    PyValueError::new_err(format!("unsupported Unicode version {unicode_version}"))
                })?;
            Ok(Self {
                remove_emails,
                remove_ips,
                email: email_replacement,
                ip: ip_replacement,
                version,
            })
        }

        /// `format()` over a batch, continuing the replacement counters at
        /// `email_index` / `ip_index`. `validator(addr)` decides whether an
        /// IP match that parses is replaced (`None`: all of them). Returns
        /// `(texts, email_index, ip_index)` with the counters afterwards.
        #[pyo3(signature = (texts, email_index, ip_index, validator, /, *, threads = None))]
        fn run<'py>(
            &self,
            py: Python<'py>,
            texts: &Bound<'py, PyAny>,
            email_index: usize,
            ip_index: usize,
            validator: Option<Bound<'py, PyAny>>,
            threads: Option<ThreadsArg>,
        ) -> PyResult<(Bound<'py, PyAny>, usize, usize)> {
            let (input, threads) = texts_and_threads(texts, threads, None)?;
            let threads = crate::input::threads(threads);
            let docs = py
                .detach(|| flat_docs(&input, threads))
                .map_err(|e| e.into_pyerr(py))?;
            let mut cur: Vec<Option<Cow<'_, str>>> =
                docs.iter().map(|d| d.map(Cow::Borrowed)).collect();
            let email_next = if self.remove_emails {
                self.email_pass(py, &mut cur, email_index, threads)?
            } else {
                email_index
            };
            let ip_next = if self.remove_ips {
                self.ip_pass(py, &mut cur, ip_index, validator.as_ref(), threads)?
            } else {
                ip_index
            };
            let out = pack_texts(py, &input, &cur, threads)?;
            Ok((out, email_next, ip_next))
        }
    }

    impl NativePii {
        /// Replace the emails in `cur`; returns the next email index.
        fn email_pass(
            &self,
            py: Python<'_>,
            cur: &mut [Option<Cow<'_, str>>],
            start: usize,
            threads: par::Threads,
        ) -> PyResult<usize> {
            let v = self.version;
            let found = py
                .detach(|| {
                    par::map(cur.len(), threads, |i| {
                        let mut m = Vec::new();
                        if let Some(d) = cur[i].as_deref() {
                            pii::email_matches(d, v, &mut m);
                        }
                        m
                    })
                })
                .map_err(|e| pool_err(py, e))?;
            let counts: Vec<usize> = found.iter().map(Vec::len).collect();
            let (at, next) = starts(&counts, start, self.email.len())?;
            let reps = &self.email;
            let replaced: Vec<Option<String>> = py
                .detach(|| {
                    par::map(cur.len(), threads, |i| {
                        let d = cur[i].as_deref().filter(|_| counts[i] > 0)?;
                        pii::replace(d, &found[i], |_| true, reps, at[i])
                            .ok()
                            .map(|(t, _)| t.into_owned())
                    })
                })
                .map_err(|e| pool_err(py, e))?;
            for (c, r) in cur.iter_mut().zip(replaced) {
                if let Some(t) = r {
                    *c = Some(Cow::Owned(t));
                }
            }
            Ok(next)
        }

        /// Replace the IP addresses in `cur` that `validator` accepts (every
        /// one that parses when it is `None`); returns the next IP index.
        fn ip_pass(
            &self,
            py: Python<'_>,
            cur: &mut [Option<Cow<'_, str>>],
            start: usize,
            validator: Option<&Bound<'_, PyAny>>,
            threads: par::Threads,
        ) -> PyResult<usize> {
            let found: Vec<Vec<(Range<usize>, bool)>> = py
                .detach(|| {
                    par::map(cur.len(), threads, |i| {
                        let Some(t) = cur[i].as_deref() else {
                            return Vec::new();
                        };
                        let mut m = Vec::new();
                        pii::ip_matches(t, &mut m);
                        m.into_iter()
                            .map(|r| {
                                let parses = pii::parse_ipv4(&t[r.clone()]).is_some();
                                (r, parses)
                            })
                            .collect()
                    })
                })
                .map_err(|e| pool_err(py, e))?;
            let valid = Self::validate(cur, &found, validator)?;
            let counts: Vec<usize> = valid
                .iter()
                .map(|r| r.iter().filter(|&&b| b).count())
                .collect();
            let (at, next) = starts(&counts, start, self.ip.len())?;
            let reps = &self.ip;
            let replaced: Vec<Option<String>> = py
                .detach(|| {
                    par::map(cur.len(), threads, |i| {
                        let t = cur[i].as_deref().filter(|_| counts[i] > 0)?;
                        let ranges: Vec<Range<usize>> =
                            found[i].iter().map(|(r, _)| r.clone()).collect();
                        pii::replace(t, &ranges, |k| valid[i][k], reps, at[i])
                            .ok()
                            .map(|(t, _)| t.into_owned())
                    })
                })
                .map_err(|e| pool_err(py, e))?;
            for (c, r) in cur.iter_mut().zip(replaced) {
                if let Some(t) = r {
                    *c = Some(Cow::Owned(t));
                }
            }
            Ok(next)
        }

        /// Which IP candidates are replaced: those that parse, and that
        /// `validator` accepts (asked once per distinct address).
        fn validate(
            cur: &[Option<Cow<'_, str>>],
            found: &[Vec<(Range<usize>, bool)>],
            validator: Option<&Bound<'_, PyAny>>,
        ) -> PyResult<Vec<Vec<bool>>> {
            let mut cache: HashMap<&str, bool> = HashMap::new();
            let mut valid = Vec::with_capacity(found.len());
            for (t, f) in cur.iter().zip(found) {
                let mut row = Vec::with_capacity(f.len());
                for (r, parses) in f {
                    let ok = match (validator, *parses) {
                        (_, false) => false,
                        (None, true) => true,
                        (Some(fun), true) => {
                            let addr = &t.as_deref().unwrap_or_default()[r.clone()];
                            if let Some(&b) = cache.get(addr) {
                                b
                            } else {
                                let b = fun.call1((addr,))?.is_truthy()?;
                                cache.insert(addr, b);
                                b
                            }
                        }
                    };
                    row.push(ok);
                }
                valid.push(row);
            }
            Ok(valid)
        }
    }

    pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<NativePii>()
    }
}
// ---- end PIIFormatter ----

// ---- C4BadWordsFilter ----
mod badwords {
    //! `C4BadWordsFilter`: the word search runs detached, in parallel; the
    //! keep-fraction draws then run in document order on numpy's PCG64.

    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyTuple;
    use pyroid_core::filters::badwords::{BadWords, Pcg64, seed_sequence};
    use pyroid_core::ftfy::UnicodeVersion;
    use pyroid_core::par;

    use super::pii::{flat_docs, pack_bools, pack_reasons};
    use crate::input::{ThreadsArg, texts_and_threads};

    /// datatrove's drop reason.
    const REASON: &str = "document_removed_with_badwords";

    /// A PCG64 `(state, inc)`.
    type RngState = (u128, u128);

    /// One language's compiled word list.
    #[pyclass(frozen, module = "pyroid.filters", name = "_BadWords")]
    #[derive(Debug)]
    struct NativeBadWords(BadWords);

    #[pymethods]
    impl NativeBadWords {
        #[new]
        fn new(words: Vec<String>, boundaries: bool) -> Self {
            Self(BadWords::new(&words, boundaries))
        }
    }

    /// `(state, inc)` of `numpy.random.default_rng(seed).bit_generator` for
    /// an int seed given as its 32-bit words, least significant first.
    #[pyfunction]
    fn _pcg64_seed(entropy: Vec<u32>) -> (u128, u128) {
        let g = seed_sequence(&entropy);
        (g.state, g.inc)
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Hit {
        /// No word list for the document's language: kept, no draw.
        Missing,
        No,
        Yes,
    }

    /// Run the filter. `lists[ids[i]]` (or `lists[lang]` for every
    /// document when `ids` is `None`) is document `i`'s list; a negative id
    /// means none (kept). `what` is `"keep"`, `"reason"`, `"all"` (both) or
    /// `"found"` (whether a word was found; no draws).
    /// Returns `(result, rng)` with the generator state after the draws.
    #[pyfunction]
    #[pyo3(signature = (
        texts, lists, lang, ids, keep_fraction, rng, what, unicode_version, /, *, threads = None
    ))]
    #[allow(clippy::too_many_arguments)] // one native entry point behind the Python API
    fn _badwords_run<'py>(
        py: Python<'py>,
        texts: &Bound<'py, PyAny>,
        lists: Vec<Py<NativeBadWords>>,
        lang: i64,
        ids: Option<Vec<i64>>,
        keep_fraction: f64,
        rng: Option<RngState>,
        what: &str,
        unicode_version: &str,
        threads: Option<ThreadsArg>,
    ) -> PyResult<(Bound<'py, PyAny>, Option<RngState>)> {
        let (input, threads) = texts_and_threads(texts, threads, None)?;
        let threads = crate::input::threads(threads);
        let v = UnicodeVersion::from_unidata_version(unicode_version).ok_or_else(|| {
            PyValueError::new_err(format!("unsupported Unicode version {unicode_version}"))
        })?;
        let lists: Vec<&BadWords> = lists.iter().map(|l| &l.get().0).collect();
        let pick = |i: usize| -> Option<&BadWords> {
            let id = ids.as_ref().map_or(lang, |ids| ids[i]);
            usize::try_from(id).ok().and_then(|k| lists.get(k).copied())
        };
        let docs = py
            .detach(|| flat_docs(&input, threads))
            .map_err(|e| e.into_pyerr(py))?;
        if ids.as_ref().is_some_and(|ids| ids.len() != docs.len()) {
            return Err(PyValueError::new_err(
                "languages must have one entry per document",
            ));
        }
        let hits: Vec<Option<Hit>> = py
            .detach(|| {
                par::map(docs.len(), threads, |i| {
                    docs[i].map(|d| match pick(i) {
                        None => Hit::Missing,
                        Some(b) if b.found(d, v) => Hit::Yes,
                        Some(_) => Hit::No,
                    })
                })
            })
            .map_err(|e| super::pii::pool_err(py, e))?;
        let mut gen_state = rng.map(|(state, inc)| Pcg64 { state, inc });
        if what == "found" {
            let vals: Vec<Option<bool>> = hits.iter().map(|h| h.map(|h| h == Hit::Yes)).collect();
            return Ok((pack_bools(py, &input, &vals)?, rng));
        }
        // datatrove: `if self.keep_fraction > 0.0 and self.uniform() < self.keep_fraction`.
        let mut keep = Vec::with_capacity(hits.len());
        for h in &hits {
            keep.push(match h {
                None => None,
                Some(Hit::Missing | Hit::No) => Some(true),
                Some(Hit::Yes) => Some(if keep_fraction > 0.0 {
                    let g = gen_state.as_mut().ok_or_else(|| {
                        PyValueError::new_err("keep_fraction > 0 needs a generator state")
                    })?;
                    g.uniform() < keep_fraction
                } else {
                    false
                }),
            });
        }
        let rng = gen_state.map(|g| (g.state, g.inc));
        let reasons = || {
            let vals: Vec<Option<&'static str>> = keep
                .iter()
                .map(|k| (*k == Some(false)).then_some(REASON))
                .collect();
            pack_reasons(py, &input, &vals, threads)
        };
        let out = match what {
            "keep" => pack_bools(py, &input, &keep)?,
            "reason" => reasons()?,
            "all" => PyTuple::new(py, [pack_bools(py, &input, &keep)?, reasons()?])?.into_any(),
            other => {
                return Err(PyValueError::new_err(format!(
                    "what must be 'keep', 'reason', 'all' or 'found', not {other:?}"
                )));
            }
        };
        Ok((out, rng))
    }

    pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<NativeBadWords>()?;
        m.add_function(wrap_pyfunction!(_pcg64_seed, m)?)?;
        m.add_function(wrap_pyfunction!(_badwords_run, m)?)
    }
}
// ---- end C4BadWordsFilter ----

// ---- URLFilter ----
mod url {
    //! The native object behind `pyroid.filters.URLFilter`: datatrove's
    //! `URLFilter` over a batch of URLs, detached, on the shared pool. The
    //! Python side builds the lists and the suffix trie (from the tldextract
    //! instance) and gives a resolver for the lookups the kernel defers.

    use std::borrow::Cow;

    use pyo3::exceptions::{PyAttributeError, PyOverflowError, PyRuntimeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::pybacked::PyBackedStr;
    use pyo3::types::{PyBool, PyList, PyString, PyTuple};
    use pyroid_core::columnar::{BoolBuffers, StringBuffers};
    use pyroid_core::filters::Reason;
    use pyroid_core::filters::url::{
        self as kernel, BuildError, StrSet, SuffixList, UrlError, UrlFilter, UrlLists,
    };
    use pyroid_core::ftfy::UnicodeVersion;
    use pyroid_core::par;

    use crate::arrow::{ArrowStrings, Chunk, Column, ColumnType, StrType, build_error};
    use crate::batch::map_values;
    use crate::errors::with_note;
    use crate::input::{TextInput, Threads, ThreadsArg, texts_and_threads, threads as pool};

    /// `(strings, self_decoding, aliases, nodes, edges, defer_xn)`: see
    /// `SuffixList::new`.
    type TrieArg = (
        Vec<String>,
        Vec<u32>,
        Vec<(String, u32)>,
        Vec<(bool, bool)>,
        Vec<(u32, u32, u32)>,
        bool,
    );
    /// `(entries, file texts)`: parsed entries, and whole files to parse as
    /// `parse_list(f, do_normalize=False)` does.
    type SetArg = (Vec<PyBackedStr>, Vec<PyBackedStr>);

    fn build_set(entries: &[PyBackedStr], texts: &[PyBackedStr]) -> Result<StrSet, BuildError> {
        let mut set = StrSet::new();
        set.reserve(entries.len(), entries.iter().map(|e| e.len()).sum());
        for e in entries {
            set.insert(e)?;
        }
        for t in texts {
            set.extend_stripped_lines(t)?;
        }
        Ok(set)
    }

    fn build_err(e: BuildError) -> PyErr {
        match e {
            BuildError::TooLarge => PyOverflowError::new_err(e.to_string()),
            BuildError::BadTrie(_) => PyValueError::new_err(e.to_string()),
        }
    }

    /// Why one URL could not be filtered.
    #[derive(Clone, Copy)]
    enum Bad {
        Empty,
        Url(UrlError),
    }

    /// (chunk, row) of item `i` of an Arrow input.
    fn locate(a: &ArrowStrings, mut i: usize) -> (usize, usize) {
        for (c, chunk) in a.chunks.iter().enumerate() {
            if i < chunk.len() {
                return (c, i);
            }
            i -= chunk.len();
        }
        (a.chunks.len(), i)
    }

    fn which(input: &TextInput, i: usize) -> String {
        match input {
            TextInput::Str(_) => "the str".to_owned(),
            TextInput::List(_) | TextInput::Tuple(_) => format!("the str at index {i}"),
            TextInput::Arrow(a) => {
                let (c, r) = locate(a, i);
                format!("row {r} of chunk {c} of the Arrow input")
            }
        }
    }

    /// Item `i` (not a null: the kernel has seen it).
    fn url_at(input: &TextInput, i: usize) -> &str {
        match input {
            TextInput::Str(s) => s,
            TextInput::List(d) | TextInput::Tuple(d) => &d[i],
            TextInput::Arrow(a) => {
                let (c, r) = locate(a, i);
                a.chunks[c].get(r).and_then(Result::ok).unwrap_or_default()
            }
        }
    }

    fn str_column<'py, S: AsRef<str> + Sync>(
        py: Python<'py>,
        a: &ArrowStrings,
        vals: &[Option<S>],
        t: par::Threads,
        large: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (mut chunks, mut start) = (Vec::new(), 0);
        for chunk in &a.chunks {
            let part = &vals[start..start + chunk.len()];
            start += chunk.len();
            if large {
                chunks.push(Chunk::LargeUtf8(
                    StringBuffers::build(part, t).map_err(build_error)?,
                ));
            } else {
                let built = StringBuffers::build_split(part, t).map_err(build_error)?;
                chunks.extend(built.into_iter().map(Chunk::Utf8));
            }
        }
        let ty = if large {
            ColumnType::LargeUtf8
        } else {
            ColumnType::Utf8
        };
        Ok(Bound::new(py, Column::new(ty, &a.field.name, chunks))?.into_any())
    }

    fn bool_column<'py>(
        py: Python<'py>,
        a: &ArrowStrings,
        vals: &[Option<bool>],
    ) -> PyResult<Bound<'py, PyAny>> {
        let (mut chunks, mut start) = (Vec::new(), 0);
        for chunk in &a.chunks {
            chunks.push(Chunk::Bool(BoolBuffers::build(
                &vals[start..start + chunk.len()],
            )));
            start += chunk.len();
        }
        Ok(Bound::new(py, Column::new(ColumnType::Bool, &a.field.name, chunks))?.into_any())
    }

    fn seq<'py, T>(
        py: Python<'py>,
        input: &TextInput,
        items: impl ExactSizeIterator<Item = T>,
    ) -> PyResult<Bound<'py, PyAny>>
    where
        T: IntoPyObject<'py>,
    {
        Ok(match input {
            TextInput::Tuple(_) => PyTuple::new(py, items)?.into_any(),
            _ => PyList::new(py, items)?.into_any(),
        })
    }

    /// datatrove 0.10.1's `URLFilter`, built once by `pyroid.filters.URLFilter`.
    #[pyclass(frozen, module = "pyroid.filters", name = "_URLFilter")]
    #[derive(Debug)]
    pub(super) struct NativeUrlFilter {
        inner: UrlFilter,
        /// `url -> (top_domain_under_public_suffix, fqdn)` from tldextract,
        /// for the lookups the kernel defers.
        resolver: Py<PyAny>,
    }

    impl NativeUrlFilter {
        fn error(py: Python<'_>, input: &TextInput, i: usize, e: Bad) -> PyErr {
            let base = match e {
                Bad::Empty => PyValueError::new_err(
                    "empty URL: datatrove's URLFilter asserts that every document has one \
                     (AssertionError: Document does not have url in its metadata)",
                ),
                Bad::Url(e @ UrlError::NoAutomaton) => PyAttributeError::new_err(e.to_string()),
                Bad::Url(e @ UrlError::Deferred) => PyRuntimeError::new_err(e.to_string()),
            };
            with_note(
                py,
                base,
                format!("pyroid.filters.URLFilter: raised by {}", which(input, i)),
            )
        }

        fn resolve(&self, py: Python<'_>, url: &str) -> PyResult<(String, String)> {
            self.resolver.bind(py).call1((url,))?.extract()
        }

        /// datatrove's verdict per URL (`None` for Arrow nulls, `Some(None)` when kept).
        #[allow(clippy::option_option)]
        fn verdicts(
            &self,
            py: Python<'_>,
            input: &TextInput,
            t: Option<Threads>,
            allow_empty: bool,
        ) -> PyResult<Vec<Option<Option<Reason>>>> {
            let f = &self.inner;
            let raw = map_values(py, input, t, |s| {
                if s.is_empty() && !allow_empty {
                    return Err(Bad::Empty);
                }
                f.check(s).map_err(Bad::Url)
            })?;
            let mut out = Vec::with_capacity(raw.len());
            for (i, v) in raw.into_iter().enumerate() {
                out.push(match v {
                    None => None,
                    Some(Ok(r)) => Some(r),
                    Some(Err(Bad::Url(UrlError::Deferred))) => {
                        let url = url_at(input, i);
                        let (rd, fqdn) = self.resolve(py, url)?;
                        let r = f.check_with_parts(url, &rd, &fqdn);
                        Some(r.map_err(|e| Self::error(py, input, i, Bad::Url(e)))?)
                    }
                    Some(Err(e)) => return Err(Self::error(py, input, i, e)),
                });
            }
            Ok(out)
        }
    }

    #[pymethods]
    impl NativeUrlFilter {
        /// Built by `pyroid.filters.URLFilter` (see `_from_parsed` there).
        #[new]
        #[pyo3(signature = (trie, domains, urls, words, unicode_version, resolver))]
        fn new(
            py: Python<'_>,
            trie: TrieArg,
            domains: SetArg,
            urls: SetArg,
            words: (Vec<String>, Vec<String>, Vec<String>, u64),
            unicode_version: &str,
            resolver: Py<PyAny>,
        ) -> PyResult<Self> {
            let version =
                UnicodeVersion::from_unidata_version(unicode_version).ok_or_else(|| {
                    PyValueError::new_err(format!("unicode_version={unicode_version:?}"))
                })?;
            let (strings, self_ids, aliases, nodes, edges, defer_xn) = trie;
            let (banned_words, soft_banned_words, banned_subwords, min_soft_words) = words;
            let inner = py
                .detach(|| -> Result<UrlFilter, BuildError> {
                    let suffixes =
                        SuffixList::new(strings, &self_ids, aliases, &nodes, &edges, defer_xn)?;
                    let lists = UrlLists {
                        domains: build_set(&domains.0, &domains.1)?,
                        urls: build_set(&urls.0, &urls.1)?,
                        banned_words,
                        soft_banned_words,
                        banned_subwords,
                        min_soft_words,
                    };
                    Ok(UrlFilter::new(suffixes, lists, version))
                })
                .map_err(build_err)?;
            Ok(Self { inner, resolver })
        }

        /// datatrove's drop reason per URL (`None` when kept).
        #[pyo3(signature = (urls, /, *, threads = None, allow_empty = false))]
        fn reasons<'py>(
            &self,
            py: Python<'py>,
            urls: &Bound<'py, PyAny>,
            threads: Option<ThreadsArg>,
            allow_empty: bool,
        ) -> PyResult<Bound<'py, PyAny>> {
            let (input, t) = texts_and_threads(urls, threads, None)?;
            let v = self.verdicts(py, &input, t, allow_empty)?;
            let obj = |r: Option<Reason>| -> Bound<'py, PyAny> {
                match r {
                    Some(r) => PyString::intern(py, &r.as_str()).into_any(),
                    None => py.None().into_bound(py),
                }
            };
            match &input {
                TextInput::Str(_) => Ok(obj(v[0].flatten())),
                TextInput::List(_) | TextInput::Tuple(_) => {
                    seq(py, &input, v.iter().map(|r| obj(r.flatten())))
                }
                TextInput::Arrow(a) => {
                    let vals: Vec<Option<Cow<'static, str>>> =
                        v.iter().map(|r| r.flatten().map(|r| r.as_str())).collect();
                    str_column(py, a, &vals, pool(t), false)
                }
            }
        }

        /// True for each URL datatrove keeps.
        #[pyo3(signature = (urls, /, *, threads = None, allow_empty = false))]
        fn keep<'py>(
            &self,
            py: Python<'py>,
            urls: &Bound<'py, PyAny>,
            threads: Option<ThreadsArg>,
            allow_empty: bool,
        ) -> PyResult<Bound<'py, PyAny>> {
            let (input, t) = texts_and_threads(urls, threads, None)?;
            let v = self.verdicts(py, &input, t, allow_empty)?;
            match &input {
                TextInput::Str(_) => Ok(PyBool::new(py, v[0].flatten().is_none())
                    .to_owned()
                    .into_any()),
                TextInput::List(_) | TextInput::Tuple(_) => {
                    seq(py, &input, v.iter().map(|r| r.flatten().is_none()))
                }
                TextInput::Arrow(a) => {
                    let vals: Vec<Option<bool>> =
                        v.iter().map(|r| r.map(|r| r.is_none())).collect();
                    bool_column(py, a, &vals)
                }
            }
        }

        /// tldextract's `(top_domain_under_public_suffix, fqdn)` per URL.
        #[pyo3(signature = (urls, /, *, threads = None))]
        fn domains<'py>(
            &self,
            py: Python<'py>,
            urls: &Bound<'py, PyAny>,
            threads: Option<ThreadsArg>,
        ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
            let (input, t) = texts_and_threads(urls, threads, None)?;
            let f = &self.inner;
            let raw = map_values(py, &input, t, |s| {
                f.domain_parts(s)
                    .ok()
                    .map(|p| (p.registered.into_owned(), p.fqdn.into_owned()))
            })?;
            let mut v = Vec::with_capacity(raw.len());
            for (i, p) in raw.into_iter().enumerate() {
                v.push(match p {
                    None => None,
                    Some(Some(p)) => Some(p),
                    Some(None) => Some(self.resolve(py, url_at(&input, i))?),
                });
            }
            match &input {
                TextInput::Str(_) => {
                    let (rd, fqdn) = v.swap_remove(0).unwrap_or_default();
                    Ok((
                        PyString::new(py, &rd).into_any(),
                        PyString::new(py, &fqdn).into_any(),
                    ))
                }
                TextInput::List(_) | TextInput::Tuple(_) => {
                    let pairs: Vec<(String, String)> = v.into_iter().flatten().collect();
                    Ok((
                        seq(py, &input, pairs.iter().map(|p| PyString::new(py, &p.0)))?,
                        seq(py, &input, pairs.iter().map(|p| PyString::new(py, &p.1)))?,
                    ))
                }
                TextInput::Arrow(a) => {
                    let large = a.field.ty == StrType::LargeUtf8;
                    let rd: Vec<Option<&str>> =
                        v.iter().map(|p| p.as_ref().map(|p| &*p.0)).collect();
                    let fqdn: Vec<Option<&str>> =
                        v.iter().map(|p| p.as_ref().map(|p| &*p.1)).collect();
                    Ok((
                        str_column(py, a, &rd, pool(t), large)?,
                        str_column(py, a, &fqdn, pool(t), large)?,
                    ))
                }
            }
        }

        /// Is `s` a blocked domain?
        fn contains_domain(&self, s: &str) -> bool {
            self.inner.domains().contains(s)
        }

        /// Is `s` a blocked URL?
        fn contains_url(&self, s: &str) -> bool {
            self.inner.urls().contains(s)
        }

        /// Number of blocked domains.
        #[getter]
        fn num_domains(&self) -> usize {
            self.inner.domains().len()
        }

        /// Number of blocked URLs.
        #[getter]
        fn num_urls(&self) -> usize {
            self.inner.urls().len()
        }

        /// Bytes held by the domain and URL sets.
        #[getter]
        fn heap_bytes(&self) -> usize {
            self.inner.heap_bytes()
        }

        /// `(nodes, strings, labels)` of the suffix trie.
        #[getter]
        fn trie_size(&self) -> (usize, usize, usize) {
            let s = self.inner.suffixes();
            (s.node_count(), s.string_count(), s.label_count())
        }

        fn __repr__(&self) -> String {
            format!(
                "<pyroid.filters native URLFilter: {} domains, {} urls>",
                self.inner.domains().len(),
                self.inner.urls().len()
            )
        }
    }

    /// tldextract's `looks_like_ipv6` (for tests).
    #[pyfunction]
    #[pyo3(name = "_looks_like_ipv6")]
    fn looks_like_ipv6(s: &str) -> bool {
        kernel::looks_like_ipv6(s)
    }

    /// tldextract's `lenient_netloc` (for tests).
    #[pyfunction]
    #[pyo3(name = "_lenient_netloc")]
    fn lenient_netloc(s: &str) -> String {
        kernel::lenient_netloc(s).to_owned()
    }

    /// `s.lower()` as the URL filter computes it (for tests).
    #[pyfunction]
    #[pyo3(name = "_py_lower")]
    fn py_lower(s: &str, unicode_version: &str) -> PyResult<String> {
        let v = UnicodeVersion::from_unidata_version(unicode_version)
            .ok_or_else(|| PyValueError::new_err(format!("unicode_version={unicode_version:?}")))?;
        Ok(kernel::py_lower(s, v).into_owned())
    }

    pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<NativeUrlFilter>()?;
        m.add_function(wrap_pyfunction!(looks_like_ipv6, m)?)?;
        m.add_function(wrap_pyfunction!(lenient_netloc, m)?)?;
        m.add_function(wrap_pyfunction!(py_lower, m)?)?;
        Ok(())
    }
}
// ---- end URLFilter ----

// ---- SymbolLinesFormatter ----
mod symbol_lines {
    //! `SymbolLinesFormatter`: a plain transform.

    use pyo3::prelude::*;
    use pyroid_core::filters::symbol_lines::SymbolLines;

    use crate::batch::transform;
    use crate::input::{ThreadsArg, texts_and_threads};

    /// A `SymbolLinesFormatter` configuration, built once by `pyroid.filters`.
    #[pyclass(frozen, module = "pyroid.filters", name = "_SymbolLines")]
    #[derive(Debug)]
    struct NativeSymbolLines(SymbolLines);

    #[pymethods]
    impl NativeSymbolLines {
        /// `symbols`: the one-character entries of `symbols_to_remove`, or
        /// `None` for datatrove's `PUNCTUATION_SET`.
        #[new]
        fn new(symbols: Option<Vec<char>>, replace_char: String) -> Self {
            Self(SymbolLines::new(symbols, replace_char))
        }

        /// `format()` per document, in the caller's kind.
        #[pyo3(signature = (texts, /, *, threads = None))]
        fn run<'py>(
            &self,
            py: Python<'py>,
            texts: &Bound<'py, PyAny>,
            threads: Option<ThreadsArg>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let (input, threads) = texts_and_threads(texts, threads, None)?;
            transform(py, input, threads, |s| self.0.format(s))
        }
    }

    pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<NativeSymbolLines>()
    }
}
// ---- end SymbolLinesFormatter ----
