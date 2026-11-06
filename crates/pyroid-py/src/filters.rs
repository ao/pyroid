// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace; modified.
//! `pyroid._pyroid.filters`: native entry points behind `pyroid.filters`.
//!
//! The typed Python module (`python/pyroid/filters.py`) owns datatrove's
//! signatures, defaults and value checks, and builds one [`NativePipeline`]
//! per distinct configuration. This module runs it over a batch: detached,
//! on the shared pool, one pass per document with a shared tokenization.

use std::borrow::Cow;

use pyo3::exceptions::{PyNotImplementedError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyBool, PyDict, PyList, PyNone, PyString, PyTuple};
use pyroid_core::columnar::{BoolBuffers, StringBuffers};
use pyroid_core::filters::{
    self, C4Quality, FilterError, FineWebQuality, GopherQuality, GopherRepetition, Outcome,
    Pipeline, Step, WordTokenizer,
};
use pyroid_core::ftfy::UnicodeVersion;
use pyroid_core::par::{self, PoolError};

use crate::arrow::{Chunk, Column, ColumnType, StrChunk, StrType, build_error};
use crate::batch::{BatchError, DocCost, detach_one};
use crate::errors::with_note;
use crate::input::{TextInput, ThreadsArg, texts_and_threads};

/// A validated filter chain, built once per distinct config by the shim.
#[pyclass(frozen, module = "pyroid.filters", name = "_Pipeline")]
#[derive(Debug)]
pub struct NativePipeline(Pipeline);

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

#[pymethods]
impl NativePipeline {
    /// `steps`: `(kind, language, params)` per filter, already validated by
    /// `pyroid.filters`.
    #[new]
    #[pyo3(signature = (steps, unicode_version))]
    fn new(
        steps: Vec<(String, String, Bound<'_, PyDict>)>,
        unicode_version: &str,
    ) -> PyResult<Self> {
        let version = UnicodeVersion::from_unidata_version(unicode_version)
            .ok_or_else(|| PyValueError::new_err(format!("unicode_version={unicode_version:?}")))?;
        let mut out = Vec::with_capacity(steps.len());
        for (kind, language, params) in &steps {
            let tok: &'static dyn WordTokenizer = filters::tokenizer_for(language, version)
                .ok_or_else(|| {
                    PyNotImplementedError::new_err(format!(
                        "pyroid.filters has no word tokenizer for language {language:?} yet; \
                         supported: the codes datatrove tokenizes with SpaCyTokenizer(\"en\"): {}",
                        filters::english_languages().join(", ")
                    ))
                })?;
            out.push((step_from(kind, params)?, tok));
        }
        Ok(Self(Pipeline { steps: out }))
    }

    fn __repr__(&self) -> String {
        format!(
            "<pyroid.filters native pipeline of {} steps>",
            self.0.steps.len()
        )
    }
}

/// The datatrove language codes the English tokenizer serves.
#[pyfunction]
fn english_languages() -> Vec<&'static str> {
    filters::english_languages().to_vec()
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
    let p = &pipeline.0;
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

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pyroid.filters")?;
    m.gil_used(false)?;
    m.add_class::<NativePipeline>()?;
    m.add_function(wrap_pyfunction!(run, &m)?)?;
    m.add_function(wrap_pyfunction!(english_languages, &m)?)?;
    parent.add_submodule(&m)
}
