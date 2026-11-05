// Derived from ftfy 6.3.1 (Apache-2.0), © Robyn Speer; modified.
//! `pyroid._pyroid.ftfy`: native entry points behind `pyroid.ftfy`.
//!
//! The typed Python shim (`python/pyroid/ftfy.py`) owns ftfy's signatures,
//! config handling and delegation; this module only runs the Rust kernel.
//! Strings with lone surrogates (which `&str` cannot hold) go through the
//! kernel's `*_wtf8` functions via Python's `"surrogatepass"` UTF-8.

use std::borrow::Cow;
use std::num::NonZeroUsize;

use pyo3::exceptions::{PyUnicodeEncodeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyString, PyTuple};
use pyroid_core::ftfy::{
    self as kernel, ExplanationStep, Normalization, TextFixerConfig, UnescapeHtml, UnicodeVersion,
};

/// Documents at least this long are fixed with the interpreter detached.
const DETACH_BYTES: usize = 4096;

/// A validated `TextFixerConfig`, built once per distinct config by the shim.
#[pyclass(frozen, module = "pyroid.ftfy", name = "_NativeConfig")]
pub struct NativeConfig(TextFixerConfig);

#[pymethods]
impl NativeConfig {
    #[new]
    #[pyo3(signature = (
        *, unescape_html, remove_terminal_escapes, fix_encoding, restore_byte_a0,
        replace_lossy_sequences, decode_inconsistent_utf8, fix_c1_controls, fix_latin_ligatures,
        fix_character_width, uncurl_quotes, fix_line_breaks, fix_surrogates, remove_control_chars,
        normalization, max_decode_length, explain, unicode_version
    ))]
    #[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
    fn new(
        unescape_html: &str,
        remove_terminal_escapes: bool,
        fix_encoding: bool,
        restore_byte_a0: bool,
        replace_lossy_sequences: bool,
        decode_inconsistent_utf8: bool,
        fix_c1_controls: bool,
        fix_latin_ligatures: bool,
        fix_character_width: bool,
        uncurl_quotes: bool,
        fix_line_breaks: bool,
        fix_surrogates: bool,
        remove_control_chars: bool,
        normalization: Option<&str>,
        max_decode_length: usize,
        explain: bool,
        unicode_version: &str,
    ) -> PyResult<Self> {
        let unescape_html = match unescape_html {
            "auto" => UnescapeHtml::Auto,
            "always" => UnescapeHtml::Always,
            "never" => UnescapeHtml::Never,
            other => return Err(PyValueError::new_err(format!("unescape_html={other:?}"))),
        };
        let normalization = match normalization {
            None => None,
            Some(n) => Some(
                Normalization::from_name(n)
                    .ok_or_else(|| PyValueError::new_err(format!("normalization={n:?}")))?,
            ),
        };
        let max_decode_length = NonZeroUsize::new(max_decode_length)
            .ok_or_else(|| PyValueError::new_err("max_decode_length must be at least 1"))?;
        let unicode_version = UnicodeVersion::from_unidata_version(unicode_version)
            .ok_or_else(|| PyValueError::new_err(format!("unicode_version={unicode_version:?}")))?;
        Ok(Self(TextFixerConfig {
            unescape_html,
            remove_terminal_escapes,
            fix_encoding,
            restore_byte_a0,
            replace_lossy_sequences,
            decode_inconsistent_utf8,
            fix_c1_controls,
            fix_latin_ligatures,
            fix_character_width,
            uncurl_quotes,
            fix_line_breaks,
            fix_surrogates,
            remove_control_chars,
            normalization,
            max_decode_length,
            explain,
            unicode_version,
        }))
    }

    fn __repr__(&self) -> String {
        format!("<pyroid.ftfy native config {:?}>", self.0)
    }
}

/// The Unicode versions the kernel has tables for (`unicodedata.unidata_version` spelling).
#[pyfunction]
fn supported_unicode_versions() -> Vec<&'static str> {
    [
        UnicodeVersion::V14_0,
        UnicodeVersion::V15_0,
        UnicodeVersion::V15_1,
        UnicodeVersion::V16_0,
        UnicodeVersion::V17_0,
    ]
    .iter()
    .map(|v| v.as_str())
    .collect()
}

/// Which kernel function to run.
#[derive(Clone, Copy)]
enum Op {
    Text,
    Segment,
    Encoding,
}

fn run_str<'a>(op: Op, s: &'a str, cfg: &TextFixerConfig) -> Cow<'a, str> {
    match op {
        Op::Text => kernel::fix_text(s, cfg),
        Op::Segment => kernel::fix_text_segment(s, cfg),
        Op::Encoding => kernel::fix_encoding(s, cfg),
    }
}

fn run_wtf8<'a>(op: Op, b: &'a [u8], cfg: &TextFixerConfig) -> Cow<'a, [u8]> {
    let r = match op {
        Op::Text => kernel::fix_text_wtf8(b, cfg),
        Op::Segment => kernel::fix_text_segment_wtf8(b, cfg),
        Op::Encoding => kernel::fix_encoding_wtf8(b, cfg),
    };
    r.expect("surrogatepass output is valid")
}

/// `text` as UTF-8 extended with surrogates. Calls `str.encode` unbound, so a
/// subclass overriding `encode` does not interfere.
fn surrogatepass_bytes<'py>(text: &Bound<'py, PyString>) -> PyResult<Bound<'py, PyBytes>> {
    let py = text.py();
    Ok(py
        .get_type::<PyString>()
        .call_method1("encode", (text, "utf-8", "surrogatepass"))?
        .cast_into::<PyBytes>()?)
}

fn str_from_wtf8<'py>(py: Python<'py>, b: &[u8]) -> PyResult<Bound<'py, PyString>> {
    match std::str::from_utf8(b) {
        Ok(s) => Ok(PyString::new(py, s)),
        Err(_) => PyString::from_encoded_object(
            PyBytes::new(py, b).as_any(),
            Some(c"utf-8"),
            Some(c"surrogatepass"),
        ),
    }
}

/// The input object itself when it is an exact `str` (ftfy returns an exact
/// `str` even for subclasses).
fn unchanged<'py>(text: &Bound<'py, PyString>) -> PyResult<Bound<'py, PyString>> {
    if text.is_exact_instance_of::<PyString>() {
        return Ok(text.clone());
    }
    match text.to_str() {
        Ok(s) => Ok(PyString::new(text.py(), s)),
        Err(_) => str_from_wtf8(text.py(), surrogatepass_bytes(text)?.as_bytes()),
    }
}

fn run<'py>(
    op: Op,
    text: &Bound<'py, PyString>,
    cfg: &NativeConfig,
) -> PyResult<Bound<'py, PyString>> {
    let py = text.py();
    let cfg = &cfg.0;
    match text.to_str() {
        Ok(s) => {
            let owned = |c: Cow<'_, str>| match c {
                Cow::Borrowed(_) => None,
                Cow::Owned(o) => Some(o),
            };
            let fixed = if s.len() >= DETACH_BYTES {
                py.detach(|| owned(run_str(op, s, cfg)))
            } else {
                owned(run_str(op, s, cfg))
            };
            match fixed {
                Some(f) if f != s => Ok(PyString::new(py, &f)),
                _ => unchanged(text),
            }
        }
        Err(e) if e.is_instance_of::<PyUnicodeEncodeError>(py) => {
            let bytes = surrogatepass_bytes(text)?;
            let b = bytes.as_bytes();
            match run_wtf8(op, b, cfg) {
                Cow::Borrowed(_) => unchanged(text),
                Cow::Owned(v) => str_from_wtf8(py, &v),
            }
        }
        Err(e) => Err(e),
    }
}

fn steps_to_py<'py>(
    py: Python<'py>,
    steps: Option<&Vec<ExplanationStep>>,
) -> PyResult<Bound<'py, PyAny>> {
    match steps {
        None => Ok(py.None().into_bound(py)),
        Some(steps) => {
            let items: Vec<Bound<'py, PyTuple>> = steps
                .iter()
                .map(|s| PyTuple::new(py, [s.action.as_str(), s.parameter]))
                .collect::<PyResult<_>>()?;
            Ok(PyList::new(py, items)?.into_any())
        }
    }
}

fn run_explained<'py>(
    text: &Bound<'py, PyString>,
    cfg: &NativeConfig,
    encoding_only: bool,
) -> PyResult<(Bound<'py, PyString>, Bound<'py, PyAny>)> {
    let py = text.py();
    let cfg = &cfg.0;
    match text.to_str() {
        Ok(s) => {
            let r = if encoding_only {
                kernel::fix_encoding_and_explain(s, cfg)
            } else {
                kernel::fix_and_explain(s, cfg)
            };
            let out = if r.text == s {
                unchanged(text)?
            } else {
                PyString::new(py, &r.text)
            };
            Ok((out, steps_to_py(py, r.explanation.as_ref())?))
        }
        Err(e) if e.is_instance_of::<PyUnicodeEncodeError>(py) => {
            let bytes = surrogatepass_bytes(text)?;
            let b = bytes.as_bytes();
            let r = if encoding_only {
                kernel::fix_encoding_and_explain_wtf8(b, cfg)
            } else {
                kernel::fix_and_explain_wtf8(b, cfg)
            }
            .expect("surrogatepass output is valid");
            let out = match r.text {
                Cow::Borrowed(_) => unchanged(text)?,
                Cow::Owned(v) => str_from_wtf8(py, &v)?,
            };
            Ok((out, steps_to_py(py, r.explanation.as_ref())?))
        }
        Err(e) => Err(e),
    }
}

/// `ftfy.fix_text(text, config)` with a prepared native config.
#[pyfunction]
fn fix_text<'py>(
    text: &Bound<'py, PyString>,
    config: &NativeConfig,
) -> PyResult<Bound<'py, PyString>> {
    run(Op::Text, text, config)
}

/// `ftfy.fix_text_segment(text, config)`.
#[pyfunction]
fn fix_text_segment<'py>(
    text: &Bound<'py, PyString>,
    config: &NativeConfig,
) -> PyResult<Bound<'py, PyString>> {
    run(Op::Segment, text, config)
}

/// `ftfy.fix_encoding(text, config)`.
#[pyfunction]
fn fix_encoding<'py>(
    text: &Bound<'py, PyString>,
    config: &NativeConfig,
) -> PyResult<Bound<'py, PyString>> {
    run(Op::Encoding, text, config)
}

/// `ftfy.fix_and_explain(text, config)` as `(text, steps | None)`.
#[pyfunction]
fn fix_and_explain<'py>(
    text: &Bound<'py, PyString>,
    config: &NativeConfig,
) -> PyResult<(Bound<'py, PyString>, Bound<'py, PyAny>)> {
    run_explained(text, config, false)
}

/// `ftfy.fix_encoding_and_explain(text, config)` as `(text, steps)`.
#[pyfunction]
fn fix_encoding_and_explain<'py>(
    text: &Bound<'py, PyString>,
    config: &NativeConfig,
) -> PyResult<(Bound<'py, PyString>, Bound<'py, PyAny>)> {
    run_explained(text, config, true)
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pyroid.ftfy")?;
    m.gil_used(false)?;
    m.add_class::<NativeConfig>()?;
    m.add_function(wrap_pyfunction!(supported_unicode_versions, &m)?)?;
    m.add_function(wrap_pyfunction!(fix_text, &m)?)?;
    m.add_function(wrap_pyfunction!(fix_text_segment, &m)?)?;
    m.add_function(wrap_pyfunction!(fix_encoding, &m)?)?;
    m.add_function(wrap_pyfunction!(fix_and_explain, &m)?)?;
    m.add_function(wrap_pyfunction!(fix_encoding_and_explain, &m)?)?;
    parent.add_submodule(&m)
}
