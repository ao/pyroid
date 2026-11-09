//! Word-tokenizer data read at run time: SudachiDict for Japanese
//! ([`crate::sudachi`]) and PyThaiNLP's corpus for Thai
//! ([`crate::pythainlp`]).

use pyo3::prelude::*;
use pyroid_core::filters;
use pyroid_core::ftfy::UnicodeVersion;

/// Load what `language`'s tokenizer needs, if it needs anything (once per
/// process).
pub(crate) fn prepare(py: Python<'_>, language: &str) -> PyResult<()> {
    crate::sudachi::prepare(py, language)?;
    crate::pythainlp::prepare(py, language)
}

/// The extra hint for a language pyroid could tokenize with installed data.
pub(crate) fn hint(language: &str) -> &'static str {
    if crate::sudachi::is_japanese(language) {
        crate::sudachi::hint(language)
    } else {
        crate::pythainlp::hint(language)
    }
}

/// Whether pyroid tokenizes `language` natively, with the run-time data
/// availability looked up at most once per call site (`ja`, `th`).
pub(crate) struct Native {
    ja: Option<bool>,
    th: Option<bool>,
}

impl Native {
    pub(crate) fn new() -> Self {
        Self { ja: None, th: None }
    }

    pub(crate) fn get(&mut self, py: Python<'_>, code: &str, v: UnicodeVersion) -> PyResult<bool> {
        if crate::sudachi::is_japanese(code) {
            if self.ja.is_none() {
                self.ja = Some(crate::sudachi::available(py)?);
            }
            return Ok(self.ja == Some(true));
        }
        if crate::pythainlp::is_thai(code) {
            if self.th.is_none() {
                self.th = Some(crate::pythainlp::available(py)?);
            }
            return Ok(self.th == Some(true));
        }
        Ok(filters::tokenizer_for(code, v).is_some())
    }
}
