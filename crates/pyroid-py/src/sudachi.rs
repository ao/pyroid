//! The SudachiDict dictionary behind datatrove's Japanese tokenizer.
//!
//! pyroid does not ship the dictionary (about 200 MB). It reads the one the
//! user installed with `pip install sudachidict-core`, the package SudachiPy
//! (and so datatrove) loads by default, or the file `PYROID_SUDACHI_DICT`
//! names. The package is located without importing it or SudachiPy.

use std::path::PathBuf;

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyroid_core::filters;
use pyroid_core::sudachi::{self, DicError};

/// Whether datatrove tokenizes `language` with `SpaCyTokenizer("ja")`.
pub(crate) fn is_japanese(language: &str) -> bool {
    filters::backend(language).is_some_and(|b| b.class == "SpaCyTokenizer" && b.arg == "ja")
}

/// `sudachidict_core/resources/system.dic` (as SudachiPy's
/// `_get_absolute_dict_path("core")`), or `$PYROID_SUDACHI_DICT`.
fn dictionary_path(py: Python<'_>) -> PyResult<Option<PathBuf>> {
    if let Some(p) = std::env::var_os("PYROID_SUDACHI_DICT") {
        return Ok(Some(PathBuf::from(p)));
    }
    let spec = py
        .import("importlib.util")?
        .call_method1("find_spec", ("sudachidict_core",))?;
    if spec.is_none() {
        return Ok(None);
    }
    let origin = spec.getattr("origin")?;
    if origin.is_none() {
        return Ok(None);
    }
    let origin: PathBuf = origin.extract()?;
    Ok(origin
        .parent()
        .map(|dir| dir.join("resources").join("system.dic")))
}

/// Whether the Japanese tokenizer can run natively: a dictionary is loaded,
/// or there is one to load.
pub(crate) fn available(py: Python<'_>) -> PyResult<bool> {
    Ok(sudachi::ja::installed().is_some() || dictionary_path(py)?.is_some())
}

/// Load the dictionary if `language` needs it (once per process). Without a
/// dictionary this does nothing, and `tokenizer_for` gives no tokenizer.
pub(crate) fn prepare(py: Python<'_>, language: &str) -> PyResult<()> {
    if sudachi::ja::installed().is_some() || !is_japanese(language) {
        return Ok(());
    }
    let Some(path) = dictionary_path(py)? else {
        return Ok(());
    };
    let shown = path.display().to_string();
    py.detach(|| sudachi::ja::install(&path))
        .map(|_| ())
        .map_err(|e| match e {
            DicError::Io(e) => {
                PyOSError::new_err(format!("cannot read the Sudachi dictionary {shown}: {e}"))
            }
            DicError::Format(m) => PyValueError::new_err(format!("{shown}: {m}")),
        })
}

/// The extra hint for a language pyroid could tokenize with a dictionary.
pub(crate) fn hint(language: &str) -> &'static str {
    if is_japanese(language) {
        "; install sudachidict-core (20261015 or newer) to tokenize it natively"
    } else {
        ""
    }
}
