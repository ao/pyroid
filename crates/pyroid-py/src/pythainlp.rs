//! PyThaiNLP's word list and sentence model behind datatrove's Thai tokenizer.
//!
//! pyroid does not ship them. It reads the files of the PyThaiNLP the user
//! installed (`pip install pythainlp`, which datatrove's `ThaiTokenizer`
//! needs anyway): `corpus/words_th.txt` and `corpus/sentenceseg_crfcut.model`,
//! the files `pythainlp.tokenize` loads, or those of the directory
//! `PYROID_PYTHAINLP_CORPUS` names. The package is located without importing
//! it. A package other than pythainlp 5.4.0 (whose algorithms are ported) gets
//! one `VersionMismatchWarning` when its files are read.

use std::path::PathBuf;

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyroid_core::filters;
use pyroid_core::pythainlp::{self, ThaiError};

/// Whether datatrove tokenizes `language` with `ThaiTokenizer`.
pub(crate) fn is_thai(language: &str) -> bool {
    filters::backend(language).is_some_and(|b| b.class == "ThaiTokenizer")
}

/// `pythainlp/corpus` of the installed package, or `$PYROID_PYTHAINLP_CORPUS`.
fn corpus_dir(py: Python<'_>) -> PyResult<Option<PathBuf>> {
    if let Some(p) = std::env::var_os("PYROID_PYTHAINLP_CORPUS") {
        return Ok(Some(PathBuf::from(p)));
    }
    let spec = py
        .import("importlib.util")?
        .call_method1("find_spec", ("pythainlp",))?;
    if spec.is_none() {
        return Ok(None);
    }
    let origin = spec.getattr("origin")?;
    if origin.is_none() {
        return Ok(None);
    }
    let origin: PathBuf = origin.extract()?;
    Ok(origin.parent().map(|dir| dir.join("corpus")))
}

/// Whether the Thai tokenizer can run natively: the corpus is loaded, or
/// there is one to load.
pub(crate) fn available(py: Python<'_>) -> PyResult<bool> {
    Ok(pythainlp::installed().is_some() || corpus_dir(py)?.is_some())
}

/// Load the corpus files if `language` needs them (once per process). Without
/// PyThaiNLP this does nothing, and `tokenizer_for` gives no tokenizer.
pub(crate) fn prepare(py: Python<'_>, language: &str) -> PyResult<()> {
    if pythainlp::installed().is_some() || !is_thai(language) {
        return Ok(());
    }
    let Some(dir) = corpus_dir(py)? else {
        return Ok(());
    };
    let mut loaded_here = false;
    py.detach(|| pythainlp::install_with(&dir, || loaded_here = true))
        .map_err(|e| match e {
            ThaiError::Io(path, e) => {
                PyOSError::new_err(format!("cannot read PyThaiNLP's {}: {e}", path.display()))
            }
            ThaiError::Format(m) => PyValueError::new_err(format!("{}: {m}", dir.display())),
        })?;
    // The package's own files: check its version (the files of PYROID_PYTHAINLP_CORPUS have none).
    if loaded_here && std::env::var_os("PYROID_PYTHAINLP_CORPUS").is_none() {
        crate::errors::check_version(
            py,
            "pyroid's Thai tokenizer",
            "pythainlp",
            &format!("pythainlp=={PYTHAINLP_VERSION}"),
        )?;
    }
    Ok(())
}

/// The PyThaiNLP release whose algorithms the port reproduces.
pub(crate) const PYTHAINLP_VERSION: &str = "5.4.0";

/// The extra hint for a language pyroid could tokenize with PyThaiNLP's data.
pub(crate) fn hint(language: &str) -> &'static str {
    if is_thai(language) {
        "; install pythainlp (5.4.0) to tokenize it natively"
    } else {
        ""
    }
}
