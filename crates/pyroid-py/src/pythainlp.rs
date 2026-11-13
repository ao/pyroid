//! PyThaiNLP's word list and sentence model behind datatrove's Thai tokenizer.
//!
//! pyroid does not ship them. It reads the files of the PyThaiNLP the user
//! installed (`pip install pythainlp`, which datatrove's `ThaiTokenizer`
//! needs anyway): `corpus/words_th.txt` and `corpus/sentenceseg_crfcut.model`,
//! the files `pythainlp.tokenize` loads, or those of the directory
//! `PYROID_PYTHAINLP_CORPUS` names. The package is located without importing
//! it.

use std::path::PathBuf;

use pyo3::exceptions::{PyOSError, PyRuntimeWarning, PyValueError};
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
    if loaded_here && std::env::var_os("PYROID_PYTHAINLP_CORPUS").is_none() {
        warn_version(py)?;
    }
    Ok(())
}

/// The PyThaiNLP release whose algorithms the port reproduces.
pub(crate) const PYTHAINLP_VERSION: &str = "5.4.0";

/// A `RuntimeWarning` if the installed pythainlp, whose word list and model
/// were just read, is not [`PYTHAINLP_VERSION`]. Called once per process (when
/// the data is loaded); `PYROID_PYTHAINLP_CORPUS` data has no version to check.
fn warn_version(py: Python<'_>) -> PyResult<()> {
    let metadata = py.import("importlib.metadata")?;
    let version: String = match metadata.call_method1("version", ("pythainlp",)) {
        Ok(v) => v.extract()?,
        // Importable but not installed (a checkout): nothing to compare.
        Err(e) if e.is_instance(py, &metadata.getattr("PackageNotFoundError")?) => {
            return Ok(());
        }
        Err(e) => return Err(e),
    };
    if version == PYTHAINLP_VERSION {
        return Ok(());
    }
    let msg = std::ffi::CString::new(format!(
        "pyroid's Thai tokenizer runs PyThaiNLP {PYTHAINLP_VERSION}'s newmm and crfcut on the \
         word list and sentence model of the installed pythainlp {version}: results can differ \
         from PyThaiNLP {version}"
    ))
    .expect("no NUL in the message");
    let category = py.get_type::<PyRuntimeWarning>();
    PyErr::warn(py, &category, &msg, caller_stacklevel(py))
}

/// The `stacklevel` that points a warning at the first frame outside pyroid
/// (as `pyroid.filters._caller_stacklevel` does).
fn caller_stacklevel(py: Python<'_>) -> i32 {
    let Ok(mut frame) = py
        .import("sys")
        .and_then(|sys| sys.call_method1("_getframe", (0,)))
    else {
        return 1;
    };
    let mut level = 1;
    loop {
        let in_pyroid = frame
            .getattr("f_globals")
            .and_then(|g| g.get_item("__name__"))
            .and_then(|n| n.extract::<String>())
            .is_ok_and(|n| n.starts_with("pyroid."));
        let back = match frame.getattr("f_back") {
            Ok(b) if !b.is_none() => b,
            _ => break,
        };
        if !in_pyroid {
            break;
        }
        frame = back;
        level += 1;
    }
    level
}

/// The extra hint for a language pyroid could tokenize with PyThaiNLP's data.
pub(crate) fn hint(language: &str) -> &'static str {
    if is_thai(language) {
        "; install pythainlp (5.4.0) to tokenize it natively"
    } else {
        ""
    }
}
