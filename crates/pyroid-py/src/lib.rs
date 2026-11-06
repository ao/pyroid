//! `PyO3` bindings: the `pyroid._pyroid` extension module.
//!
//! Keep this crate thin. Kernels live in `pyroid-core`; this crate converts
//! arguments ([`input`]), runs kernels detached on the shared pool
//! ([`batch`]), moves Arrow data across the C Data Interface ([`arrow`], the
//! only module with `unsafe`), and maps errors to the hierarchy ([`errors`]).
//!
//! Product tracks add `src/<mod>.rs` with a `register` function and one line
//! below; see `docs/dev/MODULES.md`.

mod arrow;
mod batch;
mod dedup;
mod errors;
mod fasttext;
mod filters;
mod ftfy;
mod input;
mod json_repair;
mod metrics;
mod testing;
mod text;
mod translit;

use pyo3::prelude::*;

#[pymodule(gil_used = false)]
fn _pyroid(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // The wheel's PEP 440 version (1.0.0-dev is 1.0.0.dev0), as importlib.metadata reports it.
    let version = env!("CARGO_PKG_VERSION");
    m.add(
        "__version__",
        pyroid_core::version::pep440(version).unwrap_or_else(|| version.to_owned()),
    )?;
    errors::register(m)?;
    m.add_class::<arrow::Column>()?;
    testing::register(m)?;
    // Product modules: one registration line each.
    text::register(m)?;
    ftfy::register(m)?;
    fasttext::register(m)?;
    filters::register(m)?;
    dedup::register(m)?;
    translit::register(m)?;
    metrics::register(m)?;
    json_repair::register(m)?;
    Ok(())
}
