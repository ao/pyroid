// Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
//! `pyroid._pyroid.datasketch`: the native half of `pyroid.datasketch`.
//!
//! The public API (datasketch's classes) is the Python package
//! `python/pyroid/datasketch/`; it keeps datasketch's attributes and error
//! behaviour and calls these functions for the work:
//!
//! - [`minhash`]: permutations, signature updates, `MinHash.bulk` and
//!   `minhash_batch`, LeanMinHash serialization.
//! - [`params`]: the LSH and LSH Ensemble `(b, r)` searches.
//! - [`lsh`]: the `MinHashLSH` and `MinHashLSHForest` indexes.

use pyo3::prelude::*;

mod lsh;
mod minhash;
mod params;

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pyroid.datasketch")?;
    m.gil_used(false)?;
    minhash::add(&m)?;
    params::add(&m)?;
    lsh::add(&m)?;
    parent.add_submodule(&m)
}
