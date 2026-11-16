//! `pyroid.difflib`'s native side: the state behind `SequenceMatcher`
//! ([`sm`]), the line-based functions ([`lines`]) and the batch functions.
//! The public API is the Python module `pyroid/difflib.py`, which decides
//! when these apply and otherwise runs CPython's own difflib code.

mod batch;
mod lines;
mod seq;
mod sm;

use pyo3::prelude::*;

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "pyroid.difflib")?;
    m.gil_used(false)?;
    m.add_class::<sm::BSide>()?;
    m.add_class::<sm::ASide>()?;
    m.add_class::<sm::Blocks>()?;
    m.add_function(wrap_pyfunction!(sm::_sm_init, &m)?)?;
    m.add_function(wrap_pyfunction!(sm::_sm_set_seq2, &m)?)?;
    m.add_function(wrap_pyfunction!(sm::_sm_ratio, &m)?)?;
    m.add_function(wrap_pyfunction!(sm::_sm_matching_blocks, &m)?)?;
    m.add_function(wrap_pyfunction!(sm::_sm_cached_blocks, &m)?)?;
    m.add_function(wrap_pyfunction!(sm::_sm_opcodes, &m)?)?;
    m.add_function(wrap_pyfunction!(sm::_sm_quick_ratio, &m)?)?;
    m.add_function(wrap_pyfunction!(sm::_sm_find_longest_match, &m)?)?;
    m.add_function(wrap_pyfunction!(sm::_sm_chain_attr, &m)?)?;
    m.add_function(wrap_pyfunction!(lines::_differ_compare, &m)?)?;
    m.add_function(wrap_pyfunction!(lines::_mdiff, &m)?)?;
    m.add_function(wrap_pyfunction!(lines::_line_diff, &m)?)?;
    m.add_function(wrap_pyfunction!(lines::_get_close_matches, &m)?)?;
    m.add_function(wrap_pyfunction!(batch::ratio_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(batch::_close_matches_batch, &m)?)?;
    parent.add_submodule(&m)
}
