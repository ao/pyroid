//! Sets PyO3's `cfg`s (`Py_LIMITED_API`, `Py_GIL_DISABLED`, `Py_3_x`) for this
//! crate, so its code can use the C API calls each wheel's ABI allows
//! (`arrow/fill.rs`: `PyList_SET_ITEM` exists only outside the limited API).

fn main() {
    pyo3_build_config::use_pyo3_cfgs();
}
