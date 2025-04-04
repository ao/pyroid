//! pyroid: High-performance Rust functions for Python
//!
//! This crate provides high-performance Rust implementations of common
//! operations that are typically slow in pure Python.

use pyo3::prelude::*;

mod string_ops;
mod math_ops;
mod data_ops;
mod async_ops;
mod utils;

/// The pyroid Python module
#[pymodule]
fn pyroid(py: Python, m: &PyModule) -> PyResult<()> {
    // Register the string operations
    string_ops::register(py, m)?;
    
    // Register the math operations
    math_ops::register(py, m)?;
    
    // Register the data operations
    data_ops::register(py, m)?;
    
    // Register the async operations
    async_ops::register(py, m)?;
    
    // Add module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}