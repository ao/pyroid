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
mod dataframe_ops;
mod ml_ops;
mod text_nlp_ops;
mod io_ops;
mod image_ops;

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
    
    // Register the dataframe operations
    dataframe_ops::register(py, m)?;
    
    // Register the machine learning operations
    ml_ops::register(py, m)?;
    
    // Register the text and NLP operations
    text_nlp_ops::register(py, m)?;
    
    // Register the file I/O operations
    io_ops::register(py, m)?;
    
    // Register the image processing operations
    image_ops::register(py, m)?;
    
    // Add module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}