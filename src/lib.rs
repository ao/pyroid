//! pyroid: High-performance Rust functions for Python
//!
//! This crate provides high-performance Rust implementations of common
//! operations that are typically slow in pure Python.

use pyo3::prelude::*;
use pyo3::exceptions::{PyException, PyValueError, PyRuntimeError, PyTypeError};

// Core functionality
mod core;

// Domain-specific modules
mod math;
mod text;
mod data;
mod io;
mod image;
mod ml;

// Legacy modules (for backward compatibility)
mod string_ops;
mod math_ops;
mod data_ops;
mod utils;

// Disabled modules (dependencies removed)
mod async_ops;
// mod dataframe_ops;
// mod ml_ops;
// mod text_nlp_ops;
// mod io_ops;
// mod image_ops;

/// The pyroid Python module
#[pymodule]
fn pyroid(py: Python, m: &PyModule) -> PyResult<()> {
    // Add core classes directly to the top-level module
    m.add_class::<core::config::PyConfig>()?;
    m.add_class::<core::config::ConfigContext>()?;
    m.add_class::<core::types::PySharedData>()?;
    
    // Add error classes directly to the top-level module
    // Define the base exception class
    let pyroid_error = py.get_type::<PyException>();
    m.add("PyroidError", pyroid_error)?;
    
    // Define specific exception classes using standard Python exceptions
    let input_error = py.get_type::<PyValueError>();
    m.add("InputError", input_error)?;
    
    let computation_error = py.get_type::<PyRuntimeError>();
    m.add("ComputationError", computation_error)?;
    
    let memory_error = py.get_type::<PyTypeError>();
    m.add("MemoryError", memory_error)?;
    
    let conversion_error = py.get_type::<PyTypeError>();
    m.add("ConversionError", conversion_error)?;
    
    let io_error = py.get_type::<PyValueError>();
    m.add("IoError", io_error)?;
    
    // Add async classes directly to the top-level module
    m.add_class::<async_ops::AsyncClient>()?;
    m.add_class::<async_ops::AsyncFileReader>()?;
    
    // Register the core module (for backward compatibility)
    core::register(py, m)?;
    
    // Register domain-specific modules
    math::register(py, m)?;
    text::register(py, m)?;
    data::register(py, m)?;
    io::register(py, m)?;
    image::register(py, m)?;
    ml::register(py, m)?;
    
    // Register legacy modules for backward compatibility
    string_ops::register(py, m)?;
    math_ops::register(py, m)?;
    data_ops::register(py, m)?;
    
    // Disabled module registrations
    async_ops::register(py, m)?;
    // dataframe_ops::register(py, m)?;
    // ml_ops::register(py, m)?;
    // text_nlp_ops::register(py, m)?;
    // io_ops::register(py, m)?;
    // image_ops::register(py, m)?;
    
    // Add module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}