//! Error handling for Pyroid
//!
//! This module provides a unified error handling system for Pyroid.

use pyo3::prelude::*;
use pyo3::exceptions::{PyException, PyValueError, PyRuntimeError, PyTypeError};
use std::fmt;

/// Pyroid error types
#[derive(Debug)]
pub enum PyroidError {
    /// Input validation error
    InputError(String),
    /// Computation error
    ComputationError(String),
    /// Memory error
    MemoryError(String),
    /// Type conversion error
    ConversionError(String),
    /// I/O error
    IoError(String),
    /// Other error
    Other(String),
}

impl fmt::Display for PyroidError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PyroidError::InputError(msg) => write!(f, "Input error: {}", msg),
            PyroidError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
            PyroidError::MemoryError(msg) => write!(f, "Memory error: {}", msg),
            PyroidError::ConversionError(msg) => write!(f, "Conversion error: {}", msg),
            PyroidError::IoError(msg) => write!(f, "I/O error: {}", msg),
            PyroidError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for PyroidError {}

impl From<PyroidError> for PyErr {
    fn from(err: PyroidError) -> PyErr {
        match err {
            PyroidError::InputError(msg) => PyValueError::new_err(msg),
            PyroidError::ComputationError(msg) => PyRuntimeError::new_err(msg),
            PyroidError::MemoryError(msg) => PyTypeError::new_err(format!("Memory error: {}", msg)),
            PyroidError::ConversionError(msg) => PyTypeError::new_err(msg),
            PyroidError::IoError(msg) => PyValueError::new_err(format!("I/O error: {}", msg)),
            PyroidError::Other(msg) => PyException::new_err(msg),
        }
    }
}

// Define custom Python exception types
#[pyclass]
struct PyroidErrorType {}

#[pymethods]
impl PyroidErrorType {
    #[new]
    fn new() -> Self {
        Self {}
    }
}

/// Register the error module
pub fn register(py: Python, module: &PyModule) -> PyResult<()> {
    module.add_class::<PyroidErrorType>()?;
    
    // Create the error hierarchy in Python
    let error_module = PyModule::new(py, "error")?;
    
    // Define the base exception class
    error_module.add("PyroidError", py.get_type::<PyException>())?;
    
    // Define specific exception classes using standard Python exceptions
    error_module.add("PyroidError", py.get_type::<PyException>())?;
    error_module.add("InputError", py.get_type::<PyValueError>())?;
    error_module.add("ComputationError", py.get_type::<PyRuntimeError>())?;
    error_module.add("MemoryError", py.get_type::<PyTypeError>())?;
    error_module.add("ConversionError", py.get_type::<PyTypeError>())?;
    error_module.add("IoError", py.get_type::<PyValueError>())?;
    
    // Add the error module to the parent module
    module.add_submodule(error_module)?;
    
    Ok(())
}