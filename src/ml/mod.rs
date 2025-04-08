//! Machine learning operations for Pyroid
//!
//! This module provides high-performance machine learning operations.

use pyo3::prelude::*;

// Import submodules
pub mod basic;

/// Register the ml module
pub fn register(py: Python, m: &PyModule) -> PyResult<()> {
    let ml_module = PyModule::new(py, "ml")?;
    
    // Register submodules
    basic::register(py, ml_module)?;
    
    // Add the ml module to the parent module
    m.add_submodule(ml_module)?;
    
    Ok(())
}