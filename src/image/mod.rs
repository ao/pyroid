//! Image operations for Pyroid
//!
//! This module provides high-performance image operations.

use pyo3::prelude::*;

// Import submodules
pub mod basic;

/// Register the image module
pub fn register(py: Python, m: &PyModule) -> PyResult<()> {
    let image_module = PyModule::new(py, "image")?;
    
    // Register submodules
    basic::register(py, image_module)?;
    
    // Add the image module to the parent module
    m.add_submodule(image_module)?;
    
    Ok(())
}