//! Mathematical operations for Pyroid
//!
//! This module provides high-performance mathematical operations.

use pyo3::prelude::*;
use rayon::prelude::*;
use crate::core::error::PyroidError;
use crate::core::conversion::{ToPyObject, FromPyObject};
use crate::core::config::get_config;

mod vector;
mod matrix;
mod stats;

/// Register the math module
pub fn register(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let math_module = PyModule::new(py, "math")?;
    
    // Register submodules
    vector::register(py, math_module)?;
    matrix::register(py, math_module)?;
    stats::register(py, math_module)?;
    
    // Add the math module to the parent module
    parent_module.add_submodule(math_module)?;
    
    Ok(())
}