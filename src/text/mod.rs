//! Text processing operations for Pyroid
//!
//! This module provides high-performance text processing operations.

use pyo3::prelude::*;
use rayon::prelude::*;
use crate::core::error::PyroidError;
use crate::core::conversion::{ToPyObject, FromPyObject};
use crate::core::config::get_config;

/// Register the text module
pub fn register(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let text_module = PyModule::new(py, "text")?;
    
    // Add the text module to the parent module
    parent_module.add_submodule(text_module)?;
    
    Ok(())
}