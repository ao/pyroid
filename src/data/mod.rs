//! Data operations for Pyroid
//!
//! This module provides high-performance data operations.

use pyo3::prelude::*;
use rayon::prelude::*;
use crate::core::error::PyroidError;
use crate::core::conversion::{ToPyObject, FromPyObject};
use crate::core::config::get_config;

mod dataframe;
mod collections;

/// Register the data module
pub fn register(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let data_module = PyModule::new(py, "data")?;
    
    // Register submodules
    dataframe::register(py, data_module)?;
    collections::register(py, data_module)?;
    
    // Add the data module to the parent module
    parent_module.add_submodule(data_module)?;
    
    Ok(())
}