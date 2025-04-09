//! Core functionality for Pyroid
//!
//! This module contains core functionality and shared utilities for Pyroid.

pub mod error;
pub mod types;
pub mod conversion;
pub mod config;
pub mod runtime;
pub mod buffer;
pub mod parallel;

use pyo3::prelude::*;

/// Register the core module
pub fn register(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let core_module = PyModule::new(py, "core")?;
    
    // Register submodules
    error::register(py, core_module)?;
    types::register(py, core_module)?;
    conversion::register(py, core_module)?;
    config::register(py, core_module)?;
    runtime::register(py, core_module)?;
    buffer::register(py, core_module)?;
    parallel::register(py, core_module)?;
    
    // Add the core module to the parent module
    parent_module.add_submodule(core_module)?;
    
    Ok(())
}