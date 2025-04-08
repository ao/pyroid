//! I/O operations for Pyroid
//!
//! This module provides high-performance I/O operations.

use pyo3::prelude::*;
use rayon::prelude::*;
use crate::core::error::PyroidError;
use crate::core::conversion::{ToPyObject, FromPyObject};
use crate::core::config::get_config;

mod file;
mod network;
mod async_io;

/// Register the io module
pub fn register(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let io_module = PyModule::new(py, "io")?;
    
    // Register submodules
    file::register(py, io_module)?;
    network::register(py, io_module)?;
    async_io::register(py, io_module)?;
    
    // Add the io module to the parent module
    parent_module.add_submodule(io_module)?;
    
    Ok(())
}