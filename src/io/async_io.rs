//! Async I/O operations for Pyroid
//!
//! This module provides high-performance async I/O operations.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use pyo3::exceptions::{PyRuntimeError, PyValueError, PyIOError};
use crate::core::error::PyroidError;

/// Async sleep
#[pyfunction]
fn sleep(py: Python, seconds: f64) -> PyResult<()> {
    #[cfg(feature = "io")]
    {
        // Import the async_bridge module
        let async_bridge = PyModule::import(py, "pyroid.async_bridge")?;
        
        // Call the sleep function
        async_bridge.getattr("sleep")?.call1((seconds,))?;
        
        Ok(())
    }
    
    #[cfg(not(feature = "io"))]
    {
        Err(PyroidError::IoError("Async operations are not enabled. Recompile with the 'io' feature.".to_string()).into())
    }
}

/// Async read file
#[pyfunction]
fn read_file_async(py: Python, path: &str) -> PyResult<PyObject> {
    #[cfg(feature = "io")]
    {
        // Import the async_bridge module
        let async_bridge = PyModule::import(py, "pyroid.async_bridge")?;
        
        // Call the read_file function
        let result = async_bridge.getattr("read_file")?.call1((path,))?;
        
        Ok(result.into())
    }
    
    #[cfg(not(feature = "io"))]
    {
        Err(PyroidError::IoError("Async operations are not enabled. Recompile with the 'io' feature.".to_string()).into())
    }
}

/// Async write file
#[pyfunction]
fn write_file_async(py: Python, path: &str, data: &PyBytes) -> PyResult<PyObject> {
    #[cfg(feature = "io")]
    {
        // Import the async_bridge module
        let async_bridge = PyModule::import(py, "pyroid.async_bridge")?;
        
        // Call the write_file function
        let result = async_bridge.getattr("write_file")?.call1((path, data),)?;
        
        Ok(result.into())
    }
    
    #[cfg(not(feature = "io"))]
    {
        Err(PyroidError::IoError("Async operations are not enabled. Recompile with the 'io' feature.".to_string()).into())
    }
}

/// Async HTTP GET request
#[pyfunction]
fn http_get_async(py: Python, url: &str) -> PyResult<PyObject> {
    #[cfg(feature = "io")]
    {
        // Import the async_bridge module
        let async_bridge = PyModule::import(py, "pyroid.async_bridge")?;
        
        // Call the fetch_url function
        let result = async_bridge.getattr("fetch_url")?.call1((url,))?;
        
        Ok(result.into())
    }
    
    #[cfg(not(feature = "io"))]
    {
        Err(PyroidError::IoError("Async operations are not enabled. Recompile with the 'io' feature.".to_string()).into())
    }
}

/// Async HTTP POST request
#[pyfunction]
fn http_post_async(py: Python, url: &str, data: Option<&PyBytes>, json: Option<&PyDict>) -> PyResult<PyObject> {
    #[cfg(feature = "io")]
    {
        // Import the async_bridge module
        let async_bridge = PyModule::import(py, "pyroid.async_bridge")?;
        
        // Convert PyDict to JSON if provided
        let json_dict = if let Some(json_dict) = json {
            Some(json_dict.to_object(py))
        } else {
            None
        };
        
        // Get raw bytes if provided
        let raw_data = data.map(|d| d.as_bytes().to_vec());
        
        // Call the http_post function
        let result = async_bridge.getattr("http_post")?.call1((url, raw_data, json_dict))?;
        
        Ok(result.into())
    }
    
    #[cfg(not(feature = "io"))]
    {
        Err(PyroidError::IoError("Async operations are not enabled. Recompile with the 'io' feature.".to_string()).into())
    }
}

/// Register the async_io module
pub fn register(py: Python, module: &PyModule) -> PyResult<()> {
    let async_io_module = PyModule::new(py, "async_io")?;
    
    async_io_module.add_function(wrap_pyfunction!(sleep, async_io_module)?)?;
    async_io_module.add_function(wrap_pyfunction!(read_file_async, async_io_module)?)?;
    async_io_module.add_function(wrap_pyfunction!(write_file_async, async_io_module)?)?;
    async_io_module.add_function(wrap_pyfunction!(http_get_async, async_io_module)?)?;
    async_io_module.add_function(wrap_pyfunction!(http_post_async, async_io_module)?)?;
    
    // Add the async_io module to the parent module
    module.add_submodule(async_io_module)?;
    
    Ok(())
}