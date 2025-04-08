//! File operations for Pyroid
//!
//! This module provides high-performance file operations.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict, PyBytes};
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::Path;
use crate::core::error::PyroidError;
use crate::core::conversion::{ToPyObject, FromPyObject};
use crate::core::config::get_config;

/// Read a file in parallel
#[pyfunction]
fn read_file(py: Python, path: &str) -> PyResult<PyObject> {
    let path = Path::new(path);
    
    if !path.exists() {
        return Err(PyroidError::IoError(format!("File not found: {}", path.display())).into());
    }
    
    let mut file = File::open(path)
        .map_err(|e| PyroidError::IoError(format!("Failed to open file: {}", e)))?;
    
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .map_err(|e| PyroidError::IoError(format!("Failed to read file: {}", e)))?;
    
    Ok(PyBytes::new(py, &buffer).into())
}

/// Write a file
#[pyfunction]
fn write_file(py: Python, path: &str, data: &PyBytes) -> PyResult<()> {
    let path = Path::new(path);
    
    // Create parent directories if they don't exist
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)
                .map_err(|e| PyroidError::IoError(format!("Failed to create directories: {}", e)))?;
        }
    }
    
    let mut file = File::create(path)
        .map_err(|e| PyroidError::IoError(format!("Failed to create file: {}", e)))?;
    
    let bytes = data.as_bytes();
    file.write_all(bytes)
        .map_err(|e| PyroidError::IoError(format!("Failed to write file: {}", e)))?;
    
    Ok(())
}

/// Read multiple files in parallel
#[pyfunction]
fn read_files(py: Python, paths: Vec<String>) -> PyResult<Py<PyDict>> {
    let config = get_config();
    
    let results: Result<Vec<(String, Vec<u8>)>, PyroidError> = if config.get_bool("parallel").unwrap_or(true) && paths.len() > 10 {
        paths.par_iter()
            .map(|path| {
                let path_obj = Path::new(path);
                
                if !path_obj.exists() {
                    return Err(PyroidError::IoError(format!("File not found: {}", path)));
                }
                
                let mut file = match File::open(path_obj) {
                    Ok(file) => file,
                    Err(e) => return Err(PyroidError::IoError(format!("Failed to open file {}: {}", path, e))),
                };
                
                let mut buffer = Vec::new();
                match file.read_to_end(&mut buffer) {
                    Ok(_) => Ok((path.clone(), buffer)),
                    Err(e) => Err(PyroidError::IoError(format!("Failed to read file {}: {}", path, e))),
                }
            })
            .collect()
    } else {
        let mut results = Vec::with_capacity(paths.len());
        
        for path in &paths {
            let path_obj = Path::new(path);
            
            if !path_obj.exists() {
                return Err(PyroidError::IoError(format!("File not found: {}", path)).into());
            }
            
            let mut file = match File::open(path_obj) {
                Ok(file) => file,
                Err(e) => return Err(PyroidError::IoError(format!("Failed to open file {}: {}", path, e)).into()),
            };
            
            let mut buffer = Vec::new();
            match file.read_to_end(&mut buffer) {
                Ok(_) => results.push((path.clone(), buffer)),
                Err(e) => return Err(PyroidError::IoError(format!("Failed to read file {}: {}", path, e)).into()),
            }
        }
        
        Ok(results)
    };
    
    let dict = PyDict::new(py);
    
    match results {
        Ok(file_contents) => {
            for (path, content) in file_contents {
                dict.set_item(path, PyBytes::new(py, &content))?;
            }
        },
        Err(e) => return Err(e.into()),
    }
    
    Ok(dict.into())
}

/// Register the file module
pub fn register(py: Python, module: &PyModule) -> PyResult<()> {
    let file_module = PyModule::new(py, "file")?;
    
    file_module.add_function(wrap_pyfunction!(read_file, file_module)?)?;
    file_module.add_function(wrap_pyfunction!(write_file, file_module)?)?;
    file_module.add_function(wrap_pyfunction!(read_files, file_module)?)?;
    
    // Add the file module to the parent module
    module.add_submodule(file_module)?;
    
    Ok(())
}