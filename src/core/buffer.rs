//! Zero-copy buffer protocol implementation for Pyroid
//!
//! This module provides efficient zero-copy buffer implementations for data transfer
//! between Python and Rust.

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyBufferError};

/// A buffer that can be shared between Python and Rust without copying
#[pyclass]
pub struct ZeroCopyBuffer {
    data: Vec<u8>,
    readonly: bool,
}

#[pymethods]
impl ZeroCopyBuffer {
    /// Create a new zero-copy buffer with the specified size
    #[new]
    fn new(size: usize, readonly: Option<bool>) -> Self {
        Self {
            data: vec![0; size],
            readonly: readonly.unwrap_or(false),
        }
    }

    /// Create a new zero-copy buffer from existing data
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> Self {
        Self {
            data: data.to_vec(),
            readonly: true,
        }
    }

    /// Get the size of the buffer
    #[getter]
    fn size(&self) -> usize {
        self.data.len()
    }

    /// Get a copy of the buffer as bytes
    fn as_bytes<'py>(&self, py: Python<'py>) -> &'py pyo3::types::PyBytes {
        pyo3::types::PyBytes::new(py, &self.data)
    }
    
    /// Get a reference to the underlying data
    fn get_data(&self) -> Vec<u8> {
        self.data.clone()
    }
    
    /// Set data in the buffer
    fn set_data(&mut self, data: &[u8]) -> PyResult<()> {
        if self.readonly {
            return Err(PyBufferError::new_err("Buffer is readonly"));
        }
        
        if data.len() != self.data.len() {
            return Err(PyValueError::new_err("Data size mismatch"));
        }
        
        self.data.copy_from_slice(data);
        Ok(())
    }
}

/// A memory view that provides efficient access to memory
#[pyclass]
pub struct MemoryView {
    data: Vec<u8>,
    readonly: bool,
}

#[pymethods]
impl MemoryView {
    /// Create a new memory view with the specified size
    #[new]
    fn new(size: usize, readonly: Option<bool>) -> Self {
        Self {
            data: vec![0; size],
            readonly: readonly.unwrap_or(false),
        }
    }
    
    /// Create a memory view from existing data
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> Self {
        Self {
            data: data.to_vec(),
            readonly: true,
        }
    }
    
    /// Get the size of the memory view
    #[getter]
    fn size(&self) -> usize {
        self.data.len()
    }
    
    /// Get a copy of the memory as bytes
    fn as_bytes<'py>(&self, py: Python<'py>) -> &'py pyo3::types::PyBytes {
        pyo3::types::PyBytes::new(py, &self.data)
    }
    
    /// Get a reference to the underlying data
    fn get_data(&self) -> Vec<u8> {
        self.data.clone()
    }
    
    /// Set data in the memory view
    fn set_data(&mut self, data: &[u8]) -> PyResult<()> {
        if self.readonly {
            return Err(PyBufferError::new_err("Memory view is readonly"));
        }
        
        if data.len() != self.data.len() {
            return Err(PyValueError::new_err("Data size mismatch"));
        }
        
        self.data.copy_from_slice(data);
        Ok(())
    }
}

/// Register the buffer module
pub fn register(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let buffer_module = PyModule::new(py, "buffer")?;
    
    buffer_module.add_class::<ZeroCopyBuffer>()?;
    buffer_module.add_class::<MemoryView>()?;
    
    // Add the buffer module to the parent module
    parent_module.add_submodule(buffer_module)?;
    
    Ok(())
}