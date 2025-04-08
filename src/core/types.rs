//! Core types for Pyroid
//!
//! This module provides core types used throughout Pyroid.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;
use crate::core::error::PyroidError;
use crate::core::conversion::{ToPyObject, FromPyObject};

/// A trait for objects that can be processed in parallel
pub trait Parallelizable {
    /// The type of the result
    type Output;
    
    /// Process the object in parallel
    fn process_parallel(&self) -> Self::Output;
}

/// A trait for objects that can be serialized to JSON
pub trait ToJson {
    /// Convert to a JSON string
    fn to_json(&self) -> Result<String, PyroidError>;
    
    /// Convert to a JSON value
    fn to_json_value(&self) -> Result<serde_json::Value, PyroidError>;
}

/// A trait for objects that can be deserialized from JSON
pub trait FromJson: Sized {
    /// Create from a JSON string
    fn from_json(json: &str) -> Result<Self, PyroidError>;
    
    /// Create from a JSON value
    fn from_json_value(value: &serde_json::Value) -> Result<Self, PyroidError>;
}

/// A trait for objects that can be cloned deeply
pub trait DeepClone {
    /// Create a deep clone of the object
    fn deep_clone(&self) -> Self;
}

/// A wrapper for shared data
#[derive(Clone)]
pub struct SharedData<T>(Arc<T>);

impl<T> SharedData<T> {
    /// Create a new shared data object
    pub fn new(data: T) -> Self {
        Self(Arc::new(data))
    }
    
    /// Get a reference to the data
    pub fn get(&self) -> &T {
        &self.0
    }
    
    /// Convert to Arc
    pub fn into_arc(self) -> Arc<T> {
        self.0
    }
}

impl<T> From<T> for SharedData<T> {
    fn from(data: T) -> Self {
        Self::new(data)
    }
}

impl<T> From<Arc<T>> for SharedData<T> {
    fn from(arc: Arc<T>) -> Self {
        Self(arc)
    }
}

impl<T> std::ops::Deref for SharedData<T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// A Python wrapper for SharedData
#[pyclass]
#[derive(Clone)]
pub struct PySharedData {
    data: PyObject,
}

#[pymethods]
impl PySharedData {
    #[new]
    fn new(data: PyObject) -> Self {
        Self { data }
    }
    
    /// Get the data
    fn get(&self, py: Python) -> PyObject {
        self.data.clone_ref(py)
    }
}

/// Register the types module
pub fn register(py: Python, module: &PyModule) -> PyResult<()> {
    let types_module = PyModule::new(py, "types")?;
    
    types_module.add_class::<PySharedData>()?;
    
    // Add the types module to the parent module
    module.add_submodule(types_module)?;
    
    Ok(())
}