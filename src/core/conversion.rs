//! Type conversion utilities for Pyroid
//!
//! This module provides a unified type conversion system for Pyroid.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict, PyTuple, PyString, PyBool, PyFloat, PyInt};
use std::collections::HashMap;
use crate::core::error::PyroidError;

/// Trait for converting from Python objects to Rust types
pub trait FromPyObject<T>: Sized {
    /// Convert from a Python object to a Rust type
    fn from_pyobject(obj: &PyAny) -> PyResult<T>;
}

/// Trait for converting from Rust types to Python objects
pub trait ToPyObject {
    /// Convert from a Rust type to a Python object
    fn to_pyobject(&self, py: Python) -> PyResult<PyObject>;
}

// Implementations for primitive types

impl FromPyObject<i64> for i64 {
    fn from_pyobject(obj: &PyAny) -> PyResult<i64> {
        obj.extract()
    }
}

impl ToPyObject for i64 {
    fn to_pyobject(&self, py: Python) -> PyResult<PyObject> {
        Ok(self.into_py(py))
    }
}

impl FromPyObject<f64> for f64 {
    fn from_pyobject(obj: &PyAny) -> PyResult<f64> {
        obj.extract()
    }
}

impl ToPyObject for f64 {
    fn to_pyobject(&self, py: Python) -> PyResult<PyObject> {
        Ok(self.into_py(py))
    }
}

impl FromPyObject<bool> for bool {
    fn from_pyobject(obj: &PyAny) -> PyResult<bool> {
        obj.extract()
    }
}

impl ToPyObject for bool {
    fn to_pyobject(&self, py: Python) -> PyResult<PyObject> {
        Ok(self.into_py(py))
    }
}

impl FromPyObject<String> for String {
    fn from_pyobject(obj: &PyAny) -> PyResult<String> {
        obj.extract()
    }
}

impl ToPyObject for String {
    fn to_pyobject(&self, py: Python) -> PyResult<PyObject> {
        Ok(self.into_py(py))
    }
}

impl<T> ToPyObject for Option<T>
where
    T: ToPyObject,
{
    fn to_pyobject(&self, py: Python) -> PyResult<PyObject> {
        match self {
            Some(value) => value.to_pyobject(py),
            None => Ok(py.None()),
        }
    }
}

// Implementations for collections

impl<T> FromPyObject<Vec<T>> for Vec<T>
where
    T: for<'a> pyo3::FromPyObject<'a>,
{
    fn from_pyobject(obj: &PyAny) -> PyResult<Vec<T>> {
        let py_list = obj.downcast::<PyList>()?;
        let mut result = Vec::with_capacity(py_list.len());
        
        for item in py_list.iter() {
            result.push(item.extract()?);
        }
        
        Ok(result)
    }
}

impl<T> ToPyObject for Vec<T>
where
    T: ToPyObject,
{
    fn to_pyobject(&self, py: Python) -> PyResult<PyObject> {
        let list = PyList::empty(py);
        
        for item in self {
            list.append(item.to_pyobject(py)?)?;
        }
        
        Ok(list.into())
    }
}

impl<K, V> FromPyObject<HashMap<K, V>> for HashMap<K, V>
where
    K: std::hash::Hash + Eq + for<'a> pyo3::FromPyObject<'a>,
    V: for<'a> pyo3::FromPyObject<'a>,
{
    fn from_pyobject(obj: &PyAny) -> PyResult<HashMap<K, V>> {
        let py_dict = obj.downcast::<PyDict>()?;
        let mut result = HashMap::with_capacity(py_dict.len());
        
        for (key, value) in py_dict.iter() {
            result.insert(key.extract()?, value.extract()?);
        }
        
        Ok(result)
    }
}

impl<K, V> ToPyObject for HashMap<K, V>
where
    K: ToPyObject,
    V: ToPyObject,
{
    fn to_pyobject(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        
        for (key, value) in self {
            dict.set_item(key.to_pyobject(py)?, value.to_pyobject(py)?)?;
        }
        
        Ok(dict.into())
    }
}

// Utility functions

/// Convert a Python list to a Rust Vec
pub fn pylist_to_vec<T>(py_list: &PyList) -> PyResult<Vec<T>>
where
    T: for<'a> pyo3::FromPyObject<'a>,
{
    let mut result = Vec::with_capacity(py_list.len());
    for item in py_list.iter() {
        result.push(item.extract()?);
    }
    Ok(result)
}

/// Convert a Rust Vec to a Python list
pub fn vec_to_pylist<T>(py: Python, vec: &[T]) -> PyResult<Py<PyList>>
where
    T: ToPyObject,
{
    let list = PyList::empty(py);
    for item in vec {
        list.append(item.to_pyobject(py)?)?;
    }
    Ok(list.into())
}

/// Convert a Python dict to a Rust HashMap
pub fn pydict_to_hashmap<K, V>(py_dict: &PyDict) -> PyResult<HashMap<K, V>>
where
    K: std::hash::Hash + Eq + for<'a> pyo3::FromPyObject<'a>,
    V: for<'a> pyo3::FromPyObject<'a>,
{
    let mut result = HashMap::with_capacity(py_dict.len());
    for (key, value) in py_dict.iter() {
        result.insert(key.extract()?, value.extract()?);
    }
    Ok(result)
}

/// Convert a Rust HashMap to a Python dict
pub fn hashmap_to_pydict<K, V>(py: Python, hashmap: &HashMap<K, V>) -> PyResult<Py<PyDict>>
where
    K: ToPyObject,
    V: ToPyObject,
{
    let dict = PyDict::new(py);
    for (key, value) in hashmap {
        dict.set_item(key.to_pyobject(py)?, value.to_pyobject(py)?)?;
    }
    Ok(dict.into())
}

/// Register the conversion module
pub fn register(_py: Python, _module: &PyModule) -> PyResult<()> {
    // No Python-facing functions to register
    Ok(())
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     #[test]
//     fn test_vec_conversion() {
//         Python::with_gil(|py| {
//             // Test Vec<i64>
//             let rust_vec: Vec<i64> = vec![1, 2, 3, 4, 5];
//             let py_list = rust_vec.to_pyobject(py).unwrap();
//             let back_to_rust: Vec<i64> = py_list.as_ref(py).extract().unwrap();
//             assert_eq!(rust_vec, back_to_rust);
//
//             // Test Vec<String>
//             let rust_vec: Vec<String> = vec!["a".to_string(), "b".to_string(), "c".to_string()];
//             let py_list = rust_vec.to_pyobject(py).unwrap();
//             let back_to_rust: Vec<String> = py_list.as_ref(py).extract().unwrap();
//             assert_eq!(rust_vec, back_to_rust);
//         });
//     }
//
//     #[test]
//     fn test_hashmap_conversion() {
//         Python::with_gil(|py| {
//             // Test HashMap<String, i64>
//             let mut rust_map = HashMap::new();
//             rust_map.insert("a".to_string(), 1);
//             rust_map.insert("b".to_string(), 2);
//             rust_map.insert("c".to_string(), 3);
//
//             let py_dict = rust_map.to_pyobject(py).unwrap();
//             let back_to_rust: HashMap<String, i64> = py_dict.as_ref(py).extract().unwrap();
//
//             assert_eq!(rust_map, back_to_rust);
//         });
//     }
// }