//! Type conversion utilities for Pyroid
//!
//! This module contains functions for converting between Python and Rust types.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use std::collections::HashMap;

/// Convert a Python list to a Rust Vec
pub fn pylist_to_vec<T, F>(py_list: &PyList, converter: F) -> PyResult<Vec<T>>
where
    F: Fn(&PyAny) -> PyResult<T>,
{
    let mut result = Vec::with_capacity(py_list.len());
    for item in py_list.iter() {
        result.push(converter(item)?);
    }
    Ok(result)
}

/// Convert a Rust Vec to a Python list
pub fn vec_to_pylist<T, F>(py: Python, vec: &[T], converter: F) -> PyResult<Py<PyList>>
where
    F: Fn(Python, &T) -> PyResult<PyObject>,
{
    let list = PyList::empty(py);
    for item in vec {
        list.append(converter(py, item)?)?;
    }
    Ok(list.into())
}

/// Convert a Python dict to a Rust HashMap
pub fn pydict_to_hashmap<K, V, FK, FV>(
    py_dict: &PyDict,
    key_converter: FK,
    value_converter: FV,
) -> PyResult<HashMap<K, V>>
where
    K: std::hash::Hash + Eq,
    FK: Fn(&PyAny) -> PyResult<K>,
    FV: Fn(&PyAny) -> PyResult<V>,
{
    let mut result = HashMap::with_capacity(py_dict.len());
    for (key, value) in py_dict.iter() {
        result.insert(key_converter(key)?, value_converter(value)?);
    }
    Ok(result)
}

/// Convert a Rust HashMap to a Python dict
pub fn hashmap_to_pydict<K, V, FK, FV>(
    py: Python,
    hashmap: &HashMap<K, V>,
    key_converter: FK,
    value_converter: FV,
) -> PyResult<Py<PyDict>>
where
    FK: Fn(Python, &K) -> PyResult<PyObject>,
    FV: Fn(Python, &V) -> PyResult<PyObject>,
{
    let dict = PyDict::new(py);
    for (key, value) in hashmap {
        dict.set_item(key_converter(py, key)?, value_converter(py, value)?)?;
    }
    Ok(dict.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pylist_to_vec() {
        Python::with_gil(|py| {
            let py_list = PyList::new(py, &[1, 2, 3, 4, 5]);
            let result: PyResult<Vec<i32>> = pylist_to_vec(py_list, |item| item.extract());
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), vec![1, 2, 3, 4, 5]);
        });
    }
    
    #[test]
    fn test_vec_to_pylist() {
        Python::with_gil(|py| {
            let vec = vec![1, 2, 3, 4, 5];
            let result = vec_to_pylist(py, &vec, |py, &item| Ok(item.to_object(py)));
            assert!(result.is_ok());
            
            let py_list = result.unwrap();
            let extracted: Vec<i32> = py_list.extract(py).unwrap();
            assert_eq!(extracted, vec);
        });
    }
}