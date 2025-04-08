//! Collection operations for Pyroid
//!
//! This module provides high-performance collection operations.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict, PyAny, PyTuple};
use crate::core::error::PyroidError;

/// Filter a list using a predicate function
#[pyfunction]
fn filter(py: Python, values: &PyList, predicate: &PyAny) -> PyResult<Py<PyList>> {
    if !predicate.is_callable() {
        return Err(PyroidError::InputError("Predicate must be callable".to_string()).into());
    }
    
    let filtered_list = PyList::empty(py);
    
    for i in 0..values.len() {
        let item = values.get_item(i)?;
        let args = PyTuple::new(py, &[item]);
        
        match predicate.call1(args) {
            Ok(result) => {
                if let Ok(b) = result.extract::<bool>() {
                    if b {
                        filtered_list.append(item)?;
                    }
                }
            },
            Err(_) => {}
        }
    }
    
    Ok(filtered_list.into())
}

/// Map a function over a list
#[pyfunction]
fn map(py: Python, values: &PyList, func: &PyAny) -> PyResult<Py<PyList>> {
    if !func.is_callable() {
        return Err(PyroidError::InputError("Function must be callable".to_string()).into());
    }
    
    let result_list = PyList::empty(py);
    
    for i in 0..values.len() {
        let item = values.get_item(i)?;
        let args = PyTuple::new(py, &[item]);
        
        match func.call1(args) {
            Ok(result) => {
                result_list.append(result)?;
            },
            Err(_) => {
                result_list.append(py.None())?;
            }
        }
    }
    
    Ok(result_list.into())
}

/// Reduce a list using a binary function
#[pyfunction]
fn reduce(py: Python, values: &PyList, func: &PyAny, initial: Option<&PyAny>) -> PyResult<PyObject> {
    if !func.is_callable() {
        return Err(PyroidError::InputError("Function must be callable".to_string()).into());
    }
    
    if values.is_empty() {
        if let Some(initial) = initial {
            return Ok(initial.to_object(py));
        } else {
            return Err(PyroidError::InputError("Cannot reduce empty sequence with no initial value".to_string()).into());
        }
    }
    
    let mut accumulator = if let Some(initial) = initial {
        initial.to_object(py)
    } else {
        values.get_item(0)?.to_object(py)
    };
    
    let start_idx = if initial.is_some() { 0 } else { 1 };
    
    for i in start_idx..values.len() {
        let item = values.get_item(i)?;
        let args = PyTuple::new(py, &[accumulator.as_ref(py), item]);
        accumulator = func.call1(args)?.to_object(py);
    }
    
    Ok(accumulator)
}

/// Sort a list
#[pyfunction]
fn sort(py: Python, values: &PyList, key: Option<&PyAny>, reverse: Option<bool>) -> PyResult<Py<PyList>> {
    if let Some(key_func) = key {
        if !key_func.is_callable() {
            return Err(PyroidError::InputError("Key function must be callable".to_string()).into());
        }
    }
    
    let reverse = reverse.unwrap_or(false);
    
    // Create a new list with the same items
    let result_list = PyList::empty(py);
    for i in 0..values.len() {
        result_list.append(values.get_item(i)?)?;
    }
    
    // Use Python's built-in sort method which is already optimized
    let kwargs = PyDict::new(py);
    if let Some(key_func) = key {
        kwargs.set_item("key", key_func)?;
    }
    kwargs.set_item("reverse", reverse)?;
    
    // Call the sort method
    result_list.call_method("sort", (), Some(kwargs))?;
    
    Ok(result_list.into())
}

/// Register the collections module
pub fn register(py: Python, module: &PyModule) -> PyResult<()> {
    let collections_module = PyModule::new(py, "collections")?;
    
    collections_module.add_function(wrap_pyfunction!(filter, collections_module)?)?;
    collections_module.add_function(wrap_pyfunction!(map, collections_module)?)?;
    collections_module.add_function(wrap_pyfunction!(reduce, collections_module)?)?;
    collections_module.add_function(wrap_pyfunction!(sort, collections_module)?)?;
    
    // Add the collections module to the parent module
    module.add_submodule(collections_module)?;
    
    Ok(())
}