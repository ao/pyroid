//! Data processing operations
//!
//! This module provides high-performance data processing operations.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyList, PyDict};
use rayon::prelude::*;
use crate::utils::split_into_chunks;

/// Filter a list in parallel using a predicate function
///
/// Args:
///     items: A list of items to filter
///     predicate: A function that returns true for items to keep
///
/// Returns:
///     A new list containing only the items for which the predicate returns true
#[pyfunction]
fn parallel_filter(_py: Python, items: &PyList, predicate: PyObject) -> PyResult<Py<PyList>> {
    Python::with_gil(|py| {
        // Convert Python list to Rust Vec
        let mut rust_items: Vec<PyObject> = Vec::with_capacity(items.len());
        for item in items.iter() {
            rust_items.push(item.to_object(py));
        }
        
        // Split into chunks for parallel processing
        let chunks = split_into_chunks(&rust_items, 1000);
        
        // Process each chunk in parallel
        let results: Vec<Vec<PyObject>> = chunks
            .par_iter()
            .map(|chunk| {
                Python::with_gil(|py| {
                    let mut result = Vec::new();
                    for item in chunk.iter() {
                        let item_ref = item.clone_ref(py);
                        match predicate.call1(py, (item_ref,)) {
                            Ok(value) => {
                                if value.extract::<bool>(py).unwrap_or(false) {
                                    result.push(item.clone_ref(py));
                                }
                            },
                            Err(e) => {
                                // If the predicate fails, we propagate the error
                                return Err(e);
                            }
                        }
                    }
                    Ok(result)
                })
            })
            .collect::<Result<Vec<Vec<PyObject>>, PyErr>>()?;
        
        // Flatten the results
        let mut filtered_items = Vec::new();
        for chunk_result in results {
            filtered_items.extend(chunk_result);
        }
        
        // Convert back to Python list
        let py_list = PyList::empty(py);
        for item in filtered_items {
            py_list.append(item)?;
        }
        
        Ok(py_list.into())
    })
}

/// Map a function over a list in parallel
///
/// Args:
///     items: A list of items to map
///     func: A function to apply to each item
///
/// Returns:
///     A new list containing the results of applying the function to each item
#[pyfunction]
fn parallel_map(_py: Python, items: &PyList, func: PyObject) -> PyResult<Py<PyList>> {
    Python::with_gil(|py| {
        // Convert Python list to Rust Vec
        let mut rust_items: Vec<PyObject> = Vec::with_capacity(items.len());
        for item in items.iter() {
            rust_items.push(item.to_object(py));
        }
        
        // Split into chunks for parallel processing
        let chunks = split_into_chunks(&rust_items, 1000);
        
        // Process each chunk in parallel
        let results: Vec<Vec<PyObject>> = chunks
            .par_iter()
            .map(|chunk| {
                Python::with_gil(|py| {
                    let mut result = Vec::new();
                    for item in chunk.iter() {
                        let item_ref = item.clone_ref(py);
                        match func.call1(py, (item_ref,)) {
                            Ok(value) => {
                                result.push(value);
                            },
                            Err(e) => {
                                // If the function fails, we propagate the error
                                return Err(e);
                            }
                        }
                    }
                    Ok(result)
                })
            })
            .collect::<Result<Vec<Vec<PyObject>>, PyErr>>()?;
        
        // Flatten the results
        let mut mapped_items = Vec::new();
        for chunk_result in results {
            mapped_items.extend(chunk_result);
        }
        
        // Convert back to Python list
        let py_list = PyList::empty(py);
        for item in mapped_items {
            py_list.append(item)?;
        }
        
        Ok(py_list.into())
    })
}

/// Reduce a list to a single value in parallel
///
/// Args:
///     items: A list of items to reduce
///     func: A function that takes two arguments and returns a single value
///     initial: An optional initial value
///
/// Returns:
///     The result of reducing the list
#[pyfunction]
fn parallel_reduce(_py: Python, items: &PyList, func: PyObject, initial: Option<PyObject>) -> PyResult<PyObject> {
    if items.is_empty() {
        return match initial {
            Some(value) => Ok(value),
            None => Err(PyValueError::new_err("Cannot reduce empty sequence with no initial value")),
        };
    }
    
    Python::with_gil(|py| {
        // Convert Python list to Rust Vec
        let mut rust_items: Vec<PyObject> = Vec::with_capacity(items.len());
        for item in items.iter() {
            rust_items.push(item.to_object(py));
        }
        
        // If we have an initial value, add it to the front of the list
        if let Some(init) = initial {
            rust_items.insert(0, init);
        }
        
        // Split into chunks for parallel processing
        let chunk_size = (rust_items.len() / rayon::current_num_threads()).max(1);
        let chunks = split_into_chunks(&rust_items, chunk_size);
        
        // First reduce each chunk
        let chunk_results: Vec<PyObject> = chunks
            .par_iter()
            .map(|chunk| {
                Python::with_gil(|py| {
                    if chunk.is_empty() {
                        return Err(PyValueError::new_err("Empty chunk encountered"));
                    }
                    
                    let mut acc = chunk[0].clone_ref(py);
                    for i in 1..chunk.len() {
                        let next_item = chunk[i].clone_ref(py);
                        match func.call1(py, (acc, next_item)) {
                            Ok(value) => {
                                acc = value;
                            },
                            Err(e) => {
                                return Err(e);
                            }
                        }
                    }
                    Ok(acc)
                })
            })
            .collect::<Result<Vec<PyObject>, PyErr>>()?;
        
        // Then reduce the chunk results
        if chunk_results.is_empty() {
            return Err(PyValueError::new_err("No chunks to reduce"));
        }
        
        let mut final_result = chunk_results[0].clone_ref(py);
        for i in 1..chunk_results.len() {
            let next_result = chunk_results[i].clone_ref(py);
            match func.call1(py, (final_result, next_result)) {
                Ok(value) => {
                    final_result = value;
                },
                Err(e) => {
                    return Err(e);
                }
            }
        }
        
        Ok(final_result)
    })
}

/// Sort a list in parallel
///
/// Args:
///     items: A list of items to sort
///     key: An optional function to extract a comparison key from each item
///     reverse: Whether to sort in descending order
///
/// Returns:
///     A new sorted list
#[pyfunction]
fn parallel_sort(py: Python, items: &PyList, key: Option<PyObject>, reverse: Option<bool>) -> PyResult<Py<PyList>> {
    let reverse = reverse.unwrap_or(false);
    
    // Convert Python list to Rust Vec
    let mut rust_items: Vec<PyObject> = Vec::with_capacity(items.len());
    for item in items.iter() {
        rust_items.push(item.to_object(py));
    }
    
    // If we have a key function, extract keys for all items
    let mut keys_and_items: Vec<(PyObject, PyObject)> = if let Some(key_func) = key {
        let mut result = Vec::with_capacity(rust_items.len());
        for item in rust_items.iter() {
            let item_ref = item.clone_ref(py);
            match key_func.call1(py, (item_ref,)) {
                Ok(key_value) => {
                    result.push((key_value, item.clone_ref(py)));
                },
                Err(e) => {
                    return Err(e);
                }
            }
        }
        result
    } else {
        // No key function, use the items themselves as keys
        rust_items.iter().map(|item| (item.clone_ref(py), item.clone_ref(py))).collect()
    };
    
    // Sort the items
    Python::with_gil(|_py| {
        // Sort the items - we can't use par_sort_by here because of GIL issues
        // Instead, we'll use a regular sort
        keys_and_items.sort_by(|a, b| {
            let a_key = &a.0;
            let b_key = &b.0;
            
            // Use Python's comparison operators
            let lt = Python::with_gil(|py| {
                let a_ref = a_key.as_ref(py);
                let b_ref = b_key.as_ref(py);
                match a_ref.compare(b_ref) {
                    Ok(ordering) => ordering == std::cmp::Ordering::Less,
                    Err(_) => false,
                }
            });
            
            if lt {
                std::cmp::Ordering::Less
            } else {
                let gt = Python::with_gil(|py| {
                    let a_ref = a_key.as_ref(py);
                    let b_ref = b_key.as_ref(py);
                    match a_ref.compare(b_ref) {
                        Ok(ordering) => ordering == std::cmp::Ordering::Greater,
                        Err(_) => false,
                    }
                });
                
                if gt {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            }
        });
    });
    
    // If reverse is true, reverse the sorted list
    if reverse {
        keys_and_items.reverse();
    }
    
    // Extract the sorted items
    let sorted_items: Vec<PyObject> = keys_and_items.into_iter().map(|(_, item)| item).collect();
    
    // Convert back to Python list
    let py_list = PyList::empty(py);
    for item in sorted_items {
        py_list.append(item)?;
    }
    
    Ok(py_list.into())
}

/// Register the data operations module
pub fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parallel_filter, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_map, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_reduce, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_sort, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parallel_filter() {
        Python::with_gil(|py| {
            let items = PyList::new(py, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
            
            // Create a Python function that filters even numbers
            let locals = PyDict::new(py);
            locals.set_item("items", items).unwrap();
            py.run(r#"
def is_even(x):
    return x % 2 == 0
            "#, None, Some(locals)).unwrap();
            
            let is_even = locals.get_item("is_even").unwrap().to_object(py);
            let result = parallel_filter(py, items, is_even).unwrap();
            let result_vec: Vec<i32> = result.extract(py).unwrap();
            
            assert_eq!(result_vec, vec![2, 4, 6, 8, 10]);
        });
    }
    
    #[test]
    fn test_parallel_map() {
        Python::with_gil(|py| {
            let items = PyList::new(py, &[1, 2, 3, 4, 5]);
            
            // Create a Python function that squares numbers
            let locals = PyDict::new(py);
            locals.set_item("items", items).unwrap();
            py.run(r#"
def square(x):
    return x * x
            "#, None, Some(locals)).unwrap();
            
            let square = locals.get_item("square").unwrap().to_object(py);
            let result = parallel_map(py, items, square).unwrap();
            let result_vec: Vec<i32> = result.extract(py).unwrap();
            
            assert_eq!(result_vec, vec![1, 4, 9, 16, 25]);
        });
    }
    
    #[test]
    fn test_parallel_reduce() {
        Python::with_gil(|py| {
            let items = PyList::new(py, &[1, 2, 3, 4, 5]);
            
            // Create a Python function that adds numbers
            let locals = PyDict::new(py);
            locals.set_item("items", items).unwrap();
            py.run(r#"
def add(x, y):
    return x + y
            "#, None, Some(locals)).unwrap();
            
            let add = locals.get_item("add").unwrap().to_object(py);
            let result = parallel_reduce(py, items, add, None).unwrap();
            let sum: i32 = result.extract(py).unwrap();
            
            assert_eq!(sum, 15);
        });
    }
    
    #[test]
    fn test_parallel_sort() {
        Python::with_gil(|py| {
            let items = PyList::new(py, &[5, 2, 8, 1, 9, 3]);
            
            // Sort without a key function
            let result = parallel_sort(py, items, None, None).unwrap();
            let sorted_vec: Vec<i32> = result.extract(py).unwrap();
            
            assert_eq!(sorted_vec, vec![1, 2, 3, 5, 8, 9]);
            
            // Sort in reverse
            let result = parallel_sort(py, items, None, Some(true)).unwrap();
            let sorted_vec: Vec<i32> = result.extract(py).unwrap();
            
            assert_eq!(sorted_vec, vec![9, 8, 5, 3, 2, 1]);
        });
    }
}