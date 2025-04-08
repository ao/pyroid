//! Vector operations for Pyroid
//!
//! This module provides high-performance vector operations.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyFloat, PyInt};
use rayon::prelude::*;
use crate::core::error::PyroidError;
use crate::core::conversion::{ToPyObject, FromPyObject, pylist_to_vec, vec_to_pylist};
use crate::core::config::get_config;

/// Vector class for mathematical operations
#[pyclass]
#[derive(Clone)]
pub struct Vector {
    data: Vec<f64>,
}

#[pymethods]
impl Vector {
    /// Create a new vector
    #[new]
    pub fn new(values: Option<&PyAny>) -> PyResult<Self> {
        if let Some(values) = values {
            if let Ok(py_list) = values.downcast::<PyList>() {
                let data = pylist_to_vec::<f64>(py_list)?;
                Ok(Self { data })
            } else if let Ok(py_float) = values.downcast::<PyFloat>() {
                let value = py_float.value();
                Ok(Self { data: vec![value] })
            } else if let Ok(py_int) = values.downcast::<PyInt>() {
                let value = py_int.extract::<f64>()?;
                Ok(Self { data: vec![value] })
            } else {
                Err(PyroidError::InputError("Expected a list, float, or int".to_string()).into())
            }
        } else {
            Ok(Self { data: Vec::new() })
        }
    }
    
    /// Get the length of the vector
    #[getter]
    fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Check if the vector is empty
    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Get a value at the specified index
    fn get(&self, index: usize) -> PyResult<f64> {
        self.data.get(index)
            .copied()
            .ok_or_else(|| PyroidError::InputError(format!("Index out of bounds: {}", index)).into())
    }
    
    /// Set a value at the specified index
    fn set(&mut self, index: usize, value: f64) -> PyResult<()> {
        if index < self.data.len() {
            self.data[index] = value;
            Ok(())
        } else {
            Err(PyroidError::InputError(format!("Index out of bounds: {}", index)).into())
        }
    }
    
    /// Append a value to the vector
    fn append(&mut self, value: f64) -> PyResult<()> {
        self.data.push(value);
        Ok(())
    }
    
    /// Extend the vector with values from another vector
    fn extend(&mut self, other: &Vector) -> PyResult<()> {
        self.data.extend_from_slice(&other.data);
        Ok(())
    }
    
    /// Get the sum of all elements
    fn sum(&self) -> f64 {
        let config = get_config();
        
        if config.get_bool("parallel").unwrap_or(true) && self.data.len() > 1000 {
            self.data.par_iter().sum()
        } else {
            self.data.iter().sum()
        }
    }
    
    /// Get the product of all elements
    fn product(&self) -> f64 {
        let config = get_config();
        
        if config.get_bool("parallel").unwrap_or(true) && self.data.len() > 1000 {
            self.data.par_iter().product()
        } else {
            self.data.iter().product()
        }
    }
    
    /// Get the mean of all elements
    fn mean(&self) -> PyResult<f64> {
        if self.data.is_empty() {
            return Err(PyroidError::ComputationError("Cannot compute mean of empty vector".to_string()).into());
        }
        
        Ok(self.sum() / self.data.len() as f64)
    }
    
    /// Get the standard deviation of all elements
    fn std(&self) -> PyResult<f64> {
        if self.data.is_empty() {
            return Err(PyroidError::ComputationError("Cannot compute standard deviation of empty vector".to_string()).into());
        }
        
        let mean = self.mean()?;
        let config = get_config();
        
        let variance = if config.get_bool("parallel").unwrap_or(true) && self.data.len() > 1000 {
            self.data.par_iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / self.data.len() as f64
        } else {
            self.data.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / self.data.len() as f64
        };
        
        Ok(variance.sqrt())
    }
    
    /// Add two vectors
    fn __add__(&self, other: &Vector) -> PyResult<Vector> {
        if self.data.len() != other.data.len() {
            return Err(PyroidError::InputError("Vectors must have the same length for addition".to_string()).into());
        }
        
        let config = get_config();
        
        let result = if config.get_bool("parallel").unwrap_or(true) && self.data.len() > 1000 {
            self.data.par_iter()
                .zip(other.data.par_iter())
                .map(|(&a, &b)| a + b)
                .collect()
        } else {
            self.data.iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a + b)
                .collect()
        };
        
        Ok(Vector { data: result })
    }
    
    /// Subtract two vectors
    fn __sub__(&self, other: &Vector) -> PyResult<Vector> {
        if self.data.len() != other.data.len() {
            return Err(PyroidError::InputError("Vectors must have the same length for subtraction".to_string()).into());
        }
        
        let config = get_config();
        
        let result = if config.get_bool("parallel").unwrap_or(true) && self.data.len() > 1000 {
            self.data.par_iter()
                .zip(other.data.par_iter())
                .map(|(&a, &b)| a - b)
                .collect()
        } else {
            self.data.iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a - b)
                .collect()
        };
        
        Ok(Vector { data: result })
    }
    
    /// Multiply vector by a scalar
    fn __mul__(&self, scalar: f64) -> Vector {
        let config = get_config();
        
        let result = if config.get_bool("parallel").unwrap_or(true) && self.data.len() > 1000 {
            self.data.par_iter()
                .map(|&x| x * scalar)
                .collect()
        } else {
            self.data.iter()
                .map(|&x| x * scalar)
                .collect()
        };
        
        Vector { data: result }
    }
    
    /// Divide vector by a scalar
    fn __truediv__(&self, scalar: f64) -> PyResult<Vector> {
        if scalar == 0.0 {
            return Err(PyroidError::ComputationError("Division by zero".to_string()).into());
        }
        
        let config = get_config();
        
        let result = if config.get_bool("parallel").unwrap_or(true) && self.data.len() > 1000 {
            self.data.par_iter()
                .map(|&x| x / scalar)
                .collect()
        } else {
            self.data.iter()
                .map(|&x| x / scalar)
                .collect()
        };
        
        Ok(Vector { data: result })
    }
    
    /// Convert to a list
    fn to_list(&self, py: Python) -> PyResult<Py<PyList>> {
        vec_to_pylist(py, &self.data)
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        if self.data.len() <= 10 {
            format!("Vector({})", self.data.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", "))
        } else {
            format!("Vector([{}, {}, ... {} more elements ... {}, {}])",
                self.data[0], self.data[1],
                self.data.len() - 4,
                self.data[self.data.len() - 2], self.data[self.data.len() - 1])
        }
    }
}

/// Sum a list of numbers in parallel
#[pyfunction]
fn sum(py: Python, values: &PyList) -> PyResult<f64> {
    let data = pylist_to_vec::<f64>(values)?;
    let config = get_config();
    
    let result = if config.get_bool("parallel").unwrap_or(true) && data.len() > 1000 {
        data.par_iter().sum()
    } else {
        data.iter().sum()
    };
    
    Ok(result)
}

/// Register the vector module
pub fn register(py: Python, module: &PyModule) -> PyResult<()> {
    let vector_module = PyModule::new(py, "vector")?;
    
    vector_module.add_class::<Vector>()?;
    vector_module.add_function(wrap_pyfunction!(sum, vector_module)?)?;
    
    // Add the vector module to the parent module
    module.add_submodule(vector_module)?;
    
    Ok(())
}