//! Matrix operations for Pyroid
//!
//! This module provides high-performance matrix operations.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyFloat, PyInt};
use rayon::prelude::*;
use crate::core::error::PyroidError;
use crate::core::conversion::{ToPyObject, FromPyObject, pylist_to_vec, vec_to_pylist};
use crate::core::config::get_config;
use super::vector::Vector;

/// Matrix class for mathematical operations
#[pyclass]
#[derive(Clone)]
pub struct Matrix {
    data: Vec<Vec<f64>>,
    rows: usize,
    cols: usize,
}

#[pymethods]
impl Matrix {
    /// Create a new matrix
    #[new]
    fn new(values: Option<&PyAny>) -> PyResult<Self> {
        if let Some(values) = values {
            if let Ok(py_list) = values.downcast::<PyList>() {
                // Check if it's a list of lists
                if py_list.len() == 0 {
                    return Ok(Self { data: Vec::new(), rows: 0, cols: 0 });
                }
                
                let mut data = Vec::with_capacity(py_list.len());
                let mut cols = 0;
                
                for (i, item) in py_list.iter().enumerate() {
                    if let Ok(row_list) = item.downcast::<PyList>() {
                        if i == 0 {
                            cols = row_list.len();
                        } else if row_list.len() != cols {
                            return Err(PyroidError::InputError("All rows must have the same length".to_string()).into());
                        }
                        
                        let row = pylist_to_vec::<f64>(row_list)?;
                        data.push(row);
                    } else {
                        return Err(PyroidError::InputError("Expected a list of lists".to_string()).into());
                    }
                }
                
                Ok(Self { data, rows: py_list.len(), cols })
            } else {
                Err(PyroidError::InputError("Expected a list of lists".to_string()).into())
            }
        } else {
            Ok(Self { data: Vec::new(), rows: 0, cols: 0 })
        }
    }
    
    /// Get the number of rows
    #[getter]
    fn rows(&self) -> usize {
        self.rows
    }
    
    /// Get the number of columns
    #[getter]
    fn cols(&self) -> usize {
        self.cols
    }
    
    /// Get the shape of the matrix
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    
    /// Check if the matrix is empty
    fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }
    
    /// Get a value at the specified indices
    fn get(&self, row: usize, col: usize) -> PyResult<f64> {
        if row < self.rows && col < self.cols {
            Ok(self.data[row][col])
        } else {
            Err(PyroidError::InputError(format!("Indices out of bounds: ({}, {})", row, col)).into())
        }
    }
    
    /// Set a value at the specified indices
    fn set(&mut self, row: usize, col: usize, value: f64) -> PyResult<()> {
        if row < self.rows && col < self.cols {
            self.data[row][col] = value;
            Ok(())
        } else {
            Err(PyroidError::InputError(format!("Indices out of bounds: ({}, {})", row, col)).into())
        }
    }
    
    /// Get a row as a Vector
    fn get_row(&self, row: usize) -> PyResult<Vector> {
        if row < self.rows {
            let values = self.data[row].clone();
            Python::with_gil(|py| {
                let py_list = vec_to_pylist(py, &values)?;
                Vector::new(Some(py_list.as_ref(py)))
            })
        } else {
            Err(PyroidError::InputError(format!("Row index out of bounds: {}", row)).into())
        }
    }
    
    /// Get a column as a Vector
    fn get_col(&self, col: usize) -> PyResult<Vector> {
        if col < self.cols {
            let values: Vec<f64> = self.data.iter()
                .map(|row| row[col])
                .collect();
            
            Python::with_gil(|py| {
                let py_list = vec_to_pylist(py, &values)?;
                Vector::new(Some(py_list.as_ref(py)))
            })
        } else {
            Err(PyroidError::InputError(format!("Column index out of bounds: {}", col)).into())
        }
    }
    
    /// Add two matrices
    fn __add__(&self, other: &Matrix) -> PyResult<Matrix> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(PyroidError::InputError("Matrices must have the same dimensions for addition".to_string()).into());
        }
        
        let config = get_config();
        
        let result = if config.get_bool("parallel").unwrap_or(true) && self.rows > 100 {
            let mut result = vec![vec![0.0; self.cols]; self.rows];
            
            result.par_iter_mut().enumerate().for_each(|(i, row)| {
                for j in 0..self.cols {
                    row[j] = self.data[i][j] + other.data[i][j];
                }
            });
            
            result
        } else {
            let mut result = vec![vec![0.0; self.cols]; self.rows];
            
            for i in 0..self.rows {
                for j in 0..self.cols {
                    result[i][j] = self.data[i][j] + other.data[i][j];
                }
            }
            
            result
        };
        
        Ok(Matrix { data: result, rows: self.rows, cols: self.cols })
    }
    
    /// Subtract two matrices
    fn __sub__(&self, other: &Matrix) -> PyResult<Matrix> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(PyroidError::InputError("Matrices must have the same dimensions for subtraction".to_string()).into());
        }
        
        let config = get_config();
        
        let result = if config.get_bool("parallel").unwrap_or(true) && self.rows > 100 {
            let mut result = vec![vec![0.0; self.cols]; self.rows];
            
            result.par_iter_mut().enumerate().for_each(|(i, row)| {
                for j in 0..self.cols {
                    row[j] = self.data[i][j] - other.data[i][j];
                }
            });
            
            result
        } else {
            let mut result = vec![vec![0.0; self.cols]; self.rows];
            
            for i in 0..self.rows {
                for j in 0..self.cols {
                    result[i][j] = self.data[i][j] - other.data[i][j];
                }
            }
            
            result
        };
        
        Ok(Matrix { data: result, rows: self.rows, cols: self.cols })
    }
    
    /// Multiply matrix by a scalar
    fn __mul__(&self, scalar: f64) -> Matrix {
        let config = get_config();
        
        let result = if config.get_bool("parallel").unwrap_or(true) && self.rows > 100 {
            let mut result = vec![vec![0.0; self.cols]; self.rows];
            
            result.par_iter_mut().enumerate().for_each(|(i, row)| {
                for j in 0..self.cols {
                    row[j] = self.data[i][j] * scalar;
                }
            });
            
            result
        } else {
            let mut result = vec![vec![0.0; self.cols]; self.rows];
            
            for i in 0..self.rows {
                for j in 0..self.cols {
                    result[i][j] = self.data[i][j] * scalar;
                }
            }
            
            result
        };
        
        Matrix { data: result, rows: self.rows, cols: self.cols }
    }
    
    /// Matrix multiplication
    fn matmul(&self, other: &Matrix) -> PyResult<Matrix> {
        if self.cols != other.rows {
            return Err(PyroidError::InputError(
                format!("Matrix dimensions incompatible for multiplication: {}x{} and {}x{}",
                    self.rows, self.cols, other.rows, other.cols))
                .into());
        }
        
        let config = get_config();
        
        let result = if config.get_bool("parallel").unwrap_or(true) && self.rows > 50 {
            let mut result = vec![vec![0.0; other.cols]; self.rows];
            
            result.par_iter_mut().enumerate().for_each(|(i, row)| {
                for j in 0..other.cols {
                    let mut sum = 0.0;
                    for k in 0..self.cols {
                        sum += self.data[i][k] * other.data[k][j];
                    }
                    row[j] = sum;
                }
            });
            
            result
        } else {
            let mut result = vec![vec![0.0; other.cols]; self.rows];
            
            for i in 0..self.rows {
                for j in 0..other.cols {
                    let mut sum = 0.0;
                    for k in 0..self.cols {
                        sum += self.data[i][k] * other.data[k][j];
                    }
                    result[i][j] = sum;
                }
            }
            
            result
        };
        
        Ok(Matrix { data: result, rows: self.rows, cols: other.cols })
    }
    
    /// Transpose the matrix
    fn transpose(&self) -> Matrix {
        let mut result = vec![vec![0.0; self.rows]; self.cols];
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[j][i] = self.data[i][j];
            }
        }
        
        Matrix { data: result, rows: self.cols, cols: self.rows }
    }
    
    /// Convert to a list of lists
    fn to_list(&self, py: Python) -> PyResult<Py<PyList>> {
        let outer_list = PyList::empty(py);
        
        for row in &self.data {
            let row_list = PyList::new(py, row);
            outer_list.append(row_list)?;
        }
        
        Ok(outer_list.into())
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        if self.rows == 0 || self.cols == 0 {
            return "Matrix(empty)".to_string();
        }
        
        if self.rows <= 5 && self.cols <= 5 {
            let mut result = String::from("Matrix([\n");
            
            for row in &self.data {
                result.push_str("  [");
                for (j, val) in row.iter().enumerate() {
                    if j > 0 {
                        result.push_str(", ");
                    }
                    result.push_str(&format!("{:.4}", val));
                }
                result.push_str("],\n");
            }
            
            result.push_str("])");
            result
        } else {
            format!("Matrix({}x{})", self.rows, self.cols)
        }
    }
}

/// Matrix multiplication function
#[pyfunction]
fn multiply(py: Python, a: &PyList, b: &PyList) -> PyResult<Py<PyList>> {
    let matrix_a = Matrix::new(Some(a))?;
    let matrix_b = Matrix::new(Some(b))?;
    
    let result = matrix_a.matmul(&matrix_b)?;
    result.to_list(py)
}

/// Register the matrix module
pub fn register(py: Python, module: &PyModule) -> PyResult<()> {
    let matrix_module = PyModule::new(py, "matrix")?;
    
    matrix_module.add_class::<Matrix>()?;
    matrix_module.add_function(wrap_pyfunction!(multiply, matrix_module)?)?;
    
    // Add the matrix module to the parent module
    module.add_submodule(matrix_module)?;
    
    Ok(())
}