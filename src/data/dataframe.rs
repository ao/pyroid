//! DataFrame operations for Pyroid
//!
//! This module provides high-performance DataFrame operations.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple, PyString};
use std::collections::HashMap;
use crate::core::error::PyroidError;

/// A simple column-oriented dataframe implementation
#[derive(Clone)]
struct DataFrame {
    columns: HashMap<String, Column>,
    row_count: usize,
}

/// A column in a dataframe
#[derive(Clone)]
enum Column {
    Int(Vec<Option<i64>>),
    Float(Vec<Option<f64>>),
    Bool(Vec<Option<bool>>),
    String(Vec<Option<String>>),
}

impl DataFrame {
    /// Create a new DataFrame from a Python dictionary
    fn from_pydict(dict: &PyDict) -> PyResult<Self> {
        let mut columns = HashMap::new();
        let mut row_count = 0;
        
        for (key, value) in dict.iter() {
            let key_str = key.extract::<String>()?;
            
            if let Ok(list) = value.downcast::<PyList>() {
                if list.is_empty() {
                    continue;
                }
                
                // Set row count from the first column
                if row_count == 0 {
                    row_count = list.len();
                } else if list.len() != row_count {
                    return Err(PyroidError::InputError(format!(
                        "Column {} has length {}, expected {}",
                        key_str, list.len(), row_count
                    )).into());
                }
                
                // Try to determine the type of the list
                let first_item = list.get_item(0)?;
                
                if let Ok(_) = first_item.extract::<i64>() {
                    // Integer list
                    let mut values = Vec::with_capacity(list.len());
                    for i in 0..list.len() {
                        let item = list.get_item(i)?;
                        if item.is_none() {
                            values.push(None);
                        } else {
                            values.push(Some(item.extract::<i64>()?));
                        }
                    }
                    columns.insert(key_str, Column::Int(values));
                } else if let Ok(_) = first_item.extract::<f64>() {
                    // Float list
                    let mut values = Vec::with_capacity(list.len());
                    for i in 0..list.len() {
                        let item = list.get_item(i)?;
                        if item.is_none() {
                            values.push(None);
                        } else {
                            values.push(Some(item.extract::<f64>()?));
                        }
                    }
                    columns.insert(key_str, Column::Float(values));
                } else if let Ok(_) = first_item.extract::<bool>() {
                    // Boolean list
                    let mut values = Vec::with_capacity(list.len());
                    for i in 0..list.len() {
                        let item = list.get_item(i)?;
                        if item.is_none() {
                            values.push(None);
                        } else {
                            values.push(Some(item.extract::<bool>()?));
                        }
                    }
                    columns.insert(key_str, Column::Bool(values));
                } else if let Ok(_) = first_item.extract::<String>() {
                    // String list
                    let mut values = Vec::with_capacity(list.len());
                    for i in 0..list.len() {
                        let item = list.get_item(i)?;
                        if item.is_none() {
                            values.push(None);
                        } else {
                            values.push(Some(item.extract::<String>()?));
                        }
                    }
                    columns.insert(key_str, Column::String(values));
                } else {
                    return Err(PyroidError::ConversionError(format!(
                        "Unsupported column type for {}", key_str
                    )).into());
                }
            } else {
                return Err(PyroidError::ConversionError(format!(
                    "Column values must be lists for {}", key_str
                )).into());
            }
        }
        
        Ok(Self { columns, row_count })
    }
    
    /// Convert to a Python dictionary
    fn to_pydict(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        
        for (name, column) in &self.columns {
            let py_list = PyList::empty(py);
            
            match column {
                Column::Int(values) => {
                    for value in values {
                        match value {
                            Some(v) => py_list.append(*v)?,
                            None => py_list.append(py.None())?,
                        }
                    }
                },
                Column::Float(values) => {
                    for value in values {
                        match value {
                            Some(v) => py_list.append(*v)?,
                            None => py_list.append(py.None())?,
                        }
                    }
                },
                Column::Bool(values) => {
                    for value in values {
                        match value {
                            Some(v) => py_list.append(*v)?,
                            None => py_list.append(py.None())?,
                        }
                    }
                },
                Column::String(values) => {
                    for value in values {
                        match value {
                            Some(ref v) => py_list.append(v)?,
                            None => py_list.append(py.None())?,
                        }
                    }
                },
            }
            
            dict.set_item(name, py_list)?;
        }
        
        Ok(dict.into())
    }
    
    /// Apply a function to each column
    fn apply_to_columns(&self, py: Python, func: &PyAny) -> PyResult<Self> {
        if !func.is_callable() {
            return Err(PyroidError::InputError("Function must be callable".to_string()).into());
        }
        
        let mut result_columns = HashMap::new();
        
        for (name, column) in &self.columns {
            // Convert column to Python list
            let py_list = PyList::empty(py);
            
            match column {
                Column::Int(values) => {
                    for value in values {
                        match value {
                            Some(v) => py_list.append(*v)?,
                            None => py_list.append(py.None())?,
                        }
                    }
                },
                Column::Float(values) => {
                    for value in values {
                        match value {
                            Some(v) => py_list.append(*v)?,
                            None => py_list.append(py.None())?,
                        }
                    }
                },
                Column::Bool(values) => {
                    for value in values {
                        match value {
                            Some(v) => py_list.append(*v)?,
                            None => py_list.append(py.None())?,
                        }
                    }
                },
                Column::String(values) => {
                    for value in values {
                        match value {
                            Some(ref v) => py_list.append(v)?,
                            None => py_list.append(py.None())?,
                        }
                    }
                },
            }
            
            // Call the function with the column
            let args = PyTuple::new(py, &[py_list]);
            let result = func.call1(args)?;
            
            // Convert the result back to a Column
            if let Ok(result_list) = result.downcast::<PyList>() {
                if result_list.is_empty() {
                    result_columns.insert(name.clone(), column.clone());
                    continue;
                }
                
                let first_item = result_list.get_item(0)?;
                
                if let Ok(_) = first_item.extract::<i64>() {
                    // Integer list
                    let mut values = Vec::with_capacity(result_list.len());
                    for i in 0..result_list.len() {
                        let item = result_list.get_item(i)?;
                        if item.is_none() {
                            values.push(None);
                        } else {
                            values.push(Some(item.extract::<i64>()?));
                        }
                    }
                    result_columns.insert(name.clone(), Column::Int(values));
                } else if let Ok(_) = first_item.extract::<f64>() {
                    // Float list
                    let mut values = Vec::with_capacity(result_list.len());
                    for i in 0..result_list.len() {
                        let item = result_list.get_item(i)?;
                        if item.is_none() {
                            values.push(None);
                        } else {
                            values.push(Some(item.extract::<f64>()?));
                        }
                    }
                    result_columns.insert(name.clone(), Column::Float(values));
                } else if let Ok(_) = first_item.extract::<bool>() {
                    // Boolean list
                    let mut values = Vec::with_capacity(result_list.len());
                    for i in 0..result_list.len() {
                        let item = result_list.get_item(i)?;
                        if item.is_none() {
                            values.push(None);
                        } else {
                            values.push(Some(item.extract::<bool>()?));
                        }
                    }
                    result_columns.insert(name.clone(), Column::Bool(values));
                } else if let Ok(_) = first_item.extract::<String>() {
                    // String list
                    let mut values = Vec::with_capacity(result_list.len());
                    for i in 0..result_list.len() {
                        let item = result_list.get_item(i)?;
                        if item.is_none() {
                            values.push(None);
                        } else {
                            values.push(Some(item.extract::<String>()?));
                        }
                    }
                    result_columns.insert(name.clone(), Column::String(values));
                } else {
                    return Err(PyroidError::ConversionError(format!(
                        "Unsupported result type for column {}", name
                    )).into());
                }
            } else {
                return Err(PyroidError::ConversionError(
                    "Function must return a list".to_string()
                ).into());
            }
        }
        
        Ok(Self {
            columns: result_columns,
            row_count: self.row_count,
        })
    }
}

/// DataFrame class for data operations
#[pyclass]
#[derive(Clone)]
pub struct PyDataFrame {
    df: DataFrame,
}

#[pymethods]
impl PyDataFrame {
    /// Create a new DataFrame
    #[new]
    fn new(data: &PyDict) -> PyResult<Self> {
        let df = DataFrame::from_pydict(data)?;
        Ok(Self { df })
    }
    
    /// Get the number of rows
    #[getter]
    fn rows(&self) -> usize {
        self.df.row_count
    }
    
    /// Get the number of columns
    #[getter]
    fn cols(&self) -> usize {
        self.df.columns.len()
    }
    
    /// Get the shape of the DataFrame
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.df.row_count, self.df.columns.len())
    }
    
    /// Get the column names
    #[getter]
    fn columns(&self, py: Python) -> PyResult<Py<PyList>> {
        let names: Vec<&str> = self.df.columns.keys().map(|s| s.as_str()).collect();
        let py_list = PyList::new(py, &names);
        Ok(py_list.into())
    }
    
    /// Apply a function to each column
    fn apply(&self, py: Python, func: &PyAny, axis: Option<i32>) -> PyResult<Self> {
        let axis = axis.unwrap_or(0);
        
        if axis == 0 {
            // Apply to columns
            let result_df = self.df.apply_to_columns(py, func)?;
            Ok(Self { df: result_df })
        } else {
            // Apply to rows - not implemented yet
            Err(PyroidError::InputError("Row-wise apply not implemented yet".to_string()).into())
        }
    }
    
    /// Convert to a dictionary
    fn to_dict(&self, py: Python) -> PyResult<Py<PyDict>> {
        self.df.to_pydict(py)
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!("DataFrame({}x{})", self.df.row_count, self.df.columns.len())
    }
    
    /// String representation
    fn __str__(&self) -> String {
        let mut result = String::new();
        result.push_str(&format!("DataFrame({}x{})\n", self.df.row_count, self.df.columns.len()));
        
        // Show column names
        result.push_str("Columns: [");
        let mut first = true;
        for name in self.df.columns.keys() {
            if !first {
                result.push_str(", ");
            }
            result.push_str(name);
            first = false;
        }
        result.push_str("]\n");
        
        // Show first few rows
        let max_rows = 5;
        let max_rows_to_show = std::cmp::min(self.df.row_count, max_rows);
        
        if max_rows_to_show > 0 {
            result.push_str("Data preview:\n");
            
            // Get column names
            let column_names: Vec<&String> = self.df.columns.keys().collect();
            
            // Print header
            for (i, name) in column_names.iter().enumerate() {
                if i > 0 {
                    result.push_str(" | ");
                }
                result.push_str(name);
            }
            result.push_str("\n");
            
            // Print rows
            for row_idx in 0..max_rows_to_show {
                for (i, name) in column_names.iter().enumerate() {
                    if i > 0 {
                        result.push_str(" | ");
                    }
                    
                    match &self.df.columns[*name] {
                        Column::Int(values) => {
                            match &values[row_idx] {
                                Some(v) => result.push_str(&v.to_string()),
                                None => result.push_str("null"),
                            }
                        },
                        Column::Float(values) => {
                            match &values[row_idx] {
                                Some(v) => result.push_str(&format!("{:.4}", v)),
                                None => result.push_str("null"),
                            }
                        },
                        Column::Bool(values) => {
                            match &values[row_idx] {
                                Some(v) => result.push_str(&v.to_string()),
                                None => result.push_str("null"),
                            }
                        },
                        Column::String(values) => {
                            match &values[row_idx] {
                                Some(v) => result.push_str(v),
                                None => result.push_str("null"),
                            }
                        },
                    }
                }
                result.push_str("\n");
            }
            
            // Show ellipsis if there are more rows
            if self.df.row_count > max_rows {
                result.push_str("...\n");
            }
        }
        
        result
    }
}

/// Apply a function to a DataFrame
#[pyfunction]
fn apply(py: Python, df: &PyDict, func: &PyAny, axis: Option<i32>) -> PyResult<Py<PyDict>> {
    let dataframe = PyDataFrame::new(df)?;
    let result = dataframe.apply(py, func, axis)?;
    result.to_dict(py)
}

/// Register the dataframe module
pub fn register(py: Python, module: &PyModule) -> PyResult<()> {
    let dataframe_module = PyModule::new(py, "dataframe")?;
    
    dataframe_module.add_class::<PyDataFrame>()?;
    dataframe_module.add_function(wrap_pyfunction!(apply, dataframe_module)?)?;
    
    // Add the dataframe module to the parent module
    module.add_submodule(dataframe_module)?;
    
    Ok(())
}