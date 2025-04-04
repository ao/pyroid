//! DataFrame operations
//!
//! This module provides high-performance DataFrame operations using Polars.

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use pyo3::types::{PyDict, PyList, PyTuple, PyString};
use rayon::prelude::*;
use polars::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Convert a Python dictionary to a Polars DataFrame
fn py_dict_to_df(py: Python, dict: &PyDict) -> PyResult<DataFrame> {
    let mut cols = Vec::new();
    
    for (key, value) in dict.iter() {
        let key_str = key.extract::<String>()?;
        
        if let Ok(list) = value.extract::<Vec<i64>>() {
            cols.push(Series::new(&key_str, list));
        } else if let Ok(list) = value.extract::<Vec<f64>>() {
            cols.push(Series::new(&key_str, list));
        } else if let Ok(list) = value.extract::<Vec<bool>>() {
            cols.push(Series::new(&key_str, list));
        } else if let Ok(list) = value.extract::<Vec<String>>() {
            cols.push(Series::new(&key_str, list));
        } else {
            return Err(PyValueError::new_err(format!("Unsupported column type for {}", key_str)));
        }
    }
    
    DataFrame::new(cols).map_err(|e| PyRuntimeError::new_err(format!("Failed to create DataFrame: {}", e)))
}

/// Convert a Polars DataFrame to a Python dictionary
fn df_to_py_dict(py: Python, df: &DataFrame) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    
    for col in df.get_columns() {
        let name = col.name();
        let py_list = PyList::empty(py);
        
        match col.dtype() {
            DataType::Int64 => {
                let ca = col.i64().map_err(|e| PyRuntimeError::new_err(format!("Failed to get i64 column: {}", e)))?;
                for i in 0..col.len() {
                    if let Some(v) = ca.get(i) {
                        py_list.append(v)?;
                    } else {
                        py_list.append(py.None())?;
                    }
                }
            },
            DataType::Float64 => {
                let ca = col.f64().map_err(|e| PyRuntimeError::new_err(format!("Failed to get f64 column: {}", e)))?;
                for i in 0..col.len() {
                    if let Some(v) = ca.get(i) {
                        py_list.append(v)?;
                    } else {
                        py_list.append(py.None())?;
                    }
                }
            },
            DataType::Boolean => {
                let ca = col.bool().map_err(|e| PyRuntimeError::new_err(format!("Failed to get bool column: {}", e)))?;
                for i in 0..col.len() {
                    if let Some(v) = ca.get(i) {
                        py_list.append(v)?;
                    } else {
                        py_list.append(py.None())?;
                    }
                }
            },
            DataType::String => {
                let ca = col.str().map_err(|e| PyRuntimeError::new_err(format!("Failed to get string column: {}", e)))?;
                for i in 0..col.len() {
                    if let Some(v) = ca.get(i) {
                        py_list.append(v)?;
                    } else {
                        py_list.append(py.None())?;
                    }
                }
            },
            _ => return Err(PyValueError::new_err(format!("Unsupported column type for {}", name))),
        }
        
        dict.set_item(name, py_list)?;
    }
    
    Ok(dict.into())
}

/// Apply a function to a DataFrame in parallel
///
/// Args:
///     df: A dictionary representing the DataFrame (column name -> list of values)
///     func: A function to apply to each row or column
///     axis: 0 for columns, 1 for rows (default: 0)
///
/// Returns:
///     A dictionary representing the resulting DataFrame
#[pyfunction]
fn dataframe_apply(py: Python, df: &PyDict, func: PyObject, axis: Option<i32>) -> PyResult<PyObject> {
    let axis = axis.unwrap_or(0);
    
    // Convert Python dict to Polars DataFrame
    let mut polars_df = py_dict_to_df(py, df)?;
    
    if axis == 0 {
        // Apply to columns
        let column_names: Vec<String> = polars_df.get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
            
        let results: Result<Vec<Series>, PyErr> = column_names.par_iter()
            .map(|name| {
                Python::with_gil(|py| {
                    let col = polars_df.column(name)
                        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get column {}: {}", name, e)))?;
                    
                    // Convert column to Python list
                    let py_col = match col.dtype() {
                        DataType::Int64 => {
                            let values: Vec<i64> = col.i64()
                                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                                .into_iter()
                                .collect::<Vec<Option<i64>>>()
                                .into_iter()
                                .map(|v| v.unwrap_or(0))
                                .collect();
                            PyList::new(py, &values).to_object(py)
                        },
                        DataType::Float64 => {
                            let values: Vec<f64> = col.f64()
                                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                                .into_iter()
                                .collect::<Vec<Option<f64>>>()
                                .into_iter()
                                .map(|v| v.unwrap_or(0.0))
                                .collect();
                            PyList::new(py, &values).to_object(py)
                        },
                        DataType::Boolean => {
                            let values: Vec<bool> = col.bool()
                                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                                .into_iter()
                                .collect::<Vec<Option<bool>>>()
                                .into_iter()
                                .map(|v| v.unwrap_or(false))
                                .collect();
                            PyList::new(py, &values).to_object(py)
                        },
                        DataType::String => {
                            let values: Vec<String> = col.str()
                                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                                .into_iter()
                                .collect::<Vec<Option<&str>>>()
                                .into_iter()
                                .map(|v| v.unwrap_or("").to_string())
                                .collect();
                            PyList::new(py, &values).to_object(py)
                        },
                        _ => return Err(PyValueError::new_err(format!("Unsupported column type for {}", name))),
                    };
                    
                    // Call the Python function with the column
                    let result = func.call1(py, (py_col,))?;
                    
                    // Convert the result back to a Series
                    if let Ok(values) = result.extract::<Vec<i64>>(py) {
                        Ok(Series::new(name, values))
                    } else if let Ok(values) = result.extract::<Vec<f64>>(py) {
                        Ok(Series::new(name, values))
                    } else if let Ok(values) = result.extract::<Vec<bool>>(py) {
                        Ok(Series::new(name, values))
                    } else if let Ok(values) = result.extract::<Vec<String>>(py) {
                        Ok(Series::new(name, values))
                    } else {
                        Err(PyValueError::new_err(format!("Unsupported return type for column {}", name)))
                    }
                })
            })
            .collect();
            
        // Create a new DataFrame from the results
        let result_df = DataFrame::new(results?)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create result DataFrame: {}", e)))?;
            
        // Convert back to Python dict
        df_to_py_dict(py, &result_df)
    } else {
        // Apply to rows
        let num_rows = polars_df.height();
        let column_names: Vec<String> = polars_df.get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
            
        // Process rows in parallel
        let results: Result<Vec<Vec<AnyValue>>, PyErr> = (0..num_rows).into_par_iter()
            .map(|row_idx| {
                Python::with_gil(|py| {
                    // Create a Python dict for the row
                    let row_dict = PyDict::new(py);
                    
                    for name in &column_names {
                        let col = polars_df.column(name)
                            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get column {}: {}", name, e)))?;
                            
                        match col.dtype() {
                            DataType::Int64 => {
                                let value = col.i64()
                                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                                    .get(row_idx);
                                if let Some(v) = value {
                                    row_dict.set_item(name, v)?;
                                } else {
                                    row_dict.set_item(name, py.None())?;
                                }
                            },
                            DataType::Float64 => {
                                let value = col.f64()
                                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                                    .get(row_idx);
                                if let Some(v) = value {
                                    row_dict.set_item(name, v)?;
                                } else {
                                    row_dict.set_item(name, py.None())?;
                                }
                            },
                            DataType::Boolean => {
                                let value = col.bool()
                                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                                    .get(row_idx);
                                if let Some(v) = value {
                                    row_dict.set_item(name, v)?;
                                } else {
                                    row_dict.set_item(name, py.None())?;
                                }
                            },
                            DataType::String => {
                                let value = col.str()
                                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                                    .get(row_idx);
                                if let Some(v) = value {
                                    row_dict.set_item(name, v)?;
                                } else {
                                    row_dict.set_item(name, py.None())?;
                                }
                            },
                            _ => return Err(PyValueError::new_err(format!("Unsupported column type for {}", name))),
                        }
                    }
                    
                    // Call the Python function with the row
                    let result = func.call1(py, (row_dict,))?;
                    
                    // Convert the result to a row
                    if let Ok(result_dict) = result.extract::<&PyDict>(py) {
                        let mut row_values = Vec::new();
                        
                        for name in &column_names {
                            if let Some(value) = result_dict.get_item(name) {
                                if let Ok(v) = value.extract::<i64>(py) {
                                    row_values.push(AnyValue::Int64(v));
                                } else if let Ok(v) = value.extract::<f64>(py) {
                                    row_values.push(AnyValue::Float64(v));
                                } else if let Ok(v) = value.extract::<bool>(py) {
                                    row_values.push(AnyValue::Boolean(v));
                                } else if let Ok(v) = value.extract::<String>(py) {
                                    row_values.push(AnyValue::String(v.into()));
                                } else {
                                    row_values.push(AnyValue::Null);
                                }
                            } else {
                                row_values.push(AnyValue::Null);
                            }
                        }
                        
                        Ok(row_values)
                    } else {
                        Err(PyValueError::new_err("Function must return a dictionary for row-wise operations"))
                    }
                })
            })
            .collect();
            
        // Create a new DataFrame from the results
        let rows = results?;
        let mut columns = Vec::new();
        
        for (col_idx, name) in column_names.iter().enumerate() {
            let mut col_values = Vec::new();
            
            for row in &rows {
                col_values.push(row[col_idx].clone());
            }
            
            let series = Series::from_any_values(name, &col_values)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create series: {}", e)))?;
                
            columns.push(series);
        }
        
        let result_df = DataFrame::new(columns)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create result DataFrame: {}", e)))?;
            
        // Convert back to Python dict
        df_to_py_dict(py, &result_df)
    }
}

/// Perform groupby and aggregation operations on a DataFrame in parallel
///
/// Args:
///     df: A dictionary representing the DataFrame (column name -> list of values)
///     group_cols: A list of column names to group by
///     agg_dict: A dictionary mapping column names to aggregation functions
///
/// Returns:
///     A dictionary representing the resulting DataFrame
#[pyfunction]
fn dataframe_groupby_aggregate(py: Python, df: &PyDict, group_cols: Vec<String>, agg_dict: &PyDict) -> PyResult<PyObject> {
    // Convert Python dict to Polars DataFrame
    let polars_df = py_dict_to_df(py, df)?;
    
    // Create a lazy DataFrame for efficient groupby operations
    let mut lazy_df = polars_df.lazy();
    
    // Build the groupby expression
    let mut group_by_exprs = Vec::new();
    for col in group_cols {
        group_by_exprs.push(col(col.as_str()));
    }
    
    // Build the aggregation expressions
    let mut agg_exprs = Vec::new();
    for (col_name, agg_func) in agg_dict.iter() {
        let col_name = col_name.extract::<String>()?;
        let agg_func_str = agg_func.extract::<String>()?;
        
        match agg_func_str.as_str() {
            "sum" => agg_exprs.push(col(&col_name).sum().alias(&format!("{}_sum", col_name))),
            "mean" => agg_exprs.push(col(&col_name).mean().alias(&format!("{}_mean", col_name))),
            "min" => agg_exprs.push(col(&col_name).min().alias(&format!("{}_min", col_name))),
            "max" => agg_exprs.push(col(&col_name).max().alias(&format!("{}_max", col_name))),
            "count" => agg_exprs.push(col(&col_name).count().alias(&format!("{}_count", col_name))),
            "std" => agg_exprs.push(col(&col_name).std(1).alias(&format!("{}_std", col_name))),
            _ => return Err(PyValueError::new_err(format!("Unsupported aggregation function: {}", agg_func_str))),
        }
    }
    
    // Perform the groupby and aggregation
    let result_df = lazy_df
        .group_by(group_by_exprs)
        .agg(agg_exprs)
        .collect()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to perform groupby: {}", e)))?;
    
    // Convert back to Python dict
    df_to_py_dict(py, &result_df)
}

/// Apply multiple transformations to a DataFrame in one pass
///
/// Args:
///     df: A dictionary representing the DataFrame (column name -> list of values)
///     transformations: A list of (column_name, operation, args) tuples
///
/// Returns:
///     A dictionary representing the resulting DataFrame
#[pyfunction]
fn parallel_transform(py: Python, df: &PyDict, transformations: Vec<(String, String, Option<PyObject>)>) -> PyResult<PyObject> {
    // Convert Python dict to Polars DataFrame
    let mut polars_df = py_dict_to_df(py, df)?;
    
    // Apply transformations in parallel
    let results: Result<Vec<(String, Series)>, PyErr> = transformations.par_iter()
        .map(|(col_name, operation, args)| {
            let col = polars_df.column(col_name)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to get column {}: {}", col_name, e)))?;
                
            let result_series = match operation.as_str() {
                "log" => {
                    let f_col = col.f64()
                        .map_err(|e| PyRuntimeError::new_err(format!("Column {} must be numeric for log: {}", col_name, e)))?;
                    let result: Float64Chunked = f_col.apply(|opt_v| opt_v.map(|v| v.ln()));
                    Series::new(&format!("{}_log", col_name), result)
                },
                "sqrt" => {
                    let f_col = col.f64()
                        .map_err(|e| PyRuntimeError::new_err(format!("Column {} must be numeric for sqrt: {}", col_name, e)))?;
                    let result: Float64Chunked = f_col.apply(|opt_v| opt_v.map(|v| v.sqrt()));
                    Series::new(&format!("{}_sqrt", col_name), result)
                },
                "abs" => {
                    let f_col = col.f64()
                        .map_err(|e| PyRuntimeError::new_err(format!("Column {} must be numeric for abs: {}", col_name, e)))?;
                    let result: Float64Chunked = f_col.apply(|opt_v| opt_v.map(|v| v.abs()));
                    Series::new(&format!("{}_abs", col_name), result)
                },
                "round" => {
                    let decimals = if let Some(args_obj) = args {
                        Python::with_gil(|py| args_obj.extract::<i32>(py).unwrap_or(0))
                    } else {
                        0
                    };
                    
                    let f_col = col.f64()
                        .map_err(|e| PyRuntimeError::new_err(format!("Column {} must be numeric for round: {}", col_name, e)))?;
                    let factor = 10.0_f64.powi(decimals);
                    let result: Float64Chunked = f_col.apply(move |opt_v| 
                        opt_v.map(|v| (v * factor).round() / factor)
                    );
                    Series::new(&format!("{}_round", col_name), result)
                },
                "fillna" => {
                    let fill_value = if let Some(args_obj) = args {
                        Python::with_gil(|py| {
                            if let Ok(v) = args_obj.extract::<i64>(py) {
                                AnyValue::Int64(v)
                            } else if let Ok(v) = args_obj.extract::<f64>(py) {
                                AnyValue::Float64(v)
                            } else if let Ok(v) = args_obj.extract::<String>(py) {
                                AnyValue::String(v.into())
                            } else if let Ok(v) = args_obj.extract::<bool>(py) {
                                AnyValue::Boolean(v)
                            } else {
                                return Err(PyValueError::new_err("Unsupported fill value type"));
                            }
                        })?
                    } else {
                        return Err(PyValueError::new_err("fillna requires a fill value"));
                    };
                    
                    let result = col.fill_null(fill_value)
                        .map_err(|e| PyRuntimeError::new_err(format!("Failed to fill nulls: {}", e)))?;
                    Series::new(col_name, result)
                },
                _ => return Err(PyValueError::new_err(format!("Unsupported operation: {}", operation))),
            };
            
            Ok((col_name.clone(), result_series))
        })
        .collect();
        
    // Add the transformed columns to the DataFrame
    for (_, series) in results? {
        polars_df.with_column(series)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to add column: {}", e)))?;
    }
    
    // Convert back to Python dict
    df_to_py_dict(py, &polars_df)
}

/// Join two DataFrames in parallel
///
/// Args:
///     left: A dictionary representing the left DataFrame
///     right: A dictionary representing the right DataFrame
///     on: Column name(s) to join on
///     how: Join type (inner, left, right, outer)
///
/// Returns:
///     A dictionary representing the joined DataFrame
#[pyfunction]
fn parallel_join(py: Python, left: &PyDict, right: &PyDict, on: PyObject, how: Option<String>) -> PyResult<PyObject> {
    // Convert Python dicts to Polars DataFrames
    let left_df = py_dict_to_df(py, left)?;
    let right_df = py_dict_to_df(py, right)?;
    
    // Determine join columns
    let join_cols: Vec<String> = if let Ok(col) = on.extract::<String>(py) {
        vec![col]
    } else if let Ok(cols) = on.extract::<Vec<String>>(py) {
        cols
    } else {
        return Err(PyValueError::new_err("'on' must be a string or list of strings"));
    };
    
    // Determine join type
    let join_type = match how.as_deref().unwrap_or("inner") {
        "inner" => JoinType::Inner,
        "left" => JoinType::Left,
        "right" => JoinType::Right,
        "outer" => JoinType::Outer,
        _ => return Err(PyValueError::new_err("'how' must be one of: inner, left, right, outer")),
    };
    
    // Perform the join
    let result_df = left_df.join(&right_df, &join_cols, &join_cols, join_type, None)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to join DataFrames: {}", e)))?;
    
    // Convert back to Python dict
    df_to_py_dict(py, &result_df)
}

/// Register the dataframe operations module
pub fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dataframe_apply, m)?)?;
    m.add_function(wrap_pyfunction!(dataframe_groupby_aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_transform, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_join, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dataframe_creation() {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("a", vec![1, 2, 3]).unwrap();
            dict.set_item("b", vec![4.0, 5.0, 6.0]).unwrap();
            
            let df = py_dict_to_df(py, dict).unwrap();
            assert_eq!(df.shape(), (3, 2));
        });
    }
}