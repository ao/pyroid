//! Statistical operations for Pyroid
//!
//! This module provides high-performance statistical operations.

use pyo3::prelude::*;
use std::cmp::Ordering;
use pyo3::types::{PyList, PyDict};
use rayon::prelude::*;
use crate::core::error::PyroidError;
use crate::core::conversion::{ToPyObject, FromPyObject, pylist_to_vec, vec_to_pylist};
use crate::core::config::get_config;

/// Calculate the mean of a list of numbers
#[pyfunction]
fn mean(py: Python, values: &PyList) -> PyResult<f64> {
    let data = pylist_to_vec::<f64>(values)?;
    
    if data.is_empty() {
        return Err(PyroidError::ComputationError("Cannot compute mean of empty list".to_string()).into());
    }
    
    let sum: f64 = data.iter().sum();
    Ok(sum / data.len() as f64)
}

/// Calculate the median of a list of numbers
#[pyfunction]
fn median(py: Python, values: &PyList) -> PyResult<f64> {
    let mut data = pylist_to_vec::<f64>(values)?;
    
    if data.is_empty() {
        return Err(PyroidError::ComputationError("Cannot compute median of empty list".to_string()).into());
    }
    
    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    
    if data.len() % 2 == 0 {
        // Even number of elements
        let mid = data.len() / 2;
        Ok((data[mid - 1] + data[mid]) / 2.0)
    } else {
        // Odd number of elements
        Ok(data[data.len() / 2])
    }
}

/// Calculate the standard deviation of a list of numbers
#[pyfunction]
fn calc_std(py: Python, values: &PyList, ddof: Option<usize>) -> PyResult<f64> {
    let data = pylist_to_vec::<f64>(values)?;
    let ddof = ddof.unwrap_or(0);
    
    if data.len() <= ddof {
        return Err(PyroidError::ComputationError(
            format!("Cannot compute standard deviation with {} degrees of freedom from {} samples", 
                ddof, data.len())
        ).into());
    }
    
    let mean_val = mean(py, values)?;
    let config = get_config();
    
    let variance = if config.get_bool("parallel").unwrap_or(true) && data.len() > 1000 {
        data.par_iter()
            .map(|&x| (x - mean_val).powi(2))
            .sum::<f64>() / (data.len() - ddof) as f64
    } else {
        data.iter()
            .map(|&x| (x - mean_val).powi(2))
            .sum::<f64>() / (data.len() - ddof) as f64
    };
    
    Ok(variance.sqrt())
}

/// Calculate the variance of a list of numbers
#[pyfunction]
fn variance(py: Python, values: &PyList, ddof: Option<usize>) -> PyResult<f64> {
    let std_val = calc_std(py, values, ddof)?;
    Ok(std_val.powi(2))
}

/// Calculate the correlation coefficient between two lists of numbers
#[pyfunction]
fn correlation(py: Python, x: &PyList, y: &PyList) -> PyResult<f64> {
    let x_data = pylist_to_vec::<f64>(x)?;
    let y_data = pylist_to_vec::<f64>(y)?;
    
    if x_data.len() != y_data.len() {
        return Err(PyroidError::InputError("Lists must have the same length".to_string()).into());
    }
    
    if x_data.is_empty() {
        return Err(PyroidError::ComputationError("Cannot compute correlation of empty lists".to_string()).into());
    }
    
    let n = x_data.len() as f64;
    let x_mean = x_data.iter().sum::<f64>() / n;
    let y_mean = y_data.iter().sum::<f64>() / n;
    
    let mut numerator = 0.0;
    let mut x_variance = 0.0;
    let mut y_variance = 0.0;
    
    for i in 0..x_data.len() {
        let x_diff = x_data[i] - x_mean;
        let y_diff = y_data[i] - y_mean;
        
        numerator += x_diff * y_diff;
        x_variance += x_diff.powi(2);
        y_variance += y_diff.powi(2);
    }
    
    if x_variance == 0.0 || y_variance == 0.0 {
        return Err(PyroidError::ComputationError("Cannot compute correlation when one of the variables has zero variance".to_string()).into());
    }
    
    Ok(numerator / (x_variance.sqrt() * y_variance.sqrt()))
}

/// Calculate descriptive statistics for a list of numbers
#[pyfunction]
fn describe(py: Python, values: &PyList) -> PyResult<Py<PyDict>> {
    let data = pylist_to_vec::<f64>(values)?;
    
    if data.is_empty() {
        return Err(PyroidError::ComputationError("Cannot compute statistics of empty list".to_string()).into());
    }
    
    let count = data.len();
    let mean_val = mean(py, values)?;
    let std_val = calc_std(py, values, Some(0))?;
    let min_val = *data.iter().min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal)).unwrap();
    let max_val = *data.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal)).unwrap();
    
    // Calculate percentiles
    let mut sorted_data = data.clone();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    
    let p25_idx = (0.25 * (count - 1) as f64).round() as usize;
    let p50_idx = (0.50 * (count - 1) as f64).round() as usize;
    let p75_idx = (0.75 * (count - 1) as f64).round() as usize;
    
    let dict = PyDict::new(py);
    dict.set_item("count", count)?;
    dict.set_item("mean", mean_val)?;
    dict.set_item("std", std_val)?;
    dict.set_item("min", min_val)?;
    dict.set_item("25%", sorted_data[p25_idx])?;
    dict.set_item("50%", sorted_data[p50_idx])?;
    dict.set_item("75%", sorted_data[p75_idx])?;
    dict.set_item("max", max_val)?;
    
    Ok(dict.into())
}

/// Register the stats module
pub fn register(py: Python, module: &PyModule) -> PyResult<()> {
    let stats_module = PyModule::new(py, "stats")?;
    
    stats_module.add_function(wrap_pyfunction!(mean, stats_module)?)?;
    stats_module.add_function(wrap_pyfunction!(median, stats_module)?)?;
    stats_module.add_function(wrap_pyfunction!(calc_std, stats_module)?)?;
    stats_module.add_function(wrap_pyfunction!(variance, stats_module)?)?;
    stats_module.add_function(wrap_pyfunction!(correlation, stats_module)?)?;
    stats_module.add_function(wrap_pyfunction!(describe, stats_module)?)?;
    
    // Add the stats module to the parent module
    module.add_submodule(stats_module)?;
    
    Ok(())
}