//! Basic machine learning operations for Pyroid
//!
//! This module provides basic machine learning operations without external dependencies.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use crate::core::error::PyroidError;

/// K-means clustering algorithm
#[pyfunction]
fn kmeans(py: Python, data: &PyList, k: usize, max_iterations: Option<usize>) -> PyResult<Py<PyDict>> {
    let max_iterations = max_iterations.unwrap_or(100);
    
    // Convert data to Vec<Vec<f64>>
    let mut points = Vec::with_capacity(data.len());
    let mut dimensions = 0;
    
    for i in 0..data.len() {
        let point = data.get_item(i)?;
        let point_list = point.downcast::<PyList>()?;
        
        if i == 0 {
            dimensions = point_list.len();
        } else if point_list.len() != dimensions {
            return Err(PyroidError::InputError(format!(
                "All points must have the same dimensions. Expected {}, got {}",
                dimensions, point_list.len()
            )).into());
        }
        
        let mut point_vec = Vec::with_capacity(dimensions);
        for j in 0..dimensions {
            let value = point_list.get_item(j)?.extract::<f64>()?;
            point_vec.push(value);
        }
        
        points.push(point_vec);
    }
    
    if points.len() < k {
        return Err(PyroidError::InputError(format!(
            "Number of points ({}) must be greater than or equal to k ({})",
            points.len(), k
        )).into());
    }
    
    // Initialize centroids randomly
    let mut centroids = Vec::with_capacity(k);
    let mut used_indices = std::collections::HashSet::new();
    
    while centroids.len() < k {
        let idx = (rand() * points.len() as f64) as usize;
        if used_indices.insert(idx) {
            centroids.push(points[idx].clone());
        }
    }
    
    // Run k-means algorithm
    let mut clusters = vec![0; points.len()];
    let mut iterations = 0;
    let mut changed = true;
    
    while changed && iterations < max_iterations {
        changed = false;
        iterations += 1;
        
        // Assign points to clusters
        for (i, point) in points.iter().enumerate() {
            let mut min_dist = f64::MAX;
            let mut min_cluster = 0;
            
            for (j, centroid) in centroids.iter().enumerate() {
                let dist = euclidean_distance(point, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    min_cluster = j;
                }
            }
            
            if clusters[i] != min_cluster {
                clusters[i] = min_cluster;
                changed = true;
            }
        }
        
        if !changed {
            break;
        }
        
        // Update centroids
        let mut new_centroids = vec![vec![0.0; dimensions]; k];
        let mut counts = vec![0; k];
        
        for (i, point) in points.iter().enumerate() {
            let cluster = clusters[i];
            counts[cluster] += 1;
            
            for j in 0..dimensions {
                new_centroids[cluster][j] += point[j];
            }
        }
        
        for i in 0..k {
            if counts[i] > 0 {
                for j in 0..dimensions {
                    new_centroids[i][j] /= counts[i] as f64;
                }
                centroids[i] = new_centroids[i].clone();
            }
        }
    }
    
    // Create result dictionary
    let result = PyDict::new(py);
    
    // Add centroids
    let centroids_list = PyList::empty(py);
    for centroid in centroids {
        let centroid_list = PyList::empty(py);
        for value in centroid {
            centroid_list.append(value)?;
        }
        centroids_list.append(centroid_list)?;
    }
    result.set_item("centroids", centroids_list)?;
    
    // Add cluster assignments
    let clusters_list = PyList::empty(py);
    for cluster in clusters {
        clusters_list.append(cluster)?;
    }
    result.set_item("clusters", clusters_list)?;
    
    // Add iterations
    result.set_item("iterations", iterations)?;
    
    Ok(result.into())
}

/// Linear regression
#[pyfunction]
fn linear_regression(py: Python, x: &PyList, y: &PyList) -> PyResult<Py<PyDict>> {
    if x.len() != y.len() {
        return Err(PyroidError::InputError(format!(
            "X and y must have the same length. Got {} and {}",
            x.len(), y.len()
        )).into());
    }
    
    if x.len() < 2 {
        return Err(PyroidError::InputError(
            "Need at least 2 data points for linear regression".to_string()
        ).into());
    }
    
    // Convert to Vec<f64>
    let mut x_vec = Vec::with_capacity(x.len());
    let mut y_vec = Vec::with_capacity(y.len());
    
    for i in 0..x.len() {
        x_vec.push(x.get_item(i)?.extract::<f64>()?);
        y_vec.push(y.get_item(i)?.extract::<f64>()?);
    }
    
    // Calculate means
    let n = x_vec.len() as f64;
    let mean_x = x_vec.iter().sum::<f64>() / n;
    let mean_y = y_vec.iter().sum::<f64>() / n;
    
    // Calculate slope and intercept
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    
    for i in 0..x_vec.len() {
        let x_diff = x_vec[i] - mean_x;
        let y_diff = y_vec[i] - mean_y;
        numerator += x_diff * y_diff;
        denominator += x_diff * x_diff;
    }
    
    let slope = if denominator != 0.0 { numerator / denominator } else { 0.0 };
    let intercept = mean_y - slope * mean_x;
    
    // Calculate R-squared
    let mut ss_total = 0.0;
    let mut ss_residual = 0.0;
    
    for i in 0..x_vec.len() {
        let y_pred = slope * x_vec[i] + intercept;
        ss_total += (y_vec[i] - mean_y).powi(2);
        ss_residual += (y_vec[i] - y_pred).powi(2);
    }
    
    let r_squared = if ss_total != 0.0 { 1.0 - ss_residual / ss_total } else { 0.0 };
    
    // Create result dictionary
    let result = PyDict::new(py);
    result.set_item("slope", slope)?;
    result.set_item("intercept", intercept)?;
    result.set_item("r_squared", r_squared)?;
    
    Ok(result.into())
}

/// Simple random number generator (0.0 to 1.0)
fn rand() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    (now % 10000) as f64 / 10000.0
}

/// Calculate Euclidean distance between two points
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut sum_sq = 0.0;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum_sq += diff * diff;
    }
    sum_sq.sqrt()
}

/// Normalize a vector of values
#[pyfunction]
fn normalize(py: Python, values: &PyList, method: Option<&str>) -> PyResult<Py<PyList>> {
    let method = method.unwrap_or("minmax");
    
    // Convert to Vec<f64>
    let mut data = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        data.push(values.get_item(i)?.extract::<f64>()?);
    }
    
    // Normalize
    let normalized = match method {
        "minmax" => {
            let min = data.iter().fold(f64::MAX, |a, &b| a.min(b));
            let max = data.iter().fold(f64::MIN, |a, &b| a.max(b));
            let range = max - min;
            
            if range == 0.0 {
                vec![0.5; data.len()]
            } else {
                data.iter().map(|&x| (x - min) / range).collect()
            }
        },
        "zscore" => {
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
            let std_dev = variance.sqrt();
            
            if std_dev == 0.0 {
                vec![0.0; data.len()]
            } else {
                data.iter().map(|&x| (x - mean) / std_dev).collect()
            }
        },
        _ => {
            return Err(PyroidError::InputError(format!(
                "Unknown normalization method: {}. Supported methods: minmax, zscore",
                method
            )).into());
        }
    };
    
    // Convert back to Python list
    let result = PyList::empty(py);
    for value in normalized {
        result.append(value)?;
    }
    
    Ok(result.into())
}

/// Calculate distance matrix between points
#[pyfunction]
fn distance_matrix(py: Python, points: &PyList, metric: Option<&str>) -> PyResult<Py<PyList>> {
    let metric = metric.unwrap_or("euclidean");
    
    // Convert to Vec<Vec<f64>>
    let mut data = Vec::with_capacity(points.len());
    let mut dimensions = 0;
    
    for i in 0..points.len() {
        let point = points.get_item(i)?;
        let point_list = point.downcast::<PyList>()?;
        
        if i == 0 {
            dimensions = point_list.len();
        } else if point_list.len() != dimensions {
            return Err(PyroidError::InputError(format!(
                "All points must have the same dimensions. Expected {}, got {}",
                dimensions, point_list.len()
            )).into());
        }
        
        let mut point_vec = Vec::with_capacity(dimensions);
        for j in 0..dimensions {
            let value = point_list.get_item(j)?.extract::<f64>()?;
            point_vec.push(value);
        }
        
        data.push(point_vec);
    }
    
    // Calculate distance matrix
    let n = data.len();
    let result_list = PyList::empty(py);
    
    for i in 0..n {
        let row_list = PyList::empty(py);
        
        for j in 0..n {
            let distance = match metric {
                "euclidean" => euclidean_distance(&data[i], &data[j]),
                "manhattan" => {
                    let mut sum = 0.0;
                    for k in 0..dimensions {
                        sum += (data[i][k] - data[j][k]).abs();
                    }
                    sum
                },
                _ => {
                    return Err(PyroidError::InputError(format!(
                        "Unknown distance metric: {}. Supported metrics: euclidean, manhattan",
                        metric
                    )).into());
                }
            };
            
            row_list.append(distance)?;
        }
        
        result_list.append(row_list)?;
    }
    
    Ok(result_list.into())
}

/// Register the basic module
pub fn register(py: Python, module: &PyModule) -> PyResult<()> {
    let basic_module = PyModule::new(py, "basic")?;
    
    basic_module.add_function(wrap_pyfunction!(kmeans, basic_module)?)?;
    basic_module.add_function(wrap_pyfunction!(linear_regression, basic_module)?)?;
    basic_module.add_function(wrap_pyfunction!(normalize, basic_module)?)?;
    basic_module.add_function(wrap_pyfunction!(distance_matrix, basic_module)?)?;
    
    // Add the basic module to the parent module
    module.add_submodule(basic_module)?;
    
    Ok(())
}