//! Machine Learning operations
//!
//! This module provides high-performance machine learning operations.

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use pyo3::types::{PyDict, PyList, PyTuple};
use rayon::prelude::*;
use ndarray::{Array1, Array2, Axis, s};
use ndarray_stats::QuantileExt;
use std::collections::HashMap;

/// Calculate distance matrix in parallel
///
/// Args:
///     points: A list of points (each point is a list of coordinates)
///     metric: Distance metric to use (euclidean, manhattan, cosine)
///
/// Returns:
///     A 2D array of distances between points
#[pyfunction]
fn parallel_distance_matrix(py: Python, points: Vec<Vec<f64>>, metric: Option<String>) -> PyResult<PyObject> {
    let metric = metric.unwrap_or_else(|| "euclidean".to_string());
    
    // Convert points to ndarray
    let n_points = points.len();
    let n_dims = if n_points > 0 { points[0].len() } else { 0 };
    
    if n_points == 0 {
        return Err(PyValueError::new_err("Empty points list"));
    }
    
    // Check that all points have the same dimensionality
    for (i, point) in points.iter().enumerate() {
        if point.len() != n_dims {
            return Err(PyValueError::new_err(
                format!("Point at index {} has different dimensionality ({}) than expected ({})",
                    i, point.len(), n_dims)
            ));
        }
    }
    
    // Create a 2D array for the points
    let mut points_array = Array2::<f64>::zeros((n_points, n_dims));
    for (i, point) in points.iter().enumerate() {
        for (j, &value) in point.iter().enumerate() {
            points_array[[i, j]] = value;
        }
    }
    
    // Calculate the distance matrix in parallel
    let distance_matrix = match metric.as_str() {
        "euclidean" => {
            let mut result = Array2::<f64>::zeros((n_points, n_points));
            
            // Parallelize the computation
            result.axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(i, mut row)| {
                    for j in 0..n_points {
                        if i == j {
                            row[j] = 0.0;
                        } else {
                            let mut sum_sq = 0.0;
                            for k in 0..n_dims {
                                let diff = points_array[[i, k]] - points_array[[j, k]];
                                sum_sq += diff * diff;
                            }
                            row[j] = sum_sq.sqrt();
                        }
                    }
                });
                
            result
        },
        "manhattan" => {
            let mut result = Array2::<f64>::zeros((n_points, n_points));
            
            // Parallelize the computation
            result.axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(i, mut row)| {
                    for j in 0..n_points {
                        if i == j {
                            row[j] = 0.0;
                        } else {
                            let mut sum_abs = 0.0;
                            for k in 0..n_dims {
                                let diff = points_array[[i, k]] - points_array[[j, k]];
                                sum_abs += diff.abs();
                            }
                            row[j] = sum_abs;
                        }
                    }
                });
                
            result
        },
        "cosine" => {
            let mut result = Array2::<f64>::zeros((n_points, n_points));
            
            // Precompute norms
            let norms: Vec<f64> = (0..n_points)
                .into_par_iter()
                .map(|i| {
                    let mut norm_sq = 0.0;
                    for k in 0..n_dims {
                        norm_sq += points_array[[i, k]] * points_array[[i, k]];
                    }
                    norm_sq.sqrt()
                })
                .collect();
            
            // Parallelize the computation
            result.axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(i, mut row)| {
                    for j in 0..n_points {
                        if i == j {
                            row[j] = 0.0;
                        } else {
                            let mut dot_product = 0.0;
                            for k in 0..n_dims {
                                dot_product += points_array[[i, k]] * points_array[[j, k]];
                            }
                            
                            let norm_i = norms[i];
                            let norm_j = norms[j];
                            
                            if norm_i > 0.0 && norm_j > 0.0 {
                                row[j] = 1.0 - (dot_product / (norm_i * norm_j));
                            } else {
                                row[j] = 1.0; // Maximum distance for zero vectors
                            }
                        }
                    }
                });
                
            result
        },
        _ => return Err(PyValueError::new_err(format!("Unsupported distance metric: {}", metric))),
    };
    
    // Convert the distance matrix to a Python list of lists
    let py_list = PyList::empty(py);
    
    for i in 0..n_points {
        let row = PyList::empty(py);
        for j in 0..n_points {
            row.append(distance_matrix[[i, j]])?;
        }
        py_list.append(row)?;
    }
    
    Ok(py_list.into())
}

/// Scale features in parallel
///
/// Args:
///     data: A 2D array of data (rows are samples, columns are features)
///     method: Scaling method (standard, minmax, robust)
///     with_mean: Whether to center the data before scaling (default: true)
///     with_std: Whether to scale to unit variance (default: true)
///
/// Returns:
///     A 2D array of scaled data
#[pyfunction]
fn parallel_feature_scaling(
    py: Python,
    data: Vec<Vec<f64>>,
    method: Option<String>,
    with_mean: Option<bool>,
    with_std: Option<bool>
) -> PyResult<PyObject> {
    let method = method.unwrap_or_else(|| "standard".to_string());
    let with_mean = with_mean.unwrap_or(true);
    let with_std = with_std.unwrap_or(true);
    
    // Convert data to ndarray
    let n_samples = data.len();
    let n_features = if n_samples > 0 { data[0].len() } else { 0 };
    
    if n_samples == 0 {
        return Err(PyValueError::new_err("Empty data"));
    }
    
    // Check that all samples have the same number of features
    for (i, sample) in data.iter().enumerate() {
        if sample.len() != n_features {
            return Err(PyValueError::new_err(
                format!("Sample at index {} has different number of features ({}) than expected ({})",
                    i, sample.len(), n_features)
            ));
        }
    }
    
    // Create a 2D array for the data
    let mut data_array = Array2::<f64>::zeros((n_samples, n_features));
    for (i, sample) in data.iter().enumerate() {
        for (j, &value) in sample.iter().enumerate() {
            data_array[[i, j]] = value;
        }
    }
    
    // Scale the data
    let scaled_data = match method.as_str() {
        "standard" => {
            // Calculate mean and std for each feature
            let mut means = Vec::with_capacity(n_features);
            let mut stds = Vec::with_capacity(n_features);
            
            for j in 0..n_features {
                let column = data_array.slice(s![.., j]);
                
                // Calculate mean
                let mean = if with_mean {
                    column.sum() / n_samples as f64
                } else {
                    0.0
                };
                
                // Calculate std
                let std = if with_std {
                    let mut sum_sq_diff = 0.0;
                    for &value in column.iter() {
                        let diff = value - mean;
                        sum_sq_diff += diff * diff;
                    }
                    (sum_sq_diff / n_samples as f64).sqrt()
                } else {
                    1.0
                };
                
                means.push(mean);
                stds.push(if std > 0.0 { std } else { 1.0 });
            }
            
            // Scale the data in parallel
            let mut result = Array2::<f64>::zeros((n_samples, n_features));
            
            result.axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(i, mut row)| {
                    for j in 0..n_features {
                        row[j] = (data_array[[i, j]] - means[j]) / stds[j];
                    }
                });
                
            result
        },
        "minmax" => {
            // Calculate min and max for each feature
            let mut mins = Vec::with_capacity(n_features);
            let mut maxs = Vec::with_capacity(n_features);
            
            for j in 0..n_features {
                let column = data_array.slice(s![.., j]);
                
                let min = *column.min().unwrap();
                let max = *column.max().unwrap();
                
                mins.push(min);
                maxs.push(max);
            }
            
            // Scale the data in parallel
            let mut result = Array2::<f64>::zeros((n_samples, n_features));
            
            result.axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(i, mut row)| {
                    for j in 0..n_features {
                        let min = mins[j];
                        let max = maxs[j];
                        
                        if max > min {
                            row[j] = (data_array[[i, j]] - min) / (max - min);
                        } else {
                            row[j] = 0.0;
                        }
                    }
                });
                
            result
        },
        "robust" => {
            // Calculate median and IQR for each feature
            let mut medians = Vec::with_capacity(n_features);
            let mut iqrs = Vec::with_capacity(n_features);
            
            for j in 0..n_features {
                let mut column_vec: Vec<f64> = data_array.slice(s![.., j]).to_vec();
                column_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
                
                let median = if n_samples % 2 == 0 {
                    (column_vec[n_samples / 2 - 1] + column_vec[n_samples / 2]) / 2.0
                } else {
                    column_vec[n_samples / 2]
                };
                
                let q1_idx = n_samples / 4;
                let q3_idx = n_samples * 3 / 4;
                
                let q1 = column_vec[q1_idx];
                let q3 = column_vec[q3_idx];
                
                let iqr = q3 - q1;
                
                medians.push(median);
                iqrs.push(if iqr > 0.0 { iqr } else { 1.0 });
            }
            
            // Scale the data in parallel
            let mut result = Array2::<f64>::zeros((n_samples, n_features));
            
            result.axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(i, mut row)| {
                    for j in 0..n_features {
                        row[j] = (data_array[[i, j]] - medians[j]) / iqrs[j];
                    }
                });
                
            result
        },
        _ => return Err(PyValueError::new_err(format!("Unsupported scaling method: {}", method))),
    };
    
    // Convert the scaled data to a Python list of lists
    let py_list = PyList::empty(py);
    
    for i in 0..n_samples {
        let row = PyList::empty(py);
        for j in 0..n_features {
            row.append(scaled_data[[i, j]])?;
        }
        py_list.append(row)?;
    }
    
    Ok(py_list.into())
}

/// Perform cross-validation in parallel
///
/// Args:
///     X: A 2D array of features (rows are samples, columns are features)
///     y: A 1D array of target values
///     cv: Number of folds (default: 5)
///     model_func: A Python function that takes (X_train, y_train, X_test) and returns predictions
///     scoring_func: A Python function that takes (y_true, y_pred) and returns a score
///
/// Returns:
///     A list of scores for each fold
#[pyfunction]
fn parallel_cross_validation(
    py: Python,
    X: Vec<Vec<f64>>,
    y: Vec<f64>,
    cv: Option<usize>,
    model_func: PyObject,
    scoring_func: PyObject
) -> PyResult<PyObject> {
    let cv = cv.unwrap_or(5);
    
    // Validate inputs
    let n_samples = X.len();
    
    if n_samples == 0 {
        return Err(PyValueError::new_err("Empty features array"));
    }
    
    if y.len() != n_samples {
        return Err(PyValueError::new_err(
            format!("Number of samples in X ({}) does not match number of samples in y ({})",
                n_samples, y.len())
        ));
    }
    
    if cv < 2 {
        return Err(PyValueError::new_err("cv must be at least 2"));
    }
    
    if n_samples < cv {
        return Err(PyValueError::new_err(
            format!("Number of samples ({}) is less than number of folds ({})",
                n_samples, cv)
        ));
    }
    
    // Create folds
    let fold_size = n_samples / cv;
    let mut folds = Vec::with_capacity(cv);
    
    for i in 0..cv {
        let start = i * fold_size;
        let end = if i == cv - 1 { n_samples } else { (i + 1) * fold_size };
        folds.push((start, end));
    }
    
    // Perform cross-validation in parallel
    let scores: Result<Vec<f64>, PyErr> = folds.par_iter()
        .map(|&(test_start, test_end)| {
            Python::with_gil(|py| {
                // Create train and test sets
                let mut X_train = Vec::new();
                let mut y_train = Vec::new();
                let mut X_test = Vec::new();
                let mut y_test = Vec::new();
                
                for i in 0..n_samples {
                    if i >= test_start && i < test_end {
                        X_test.push(X[i].clone());
                        y_test.push(y[i]);
                    } else {
                        X_train.push(X[i].clone());
                        y_train.push(y[i]);
                    }
                }
                
                // Convert to Python lists
                let py_X_train = PyList::empty(py);
                for row in &X_train {
                    let py_row = PyList::new(py, row);
                    py_X_train.append(py_row)?;
                }
                
                let py_y_train = PyList::new(py, &y_train);
                
                let py_X_test = PyList::empty(py);
                for row in &X_test {
                    let py_row = PyList::new(py, row);
                    py_X_test.append(py_row)?;
                }
                
                let py_y_test = PyList::new(py, &y_test);
                
                // Call the model function to get predictions
                let py_y_pred = model_func.call1(py, (py_X_train, py_y_train, py_X_test))?;
                
                // Call the scoring function to get the score
                let score = scoring_func.call1(py, (py_y_test, py_y_pred))?
                    .extract::<f64>(py)?;
                
                Ok(score)
            })
        })
        .collect();
    
    // Convert scores to a Python list
    let py_scores = PyList::new(py, &scores?);
    
    Ok(py_scores.into())
}

/// Register the machine learning operations module
pub fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parallel_distance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_feature_scaling, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_cross_validation, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_distance_matrix_euclidean() {
        Python::with_gil(|py| {
            let points = vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
            ];
            
            let result = parallel_distance_matrix(py, points, Some("euclidean".to_string())).unwrap();
            let matrix: Vec<Vec<f64>> = result.extract(py).unwrap();
            
            assert_eq!(matrix.len(), 3);
            assert!((matrix[0][1] - 1.0).abs() < 1e-10);
            assert!((matrix[0][2] - 1.0).abs() < 1e-10);
            assert!((matrix[1][2] - 2.0_f64.sqrt()).abs() < 1e-10);
        });
    }
    
    #[test]
    fn test_feature_scaling_standard() {
        Python::with_gil(|py| {
            let data = vec![
                vec![0.0, 0.0],
                vec![1.0, 2.0],
                vec![2.0, 4.0],
            ];
            
            let result = parallel_feature_scaling(py, data, Some("standard".to_string()), Some(true), Some(true)).unwrap();
            let scaled: Vec<Vec<f64>> = result.extract(py).unwrap();
            
            assert_eq!(scaled.len(), 3);
            assert!((scaled[0][0] + 1.0).abs() < 1e-10); // (0 - 1) / 1 = -1
            assert!((scaled[1][0] - 0.0).abs() < 1e-10); // (1 - 1) / 1 = 0
            assert!((scaled[2][0] - 1.0).abs() < 1e-10); // (2 - 1) / 1 = 1
        });
    }
}