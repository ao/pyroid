//! Mathematical operations
//!
//! This module provides high-performance mathematical operations.
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyList;
use rayon::prelude::*;
use crate::utils::split_into_chunks;

/// Calculate the sum of a list of numbers in parallel
///
/// Args:
///     numbers: A list of numbers to sum
///
/// Returns:
///     The sum of the numbers
#[pyfunction]
fn parallel_sum(numbers: Vec<f64>) -> f64 {
    numbers.par_iter().sum()
}

/// Calculate the product of a list of numbers in parallel
///
/// Args:
///     numbers: A list of numbers to multiply
///
/// Returns:
///     The product of the numbers
#[pyfunction]
fn parallel_product(numbers: Vec<f64>) -> f64 {
    numbers.par_iter().product()
}

/// Calculate the mean of a list of numbers in parallel
///
/// Args:
///     numbers: A list of numbers
///
/// Returns:
///     The mean (average) of the numbers
#[pyfunction]
fn parallel_mean(numbers: Vec<f64>) -> PyResult<f64> {
    if numbers.is_empty() {
        return Err(PyValueError::new_err("Cannot calculate mean of empty list"));
    }
    
    let sum: f64 = numbers.par_iter().sum();
    Ok(sum / numbers.len() as f64)
}

/// Calculate the standard deviation of a list of numbers in parallel
///
/// Args:
///     numbers: A list of numbers
///     ddof: Delta degrees of freedom (default: 0)
///
/// Returns:
///     The standard deviation of the numbers
#[pyfunction]
fn parallel_std(numbers: Vec<f64>, ddof: Option<usize>) -> PyResult<f64> {
    let ddof = ddof.unwrap_or(0);
    
    if numbers.is_empty() {
        return Err(PyValueError::new_err("Cannot calculate standard deviation of empty list"));
    }
    
    if numbers.len() <= ddof {
        return Err(PyValueError::new_err(
            format!("Sample size {} is less than or equal to ddof {}", numbers.len(), ddof)
        ));
    }
    
    let mean = parallel_mean(numbers.clone())?;
    
    let variance: f64 = numbers.par_iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / (numbers.len() - ddof) as f64;
    
    Ok(variance.sqrt())
}

/// Apply a function to each element of a list in parallel
///
/// Args:
///     numbers: A list of numbers
///     operation: The operation to apply ('sqrt', 'log', 'exp', 'abs', 'sin', 'cos', 'tan')
///
/// Returns:
///     A list with the operation applied to each element
#[pyfunction]
fn parallel_apply(numbers: Vec<f64>, operation: &str) -> PyResult<Vec<f64>> {
    let op_func = match operation {
        "sqrt" => |x: f64| x.sqrt(),
        "log" => |x: f64| x.ln(),
        "exp" => |x: f64| x.exp(),
        "abs" => |x: f64| x.abs(),
        "sin" => |x: f64| x.sin(),
        "cos" => |x: f64| x.cos(),
        "tan" => |x: f64| x.tan(),
        _ => return Err(PyValueError::new_err(format!("Unknown operation: {}", operation))),
    };
    
    Ok(numbers.par_iter().map(|&x| op_func(x)).collect())
}

/// Perform matrix multiplication
///
/// This is a placeholder for a more sophisticated implementation that would
/// use ndarray and handle NumPy arrays properly.
///
/// Args:
///     a: First matrix (as nested lists)
///     b: Second matrix (as nested lists)
///
/// Returns:
///     The result of matrix multiplication
#[pyfunction]
fn matrix_multiply(py: Python, a: &PyAny, b: &PyAny) -> PyResult<PyObject> {
    // This is a simplified implementation that assumes a and b are lists of lists
    // A proper implementation would use ndarray and handle NumPy arrays
    
    // Extract dimensions
    let a_rows = a.len()?;
    if a_rows == 0 {
        return Err(PyValueError::new_err("First matrix is empty"));
    }
    
    let a_first_row = a.get_item(0)?;
    let a_cols = a_first_row.len()?;
    
    let b_rows = b.len()?;
    if b_rows == 0 {
        return Err(PyValueError::new_err("Second matrix is empty"));
    }
    
    let b_first_row = b.get_item(0)?;
    let b_cols = b_first_row.len()?;
    
    // Check dimensions
    if a_cols != b_rows {
        return Err(PyValueError::new_err(
            format!("Matrix dimensions don't match: {}x{} and {}x{}", a_rows, a_cols, b_rows, b_cols)
        ));
    }
    
    // Convert to Rust data structure
    let mut a_data: Vec<Vec<f64>> = Vec::with_capacity(a_rows);
    for i in 0..a_rows {
        let row = a.get_item(i)?;
        let mut a_row: Vec<f64> = Vec::with_capacity(a_cols);
        for j in 0..a_cols {
            let val = row.get_item(j)?.extract::<f64>()?;
            a_row.push(val);
        }
        a_data.push(a_row);
    }
    
    let mut b_data: Vec<Vec<f64>> = Vec::with_capacity(b_rows);
    for i in 0..b_rows {
        let row = b.get_item(i)?;
        let mut b_row: Vec<f64> = Vec::with_capacity(b_cols);
        for j in 0..b_cols {
            let val = row.get_item(j)?.extract::<f64>()?;
            b_row.push(val);
        }
        b_data.push(b_row);
    }
    
    // Perform matrix multiplication
    let mut result: Vec<Vec<f64>> = vec![vec![0.0; b_cols]; a_rows];
    
    // Parallelize the outer loop
    result.par_iter_mut().enumerate().for_each(|(i, row)| {
        for j in 0..b_cols {
            let mut sum = 0.0;
            for k in 0..a_cols {
                sum += a_data[i][k] * b_data[k][j];
            }
            row[j] = sum;
        }
    });
    
    // Convert back to Python
    let py_result = PyList::empty(py);
    for row in result {
        let py_row = PyList::empty(py);
        for val in row {
            py_row.append(val)?;
        }
        py_result.append(py_row)?;
    }
    
    Ok(py_result.into())
}

/// Register the math operations module
pub fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parallel_sum, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_product, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_mean, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_std, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_apply, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_multiply, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parallel_sum() {
        let numbers = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = parallel_sum(numbers);
        assert_eq!(result, 15.0);
    }
    
    #[test]
    fn test_parallel_product() {
        let numbers = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = parallel_product(numbers);
        assert_eq!(result, 120.0);
    }
    
    #[test]
    fn test_parallel_mean() {
        let numbers = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = parallel_mean(numbers).unwrap();
        assert_eq!(result, 3.0);
    }
    
    #[test]
    fn test_parallel_std() {
        let numbers = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = parallel_std(numbers, Some(1)).unwrap();
        assert!((result - 1.5811388300841898).abs() < 1e-10);
    }
    
    #[test]
    fn test_parallel_apply() {
        let numbers = vec![1.0, 4.0, 9.0, 16.0];
        let result = parallel_apply(numbers, "sqrt").unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }
}