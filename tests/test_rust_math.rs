//! Integration tests for the Rust math module

mod common;
use common::math::{Vector, Matrix, stats};

#[test]
fn test_vector_operations() {
    // Create vectors
    let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
    let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
    
    // Test addition
    let v3 = v1.add(&v2);
    assert_eq!(v3.data(), &[5.0, 7.0, 9.0]);
    
    // Test subtraction
    let v4 = v2.sub(&v1);
    assert_eq!(v4.data(), &[3.0, 3.0, 3.0]);
    
    // Test scalar multiplication
    let v5 = v1.mul(2.0);
    assert_eq!(v5.data(), &[2.0, 4.0, 6.0]);
    
    // Test dot product
    let dot = v1.dot(&v2);
    assert_eq!(dot, 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0);
    
    // Test norm
    let norm = v1.norm();
    assert!(f64::abs(norm - f64::sqrt(1.0*1.0 + 2.0*2.0 + 3.0*3.0)) < 1e-10);
}

#[test]
fn test_matrix_operations() {
    // Create matrices
    let m1 = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let m2 = Matrix::new(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
    
    // Test addition
    let m3 = m1.add(&m2);
    assert_eq!(m3.data(), &[vec![6.0, 8.0], vec![10.0, 12.0]]);
    
    // Test subtraction
    let m4 = m2.sub(&m1);
    assert_eq!(m4.data(), &[vec![4.0, 4.0], vec![4.0, 4.0]]);
    
    // Test scalar multiplication
    let m5 = m1.mul_scalar(2.0);
    assert_eq!(m5.data(), &[vec![2.0, 4.0], vec![6.0, 8.0]]);
    
    // Test matrix multiplication
    let m6 = m1.mul_matrix(&m2);
    assert_eq!(m6.data(), &[vec![19.0, 22.0], vec![43.0, 50.0]]);
    
    // Test transpose
    let m7 = m1.transpose();
    assert_eq!(m7.data(), &[vec![1.0, 3.0], vec![2.0, 4.0]]);
}

#[test]
fn test_stats_functions() {
    // Create test data
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    
    // Test sum
    let sum_result = stats::sum(&data);
    assert_eq!(sum_result, 15.0);
    
    // Test mean
    let mean_result = stats::mean(&data);
    assert_eq!(mean_result, 3.0);
    
    // Test median
    let median_result = stats::median(&data);
    assert_eq!(median_result, 3.0);
    
    // Test standard deviation
    let std_result = stats::std(&data);
    assert!(f64::abs(std_result - f64::sqrt(2.0)) < 1e-10);
    
    // Test variance
    let var_result = stats::variance(&data);
    assert!(f64::abs(var_result - 2.0) < 1e-10);
    
    // Test with empty data
    let empty_data: Vec<f64> = vec![];
    assert_eq!(stats::sum(&empty_data), 0.0);
    assert_eq!(stats::mean(&empty_data), 0.0);
    assert_eq!(stats::median(&empty_data), 0.0);
    assert_eq!(stats::std(&empty_data), 0.0);
    assert_eq!(stats::variance(&empty_data), 0.0);
    
    // Test with single value
    let single_data = vec![5.0];
    assert_eq!(stats::sum(&single_data), 5.0);
    assert_eq!(stats::mean(&single_data), 5.0);
    assert_eq!(stats::median(&single_data), 5.0);
    assert_eq!(stats::std(&single_data), 0.0);
    assert_eq!(stats::variance(&single_data), 0.0);
}

#[test]
fn test_correlation() {
    // Create test data with perfect positive correlation
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    
    let corr_result = stats::correlation(&x, &y);
    assert!(f64::abs(corr_result - 1.0) < 1e-10);
    
    // Test with perfect negative correlation
    let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
    let corr_neg_result = stats::correlation(&x, &y_neg);
    assert!(f64::abs(corr_neg_result - (-1.0)) < 1e-10);
    
    // Test with no correlation
    let y_no_corr = vec![5.0, 5.0, 5.0, 5.0, 5.0];
    let corr_no_result = stats::correlation(&x, &y_no_corr);
    assert!(f64::abs(corr_no_result - 0.0) < 1e-10);
    
    // Test with empty data
    let empty_data: Vec<f64> = vec![];
    assert_eq!(stats::correlation(&empty_data, &empty_data), 0.0);
    
    // Test with single value
    let single_data = vec![5.0];
    assert_eq!(stats::correlation(&single_data, &single_data), 0.0);
}