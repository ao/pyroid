//! Integration tests for the ML module

extern crate rand;
mod common;
use common::ml;

#[test]
fn test_kmeans() {
    // Create some test data with clear clusters
    let data = vec![
        vec![1.0, 1.0],
        vec![1.5, 2.0],
        vec![10.0, 10.0],
        vec![10.5, 9.5]
    ];
    
    // Run k-means with a fixed seed for reproducibility
    let result = ml::kmeans(&data, 2, 100, 42);
    
    // Check that we have the right number of centroids
    assert_eq!(result.centroids.len(), 2);
    
    // Check that we have the right number of clusters
    assert_eq!(result.clusters.len(), 2);
    
    // Check that each point is assigned to the correct cluster
    // We don't know which cluster is which, so we need to check both possibilities
    let cluster_a = &result.clusters[0];
    let cluster_b = &result.clusters[1];
    
    // Either cluster_a has the first two points and cluster_b has the last two,
    // or vice versa
    let correct_clustering = (
        (cluster_a.contains(&0) && cluster_a.contains(&1) &&
         cluster_b.contains(&2) && cluster_b.contains(&3))
        ||
        (cluster_b.contains(&0) && cluster_b.contains(&1) &&
         cluster_a.contains(&2) && cluster_a.contains(&3))
    );
    
    assert!(correct_clustering);
    
    // Test with empty data
    let empty_data: Vec<Vec<f64>> = vec![];
    let result = ml::kmeans(&empty_data, 2, 100, 42);
    assert_eq!(result.centroids.len(), 0);
    assert_eq!(result.clusters.len(), 0);
}

#[test]
fn test_linear_regression() {
    // Create some test data with a clear linear relationship
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];  // y = 2*x
    
    // Run linear regression
    let result = ml::linear_regression(&x, &y);
    
    // Check the coefficients
    assert!((result.coefficients[0] - 2.0).abs() < 1e-10);
    
    // Check the intercept
    assert!(result.intercept.abs() < 1e-10);
    
    // Check the R-squared
    assert!((result.r_squared - 1.0).abs() < 1e-10);
    
    // Test with empty data
    let empty_x: Vec<f64> = vec![];
    let empty_y: Vec<f64> = vec![];
    let result = ml::linear_regression(&empty_x, &empty_y);
    assert_eq!(result.coefficients.len(), 0);
    assert_eq!(result.intercept, 0.0);
    assert_eq!(result.r_squared, 0.0);
    
    // Test with mismatched data
    let x = vec![1.0, 2.0];
    let y = vec![1.0];
    let result = ml::linear_regression(&x, &y);
    assert_eq!(result.coefficients.len(), 0);
    assert_eq!(result.intercept, 0.0);
    assert_eq!(result.r_squared, 0.0);
}

#[test]
fn test_normalize() {
    // Create some test data
    let data = vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
        vec![5.0, 6.0]
    ];
    
    // Run min-max normalization
    let result = ml::normalize(&data, "min-max");
    
    // Check the result
    assert!((result[0][0] - 0.0).abs() < 1e-10);  // min value
    assert!((result[2][0] - 1.0).abs() < 1e-10);  // max value
    assert!((result[0][1] - 0.0).abs() < 1e-10);  // min value
    assert!((result[2][1] - 1.0).abs() < 1e-10);  // max value
    
    // Run z-score normalization
    let result = ml::normalize(&data, "z-score");
    
    // Check the result (mean should be 0, std should be 1)
    let mean_col0 = result.iter().map(|row| row[0]).sum::<f64>() / result.len() as f64;
    let mean_col1 = result.iter().map(|row| row[1]).sum::<f64>() / result.len() as f64;
    assert!(mean_col0.abs() < 1e-10);
    assert!(mean_col1.abs() < 1e-10);
    
    let std_col0 = (result.iter().map(|row| (row[0] - mean_col0).powi(2)).sum::<f64>() / result.len() as f64).sqrt();
    let std_col1 = (result.iter().map(|row| (row[1] - mean_col1).powi(2)).sum::<f64>() / result.len() as f64).sqrt();
    assert!((std_col0 - 1.0).abs() < 1e-10);
    assert!((std_col1 - 1.0).abs() < 1e-10);
    
    // Test with empty data
    let empty_data: Vec<Vec<f64>> = vec![];
    let result = ml::normalize(&empty_data, "min-max");
    assert_eq!(result.len(), 0);
}

#[test]
fn test_distance_matrix() {
    // Create some test data
    let data = vec![
        vec![1.0, 1.0],
        vec![4.0, 5.0],
        vec![10.0, 10.0]
    ];
    
    // Calculate Euclidean distance matrix
    let result = ml::distance_matrix(&data, "euclidean");
    
    // Check the result
    assert_eq!(result.len(), 3);  // 3x3 matrix
    assert_eq!(result[0].len(), 3);
    
    // Check some specific distances
    assert!(result[0][0].abs() < 1e-10);  // Distance to self is 0
    
    // Calculate expected Euclidean distance
    let expected_dist1 = ((4.0-1.0).powi(2) + (5.0-1.0).powi(2)).sqrt();
    assert!((result[0][1] - expected_dist1).abs() < 1e-10);
    
    let expected_dist2 = ((10.0-1.0).powi(2) + (10.0-1.0).powi(2)).sqrt();
    assert!((result[0][2] - expected_dist2).abs() < 1e-10);
    
    // Calculate Manhattan distance matrix
    let result = ml::distance_matrix(&data, "manhattan");
    
    // Check the result
    assert_eq!(result.len(), 3);  // 3x3 matrix
    assert_eq!(result[0].len(), 3);
    
    // Check some specific distances
    assert!(result[0][0].abs() < 1e-10);  // Distance to self is 0
    
    // Calculate expected Manhattan distance
    let expected_dist1 = (4.0-1.0).abs() + (5.0-1.0).abs();
    assert!((result[0][1] - expected_dist1).abs() < 1e-10);
    
    let expected_dist2 = (10.0-1.0).abs() + (10.0-1.0).abs();
    assert!((result[0][2] - expected_dist2).abs() < 1e-10);
    
    // Test with empty data
    let empty_data: Vec<Vec<f64>> = vec![];
    let result = ml::distance_matrix(&empty_data, "euclidean");
    assert_eq!(result.len(), 0);
}