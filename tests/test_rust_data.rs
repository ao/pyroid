//! Integration tests for the Rust data module

mod common;
use common::data;

#[test]
fn test_filter() {
    // Create test data
    let data = vec![1, 2, 3, 4, 5];
    
    // Filter even numbers
    let result = data::filter(&data, |&x| x % 2 == 0);
    assert_eq!(result, vec![2, 4]);
    
    // Filter with empty result
    let result = data::filter(&data, |&x| x > 10);
    assert_eq!(result, Vec::<i32>::new());
    
    // Filter with empty input
    let empty_data: Vec<i32> = vec![];
    let result = data::filter(&empty_data, |&x| x % 2 == 0);
    assert_eq!(result, Vec::<i32>::new());
}

#[test]
fn test_map() {
    // Create test data
    let data = vec![1, 2, 3, 4, 5];
    
    // Map to double each value
    let result = data::map(&data, |&x| x * 2);
    assert_eq!(result, vec![2, 4, 6, 8, 10]);
    
    // Map with empty input
    let empty_data: Vec<i32> = vec![];
    let result = data::map(&empty_data, |&x| x * 2);
    assert_eq!(result, Vec::<i32>::new());
}

#[test]
fn test_reduce() {
    // Create test data
    let data = vec![1, 2, 3, 4, 5];
    
    // Reduce to sum
    let result = data::reduce(&data, |acc, &x| acc + x, 0);
    assert_eq!(result, 15);
    
    // Reduce with initial value
    let result = data::reduce(&data, |acc, &x| acc + x, 10);
    assert_eq!(result, 25);
    
    // Reduce with empty input
    let empty_data: Vec<i32> = vec![];
    let result = data::reduce(&empty_data, |acc, &x| acc + x, 0);
    assert_eq!(result, 0);
    
    // Reduce with single value
    let single_data = vec![5];
    let result = data::reduce(&single_data, |acc, &x| acc + x, 0);
    assert_eq!(result, 5);
}

#[test]
fn test_sort() {
    // Create test data
    let mut data = vec![5, 3, 1, 4, 2];
    
    // Sort in ascending order
    data::sort(&mut data, false);
    assert_eq!(data, vec![1, 2, 3, 4, 5]);
    
    // Sort in descending order
    data::sort(&mut data, true);
    assert_eq!(data, vec![5, 4, 3, 2, 1]);
    
    // Sort with empty input
    let mut empty_data: Vec<i32> = vec![];
    data::sort(&mut empty_data, false);
    assert_eq!(empty_data, Vec::<i32>::new());
    
    // Sort with single value
    let mut single_data = vec![5];
    data::sort(&mut single_data, false);
    assert_eq!(single_data, vec![5]);
}

#[test]
fn test_dataframe() {
    use std::collections::HashMap;
    
    // Create a DataFrame
    let mut data = HashMap::new();
    data.insert("a".to_string(), vec!["1".to_string(), "2".to_string(), "3".to_string()]);
    data.insert("b".to_string(), vec!["4".to_string(), "5".to_string(), "6".to_string()]);
    
    let df = data::DataFrame::new(data);
    
    // Test get_column
    let col_a = df.get_column("a").unwrap();
    assert_eq!(col_a, &vec!["1".to_string(), "2".to_string(), "3".to_string()]);
    
    let col_b = df.get_column("b").unwrap();
    assert_eq!(col_b, &vec!["4".to_string(), "5".to_string(), "6".to_string()]);
    
    // Test get_column with nonexistent column
    assert!(df.get_column("c").is_none());
    
    // Test num_rows
    assert_eq!(df.num_rows(), 3);
    
    // Test column_names
    let names = df.column_names();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"a".to_string()));
    assert!(names.contains(&"b".to_string()));
}

#[test]
fn test_dataframe_apply() {
    use std::collections::HashMap;
    
    // Create a DataFrame
    let mut data = HashMap::new();
    data.insert("a".to_string(), vec!["1".to_string(), "2".to_string(), "3".to_string()]);
    data.insert("b".to_string(), vec!["4".to_string(), "5".to_string(), "6".to_string()]);
    
    let df = data::DataFrame::new(data);
    
    // Apply a function to append "x" to each value
    let result = df.apply(|col| data::map(col, |s| format!("{}x", s)));
    
    // Check the result
    let col_a = result.get_column("a").unwrap();
    assert_eq!(col_a, &vec!["1x".to_string(), "2x".to_string(), "3x".to_string()]);
    
    let col_b = result.get_column("b").unwrap();
    assert_eq!(col_b, &vec!["4x".to_string(), "5x".to_string(), "6x".to_string()]);
}

#[test]
fn test_dataframe_groupby() {
    use std::collections::HashMap;
    
    // Create a DataFrame
    let mut data = HashMap::new();
    data.insert("category".to_string(), vec!["A".to_string(), "B".to_string(), "A".to_string(), "B".to_string(), "C".to_string()]);
    data.insert("value".to_string(), vec!["1".to_string(), "2".to_string(), "3".to_string(), "4".to_string(), "5".to_string()]);
    
    let df = data::DataFrame::new(data);
    
    // Group by category and calculate mean of value
    let mut aggs = HashMap::new();
    aggs.insert("value".to_string(), "mean".to_string());
    
    let result = df.groupby("category", aggs);
    
    // Check the result
    assert_eq!(result.num_rows(), 3); // 3 unique categories
    
    // Check that the categories are present
    let categories = result.get_column("category").unwrap();
    assert!(categories.contains(&"A".to_string()));
    assert!(categories.contains(&"B".to_string()));
    assert!(categories.contains(&"C".to_string()));
    
    // Check the aggregated values
    // This is more complex because the order is not guaranteed
    // We'll need to find each category and check its value
    let values = result.get_column("value_mean").unwrap();
    
    for i in 0..result.num_rows() {
        let category = &categories[i];
        let value = &values[i];
        
        if category == "A" {
            assert_eq!(value, "2"); // (1 + 3) / 2
        } else if category == "B" {
            assert_eq!(value, "3"); // (2 + 4) / 2
        } else if category == "C" {
            assert_eq!(value, "5"); // 5 / 1
        }
    }
}