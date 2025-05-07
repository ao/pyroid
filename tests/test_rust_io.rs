//! Integration tests for the Rust IO module

mod common;
use common::io;
use std::fs;
use std::path::Path;

#[test]
fn test_file_operations() {
    // Create a temporary file path
    let test_file = "test_io_file.txt";
    
    // Clean up any existing file
    if Path::new(test_file).exists() {
        fs::remove_file(test_file).unwrap();
    }
    
    // Test file_exists (should be false)
    assert!(!io::file_exists(test_file));
    
    // Test write_file
    let content = "Hello, world!";
    io::write_file(test_file, content).unwrap();
    
    // Test file_exists (should be true now)
    assert!(io::file_exists(test_file));
    
    // Test read_file
    let read_content = io::read_file(test_file).unwrap();
    assert_eq!(read_content, content);
    
    // Test append_file
    let append_content = "\nAppended content";
    io::append_file(test_file, append_content).unwrap();
    
    // Test read_file after append
    let read_content = io::read_file(test_file).unwrap();
    assert_eq!(read_content, format!("{}{}", content, append_content));
    
    // Test delete_file
    io::delete_file(test_file).unwrap();
    
    // Test file_exists after delete (should be false)
    assert!(!io::file_exists(test_file));
}

#[test]
fn test_write_file_creates_directories() {
    // Create a temporary file path with directories
    let test_dir = "test_io_dir";
    let test_subdir = format!("{}/subdir", test_dir);
    let test_file = format!("{}/test_file.txt", test_subdir);
    
    // Clean up any existing directory
    if Path::new(test_dir).exists() {
        fs::remove_dir_all(test_dir).unwrap();
    }
    
    // Test write_file with directories
    let content = "Hello, world!";
    io::write_file(&test_file, content).unwrap();
    
    // Test file_exists (should be true)
    assert!(io::file_exists(&test_file));
    
    // Test read_file
    let read_content = io::read_file(&test_file).unwrap();
    assert_eq!(read_content, content);
    
    // Clean up
    fs::remove_dir_all(test_dir).unwrap();
}

#[test]
fn test_error_handling() {
    // Test read_file with nonexistent file
    let result = io::read_file("nonexistent_file.txt");
    assert!(result.is_err());
    
    // Test delete_file with nonexistent file
    let result = io::delete_file("nonexistent_file.txt");
    assert!(result.is_err());
}