//! Integration tests for the Rust core module

mod common;
use common::core;

#[test]
fn test_config() {
    // Create a new Config
    let mut config = core::Config::new();
    
    // Set some options
    config.set("parallel", true);
    config.set("chunk_size", 1000);
    config.set("threshold", 0.5);
    config.set("name", "test");
    
    // Get the options
    assert_eq!(config.get::<bool>("parallel").unwrap(), true);
    assert_eq!(config.get::<i32>("chunk_size").unwrap(), 1000);
    assert_eq!(config.get::<f64>("threshold").unwrap(), 0.5);
    assert_eq!(config.get::<String>("name").unwrap(), "test");
    
    // Get a nonexistent option
    assert!(config.get::<bool>("nonexistent").is_none());
    
    // Update an existing option
    config.set("parallel", false);
    assert_eq!(config.get::<bool>("parallel").unwrap(), false);
}

#[test]
fn test_shared_data() {
    // Create a SharedData object with a vector
    let data = vec![1, 2, 3, 4, 5];
    let shared = core::SharedData::new(data.clone());
    
    // Get the data
    let retrieved: Vec<i32> = shared.get();
    assert_eq!(retrieved, data);
    
    // Create a SharedData object with a string
    let text = "Hello, world!".to_string();
    let shared = core::SharedData::new(text.clone());
    
    // Get the data
    let retrieved: String = shared.get();
    assert_eq!(retrieved, text);
}

#[test]
fn test_thread_local_config() {
    // Create a Config
    let mut config = core::Config::new();
    config.set("parallel", true);
    
    // Use the config in a closure
    let result = core::with_config(config, || {
        // Get the thread-local config
        let thread_config = core::get_config();
        
        // Check that it has the right value
        thread_config.get::<bool>("parallel").unwrap()
    });
    
    assert_eq!(result, true);
}