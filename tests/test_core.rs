//! Integration tests for the core module

use pyroid::core::config::Config;
use pyroid::core::types::SharedData;

#[test]
fn test_config() {
    // Create a new Config
    let mut config = Config::new();
    
    // Set some options
    config.set("parallel", true);
    config.set("chunk_size", 1000);
    
    // Get the options
    assert_eq!(config.get::<bool>("parallel").unwrap(), true);
    assert_eq!(config.get::<i32>("chunk_size").unwrap(), 1000);
    
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
    let shared = SharedData::new(data.clone());
    
    // Get the data
    let retrieved: Vec<i32> = shared.get();
    assert_eq!(retrieved, data);
}

#[test]
fn test_thread_local_config() {
    use pyroid::core::config::with_config;
    
    // Create a Config
    let mut config = Config::new();
    config.set("parallel", true);
    
    // Use the config in a closure
    let result = with_config(config, || {
        // Get the thread-local config
        let thread_config = pyroid::core::config::get_config();
        
        // Check that it has the right value
        thread_config.get::<bool>("parallel").unwrap()
    });
    
    assert_eq!(result, true);
}