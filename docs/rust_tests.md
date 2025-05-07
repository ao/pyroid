# Rust Tests Documentation

This document provides detailed information about the Rust tests in the Pyroid project, including how to run them and what they test.

## Overview

The Pyroid project includes comprehensive tests at both the Python and Rust levels:

- **Python Tests**: Test the Python API and fallback implementations
- **Rust Tests**: Test the core Rust implementations directly, without going through Python bindings

The Rust tests ensure that the underlying Rust code works correctly before it's exposed through the Python bindings, providing an additional layer of quality assurance.

## Running the Tests

### Running All Rust Tests

To run all Rust tests:

```bash
cargo test --test test_rust_*
```

### Running Specific Module Tests

To run tests for a specific module:

```bash
# Core functionality tests
cargo test --test test_rust_core

# Math operations tests
cargo test --test test_rust_math

# Data operations tests
cargo test --test test_rust_data

# Text operations tests
cargo test --test test_rust_text

# I/O operations tests
cargo test --test test_rust_io

# Image operations tests
cargo test --test test_rust_image

# ML operations tests
cargo test --test test_rust_impl
```

### Running a Specific Test

To run a specific test:

```bash
cargo test --test test_rust_core test_config
```

### Running Tests with Verbose Output

To see more detailed output, including output from passing tests:

```bash
cargo test --test test_rust_* -- --nocapture
```

## Test Modules

### Core Tests (`test_rust_core.rs`)

Tests for the core functionality of the Pyroid library:

- **Config**: Tests for the configuration system, including setting and getting options
- **SharedData**: Tests for the shared data container
- **Thread-local Config**: Tests for thread-local configuration

### Math Tests (`test_rust_math.rs`)

Tests for mathematical operations:

- **Vector Operations**: Tests for vector addition, subtraction, scalar multiplication, dot product, and norm
- **Matrix Operations**: Tests for matrix addition, subtraction, scalar multiplication, matrix multiplication, and transpose
- **Statistical Functions**: Tests for sum, mean, median, variance, and standard deviation
- **Correlation**: Tests for correlation calculation

### Data Tests (`test_rust_data.rs`)

Tests for data operations:

- **Filter**: Tests for filtering collections
- **Map**: Tests for mapping operations
- **Reduce**: Tests for reduction operations
- **Sort**: Tests for sorting operations
- **DataFrame**: Tests for DataFrame creation and column access
- **DataFrame Apply**: Tests for applying functions to DataFrame columns
- **DataFrame GroupBy**: Tests for grouping and aggregating DataFrame data

### Text Tests (`test_rust_text.rs`)

Tests for text operations:

- **Reverse**: Tests for string reversal
- **Base64**: Tests for base64 encoding and decoding
- **Split**: Tests for string splitting
- **Join**: Tests for string joining
- **Replace**: Tests for string replacement
- **Regex Replace**: Tests for regex-based replacement
- **Case Conversion**: Tests for uppercase and lowercase conversion
- **Tokenize**: Tests for text tokenization
- **N-grams**: Tests for n-gram generation

### IO Tests (`test_rust_io.rs`)

Tests for input/output operations:

- **File Operations**: Tests for reading, writing, and appending to files
- **Directory Creation**: Tests for creating directories when writing files
- **Error Handling**: Tests for handling errors with nonexistent files

### Image Tests (`test_rust_image.rs`)

Tests for image processing:

- **Image Creation**: Tests for creating new images
- **Pixel Manipulation**: Tests for getting and setting pixel values
- **Resize**: Tests for resizing images
- **Grayscale**: Tests for converting images to grayscale
- **Blur**: Tests for blurring images

### ML Tests (`test_rust_impl.rs`)

Tests for machine learning operations:

- **K-means Clustering**: Tests for k-means clustering algorithm
- **Linear Regression**: Tests for linear regression
- **Normalize**: Tests for data normalization
- **Distance Matrix**: Tests for calculating distance matrices

## Test Implementation

The tests are implemented using Rust's built-in testing framework. The common module (`tests/common/mod.rs`) provides implementations of the functionality being tested, which are then used by the test modules.

### Common Module

The common module contains pure Rust implementations of all the functionality tested:

```rust
pub mod core { /* ... */ }
pub mod math { /* ... */ }
pub mod text { /* ... */ }
pub mod data { /* ... */ }
pub mod io { /* ... */ }
pub mod image { /* ... */ }
pub mod ml { /* ... */ }
```

Each module provides implementations of the corresponding functionality in the Pyroid library.

### Test Structure

Each test module follows a similar structure:

1. Import the common module
2. Define test functions for each piece of functionality
3. Use assertions to verify that the functionality works as expected

For example:

```rust
#[test]
fn test_config() {
    // Create a new Config
    let mut config = core::Config::new();
    
    // Set some options
    config.set("parallel", true);
    
    // Get the options
    assert_eq!(config.get::<bool>("parallel").unwrap(), true);
}
```

## Adding New Tests

To add a new test:

1. Identify the appropriate test module for your test
2. Add a new test function with the `#[test]` attribute
3. Implement the test logic
4. Run the test to ensure it passes

If you need to test new functionality:

1. Add the necessary implementation to the common module
2. Create a new test function in the appropriate test module
3. Run the test to ensure it passes

## Troubleshooting

### Common Issues

- **Test fails with "assertion failed"**: Check that the implementation in the common module matches the expected behavior
- **Test fails with "panicked at..."**: Check for runtime errors in the implementation
- **Dead code warnings**: These are normal and can be ignored, as not all functions in the common module are used in every test

### Debugging Tests

To debug a failing test:

1. Run the test with verbose output: `cargo test --test test_rust_* -- --nocapture`
2. Add print statements to the test to see intermediate values
3. Check the implementation in the common module for errors