# Pyroid API Reference

This is the API reference for Pyroid, a high-performance Rust-powered library for Python. Pyroid provides accelerated implementations of common operations across multiple domains, designed to eliminate Python's performance bottlenecks.

## Module Overview

Pyroid is organized into several modules, each focusing on a specific domain:

| Module | Description | Key Functions |
|--------|-------------|--------------|
| [Math Operations](./math_ops.md) | Fast numerical computations | `parallel_sum`, `matrix_multiply` |
| [String Operations](./string_ops.md) | Efficient text processing | `parallel_regex_replace`, `parallel_text_cleanup` |
| [Data Operations](./data_ops.md) | Collection manipulation | `parallel_filter`, `parallel_map`, `parallel_sort` |
| [DataFrame Operations](./dataframe_ops.md) | Fast pandas-like operations | `dataframe_apply`, `dataframe_groupby_aggregate` |
| [Machine Learning Operations](./ml_ops.md) | Basic ML algorithms | `kmeans`, `linear_regression`, `normalize`, `distance_matrix` |
| [Text & NLP Operations](./text_nlp_ops.md) | Text analysis tools | `parallel_tokenize`, `parallel_tfidf` |
| [Async Operations](./async_ops.md) | Non-blocking I/O | `sleep`, `read_file_async`, `http_get_async` |
| [File I/O Operations](./io_ops.md) | File processing | `read_file`, `write_file`, `read_files` |
| [Image Processing Operations](./image_ops.md) | Basic image manipulation | `create_image`, `to_grayscale`, `resize`, `blur` |

## Common Patterns

Throughout the Pyroid API, you'll notice several common patterns:

### Simple and Consistent API

Pyroid functions are designed with simplicity and consistency in mind:

```python
# Simple function calls with clear parameters
result = pyroid.ml.basic.kmeans(data, k=3)
```

### Optional Parameters

Many functions accept optional parameters with sensible defaults:

```python
# Use default parameters
result = pyroid.parallel_tokenize(texts)

# Override defaults
result = pyroid.parallel_tokenize(texts, lowercase=False, remove_punct=False)
```

### Return Types

Pyroid functions typically return Python-native data structures:

- Lists for collections of items
- Dictionaries for structured data
- Tuples for multiple return values

This makes it easy to integrate Pyroid with existing Python code.

## Function Categories

### Math and Data Functions

These functions handle mathematical and data operations:

- `vector_add`, `vector_subtract`: Vector operations
- `matrix_multiply`: Matrix multiplication
- `sum`, `mean`, `median`: Statistical operations
- `filter`, `map`, `reduce`: Collection operations
- `sort`: Sort items in a collection
- `dataframe_apply`: Apply a function to DataFrame data

### Machine Learning Functions

These functions provide basic machine learning capabilities:

- `kmeans`: K-means clustering algorithm
- `linear_regression`: Simple linear regression
- `normalize`: Data normalization
- `distance_matrix`: Calculate distances between points

### Image Processing Functions

These functions handle image operations:

- `create_image`: Create a new image
- `to_grayscale`: Convert image to grayscale
- `resize`: Resize an image
- `blur`: Apply blur filter to an image
- `adjust_brightness`: Adjust image brightness

### I/O Functions

These functions handle input/output operations:

- `read_file`, `write_file`: File operations
- `read_files`: Read multiple files
- `get`, `post`: HTTP requests
- `sleep`, `read_file_async`, `http_get_async`: Async operations

## Performance Considerations

Pyroid is designed for good performance with minimal dependencies:

1. **Rust Implementation**: Core operations are implemented in Rust, providing better performance than pure Python.

2. **Minimal Dependencies**: The library has minimal external dependencies, making it more reliable and easier to install.

3. **Memory Efficiency**: Operations are designed to be memory-efficient, with careful management of resources.

4. **Data Conversion**: Converting between Python and Rust data structures has some overhead. For very large datasets, this overhead may be noticeable.

5. **Image and ML Operations**: Basic image and machine learning operations are implemented without external dependencies, providing good baseline performance.

For more detailed performance optimization tips, see the [Performance Optimization Guide](../guides/performance.md).

## Error Handling

Pyroid functions raise Python exceptions when errors occur. Common exceptions include:

- `ValueError`: Invalid input values
- `TypeError`: Incompatible input types
- `RuntimeError`: Errors during execution
- `FileNotFoundError`: File not found (for I/O operations)

Example of error handling:

```python
try:
    result = pyroid.parallel_sum(data)
except ValueError as e:
    print(f"Invalid input: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
```

## Thread Safety

Pyroid functions are generally thread-safe, meaning they can be called from multiple Python threads concurrently. However, some functions that maintain internal state (like `AsyncClient`) may not be thread-safe and should not be shared across threads without proper synchronization.

## Module Details

For detailed information about each module, including function signatures, parameters, return values, and examples, click on the module links in the table above.