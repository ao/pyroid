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
| [Machine Learning Operations](./ml_ops.md) | ML algorithm acceleration | `parallel_distance_matrix`, `parallel_feature_scaling` |
| [Text & NLP Operations](./text_nlp_ops.md) | Text analysis tools | `parallel_tokenize`, `parallel_tfidf` |
| [Async Operations](./async_ops.md) | Non-blocking I/O | `AsyncClient`, `fetch_many` |
| [File I/O Operations](./io_ops.md) | Parallel file processing | `parallel_read_csv`, `parallel_json_parse` |
| [Image Processing Operations](./image_ops.md) | Efficient image manipulation | `parallel_resize`, `parallel_filter` |

## Common Patterns

Throughout the Pyroid API, you'll notice several common patterns:

### Parallel Processing

Most functions in Pyroid are designed to process multiple items in parallel. This is indicated by the `parallel_` prefix in the function name:

```python
# Process multiple items in parallel
results = pyroid.parallel_map(items, func)
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

### Transformation Functions

These functions transform input data into output data:

- `parallel_map`: Apply a function to each item in a collection
- `parallel_filter`: Filter items in a collection based on a predicate
- `parallel_sort`: Sort items in a collection
- `dataframe_apply`: Apply a function to each column or row of a DataFrame
- `parallel_feature_scaling`: Scale features using various methods
- `parallel_tokenize`: Tokenize texts
- `parallel_resize`: Resize images

### Aggregation Functions

These functions aggregate input data into a single result:

- `parallel_sum`: Calculate the sum of a collection
- `parallel_mean`: Calculate the mean of a collection
- `parallel_reduce`: Reduce a collection to a single value using a function
- `dataframe_groupby_aggregate`: Group by and aggregate DataFrame data

### Analysis Functions

These functions analyze input data and provide insights:

- `parallel_distance_matrix`: Calculate distances between points
- `parallel_tfidf`: Calculate TF-IDF matrix
- `parallel_document_similarity`: Calculate document similarity matrix
- `parallel_extract_metadata`: Extract metadata from images

### I/O Functions

These functions handle input/output operations:

- `parallel_read_csv`: Read multiple CSV files
- `parallel_json_parse`: Parse multiple JSON strings
- `parallel_compress`: Compress data
- `parallel_decompress`: Decompress data
- `AsyncClient.fetch_many`: Fetch multiple URLs concurrently

## Performance Considerations

Pyroid is designed for high performance, but there are some considerations to keep in mind:

1. **Data Size**: Pyroid's performance advantages are most noticeable with large datasets. For small datasets, the overhead of parallelization may outweigh the benefits.

2. **Memory Usage**: Processing large datasets in parallel can consume significant memory. Be mindful of memory constraints, especially when working with very large datasets.

3. **CPU Utilization**: Pyroid is designed to utilize multiple CPU cores. Ensure your system has multiple cores available for best performance.

4. **Data Conversion**: Converting between Python and Rust data structures has some overhead. Minimize the number of conversions by performing multiple operations in Rust before converting back to Python.

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