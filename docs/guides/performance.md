# Performance Optimization Guide

This guide provides tips and best practices for optimizing the performance of your Pyroid applications. Pyroid is already designed to be high-performance, but following these guidelines can help you get the most out of it.

## Understanding Pyroid's Performance Advantages

Pyroid achieves its performance advantages through several key mechanisms:

1. **Rust Implementation**: Core operations are implemented in Rust, a systems programming language that offers performance comparable to C/C++ with memory safety guarantees.

2. **Parallel Processing**: Pyroid uses Rayon, a data parallelism library for Rust, to automatically parallelize operations across multiple CPU cores.

3. **Efficient Memory Management**: Rust's ownership model allows for efficient memory management without the overhead of garbage collection.

4. **SIMD Instructions**: Where applicable, Pyroid leverages SIMD (Single Instruction, Multiple Data) instructions for vectorized operations.

5. **Optimized Algorithms**: Pyroid implements optimized versions of common algorithms, taking advantage of Rust's performance characteristics.

## General Performance Tips

### 1. Process Data in Batches

Pyroid's parallel processing capabilities shine when processing multiple items at once. Instead of processing items one by one, collect them into a batch and process them all at once.

```python
# Less efficient
results = []
for item in items:
    result = pyroid.some_function(item)
    results.append(result)

# More efficient
results = pyroid.parallel_some_function(items)
```

### 2. Choose Appropriate Batch Sizes

While processing data in batches is more efficient, extremely large batches can lead to memory issues. Find a balance that works for your system:

```python
# Process data in manageable chunks
batch_size = 10000
results = []

for i in range(0, len(items), batch_size):
    batch = items[i:i+batch_size]
    batch_results = pyroid.parallel_some_function(batch)
    results.extend(batch_results)
```

### 3. Minimize Data Conversion

Converting between Python and Rust data structures has some overhead. Minimize the number of conversions by performing multiple operations in Rust before converting back to Python:

```python
# Less efficient (multiple conversions)
intermediate = pyroid.operation1(data)
result = pyroid.operation2(intermediate)

# More efficient (if available)
result = pyroid.combined_operations(data)
```

### 4. Use Type-Specific Functions

When available, use functions that are specific to your data types rather than generic functions:

```python
# Less efficient (generic)
result = pyroid.parallel_map(numbers, lambda x: x * 2)

# More efficient (specific to numeric data)
result = pyroid.parallel_multiply(numbers, 2)
```

### 5. Reuse Objects When Possible

Some Pyroid operations involve creating internal objects that can be reused:

```python
# Less efficient
for url in urls:
    client = pyroid.AsyncClient()
    response = await client.fetch(url)
    # Process response

# More efficient
client = pyroid.AsyncClient()
for url in urls:
    response = await client.fetch(url)
    # Process response
```

## Domain-Specific Optimization Tips

### Math Operations

1. **Use Specialized Functions**: Use specialized functions like `parallel_sum` instead of generic functions like `parallel_reduce` for better performance.

2. **Consider Data Layout**: For matrix operations, ensure your matrices are in the correct layout (row-major or column-major) for the operation you're performing.

3. **Use Appropriate Data Types**: Use the appropriate data types for your calculations. For example, use integers instead of floats when possible.

```python
# Example: Optimized matrix multiplication
import pyroid
import time

# Generate test matrices
n = 1000
matrix_a = [[i + j for j in range(n)] for i in range(n)]
matrix_b = [[i * j for j in range(n)] for i in range(n)]

# Measure performance
start = time.time()
result = pyroid.matrix_multiply(matrix_a, matrix_b)
end = time.time()

print(f"Time taken: {end - start:.2f} seconds")
```

### String Operations

1. **Precompile Regular Expressions**: For repeated regex operations on different strings, precompile the regex pattern if possible.

2. **Batch Text Processing**: Process multiple texts at once using functions like `parallel_regex_replace` or `parallel_text_cleanup`.

3. **Consider Text Encoding**: Be aware of text encoding issues, especially when processing non-ASCII text.

```python
# Example: Efficient batch text processing
import pyroid
import time

# Generate test data
texts = ["Hello, world! " * 100 for _ in range(10000)]

# Measure performance
start = time.time()
cleaned = pyroid.parallel_text_cleanup(texts)
end = time.time()

print(f"Time taken: {end - start:.2f} seconds")
```

### Data Operations

1. **Use Appropriate Comparison Functions**: For sorting, provide a specialized comparison function if the default comparison is not optimal.

2. **Consider Memory Usage**: For very large datasets, be mindful of memory usage and consider processing data in chunks.

3. **Avoid Unnecessary Copies**: When possible, use in-place operations to avoid unnecessary data copying.

```python
# Example: Efficient sorting with custom comparison
import pyroid
import time
import random

# Generate test data
data = [(random.random(), random.random()) for _ in range(1000000)]

# Define custom comparison function (sort by second element)
def compare(a, b):
    return -1 if a[1] < b[1] else (1 if a[1] > b[1] else 0)

# Measure performance
start = time.time()
sorted_data = pyroid.parallel_sort(data, compare, False)
end = time.time()

print(f"Time taken: {end - start:.2f} seconds")
```

### DataFrame Operations

1. **Specify Schema**: When working with DataFrames, specify the schema to avoid type inference.

2. **Use Column-Wise Operations**: Column-wise operations (axis=0) are generally faster than row-wise operations (axis=1).

3. **Combine Multiple Transformations**: Use `parallel_transform` to apply multiple transformations in a single pass.

```python
# Example: Efficient DataFrame operations
import pyroid
import time
import random

# Generate test data
n_rows = 1000000
df = {
    'A': [random.random() for _ in range(n_rows)],
    'B': [random.random() for _ in range(n_rows)],
    'C': [random.random() for _ in range(n_rows)]
}

# Apply multiple transformations in one pass
transformations = [
    ('A', 'log', None),
    ('B', 'sqrt', None),
    ('C', 'round', 2)
]

# Measure performance
start = time.time()
result = pyroid.parallel_transform(df, transformations)
end = time.time()

print(f"Time taken: {end - start:.2f} seconds")
```

### Machine Learning Operations

1. **Scale Features Appropriately**: Use `parallel_feature_scaling` with the appropriate method for your data.

2. **Choose the Right Distance Metric**: Different distance metrics have different performance characteristics. Choose the one that best fits your needs.

3. **Optimize Cross-Validation**: For cross-validation, ensure your model function is as efficient as possible.

```python
# Example: Efficient distance matrix calculation
import pyroid
import time
import random

# Generate test data
n_points = 5000
n_dims = 10
points = [[random.random() for _ in range(n_dims)] for _ in range(n_points)]

# Measure performance
start = time.time()
distances = pyroid.parallel_distance_matrix(points, "euclidean")
end = time.time()

print(f"Time taken: {end - start:.2f} seconds")
```

### Text and NLP Operations

1. **Preprocess Text**: Clean and normalize your text data before processing to improve results quality and performance.

2. **Use Tokenized Input**: If you already have tokenized texts, set `tokenized=True` to avoid redundant tokenization.

3. **Filter Out Rare and Common Terms**: Use the `min_df` and `max_df` parameters in `parallel_tfidf` to filter out terms that are too rare or too common.

```python
# Example: Efficient TF-IDF calculation
import pyroid
import time

# Sample documents
documents = [
    "This is the first document",
    "This document is the second document",
    "And this is the third one",
    "Is this the first document?"
] * 250  # Repeat to create a larger dataset

# Measure performance
start = time.time()
tfidf_matrix, vocabulary = pyroid.parallel_tfidf(documents, False, min_df=2, max_df=0.9)
end = time.time()

print(f"Time taken: {end - start:.2f} seconds")
```

### Async Operations

1. **Set Appropriate Concurrency**: Set an appropriate concurrency level based on your system resources and the nature of the I/O operations.

2. **Reuse Client**: Create the `AsyncClient` once and reuse it for multiple operations.

3. **Handle Errors Properly**: Ensure proper error handling to avoid performance degradation due to unhandled exceptions.

```python
# Example: Efficient async HTTP requests
import pyroid
import asyncio
import time

async def main():
    # Create client once
    client = pyroid.AsyncClient()
    
    # Generate test URLs
    urls = ["https://httpbin.org/delay/1"] * 10
    
    # Measure performance
    start = time.time()
    responses = await client.fetch_many(urls, concurrency=5)
    end = time.time()
    
    print(f"Time taken: {end - start:.2f} seconds")
    print(f"Average time per request: {(end - start) / len(urls):.2f} seconds")

asyncio.run(main())
```

### File I/O Operations

1. **Specify Schema for CSV Files**: When reading CSV files, specify a schema to improve performance and ensure correct data types.

2. **Choose Appropriate Compression Level**: For compression, choose a compression level that balances speed and compression ratio.

3. **Batch Process Large Files**: For very large files, consider processing them in batches to avoid memory issues.

```python
# Example: Efficient CSV reading
import pyroid
import time
import pandas as pd
import os

# Create a sample CSV file
df = pd.DataFrame({
    'id': range(100000),
    'name': [f"item_{i}" for i in range(100000)],
    'value': [i / 100 for i in range(100000)],
    'flag': [i % 2 == 0 for i in range(100000)]
})
df.to_csv("test_data.csv", index=False)

# Define schema
schema = {
    'id': 'int',
    'value': 'float',
    'flag': 'bool'
}

# Measure performance
start = time.time()
data = pyroid.parallel_read_csv(["test_data.csv"], schema)
end = time.time()

print(f"Time taken: {end - start:.2f} seconds")

# Clean up
os.remove("test_data.csv")
```

### Image Processing Operations

1. **Choose the Appropriate Resampling Filter**: Different resampling filters have different quality and performance characteristics.

2. **Optimize Image Dimensions**: Resize images to the dimensions you need before applying filters or other operations.

3. **Use Appropriate Image Formats**: Choose the right format for your use case to balance quality, file size, and processing time.

```python
# Example: Efficient image batch processing
import pyroid
import time
from PIL import Image
import io
import os

# Create sample images
os.makedirs("test_images", exist_ok=True)
images = []
for i in range(10):
    img = Image.new('RGB', (800, 600), color=(i*20, i*20, i*20))
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    images.append(buffer.getvalue())

# Measure performance
start = time.time()
resized = pyroid.parallel_resize(images, (400, 300), "lanczos3")
filtered = pyroid.parallel_filter(resized, "blur", {"sigma": 2.0})
converted = pyroid.parallel_convert(filtered, None, "png")
end = time.time()

print(f"Time taken: {end - start:.2f} seconds")
```

## System-Level Optimization

### 1. CPU Utilization

Pyroid is designed to utilize multiple CPU cores efficiently. To get the best performance:

- Ensure your system has multiple CPU cores available.
- Monitor CPU utilization during processing to ensure all cores are being used.
- Avoid running other CPU-intensive tasks simultaneously.

### 2. Memory Management

Efficient memory management is crucial for performance:

- Monitor memory usage during processing to ensure you're not running out of memory.
- For very large datasets, process data in chunks to avoid memory issues.
- Consider using memory-mapped files for very large datasets.

### 3. I/O Optimization

I/O operations can be a bottleneck:

- Use SSDs instead of HDDs for better I/O performance.
- Minimize disk I/O by processing data in memory when possible.
- Use async I/O operations for network and disk operations.

### 4. Environment Variables

Pyroid uses Rayon for parallelism, which respects the `RAYON_NUM_THREADS` environment variable:

```bash
# Set the number of threads for Rayon
export RAYON_NUM_THREADS=8
```

This can be useful for controlling the level of parallelism, especially in environments with limited resources or when running multiple parallel processes.

## Benchmarking and Profiling

### 1. Simple Benchmarking

Use a simple benchmarking function to measure the performance of your code:

```python
import time

def benchmark(func, *args, **kwargs):
    """Simple benchmarking function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")
    return result
```

### 2. Comparing Implementations

Compare different implementations to find the most efficient one:

```python
import pyroid
import time
import numpy as np

# Generate test data
data = [random.random() for _ in range(1000000)]
data_np = np.array(data)

# Python implementation
start = time.time()
python_result = sum(data)
python_time = time.time() - start

# NumPy implementation
start = time.time()
numpy_result = np.sum(data_np)
numpy_time = time.time() - start

# Pyroid implementation
start = time.time()
pyroid_result = pyroid.parallel_sum(data)
pyroid_time = time.time() - start

print(f"Python time: {python_time:.2f}s, result: {python_result}")
print(f"NumPy time: {numpy_time:.2f}s, result: {numpy_result}")
print(f"Pyroid time: {pyroid_time:.2f}s, result: {pyroid_result}")
print(f"Speedup vs Python: {python_time / pyroid_time:.1f}x")
print(f"Speedup vs NumPy: {numpy_time / pyroid_time:.1f}x")
```

### 3. Profiling

For more detailed performance analysis, use a profiler like cProfile:

```python
import cProfile
import pyroid

def test_function():
    data = [i for i in range(1000000)]
    result = pyroid.parallel_sum(data)
    return result

cProfile.run('test_function()')
```

Or use a more sophisticated profiler like py-spy:

```bash
pip install py-spy
py-spy record -o profile.svg -- python your_script.py
```

## Common Performance Pitfalls

### 1. Small Data Sizes

Pyroid's performance advantages are most noticeable with large datasets. For small datasets, the overhead of parallelization may outweigh the benefits:

```python
# For small datasets, Python's built-in functions may be faster
small_data = [1, 2, 3, 4, 5]
result = sum(small_data)  # Faster than pyroid.parallel_sum for small data
```

### 2. Non-Parallelizable Operations

Some operations are inherently sequential and don't benefit from parallelization:

```python
# Operations with dependencies between steps don't parallelize well
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

### 3. Excessive Data Conversion

Converting between Python and Rust data structures has overhead:

```python
# Avoid unnecessary conversions
data = [1, 2, 3, 4, 5]
result1 = pyroid.operation1(data)
result2 = pyroid.operation2(result1)  # Converts back to Python and then to Rust again
```

### 4. Ignoring Memory Constraints

Processing very large datasets without considering memory constraints can lead to performance issues:

```python
# This might cause memory issues for very large files
data = pyroid.parallel_read_csv(["very_large_file.csv"])

# Better approach: process in chunks
chunk_size = 100000
# ... implement chunked processing
```

### 5. Inappropriate Parallelism Level

Setting too high or too low a level of parallelism can hurt performance:

```python
# Too low concurrency wastes resources
responses = await client.fetch_many(urls, concurrency=1)  # Too low

# Too high concurrency might overwhelm the system
responses = await client.fetch_many(urls, concurrency=1000)  # Too high

# Find the right balance
responses = await client.fetch_many(urls, concurrency=10)  # Better
```

## Conclusion

Optimizing performance with Pyroid involves understanding its parallel processing capabilities, choosing the right functions for your tasks, and being mindful of system resources. By following the guidelines in this document, you can ensure that your Pyroid applications run as efficiently as possible.

Remember that performance optimization is often an iterative process. Measure, optimize, and measure again to ensure that your optimizations are actually improving performance.

For more detailed information about specific functions and their performance characteristics, refer to the [API documentation](../api/).