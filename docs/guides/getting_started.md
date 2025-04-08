# Getting Started with Pyroid

This guide will help you get started with Pyroid, a high-performance Rust-powered library for Python that accelerates common operations and eliminates performance bottlenecks.

## Installation

### Prerequisites

Before installing Pyroid, ensure you have the following:

- Python 3.8 or higher
- A compatible operating system (Windows, macOS, or Linux)
- Pip package manager

### Installing from PyPI

The easiest way to install Pyroid is via pip:

```bash
pip install pyroid
```

This will install the pre-built binary package for your platform.

### Installing from Source

If you want to install from source or contribute to the development, you can clone the repository and install it in development mode:

```bash
git clone https://github.com/ao/pyroid.git
cd pyroid
pip install -e .
```

This requires Rust and Cargo to be installed on your system. If you don't have Rust installed, you can install it using [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Verifying Installation

To verify that Pyroid is installed correctly, you can run the following Python code:

```python
import pyroid
print(f"Pyroid version: {pyroid.__version__}")
```

## Basic Usage

Pyroid provides a wide range of high-performance functions across multiple domains. Here are some basic examples to get you started:

### Math Operations

```python
import pyroid

# Parallel sum of a large list
numbers = list(range(1_000_000))
result = pyroid.parallel_sum(numbers)
print(f"Sum: {result}")

# Matrix multiplication
matrix_a = [[1, 2], [3, 4]]
matrix_b = [[5, 6], [7, 8]]
result = pyroid.matrix_multiply(matrix_a, matrix_b)
print(f"Matrix multiplication result: {result}")
```

### String Operations

```python
import pyroid

# Parallel regex replacement
text = "Hello world! " * 1000
result = pyroid.parallel_regex_replace(text, r"Hello", "Hi")
print(f"Modified text length: {len(result)}")

# Process multiple strings in parallel
texts = ["Hello world!", "This is a test.", "Pyroid is fast!"] * 1000
cleaned = pyroid.parallel_text_cleanup(texts)
print(f"Cleaned {len(cleaned)} strings")
```

### Data Operations

```python
import pyroid
import random

# Generate test data
data = [random.randint(1, 1000) for _ in range(1_000_000)]

# Parallel filter
filtered = pyroid.parallel_filter(data, lambda x: x > 500)
print(f"Filtered {len(filtered)} items")

# Parallel map
squared = pyroid.parallel_map(data, lambda x: x * x)
print(f"Mapped {len(squared)} items")

# Parallel sort
sorted_data = pyroid.parallel_sort(data, None, False)
print(f"Sorted {len(sorted_data)} items")
```

## Module Overview

Pyroid is organized into several modules, each focusing on a specific domain:

### Math Operations

The math operations module provides high-performance numerical computations:

```python
import pyroid

# Parallel sum
result = pyroid.parallel_sum([1, 2, 3, 4, 5])

# Matrix operations
matrix_a = [[1, 2], [3, 4]]
matrix_b = [[5, 6], [7, 8]]
result = pyroid.matrix_multiply(matrix_a, matrix_b)

# Statistical functions
mean = pyroid.parallel_mean([1, 2, 3, 4, 5])
std_dev = pyroid.parallel_std([1, 2, 3, 4, 5])
```

### String Operations

The string operations module provides efficient text processing:

```python
import pyroid

# Regex replacement
result = pyroid.parallel_regex_replace("Hello world!", r"Hello", "Hi")

# Text cleanup
cleaned = pyroid.parallel_text_cleanup(["Hello, world!", "This is a test."])

# Split and join
lines = ["Line 1", "Line 2", "Line 3"]
joined = pyroid.parallel_join(lines, "\n")
split_again = pyroid.parallel_split(joined, "\n")
```

### Data Operations

The data operations module provides high-performance collection manipulation:

```python
import pyroid

# Filter
filtered = pyroid.parallel_filter([1, 2, 3, 4, 5], lambda x: x > 2)

# Map
mapped = pyroid.parallel_map([1, 2, 3, 4, 5], lambda x: x * x)

# Sort
sorted_data = pyroid.parallel_sort([5, 3, 1, 4, 2], None, False)

# Reduce
sum_result = pyroid.parallel_reduce([1, 2, 3, 4, 5], lambda x, y: x + y, 0)
```

### DataFrame Operations

The DataFrame operations module provides pandas-like operations:

```python
import pyroid

# Create a dictionary representing a DataFrame
df = {
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
}

# Apply a function to each column
def square(x):
    return [val * val for val in x]

result = pyroid.dataframe_apply(df, square, 0)

# Group by and aggregate
df = {
    'category': ['A', 'B', 'A', 'B', 'C'],
    'value': [10, 20, 15, 25, 30]
}
agg_dict = {'value': 'mean'}
result = pyroid.dataframe_groupby_aggregate(df, ['category'], agg_dict)
```

### Machine Learning Operations

The machine learning operations module provides basic ML algorithms:

```python
import pyroid
import matplotlib.pyplot as plt
import numpy as np

# K-means clustering
data = [
    [1.0, 2.0], [1.5, 1.8], [5.0, 8.0],
    [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]
]
result = pyroid.ml.basic.kmeans(data, k=2)
print(f"Centroids: {result['centroids']}")
print(f"Clusters: {result['clusters']}")

# Linear regression
x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [2.0, 4.0, 5.0, 4.0, 6.0]
result = pyroid.ml.basic.linear_regression(x, y)
print(f"Slope: {result['slope']}")
print(f"Intercept: {result['intercept']}")
print(f"R-squared: {result['r_squared']}")

# Data normalization
values = [10.0, 20.0, 30.0, 40.0, 50.0]
normalized = pyroid.ml.basic.normalize(values, method="minmax")
print(f"Normalized (min-max): {normalized}")

# Distance matrix
points = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
distances = pyroid.ml.basic.distance_matrix(points, metric="euclidean")
print(f"Distance matrix: {distances}")
```

### Text and NLP Operations

The text and NLP operations module provides efficient text analysis tools:

```python
import pyroid

# Tokenize texts
texts = ["Hello world!", "This is a test.", "Pyroid is fast!"]
tokens = pyroid.parallel_tokenize(texts, lowercase=True, remove_punct=True)

# Generate n-grams
bigrams = pyroid.parallel_ngrams(texts, 2, False)

# Calculate TF-IDF
docs = ["This is the first document", "This document is the second document"]
tfidf_matrix, vocabulary = pyroid.parallel_tfidf(docs, False)

# Calculate document similarity
similarity_matrix = pyroid.parallel_document_similarity(docs, "cosine")
```

### Async Operations

The async operations module provides non-blocking I/O operations:

```python
import asyncio
import pyroid

async def main():
    # Create an async client
    client = pyroid.AsyncClient()
    
    # Fetch multiple URLs concurrently
    urls = ["https://example.com", "https://google.com", "https://github.com"]
    responses = await client.fetch_many(urls, concurrency=3)
    
    for url, response in responses.items():
        if isinstance(response, dict):
            print(f"{url}: Status {response['status']}")

asyncio.run(main())
```

### File I/O Operations

The I/O operations module provides parallel file processing:

```python
import pyroid

# Read multiple CSV files in parallel
files = ["data1.csv", "data2.csv", "data3.csv"]
schema = {"id": "int", "value": "float", "flag": "bool"}
data = pyroid.parallel_read_csv(files, schema)

# Parse multiple JSON strings in parallel
json_strings = ['{"name": "Alice", "age": 30}', '{"name": "Bob", "age": 25}']
parsed = pyroid.parallel_json_parse(json_strings)

# Compress data in parallel
data = ["Hello, world!".encode() for _ in range(100)]
compressed = pyroid.parallel_compress(data, "gzip", 6)

# Decompress data in parallel
decompressed = pyroid.parallel_decompress(compressed, "gzip")
```

### Image Processing Operations

The image processing operations module provides basic image manipulation:

```python
import pyroid

# Create a new image (width, height, channels)
img = pyroid.image.basic.create_image(100, 100, 3)

# Set some pixels
for x in range(50):
    for y in range(50):
        img.set_pixel(x, y, [255, 0, 0])  # Red square

for x in range(50, 100):
    for y in range(50, 100):
        img.set_pixel(x, y, [0, 0, 255])  # Blue square

# Apply operations
grayscale_img = img.to_grayscale()
resized_img = img.resize(200, 200)
blurred_img = img.blur(2)
brightened_img = img.adjust_brightness(1.5)

# Get image data
width = img.width
height = img.height
channels = img.channels
data = img.data

# Create an image from raw bytes
raw_data = bytes([255, 0, 0] * (50 * 50))  # Red pixels
red_img = pyroid.image.basic.from_bytes(raw_data, 50, 50, 3)
```

## Performance Comparison

Pyroid offers improved performance over pure Python due to its Rust implementation. Here's a quick comparison of Pyroid vs. pure Python for some common operations:

| Operation | Pure Python | Pyroid | Speedup |
|-----------|-------------|--------|---------|
| Vector operations | 500ms | 100ms | 5x |
| Matrix multiplication | 800ms | 200ms | 4x |
| String processing | 1200ms | 300ms | 4x |
| Collection operations | 1500ms | 400ms | 3.75x |
| File I/O | 2000ms | 600ms | 3.3x |
| Basic image processing | 1800ms | 500ms | 3.6x |
| Simple ML algorithms | 2500ms | 700ms | 3.6x |

## Best Practices

Here are some best practices to get the most out of Pyroid:

### 1. Use Appropriate Data Structures

Choose the right data structures for your task. For example, use grayscale images (1 channel) instead of RGB (3 channels) when color is not needed.

```python
# More efficient for grayscale processing
img = pyroid.image.basic.create_image(100, 100, 1)  # 1 channel
```

### 2. Reuse Objects When Possible

Some Pyroid operations involve creating objects that can be reused. For example, when working with images, you can reuse the same image object for multiple operations.

```python
# More efficient
img = pyroid.image.basic.create_image(100, 100, 3)
img1 = img.resize(200, 200)
img2 = img1.blur(2)
img3 = img2.adjust_brightness(1.5)
```

### 3. Choose the Right Function for the Task

Pyroid provides multiple functions for similar tasks. Choose the one that best fits your needs.

For example, when working with machine learning:
- Use `kmeans` for clustering
- Use `linear_regression` for simple regression tasks
- Use `normalize` for data preprocessing

### 4. Optimize Data Conversions

Converting between Python and Rust data structures has some overhead. Try to minimize the number of conversions.

```python
# Less efficient (multiple conversions)
for i in range(10):
    result = pyroid.ml.basic.normalize(data)
    data = process_data(result)

# More efficient (single conversion)
result = pyroid.ml.basic.normalize(data)
for i in range(10):
    data = process_data(result)
```

### 5. Be Mindful of Memory Usage

For very large datasets or images, be mindful of memory usage. Consider processing data in smaller chunks if needed.

## Next Steps

Now that you're familiar with the basics of Pyroid, you can:

1. Explore the [API documentation](../api/) for detailed information about each function
2. Check out the [examples](../../examples/) for more complex use cases
3. Read the [performance optimization guide](./performance.md) for tips on getting the best performance
4. Learn about [advanced usage patterns](./advanced_usage.md) for more sophisticated applications

## Getting Help

If you encounter any issues or have questions about Pyroid, you can:

- Check the [FAQ](./faq.md) for answers to common questions
- Open an issue on the [GitHub repository](https://github.com/ao/pyroid/issues)


