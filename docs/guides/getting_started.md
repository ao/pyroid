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

The machine learning operations module provides accelerated ML primitives:

```python
import pyroid

# Calculate distance matrix
points = [[1, 2], [3, 4], [5, 6]]
distances = pyroid.parallel_distance_matrix(points, "euclidean")

# Scale features
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
scaled = pyroid.parallel_feature_scaling(data, "standard")

# Cross-validation
X = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [0, 1, 0, 1]

def knn_model(X_train, y_train, X_test):
    # Simple 1-NN implementation
    predictions = []
    for test_point in X_test:
        min_dist = float('inf')
        min_idx = 0
        for i, train_point in enumerate(X_train):
            dist = sum((a - b) ** 2 for a, b in zip(test_point, train_point)) ** 0.5
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        predictions.append(y_train[min_idx])
    return predictions

def accuracy(y_true, y_pred):
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)

scores = pyroid.parallel_cross_validation(X, y, 2, knn_model, accuracy)
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

The image processing operations module provides efficient image manipulation:

```python
import pyroid
from PIL import Image
import io

# Load some images
with open("image1.jpg", "rb") as f:
    image1 = f.read()
with open("image2.jpg", "rb") as f:
    image2 = f.read()

# Resize images in parallel
images = [image1, image2]
resized = pyroid.parallel_resize(images, (800, 600), "lanczos3")

# Apply filters
filtered = pyroid.parallel_filter(images, "blur", {"sigma": 2.0})

# Convert formats
converted = pyroid.parallel_convert(images, None, "png")

# Extract metadata
metadata = pyroid.parallel_extract_metadata(images)
```

## Performance Comparison

One of the main advantages of Pyroid is its performance. Here's a quick comparison of Pyroid vs. pure Python for some common operations:

| Operation | Pure Python | Pyroid | Speedup |
|-----------|-------------|--------|---------|
| Sum 10M numbers | 1000ms | 50ms | 20x |
| Regex on 10MB text | 2500ms | 200ms | 12.5x |
| Sort 10M items | 3500ms | 300ms | 11.7x |
| 100 HTTP requests | 5000ms | 500ms | 10x |
| DataFrame groupby | 3000ms | 200ms | 15x |
| TF-IDF calculation | 4000ms | 300ms | 13.3x |
| Image batch resize | 2000ms | 150ms | 13.3x |

## Best Practices

Here are some best practices to get the most out of Pyroid:

### 1. Process Data in Batches

Pyroid is designed for parallel processing, so it's most efficient when processing multiple items at once. Instead of processing items one by one in a loop, collect them into a batch and process them all at once.

```python
# Less efficient
results = []
for item in items:
    result = pyroid.some_function(item)
    results.append(result)

# More efficient
results = pyroid.parallel_some_function(items)
```

### 2. Reuse Objects When Possible

Some Pyroid operations involve creating internal objects that can be reused. For example, when using the `AsyncClient` for multiple HTTP requests, create the client once and reuse it.

```python
# Less efficient
async def fetch_url(url):
    client = pyroid.AsyncClient()
    response = await client.fetch(url)
    return response

# More efficient
client = pyroid.AsyncClient()

async def fetch_url(url):
    response = await client.fetch(url)
    return response
```

### 3. Choose the Right Function for the Task

Pyroid provides multiple functions for similar tasks, each optimized for different scenarios. Choose the one that best fits your needs.

For example, when working with DataFrames:
- Use `dataframe_apply` for applying a function to each column or row
- Use `parallel_transform` for applying multiple transformations in one pass
- Use `dataframe_groupby_aggregate` for grouping and aggregation

### 4. Specify Types When Possible

Some Pyroid functions accept type information to avoid type inference, which can improve performance. For example, when reading CSV files, specify a schema:

```python
schema = {
    'id': 'int',
    'value': 'float',
    'flag': 'bool'
}
data = pyroid.parallel_read_csv(files, schema)
```

### 5. Be Mindful of Memory Usage

While Pyroid is generally more memory-efficient than pure Python, it still needs to load data into memory. For very large datasets, consider processing data in chunks.

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
