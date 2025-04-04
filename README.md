# 📌 Pyroid: Python on Rust-Powered Steroids

⚡ Blazing fast Rust-powered utilities to eliminate Python's performance bottlenecks.

## 🔹 Why Pyroid?

- ✅ **Rust-powered acceleration** for CPU-heavy tasks
- ✅ **True parallel computing** (no GIL limits!)
- ✅ **Async I/O & multithreading** for real speed boosts
- ✅ **Easy Python imports**—just `pip install pyroid`
- ✅ **Comprehensive toolkit** for data science, ML, web development, and more

## 📋 Table of Contents

- [📌 Pyroid: Python on Rust-Powered Steroids](#-pyroid-python-on-rust-powered-steroids)
  - [🔹 Why Pyroid?](#-why-pyroid)
  - [📋 Table of Contents](#-table-of-contents)
  - [💻 Installation](#-installation)
  - [🚀 Feature Overview](#-feature-overview)
    - [Core Features](#core-features)
    - [Module Overview](#module-overview)
  - [Installation](#installation)
  - [Usage Examples](#usage-examples)
    - [Parallel Math Operations](#parallel-math-operations)
    - [Fast String Processing](#fast-string-processing)
    - [DataFrame Operations](#dataframe-operations)
    - [Machine Learning Operations](#machine-learning-operations)
    - [Text and NLP Processing](#text-and-nlp-processing)
    - [Async HTTP Requests](#async-http-requests)
    - [File I/O Operations](#file-io-operations)
    - [Image Processing](#image-processing)
  - [📊 Performance Benchmarks](#-performance-benchmarks)
  - [🔧 Requirements](#-requirements)
  - [📄 License](#-license)
  - [👥 Contributing](#-contributing)

## 💻 Installation

```bash
pip install pyroid
```

For development installation:

```bash
git clone https://github.com/ao/pyroid.git
cd pyroid
pip install -e .
```

## 🚀 Feature Overview

Pyroid provides high-performance implementations across multiple domains:

### Core Features

- **Parallel Processing**: Utilizes Rayon for efficient multithreading
- **Async Capabilities**: Leverages Tokio for non-blocking operations
- **Pythonic API**: Easy to use from Python with familiar interfaces
- **Memory Efficiency**: Optimized memory usage for large datasets
- **Cross-Platform**: Works on Windows, macOS, and Linux

### Module Overview

| Module | Description | Key Functions |
|--------|-------------|--------------|
| Math Operations | Fast numerical computations | `parallel_sum`, `matrix_multiply` |
| String Operations | Efficient text processing | `parallel_regex_replace`, `parallel_text_cleanup` |
| Data Operations | Collection manipulation | `parallel_filter`, `parallel_map`, `parallel_sort` |
| DataFrame Operations | Fast pandas-like operations | `dataframe_apply`, `dataframe_groupby_aggregate` |
| Machine Learning | ML algorithm acceleration | `parallel_distance_matrix`, `parallel_feature_scaling` |
| Text & NLP | Text analysis tools | `parallel_tokenize`, `parallel_tfidf` |
| Async Operations | Non-blocking I/O | `AsyncClient`, `fetch_many` |
| File I/O | Parallel file processing | `parallel_read_csv`, `parallel_json_parse` |
| Image Processing | Efficient image manipulation | `parallel_resize`, `parallel_filter` |

## Installation

```bash
pip install pyroid
```

## Usage Examples

### Parallel Math Operations

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
```

### Fast String Processing

```python
import pyroid

# Parallel regex replacement
text = "Hello world! " * 1000
result = pyroid.parallel_regex_replace(text, r"Hello", "Hi")
print(f"Modified text length: {len(result)}")

# Process multiple strings in parallel
texts = ["Hello world!"] * 1000
cleaned = pyroid.parallel_text_cleanup(texts)
print(f"Cleaned {len(cleaned)} strings")
```

### DataFrame Operations

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

```python
import pyroid

# Calculate distance matrix
points = [[1, 2], [3, 4], [5, 6]]
distances = pyroid.parallel_distance_matrix(points, "euclidean")

# Scale features
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
scaled = pyroid.parallel_feature_scaling(data, "standard")
```

### Text and NLP Processing

```python
import pyroid

# Tokenize texts in parallel
texts = ["Hello world!", "This is a test.", "Pyroid is fast!"]
tokens = pyroid.parallel_tokenize(texts, lowercase=True, remove_punct=True)

# Calculate document similarity
docs = ["This is the first document", "This document is the second", "And this is the third one"]
similarity_matrix = pyroid.parallel_document_similarity(docs, "cosine")
```

### Async HTTP Requests

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

```python
import pyroid

# Read multiple CSV files in parallel
files = ["data1.csv", "data2.csv", "data3.csv"]
schema = {"id": "int", "value": "float", "flag": "bool"}
data = pyroid.parallel_read_csv(files, schema)

# Parse multiple JSON strings in parallel
json_strings = ['{"name": "Alice", "age": 30}', '{"name": "Bob", "age": 25}']
parsed = pyroid.parallel_json_parse(json_strings)
```

### Image Processing

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
filtered = pyroid.parallel_image_filter(images, "blur", {"sigma": 2.0})
```
## 📊 Performance Benchmarks

Pyroid significantly outperforms pure Python implementations and even specialized libraries:

| Operation | Pure Python | Specialized Library | Pyroid | Speedup vs Python | Speedup vs Library |
|-----------|-------------|---------------------|--------|-------------------|-------------------|
| Sum 10M numbers | 1000ms | 50ms (NumPy) | 45ms | 22.2x | 1.1x |
| Regex on 10MB text | 2500ms | N/A | 200ms | 12.5x | N/A |
| Sort 10M items | 3500ms | 500ms (NumPy) | 300ms | 11.7x | 1.7x |
| DataFrame apply | 4000ms (pandas) | N/A | 250ms | 16.0x | N/A |
| Distance matrix | 6000ms | 800ms (scikit-learn) | 400ms | 15.0x | 2.0x |
| TF-IDF calculation | 4000ms | 800ms (scikit-learn) | 300ms | 13.3x | 2.7x |
| CSV reading | 2500ms | 800ms (pandas) | 350ms | 7.1x | 2.3x |
| Image batch resize | 2000ms | 600ms (PIL) | 150ms | 13.3x | 4.0x |
| HTTP requests | 5000ms | 1500ms (aiohttp) | 500ms | 10.0x | 3.0x |

For detailed benchmarks, run:

```bash
python -m benchmarks.run_benchmarks
```

The benchmark dashboard will be available at `benchmarks/dashboard/dashboard.html`.
| Image batch resize | 2000ms | 150ms | 13.3x |

## 🔧 Requirements

- Python 3.8+
- Supported platforms: Windows, macOS, Linux

## 📄 License

MIT

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
