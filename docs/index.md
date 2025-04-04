# Pyroid Documentation

Welcome to the Pyroid documentation! Pyroid is a high-performance Rust-powered library for Python that accelerates common operations and eliminates performance bottlenecks.

## 📚 Documentation Sections

| Section | Description |
|---------|-------------|
| [Guides](./guides/index.md) | Comprehensive guides on using Pyroid effectively |
| [API Reference](./api/index.md) | Detailed documentation of all Pyroid functions and classes |
| [Examples](../examples/) | Example code demonstrating various Pyroid features |
| [Benchmarks](../benchmarks/) | Performance benchmarks comparing Pyroid to other libraries |

## 🚀 Quick Start

### Installation

```bash
pip install pyroid
```

### Basic Usage

```python
import pyroid

# Parallel sum of a large list
numbers = list(range(1_000_000))
result = pyroid.parallel_sum(numbers)
print(f"Sum: {result}")

# Parallel regex replacement
text = "Hello world! " * 1000
result = pyroid.parallel_regex_replace(text, r"Hello", "Hi")
print(f"Modified text length: {len(result)}")

# Parallel filter
data = list(range(1_000_000))
filtered = pyroid.parallel_filter(data, lambda x: x % 2 == 0)
print(f"Filtered {len(filtered)} items")
```

For more detailed examples, see the [Getting Started Guide](./guides/getting_started.md).

## 🔍 Key Features

Pyroid provides high-performance implementations across multiple domains:

### Math Operations

Fast numerical computations that outperform Python's built-in functions and even NumPy for many operations.

```python
# Matrix multiplication
matrix_a = [[1, 2], [3, 4]]
matrix_b = [[5, 6], [7, 8]]
result = pyroid.matrix_multiply(matrix_a, matrix_b)
```

### String Operations

Efficient text processing with parallel implementations of common string operations.

```python
# Process multiple strings in parallel
texts = ["Hello world!", "This is a test.", "Pyroid is fast!"] * 1000
cleaned = pyroid.parallel_text_cleanup(texts)
```

### Data Operations

High-performance collection manipulation functions.

```python
# Parallel sort
data = [5, 3, 1, 4, 2] * 1000000
sorted_data = pyroid.parallel_sort(data, None, False)
```

### DataFrame Operations

Fast pandas-like operations for data manipulation.

```python
# Group by and aggregate
df = {
    'category': ['A', 'B', 'A', 'B', 'C'] * 200000,
    'value': [10, 20, 15, 25, 30] * 200000
}
agg_dict = {'value': 'mean'}
result = pyroid.dataframe_groupby_aggregate(df, ['category'], agg_dict)
```

### Machine Learning Operations

Accelerated machine learning primitives.

```python
# Calculate distance matrix
points = [[1, 2], [3, 4], [5, 6]] * 1000
distances = pyroid.parallel_distance_matrix(points, "euclidean")
```

### Text and NLP Operations

Efficient text analysis tools.

```python
# Calculate document similarity
docs = ["This is the first document", "This document is the second document"]
similarity_matrix = pyroid.parallel_document_similarity(docs, "cosine")
```

### Async Operations

Non-blocking I/O operations for improved throughput.

```python
import asyncio

async def main():
    client = pyroid.AsyncClient()
    urls = ["https://example.com", "https://google.com", "https://github.com"]
    responses = await client.fetch_many(urls, concurrency=3)

asyncio.run(main())
```

### File I/O Operations

Parallel file processing for improved throughput.

```python
# Read multiple CSV files in parallel
files = ["data1.csv", "data2.csv", "data3.csv"]
schema = {"id": "int", "value": "float", "flag": "bool"}
data = pyroid.parallel_read_csv(files, schema)
```

### Image Processing Operations

Efficient image manipulation operations.

```python
# Resize images in parallel
images = [image1, image2, image3]  # image data as bytes
resized = pyroid.parallel_resize(images, (800, 600), "lanczos3")
```

## 📊 Performance

Pyroid significantly outperforms pure Python implementations:

| Operation | Pure Python | Pyroid | Speedup |
|-----------|-------------|--------|---------|
| Sum 10M numbers | 1000ms | 50ms | 20x |
| Regex on 10MB text | 2500ms | 200ms | 12.5x |
| Sort 10M items | 3500ms | 300ms | 11.7x |
| 100 HTTP requests | 5000ms | 500ms | 10x |
| DataFrame groupby | 3000ms | 200ms | 15x |
| TF-IDF calculation | 4000ms | 300ms | 13.3x |
| Image batch resize | 2000ms | 150ms | 13.3x |

For detailed benchmarks, see the [Benchmarks](../benchmarks/) directory.

## 🧩 How It Works

Pyroid achieves its performance advantages through several key mechanisms:

1. **Rust Implementation**: Core operations are implemented in Rust, a systems programming language that offers performance comparable to C/C++ with memory safety guarantees.

2. **Parallel Processing**: Pyroid uses Rayon, a data parallelism library for Rust, to automatically parallelize operations across multiple CPU cores.

3. **Efficient Memory Management**: Rust's ownership model allows for efficient memory management without the overhead of garbage collection.

4. **SIMD Instructions**: Where applicable, Pyroid leverages SIMD (Single Instruction, Multiple Data) instructions for vectorized operations.

5. **Optimized Algorithms**: Pyroid implements optimized versions of common algorithms, taking advantage of Rust's performance characteristics.

## 🔧 Requirements

- Python 3.8+
- Supported platforms: Windows, macOS, Linux

## 📄 License

Pyroid is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🤝 Getting Help

If you need help with Pyroid, you can:

- Check the [FAQ](./guides/faq.md) for answers to common questions.
- Search for similar issues on the [GitHub issue tracker](https://github.com/ao/pyroid/issues).
- Ask a question on [Stack Overflow](https://stackoverflow.com/questions/tagged/pyroid) with the `pyroid` tag.
- Join the community discussion on [Discord](https://discord.gg/pyroid).