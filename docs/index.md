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

# Vector operations
v1 = pyroid.math.Vector([1, 2, 3])
v2 = pyroid.math.Vector([4, 5, 6])
v3 = v1 + v2
print(f"Vector sum: {v3}")

# Matrix operations
m1 = pyroid.math.Matrix([[1, 2], [3, 4]])
m2 = pyroid.math.Matrix([[5, 6], [7, 8]])
m3 = m1 * m2
print(f"Matrix product: {m3}")

# Collection operations
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = pyroid.data.filter(numbers, lambda x: x % 2 == 0)
print(f"Even numbers: {even_numbers}")
```

For more detailed examples, see the [Getting Started Guide](./guides/getting_started.md).

## 🔍 Key Features

Pyroid provides high-performance implementations across multiple domains:

### Math Operations

Fast numerical computations with vector and matrix operations.

```python
# Vector operations
v1 = pyroid.math.Vector([1, 2, 3])
v2 = pyroid.math.Vector([4, 5, 6])
dot_product = v1.dot(v2)
```

### String Operations

Efficient text processing with common string operations.

```python
# Basic string operations
text = "Hello, world!"
reversed_text = pyroid.text.reverse(text)
uppercase = pyroid.text.to_uppercase(text)
```

### Data Operations

High-performance collection manipulation functions.

```python
# Filter, map, reduce operations
numbers = [1, 2, 3, 4, 5]
squared = pyroid.data.map(numbers, lambda x: x * x)
```

### DataFrame Operations

Fast data manipulation for structured data.

```python
# Create a DataFrame
df = pyroid.data.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 40, 45]
})
```

### Machine Learning Operations

Basic machine learning algorithms implemented in Rust.

```python
# K-means clustering
data = [[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]]
kmeans_result = pyroid.ml.basic.kmeans(data, k=2)
```

### Text and NLP Operations

Efficient text analysis tools.

```python
# Tokenization and n-grams
text = "Hello, World!"
tokens = pyroid.text.tokenize(text)
ngrams = pyroid.text.ngrams(text, 2)
```

### Async Operations

Non-blocking I/O operations for improved throughput.

```python
import asyncio

async def main():
    content = await pyroid.io.read_file_async("example.txt")
    print(f"File content: {content}")

asyncio.run(main())
```

### File I/O Operations

Efficient file operations.

```python
# Read and write files
content = pyroid.io.read_file("example.txt")
pyroid.io.write_file("output.txt", "Hello, world!")
```

### Image Processing Operations

Basic image manipulation operations.

```python
# Create and manipulate images
img = pyroid.image.basic.create_image(100, 100, 3)
grayscale_img = img.to_grayscale()
```

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