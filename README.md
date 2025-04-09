# 📌 Pyroid: Python on Rust-Powered Steroids

⚡ Blazing fast Rust-powered utilities to eliminate Python's performance bottlenecks.

## 🔹 Why Pyroid?

- ✅ **Rust-powered acceleration** for CPU-heavy tasks
- ✅ **Simplified architecture** with minimal dependencies
- ✅ **Domain-driven design** for better organization
- ✅ **Easy Python imports**—just `pip install pyroid`
- ✅ **Modular toolkit** with optional features
- ✅ **Optimized async operations** with unified runtime
- ✅ **Zero-copy buffer protocol** for efficient memory usage
- ✅ **Parallel processing** for high-throughput workloads

## 📋 Table of Contents

- [📌 Pyroid: Python on Rust-Powered Steroids](#-pyroid-python-on-rust-powered-steroids)
  - [🔹 Why Pyroid?](#-why-pyroid)
  - [📋 Table of Contents](#-table-of-contents)
  - [💻 Installation](#-installation)
  - [🚀 Feature Overview](#-feature-overview)
    - [Core Features](#core-features)
    - [Module Overview](#module-overview)
  - [🔧 Feature Flags](#-feature-flags)
  - [Usage Examples](#usage-examples)
    - [Math Operations](#math-operations)
    - [String Processing](#string-processing)
    - [DataFrame Operations](#dataframe-operations)
    - [Collection Operations](#collection-operations)
    - [File I/O Operations](#file-io-operations)
    - [Network Operations](#network-operations)
    - [Async Operations](#async-operations)
    - [Image Processing](#image-processing)
    - [Machine Learning](#machine-learning)
  - [📊 Performance Considerations](#-performance-considerations)
  - [Building from Source](#building-from-source)
  - [Running Benchmarks and Examples](#running-benchmarks-and-examples)
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
python build_and_install.py
```

This script will:
1. Check if Rust is installed and install it if needed
2. Build the Rust code with optimizations
3. Install the Python package in development mode

## 🚀 Feature Overview

Pyroid provides high-performance implementations across multiple domains:

### Core Features

- **Simplified Architecture**: Minimal external dependencies for better maintainability
- **Domain-Driven Design**: Organized by functionality domains
- **Pythonic API**: Easy to use from Python with familiar interfaces
- **Memory Efficiency**: Optimized memory usage for large datasets
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Unified Async Runtime**: Shared Tokio runtime for all async operations
- **Zero-Copy Buffer Protocol**: Efficient memory management without copying
- **GIL-Aware Scheduling**: Optimized task scheduling with Python's GIL
- **Parallel Processing**: Efficient batch processing with adaptive sizing

### Module Overview

| Module | Description | Key Functions |
|--------|-------------|--------------|
| Math | Numerical computations | `vector_operations`, `matrix_operations`, `statistics` |
| String | Text processing | `reverse`, `base64_encode`, `base64_decode` |
| Data | Collection and DataFrame operations | `filter`, `map`, `reduce`, `dataframe_apply` |
| I/O | File and network operations | `read_file`, `write_file`, `http_get`, `http_post` |
| Image | Basic image manipulation | `create_image`, `to_grayscale`, `resize`, `blur` |
| ML | Basic machine learning | `kmeans`, `linear_regression`, `normalize`, `distance_matrix` |
| Core | Core functionality | `runtime`, `buffer`, `parallel` |

## 🔧 Feature Flags

Pyroid uses feature flags to allow selective compilation of components:

| Feature Flag | Description | Default |
|--------------|-------------|---------|
| `math` | Math operations | Enabled |
| `text` | Text processing | Enabled |
| `data` | Collection and DataFrame operations | Enabled |
| `io` | File and network operations | Enabled |
| `image` | Basic image processing | Enabled |
| `ml` | Basic machine learning | Enabled |

To compile with only specific features, modify your `Cargo.toml`:

```toml
[dependencies]
pyroid = { version = "0.1.0", default-features = false, features = ["math", "data"] }
```

## Usage Examples

### Math Operations

```python
import pyroid

# Vector operations
v1 = pyroid.math.Vector([1, 2, 3])
v2 = pyroid.math.Vector([4, 5, 6])
v3 = v1 + v2
print(f"Vector sum: {v3}")
print(f"Dot product: {v1.dot(v2)}")

# Matrix operations
m1 = pyroid.math.Matrix([[1, 2], [3, 4]])
m2 = pyroid.math.Matrix([[5, 6], [7, 8]])
m3 = m1 * m2
print(f"Matrix product: {m3}")

# Statistical functions
numbers = [1, 2, 3, 4, 5]
mean = pyroid.math.stats.mean(numbers)
median = pyroid.math.stats.median(numbers)
std_dev = pyroid.math.stats.calc_std(numbers)
print(f"Mean: {mean}, Median: {median}, StdDev: {std_dev}")
```

### String Processing

```python
import pyroid

# Basic string operations
text = "Hello, world!"
reversed_text = pyroid.text.reverse(text)
uppercase = pyroid.text.to_uppercase(text)
lowercase = pyroid.text.to_lowercase(text)

# Base64 encoding/decoding
encoded = pyroid.text.base64_encode(text)
decoded = pyroid.text.base64_decode(encoded)
print(f"Original: {text}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
```

### DataFrame Operations

```python
import pyroid

# Create a DataFrame
df = pyroid.data.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 40, 45]
})

# Apply a function to each column
result = pyroid.data.apply(df, lambda x: x * 2, axis=0)
print(f"DataFrame: {df}")
print(f"Applied function: {result}")

# Group by and aggregate
grouped = pyroid.data.groupby_aggregate(df, "age", {"name": "count"})
print(f"Grouped by age: {grouped}")
```

### Collection Operations

```python
import pyroid

# Filter a list
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = pyroid.data.filter(numbers, lambda x: x % 2 == 0)
print(f"Even numbers: {even_numbers}")

# Map a function over a list
squared = pyroid.data.map(numbers, lambda x: x * x)
print(f"Squared numbers: {squared}")

# Reduce a list
sum_result = pyroid.data.reduce(numbers, lambda x, y: x + y)
print(f"Sum: {sum_result}")

# Sort a list
unsorted = [5, 2, 8, 1, 9, 3]
sorted_list = pyroid.data.sort(unsorted, reverse=True)  # Note: supports reverse parameter
print(f"Sorted (descending): {sorted_list}")
```

### File I/O Operations

```python
import pyroid

# Read a file
content = pyroid.io.read_file("example.txt")
print(f"File content length: {len(content)}")

# Write a file
pyroid.io.write_file("output.txt", "Hello, world!")  # Note: string, not bytes

# Read multiple files
files = ["file1.txt", "file2.txt", "file3.txt"]
contents = pyroid.io.read_files(files)
print(f"Read multiple files: {contents}")
```

### Network Operations

```python
import pyroid

# Make a GET request
response = pyroid.io.get("https://example.com")
print(f"HTTP GET response length: {len(response)}")

# Note: POST requests might not be implemented in the current version
# If you need to make POST requests, check the documentation for the latest API
```

### Async Operations

```python
import asyncio
import pyroid
from pyroid.core import runtime, buffer, parallel

async def main():
    # Initialize the runtime
    runtime.init()
    print(f"Runtime initialized with {runtime.get_worker_threads()} worker threads")
    
    # Create an AsyncClient for HTTP operations
    client = pyroid.AsyncClient()
    
    # Fetch a URL asynchronously
    response = await client.fetch("https://example.com")
    print(f"Status code: {response['status']}")
    
    # Fetch multiple URLs concurrently
    urls = [f"https://example.com/{i}" for i in range(10)]
    responses = await client.fetch_many(urls, concurrency=5)
    print(f"Fetched {len(responses)} URLs")
    
    # Async file operations
    file_reader = pyroid.AsyncFileReader("example.txt")
    content = await file_reader.read_all()
    print(f"File content length: {len(content)}")
    
    # Zero-copy buffer operations
    zero_copy_buffer = buffer.ZeroCopyBuffer(1024)  # 1KB buffer
    data = zero_copy_buffer.get_data()
    # Modify data...
    zero_copy_buffer.set_data(data)
    
    # Parallel processing
    processor = parallel.BatchProcessor(batch_size=1000, adaptive=True)
    items = list(range(1000000))
    
    def process_item(x):
        return x * x
    
    results = processor.map(items, process_item)
    print(f"Processed {len(results)} items")

# Run the async main function
asyncio.run(main())
```

For a more comprehensive example, see `examples/optimized_async_example.py`.

### Image Processing

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
```

### Machine Learning

```python
import pyroid

# K-means clustering
data = [
    [1.0, 2.0], [1.5, 1.8], [5.0, 8.0],
    [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]
]
kmeans_result = pyroid.ml.basic.kmeans(data, k=2)
print(f"K-means centroids: {kmeans_result['centroids']}")
print(f"K-means clusters: {kmeans_result['clusters']}")

# Linear regression
# Note: Linear regression expects X as 2D array and y as 1D array
X = [[1, 1], [1, 2], [2, 2], [2, 3]]
y = [6, 8, 9, 11]
regression_result = pyroid.ml.basic.linear_regression(X, y)
print(f"Linear regression coefficients: {regression_result['coefficients']}")
print(f"Linear regression intercept: {regression_result['intercept']}")
print(f"Linear regression R-squared: {regression_result['r_squared']}")

# Data normalization
normalized_data = pyroid.ml.basic.normalize(data, method="min-max")
print(f"Normalized data (min-max): {normalized_data}")

# Distance matrix
distance_matrix = pyroid.ml.basic.distance_matrix(data, metric="euclidean")
print(f"Distance matrix shape: {len(distance_matrix)}x{len(distance_matrix[0])}")
```

## 📊 Performance Considerations

Pyroid offers significant performance improvements over pure Python:

- **Math operations**: Optimized vector and matrix operations
- **String processing**: Efficient string manipulation and base64 encoding/decoding
- **Data operations**: Improved collection operations and DataFrame handling
- **I/O operations**: Efficient file and network operations with async support
- **Image processing**: Basic image manipulation without external dependencies
- **Machine learning**: Simple ML algorithms implemented in pure Rust
- **Async operations**: High-performance async operations with unified runtime
- **Zero-copy buffers**: Efficient memory management without copying
- **Parallel processing**: Batch processing with adaptive sizing for optimal performance

## Building from Source

To build pyroid from source, you need:

1. **Rust** (1.70.0 or later)
2. **Python** (3.8 or later)
3. **Cargo** (comes with Rust)

The easiest way to build and install pyroid is to use the provided script:

```bash
python build_and_install.py
```

Alternatively, you can build manually:

```bash
# Build the Rust code
cargo build --release

# Install the Python package in development mode
pip install -e .
```

For performance-critical applications, consider using the following optimizations:

1. **Unified Runtime**: Initialize the runtime once at the start of your application
   ```python
   from pyroid.core import runtime
   runtime.init()
   ```

2. **Zero-Copy Buffers**: Use zero-copy buffers for large data transfers
   ```python
   from pyroid.core import buffer
   zero_copy_buffer = buffer.ZeroCopyBuffer(size)
   ```

3. **Parallel Processing**: Use batch processing for CPU-intensive operations
   ```python
   from pyroid.core import parallel
   processor = parallel.BatchProcessor(adaptive=True)
   results = processor.map(items, process_function)
   ```

4. **Concurrency Control**: Adjust concurrency levels based on your workload
   ```python
   client = pyroid.AsyncClient()
   responses = await client.fetch_many(urls, concurrency=optimal_value)
   ```

## Running Benchmarks and Examples

To run the benchmarks and see the performance improvements:

```bash
# Build and install pyroid first
python build_and_install.py

# Run the async benchmarks
python -m benchmarks.run_benchmarks --size small --suite async --no-dashboard

# Run the high-throughput benchmark
python -m benchmarks.run_benchmarks --size small --suite high-throughput --no-dashboard
```

To run the example demonstrating all the optimized features:

```bash
# Build and install pyroid first
python build_and_install.py

# Run the example
python examples/optimized_async_example.py
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
