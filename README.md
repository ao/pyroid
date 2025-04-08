# 📌 Pyroid: Python on Rust-Powered Steroids

⚡ Blazing fast Rust-powered utilities to eliminate Python's performance bottlenecks.

## 🔹 Why Pyroid?

- ✅ **Rust-powered acceleration** for CPU-heavy tasks
- ✅ **Simplified architecture** with minimal dependencies
- ✅ **Domain-driven design** for better organization
- ✅ **Easy Python imports**—just `pip install pyroid`
- ✅ **Modular toolkit** with optional features

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

- **Simplified Architecture**: Minimal external dependencies for better maintainability
- **Domain-Driven Design**: Organized by functionality domains
- **Pythonic API**: Easy to use from Python with familiar interfaces
- **Memory Efficiency**: Optimized memory usage for large datasets
- **Cross-Platform**: Works on Windows, macOS, and Linux

### Module Overview

| Module | Description | Key Functions |
|--------|-------------|--------------|
| Math | Numerical computations | `vector_operations`, `matrix_operations`, `statistics` |
| String | Text processing | `reverse`, `base64_encode`, `base64_decode` |
| Data | Collection and DataFrame operations | `filter`, `map`, `reduce`, `dataframe_apply` |
| I/O | File and network operations | `read_file`, `write_file`, `http_get`, `http_post` |
| Image | Basic image manipulation | `create_image`, `to_grayscale`, `resize`, `blur` |
| ML | Basic machine learning | `kmeans`, `linear_regression`, `normalize`, `distance_matrix` |

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
reversed_text = pyroid.string.reverse(text)
uppercase = pyroid.string.to_uppercase(text)
lowercase = pyroid.string.to_lowercase(text)

# Base64 encoding/decoding
encoded = pyroid.string.base64_encode(text)
decoded = pyroid.string.base64_decode(encoded)
print(f"Original: {text}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
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

# Create a DataFrame object
dataframe = pyroid.data.dataframe.PyDataFrame(df)

# Apply a function to each column
def square(x):
    return [val * val for val in x]

result = dataframe.apply(square)
print(f"DataFrame shape: {dataframe.shape}")
print(f"DataFrame columns: {dataframe.columns}")
print(f"Result: {result.to_dict()}")
```

### Collection Operations

```python
import pyroid

# Filter a list
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = pyroid.data.collections.filter(numbers, lambda x: x % 2 == 0)
print(f"Even numbers: {even_numbers}")

# Map a function over a list
squared = pyroid.data.collections.map(numbers, lambda x: x * x)
print(f"Squared numbers: {squared}")

# Reduce a list
sum_result = pyroid.data.collections.reduce(numbers, lambda x, y: x + y)
print(f"Sum: {sum_result}")

# Sort a list
unsorted = [5, 2, 8, 1, 9, 3]
sorted_list = pyroid.data.collections.sort(unsorted)
print(f"Sorted: {sorted_list}")
```

### File I/O Operations

```python
import pyroid

# Read a file
content = pyroid.io.file.read_file("example.txt")
print(f"File content length: {len(content)}")

# Write a file
pyroid.io.file.write_file("output.txt", b"Hello, world!")

# Read multiple files
files = ["file1.txt", "file2.txt", "file3.txt"]
contents = pyroid.io.file.read_files(files)
for file, content in contents.items():
    print(f"{file}: {len(content)} bytes")
```

### Network Operations

```python
import pyroid

# Make a GET request
response = pyroid.io.network.get("https://example.com")
print(f"Status: {response['status']}")
print(f"Body length: {len(response['body'])}")

# Make a POST request with JSON data
data = {"name": "John", "age": 30}
response = pyroid.io.network.post("https://example.com/api", json=data)
print(f"Status: {response['status']}")
```

### Async Operations

```python
import asyncio
import pyroid

async def main():
    # Async sleep
    print("Sleeping for 1 second...")
    await pyroid.io.async_io.sleep(1.0)
    print("Awake!")
    
    # Async file operations
    await pyroid.io.async_io.write_file_async("async_test.txt", b"Hello, async world!")
    content = await pyroid.io.async_io.read_file_async("async_test.txt")
    print(f"File content: {content}")
    
    # Async HTTP requests
    response = await pyroid.io.async_io.http_get_async("https://example.com")
    print(f"Response length: {len(response)}")
    
    # Async HTTP POST
    data = {"name": "John", "age": 30}
    response = await pyroid.io.async_io.http_post_async("https://example.com/api", json=data)
    print(f"Response length: {len(response)}")

# Run the async main function
asyncio.run(main())
```

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

## 📊 Performance Considerations

While Pyroid has been simplified to reduce dependencies, it still offers performance improvements over pure Python:

- **Math operations**: Optimized vector and matrix operations
- **String processing**: Efficient string manipulation and base64 encoding/decoding
- **Data operations**: Improved collection operations and DataFrame handling
- **I/O operations**: Efficient file and network operations
- **Image processing**: Basic image manipulation without external dependencies
- **Machine learning**: Simple ML algorithms implemented in pure Rust

Note that some advanced parallel processing features have been simplified to improve maintainability. For extremely performance-critical applications, you may need to enable specific optimizations.

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
