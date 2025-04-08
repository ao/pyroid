# Getting Started with Pyroid

This guide will help you get started with Pyroid, a high-performance Rust-powered library for Python.

## Installation

You can install Pyroid using pip:

```bash
pip install pyroid
```

For development installation:

```bash
git clone https://github.com/ao/pyroid.git
cd pyroid
pip install -e .
```

## Basic Usage

### Importing Pyroid

```python
import pyroid
```

### Math Operations

```python
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

### Data Operations

```python
# Collection operations
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = pyroid.data.filter(numbers, lambda x: x % 2 == 0)
squared = pyroid.data.map(numbers, lambda x: x * x)
sum_result = pyroid.data.reduce(numbers, lambda x, y: x + y)
sorted_list = pyroid.data.sort([5, 2, 8, 1, 9, 3], reverse=True)

print(f"Even numbers: {even_numbers}")
print(f"Squared numbers: {squared}")
print(f"Sum: {sum_result}")
print(f"Sorted (descending): {sorted_list}")

# DataFrame operations
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

### File I/O Operations

```python
# Read a file
content = pyroid.io.read_file("example.txt")
print(f"File content length: {len(content)}")

# Write a file
pyroid.io.write_file("output.txt", "Hello, world!")

# Read multiple files
files = ["file1.txt", "file2.txt", "file3.txt"]
contents = pyroid.io.read_files(files)
print(f"Read multiple files: {contents}")
```

### Network Operations

```python
# Make a GET request
response = pyroid.io.get("https://example.com")
print(f"HTTP GET response length: {len(response)}")
```

### Async Operations

```python
import asyncio

async def main():
    # Async sleep
    print("Sleeping for 0.1 seconds...")
    await pyroid.io.sleep(0.1)
    print("Awake!")
    
    # Async file operations
    content = await pyroid.io.read_file_async("example.txt")
    print(f"File content: {content}")

# Run the async main function
asyncio.run(main())
```

### Image Processing

```python
# Create a new image (width, height, channels)
img = pyroid.image.basic.create_image(100, 100, 3)

# Set some pixels
for x in range(50):
    for y in range(50):
        img.set_pixel(x, y, [255, 0, 0])  # Red square

# Apply operations
grayscale_img = img.to_grayscale()
resized_img = img.resize(200, 200)
blurred_img = img.blur(2)
brightened_img = img.adjust_brightness(1.5)

# Get image data
width = img.width
height = img.height
channels = img.channels
```

### Machine Learning

```python
# K-means clustering
data = [
    [1.0, 2.0], [1.5, 1.8], [5.0, 8.0],
    [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]
]
kmeans_result = pyroid.ml.basic.kmeans(data, k=2)
print(f"K-means centroids: {kmeans_result['centroids']}")
print(f"K-means clusters: {kmeans_result['clusters']}")

# Linear regression
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

## Configuration

Pyroid provides a configuration system that allows you to customize its behavior:

```python
# Create a config object
config = pyroid.core.Config()

# Set configuration
config.set("parallel", True)
config.set("chunk_size", 1000)

# Get configuration
parallel = config.get("parallel")
chunk_size = config.get("chunk_size")

# Use context manager for temporary configuration
with pyroid.config(parallel=False, chunk_size=500):
    # Operations here will use this configuration
    pass
```

## Next Steps

Now that you're familiar with the basics of Pyroid, you can explore the following resources:

- [API Reference](../api/index.md) for detailed documentation of all Pyroid functions and classes
- [Examples](../../examples/) for practical examples of using Pyroid
- [Performance Guide](./performance.md) for tips on optimizing performance with Pyroid
