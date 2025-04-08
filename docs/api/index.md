# Pyroid API Reference

This section provides detailed documentation for all Pyroid modules, classes, and functions.

## Core Module

The core module provides configuration and utility functions for Pyroid.

```python
import pyroid.core

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

## Math Module

The math module provides numerical computation functions and classes.

- [Vector Operations](./vector_ops.md)
- [Matrix Operations](./matrix_ops.md)
- [Statistical Functions](./stats_ops.md)

```python
import pyroid.math

# Vector operations
v1 = pyroid.math.Vector([1, 2, 3])
v2 = pyroid.math.Vector([4, 5, 6])
v3 = v1 + v2
dot_product = v1.dot(v2)

# Matrix operations
m1 = pyroid.math.Matrix([[1, 2], [3, 4]])
m2 = pyroid.math.Matrix([[5, 6], [7, 8]])
m3 = m1 * m2

# Statistical functions
numbers = [1, 2, 3, 4, 5]
mean = pyroid.math.stats.mean(numbers)
median = pyroid.math.stats.median(numbers)
std_dev = pyroid.math.stats.calc_std(numbers)
variance = pyroid.math.stats.variance(numbers)
```

## Text Module

The text module provides string processing functions.

- [Basic String Operations](./text_ops.md)
- [Encoding/Decoding](./encoding_ops.md)
- [NLP Functions](./nlp_ops.md)

```python
import pyroid.text

# Basic operations
text = "Hello, World!"
reversed_text = pyroid.text.reverse(text)
upper = pyroid.text.to_uppercase(text)
lower = pyroid.text.to_lowercase(text)

# Split and join
words = pyroid.text.split(text, " ")
joined = pyroid.text.join(words, "-")

# Base64 encoding/decoding
encoded = pyroid.text.base64_encode(text)
decoded = pyroid.text.base64_decode(encoded)

# Regex replace
replaced = pyroid.text.regex_replace(text, r"World", "Python")

# NLP operations
tokens = pyroid.text.tokenize(text)
ngrams = pyroid.text.ngrams(text, 2)
```

## String Module

The string module is an alias for the text module.

```python
import pyroid.string

# Same functions as text module
upper = pyroid.string.to_uppercase("Hello, World!")
lower = pyroid.string.to_lowercase("Hello, World!")
```

## Data Module

The data module provides collection and DataFrame operations.

- [Collection Operations](./collection_ops.md)
- [DataFrame Operations](./dataframe_ops.md)

```python
import pyroid.data

# Collection operations
items = [1, 2, 3, 4, 5]
filtered = pyroid.data.filter(items, lambda x: x % 2 == 0)
mapped = pyroid.data.map(items, lambda x: x * 2)
reduced = pyroid.data.reduce(items, lambda x, y: x + y)
sorted_items = pyroid.data.sort(items, reverse=True)

# DataFrame operations
df = pyroid.data.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "age": [25, 30, 35, 40, 45]
})

# Apply function
result = pyroid.data.apply(df, lambda x: x * 2, axis=0)

# Group by and aggregate
grouped = pyroid.data.groupby_aggregate(df, "age", {"name": "count"})
```

## I/O Module

The I/O module provides file and network operations.

- [File Operations](./file_ops.md)
- [Network Operations](./network_ops.md)
- [Async Operations](./async_ops.md)

```python
import pyroid.io
import asyncio

# File operations
content = pyroid.io.read_file("example.txt")
pyroid.io.write_file("output.txt", "Hello, world!")
files = pyroid.io.read_files(["file1.txt", "file2.txt"])

# Network operations
response = pyroid.io.get("https://example.com")

# Async operations
async def main():
    await pyroid.io.sleep(0.1)
    content = await pyroid.io.read_file_async("example.txt")
    return content

asyncio.run(main())
```

## Image Module

The image module provides basic image manipulation functions.

- [Image Creation](./image_creation.md)
- [Image Manipulation](./image_manipulation.md)

```python
import pyroid.image.basic

# Create an image
img = pyroid.image.basic.create_image(100, 100, 3)

# Set pixels
for x in range(50):
    for y in range(50):
        img.set_pixel(x, y, [255, 0, 0])  # Red square

# Get a pixel
pixel = img.get_pixel(25, 25)

# Image transformations
grayscale_img = img.to_grayscale()
resized_img = img.resize(50, 50)
blurred_img = img.blur(2)
brightened_img = img.adjust_brightness(1.5)
```

## Machine Learning Module

The ML module provides basic machine learning algorithms.

- [Clustering](./clustering.md)
- [Regression](./regression.md)
- [Data Preprocessing](./preprocessing.md)

```python
import pyroid.ml.basic

# K-means clustering
data = [
    [1.0, 2.0], [1.5, 1.8], [5.0, 8.0],
    [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]
]
kmeans_result = pyroid.ml.basic.kmeans(data, k=2)

# Linear regression
X = [[1, 1], [1, 2], [2, 2], [2, 3]]
y = [6, 8, 9, 11]
regression_result = pyroid.ml.basic.linear_regression(X, y)

# Data normalization
normalized_data = pyroid.ml.basic.normalize(data, method="min-max")

# Distance matrix
distance_matrix = pyroid.ml.basic.distance_matrix(data, metric="euclidean")