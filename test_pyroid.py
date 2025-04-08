#!/usr/bin/env python3
"""
Test script for Pyroid
"""

import sys
import os

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

try:
    import pyroid
    print(f"Successfully imported pyroid version {pyroid.__version__}")
except ImportError as e:
    print(f"Failed to import pyroid: {e}")
    sys.exit(1)

# Test vector operations
try:
    v1 = pyroid.math.Vector([1, 2, 3])
    v2 = pyroid.math.Vector([4, 5, 6])
    v3 = v1 + v2
    print(f"Vector sum: {v3}")
    print(f"Dot product: {v1.dot(v2)}")
except Exception as e:
    print(f"Vector operations error: {e}")

# Test matrix operations
try:
    m1 = pyroid.math.Matrix([[1, 2], [3, 4]])
    m2 = pyroid.math.Matrix([[5, 6], [7, 8]])
    m3 = m1 * m2
    print(f"Matrix product: {m3}")
except Exception as e:
    print(f"Matrix operations error: {e}")

# Test statistical functions
try:
    numbers = [1, 2, 3, 4, 5]
    mean = pyroid.math.stats.mean(numbers)
    median = pyroid.math.stats.median(numbers)
    std_dev = pyroid.math.stats.calc_std(numbers)
    print(f"Mean: {mean}, Median: {median}, StdDev: {std_dev}")
except Exception as e:
    print(f"Statistical functions error: {e}")

# Test image operations
try:
    img = pyroid.image.basic.create_image(100, 100, 3)
    for x in range(50):
        for y in range(50):
            img.set_pixel(x, y, [255, 0, 0])  # Red square
    grayscale_img = img.to_grayscale()
    print(f"Created image: {img.width}x{img.height} with {img.channels} channels")
    print(f"Grayscale image: {grayscale_img.width}x{grayscale_img.height} with {grayscale_img.channels} channels")
except Exception as e:
    print(f"Image operations error: {e}")

# Test ML operations
try:
    data = [
        [1.0, 2.0], [1.5, 1.8], [5.0, 8.0],
        [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]
    ]
    result = pyroid.ml.basic.kmeans(data, k=2)
    print(f"K-means centroids: {result['centroids']}")
    print(f"K-means clusters: {result['clusters']}")
except Exception as e:
    print(f"ML operations error: {e}")

print("All tests completed.")