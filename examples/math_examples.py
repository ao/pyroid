#!/usr/bin/env python3
"""
Math operation examples for pyroid.

This script demonstrates the mathematical capabilities of pyroid.
"""

import time
import pyroid

def benchmark(func, *args, **kwargs):
    """Simple benchmarking function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")
    return result

def main():
    print("Pyroid Math Operations Examples")
    print("==============================")
    
    # Example 1: Vector operations
    print("\n1. Vector Operations")
    
    # Create vectors
    v1 = pyroid.math.Vector([1, 2, 3])
    v2 = pyroid.math.Vector([4, 5, 6])
    
    # Vector addition
    v3 = v1 + v2
    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}")
    print(f"Vector sum: {v3}")
    
    # Dot product
    dot_product = v1.dot(v2)
    print(f"Dot product: {dot_product}")
    
    # Try vector operations that might not be implemented
    try:
        # Vector subtraction
        v4 = v1 - v2
        print(f"Vector subtraction: {v4}")
    except Exception as e:
        print(f"Vector subtraction not implemented: {e}")
        # Manual subtraction
        v4 = pyroid.math.Vector([v1.values[i] - v2.values[i] for i in range(len(v1.values))])
        print(f"Vector subtraction (manual): {v4}")
    
    try:
        # Vector scalar multiplication
        v5 = v1 * 2
        print(f"Vector scalar multiplication: {v5}")
    except Exception as e:
        print(f"Vector scalar multiplication not implemented: {e}")
        # Manual multiplication
        v5 = pyroid.math.Vector([x * 2 for x in v1.values])
        print(f"Vector scalar multiplication (manual): {v5}")
    
    # Manual division
    v6 = pyroid.math.Vector([x / 2 for x in v1.values])
    print(f"Vector scalar division (manual): {v6}")
    
    # Try vector magnitude and normalization
    try:
        magnitude = v1.magnitude()
        print(f"Vector magnitude: {magnitude}")
    except Exception as e:
        print(f"Vector magnitude not implemented: {e}")
    
    try:
        normalized = v1.normalize()
        print(f"Vector normalized: {normalized}")
    except Exception as e:
        print(f"Vector normalization not implemented: {e}")
    
    # Example 2: Matrix operations
    print("\n2. Matrix Operations")
    
    # Create matrices
    m1 = pyroid.math.Matrix([[1, 2], [3, 4]])
    m2 = pyroid.math.Matrix([[5, 6], [7, 8]])
    
    print(f"Matrix 1: {m1}")
    print(f"Matrix 2: {m2}")
    
    # Matrix operations
    try:
        # Matrix addition
        m3 = m1 + m2
        print(f"Matrix addition: {m3}")
    except Exception as e:
        print(f"Matrix addition not implemented: {e}")
        # Manual addition
        m3 = pyroid.math.Matrix([[m1.values[i][j] + m2.values[i][j] for j in range(len(m1.values[0]))] for i in range(len(m1.values))])
        print(f"Matrix addition (manual): {m3}")
    
    try:
        # Matrix subtraction
        m4 = m1 - m2
        print(f"Matrix subtraction: {m4}")
    except Exception as e:
        print(f"Matrix subtraction not implemented: {e}")
        # Manual subtraction
        m4 = pyroid.math.Matrix([[m1.values[i][j] - m2.values[i][j] for j in range(len(m1.values[0]))] for i in range(len(m1.values))])
        print(f"Matrix subtraction (manual): {m4}")
    
    # Matrix multiplication
    m5 = m1 * m2
    print(f"Matrix multiplication: {m5}")
    
    # Try matrix transpose and determinant
    try:
        transposed = m1.transpose()
        print(f"Matrix transpose: {transposed}")
    except Exception as e:
        print(f"Matrix transpose not implemented: {e}")
    
    try:
        det = m1.determinant()
        print(f"Matrix determinant: {det}")
    except Exception as e:
        print(f"Matrix determinant not implemented: {e}")
    
    # Example 3: Statistical functions
    print("\n3. Statistical Functions")
    
    # Create a list of numbers
    numbers = [1, 2, 3, 4, 5]
    print(f"Numbers: {numbers}")
    
    # Calculate statistics
    mean = pyroid.math.stats.mean(numbers)
    median = pyroid.math.stats.median(numbers)
    std_dev = pyroid.math.stats.calc_std(numbers)
    variance = pyroid.math.stats.variance(numbers)
    
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Standard deviation: {std_dev}")
    print(f"Variance: {variance}")
    
    # Correlation
    x = [1, 2, 3, 4, 5]
    y = [5, 4, 3, 2, 1]
    correlation = pyroid.math.stats.correlation(x, y)
    print(f"Correlation between {x} and {y}: {correlation}")
    
    # Descriptive statistics
    stats = pyroid.math.stats.describe(numbers)
    print(f"Descriptive statistics: {stats}")

if __name__ == "__main__":
    main()