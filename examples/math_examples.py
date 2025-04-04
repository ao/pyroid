#!/usr/bin/env python3
"""
Math operation examples for pyroid.

This script demonstrates the mathematical capabilities of pyroid.
"""

import time
import random
import numpy as np
import pyroid

def benchmark(func, *args, **kwargs):
    """Simple benchmarking function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")
    return result

def python_sum(numbers):
    """Pure Python implementation of sum."""
    return sum(numbers)

def numpy_sum(numbers):
    """NumPy implementation of sum."""
    return np.sum(numbers)

def main():
    print("pyroid Math Operations Examples")
    print("===============================")
    
    # Example 1: Parallel sum
    print("\n1. Parallel Sum")
    numbers = [random.random() for _ in range(10_000_000)]
    
    print("\nSumming 10 million numbers using different methods:")
    
    print("\nPure Python sum:")
    py_result = benchmark(python_sum, numbers)
    
    print("\nNumPy sum:")
    np_result = benchmark(numpy_sum, numbers)
    
    print("\npyroid parallel_sum:")
    rust_result = benchmark(pyroid.parallel_sum, numbers)
    
    print("\nResults:")
    print(f"Python: {py_result}")
    print(f"NumPy:  {np_result}")
    print(f"pyroid: {rust_result}")
    
    # Example 2: Parallel product
    print("\n2. Parallel Product")
    numbers = [1.0 + random.random() * 0.01 for _ in range(1_000_000)]
    
    print("\nCalculating product of 1 million numbers:")
    result = benchmark(pyroid.parallel_product, numbers)
    print(f"Result: {result}")
    
    # Example 3: Parallel mean and standard deviation
    print("\n3. Statistical Functions")
    numbers = [random.gauss(0, 1) for _ in range(5_000_000)]
    
    print("\nCalculating mean of 5 million numbers:")
    mean = benchmark(pyroid.parallel_mean, numbers)
    print(f"Mean: {mean}")
    
    print("\nCalculating standard deviation of 5 million numbers:")
    std = benchmark(pyroid.parallel_std, numbers, 1)  # Using ddof=1 for sample std
    print(f"Standard deviation: {std}")
    
    # Compare with NumPy
    print("\nNumPy mean:")
    np_mean = benchmark(np.mean, numbers)
    print(f"Mean: {np_mean}")
    
    print("\nNumPy std:")
    np_std = benchmark(np.std, numbers, ddof=1)
    print(f"Standard deviation: {np_std}")
    
    # Example 4: Parallel apply
    print("\n4. Parallel Apply")
    numbers = [random.random() for _ in range(5_000_000)]
    
    print("\nApplying sqrt to 5 million numbers:")
    results = benchmark(pyroid.parallel_apply, numbers, "sqrt")
    print(f"First 5 results: {results[:5]}")
    
    # Example 5: Matrix multiplication
    print("\n5. Matrix Multiplication")
    size = 500
    a = [[random.random() for _ in range(size)] for _ in range(size)]
    b = [[random.random() for _ in range(size)] for _ in range(size)]
    
    print(f"\nMultiplying two {size}x{size} matrices:")
    result = benchmark(pyroid.matrix_multiply, a, b)
    print(f"Result shape: {len(result)}x{len(result[0])}")
    print(f"Sample value at [0][0]: {result[0][0]}")

if __name__ == "__main__":
    main()