"""
Math operation benchmarks for Pyroid.

This module provides benchmarks for comparing Pyroid's math operations with
pure Python and NumPy implementations.
"""

import random
import time
import numpy as np

try:
    import pyroid
except ImportError:
    print("Warning: pyroid not found. Math benchmarks will not run correctly.")

from ..core.benchmark import Benchmark
from ..core.reporter import BenchmarkReporter


def run_math_benchmarks(sizes=[1_000, 10_000, 100_000, 1_000_000, 10_000_000]):
    """Run math benchmarks.
    
    Args:
        sizes: List of dataset sizes to benchmark.
        
    Returns:
        List of Benchmark objects with results.
    """
    results = []
    
    for size in sizes:
        # Generate test data
        numbers = [random.random() for _ in range(size)]
        
        # Sum benchmark
        sum_benchmark = Benchmark(f"Sum {size:,} numbers", f"Sum {size:,} random floating-point numbers")
        
        # Set appropriate timeouts based on dataset size
        python_timeout = 2 if size <= 100_000 else 10
        numpy_timeout = 5 if size <= 1_000_000 else 10
        pyroid_timeout = 10  # Pyroid should be fast, but set a reasonable timeout
        
        sum_benchmark.run_test("Python sum", "Python", sum, python_timeout, numbers)
        
        if size <= 1_000_000:  # NumPy might struggle with very large arrays
            sum_benchmark.run_test("NumPy sum", "NumPy", np.sum, numpy_timeout, numbers)
            
        sum_benchmark.run_test("pyroid sum", "pyroid", pyroid.math.sum, pyroid_timeout, numbers)
        
        BenchmarkReporter.print_results(sum_benchmark)
        results.append(sum_benchmark)
        
        # Mean benchmark
        if size <= 1_000_000:  # Skip very large datasets for mean
            mean_benchmark = Benchmark(f"Mean {size:,} numbers", f"Calculate mean of {size:,} random floating-point numbers")
            
            # Python mean
            def python_mean(numbers):
                return sum(numbers) / len(numbers)
            
            mean_benchmark.run_test("Python mean", "Python", python_mean, python_timeout, numbers)
            mean_benchmark.run_test("NumPy mean", "NumPy", np.mean, numpy_timeout, numbers)
            mean_benchmark.run_test("pyroid mean", "pyroid", pyroid.math.mean, pyroid_timeout, numbers)
            
            BenchmarkReporter.print_results(mean_benchmark)
            results.append(mean_benchmark)
        
        # Standard deviation benchmark
        if size <= 1_000_000:  # Skip very large datasets for std
            std_benchmark = Benchmark(f"Std {size:,} numbers", f"Calculate standard deviation of {size:,} random floating-point numbers")
            
            # Python std
            def python_std(numbers):
                mean = sum(numbers) / len(numbers)
                return (sum((x - mean) ** 2 for x in numbers) / (len(numbers) - 1)) ** 0.5
            
            std_benchmark.run_test("Python std", "Python", python_std, python_timeout, numbers)
            std_benchmark.run_test("NumPy std", "NumPy", lambda x: np.std(x, ddof=1), numpy_timeout, numbers)
            std_benchmark.run_test("pyroid std", "pyroid", lambda x: pyroid.math.std(x), pyroid_timeout, numbers)
            
            BenchmarkReporter.print_results(std_benchmark)
            results.append(std_benchmark)
        
        # Matrix multiplication benchmark (only for smaller sizes)
        if size <= 10_000:
            matrix_size = min(500, int(size ** 0.5))
            
            # Generate random matrices
            a = [[random.random() for _ in range(matrix_size)] for _ in range(matrix_size)]
            b = [[random.random() for _ in range(matrix_size)] for _ in range(matrix_size)]
            
            # Convert to NumPy arrays for NumPy benchmark
            a_np = np.array(a)
            b_np = np.array(b)
            
            matrix_benchmark = Benchmark(f"Matrix multiply {matrix_size}x{matrix_size}", f"Multiply two {matrix_size}x{matrix_size} matrices")
            
            # Python matrix multiplication
            def python_matmul(a, b):
                n = len(a)
                c = [[0 for _ in range(n)] for _ in range(n)]
                for i in range(n):
                    for j in range(n):
                        for k in range(n):
                            c[i][j] += a[i][k] * b[k][j]
                return c
            
            # Only run Python implementation for very small matrices
            if matrix_size <= 100:
                matrix_benchmark.run_test("Python matmul", "Python", python_matmul, 30, a, b)
            
            matrix_benchmark.run_test("NumPy matmul", "NumPy", np.matmul, 30, a_np, b_np)
            matrix_benchmark.run_test("pyroid multiply", "pyroid", pyroid.math.multiply, 30, a, b)
            
            BenchmarkReporter.print_results(matrix_benchmark)
            results.append(matrix_benchmark)
    
    return results


if __name__ == "__main__":
    print("Running math benchmarks...")
    run_math_benchmarks()