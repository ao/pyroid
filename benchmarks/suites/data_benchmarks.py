"""
Data operation benchmarks for Pyroid.

This module provides benchmarks for comparing Pyroid's data operations with
pure Python implementations.
"""

import random
import time
import functools

try:
    import pyroid
except ImportError:
    print("Warning: pyroid not found. Data benchmarks will not run correctly.")

from ..core.benchmark import Benchmark
from ..core.reporter import BenchmarkReporter


def run_data_benchmarks(sizes=[1_000, 10_000, 100_000, 1_000_000]):
    """Run data benchmarks.
    
    Args:
        sizes: List of dataset sizes to benchmark.
        
    Returns:
        List of Benchmark objects with results.
    """
    results = []
    
    for size in sizes:
        # Generate test data
        data = [random.randint(1, 1000) for _ in range(size)]
        
        # Filter benchmark
        filter_benchmark = Benchmark(f"Filter {size:,} items", f"Filter even numbers from {size:,} integers")
        
        # Define filter function
        def is_even(x):
            return x % 2 == 0
        
        # Set appropriate timeouts based on dataset size
        python_timeout = 2 if size <= 100_000 else 10
        pyroid_timeout = 10  # Pyroid should be fast, but set a reasonable timeout
        
        filter_benchmark.run_test("Python filter", "Python", lambda d: list(filter(is_even, d)), python_timeout, data)
        filter_benchmark.run_test("pyroid parallel_filter", "pyroid", pyroid.parallel_filter, pyroid_timeout, data, is_even)
        
        BenchmarkReporter.print_results(filter_benchmark)
        results.append(filter_benchmark)
        
        # Map benchmark
        map_benchmark = Benchmark(f"Map {size:,} items", f"Square {size:,} integers")
        
        # Define map function
        def square(x):
            return x * x
        
        map_benchmark.run_test("Python map", "Python", lambda d: list(map(square, d)), python_timeout, data)
        map_benchmark.run_test("pyroid parallel_map", "pyroid", pyroid.parallel_map, pyroid_timeout, data, square)
        
        BenchmarkReporter.print_results(map_benchmark)
        results.append(map_benchmark)
        
        # Reduce benchmark
        reduce_benchmark = Benchmark(f"Reduce {size:,} items", f"Sum {size:,} integers")
        
        # Define reduce function
        def add(x, y):
            return x + y
        
        # Use a smaller dataset for reduce to avoid overflow
        reduce_data = data[:min(size, 10_000)]
        
        reduce_benchmark.run_test("Python sum", "Python", sum, python_timeout, reduce_data)
        reduce_benchmark.run_test("Python functools.reduce", "Python", lambda d: functools.reduce(add, d), python_timeout, reduce_data)
        reduce_benchmark.run_test("pyroid parallel_reduce", "pyroid", pyroid.parallel_reduce, pyroid_timeout, reduce_data, add)
        
        BenchmarkReporter.print_results(reduce_benchmark)
        results.append(reduce_benchmark)
        
        # Sort benchmark
        sort_benchmark = Benchmark(f"Sort {size:,} items", f"Sort {size:,} integers")
        
        # Use a copy of the data to avoid modifying the original
        sort_data = data.copy()
        
        sort_benchmark.run_test("Python sorted", "Python", sorted, python_timeout, sort_data)
        sort_benchmark.run_test("pyroid parallel_sort", "pyroid", lambda d: pyroid.parallel_sort(d, None, False), pyroid_timeout, sort_data)
        
        BenchmarkReporter.print_results(sort_benchmark)
        results.append(sort_benchmark)
        
        # Sort with key function benchmark
        # Create a list of tuples (id, value)
        tuple_data = [(i, random.randint(1, 1000)) for i in range(size)]
        
        key_sort_benchmark = Benchmark(f"Sort {size:,} items with key", f"Sort {size:,} tuples by second element")
        
        # Define key function
        def get_second(item):
            return item[1]
        
        key_sort_benchmark.run_test("Python sorted with key", "Python", lambda d: sorted(d, key=get_second), python_timeout, tuple_data)
        key_sort_benchmark.run_test("pyroid parallel_sort with key", "pyroid", lambda d: pyroid.parallel_sort(d, get_second, False), pyroid_timeout, tuple_data)
        
        BenchmarkReporter.print_results(key_sort_benchmark)
        results.append(key_sort_benchmark)
    
    return results


if __name__ == "__main__":
    print("Running data benchmarks...")
    run_data_benchmarks()