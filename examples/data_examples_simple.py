#!/usr/bin/env python3
"""
Data operation examples for pyroid (simplified version).

This script demonstrates the data processing capabilities of pyroid with smaller datasets.
"""

import time
import random
import pyroid

def benchmark(func, *args, **kwargs):
    """Simple benchmarking function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")
    return result

def main():
    print("pyroid Data Operations Examples (Simplified)")
    print("=========================================")
    
    # Generate a small dataset for testing
    print("\nGenerating test data...")
    data = [random.randint(1, 1000) for _ in range(1_000)]
    print(f"Generated {len(data):,} items")
    
    # Example 1: Parallel filter
    print("\n1. Parallel Filter")
    
    def is_even(x):
        return x % 2 == 0
    
    print("\nFiltering even numbers from 1,000 items:")
    filtered = benchmark(pyroid.parallel_filter, data, is_even)
    print(f"Found {len(filtered):,} even numbers")
    print(f"First 5 items: {filtered[:5]}")
    
    # Example 2: Parallel map
    print("\n2. Parallel Map")
    
    def square(x):
        return x * x
    
    print("\nSquaring 1,000 numbers:")
    squared = benchmark(pyroid.parallel_map, data, square)
    print(f"First 5 items: {squared[:5]}")
    
    # Example 3: Parallel reduce
    print("\n3. Parallel Reduce")
    
    def add(x, y):
        return x + y
    
    print("\nSumming 1,000 numbers using parallel_reduce:")
    total = benchmark(pyroid.parallel_reduce, data, add)
    print(f"Sum: {total:,}")
    
    # Example 4: Parallel sort
    print("\n4. Parallel Sort")
    
    print("\nSorting 1,000 numbers:")
    sorted_data = benchmark(pyroid.parallel_sort, data, None, False)
    print(f"First 5 items: {sorted_data[:5]}")
    print(f"Last 5 items: {sorted_data[-5:]}")

if __name__ == "__main__":
    main()