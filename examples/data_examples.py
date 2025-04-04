#!/usr/bin/env python3
"""
Data operation examples for pyroid.

This script demonstrates the data processing capabilities of pyroid.
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
    print("pyroid Data Operations Examples")
    print("==============================")
    
    # Generate a smaller dataset for testing
    print("\nGenerating test data...")
    data = [random.randint(1, 1000) for _ in range(10_000)]
    print(f"Generated {len(data):,} items")
    
    # Example 1: Parallel filter
    print("\n1. Parallel Filter")
    
    def is_even(x):
        return x % 2 == 0
    
    print("\nFiltering even numbers from 10,000 items:")
    filtered = benchmark(pyroid.parallel_filter, data, is_even)
    print(f"Found {len(filtered):,} even numbers")
    print(f"First 5 items: {filtered[:5]}")
    
    # Compare with Python's filter
    print("\nPython's built-in filter:")
    py_filtered = benchmark(lambda d: list(filter(is_even, d)), data)
    print(f"Found {len(py_filtered):,} even numbers")
    print(f"Results match: {len(filtered) == len(py_filtered)}")
    
    # Example 2: Parallel map
    print("\n2. Parallel Map")
    
    def square(x):
        return x * x
    
    print("\nSquaring 10,000 numbers:")
    squared = benchmark(pyroid.parallel_map, data, square)
    print(f"First 5 items: {squared[:5]}")
    
    # Compare with Python's map
    print("\nPython's built-in map:")
    py_squared = benchmark(lambda d: list(map(square, d)), data)
    print(f"Results match: {squared[:100] == py_squared[:100]}")
    
    # Example 3: Parallel reduce
    print("\n3. Parallel Reduce")
    
    def add(x, y):
        return x + y
    
    # Use a smaller dataset for reduce to avoid overflow
    reduce_data = data[:10_000]
    
    print("\nSumming 10,000 numbers using parallel_reduce:")
    total = benchmark(pyroid.parallel_reduce, reduce_data, add)
    print(f"Sum: {total:,}")
    
    # Compare with Python's sum
    print("\nPython's built-in sum:")
    py_total = benchmark(sum, reduce_data)
    print(f"Sum: {py_total:,}")
    print(f"Results match: {total == py_total}")
    
    # Example 4: Parallel sort
    print("\n4. Parallel Sort")
    
    # Use a smaller dataset for sorting
    sort_data = data[:10_000]
    
    print("\nSorting 10,000 numbers:")
    sorted_data = benchmark(pyroid.parallel_sort, sort_data, None, False)
    print(f"First 5 items: {sorted_data[:5]}")
    print(f"Last 5 items: {sorted_data[-5:]}")
    
    # Compare with Python's sort
    print("\nPython's built-in sorted:")
    py_sorted = benchmark(sorted, sort_data)
    print(f"Results match: {sorted_data[:100] == py_sorted[:100]}")
    
    # Example 5: Parallel sort with key function
    print("\n5. Parallel Sort with Key Function")
    
    # Create a list of tuples (id, value)
    tuple_data = [(i, random.randint(1, 1000)) for i in range(10_000)]
    
    def get_second(item):
        return item[1]
    
    print("\nSorting 10,000 tuples by second element:")
    sorted_tuples = benchmark(pyroid.parallel_sort, tuple_data, get_second, False)
    print(f"First 5 items: {sorted_tuples[:5]}")
    
    # Compare with Python's sort
    print("\nPython's built-in sorted with key:")
    py_sorted_tuples = benchmark(lambda d: sorted(d, key=get_second), tuple_data)
    print(f"Results match: {[t[1] for t in sorted_tuples[:10]] == [t[1] for t in py_sorted_tuples[:10]]}")
    
    # Example 6: Parallel sort in reverse
    print("\n6. Parallel Sort in Reverse")
    
    print("\nSorting 10,000 numbers in reverse:")
    reverse_sorted = benchmark(pyroid.parallel_sort, sort_data, None, True)
    print(f"First 5 items: {reverse_sorted[:5]}")
    print(f"Last 5 items: {reverse_sorted[-5:]}")
    
    # Compare with Python's sort
    print("\nPython's built-in sorted in reverse:")
    py_reverse_sorted = benchmark(lambda d: sorted(d, reverse=True), sort_data)
    print(f"Results match: {reverse_sorted[:10] == py_reverse_sorted[:10]}")

if __name__ == "__main__":
    main()