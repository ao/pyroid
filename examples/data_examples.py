#!/usr/bin/env python3
"""
Data operation examples for pyroid.

This script demonstrates the data processing capabilities of pyroid.
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
    print("Pyroid Data Operations Examples")
    print("=============================")
    
    # Example 1: Collection operations
    print("\n1. Collection Operations")
    
    # Create a list of numbers
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"Original numbers: {numbers}")
    
    # Filter even numbers
    even_numbers = pyroid.data.filter(numbers, lambda x: x % 2 == 0)
    print(f"Even numbers: {even_numbers}")
    
    # Map: double each number
    doubled = pyroid.data.map(numbers, lambda x: x * 2)
    print(f"Doubled numbers: {doubled}")
    
    # Reduce: sum all numbers
    total = pyroid.data.reduce(numbers, lambda x, y: x + y)
    print(f"Sum of numbers: {total}")
    
    # Sort in descending order
    sorted_desc = pyroid.data.sort(numbers, reverse=True)
    print(f"Sorted (descending): {sorted_desc}")
    
    # Example 2: DataFrame operations
    print("\n2. DataFrame Operations")
    
    # Create a DataFrame
    df = pyroid.data.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, 35, 40, 45]
    })
    
    print(f"DataFrame: {df}")
    
    # Apply a function to each column
    result = pyroid.data.apply(df, lambda x: x * 2, axis=0)
    print(f"Applied function (double): {result}")
    
    # Group by age and count names
    try:
        grouped = pyroid.data.groupby_aggregate(df, "age", {"name": "count"})
        print(f"Grouped by age: {grouped}")
    except Exception as e:
        print(f"Group by operation not fully implemented: {e}")
    
    # Example 3: Performance comparison
    print("\n3. Performance Comparison")
    
    # Create a large list for benchmarking
    large_list = list(range(1000000))
    print(f"Large list length: {len(large_list)}")
    
    # Benchmark Python's built-in filter
    print("\nPython built-in filter:")
    python_result = benchmark(lambda: list(filter(lambda x: x % 2 == 0, large_list)))
    print(f"Result length: {len(python_result)}")
    
    # Benchmark Pyroid's filter
    print("\nPyroid filter:")
    pyroid_result = benchmark(lambda: pyroid.data.filter(large_list, lambda x: x % 2 == 0))
    print(f"Result length: {len(pyroid_result)}")
    
    # Benchmark Python's built-in map
    print("\nPython built-in map:")
    python_result = benchmark(lambda: list(map(lambda x: x * 2, large_list)))
    print(f"Result length: {len(python_result)}")
    
    # Benchmark Pyroid's map
    print("\nPyroid map:")
    pyroid_result = benchmark(lambda: pyroid.data.map(large_list, lambda x: x * 2))
    print(f"Result length: {len(pyroid_result)}")
    
    # Example 4: Advanced DataFrame operations
    print("\n4. Advanced DataFrame Operations")
    
    # Create a more complex DataFrame
    df2 = pyroid.data.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "category": ["A", "B", "A", "B", "A", "C"],
        "value": [10, 20, 15, 25, 30, 35]
    })
    
    print(f"Complex DataFrame: {df2}")
    
    # Try to filter rows
    try:
        filtered = pyroid.data.filter(df2, lambda row: row["value"] > 20)
        print(f"Filtered rows (value > 20): {filtered}")
    except Exception as e:
        print(f"DataFrame filter not implemented: {e}")
    
    # Try to sort by value
    try:
        sorted_df = pyroid.data.sort(df2, "value", ascending=False)
        print(f"Sorted by value (descending): {sorted_df}")
    except Exception as e:
        print(f"DataFrame sort not implemented: {e}")
    
    # Try to join DataFrames
    try:
        categories = pyroid.data.DataFrame({
            "category": ["A", "B", "C"],
            "description": ["Category A", "Category B", "Category C"]
        })
        
        joined = pyroid.data.join(df2, categories, "category")
        print(f"Joined DataFrames: {joined}")
    except Exception as e:
        print(f"DataFrame join not implemented: {e}")

if __name__ == "__main__":
    main()