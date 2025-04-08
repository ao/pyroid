#!/usr/bin/env python3
"""
DataFrame operation examples for pyroid.

This script demonstrates the DataFrame capabilities of pyroid.
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
    print("Pyroid DataFrame Operations Examples")
    print("=================================")
    
    # Example 1: Creating a DataFrame
    print("\n1. Creating a DataFrame")
    
    # Create a DataFrame
    df = pyroid.data.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, 35, 40, 45],
        "salary": [50000, 60000, 70000, 80000, 90000]
    })
    
    print(f"DataFrame created with {len(df.get('id'))} rows")
    print(f"Columns: {list(df.keys())}")
    
    # Example 2: Accessing DataFrame data
    print("\n2. Accessing DataFrame Data")
    
    # Access a column
    ids = df.get("id")
    names = df.get("name")
    
    print(f"IDs: {ids}")
    print(f"Names: {names}")
    
    # Try to access a row (may not be implemented)
    try:
        row = df.get_row(0)
        print(f"First row: {row}")
    except Exception as e:
        print(f"Row access not implemented: {e}")
    
    # Example 3: Applying functions to DataFrame
    print("\n3. Applying Functions to DataFrame")
    
    # Apply a function to each column
    result = pyroid.data.apply(df, lambda x: x * 2, axis=0)
    
    print("Original values:")
    print(f"IDs: {df.get('id')}")
    print(f"Ages: {df.get('age')}")
    
    print("\nAfter applying (x * 2):")
    print(f"IDs: {result.get('id')}")
    print(f"Ages: {result.get('age')}")
    
    # Example 4: Filtering DataFrame
    print("\n4. Filtering DataFrame")
    
    # Try to filter rows (may not be implemented)
    try:
        filtered = pyroid.data.filter(df, lambda row: row["age"] > 30)
        print(f"Filtered DataFrame (age > 30): {len(filtered.get('id'))} rows")
        print(f"Ages: {filtered.get('age')}")
    except Exception as e:
        print(f"DataFrame filtering not implemented: {e}")
        
        # Manual filtering as a workaround
        indices = [i for i, age in enumerate(df.get("age")) if age > 30]
        filtered_ids = [df.get("id")[i] for i in indices]
        filtered_names = [df.get("name")[i] for i in indices]
        filtered_ages = [df.get("age")[i] for i in indices]
        
        print("Manual filtering (age > 30):")
        print(f"IDs: {filtered_ids}")
        print(f"Names: {filtered_names}")
        print(f"Ages: {filtered_ages}")
    
    # Example 5: Group by and aggregate
    print("\n5. Group By and Aggregate")
    
    # Create a DataFrame with duplicate age values
    df2 = pyroid.data.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
        "age": [25, 30, 25, 30, 35, 35],
        "department": ["HR", "IT", "HR", "IT", "Finance", "Finance"]
    })
    
    print(f"DataFrame created with {len(df2.get('id'))} rows")
    
    # Group by age and count
    try:
        grouped = pyroid.data.groupby_aggregate(df2, "age", {"name": "count"})
        print("Grouped by age:")
        print(f"Ages: {grouped.get('age')}")
        print(f"Counts: {grouped.get('name_count')}")
    except Exception as e:
        print(f"Group by operation not fully implemented: {e}")
        
        # Manual groupby as a workaround
        ages = df2.get("age")
        unique_ages = list(set(ages))
        counts = [ages.count(age) for age in unique_ages]
        
        print("Manual groupby by age:")
        print(f"Ages: {unique_ages}")
        print(f"Counts: {counts}")
    
    # Example 6: Performance comparison
    print("\n6. Performance Comparison")
    
    # Create a larger DataFrame for benchmarking
    large_df = {}
    n_rows = 10000
    
    for i in range(10):
        large_df[f"col_{i}"] = list(range(n_rows))
    
    df_large = pyroid.data.DataFrame(large_df)
    print(f"Large DataFrame created with {n_rows} rows and {len(large_df)} columns")
    
    # Benchmark apply operation
    print("\nBenchmarking apply operation (multiply by 2):")
    result = benchmark(lambda: pyroid.data.apply(df_large, lambda x: x * 2, axis=0))
    print(f"Result has {len(result.get('col_0'))} rows")

if __name__ == "__main__":
    main()