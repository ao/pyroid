#!/usr/bin/env python3
"""
Benchmark script to compare the performance of Pyroid DataFrame operations
with native Python and pandas implementations.
"""

import time
import random
import pandas as pd
import numpy as np
import pyroid
from typing import List, Dict, Any, Callable

def generate_data(rows: int, cols: int) -> Dict[str, List[float]]:
    """Generate random data for benchmarking."""
    data = {}
    for i in range(cols):
        col_name = f"col_{i}"
        data[col_name] = [random.random() for _ in range(rows)]
    return data

def time_function(func: Callable, *args, **kwargs) -> float:
    """Time a function execution."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time, result

def benchmark_creation(data: Dict[str, List[float]]) -> Dict[str, float]:
    """Benchmark DataFrame creation."""
    results = {}
    
    # Pyroid DataFrame creation
    pyroid_time, _ = time_function(pyroid.data.DataFrame, data)
    results["pyroid"] = pyroid_time
    
    # Pandas DataFrame creation
    pandas_time, _ = time_function(pd.DataFrame, data)
    results["pandas"] = pandas_time
    
    return results

def benchmark_filter(data: Dict[str, List[float]], column: str, threshold: float) -> Dict[str, float]:
    """Benchmark filtering operation."""
    results = {}
    
    # Pyroid DataFrame filter
    pyroid_df = pyroid.data.DataFrame(data)
    pyroid_time, _ = time_function(
        lambda df, col, val: pyroid.data.filter(df[col], lambda x: x > val),
        pyroid_df, column, threshold
    )
    results["pyroid"] = pyroid_time
    
    # Pandas DataFrame filter
    pandas_df = pd.DataFrame(data)
    pandas_time, _ = time_function(
        lambda df, col, val: df[df[col] > val],
        pandas_df, column, threshold
    )
    results["pandas"] = pandas_time
    
    # Native Python filter
    native_time, _ = time_function(
        lambda data, col, val: [i for i, x in enumerate(data[col]) if x > val],
        data, column, threshold
    )
    results["native"] = native_time
    
    return results

def benchmark_map(data: Dict[str, List[float]], column: str) -> Dict[str, float]:
    """Benchmark mapping operation."""
    results = {}
    
    # Pyroid DataFrame map
    pyroid_df = pyroid.data.DataFrame(data)
    pyroid_time, _ = time_function(
        lambda df, col: pyroid.data.map(df[col], lambda x: x * 2),
        pyroid_df, column
    )
    results["pyroid"] = pyroid_time
    
    # Pandas DataFrame map
    pandas_df = pd.DataFrame(data)
    pandas_time, _ = time_function(
        lambda df, col: df[col] * 2,
        pandas_df, column
    )
    results["pandas"] = pandas_time
    
    # Native Python map
    native_time, _ = time_function(
        lambda data, col: [x * 2 for x in data[col]],
        data, column
    )
    results["native"] = native_time
    
    return results

def benchmark_groupby(data: Dict[str, List[Any]], group_col: str, agg_col: str) -> Dict[str, float]:
    """Benchmark groupby operation."""
    results = {}
    
    # Prepare categorical data for groupby
    categories = ["A", "B", "C", "D", "E"]
    data[group_col] = [random.choice(categories) for _ in range(len(data[agg_col]))]
    
    # Pyroid DataFrame groupby
    pyroid_df = pyroid.data.DataFrame(data)
    pyroid_time, _ = time_function(
        lambda df, g_col, a_col: pyroid.data.groupby_aggregate(df, g_col, {a_col: "mean"}),
        pyroid_df, group_col, agg_col
    )
    results["pyroid"] = pyroid_time
    
    # Pandas DataFrame groupby
    pandas_df = pd.DataFrame(data)
    pandas_time, _ = time_function(
        lambda df, g_col, a_col: df.groupby(g_col)[a_col].mean().reset_index(),
        pandas_df, group_col, agg_col
    )
    results["pandas"] = pandas_time
    
    # Native Python groupby
    def native_groupby(data, g_col, a_col):
        groups = {}
        for i in range(len(data[g_col])):
            key = data[g_col][i]
            if key not in groups:
                groups[key] = []
            groups[key].append(data[a_col][i])
        
        result = {g_col: [], f"{a_col}_mean": []}
        for key, values in groups.items():
            result[g_col].append(key)
            result[f"{a_col}_mean"].append(sum(values) / len(values))
        
        return result
    
    native_time, _ = time_function(
        native_groupby,
        data, group_col, agg_col
    )
    results["native"] = native_time
    
    return results

def run_benchmarks():
    """Run all benchmarks."""
    print("Running DataFrame benchmarks...")
    
    # Test with different data sizes
    sizes = [
        (1000, 5),
        (10000, 10),
        (100000, 20),
        (1000000, 30)
    ]
    
    for rows, cols in sizes:
        print(f"\nBenchmarking with {rows} rows and {cols} columns:")
        data = generate_data(rows, cols)
        
        # Benchmark DataFrame creation
        creation_results = benchmark_creation(data)
        print(f"  DataFrame Creation:")
        print(f"    Pyroid: {creation_results['pyroid']:.6f} seconds")
        print(f"    Pandas: {creation_results['pandas']:.6f} seconds")
        print(f"    Speedup: {creation_results['pandas'] / creation_results['pyroid']:.2f}x")
        
        # Benchmark filtering
        filter_results = benchmark_filter(data, "col_0", 0.5)
        print(f"  Filtering (col_0 > 0.5):")
        print(f"    Pyroid: {filter_results['pyroid']:.6f} seconds")
        print(f"    Pandas: {filter_results['pandas']:.6f} seconds")
        print(f"    Native: {filter_results['native']:.6f} seconds")
        print(f"    Speedup vs Native: {filter_results['native'] / filter_results['pyroid']:.2f}x")
        print(f"    Speedup vs Pandas: {filter_results['pandas'] / filter_results['pyroid']:.2f}x")
        
        # Benchmark mapping
        map_results = benchmark_map(data, "col_0")
        print(f"  Mapping (col_0 * 2):")
        print(f"    Pyroid: {map_results['pyroid']:.6f} seconds")
        print(f"    Pandas: {map_results['pandas']:.6f} seconds")
        print(f"    Native: {map_results['native']:.6f} seconds")
        print(f"    Speedup vs Native: {map_results['native'] / map_results['pyroid']:.2f}x")
        print(f"    Speedup vs Pandas: {map_results['pandas'] / map_results['pyroid']:.2f}x")
        
        # Benchmark groupby (only for smaller datasets to avoid excessive memory usage)
        if rows <= 100000:
            groupby_results = benchmark_groupby(data, "category", "col_0")
            print(f"  GroupBy (category) and Aggregate (col_0, mean):")
            print(f"    Pyroid: {groupby_results['pyroid']:.6f} seconds")
            print(f"    Pandas: {groupby_results['pandas']:.6f} seconds")
            print(f"    Native: {groupby_results['native']:.6f} seconds")
            print(f"    Speedup vs Native: {groupby_results['native'] / groupby_results['pyroid']:.2f}x")
            print(f"    Speedup vs Pandas: {groupby_results['pandas'] / groupby_results['pyroid']:.2f}x")

def check_implementation():
    """Check if we're using the Rust implementation or Python implementation."""
    try:
        # Try to access a private attribute that would only exist in our Python implementation
        pyroid.data.DataFrame._validate
        print("Using Python implementation of DataFrame")
    except AttributeError:
        print("Using Rust implementation of DataFrame")

if __name__ == "__main__":
    print("Checking DataFrame implementation:")
    check_implementation()
    run_benchmarks()