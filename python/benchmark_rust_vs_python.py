#!/usr/bin/env python3
"""
Benchmark script to compare the performance of Pyroid Rust implementation
with our Python implementation and pandas.
"""

import time
import random
import pandas as pd
import numpy as np
import sys
import os
from typing import List, Dict, Any, Callable, Tuple, Optional

def generate_data(rows: int, cols: int) -> Dict[str, List[float]]:
    """Generate random data for benchmarking."""
    data = {}
    for i in range(cols):
        col_name = f"col_{i}"
        data[col_name] = [random.random() for _ in range(rows)]
    return data

def time_function(func: Callable, *args, **kwargs) -> Tuple[float, Any]:
    """Time a function execution."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time, result

class BenchmarkRunner:
    """Class to run benchmarks with different DataFrame implementations."""
    
    def __init__(self):
        """Initialize the benchmark runner."""
        self.implementations = {}
        self.setup_implementations()
    
    def setup_implementations(self):
        """Set up the different DataFrame implementations to benchmark."""
        # Always add pandas
        self.implementations["pandas"] = {
            "create": lambda data: pd.DataFrame(data),
            "filter": lambda df, col, val: df[df[col] > val],
            "map": lambda df, col: df[col] * 2,
            "groupby": lambda df, g_col, a_col: df.groupby(g_col)[a_col].mean().reset_index()
        }
        
        # Always add native Python
        self.implementations["native"] = {
            "create": lambda data: data,
            "filter": lambda data, col, val: [i for i, x in enumerate(data[col]) if x > val],
            "map": lambda data, col: [x * 2 for x in data[col]],
            "groupby": self.native_groupby
        }
        
        # Try to import the Rust implementation
        try:
            # First, try to import directly from the Rust extension
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
            import pyroid.pyroid as rust_pyroid
            
            # Check if the DataFrame class exists
            if hasattr(rust_pyroid, "DataFrame"):
                print("Found Rust implementation of DataFrame")
                self.implementations["rust"] = {
                    "create": lambda data: rust_pyroid.DataFrame(data),
                    "filter": lambda df, col, val: rust_pyroid.filter(df[col], lambda x: x > val),
                    "map": lambda df, col: rust_pyroid.map(df[col], lambda x: x * 2),
                    "groupby": lambda df, g_col, a_col: rust_pyroid.groupby_aggregate(df, g_col, {a_col: "mean"})
                }
        except (ImportError, AttributeError):
            print("Rust implementation not available")
        
        # Add our Python implementation
        import pyroid
        print("Using Python implementation of DataFrame")
        self.implementations["python"] = {
            "create": lambda data: pyroid.data.DataFrame(data),
            "filter": lambda df, col, val: pyroid.data.filter(df[col], lambda x: x > val),
            "map": lambda df, col: pyroid.data.map(df[col], lambda x: x * 2),
            "groupby": lambda df, g_col, a_col: pyroid.data.groupby_aggregate(df, g_col, {a_col: "mean"})
        }
    
    def native_groupby(self, data, g_col, a_col):
        """Native Python implementation of groupby."""
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
    
    def benchmark_creation(self, data: Dict[str, List[float]]) -> Dict[str, float]:
        """Benchmark DataFrame creation."""
        results = {}
        
        for name, impl in self.implementations.items():
            if name == "native":
                # Native Python doesn't need to create a DataFrame
                results[name] = 0.0
                continue
                
            time_taken, _ = time_function(impl["create"], data)
            results[name] = time_taken
        
        return results
    
    def benchmark_filter(self, data: Dict[str, List[float]], column: str, threshold: float) -> Dict[str, float]:
        """Benchmark filtering operation."""
        results = {}
        
        for name, impl in self.implementations.items():
            if name == "native":
                # Native Python operates directly on the data
                time_taken, _ = time_function(impl["filter"], data, column, threshold)
            else:
                # Create the DataFrame first
                df = impl["create"](data)
                time_taken, _ = time_function(impl["filter"], df, column, threshold)
                
            results[name] = time_taken
        
        return results
    
    def benchmark_map(self, data: Dict[str, List[float]], column: str) -> Dict[str, float]:
        """Benchmark mapping operation."""
        results = {}
        
        for name, impl in self.implementations.items():
            if name == "native":
                # Native Python operates directly on the data
                time_taken, _ = time_function(impl["map"], data, column)
            else:
                # Create the DataFrame first
                df = impl["create"](data)
                time_taken, _ = time_function(impl["map"], df, column)
                
            results[name] = time_taken
        
        return results
    
    def benchmark_groupby(self, data: Dict[str, List[Any]], group_col: str, agg_col: str) -> Dict[str, float]:
        """Benchmark groupby operation."""
        results = {}
        
        # Prepare categorical data for groupby
        categories = ["A", "B", "C", "D", "E"]
        data[group_col] = [random.choice(categories) for _ in range(len(data[agg_col]))]
        
        for name, impl in self.implementations.items():
            if name == "native":
                # Native Python operates directly on the data
                time_taken, _ = time_function(impl["groupby"], data, group_col, agg_col)
            else:
                # Create the DataFrame first
                df = impl["create"](data)
                time_taken, _ = time_function(impl["groupby"], df, group_col, agg_col)
                
            results[name] = time_taken
        
        return results
    
    def run_benchmarks(self):
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
            creation_results = self.benchmark_creation(data)
            print(f"  DataFrame Creation:")
            for name, time_taken in creation_results.items():
                if name != "native":
                    print(f"    {name}: {time_taken:.6f} seconds")
            
            # Calculate speedups
            if "rust" in creation_results and "python" in creation_results:
                print(f"    Rust vs Python: {creation_results['python'] / max(creation_results['rust'], 1e-10):.2f}x")
            
            if "python" in creation_results and "pandas" in creation_results:
                print(f"    Python vs Pandas: {creation_results['pandas'] / max(creation_results['python'], 1e-10):.2f}x")
            
            # Benchmark filtering
            filter_results = self.benchmark_filter(data, "col_0", 0.5)
            print(f"  Filtering (col_0 > 0.5):")
            for name, time_taken in filter_results.items():
                print(f"    {name}: {time_taken:.6f} seconds")
            
            # Calculate speedups
            if "rust" in filter_results and "python" in filter_results:
                print(f"    Rust vs Python: {filter_results['python'] / max(filter_results['rust'], 1e-10):.2f}x")
            
            if "python" in filter_results and "pandas" in filter_results:
                print(f"    Python vs Pandas: {filter_results['pandas'] / max(filter_results['python'], 1e-10):.2f}x")
            
            if "python" in filter_results and "native" in filter_results:
                print(f"    Python vs Native: {filter_results['native'] / max(filter_results['python'], 1e-10):.2f}x")
            
            # Benchmark mapping
            map_results = self.benchmark_map(data, "col_0")
            print(f"  Mapping (col_0 * 2):")
            for name, time_taken in map_results.items():
                print(f"    {name}: {time_taken:.6f} seconds")
            
            # Calculate speedups
            if "rust" in map_results and "python" in map_results:
                print(f"    Rust vs Python: {map_results['python'] / max(map_results['rust'], 1e-10):.2f}x")
            
            if "python" in map_results and "pandas" in map_results:
                print(f"    Python vs Pandas: {map_results['pandas'] / max(map_results['python'], 1e-10):.2f}x")
            
            if "python" in map_results and "native" in map_results:
                print(f"    Python vs Native: {map_results['native'] / max(map_results['python'], 1e-10):.2f}x")
            
            # Benchmark groupby (only for smaller datasets to avoid excessive memory usage)
            if rows <= 100000:
                groupby_results = self.benchmark_groupby(data, "category", "col_0")
                print(f"  GroupBy (category) and Aggregate (col_0, mean):")
                for name, time_taken in groupby_results.items():
                    print(f"    {name}: {time_taken:.6f} seconds")
                
                # Calculate speedups
                if "rust" in groupby_results and "python" in groupby_results:
                    print(f"    Rust vs Python: {groupby_results['python'] / max(groupby_results['rust'], 1e-10):.2f}x")
                
                if "python" in groupby_results and "pandas" in groupby_results:
                    print(f"    Python vs Pandas: {groupby_results['pandas'] / max(groupby_results['python'], 1e-10):.2f}x")
                
                if "python" in groupby_results and "native" in groupby_results:
                    print(f"    Python vs Native: {groupby_results['native'] / max(groupby_results['python'], 1e-10):.2f}x")

if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run_benchmarks()