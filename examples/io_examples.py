#!/usr/bin/env python3
"""
File I/O operation examples for pyroid.

This script demonstrates the file I/O capabilities of pyroid.
"""

import time
import random
import os
import json
import csv
import gzip
import pandas as pd
import pyroid

def benchmark(func, *args, **kwargs):
    """Simple benchmarking function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")
    return result

def create_test_csv_files(num_files, rows_per_file):
    """Create test CSV files for benchmarking."""
    os.makedirs("test_data", exist_ok=True)
    
    file_paths = []
    for i in range(num_files):
        file_path = f"test_data/test_file_{i}.csv"
        file_paths.append(file_path)
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'value', 'flag'])
            
            for j in range(rows_per_file):
                writer.writerow([
                    j,
                    f"item_{j}",
                    random.random() * 100,
                    random.choice(['true', 'false'])
                ])
    
    return file_paths

def create_test_json_strings(num_strings, items_per_string):
    """Create test JSON strings for benchmarking."""
    json_strings = []
    
    for i in range(num_strings):
        data = {
            "id": i,
            "name": f"record_{i}",
            "values": [random.random() for _ in range(items_per_string)],
            "metadata": {
                "created": "2025-04-04",
                "version": "1.0",
                "tags": ["test", "benchmark", f"tag_{i}"]
            }
        }
        json_strings.append(json.dumps(data))
    
    return json_strings

def main():
    print("pyroid File I/O Operations Examples")
    print("==================================")
    
    # Example 1: Reading CSV Files
    print("\n1. Reading CSV Files")
    
    # Create test CSV files
    num_files = 5
    rows_per_file = 10000
    file_paths = create_test_csv_files(num_files, rows_per_file)
    
    print(f"\nReading {num_files} CSV files with {rows_per_file} rows each:")
    
    print("\nPandas read_csv:")
    def pandas_read_csv(file_paths):
        return [pd.read_csv(file_path) for file_path in file_paths]
    
    pandas_result = benchmark(lambda: pandas_read_csv(file_paths))
    
    print("\npyroid parallel_read_csv:")
    schema = {
        'id': 'int',
        'value': 'float',
        'flag': 'bool'
    }
    pyroid_result = benchmark(lambda: pyroid.parallel_read_csv(file_paths, schema))
    
    print("\nResults (first file):")
    print(f"Pandas: {pandas_result[0].shape}")
    print(f"pyroid: ({len(next(iter(pyroid_result[0].values())))}, {len(pyroid_result[0])})")
    
    # Example 2: JSON Parsing
    print("\n2. JSON Parsing")
    
    # Create test JSON strings
    num_strings = 10000
    items_per_string = 100
    json_strings = create_test_json_strings(num_strings, items_per_string)
    
    print(f"\nParsing {num_strings} JSON strings:")
    
    print("\nPython json.loads:")
    def python_json_parse(json_strings):
        return [json.loads(s) for s in json_strings]
    
    python_result = benchmark(lambda: python_json_parse(json_strings))
    
    print("\npyroid parallel_json_parse:")
    pyroid_result = benchmark(lambda: pyroid.parallel_json_parse(json_strings))
    
    print("\nResults (first JSON):")
    print(f"Python: {list(python_result[0].keys())}")
    print(f"pyroid: {list(pyroid_result[0].keys())}")
    
    # Example 3: Compression
    print("\n3. Compression")
    
    # Create test data
    num_items = 1000
    data_size = 10000
    data = [(''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(data_size))).encode() for _ in range(num_items)]
    
    print(f"\nCompressing {num_items} items of {data_size} bytes each:")
    
    print("\nPython gzip compression:")
    def python_compress(data):
        return [gzip.compress(d) for d in data]
    
    python_result = benchmark(lambda: python_compress(data))
    
    print("\npyroid parallel_compress:")
    pyroid_result = benchmark(lambda: pyroid.parallel_compress(data, "gzip", 6))
    
    print("\nResults (compression ratio):")
    python_ratio = sum(len(c) for c in python_result) / sum(len(d) for d in data)
    pyroid_ratio = sum(len(c) for c in pyroid_result) / sum(len(d) for d in data)
    print(f"Python: {python_ratio:.4f}")
    print(f"pyroid: {pyroid_ratio:.4f}")
    
    # Example 4: Decompression
    print("\n4. Decompression")
    
    print(f"\nDecompressing {num_items} compressed items:")
    
    print("\nPython gzip decompression:")
    def python_decompress(compressed_data):
        return [gzip.decompress(d) for d in compressed_data]
    
    python_decomp_result = benchmark(lambda: python_decompress(python_result))
    
    print("\npyroid parallel_decompress:")
    pyroid_decomp_result = benchmark(lambda: pyroid.parallel_decompress(pyroid_result, "gzip"))
    
    print("\nResults (verification):")
    python_match = all(a == b for a, b in zip(data, python_decomp_result))
    pyroid_match = all(a == b for a, b in zip(data, pyroid_decomp_result))
    print(f"Python decompression matches original: {python_match}")
    print(f"pyroid decompression matches original: {pyroid_match}")
    
    # Clean up test files
    for file_path in file_paths:
        os.remove(file_path)
    os.rmdir("test_data")

if __name__ == "__main__":
    main()