"""
Data processing pipeline benchmark for Pyroid.

This module provides a benchmark that simulates a data processing pipeline
to showcase Pyroid's performance advantages in ETL (Extract, Transform, Load) operations.
"""

import random
import time

try:
    import pyroid
except ImportError:
    print("Warning: pyroid not found. Data pipeline benchmark will not run correctly.")

from ...core.benchmark import Benchmark
from ...core.reporter import BenchmarkReporter


def run_data_processing_pipeline_benchmark(size=1_000_000):
    """Run a data processing pipeline benchmark.
    
    Args:
        size: Number of records to process.
        
    Returns:
        A Benchmark object with results.
    """
    # Generate test data
    print(f"Generating {size:,} records of test data...")
    data = [{"id": i, "value": random.random(), "category": random.choice(["A", "B", "C", "D"])} for i in range(size)]
    print("Data generation complete.")
    
    pipeline_benchmark = Benchmark("Data Processing Pipeline", f"ETL pipeline on {size:,} records")
    
    # Python implementation
    def python_pipeline(data):
        print("Running Python pipeline...")
        
        # Step 1: Filter records where value > 0.5
        print("  Step 1: Filtering records...")
        filtered = [item for item in data if item["value"] > 0.5]
        print(f"  Filtered to {len(filtered):,} records")
        
        # Step 2: Transform values (multiply by 10)
        print("  Step 2: Transforming values...")
        transformed = [{"id": item["id"], "value": item["value"] * 10, "category": item["category"]} for item in filtered]
        
        # Step 3: Group by category
        print("  Step 3: Grouping by category...")
        grouped = {}
        for item in transformed:
            category = item["category"]
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(item)
        
        # Step 4: Aggregate
        print("  Step 4: Aggregating results...")
        results = []
        for category, items in grouped.items():
            total = sum(item["value"] for item in items)
            count = len(items)
            results.append({"category": category, "total": total, "count": count, "average": total / count})
        
        # Step 5: Sort by average
        print("  Step 5: Sorting results...")
        results.sort(key=lambda x: x["average"], reverse=True)
        
        print("Python pipeline complete.")
        return results
    
    # pyroid implementation
    def pyroid_pipeline(data):
        print("Running pyroid pipeline...")
        
        # Step 1: Filter records where value > 0.5
        print("  Step 1: Filtering records...")
        filtered = pyroid.parallel_filter(data, lambda item: item["value"] > 0.5)
        print(f"  Filtered to {len(filtered):,} records")
        
        # Step 2: Transform values (multiply by 10)
        print("  Step 2: Transforming values...")
        transformed = pyroid.parallel_map(filtered, lambda item: {"id": item["id"], "value": item["value"] * 10, "category": item["category"]})
        
        # Step 3: Group by category (still using Python as pyroid doesn't have a direct equivalent)
        print("  Step 3: Grouping by category...")
        grouped = {}
        for item in transformed:
            category = item["category"]
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(item)
        
        # Step 4: Aggregate using pyroid for each group
        print("  Step 4: Aggregating results...")
        results = []
        for category, items in grouped.items():
            values = pyroid.parallel_map(items, lambda item: item["value"])
            total = pyroid.parallel_sum(values)
            count = len(items)
            results.append({"category": category, "total": total, "count": count, "average": total / count})
        
        # Step 5: Sort by average
        print("  Step 5: Sorting results...")
        results = pyroid.parallel_sort(results, lambda x: x["average"], True)
        
        print("pyroid pipeline complete.")
        return results
    
    # Set appropriate timeouts
    python_timeout = 30  # Complex pipeline might take longer
    pyroid_timeout = 10
    
    pipeline_benchmark.run_test("Python pipeline", "Python", python_pipeline, python_timeout, data)
    pipeline_benchmark.run_test("pyroid pipeline", "pyroid", pyroid_pipeline, pyroid_timeout, data)
    
    BenchmarkReporter.print_results(pipeline_benchmark)
    return pipeline_benchmark


if __name__ == "__main__":
    print("Running data processing pipeline benchmark...")
    run_data_processing_pipeline_benchmark()