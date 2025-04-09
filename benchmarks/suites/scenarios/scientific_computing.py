"""
Scientific computing benchmark for Pyroid.

This module provides a benchmark that simulates scientific computing tasks
to showcase Pyroid's performance advantages in data analysis and statistical operations.
"""

import random
import time
import math
import numpy as np

try:
    import pyroid
except ImportError:
    print("Warning: pyroid not found. Scientific computing benchmark will not run correctly.")

from ...core.benchmark import Benchmark
from ...core.reporter import BenchmarkReporter


def run_scientific_computing_benchmark(size=1_000_000):
    """Run a scientific computing benchmark.
    
    Args:
        size: Size of the dataset to process.
        
    Returns:
        A Benchmark object with results.
    """
    # Generate test data
    print(f"Generating {size:,} data points...")
    
    # Generate random data with a normal distribution
    data = [random.gauss(0, 1) for _ in range(size)]
    
    # Convert to NumPy array for NumPy operations
    data_np = np.array(data)
    
    benchmark = Benchmark("Scientific Computing", f"Statistical analysis on {size:,} data points")
    
    # Python implementation
    def python_scientific(data):
        print("Running Python scientific computing pipeline...")
        
        # Step 1: Basic statistics
        print("  Step 1: Calculating basic statistics...")
        mean = sum(data) / len(data)
        
        # Calculate variance
        variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
        std_dev = math.sqrt(variance)
        
        # Calculate median (requires sorting)
        sorted_data = sorted(data)
        if len(sorted_data) % 2 == 0:
            median = (sorted_data[len(sorted_data) // 2 - 1] + sorted_data[len(sorted_data) // 2]) / 2
        else:
            median = sorted_data[len(sorted_data) // 2]
        
        # Step 2: Z-score normalization
        print("  Step 2: Performing Z-score normalization...")
        normalized = [(x - mean) / std_dev for x in data]
        
        # Step 3: Moving average (window size 100)
        print("  Step 3: Calculating moving average...")
        window_size = 100
        moving_avg = []
        for i in range(len(normalized) - window_size + 1):
            window = normalized[i:i + window_size]
            avg = sum(window) / window_size
            moving_avg.append(avg)
        
        # Step 4: Find peaks (local maxima)
        print("  Step 4: Finding peaks...")
        peaks = []
        for i in range(1, len(moving_avg) - 1):
            if moving_avg[i] > moving_avg[i - 1] and moving_avg[i] > moving_avg[i + 1]:
                peaks.append((i, moving_avg[i]))
        
        # Step 5: Calculate correlation with shifted data
        print("  Step 5: Calculating autocorrelation...")
        shift = 10
        data1 = normalized[:-shift]
        data2 = normalized[shift:]
        
        # Calculate correlation
        mean1 = sum(data1) / len(data1)
        mean2 = sum(data2) / len(data2)
        
        numerator = sum((data1[i] - mean1) * (data2[i] - mean2) for i in range(len(data1)))
        denom1 = math.sqrt(sum((x - mean1) ** 2 for x in data1))
        denom2 = math.sqrt(sum((x - mean2) ** 2 for x in data2))
        
        correlation = numerator / (denom1 * denom2)
        
        print("Python scientific computing pipeline complete.")
        return {
            "mean": mean,
            "std_dev": std_dev,
            "median": median,
            "normalized_length": len(normalized),
            "moving_avg_length": len(moving_avg),
            "peaks_count": len(peaks),
            "correlation": correlation
        }
    
    # NumPy implementation
    def numpy_scientific(data_np):
        print("Running NumPy scientific computing pipeline...")
        
        # Step 1: Basic statistics
        print("  Step 1: Calculating basic statistics...")
        mean = np.mean(data_np)
        std_dev = np.std(data_np, ddof=1)
        median = np.median(data_np)
        
        # Step 2: Z-score normalization
        print("  Step 2: Performing Z-score normalization...")
        normalized = (data_np - mean) / std_dev
        
        # Step 3: Moving average (window size 100)
        print("  Step 3: Calculating moving average...")
        window_size = 100
        moving_avg = np.convolve(normalized, np.ones(window_size) / window_size, mode='valid')
        
        # Step 4: Find peaks (local maxima)
        print("  Step 4: Finding peaks...")
        peaks = []
        for i in range(1, len(moving_avg) - 1):
            if moving_avg[i] > moving_avg[i - 1] and moving_avg[i] > moving_avg[i + 1]:
                peaks.append((i, moving_avg[i]))
        
        # Step 5: Calculate correlation with shifted data
        print("  Step 5: Calculating autocorrelation...")
        shift = 10
        data1 = normalized[:-shift]
        data2 = normalized[shift:]
        correlation = np.corrcoef(data1, data2)[0, 1]
        
        print("NumPy scientific computing pipeline complete.")
        return {
            "mean": mean,
            "std_dev": std_dev,
            "median": median,
            "normalized_length": len(normalized),
            "moving_avg_length": len(moving_avg),
            "peaks_count": len(peaks),
            "correlation": correlation
        }
    
    # pyroid implementation
    def pyroid_scientific(data):
        print("Running pyroid scientific computing pipeline...")
        
        # Step 1: Basic statistics
        print("  Step 1: Calculating basic statistics...")
        # Use math.mean and math.std or fallback
        try:
            mean = pyroid.math.mean(data)
            std_dev = pyroid.math.std(data)
        except AttributeError:
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
            std_dev = math.sqrt(variance)
        
        # Calculate median (requires sorting)
        # Use data.collections.sort or fallback to sorted
        try:
            sorted_data = pyroid.data.collections.sort(data, None, False)
        except AttributeError:
            sorted_data = sorted(data)
        if len(sorted_data) % 2 == 0:
            median = (sorted_data[len(sorted_data) // 2 - 1] + sorted_data[len(sorted_data) // 2]) / 2
        else:
            median = sorted_data[len(sorted_data) // 2]
        
        # Step 2: Z-score normalization
        print("  Step 2: Performing Z-score normalization...")
        # Use data.collections.map or fallback to map
        try:
            normalized = pyroid.data.collections.map(data, lambda x: (x - mean) / std_dev)
        except AttributeError:
            # Avoid division by zero
            if std_dev == 0:
                normalized = [0.0] * len(data)
            else:
                normalized = list(map(lambda x: (x - mean) / std_dev, data))
        
        # Step 3: Moving average (window size 100)
        print("  Step 3: Calculating moving average...")
        window_size = 100
        moving_avg = []
        for i in range(len(normalized) - window_size + 1):
            window = normalized[i:i + window_size]
            # Use math.mean or fallback to sum/len
            try:
                avg = pyroid.math.mean(window)
            except AttributeError:
                avg = sum(window) / len(window)
            moving_avg.append(avg)
        
        # Step 4: Find peaks (local maxima)
        print("  Step 4: Finding peaks...")
        peaks = []
        for i in range(1, len(moving_avg) - 1):
            if moving_avg[i] > moving_avg[i - 1] and moving_avg[i] > moving_avg[i + 1]:
                peaks.append((i, moving_avg[i]))
        
        # Step 5: Calculate correlation with shifted data
        print("  Step 5: Calculating autocorrelation...")
        shift = 10
        data1 = normalized[:-shift]
        data2 = normalized[shift:]
        
        # Calculate correlation using parallel operations
        # Use math.mean or fallback to sum/len
        try:
            mean1 = pyroid.math.mean(data1)
            mean2 = pyroid.math.mean(data2)
        except AttributeError:
            mean1 = sum(data1) / len(data1)
            mean2 = sum(data2) / len(data2)
        
        # Calculate numerator
        def calc_numerator(i):
            return (data1[i] - mean1) * (data2[i] - mean2)
        
        # Use data.collections.map and math.sum or fallback
        try:
            numerator_values = pyroid.data.collections.map(range(len(data1)), calc_numerator)
            numerator = pyroid.math.sum(numerator_values)
        except AttributeError:
            numerator_values = list(map(calc_numerator, range(len(data1))))
            numerator = sum(numerator_values)
        
        # Calculate denominators
        # Use data.collections.map or fallback to map
        try:
            denom1_values = pyroid.data.collections.map(data1, lambda x: (x - mean1) ** 2)
            denom2_values = pyroid.data.collections.map(data2, lambda x: (x - mean2) ** 2)
        except AttributeError:
            denom1_values = list(map(lambda x: (x - mean1) ** 2, data1))
            denom2_values = list(map(lambda x: (x - mean2) ** 2, data2))
        
        # Use math.sum or fallback to sum
        try:
            denom1 = math.sqrt(pyroid.math.sum(denom1_values))
            denom2 = math.sqrt(pyroid.math.sum(denom2_values))
        except AttributeError:
            denom1 = math.sqrt(sum(denom1_values))
            denom2 = math.sqrt(sum(denom2_values))
        
        # Avoid division by zero
        if denom1 == 0 or denom2 == 0:
            correlation = 0.0
        else:
            correlation = numerator / (denom1 * denom2)
        
        print("pyroid scientific computing pipeline complete.")
        return {
            "mean": mean,
            "std_dev": std_dev,
            "median": median,
            "normalized_length": len(normalized),
            "moving_avg_length": len(moving_avg),
            "peaks_count": len(peaks),
            "correlation": correlation
        }
    
    # Set appropriate timeouts
    python_timeout = 30  # Complex pipeline might take longer
    numpy_timeout = 10
    pyroid_timeout = 10
    
    benchmark.run_test("Python scientific", "Python", python_scientific, python_timeout, data)
    benchmark.run_test("NumPy scientific", "NumPy", numpy_scientific, numpy_timeout, data_np)
    benchmark.run_test("pyroid scientific", "pyroid", pyroid_scientific, pyroid_timeout, data)
    
    BenchmarkReporter.print_results(benchmark)
    return benchmark


if __name__ == "__main__":
    print("Running scientific computing benchmark...")
    run_scientific_computing_benchmark()