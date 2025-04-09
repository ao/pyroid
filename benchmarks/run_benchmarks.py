#!/usr/bin/env python3
"""
Main entry point for running the Pyroid benchmark suite.

This script runs all benchmarks and generates reports and visualizations.
"""

import os
import sys
import time
import asyncio
import argparse
from typing import List, Dict, Any, Optional

try:
    import pyroid
    PYROID_AVAILABLE = True
except ImportError:
    print("Warning: pyroid not found. Please install pyroid to run benchmarks.")
    PYROID_AVAILABLE = False

from benchmarks.core.benchmark import Benchmark
from benchmarks.core.reporter import BenchmarkReporter
from benchmarks.core.visualizer import BenchmarkDashboard

# Import benchmark suites
from benchmarks.suites.math_benchmarks import run_math_benchmarks
from benchmarks.suites.string_benchmarks import run_string_benchmarks
from benchmarks.suites.data_benchmarks import run_data_benchmarks
from benchmarks.suites.async_benchmarks import run_async_benchmarks, run_web_scraping_benchmark

# Import real-world scenarios
from benchmarks.suites.scenarios.data_pipeline import run_data_processing_pipeline_benchmark
from benchmarks.suites.scenarios.text_processing import run_text_processing_benchmark
from benchmarks.suites.scenarios.scientific_computing import run_scientific_computing_benchmark
from benchmarks.suites.scenarios.web_scraping import run_web_scraping_benchmark as run_detailed_web_scraping_benchmark
from benchmarks.suites.scenarios.high_throughput import run_high_throughput_benchmark


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Pyroid benchmarks")
    
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results",
        help="Directory to write results to (default: benchmarks/results)"
    )
    
    parser.add_argument(
        "--dashboard-dir",
        default="benchmarks/dashboard",
        help="Directory to write dashboard to (default: benchmarks/dashboard)"
    )
    
    parser.add_argument(
        "--size",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
        help="Size of benchmarks to run (default: medium)"
    )
    
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["all", "math", "string", "data", "async", "scenarios", "high-throughput"],
        help="Benchmark suite to run (default: all)"
    )
    
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Concurrency level for async operations"
    )
    
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Skip generating the dashboard"
    )
    
    return parser.parse_args()


def get_sizes(size_option: str) -> Dict[str, List[int]]:
    """Get dataset sizes based on the size option.
    
    Args:
        size_option: Size option (small, medium, large).
        
    Returns:
        Dictionary of dataset sizes for each benchmark type.
    """
    if size_option == "small":
        return {
            "math": [1_000, 10_000, 100_000],
            "string": [1_000, 10_000],
            "data": [1_000, 10_000],
            "async": 10,
            "data_pipeline": 100_000,
            "text_processing": 1_000,
            "scientific": 100_000,
            "web_scraping": 10,
            "high_throughput": 1_000
        }
    elif size_option == "medium":
        return {
            "math": [1_000, 10_000, 100_000, 1_000_000],
            "string": [1_000, 10_000, 100_000],
            "data": [1_000, 10_000, 100_000],
            "async": 25,
            "data_pipeline": 500_000,
            "text_processing": 5_000,
            "scientific": 500_000,
            "web_scraping": 25,
            "high_throughput": 5_000
        }
    else:  # large
        return {
            "math": [1_000, 10_000, 100_000, 1_000_000, 10_000_000],
            "string": [1_000, 10_000, 100_000, 1_000_000],
            "data": [1_000, 10_000, 100_000, 1_000_000],
            "async": 50,
            "data_pipeline": 1_000_000,
            "text_processing": 10_000,
            "scientific": 1_000_000,
            "web_scraping": 50,
            "high_throughput": 10_000
        }


async def main():
    """Run all benchmarks and generate reports."""
    args = parse_args()
    
    if not PYROID_AVAILABLE:
        print("Error: pyroid is required to run benchmarks.")
        sys.exit(1)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.dashboard_dir, exist_ok=True)
    
    # Get dataset sizes
    sizes = get_sizes(args.size)
    
    # Track all benchmarks
    all_benchmarks = []
    
    # Run benchmarks based on the selected suite
    if args.suite in ["all", "math"]:
        print("\n=== Running Math Benchmarks ===\n")
        math_benchmarks = run_math_benchmarks(sizes["math"])
        all_benchmarks.extend(math_benchmarks)
    
    if args.suite in ["all", "string"]:
        print("\n=== Running String Benchmarks ===\n")
        string_benchmarks = run_string_benchmarks(sizes["string"])
        all_benchmarks.extend(string_benchmarks)
    
    if args.suite in ["all", "data"]:
        print("\n=== Running Data Benchmarks ===\n")
        data_benchmarks = run_data_benchmarks(sizes["data"])
        all_benchmarks.extend(data_benchmarks)
    if args.suite in ["all", "async"]:
        print("\n=== Running Async Benchmarks ===\n")
        async_benchmarks = await run_async_benchmarks(concurrency=args.concurrency)
        all_benchmarks.extend(async_benchmarks)
        
        print("\n=== Running Web Scraping Benchmark ===\n")
        web_benchmark = await run_web_scraping_benchmark(sizes["web_scraping"], concurrency=args.concurrency)
        all_benchmarks.append(web_benchmark)
    
    if args.suite in ["all", "scenarios"]:
        print("\n=== Running Data Processing Pipeline Benchmark ===\n")
        pipeline_benchmark = run_data_processing_pipeline_benchmark(sizes["data_pipeline"])
        all_benchmarks.append(pipeline_benchmark)
        
        print("\n=== Running Text Processing Benchmark ===\n")
        text_benchmark = run_text_processing_benchmark(sizes["text_processing"])
        all_benchmarks.append(text_benchmark)
        
        print("\n=== Running Scientific Computing Benchmark ===\n")
        scientific_benchmark = run_scientific_computing_benchmark(sizes["scientific"])
        all_benchmarks.append(scientific_benchmark)
        
        print("\n=== Running Detailed Web Scraping Benchmark ===\n")
        detailed_web_benchmark = await run_detailed_web_scraping_benchmark(sizes["web_scraping"])
        all_benchmarks.append(detailed_web_benchmark)
        
        print("\n=== Running High-Throughput Data Processing Benchmark ===\n")
        high_throughput_benchmark = await run_high_throughput_benchmark(sizes["high_throughput"])
        all_benchmarks.append(high_throughput_benchmark)
    
    elif args.suite == "high-throughput":
        print("\n=== Running High-Throughput Data Processing Benchmark ===\n")
        high_throughput_benchmark = await run_high_throughput_benchmark(sizes["high_throughput"])
        all_benchmarks.append(high_throughput_benchmark)
    
    # Generate reports
    print("\n=== Generating Reports ===\n")
    
    # Generate markdown table
    markdown_table = BenchmarkReporter.generate_markdown_table(all_benchmarks)
    with open(os.path.join(args.output_dir, "results.md"), "w") as f:
        f.write("# Pyroid Benchmark Results\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(markdown_table)
    
    print(f"Markdown results saved to {os.path.join(args.output_dir, 'results.md')}")
    
    # Export JSON results
    BenchmarkReporter.export_json(all_benchmarks, os.path.join(args.output_dir, "results.json"))
    print(f"JSON results saved to {os.path.join(args.output_dir, 'results.json')}")
    
    # Generate dashboard
    if not args.no_dashboard:
        print("\n=== Generating Dashboard ===\n")
        dashboard = BenchmarkDashboard(all_benchmarks, args.dashboard_dir)
        dashboard.generate_dashboard()
    
    print("\nBenchmark suite completed!")


if __name__ == "__main__":
    asyncio.run(main())