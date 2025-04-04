"""
Reporting utilities for the Pyroid benchmark suite.

This module provides utilities for generating reports from benchmark results.
"""

import json
import os
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from .benchmark import Benchmark


class BenchmarkReporter:
    """Class for generating reports from benchmark results."""
    
    @staticmethod
    def print_results(benchmark: Benchmark) -> None:
        """Print the results of a benchmark to the console.
        
        Args:
            benchmark: The benchmark to print results for.
        """
        comparisons = benchmark.compare_results()
        if not comparisons:
            print(f"No comparison results for {benchmark.name}")
            return
            
        print(f"\n{benchmark.name}: {benchmark.description}")
        print("-" * 80)
        print(f"{'Implementation':<20} {'Time':<25} {'Speedup':<10}")
        print("-" * 80)
        
        for comp in comparisons:
            speedup_str = f"{comp['speedup']:.2f}x" if comp['speedup'] is not None else "N/A"
            print(f"{comp['implementation']:<20} {comp['display_time']:<25} {speedup_str:<10}")
    
    @staticmethod
    def generate_markdown_table(benchmarks: List[Benchmark]) -> str:
        """Generate a markdown table from benchmark results.
        
        Args:
            benchmarks: The benchmarks to include in the table.
            
        Returns:
            A markdown table as a string.
        """
        markdown = "| Operation | Pure Python | NumPy | pyroid | Speedup vs Python | Speedup vs NumPy |\n"
        markdown += "|-----------|------------|-------|--------|-------------------|------------------|\n"
        
        for benchmark in benchmarks:
            comparisons = benchmark.compare_results()
            if not comparisons:
                continue
                
            python_result = next((c for c in comparisons if c["implementation"] == "Python"), None)
            numpy_result = next((c for c in comparisons if c["implementation"] == "NumPy"), None)
            pyroid_result = next((c for c in comparisons if c["implementation"] == "pyroid"), None)
            
            if not python_result or not pyroid_result:
                continue
                
            python_time = python_result["display_time"]
            numpy_time = numpy_result["display_time"] if numpy_result else "N/A"
            pyroid_time = pyroid_result["display_time"]
            
            speedup_vs_python = f"{pyroid_result['speedup']:.2f}x" if pyroid_result['speedup'] is not None else "N/A"
            
            if numpy_result and not numpy_result["timed_out"] and not pyroid_result["timed_out"]:
                speedup_vs_numpy = f"{numpy_result['duration_ms'] / pyroid_result['duration_ms']:.2f}x" if numpy_result['duration_ms'] and pyroid_result['duration_ms'] else "N/A"
            else:
                speedup_vs_numpy = "N/A"
            
            markdown += f"| {benchmark.name} | {python_time} | {numpy_time} | {pyroid_time} | {speedup_vs_python} | {speedup_vs_numpy} |\n"
            
        return markdown
    
    @staticmethod
    def export_json(benchmarks: List[Benchmark], output_path: str) -> None:
        """Export benchmark results to a JSON file.
        
        Args:
            benchmarks: The benchmarks to export.
            output_path: The path to write the JSON file to.
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert benchmarks to a serializable format
        data = {
            "generated_at": datetime.now().isoformat(),
            "benchmarks": []
        }
        
        for benchmark in benchmarks:
            benchmark_data = {
                "name": benchmark.name,
                "description": benchmark.description,
                "results": []
            }
            
            for result in benchmark.results:
                result_data = {
                    "name": result.name,
                    "implementation": result.implementation,
                    "duration_ms": result.duration_ms,
                    "timed_out": result.timed_out,
                    "timeout_seconds": result.timeout_seconds,
                    "display_time": result.display_time
                }
                benchmark_data["results"].append(result_data)
            
            data["benchmarks"].append(benchmark_data)
        
        # Write the JSON file
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)