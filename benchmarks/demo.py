#!/usr/bin/env python3
"""
Demo script for showcasing Pyroid's performance advantages.

This script runs a subset of benchmarks that demonstrate the most impressive
performance improvements for demonstration purposes.
"""

import time
import random
import asyncio
import re
import numpy as np

try:
    import pyroid
    PYROID_AVAILABLE = True
except ImportError:
    print("Error: pyroid is required to run the demo.")
    PYROID_AVAILABLE = False
    import sys
    sys.exit(1)

from benchmarks.core.benchmark import time_limit, TimeoutException


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def print_result(name, time_ms=None, timed_out=False, timeout=None):
    """Print a formatted benchmark result."""
    if timed_out:
        print(f"{name}: Timed out after {timeout} seconds")
    else:
        print(f"{name}: {time_ms:.2f}ms")


def print_speedup(python_time=None, pyroid_time=None, python_timed_out=False, timeout=None):
    """Print the speedup factor."""
    if python_timed_out:
        # Calculate minimum speedup based on timeout
        min_speedup = (timeout * 1000) / pyroid_time
        print(f"\nSpeedup: >{min_speedup:.1f}x (Python timed out)")
    elif python_time is not None and pyroid_time is not None:
        speedup = python_time / pyroid_time
        print(f"\nSpeedup: {speedup:.1f}x")
    else:
        print("\nSpeedup: N/A")


def demo_sum():
    """Demo summing a large list of numbers."""
    print_header("Demo 1: Summing 10 Million Numbers")
    
    # Generate test data
    print("Generating 10 million random numbers...")
    numbers = [random.random() for _ in range(10_000_000)]
    print("Data generation complete.")
    
    # Python sum
    print("\nPure Python:")
    python_time = None
    python_timed_out = False
    try:
        with time_limit(10):  # 10-second timeout
            start = time.time()
            python_result = sum(numbers)
            end = time.time()
            python_time = (end - start) * 1000
            print(f"Result: {python_result}")
            print_result("Time", python_time)
    except TimeoutException:
        python_timed_out = True
        print_result("Time", timed_out=True, timeout=10)
    
    # NumPy sum
    print("\nNumPy:")
    start = time.time()
    numpy_result = np.sum(numbers)
    end = time.time()
    numpy_time = (end - start) * 1000
    print(f"Result: {numpy_result}")
    print_result("Time", numpy_time)
    
    # pyroid sum
    print("\npyroid:")
    start = time.time()
    pyroid_result = pyroid.parallel_sum(numbers)
    end = time.time()
    pyroid_time = (end - start) * 1000
    print(f"Result: {pyroid_result}")
    print_result("Time", pyroid_time)
    
    # Print speedups
    print("\nSpeedup vs NumPy: {:.1f}x".format(numpy_time / pyroid_time))
    print_speedup(python_time, pyroid_time, python_timed_out, 10)


def demo_regex():
    """Demo regex replacement on a large text."""
    print_header("Demo 2: Regex Replacement on Large Text")
    
    # Generate test data
    print("Generating 1 million characters of text...")
    text = "Hello world! " * 100000
    print(f"Text length: {len(text):,} characters")
    
    # Python regex
    print("\nPure Python:")
    start = time.time()
    python_result = re.sub(r"Hello", "Hi", text)
    end = time.time()
    python_time = (end - start) * 1000
    print(f"Result length: {len(python_result):,} characters")
    print_result("Time", python_time)
    
    # pyroid regex
    print("\npyroid:")
    start = time.time()
    pyroid_result = pyroid.parallel_regex_replace(text, r"Hello", "Hi")
    end = time.time()
    pyroid_time = (end - start) * 1000
    print(f"Result length: {len(pyroid_result):,} characters")
    print_result("Time", pyroid_time)
    
    # Print speedup
    print_speedup(python_time, pyroid_time)


def demo_sort():
    """Demo sorting a large list."""
    print_header("Demo 3: Sorting 1 Million Items")
    
    # Generate test data
    print("Generating 1 million random integers...")
    data = [random.randint(1, 1000000) for _ in range(1_000_000)]
    print("Data generation complete.")
    
    # Python sort
    print("\nPure Python:")
    start = time.time()
    python_result = sorted(data)
    end = time.time()
    python_time = (end - start) * 1000
    print(f"First 5 items: {python_result[:5]}")
    print_result("Time", python_time)
    
    # pyroid sort
    print("\npyroid:")
    start = time.time()
    pyroid_result = pyroid.parallel_sort(data, None, False)
    end = time.time()
    pyroid_time = (end - start) * 1000
    print(f"First 5 items: {pyroid_result[:5]}")
    print_result("Time", pyroid_time)
    
    # Print speedup
    print_speedup(python_time, pyroid_time)


async def demo_http():
    """Demo concurrent HTTP requests."""
    print_header("Demo 4: Concurrent HTTP Requests")
    
    # Generate URLs
    url_count = 50
    print(f"Preparing to fetch {url_count} URLs concurrently...")
    urls = [f"https://httpbin.org/get?id={i}" for i in range(url_count)]
    
    # Python asyncio
    import aiohttp
    print("\nPure Python (aiohttp):")
    start = time.time()
    
    async def fetch_all_python():
        async with aiohttp.ClientSession() as session:
            tasks = []
            for url in urls:
                tasks.append(asyncio.create_task(session.get(url)))
            responses = await asyncio.gather(*tasks)
            
            results = []
            for response in responses:
                text = await response.text()
                results.append({"status": response.status, "length": len(text)})
            return results
    
    python_result = await fetch_all_python()
    end = time.time()
    python_time = (end - start) * 1000
    print(f"Fetched {len(python_result)} URLs")
    print_result("Time", python_time)
    
    # pyroid AsyncClient
    print("\npyroid:")
    start = time.time()
    client = pyroid.AsyncClient()
    responses = await client.fetch_many(urls, concurrency=10)
    end = time.time()
    pyroid_time = (end - start) * 1000
    print(f"Fetched {len(responses)} URLs")
    print_result("Time", pyroid_time)
    
    # Print speedup
    print_speedup(python_time, pyroid_time)


def demo_data_pipeline():
    """Demo data processing pipeline."""
    print_header("Demo 5: Data Processing Pipeline")
    
    # Generate test data
    size = 500_000
    print(f"Generating {size:,} records of test data...")
    data = [{"id": i, "value": random.random(), "category": random.choice(["A", "B", "C", "D"])} for i in range(size)]
    print("Data generation complete.")
    
    # Python implementation
    print("\nPure Python:")
    python_time = None
    python_timed_out = False
    try:
        with time_limit(20):  # 20-second timeout
            start = time.time()
            
            # Step 1: Filter records where value > 0.5
            filtered = [item for item in data if item["value"] > 0.5]
            
            # Step 2: Transform values (multiply by 10)
            transformed = [{"id": item["id"], "value": item["value"] * 10, "category": item["category"]} for item in filtered]
            
            # Step 3: Group by category
            grouped = {}
            for item in transformed:
                category = item["category"]
                if category not in grouped:
                    grouped[category] = []
                grouped[category].append(item)
            
            # Step 4: Aggregate
            results = []
            for category, items in grouped.items():
                total = sum(item["value"] for item in items)
                count = len(items)
                results.append({"category": category, "total": total, "count": count, "average": total / count})
            
            # Step 5: Sort by average
            results.sort(key=lambda x: x["average"], reverse=True)
            
            end = time.time()
            python_time = (end - start) * 1000
            print(f"Processed {len(filtered):,} records after filtering")
            print(f"Results: {len(results)} categories")
            print_result("Time", python_time)
    except TimeoutException:
        python_timed_out = True
        print_result("Time", timed_out=True, timeout=20)
    
    # pyroid implementation
    print("\npyroid:")
    start = time.time()
    
    # Step 1: Filter records where value > 0.5
    filtered = pyroid.parallel_filter(data, lambda item: item["value"] > 0.5)
    
    # Step 2: Transform values (multiply by 10)
    transformed = pyroid.parallel_map(filtered, lambda item: {"id": item["id"], "value": item["value"] * 10, "category": item["category"]})
    
    # Step 3: Group by category (still using Python as pyroid doesn't have a direct equivalent)
    grouped = {}
    for item in transformed:
        category = item["category"]
        if category not in grouped:
            grouped[category] = []
        grouped[category].append(item)
    
    # Step 4: Aggregate using pyroid for each group
    results = []
    for category, items in grouped.items():
        values = pyroid.parallel_map(items, lambda item: item["value"])
        total = pyroid.parallel_sum(values)
        count = len(items)
        results.append({"category": category, "total": total, "count": count, "average": total / count})
    
    # Step 5: Sort by average
    results = pyroid.parallel_sort(results, lambda x: x["average"], True)
    
    end = time.time()
    pyroid_time = (end - start) * 1000
    print(f"Processed {len(filtered):,} records after filtering")
    print(f"Results: {len(results)} categories")
    print_result("Time", pyroid_time)
    
    # Print speedup
    print_speedup(python_time, pyroid_time, python_timed_out, 20)


async def main():
    """Run all demos."""
    if not PYROID_AVAILABLE:
        return
    
    print_header("Pyroid Performance Demo")
    print("This demo showcases Pyroid's performance advantages over pure Python implementations.")
    
    # Run demos
    demo_sum()
    demo_regex()
    demo_sort()
    await demo_http()
    demo_data_pipeline()
    
    print_header("Demo Complete")
    print("Pyroid consistently outperforms pure Python implementations across a variety of tasks.")
    print("For more detailed benchmarks, run the full benchmark suite with:")
    print("  python -m benchmarks.run_benchmarks")


if __name__ == "__main__":
    asyncio.run(main())