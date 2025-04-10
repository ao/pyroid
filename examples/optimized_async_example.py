"""
Optimized async example for pyroid.

This example demonstrates the optimized async operations in pyroid.
"""

import asyncio
import time
import sys
import os
from typing import List, Dict, Any

# Add the parent directory to the path so we can import pyroid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import pyroid
    from pyroid.core import runtime
except ImportError:
    print("pyroid not found. Make sure it's installed or built.")
    sys.exit(1)

# Initialize the runtime
runtime.init()

# URLs for testing
TEST_URLS = [
    "https://httpbin.org/get?id=1",
    "https://httpbin.org/get?id=2",
    "https://httpbin.org/get?id=3",
    "https://httpbin.org/get?id=4",
    "https://httpbin.org/get?id=5",
    "https://httpbin.org/get?id=6",
    "https://httpbin.org/get?id=7",
    "https://httpbin.org/get?id=8",
    "https://httpbin.org/get?id=9",
    "https://httpbin.org/get?id=10",
]

# URLs grouped by host for more realistic testing
MIXED_URLS = [
    "https://httpbin.org/get?id=1",
    "https://httpbin.org/get?id=2",
    "https://example.com/",
    "https://httpbin.org/get?id=3",
    "https://example.org/",
    "https://httpbin.org/get?id=4",
    "https://example.net/",
    "https://httpbin.org/get?id=5",
]

async def fetch_with_standard_python(urls: List[str], concurrency: int = 10) -> Dict[str, Any]:
    """Fetch URLs using standard Python async/await with aiohttp."""
    import aiohttp
    
    results = {}
    semaphore = asyncio.Semaphore(concurrency)
    
    async def fetch_url(url: str) -> None:
        async with semaphore:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        results[url] = {
                            "status": response.status,
                            "text": await response.text()
                        }
            except Exception as e:
                results[url] = {"error": str(e)}
    
    await asyncio.gather(*(fetch_url(url) for url in urls))
    return results
async def fetch_with_pyroid(urls: List[str], concurrency: int = 10) -> Dict[str, Any]:
    """Fetch URLs using pyroid's AsyncClient."""
    client = pyroid.AsyncClient(concurrency=concurrency, adaptive_concurrency=True)
    return await client.fetch_many(urls, concurrency)
    return client.fetch_many(urls, concurrency)

async def benchmark_fetch(urls: List[str], concurrency: int = 10, iterations: int = 5) -> None:
    """Benchmark URL fetching with both standard Python and pyroid."""
    print(f"Benchmarking URL fetching with concurrency={concurrency}, iterations={iterations}")
    print(f"URLs: {len(urls)}")
    
    # Warm up
    print("Warming up...")
    await fetch_with_standard_python(urls[:2], concurrency)
    await fetch_with_pyroid(urls[:2], concurrency)
    
    # Benchmark standard Python
    python_times = []
    for i in range(iterations):
        start_time = time.time()
        await fetch_with_standard_python(urls, concurrency)
        elapsed = time.time() - start_time
        python_times.append(elapsed)
        print(f"Python iteration {i+1}: {elapsed:.4f}s")
    
    avg_python_time = sum(python_times) / len(python_times)
    print(f"Average Python time: {avg_python_time:.4f}s")
    
    # Benchmark pyroid
    pyroid_times = []
    for i in range(iterations):
        start_time = time.time()
        await fetch_with_pyroid(urls, concurrency)
        elapsed = time.time() - start_time
        pyroid_times.append(elapsed)
        print(f"pyroid iteration {i+1}: {elapsed:.4f}s")
    
    avg_pyroid_time = sum(pyroid_times) / len(pyroid_times)
    print(f"Average pyroid time: {avg_pyroid_time:.4f}s")
    
    # Calculate speedup
    speedup = avg_python_time / avg_pyroid_time if avg_pyroid_time > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x")
    print()

async def benchmark_zero_copy_buffer() -> None:
    """Benchmark zero-copy buffer operations."""
    print("Benchmarking zero-copy buffer operations")
    
    # Create a large buffer
    data_size = 10 * 1024 * 1024  # 10 MB
    data = b"x" * data_size
    
    # Standard Python approach
    start_time = time.time()
    for _ in range(100):
        # Copy the data
        data_copy = data[:]
        # Do something with the copy
        _ = len(data_copy)
    python_time = time.time() - start_time
    print(f"Python time: {python_time:.4f}s")
    
    # pyroid approach
    start_time = time.time()
    for _ in range(100):
        # Create a zero-copy buffer
        buffer = pyroid.core.buffer.ZeroCopyBuffer.from_bytes(data)
        # Do something with the buffer
        _ = buffer.size
    pyroid_time = time.time() - start_time
    print(f"pyroid time: {pyroid_time:.4f}s")
    
    # Calculate speedup
    speedup = python_time / pyroid_time if pyroid_time > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x")
    print()

async def benchmark_parallel_processing() -> None:
    """Benchmark parallel processing."""
    print("Benchmarking parallel processing")
    
    # Create a large list of items
    items = list(range(1000000))
    
    # Define a CPU-intensive function
    def process_item(x):
        # Simulate CPU-intensive work
        result = x
        for _ in range(100):
            result = (result * 2) % 10000007
        return result
    
    # Standard Python approach
    start_time = time.time()
    python_results = list(map(process_item, items[:10000]))  # Process fewer items for Python
    python_time = time.time() - start_time
    print(f"Python time (10K items): {python_time:.4f}s")
    
    # Extrapolate to full dataset
    extrapolated_python_time = python_time * (len(items) / 10000)
    print(f"Extrapolated Python time (1M items): {extrapolated_python_time:.4f}s")
    
    # pyroid approach with BatchProcessor
    batch_processor = pyroid.core.parallel.BatchProcessor(batch_size=10000, adaptive=True)
    
    start_time = time.time()
    pyroid_results = batch_processor.map(items, process_item)
    pyroid_time = time.time() - start_time
    print(f"pyroid time (1M items): {pyroid_time:.4f}s")
    
    # Calculate speedup based on extrapolated time
    speedup = extrapolated_python_time / pyroid_time if pyroid_time > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x")
    print()

async def main() -> None:
    """Run all benchmarks."""
    print("=== pyroid Optimized Async Benchmarks ===\n")
    
    # Benchmark URL fetching with different concurrency levels
    await benchmark_fetch(TEST_URLS, concurrency=2)
    await benchmark_fetch(TEST_URLS, concurrency=5)
    await benchmark_fetch(TEST_URLS, concurrency=10)
    
    # Benchmark URL fetching with mixed hosts
    await benchmark_fetch(MIXED_URLS, concurrency=5)
    
    # Benchmark zero-copy buffer operations
    await benchmark_zero_copy_buffer()
    
    # Benchmark parallel processing
    await benchmark_parallel_processing()

if __name__ == "__main__":
    asyncio.run(main())