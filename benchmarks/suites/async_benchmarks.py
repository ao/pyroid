"""
Async operation benchmarks for Pyroid.

This module provides benchmarks for comparing Pyroid's async operations with
pure Python implementations.
"""

import time
import asyncio
import aiohttp
import os

try:
    import pyroid
except ImportError:
    print("Warning: pyroid not found. Async benchmarks will not run correctly.")

from ..core.benchmark import Benchmark, BenchmarkResult
from ..core.reporter import BenchmarkReporter


async def benchmark_async(name, implementation, func, timeout, *args, **kwargs):
    """Benchmark an async function.
    
    Args:
        name: The name of the benchmark.
        implementation: The implementation being benchmarked (e.g., "Python", "pyroid").
        func: The async function to benchmark.
        timeout: The timeout in seconds.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
        
    Returns:
        A BenchmarkResult object containing the results of the benchmark.
    """
    try:
        start_time = time.time()
        result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        return BenchmarkResult(
            name=name, 
            implementation=implementation, 
            duration_ms=duration_ms, 
            result=result
        )
    except asyncio.TimeoutError:
        return BenchmarkResult(
            name=name, 
            implementation=implementation, 
            timed_out=True, 
            timeout_seconds=timeout
        )


async def python_fetch(url):
    """Fetch a URL using Python's aiohttp."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            text = await response.text()
            return {
                "status": response.status,
                "text": text
            }


async def run_async_benchmarks():
    """Run async benchmarks.
    
    Returns:
        List of Benchmark objects with results.
    """
    results = []
    
    # Example 1: Single URL fetch
    url = "https://httpbin.org/get"
    fetch_benchmark = Benchmark("Fetch single URL", "Fetch a single URL using async HTTP client")
    
    # Create an AsyncClient
    client = pyroid.AsyncClient()
    
    # Set timeouts
    python_timeout = 10
    pyroid_timeout = 10
    
    # Run benchmarks
    python_result = await benchmark_async("Python aiohttp", "Python", python_fetch, python_timeout, url)
    fetch_benchmark.results.append(python_result)
    
    pyroid_result = await benchmark_async("pyroid fetch", "pyroid", client.fetch, pyroid_timeout, url)
    fetch_benchmark.results.append(pyroid_result)
    
    BenchmarkReporter.print_results(fetch_benchmark)
    results.append(fetch_benchmark)
    
    # Example 2: Multiple URL fetch
    urls = [
        "https://httpbin.org/get",
        "https://httpbin.org/headers",
        "https://httpbin.org/user-agent",
        "https://httpbin.org/ip",
        "https://httpbin.org/uuid"
    ]
    
    # Duplicate the URLs to create a larger set (25 URLs)
    urls = urls * 5
    
    multi_fetch_benchmark = Benchmark(f"Fetch {len(urls)} URLs", f"Fetch {len(urls)} URLs concurrently")
    
    # Python's asyncio.gather
    async def fetch_all_python():
        return await asyncio.gather(*[python_fetch(url) for url in urls])
    
    python_multi_result = await benchmark_async("Python asyncio.gather", "Python", fetch_all_python, python_timeout)
    multi_fetch_benchmark.results.append(python_multi_result)
    
    # pyroid fetch_many
    pyroid_multi_result = await benchmark_async("pyroid fetch_many", "pyroid", client.fetch_many, pyroid_timeout, urls, 5)
    multi_fetch_benchmark.results.append(pyroid_multi_result)
    
    BenchmarkReporter.print_results(multi_fetch_benchmark)
    results.append(multi_fetch_benchmark)
    
    # Example 3: Async sleep
    sleep_benchmark = Benchmark("Async sleep", "Sleep for 0.5 seconds using async sleep")
    
    python_sleep_result = await benchmark_async("Python asyncio.sleep", "Python", asyncio.sleep, python_timeout, 0.5)
    sleep_benchmark.results.append(python_sleep_result)
    
    pyroid_sleep_result = await benchmark_async("pyroid async_sleep", "pyroid", pyroid.async_sleep, pyroid_timeout, 0.5)
    sleep_benchmark.results.append(pyroid_sleep_result)
    
    BenchmarkReporter.print_results(sleep_benchmark)
    results.append(sleep_benchmark)
    
    # Example 4: Async file operations
    # Create a test file
    test_file = "test_async_file.txt"
    with open(test_file, "w") as f:
        f.write("Line 1: Hello, world!\n")
        f.write("Line 2: This is a test file.\n")
        f.write("Line 3: For testing async file operations.\n")
    
    file_benchmark = Benchmark("Async file read", "Read a file asynchronously")
    
    # Python's async file read
    async def python_read_file():
        result = ""
        async with aiohttp.ClientSession() as session:
            with open(test_file, "r") as f:
                result = f.read()
        return result
    
    python_file_result = await benchmark_async("Python file read", "Python", python_read_file, python_timeout)
    file_benchmark.results.append(python_file_result)
    
    # pyroid AsyncFileReader
    file_reader = pyroid.AsyncFileReader(test_file)
    pyroid_file_result = await benchmark_async("pyroid read_all", "pyroid", file_reader.read_all, pyroid_timeout)
    file_benchmark.results.append(pyroid_file_result)
    
    BenchmarkReporter.print_results(file_benchmark)
    results.append(file_benchmark)
    
    # Example 5: Gather
    gather_benchmark = Benchmark("Gather tasks", "Run multiple tasks concurrently and gather results")
    
    async def task1():
        await asyncio.sleep(0.2)
        return "Result from task 1"
    
    async def task2():
        await asyncio.sleep(0.1)
        return "Result from task 2"
    
    async def task3():
        await asyncio.sleep(0.3)
        return "Result from task 3"
    
    # Python's asyncio.gather
    async def python_gather():
        return await asyncio.gather(task1(), task2(), task3())
    
    python_gather_result = await benchmark_async("Python asyncio.gather", "Python", python_gather, python_timeout)
    gather_benchmark.results.append(python_gather_result)
    
    # pyroid gather
    pyroid_gather_result = await benchmark_async("pyroid gather", "pyroid", pyroid.gather, pyroid_timeout, [task1(), task2(), task3()])
    gather_benchmark.results.append(pyroid_gather_result)
    
    BenchmarkReporter.print_results(gather_benchmark)
    results.append(gather_benchmark)
    
    return results


async def run_web_scraping_benchmark(urls_count=50):
    """Run a real-world web scraping benchmark.
    
    Args:
        urls_count: Number of URLs to scrape.
        
    Returns:
        A Benchmark object with results.
    """
    # Generate a list of URLs to scrape
    urls = [f"https://httpbin.org/get?id={i}" for i in range(urls_count)]
    
    web_benchmark = Benchmark("Web Scraping", f"Scrape {urls_count} URLs and process the results")
    
    # Python implementation
    async def python_web_scraping():
        # Fetch all URLs
        async with aiohttp.ClientSession() as session:
            tasks = []
            for url in urls:
                tasks.append(asyncio.create_task(session.get(url)))
            responses = await asyncio.gather(*tasks)
            
            # Process the responses
            results = []
            for response in responses:
                text = await response.text()
                # Extract some data
                data = {"url": str(response.url), "length": len(text)}
                results.append(data)
            
            # Sort by URL length
            results.sort(key=lambda x: len(x["url"]))
            
            return results
    
    # pyroid implementation
    async def pyroid_web_scraping():
        # Create an AsyncClient
        client = pyroid.AsyncClient()
        
        # Fetch all URLs
        responses = await client.fetch_many(urls, concurrency=10)
        
        # Process the responses
        results = []
        for url, response in responses.items():
            if isinstance(response, dict):
                # Extract some data
                data = {"url": url, "length": len(response.get("text", ""))}
                results.append(data)
        
        # Sort by URL length
        results = pyroid.parallel_sort(results, lambda x: len(x["url"]), False)
        
        return results
    
    # Set timeouts
    python_timeout = 30
    pyroid_timeout = 30
    
    # Run benchmarks
    python_result = await benchmark_async("Python web scraping", "Python", python_web_scraping, python_timeout)
    web_benchmark.results.append(python_result)
    
    pyroid_result = await benchmark_async("pyroid web scraping", "pyroid", pyroid_web_scraping, pyroid_timeout)
    web_benchmark.results.append(pyroid_result)
    
    BenchmarkReporter.print_results(web_benchmark)
    
    return web_benchmark


if __name__ == "__main__":
    print("Running async benchmarks...")
    asyncio.run(run_async_benchmarks())
    
    print("\nRunning web scraping benchmark...")
    asyncio.run(run_web_scraping_benchmark())