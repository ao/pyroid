#!/usr/bin/env python3
"""
Async operation examples for pyroid.

This script demonstrates the async capabilities of pyroid using Tokio.
"""

import time
import asyncio
import aiohttp
import pyroid

async def python_fetch(url):
    """Fetch a URL using Python's aiohttp."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            text = await response.text()
            return {
                "status": response.status,
                "text": text
            }

async def benchmark_async(name, func, *args, **kwargs):
    """Benchmark an async function."""
    print(f"\nRunning {name}...")
    start_time = time.time()
    result = await func(*args, **kwargs)
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")
    return result

async def main():
    print("pyroid Async Operations Examples")
    print("===============================")
    
    # Example 1: AsyncClient for HTTP requests
    print("\n1. Async HTTP Client")
    
    # Create an AsyncClient
    client = pyroid.AsyncClient()
    
    # Single URL fetch
    url = "https://httpbin.org/get"
    print(f"\nFetching {url}")
    
    # Compare Pyroid vs Python's aiohttp
    rust_response = await benchmark_async("pyroid fetch", client.fetch, url)
    python_response = await benchmark_async("Python aiohttp", python_fetch, url)
    
    print(f"\npyroid status: {rust_response['status']}")
    print(f"Python status: {python_response['status']}")
    
    # Example 2: Fetch multiple URLs concurrently
    print("\n2. Concurrent URL Fetching")
    
    urls = [
        "https://httpbin.org/get",
        "https://httpbin.org/headers",
        "https://httpbin.org/user-agent",
        "https://httpbin.org/ip",
        "https://httpbin.org/uuid"
    ]
    
    print(f"\nFetching {len(urls)} URLs concurrently")
    
    # Pyroid fetch_many
    rust_responses = await benchmark_async("pyroid fetch_many", client.fetch_many, urls, 5)
    
    # Python's asyncio.gather
    async def fetch_all_python():
        return await asyncio.gather(*[python_fetch(url) for url in urls])
    
    python_responses = await benchmark_async("Python asyncio.gather", fetch_all_python)
    
    print(f"\nAll URLs fetched successfully with pyroid: {len(rust_responses) == len(urls)}")
    
    # Skip Example 3 (AsyncChannel) due to implementation issues
    
    # Example 3: Async sleep (skipping AsyncChannel)
    
    # Example 4: Async sleep
    print("\n4. Async Sleep")
    
    print("\nSleeping for 0.5 seconds using pyroid async_sleep")
    await benchmark_async("pyroid async_sleep", pyroid.async_sleep, 0.5)
    
    print("\nSleeping for 0.5 seconds using Python asyncio.sleep")
    await benchmark_async("Python asyncio.sleep", asyncio.sleep, 0.5)
    
    # Example 5: AsyncFileReader
    print("\n5. Async File Operations")
    
    # Create a test file
    test_file = "test_async_file.txt"
    with open(test_file, "w") as f:
        f.write("Line 1: Hello, world!\n")
        f.write("Line 2: This is a test file.\n")
        f.write("Line 3: For testing async file operations.\n")
    
    # Create an AsyncFileReader
    file_reader = pyroid.AsyncFileReader(test_file)
    
    # Read the entire file
    print("\nReading entire file")
    content = await benchmark_async("pyroid read_all", file_reader.read_all)
    print(f"Content length: {len(content)} bytes")
    
    # Read the file line by line
    print("\nReading file line by line")
    lines = await benchmark_async("pyroid read_lines", file_reader.read_lines)
    print(f"Number of lines: {len(lines)}")
    for line in lines:
        print(f"  {line}")
    
    # Example 6: Gather
    print("\n6. Gather")
    
    async def task1():
        await asyncio.sleep(0.2)
        return "Result from task 1"
    
    async def task2():
        await asyncio.sleep(0.1)
        return "Result from task 2"
    
    async def task3():
        await asyncio.sleep(0.3)
        return "Result from task 3"
    
    print("\nRunning multiple tasks concurrently with pyroid gather")
    results = await pyroid.gather([task1(), task2(), task3()])
    
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"Task {i+1}: {result}")

if __name__ == "__main__":
    asyncio.run(main())