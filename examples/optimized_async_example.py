"""
Optimized async operations example for Pyroid.

This example demonstrates how to use Pyroid's optimized async operations,
zero-copy buffers, and parallel processing capabilities.
"""

import asyncio
import time
import numpy as np
from typing import List, Dict, Any

try:
    import pyroid
    from pyroid.core import buffer, parallel, runtime
except ImportError:
    print("Error: pyroid not found. Please install pyroid to run this example.")
    exit(1)


async def demonstrate_async_client():
    """Demonstrate the optimized AsyncClient."""
    print("\n=== Demonstrating AsyncClient ===")
    
    # Initialize the runtime
    runtime.init()
    print(f"Runtime initialized with {runtime.get_worker_threads()} worker threads")
    
    # Create an AsyncClient
    client = pyroid.AsyncClient()
    print("AsyncClient created")
    
    # Fetch a single URL
    print("\nFetching a single URL...")
    start_time = time.time()
    response = await client.fetch("https://httpbin.org/get")
    end_time = time.time()
    print(f"Fetched URL in {(end_time - start_time) * 1000:.2f}ms")
    print(f"Status code: {response['status']}")
    
    # Fetch multiple URLs concurrently
    urls = [f"https://httpbin.org/get?id={i}" for i in range(10)]
    print(f"\nFetching {len(urls)} URLs concurrently...")
    start_time = time.time()
    responses = await client.fetch_many(urls, concurrency=5)
    end_time = time.time()
    print(f"Fetched {len(responses)} URLs in {(end_time - start_time) * 1000:.2f}ms")
    print(f"Average time per URL: {(end_time - start_time) * 1000 / len(urls):.2f}ms")
    
    return "AsyncClient demonstration completed"


async def demonstrate_zero_copy_buffer():
    """Demonstrate the zero-copy buffer protocol."""
    print("\n=== Demonstrating Zero-Copy Buffer ===")
    
    # Create a zero-copy buffer
    buffer_size = 1024 * 1024  # 1MB
    zero_copy_buffer = buffer.ZeroCopyBuffer(buffer_size)
    print(f"Created a {buffer_size / 1024 / 1024:.1f}MB zero-copy buffer")
    
    # Fill the buffer with data
    data = zero_copy_buffer.get_data()
    for i in range(0, len(data), 4):
        if i + 4 <= len(data):
            data[i:i+4] = (i % 256).to_bytes(4, byteorder='little')
    
    # Update the buffer with the modified data
    zero_copy_buffer.set_data(data)
    print("Buffer filled with data")
    
    # Create a memory view
    memory_view = buffer.MemoryView(1024)
    memory_view.set_data(b"Hello, world!" * 78 + b"!")
    print(f"Created a memory view with size {memory_view.size()} bytes")
    
    # Get data from the memory view
    view_data = memory_view.get_data()
    print(f"First 20 bytes from memory view: {view_data[:20]}")
    
    return "Zero-copy buffer demonstration completed"


def demonstrate_parallel_processing():
    """Demonstrate parallel processing capabilities."""
    print("\n=== Demonstrating Parallel Processing ===")
    
    # Create a batch processor
    processor = parallel.BatchProcessor(batch_size=1000, adaptive=True)
    print("Created a batch processor with adaptive batch sizing")
    
    # Create a large list of items
    items = list(range(1000000))
    print(f"Created a list with {len(items)} items")
    
    # Define a processing function
    def process_item(x):
        # Simulate some CPU-bound work
        result = 0
        for i in range(100):
            result += (x * i) % 1000
        return result
    
    # Process items in parallel
    print("\nProcessing items in parallel...")
    start_time = time.time()
    results = processor.map(items, process_item)
    end_time = time.time()
    print(f"Processed {len(results)} items in {end_time - start_time:.2f} seconds")
    print(f"Average time per item: {(end_time - start_time) * 1000000 / len(items):.2f} nanoseconds")
    
    # Filter items in parallel
    print("\nFiltering items in parallel...")
    start_time = time.time()
    filtered = processor.filter(items, lambda x: x % 100 == 0)
    end_time = time.time()
    print(f"Filtered to {len(filtered)} items in {end_time - start_time:.2f} seconds")
    
    # Sort items in parallel
    print("\nSorting items in parallel...")
    unsorted = np.random.randint(0, 1000000, 100000).tolist()
    start_time = time.time()
    sorted_items = processor.sort(unsorted, key=lambda x: x, reverse=False)
    end_time = time.time()
    print(f"Sorted {len(sorted_items)} items in {end_time - start_time:.2f} seconds")
    
    return "Parallel processing demonstration completed"


async def run_complete_pipeline():
    """Run a complete pipeline using all optimized features."""
    print("\n=== Running Complete Optimized Pipeline ===")
    
    # Initialize the runtime
    runtime.init()
    
    # Create an AsyncClient
    client = pyroid.AsyncClient()
    
    # Create a batch processor
    processor = parallel.BatchProcessor(batch_size=1000, adaptive=True)
    
    # Step 1: Fetch data
    print("\nStep 1: Fetching data...")
    urls = [f"https://httpbin.org/get?id={i}" for i in range(20)]
    start_time = time.time()
    responses = await client.fetch_many(urls, concurrency=10)
    fetch_time = time.time() - start_time
    print(f"Fetched {len(responses)} URLs in {fetch_time:.2f} seconds")
    
    # Step 2: Process data in parallel
    print("\nStep 2: Processing data in parallel...")
    data = list(responses.values())
    
    def extract_data(response):
        if isinstance(response, dict) and 'text' in response:
            # Extract and process data
            return {
                'length': len(response['text']),
                'processed': True
            }
        return {'length': 0, 'processed': False}
    
    start_time = time.time()
    processed_data = processor.map(data, extract_data)
    process_time = time.time() - start_time
    print(f"Processed {len(processed_data)} items in {process_time:.2f} seconds")
    
    # Step 3: Store results using zero-copy buffer
    print("\nStep 3: Storing results...")
    result_str = str(processed_data).encode('utf-8')
    
    # Create a zero-copy buffer
    result_buffer = buffer.ZeroCopyBuffer.from_bytes(result_str)
    
    start_time = time.time()
    # Simulate writing to a file
    await asyncio.sleep(0.01)
    store_time = time.time() - start_time
    print(f"Stored {len(result_buffer.get_data())} bytes in {store_time:.2f} seconds")
    
    # Calculate total time
    total_time = fetch_time + process_time + store_time
    print(f"\nTotal pipeline time: {total_time:.2f} seconds")
    
    return "Complete pipeline demonstration completed"


async def main():
    """Run all demonstrations."""
    print("=== Pyroid Optimized Features Demonstration ===")
    
    # Demonstrate AsyncClient
    await demonstrate_async_client()
    
    # Demonstrate zero-copy buffer
    await demonstrate_zero_copy_buffer()
    
    # Demonstrate parallel processing
    demonstrate_parallel_processing()
    
    # Run complete pipeline
    await run_complete_pipeline()
    
    print("\n=== Demonstration Completed ===")


if __name__ == "__main__":
    asyncio.run(main())