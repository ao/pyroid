#!/usr/bin/env python3
"""
Async operation examples for pyroid.

This script demonstrates the async capabilities of pyroid.
"""

import time
import asyncio
import os
import pyroid

async def benchmark_async(name, func, *args, **kwargs):
    """Benchmark an async function."""
    print(f"\nRunning {name}...")
    start_time = time.time()
    result = await func(*args, **kwargs)
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")
    return result

async def main():
    print("Pyroid Async Operations Examples")
    print("=============================")
    
    # Example 1: Async sleep
    print("\n1. Async Sleep")
    
    print("\nSleeping for 0.1 seconds using pyroid.io.sleep:")
    await benchmark_async("pyroid.io.sleep", pyroid.io.sleep, 0.1)
    
    print("\nSleeping for 0.1 seconds using asyncio.sleep:")
    await benchmark_async("asyncio.sleep", asyncio.sleep, 0.1)
    
    # Example 2: Async file operations
    print("\n2. Async File Operations")
    
    # Create a test file
    test_file = "async_test_file.txt"
    content = "Hello, Pyroid! This is a test file for async operations."
    
    # Write to the file using synchronous API
    pyroid.io.write_file(test_file, content)
    
    # Read the file asynchronously
    print("\nReading file asynchronously:")
    try:
        read_content = await benchmark_async("pyroid.io.read_file_async", pyroid.io.read_file_async, test_file)
        print(f"Content read: {len(read_content)} bytes")
        print(f"Content matches: {content == read_content}")
    except Exception as e:
        print(f"Async file read error: {e}")
    
    # Example 3: Concurrent async operations
    print("\n3. Concurrent Async Operations")
    
    # Create multiple test files
    test_files = []
    for i in range(5):
        file_name = f"async_test_file_{i}.txt"
        test_files.append(file_name)
        pyroid.io.write_file(file_name, f"This is test file {i} for concurrent async operations.")
    
    # Read all files concurrently
    print("\nReading multiple files concurrently:")
    try:
        async def read_file_async(file_path):
            content = await pyroid.io.read_file_async(file_path)
            return file_path, content
        
        tasks = [read_file_async(file) for file in test_files]
        results = await benchmark_async("asyncio.gather", asyncio.gather, *tasks)
        
        for file_path, content in results:
            print(f"{file_path}: {len(content)} bytes")
    except Exception as e:
        print(f"Concurrent async file read error: {e}")
    
    # Example 4: Async with timeout
    print("\n4. Async with Timeout")
    
    print("\nRunning async operation with timeout:")
    try:
        async def slow_operation():
            await pyroid.io.sleep(0.5)
            return "Operation completed"
        
        # Try with sufficient timeout
        result = await benchmark_async("asyncio.wait_for (sufficient timeout)", 
                                      lambda: asyncio.wait_for(slow_operation(), timeout=1.0))
        print(f"Result: {result}")
        
        # Try with insufficient timeout
        try:
            result = await benchmark_async("asyncio.wait_for (insufficient timeout)", 
                                          lambda: asyncio.wait_for(slow_operation(), timeout=0.1))
            print(f"Result: {result}")
        except asyncio.TimeoutError:
            print("Operation timed out as expected")
    except Exception as e:
        print(f"Async timeout error: {e}")
    
    # Example 5: Async network operations
    print("\n5. Async Network Operations")
    
    try:
        print("\nMaking async GET request:")
        async def async_get(url):
            try:
                # Check if async HTTP GET is implemented
                if hasattr(pyroid.io, "get_async"):
                    response = await pyroid.io.get_async(url)
                    return response
                else:
                    # Fallback to synchronous GET
                    print("Async GET not implemented, falling back to synchronous GET")
                    return pyroid.io.get(url)
            except Exception as e:
                print(f"Async GET error: {e}")
                return None
        
        response = await benchmark_async("async_get", async_get, "https://example.com")
        if response:
            print(f"Response length: {len(response)} bytes")
            print(f"Response preview: {response[:100]}...")
    except Exception as e:
        print(f"Async network operations error: {e}")
    
    # Example 6: Async context manager
    print("\n6. Async Context Manager")
    
    try:
        print("\nUsing async context manager:")
        
        class AsyncFileContext:
            def __init__(self, file_path, mode="r"):
                self.file_path = file_path
                self.mode = mode
                self.content = None
            
            async def __aenter__(self):
                if self.mode == "r":
                    self.content = await pyroid.io.read_file_async(self.file_path)
                return self
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                self.content = None
                return False
        
        async with AsyncFileContext(test_file) as file:
            print(f"File content length: {len(file.content)} bytes")
            print(f"File content preview: {file.content[:50]}...")
    except Exception as e:
        print(f"Async context manager error: {e}")
    
    # Clean up
    print("\nCleaning up test files...")
    os.remove(test_file)
    for file in test_files:
        os.remove(file)
    print("Cleanup complete.")
    
    print("\nAsync operations examples completed.")

if __name__ == "__main__":
    asyncio.run(main())