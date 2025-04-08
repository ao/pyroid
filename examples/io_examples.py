#!/usr/bin/env python3
"""
I/O operation examples for pyroid.

This script demonstrates the I/O capabilities of pyroid.
"""

import time
import os
import pyroid

def benchmark(func, *args, **kwargs):
    """Simple benchmarking function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")
    return result

def main():
    print("Pyroid I/O Operations Examples")
    print("===========================")
    
    # Example 1: File operations
    print("\n1. File Operations")
    
    # Create a temporary file
    temp_file = "temp_test_file.txt"
    content = "Hello, Pyroid! This is a test file.\nIt has multiple lines.\nTesting file I/O operations."
    
    # Write to the file
    print(f"\nWriting to {temp_file}:")
    pyroid.io.write_file(temp_file, content)
    print(f"Content written: {len(content)} bytes")
    
    # Read from the file
    print(f"\nReading from {temp_file}:")
    read_content = pyroid.io.read_file(temp_file)
    print(f"Content read: {len(read_content)} bytes")
    print(f"Content matches: {content == read_content}")
    
    # Create additional test files
    temp_file2 = "temp_test_file2.txt"
    temp_file3 = "temp_test_file3.txt"
    
    pyroid.io.write_file(temp_file2, "This is the second test file.")
    pyroid.io.write_file(temp_file3, "This is the third test file.")
    
    # Read multiple files
    print("\nReading multiple files:")
    files = [temp_file, temp_file2, temp_file3]
    contents = pyroid.io.read_files(files)
    
    for file_path, file_content in contents.items():
        print(f"{file_path}: {len(file_content)} bytes")
    
    # Example 2: Network operations
    print("\n2. Network Operations")
    
    # Make a GET request
    try:
        print("\nMaking a GET request to https://example.com:")
        response = pyroid.io.get("https://example.com")
        print(f"Response length: {len(response)} bytes")
        print(f"Response preview: {response[:100]}...")
    except Exception as e:
        print(f"Network operations error: {e}")
    
    # Example 3: Async operations
    print("\n3. Async Operations")
    
    # Define an async function
    async def test_async():
        print("\nTesting async sleep:")
        print("Sleeping for 0.1 seconds...")
        await pyroid.io.sleep(0.1)
        print("Awake!")
        
        print("\nTesting async file read:")
        try:
            content = await pyroid.io.read_file_async(temp_file)
            print(f"Async read content length: {len(content)} bytes")
        except Exception as e:
            print(f"Async file read error: {e}")
    
    # Run the async function
    try:
        import asyncio
        print("\nRunning async operations:")
        asyncio.run(test_async())
    except Exception as e:
        print(f"Async operations error: {e}")
    
    # Example 4: Performance comparison
    print("\n4. Performance Comparison")
    
    # Create a large file for benchmarking
    large_file = "large_test_file.txt"
    large_content = "X" * 1000000  # 1MB of data
    
    print(f"\nCreating a large file ({len(large_content)} bytes):")
    with open(large_file, "w") as f:
        f.write(large_content)
    
    # Benchmark Python's built-in file read
    print("\nPython built-in file read:")
    def python_read_file(path):
        with open(path, "r") as f:
            return f.read()
    
    python_result = benchmark(lambda: python_read_file(large_file))
    print(f"Result length: {len(python_result)} bytes")
    
    # Benchmark Pyroid's file read
    print("\nPyroid file read:")
    pyroid_result = benchmark(lambda: pyroid.io.read_file(large_file))
    print(f"Result length: {len(pyroid_result)} bytes")
    
    # Clean up
    print("\nCleaning up temporary files...")
    os.remove(temp_file)
    os.remove(temp_file2)
    os.remove(temp_file3)
    os.remove(large_file)
    print("Cleanup complete.")

if __name__ == "__main__":
    main()