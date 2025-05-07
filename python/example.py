#!/usr/bin/env python3
"""
Example script to test the Pyroid library.
"""

import pyroid
import asyncio

def test_core():
    """Test core functionality."""
    print("\n=== Testing Core ===")
    
    # Create a Config object
    config = pyroid.Config({"parallel": True, "chunk_size": 1000})
    print(f"Config: {config.options}")
    
    # Use the config context
    with pyroid.config(parallel=True, chunk_size=2000):
        print("Inside config context")
    
    # Create a SharedData object
    data = pyroid.SharedData([1, 2, 3, 4, 5])
    print(f"SharedData: {data.get()}")

def test_math():
    """Test math functionality."""
    print("\n=== Testing Math ===")
    
    # Create a Vector
    vector = pyroid.math.Vector([1, 2, 3])
    print(f"Vector: {vector.values}")
    
    # Create a Matrix
    matrix = pyroid.math.Matrix([[1, 2], [3, 4]])
    print(f"Matrix: {matrix.values}")
    
    # Calculate sum
    result = pyroid.math.sum([1, 2, 3, 4, 5])
    print(f"Sum: {result}")
    
    # Calculate mean
    result = pyroid.math.mean([1, 2, 3, 4, 5])
    print(f"Mean: {result}")

def test_data():
    """Test data functionality."""
    print("\n=== Testing Data ===")
    
    # Create a DataFrame
    df = pyroid.data.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    print(f"DataFrame: {df.to_dict()}")
    
    # Filter a list
    result = pyroid.data.filter([1, 2, 3, 4, 5], lambda x: x % 2 == 0)
    print(f"Filter: {result}")
    
    # Map a function
    result = pyroid.data.map([1, 2, 3, 4, 5], lambda x: x * 2)
    print(f"Map: {result}")

def test_text():
    """Test text functionality."""
    print("\n=== Testing Text ===")
    
    # Reverse a string
    result = pyroid.text.reverse("Hello, world!")
    print(f"Reverse: {result}")
    
    # Base64 encode
    result = pyroid.text.base64_encode("Hello, world!")
    print(f"Base64 encode: {result}")
    
    # Base64 decode
    result = pyroid.text.base64_decode(result)
    print(f"Base64 decode: {result}")

def test_io():
    """Test I/O functionality."""
    print("\n=== Testing I/O ===")
    
    # Write a file
    pyroid.io.write_file("test_file.txt", "Hello, world!")
    print("Wrote test_file.txt")
    
    # Read a file
    result = pyroid.io.read_file("test_file.txt")
    print(f"Read test_file.txt: {result}")

def test_image():
    """Test image functionality."""
    print("\n=== Testing Image ===")
    
    # Create an image
    image = pyroid.image.basic.create_image(10, 10, 3)
    print(f"Created image: {image.width}x{image.height}x{image.channels}")
    
    # Convert to grayscale
    gray = image.to_grayscale()
    print(f"Grayscale image: {gray.width}x{gray.height}x{gray.channels}")

def test_ml():
    """Test machine learning functionality."""
    print("\n=== Testing ML ===")
    
    # Normalize data
    data = [[1, 2], [3, 4], [5, 6]]
    result = pyroid.ml.basic.normalize(data)
    print(f"Normalized data: {result}")
    
    # Calculate distance matrix
    result = pyroid.ml.basic.distance_matrix(data)
    print(f"Distance matrix: {result}")

async def test_async():
    """Test async functionality."""
    print("\n=== Testing Async ===")
    
    # Sleep
    print("Sleeping for 1 second...")
    await pyroid.async_ops.sleep(1)
    print("Woke up!")
    
    # Read a file asynchronously
    result = await pyroid.async_ops.read_file_async("test_file.txt")
    print(f"Read test_file.txt asynchronously: {result}")

def main():
    """Run all tests."""
    print("Testing Pyroid Library")
    
    # Test core functionality
    test_core()
    
    # Test math functionality
    test_math()
    
    # Test data functionality
    test_data()
    
    # Test text functionality
    test_text()
    
    # Test I/O functionality
    test_io()
    
    # Test image functionality
    test_image()
    
    # Test machine learning functionality
    test_ml()
    
    # Test async functionality
    asyncio.run(test_async())
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()