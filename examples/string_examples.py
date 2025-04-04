#!/usr/bin/env python3
"""
String operation examples for pyroid.

This script demonstrates the string processing capabilities of pyroid.
"""

import time
import pyroid

def benchmark(func, *args, **kwargs):
    """Simple benchmarking function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")
    return result

def main():
    print("pyroid String Operations Examples")
    print("=================================")
    
    # Example 1: Parallel regex replacement
    print("\n1. Parallel Regex Replacement")
    text = "Hello world! " * 100000
    print(f"Input text length: {len(text)}")
    
    print("\nReplacing 'Hello' with 'Hi' using parallel_regex_replace:")
    result = benchmark(pyroid.parallel_regex_replace, text, r"Hello", "Hi")
    print(f"Output text length: {len(result)}")
    print(f"First 50 characters: {result[:50]}...")
    
    # Example 2: Parallel text cleanup
    print("\n2. Parallel Text Cleanup")
    texts = [
        "  Hello, World! ",
        "123 Testing 456",
        "Special @#$% Characters",
        "UPPERCASE text",
        "mixed CASE text"
    ] * 10000
    
    print(f"\nCleaning {len(texts)} texts in parallel:")
    results = benchmark(pyroid.parallel_text_cleanup, texts)
    print(f"Processed {len(results)} texts")
    print("Sample results:")
    for i in range(min(5, len(results))):
        print(f"  Original: '{texts[i]}'")
        print(f"  Cleaned:  '{results[i]}'")
    
    # Example 3: Base64 encoding/decoding
    print("\n3. Base64 Encoding/Decoding")
    data = "Hello, world! This is a test of base64 encoding and decoding." * 10000
    print(f"Original data length: {len(data)}")
    
    print("\nEncoding data using parallel_base64_encode:")
    encoded = benchmark(pyroid.parallel_base64_encode, data)
    print(f"Encoded data length: {len(encoded)}")
    print(f"First 50 characters: {encoded[:50]}...")
    
    # Make sure the encoded string is valid base64 by using a smaller test
    test_data = "Hello, world! This is a test of base64 encoding and decoding."
    test_encoded = pyroid.parallel_base64_encode(test_data)
    
    print("\nDecoding test data using parallel_base64_decode:")
    test_decoded = benchmark(pyroid.parallel_base64_decode, test_encoded)
    print(f"Decoded test data length: {len(test_decoded)}")
    print(f"Test data: {test_decoded}")
    
    # Verify the test result
    print(f"\nOriginal and test data match: {test_data == test_decoded}")

if __name__ == "__main__":
    main()