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
    print("Pyroid String Operations Examples")
    print("===============================")
    
    # Example 1: Basic string operations
    print("\n1. Basic String Operations")
    
    # Create a test string
    text = "Hello, World!"
    print(f"Original text: {text}")
    
    # Reverse the string
    reversed_text = pyroid.text.reverse(text)
    print(f"Reversed: {reversed_text}")
    
    # Convert to uppercase
    upper = pyroid.text.to_uppercase(text)
    print(f"Uppercase: {upper}")
    
    # Convert to lowercase
    lower = pyroid.text.to_lowercase(text)
    print(f"Lowercase: {lower}")
    
    # Example 2: String manipulation
    print("\n2. String Manipulation")
    
    # Split the string
    words = pyroid.text.split(text, " ")
    print(f"Split by space: {words}")
    
    # Join the words
    joined = pyroid.text.join(words, "-")
    print(f"Joined with hyphens: {joined}")
    
    # Regex replace
    replaced = pyroid.text.regex_replace(text, r"World", "Python")
    print(f"Regex replace 'World' with 'Python': {replaced}")
    
    # Example 3: Base64 encoding/decoding
    print("\n3. Base64 Encoding/Decoding")
    
    # Encode to base64
    encoded = pyroid.text.base64_encode(text)
    print(f"Base64 encoded: {encoded}")
    
    # Decode from base64
    decoded = pyroid.text.base64_decode(encoded)
    print(f"Base64 decoded: {decoded}")
    
    # Example 4: NLP operations
    print("\n4. NLP Operations")
    
    # Tokenize the text
    tokens = pyroid.text.tokenize(text)
    print(f"Tokenized: {tokens}")
    
    # Generate n-grams
    ngrams = pyroid.text.ngrams(text, 2)
    print(f"Bigrams: {ngrams}")
    
    # Example 5: Using the string module (alias for text)
    print("\n5. String Module (Alias for Text)")
    
    # Convert to uppercase using string module
    upper_str = pyroid.string.to_uppercase(text)
    print(f"Uppercase (string module): {upper_str}")
    
    # Convert to lowercase using string module
    lower_str = pyroid.string.to_lowercase(text)
    print(f"Lowercase (string module): {lower_str}")
    
    # Example 6: Performance comparison
    print("\n6. Performance Comparison")
    
    # Create a large text for benchmarking
    large_text = "Hello, World! " * 10000
    print(f"Large text length: {len(large_text)}")
    
    # Benchmark Python's built-in reverse
    print("\nPython built-in reverse:")
    python_result = benchmark(lambda: large_text[::-1])
    print(f"Result length: {len(python_result)}")
    
    # Benchmark Pyroid's reverse
    print("\nPyroid reverse:")
    pyroid_result = benchmark(lambda: pyroid.text.reverse(large_text))
    print(f"Result length: {len(pyroid_result)}")
    
    # Benchmark Python's built-in uppercase
    print("\nPython built-in uppercase:")
    python_result = benchmark(lambda: large_text.upper())
    print(f"Result length: {len(python_result)}")
    
    # Benchmark Pyroid's uppercase
    print("\nPyroid uppercase:")
    pyroid_result = benchmark(lambda: pyroid.text.to_uppercase(large_text))
    print(f"Result length: {len(pyroid_result)}")

if __name__ == "__main__":
    main()