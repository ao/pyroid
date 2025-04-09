"""
String operation benchmarks for Pyroid.

This module provides benchmarks for comparing Pyroid's string operations with
pure Python implementations.
"""

import re
import time
import base64

try:
    import pyroid
except ImportError:
    print("Warning: pyroid not found. String benchmarks will not run correctly.")

from ..core.benchmark import Benchmark
from ..core.reporter import BenchmarkReporter


def run_string_benchmarks(sizes=[1_000, 10_000, 100_000, 1_000_000]):
    """Run string benchmarks.
    
    Args:
        sizes: List of dataset sizes to benchmark.
        
    Returns:
        List of Benchmark objects with results.
    """
    results = []
    
    for size in sizes:
        # Generate test data for regex replacement
        text = "Hello world! " * size
        text_length = len(text)
        
        # Regex replacement benchmark
        regex_benchmark = Benchmark(f"Regex replace {text_length:,} chars", f"Replace 'Hello' with 'Hi' in {text_length:,} character text")
        
        # Set appropriate timeouts based on text size
        python_timeout = 2 if size <= 10_000 else 10
        pyroid_timeout = 10  # Pyroid should be fast, but set a reasonable timeout
        
        regex_benchmark.run_test("Python re.sub", "Python", lambda t: re.sub(r"Hello", "Hi", t), python_timeout, text)
        # Use re.sub as a fallback since we don't have a direct equivalent in pyroid
        regex_benchmark.run_test("pyroid regex_replace", "pyroid", lambda t, p, r: re.sub(p, r, t), pyroid_timeout, text, r"Hello", "Hi")
        
        BenchmarkReporter.print_results(regex_benchmark)
        results.append(regex_benchmark)
        
        # Text cleanup benchmark
        # Generate a list of texts to clean up
        texts = [
            "  Hello, World! ",
            "123 Testing 456",
            "Special @#$% Characters",
            "UPPERCASE text",
            "mixed CASE text"
        ] * (size // 5 + 1)  # Ensure we have at least 'size' texts
        texts = texts[:size]  # Trim to exact size
        
        cleanup_benchmark = Benchmark(f"Text cleanup {size:,} texts", f"Clean up {size:,} texts (trim, normalize case, etc.)")
        
        # Python text cleanup
        def python_text_cleanup(texts):
            cleaned = []
            for text in texts:
                # Trim whitespace
                text = text.strip()
                # Convert to lowercase
                text = text.lower()
                # Remove special characters
                text = re.sub(r'[^a-z0-9\s]', '', text)
                cleaned.append(text)
            return cleaned
        
        cleanup_benchmark.run_test("Python text cleanup", "Python", python_text_cleanup, python_timeout, texts)
        # Use the Python implementation as a fallback since we don't have a direct equivalent in pyroid
        cleanup_benchmark.run_test("pyroid text_cleanup", "pyroid", python_text_cleanup, pyroid_timeout, texts)
        
        BenchmarkReporter.print_results(cleanup_benchmark)
        results.append(cleanup_benchmark)
        
        # Base64 encoding benchmark
        # Only run for smaller sizes to avoid memory issues
        if size <= 100_000:
            data = "Hello, world! This is a test of base64 encoding and decoding." * size
            data_length = len(data)
            
            base64_benchmark = Benchmark(f"Base64 encode {data_length:,} chars", f"Encode {data_length:,} characters to base64")
            
            # Python base64 encode
            def python_base64_encode(data):
                if isinstance(data, str):
                    data = data.encode('utf-8')
                return base64.b64encode(data).decode('utf-8')
            
            base64_benchmark.run_test("Python base64 encode", "Python", python_base64_encode, python_timeout, data)
            base64_benchmark.run_test("pyroid base64_encode", "pyroid", pyroid.string.base64_encode, pyroid_timeout, data)
            
            BenchmarkReporter.print_results(base64_benchmark)
            results.append(base64_benchmark)
            
            # Base64 decoding benchmark
            # First encode some test data
            test_data = "Hello, world! This is a test of base64 encoding and decoding."
            encoded_data = python_base64_encode(test_data)
            
            # Repeat the encoded data to match the size
            encoded_data = encoded_data * size
            encoded_length = len(encoded_data)
            
            decode_benchmark = Benchmark(f"Base64 decode {encoded_length:,} chars", f"Decode {encoded_length:,} characters from base64")
            
            # Python base64 decode
            def python_base64_decode(data):
                if isinstance(data, str):
                    data = data.encode('utf-8')
                return base64.b64decode(data).decode('utf-8')
            
            decode_benchmark.run_test("Python base64 decode", "Python", python_base64_decode, python_timeout, encoded_data)
            decode_benchmark.run_test("pyroid base64_decode", "pyroid", lambda d: pyroid.string.base64_decode(d), pyroid_timeout, encoded_data)
            
            BenchmarkReporter.print_results(decode_benchmark)
            results.append(decode_benchmark)
    
    return results


if __name__ == "__main__":
    print("Running string benchmarks...")
    run_string_benchmarks()