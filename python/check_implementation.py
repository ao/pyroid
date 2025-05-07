#!/usr/bin/env python3
"""
Script to check whether you're using the Rust implementation or the Python fallback.
This helps users verify that they have the high-performance Rust extensions installed.
"""

import sys
import os
import importlib
import inspect
import time
import random

def check_module_source(module_name):
    """Check if a module is implemented in Python or Rust."""
    try:
        module = importlib.import_module(module_name)
        
        # Try to get the file path
        try:
            file_path = inspect.getfile(module)
            if file_path.endswith(('.py', '.pyc')):
                return "Python"
            elif file_path.endswith(('.so', '.pyd', '.dll')):
                return "Rust"
            else:
                return f"Unknown ({file_path})"
        except TypeError:
            # This typically happens for built-in modules, which includes Rust extensions
            return "Rust (built-in)"
    except ImportError:
        return "Not available"

def check_implementation_by_attribute(module_name, attribute):
    """Check implementation by looking for Python-specific attributes."""
    try:
        module = importlib.import_module(module_name)
        obj = getattr(module, attribute)
        
        # Check if this is a Python class with Python-specific attributes
        if hasattr(obj, "__dict__") and isinstance(obj.__dict__, dict):
            return "Python"
        else:
            return "Rust"
    except (ImportError, AttributeError):
        return "Not available"

def run_performance_test():
    """Run a simple performance test to verify implementation."""
    try:
        import pyroid
        
        # Generate test data
        data_size = 100000
        data = {"values": [random.random() for _ in range(data_size)]}
        
        # Test DataFrame creation
        start = time.time()
        df = pyroid.data.DataFrame(data)
        creation_time = time.time() - start
        
        # Test filtering
        start = time.time()
        filtered = pyroid.data.filter(df["values"], lambda x: x > 0.5)
        filter_time = time.time() - start
        
        # Compare with expected performance
        if creation_time < 0.0001 and filter_time < 0.001:
            return "Performance matches expected Rust implementation"
        else:
            return f"Performance suggests Python implementation (creation: {creation_time:.6f}s, filter: {filter_time:.6f}s)"
    except Exception as e:
        return f"Error running performance test: {str(e)}"

def main():
    """Check which implementation of pyroid is being used."""
    print("Pyroid Implementation Check")
    print("==========================\n")
    
    # Check if pyroid is installed
    try:
        import pyroid
        print(f"Pyroid version: {getattr(pyroid, '__version__', 'Unknown')}")
    except ImportError:
        print("Pyroid is not installed. Please install it first.")
        return
    
    # Check core modules
    modules = [
        ("pyroid.pyroid", "Core Rust module"),
        ("pyroid.core_impl", "Core Python fallback"),
        ("pyroid.math_impl", "Math Python fallback"),
        ("pyroid.data_impl", "Data Python fallback"),
        ("pyroid.text_impl", "Text Python fallback"),
        ("pyroid.io_impl", "I/O Python fallback"),
        ("pyroid.image_impl", "Image Python fallback"),
        ("pyroid.ml_impl", "ML Python fallback"),
        ("pyroid.async_impl", "Async Python fallback"),
    ]
    
    print("\nModule Availability:")
    print("------------------")
    for module_name, description in modules:
        try:
            importlib.import_module(module_name)
            status = "Available"
        except ImportError:
            status = "Not available"
        print(f"{description}: {status}")
    
    # Check implementation of key classes
    classes = [
        ("pyroid.data", "DataFrame", "DataFrame"),
        ("pyroid.math", "Vector", "Vector"),
        ("pyroid.text", "reverse", "Text operations"),
        ("pyroid.io", "read_file", "I/O operations"),
        ("pyroid.async_ops", "AsyncClient", "Async operations"),
    ]
    
    print("\nImplementation Check:")
    print("-------------------")
    for module_name, attribute, description in classes:
        impl = check_implementation_by_attribute(module_name, attribute)
        print(f"{description}: {impl}")
    
    # Run performance test
    print("\nPerformance Check:")
    print("----------------")
    perf_result = run_performance_test()
    print(perf_result)
    
    # Provide conclusion
    print("\nConclusion:")
    print("-----------")
    try:
        import pyroid.pyroid
        # Check if performance matches Rust expectations
        if "Python implementation" in perf_result:
            print("You have the Rust module installed, but performance tests indicate")
            print("you may be using the Python fallback implementation for some operations.")
            print("This could be due to a partial or incomplete Rust installation.")
            print("\nFor optimal performance, reinstall the Rust components:")
            print("    python build_and_install.py")
        else:
            print("You are using the high-performance Rust implementation of Pyroid.")
    except ImportError:
        print("You are using the Python fallback implementation of Pyroid.")
        print("For optimal performance, install the Rust components:")
        print("    python build_and_install.py")

if __name__ == "__main__":
    main()