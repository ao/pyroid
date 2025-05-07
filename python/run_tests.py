#!/usr/bin/env python3
"""
Test runner for Pyroid Python implementation tests.

This script runs all the tests for the Python implementations of Pyroid modules.
It helps ensure that our Python fallback implementations behave correctly and
consistently with the Rust implementations.
"""

import unittest
import sys
import os
import argparse
import time

# Add the parent directory to the path so we can import pyroid
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import test modules
from tests.test_python_impl import TestCoreImpl
from tests.test_math_impl import TestMathImpl
from tests.test_data_impl import TestDataImpl
from tests.test_text_impl import TestTextImpl
from tests.test_io_impl import TestIOImpl
from tests.test_image_impl import TestImageImpl
from tests.test_ml_impl import TestMLImpl
from tests.test_async_impl import TestAsyncImpl

def run_tests(module_names=None, verbose=False):
    """
    Run the specified test modules.
    
    Args:
        module_names: List of module names to test, or None to test all modules
        verbose: Whether to print verbose output
    
    Returns:
        True if all tests pass, False otherwise
    """
    # Define all test modules
    all_modules = {
        "core": TestCoreImpl,
        "math": TestMathImpl,
        "data": TestDataImpl,
        "text": TestTextImpl,
        "io": TestIOImpl,
        "image": TestImageImpl,
        "ml": TestMLImpl,
        "async": TestAsyncImpl,
    }
    
    # Determine which modules to test
    if module_names:
        modules = {name: all_modules[name] for name in module_names if name in all_modules}
        if not modules:
            print(f"Error: No valid module names specified. Valid names are: {', '.join(all_modules.keys())}")
            return False
    else:
        modules = all_modules
    
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add tests from each module
    for name, module in modules.items():
        print(f"Adding tests for {name} module...")
        suite.addTest(unittest.makeSuite(module))
    
    # Run the tests
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print("\nTest Summary:")
    print(f"  Ran {result.testsRun} tests in {end_time - start_time:.2f} seconds")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    
    # Return True if all tests pass
    return len(result.failures) == 0 and len(result.errors) == 0

def check_implementation():
    """Check which implementation of pyroid is being used."""
    try:
        import pyroid
        print(f"Pyroid version: {getattr(pyroid, '__version__', 'Unknown')}")
        
        # Check if we're using the Python implementation
        try:
            import pyroid.core_impl
            print("Using Python implementation")
        except ImportError:
            print("Python implementation not available")
        
        # Check if we're using the Rust implementation
        try:
            import pyroid.pyroid
            print("Rust implementation available")
        except ImportError:
            print("Rust implementation not available")
    except ImportError:
        print("Pyroid is not installed")

def main():
    """Run the tests."""
    parser = argparse.ArgumentParser(description="Run tests for Pyroid Python implementations")
    parser.add_argument("--modules", nargs="+", help="Modules to test (core, math, data, text, io, image, ml, async)")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    args = parser.parse_args()
    
    print("Pyroid Python Implementation Tests")
    print("=================================\n")
    
    # Check which implementation we're using
    check_implementation()
    print()
    
    # Run the tests
    success = run_tests(args.modules, args.verbose)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()