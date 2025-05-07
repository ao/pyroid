#!/usr/bin/env python3
"""
Integration tests for Pyroid core functionality.

This script contains tests to verify that the core Rust extensions are working correctly.
"""

import unittest
import pyroid

class TestConfig(unittest.TestCase):
    """Test the Config class and related functionality."""
    
    def test_config_creation(self):
        """Test creating a Config object."""
        config = pyroid.Config({"parallel": True, "chunk_size": 1000})
        self.assertTrue(config.get("parallel"))
        self.assertEqual(config.get("chunk_size"), 1000)
    
    def test_config_context(self):
        """Test the ConfigContext context manager."""
        with pyroid.config(parallel=True, chunk_size=1000):
            # This test just verifies that the context manager doesn't raise an exception
            pass
    
    def test_shared_data(self):
        """Test the SharedData class."""
        data = [1, 2, 3, 4, 5]
        shared = pyroid.SharedData(data)
        # This test just verifies that SharedData can be created
        self.assertIsNotNone(shared)

class TestExceptions(unittest.TestCase):
    """Test the exception classes."""
    
    def test_exception_hierarchy(self):
        """Test the exception hierarchy."""
        self.assertTrue(issubclass(pyroid.InputError, pyroid.PyroidError))
        self.assertTrue(issubclass(pyroid.ComputationError, pyroid.PyroidError))
        self.assertTrue(issubclass(pyroid.MemoryError, pyroid.PyroidError))
        self.assertTrue(issubclass(pyroid.ConversionError, pyroid.PyroidError))
        self.assertTrue(issubclass(pyroid.IoError, pyroid.PyroidError))

if __name__ == "__main__":
    unittest.main()