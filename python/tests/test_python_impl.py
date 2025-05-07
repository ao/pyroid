#!/usr/bin/env python3
"""
Test suite for Python implementations of Pyroid modules.

This ensures that our Python fallback implementations behave the same way
as the Rust implementations, so we can catch any regressions or inconsistencies.
"""

import unittest
import sys
import os
import random
import numpy as np
from typing import List, Dict, Any

# Add the parent directory to the path so we can import pyroid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Python implementations directly
import pyroid.core_impl
import pyroid.math_impl
import pyroid.data_impl
import pyroid.text_impl
import pyroid.io_impl
import pyroid.image_impl
import pyroid.ml_impl
import pyroid.async_impl

class TestCoreImpl(unittest.TestCase):
    """Test the core_impl module."""

    def test_config(self):
        """Test the Config class."""
        # Create a Config object
        config = pyroid.core_impl.Config({"parallel": True, "chunk_size": 1000})
        
        # Test get method
        self.assertEqual(config.get("parallel"), True)
        self.assertEqual(config.get("chunk_size"), 1000)
        self.assertEqual(config.get("nonexistent"), None)
        self.assertEqual(config.get("nonexistent", "default"), "default")
        
        # Test set method
        config.set("new_option", "value")
        self.assertEqual(config.get("new_option"), "value")
        
        # Test updating an existing option
        config.set("parallel", False)
        self.assertEqual(config.get("parallel"), False)

    def test_config_context(self):
        """Test the ConfigContext class."""
        # Create a Config object
        config = pyroid.core_impl.Config({"parallel": True})
        
        # Create a ConfigContext
        context = pyroid.core_impl.ConfigContext(config)
        
        # Test context manager
        with context:
            # Inside the context, the thread-local config should be set
            thread_local_config = pyroid.core_impl._thread_local.config
            self.assertEqual(thread_local_config.get("parallel"), True)
        
        # Outside the context, the thread-local config should be None
        self.assertFalse(hasattr(pyroid.core_impl._thread_local, 'config'))

    def test_shared_data(self):
        """Test the SharedData class."""
        # Create a SharedData object
        data = [1, 2, 3, 4, 5]
        shared = pyroid.core_impl.SharedData(data)
        
        # Test get method
        self.assertEqual(shared.get(), data)
        
        # Test that the data is actually shared (reference, not copy)
        data.append(6)
        self.assertEqual(shared.get(), [1, 2, 3, 4, 5, 6])

    def test_config_function(self):
        """Test the config function."""
        # Use the config function
        with pyroid.core_impl.config(parallel=True, chunk_size=1000):
            # Inside the context, the thread-local config should be set
            thread_local_config = pyroid.core_impl._thread_local.config
            self.assertEqual(thread_local_config.get("parallel"), True)
            self.assertEqual(thread_local_config.get("chunk_size"), 1000)
        
        # Outside the context, the thread-local config should be None
        self.assertFalse(hasattr(pyroid.core_impl._thread_local, 'config'))

    def test_get_config(self):
        """Test the get_config function."""
        # Without a thread-local config, should return the global config
        config = pyroid.core_impl.get_config()
        self.assertEqual(config, pyroid.core_impl._global_config)
        
        # With a thread-local config, should return that
        with pyroid.core_impl.config(parallel=True):
            config = pyroid.core_impl.get_config()
            self.assertEqual(config.get("parallel"), True)

if __name__ == "__main__":
    unittest.main()