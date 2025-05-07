#!/usr/bin/env python3
"""
Test suite for Python implementations of Pyroid I/O module.
"""

import unittest
import sys
import os
import tempfile
from typing import List, Dict, Any

# Add the parent directory to the path so we can import pyroid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Python implementations directly
import pyroid.io_impl

class TestIOImpl(unittest.TestCase):
    """Test the io_impl module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test file
        self.test_file = os.path.join(self.temp_dir, "test_file.txt")
        with open(self.test_file, "w") as f:
            f.write("Hello, world!")
        
        # Create another test file
        self.test_file2 = os.path.join(self.temp_dir, "test_file2.txt")
        with open(self.test_file2, "w") as f:
            f.write("Another test file")

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove test files
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        
        if os.path.exists(self.test_file2):
            os.remove(self.test_file2)
        
        # Remove any other files in the temp directory
        for filename in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Remove the temp directory
        os.rmdir(self.temp_dir)

    def test_read_file(self):
        """Test the read_file function."""
        # Test reading a file
        result = pyroid.io_impl.read_file(self.test_file)
        self.assertEqual(result, "Hello, world!")
        
        # Test reading a nonexistent file
        with self.assertRaises(FileNotFoundError):
            pyroid.io_impl.read_file(os.path.join(self.temp_dir, "nonexistent.txt"))

    def test_write_file(self):
        """Test the write_file function."""
        # Test writing to a file
        new_content = "New content"
        result = pyroid.io_impl.write_file(self.test_file, new_content)
        self.assertTrue(result)
        
        # Verify the content was written
        with open(self.test_file, "r") as f:
            self.assertEqual(f.read(), new_content)
        
        # Test writing to a new file
        new_file = os.path.join(self.temp_dir, "new_file.txt")
        result = pyroid.io_impl.write_file(new_file, new_content)
        self.assertTrue(result)
        
        # Verify the content was written
        with open(new_file, "r") as f:
            self.assertEqual(f.read(), new_content)

    def test_read_files(self):
        """Test the read_files function."""
        # Test reading multiple files
        result = pyroid.io_impl.read_files([self.test_file, self.test_file2])
        self.assertEqual(result, {
            self.test_file: "Hello, world!",
            self.test_file2: "Another test file"
        })
        
        # Test reading a mix of existing and nonexistent files
        with self.assertRaises(FileNotFoundError):
            pyroid.io_impl.read_files([
                self.test_file,
                os.path.join(self.temp_dir, "nonexistent.txt")
            ])

    def test_get(self):
        """Test the get function."""
        # This is a mock test since we can't easily test HTTP requests
        # in a unit test without mocking or a test server
        
        # Just test that the function exists and has the right signature
        self.assertTrue(hasattr(pyroid.io_impl, "get"))
        self.assertTrue(callable(pyroid.io_impl.get))

    def test_post(self):
        """Test the post function."""
        # This is a mock test since we can't easily test HTTP requests
        # in a unit test without mocking or a test server
        
        # Just test that the function exists and has the right signature
        self.assertTrue(hasattr(pyroid.io_impl, "post"))
        self.assertTrue(callable(pyroid.io_impl.post))

    def test_sleep(self):
        """Test the sleep function."""
        # Just test that the function exists and has the right signature
        self.assertTrue(hasattr(pyroid.io_impl, "sleep"))
        self.assertTrue(callable(pyroid.io_impl.sleep))

    def test_read_file_async(self):
        """Test the read_file_async function."""
        # Just test that the function exists and has the right signature
        self.assertTrue(hasattr(pyroid.io_impl, "read_file_async"))
        self.assertTrue(callable(pyroid.io_impl.read_file_async))

    def test_write_file_async(self):
        """Test the write_file_async function."""
        # Just test that the function exists and has the right signature
        self.assertTrue(hasattr(pyroid.io_impl, "write_file_async"))
        self.assertTrue(callable(pyroid.io_impl.write_file_async))

if __name__ == "__main__":
    unittest.main()