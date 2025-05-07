#!/usr/bin/env python3
"""
Integration tests for Pyroid I/O functionality.

This script contains tests to verify that the I/O Rust extensions are working correctly.
"""

import unittest
import os
import tempfile
import asyncio
import pyroid

class TestFileOperations(unittest.TestCase):
    """Test the file operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test_file.txt")
        self.test_content = "Hello, world!\nThis is a test file.\n"
        
        # Write test content to the file
        with open(self.test_file, "w") as f:
            f.write(self.test_content)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove test files
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        
        # Remove test directory
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)
    
    def test_read_file(self):
        """Test reading a file."""
        content = pyroid.io.read_file(self.test_file)
        self.assertEqual(content, self.test_content)
    
    def test_write_file(self):
        """Test writing a file."""
        new_file = os.path.join(self.test_dir, "new_file.txt")
        new_content = "This is a new file.\n"
        
        result = pyroid.io.write_file(new_file, new_content)
        self.assertTrue(result)
        
        # Verify the content was written correctly
        with open(new_file, "r") as f:
            content = f.read()
        self.assertEqual(content, new_content)
        
        # Clean up
        if os.path.exists(new_file):
            os.remove(new_file)
    
    def test_read_files(self):
        """Test reading multiple files."""
        # Create additional test files
        files = []
        contents = []
        for i in range(3):
            file_path = os.path.join(self.test_dir, f"test_file_{i}.txt")
            content = f"Content of file {i}\n"
            with open(file_path, "w") as f:
                f.write(content)
            files.append(file_path)
            contents.append(content)
        
        # Read all files
        result = pyroid.io.read_files(files)
        
        # Verify results
        for i, file_path in enumerate(files):
            self.assertEqual(result[file_path], contents[i])
        
        # Clean up
        for file_path in files:
            if os.path.exists(file_path):
                os.remove(file_path)

class TestNetworkOperations(unittest.TestCase):
    """Test the network operations."""
    
    @unittest.skip("Network tests require an internet connection")
    def test_get(self):
        """Test HTTP GET request."""
        url = "https://httpbin.org/get"
        result = pyroid.io.get(url)
        self.assertIsNotNone(result)
        self.assertIn("url", result)
    
    @unittest.skip("Network tests require an internet connection")
    def test_post(self):
        """Test HTTP POST request."""
        url = "https://httpbin.org/post"
        data = {"key": "value"}
        result = pyroid.io.post(url, data)
        self.assertIsNotNone(result)
        self.assertIn("json", result)
        self.assertEqual(result["json"]["key"], "value")

class TestAsyncOperations(unittest.TestCase):
    """Test the async operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test_file.txt")
        self.test_content = "Hello, world!\nThis is a test file.\n"
        
        # Write test content to the file
        with open(self.test_file, "w") as f:
            f.write(self.test_content)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove test files
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        
        # Remove test directory
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)
    
    def test_sleep(self):
        """Test async sleep."""
        async def test_sleep_coroutine():
            start_time = asyncio.get_event_loop().time()
            await pyroid.io.sleep(0.1)
            end_time = asyncio.get_event_loop().time()
            return end_time - start_time
        
        # Run the coroutine
        loop = asyncio.get_event_loop()
        duration = loop.run_until_complete(test_sleep_coroutine())
        
        # Verify that at least 0.1 seconds passed
        self.assertGreaterEqual(duration, 0.1)
    
    def test_read_file_async(self):
        """Test async file reading."""
        async def test_read_file_coroutine():
            content = await pyroid.io.read_file_async(self.test_file)
            return content
        
        # Run the coroutine
        loop = asyncio.get_event_loop()
        content = loop.run_until_complete(test_read_file_coroutine())
        
        # Verify the content
        self.assertEqual(content, self.test_content)
    
    def test_write_file_async(self):
        """Test async file writing."""
        async def test_write_file_coroutine():
            new_file = os.path.join(self.test_dir, "new_file_async.txt")
            new_content = "This is a new file (async).\n"
            
            result = await pyroid.io.write_file_async(new_file, new_content)
            
            # Verify the content was written correctly
            with open(new_file, "r") as f:
                content = f.read()
            
            # Clean up
            if os.path.exists(new_file):
                os.remove(new_file)
                
            return result, content == new_content
        
        # Run the coroutine
        loop = asyncio.get_event_loop()
        result, content_match = loop.run_until_complete(test_write_file_coroutine())
        
        # Verify the result
        self.assertTrue(result)
        self.assertTrue(content_match)

if __name__ == "__main__":
    unittest.main()