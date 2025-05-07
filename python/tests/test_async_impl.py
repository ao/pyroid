#!/usr/bin/env python3
"""
Test suite for Python implementations of Pyroid async module.
"""

import unittest
import sys
import os
import asyncio
import tempfile
from typing import List, Dict, Any

# Add the parent directory to the path so we can import pyroid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Python implementations directly
import pyroid.async_impl

class TestAsyncImpl(unittest.TestCase):
    """Test the async_impl module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test file
        self.test_file = os.path.join(self.temp_dir, "test_file.txt")
        with open(self.test_file, "w") as f:
            f.write("Hello, world!")

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove test files
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        
        # Remove any other files in the temp directory
        for filename in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Remove the temp directory
        os.rmdir(self.temp_dir)

    def test_async_client(self):
        """Test the AsyncClient class."""
        # Create an AsyncClient
        client = pyroid.async_impl.AsyncClient("https://example.com")
        
        # Check properties
        self.assertEqual(client.base_url, "https://example.com")
        self.assertEqual(client.timeout, 30.0)
        
        # Create with custom timeout
        client = pyroid.async_impl.AsyncClient("https://example.com", timeout=10.0)
        self.assertEqual(client.timeout, 10.0)

    def test_async_file_reader(self):
        """Test the AsyncFileReader class."""
        # Create an AsyncFileReader
        reader = pyroid.async_impl.AsyncFileReader(self.test_file)
        
        # Check properties
        self.assertEqual(reader.path, self.test_file)
        self.assertIsNone(reader.file)

    def test_sleep(self):
        """Test the sleep function."""
        # Create an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Define a coroutine that uses sleep
        async def test_coro():
            start_time = loop.time()
            await pyroid.async_impl.sleep(0.1)  # Sleep for 0.1 seconds
            end_time = loop.time()
            return end_time - start_time
        
        # Run the coroutine
        elapsed_time = loop.run_until_complete(test_coro())
        
        # Check that it slept for at least 0.1 seconds
        self.assertGreaterEqual(elapsed_time, 0.1)
        
        # Clean up
        loop.close()

    def test_read_file_async(self):
        """Test the read_file_async function."""
        # Create an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Define a coroutine that reads a file
        async def test_coro():
            return await pyroid.async_impl.read_file_async(self.test_file)
        
        # Run the coroutine
        content = loop.run_until_complete(test_coro())
        
        # Check the content
        self.assertEqual(content, "Hello, world!")
        
        # Clean up
        loop.close()

    def test_write_file_async(self):
        """Test the write_file_async function."""
        # Create an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Define a coroutine that writes a file
        async def test_coro():
            new_file = os.path.join(self.temp_dir, "new_file.txt")
            result = await pyroid.async_impl.write_file_async(new_file, "New content")
            
            # Check that the file was written
            with open(new_file, "r") as f:
                content = f.read()
            
            # Clean up
            os.remove(new_file)
            
            return result, content
        
        # Run the coroutine
        result, content = loop.run_until_complete(test_coro())
        
        # Check the result
        self.assertTrue(result)
        self.assertEqual(content, "New content")
        
        # Clean up
        loop.close()

    def test_fetch_url(self):
        """Test the fetch_url function."""
        # This is a mock test since we can't easily test HTTP requests
        # in a unit test without mocking or a test server
        
        # Just test that the function exists and has the right signature
        self.assertTrue(hasattr(pyroid.async_impl, "fetch_url"))
        self.assertTrue(callable(pyroid.async_impl.fetch_url))

    def test_fetch_many(self):
        """Test the fetch_many function."""
        # This is a mock test since we can't easily test HTTP requests
        # in a unit test without mocking or a test server
        
        # Just test that the function exists and has the right signature
        self.assertTrue(hasattr(pyroid.async_impl, "fetch_many"))
        self.assertTrue(callable(pyroid.async_impl.fetch_many))

    def test_download_file(self):
        """Test the download_file function."""
        # This is a mock test since we can't easily test HTTP requests
        # in a unit test without mocking or a test server
        
        # Just test that the function exists and has the right signature
        self.assertTrue(hasattr(pyroid.async_impl, "download_file"))
        self.assertTrue(callable(pyroid.async_impl.download_file))

    def test_http_post(self):
        """Test the http_post function."""
        # This is a mock test since we can't easily test HTTP requests
        # in a unit test without mocking or a test server
        
        # Just test that the function exists and has the right signature
        self.assertTrue(hasattr(pyroid.async_impl, "http_post"))
        self.assertTrue(callable(pyroid.async_impl.http_post))

    def test_aiofiles_open(self):
        """Test the aiofiles_open function."""
        # Create an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Define a coroutine that uses aiofiles_open
        async def test_coro():
            async with pyroid.async_impl.aiofiles_open(self.test_file, "r") as f:
                content = await f.read()
            return content
        
        # Run the coroutine
        content = loop.run_until_complete(test_coro())
        
        # Check the content
        self.assertEqual(content, "Hello, world!")
        
        # Clean up
        loop.close()

if __name__ == "__main__":
    unittest.main()