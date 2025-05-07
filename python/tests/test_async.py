#!/usr/bin/env python3
"""
Integration tests for Pyroid async functionality.

This script contains tests to verify that the async Rust extensions are working correctly.
"""

import unittest
import os
import tempfile
import asyncio
import pyroid

class TestAsyncClient(unittest.TestCase):
    """Test the AsyncClient class."""
    
    def test_client_creation(self):
        """Test creating an AsyncClient object."""
        client = pyroid.AsyncClient()
        self.assertIsNotNone(client)
    
    def test_client_with_options(self):
        """Test creating an AsyncClient with options."""
        client = pyroid.AsyncClient(timeout=30, concurrency=5, adaptive_concurrency=False)
        self.assertIsNotNone(client)
    
    @unittest.skip("Network tests require an internet connection")
    def test_fetch(self):
        """Test fetching a URL."""
        async def test_fetch_coroutine():
            client = pyroid.AsyncClient()
            response = await client.fetch("https://httpbin.org/get")
            return response
        
        # Run the coroutine
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(test_fetch_coroutine())
        
        # Verify the response
        self.assertIsNotNone(response)
        self.assertIn("url", response)
    
    @unittest.skip("Network tests require an internet connection")
    def test_fetch_many(self):
        """Test fetching multiple URLs."""
        async def test_fetch_many_coroutine():
            client = pyroid.AsyncClient()
            urls = [
                "https://httpbin.org/get?id=1",
                "https://httpbin.org/get?id=2",
                "https://httpbin.org/get?id=3"
            ]
            responses = await client.fetch_many(urls)
            return responses
        
        # Run the coroutine
        loop = asyncio.get_event_loop()
        responses = loop.run_until_complete(test_fetch_many_coroutine())
        
        # Verify the responses
        self.assertEqual(len(responses), 3)
        for response in responses:
            self.assertIsNotNone(response)
            self.assertIn("url", response)
    
    @unittest.skip("Network tests require an internet connection")
    def test_download_file(self):
        """Test downloading a file."""
        async def test_download_file_coroutine():
            client = pyroid.AsyncClient()
            temp_dir = tempfile.mkdtemp()
            temp_file = os.path.join(temp_dir, "test_download.txt")
            
            # Download a small text file
            await client.download_file("https://httpbin.org/robots.txt", temp_file)
            
            # Check that the file exists
            exists = os.path.exists(temp_file)
            
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
                
            return exists
        
        # Run the coroutine
        loop = asyncio.get_event_loop()
        exists = loop.run_until_complete(test_download_file_coroutine())
        
        # Verify the file was downloaded
        self.assertTrue(exists)
    
    @unittest.skip("Network tests require an internet connection")
    def test_connection_pool_stats(self):
        """Test getting connection pool stats."""
        async def test_stats_coroutine():
            client = pyroid.AsyncClient()
            
            # Make some requests to populate the stats
            await client.fetch("https://httpbin.org/get")
            await client.fetch("https://httpbin.org/get")
            
            # Get the stats
            stats = client.connection_pool_stats()
            return stats
        
        # Run the coroutine
        loop = asyncio.get_event_loop()
        stats = loop.run_until_complete(test_stats_coroutine())
        
        # Verify the stats
        self.assertIsNotNone(stats)
        self.assertIn("optimal_concurrency", stats)
        self.assertIn("response_times_count", stats)

class TestAsyncFileReader(unittest.TestCase):
    """Test the AsyncFileReader class."""
    
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
    
    def test_reader_creation(self):
        """Test creating an AsyncFileReader object."""
        reader = pyroid.AsyncFileReader(self.test_file)
        self.assertIsNotNone(reader)
    
    def test_read_all(self):
        """Test reading an entire file."""
        async def test_read_all_coroutine():
            reader = pyroid.AsyncFileReader(self.test_file)
            content = await reader.read_all()
            return content
        
        # Run the coroutine
        loop = asyncio.get_event_loop()
        content = loop.run_until_complete(test_read_all_coroutine())
        
        # Verify the content
        self.assertEqual(content, self.test_content)
    
    def test_read_lines(self):
        """Test reading a file line by line."""
        async def test_read_lines_coroutine():
            reader = pyroid.AsyncFileReader(self.test_file)
            lines = await reader.read_lines()
            return lines
        
        # Run the coroutine
        loop = asyncio.get_event_loop()
        lines = loop.run_until_complete(test_read_lines_coroutine())
        
        # Verify the lines
        expected_lines = self.test_content.splitlines()
        self.assertEqual(len(lines), len(expected_lines))
        for i, line in enumerate(lines):
            self.assertEqual(line, expected_lines[i])

class TestAsyncOperations(unittest.TestCase):
    """Test the async operations from the async_ops module."""
    
    def test_sleep(self):
        """Test async sleep."""
        async def test_sleep_coroutine():
            start_time = asyncio.get_event_loop().time()
            await pyroid.async_ops.sleep(0.1)
            end_time = asyncio.get_event_loop().time()
            return end_time - start_time
        
        # Run the coroutine
        loop = asyncio.get_event_loop()
        duration = loop.run_until_complete(test_sleep_coroutine())
        
        # Verify that at least 0.1 seconds passed
        self.assertGreaterEqual(duration, 0.1)
    
    def test_read_file_async(self):
        """Test async file reading."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Hello, world!")
            temp_file = f.name
        
        async def test_read_file_coroutine():
            content = await pyroid.async_ops.read_file_async(temp_file)
            return content
        
        try:
            # Run the coroutine
            loop = asyncio.get_event_loop()
            content = loop.run_until_complete(test_read_file_coroutine())
            
            # Verify the content
            self.assertEqual(content, "Hello, world!")
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_write_file_async(self):
        """Test async file writing."""
        # Create a temporary file path
        temp_file = tempfile.mktemp()
        
        async def test_write_file_coroutine():
            result = await pyroid.async_ops.write_file_async(temp_file, "Hello, world!")
            return result
        
        try:
            # Run the coroutine
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(test_write_file_coroutine())
            
            # Verify the result
            self.assertTrue(result)
            
            # Verify the content was written correctly
            with open(temp_file, "r") as f:
                content = f.read()
            self.assertEqual(content, "Hello, world!")
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    unittest.main()