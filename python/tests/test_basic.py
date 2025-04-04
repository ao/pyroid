#!/usr/bin/env python3
"""
Basic tests for Pyroid.

This script contains basic tests to verify that Pyroid is working correctly.
"""

import unittest
import pyroid

class TestStringOps(unittest.TestCase):
    def test_parallel_regex_replace(self):
        text = "Hello world! Hello universe!"
        result = pyroid.parallel_regex_replace(text, r"Hello", "Hi")
        self.assertEqual(result, "Hi world! Hi universe!")
    
    def test_parallel_text_cleanup(self):
        texts = ["  Hello, World! ", "123 Testing 456", "Special @#$% Characters"]
        results = pyroid.parallel_text_cleanup(texts)
        self.assertEqual(results[0], "hello world")
        self.assertEqual(results[1], "123 testing 456")
        self.assertEqual(results[2], "special  characters")
    
    def test_parallel_base64(self):
        text = "Hello, world!"
        encoded = pyroid.parallel_base64_encode(text)
        decoded = pyroid.parallel_base64_decode(encoded)
        self.assertEqual(decoded, text)

class TestMathOps(unittest.TestCase):
    def test_parallel_sum(self):
        numbers = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = pyroid.parallel_sum(numbers)
        self.assertEqual(result, 15.0)
    
    def test_parallel_product(self):
        numbers = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = pyroid.parallel_product(numbers)
        self.assertEqual(result, 120.0)
    
    def test_parallel_mean(self):
        numbers = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = pyroid.parallel_mean(numbers)
        self.assertEqual(result, 3.0)
    
    def test_parallel_std(self):
        numbers = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = pyroid.parallel_std(numbers, 1)
        self.assertAlmostEqual(result, 1.5811388300841898, places=10)
    
    def test_parallel_apply(self):
        numbers = [1.0, 4.0, 9.0, 16.0]
        result = pyroid.parallel_apply(numbers, "sqrt")
        self.assertEqual(result, [1.0, 2.0, 3.0, 4.0])
    
    def test_matrix_multiply(self):
        a = [[1, 2], [3, 4]]
        b = [[5, 6], [7, 8]]
        result = pyroid.matrix_multiply(a, b)
        self.assertEqual(result, [[19, 22], [43, 50]])

class TestDataOps(unittest.TestCase):
    def test_parallel_filter(self):
        items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = pyroid.parallel_filter(items, lambda x: x % 2 == 0)
        self.assertEqual(result, [2, 4, 6, 8, 10])
    
    def test_parallel_map(self):
        items = [1, 2, 3, 4, 5]
        result = pyroid.parallel_map(items, lambda x: x * x)
        self.assertEqual(result, [1, 4, 9, 16, 25])
    
    def test_parallel_reduce(self):
        items = [1, 2, 3, 4, 5]
        result = pyroid.parallel_reduce(items, lambda x, y: x + y)
        self.assertEqual(result, 15)
    
    def test_parallel_sort(self):
        items = [5, 2, 8, 1, 9, 3]
        result = pyroid.parallel_sort(items)
        self.assertEqual(result, [1, 2, 3, 5, 8, 9])
        
        # Test with reverse
        result = pyroid.parallel_sort(items, reverse=True)
        self.assertEqual(result, [9, 8, 5, 3, 2, 1])
        
        # Test with key function
        items = [(1, 5), (2, 2), (3, 8), (4, 1), (5, 9), (6, 3)]
        result = pyroid.parallel_sort(items, key=lambda x: x[1])
        self.assertEqual(result, [(4, 1), (2, 2), (6, 3), (1, 5), (3, 8), (5, 9)])

class TestAsyncOps(unittest.TestCase):
    def test_async_client_creation(self):
        client = pyroid.AsyncClient()
        self.assertIsNotNone(client)
    
    def test_async_channel_creation(self):
        channel = pyroid.AsyncChannel(10)
        self.assertIsNotNone(channel)
    
    def test_async_file_reader_creation(self):
        reader = pyroid.AsyncFileReader("test_file.txt")
        self.assertIsNotNone(reader)

if __name__ == "__main__":
    unittest.main()