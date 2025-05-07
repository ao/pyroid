#!/usr/bin/env python3
"""
Integration tests for Pyroid text functionality.

This script contains tests to verify that the text Rust extensions are working correctly.
"""

import unittest
import pyroid
import base64

class TestStringOperations(unittest.TestCase):
    """Test the string operations."""
    
    def test_reverse(self):
        """Test string reversal."""
        text = "Hello, world!"
        result = pyroid.text.reverse(text)
        self.assertEqual(result, "!dlrow ,olleH")
    
    def test_base64_encode_decode(self):
        """Test base64 encoding and decoding."""
        text = "Hello, world!"
        encoded = pyroid.text.base64_encode(text)
        # Verify against Python's built-in base64
        expected = base64.b64encode(text.encode()).decode()
        self.assertEqual(encoded, expected)
        
        # Test decoding
        decoded = pyroid.text.base64_decode(encoded)
        self.assertEqual(decoded, text)
    
    def test_split(self):
        """Test string splitting."""
        text = "apple,banana,cherry,date"
        result = pyroid.text.split(text, ",")
        self.assertEqual(result, ["apple", "banana", "cherry", "date"])
    
    def test_join(self):
        """Test string joining."""
        items = ["apple", "banana", "cherry", "date"]
        result = pyroid.text.join(items, ", ")
        self.assertEqual(result, "apple, banana, cherry, date")
    
    def test_replace(self):
        """Test string replacement."""
        text = "Hello, world!"
        result = pyroid.text.replace(text, "world", "universe")
        self.assertEqual(result, "Hello, universe!")
    
    def test_regex_replace(self):
        """Test regex replacement."""
        text = "Hello, world! Hello, universe!"
        result = pyroid.text.regex_replace(text, r"Hello", "Hi")
        self.assertEqual(result, "Hi, world! Hi, universe!")
    
    def test_to_uppercase(self):
        """Test converting to uppercase."""
        text = "Hello, world!"
        result = pyroid.text.to_uppercase(text)
        self.assertEqual(result, "HELLO, WORLD!")
    
    def test_to_lowercase(self):
        """Test converting to lowercase."""
        text = "Hello, WORLD!"
        result = pyroid.text.to_lowercase(text)
        self.assertEqual(result, "hello, world!")

class TestNLPOperations(unittest.TestCase):
    """Test the NLP operations."""
    
    def test_tokenize(self):
        """Test tokenization."""
        text = "Hello, world! This is a test."
        result = pyroid.text.tokenize(text)
        self.assertEqual(result, ["hello", "world", "this", "is", "a", "test"])
        
        # Test with options
        result = pyroid.text.tokenize(text, lowercase=False, remove_punct=False)
        self.assertEqual(result, ["Hello", "world", "This", "is", "a", "test"])
    
    def test_ngrams(self):
        """Test n-gram generation."""
        text = "This is a test"
        result = pyroid.text.ngrams(text, 2)
        self.assertEqual(result, [["this", "is"], ["is", "a"], ["a", "test"]])
        
        # Test with n=3
        result = pyroid.text.ngrams(text, 3)
        self.assertEqual(result, [["this", "is", "a"], ["is", "a", "test"]])

if __name__ == "__main__":
    unittest.main()