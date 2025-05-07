#!/usr/bin/env python3
"""
Test suite for Python implementations of Pyroid text module.
"""

import unittest
import sys
import os
import re
from typing import List, Dict, Any

# Add the parent directory to the path so we can import pyroid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Python implementations directly
import pyroid.text_impl

class TestTextImpl(unittest.TestCase):
    """Test the text_impl module."""

    def test_reverse(self):
        """Test the reverse function."""
        # Test with a simple string
        result = pyroid.text_impl.reverse("hello")
        self.assertEqual(result, "olleh")
        
        # Test with an empty string
        result = pyroid.text_impl.reverse("")
        self.assertEqual(result, "")
        
        # Test with a string containing spaces
        result = pyroid.text_impl.reverse("hello world")
        self.assertEqual(result, "dlrow olleh")

    def test_base64(self):
        """Test the base64_encode and base64_decode functions."""
        # Test encoding
        result = pyroid.text_impl.base64_encode("hello")
        self.assertEqual(result, "aGVsbG8=")
        
        # Test decoding
        result = pyroid.text_impl.base64_decode("aGVsbG8=")
        self.assertEqual(result, "hello")
        
        # Test round trip
        original = "Hello, world! 123"
        encoded = pyroid.text_impl.base64_encode(original)
        decoded = pyroid.text_impl.base64_decode(encoded)
        self.assertEqual(decoded, original)

    def test_split(self):
        """Test the split function."""
        # Test with a simple string
        result = pyroid.text_impl.split("a,b,c", ",")
        self.assertEqual(result, ["a", "b", "c"])
        
        # Test with an empty string
        result = pyroid.text_impl.split("", ",")
        self.assertEqual(result, [""])
        
        # Test with a string that doesn't contain the delimiter
        result = pyroid.text_impl.split("abc", ",")
        self.assertEqual(result, ["abc"])

    def test_join(self):
        """Test the join function."""
        # Test with a list of strings
        result = pyroid.text_impl.join(["a", "b", "c"], ",")
        self.assertEqual(result, "a,b,c")
        
        # Test with an empty list
        result = pyroid.text_impl.join([], ",")
        self.assertEqual(result, "")
        
        # Test with a list containing empty strings
        result = pyroid.text_impl.join(["", "", ""], ",")
        self.assertEqual(result, ",,")

    def test_replace(self):
        """Test the replace function."""
        # Test with a simple string
        result = pyroid.text_impl.replace("hello world", "world", "universe")
        self.assertEqual(result, "hello universe")
        
        # Test with a string that doesn't contain the substring
        result = pyroid.text_impl.replace("hello world", "universe", "galaxy")
        self.assertEqual(result, "hello world")
        
        # Test with an empty string
        result = pyroid.text_impl.replace("", "world", "universe")
        self.assertEqual(result, "")

    def test_regex_replace(self):
        """Test the regex_replace function."""
        # Test with a simple pattern
        result = pyroid.text_impl.regex_replace("hello 123 world", r"\d+", "456")
        self.assertEqual(result, "hello 456 world")
        
        # Test with a pattern that doesn't match
        result = pyroid.text_impl.regex_replace("hello world", r"\d+", "456")
        self.assertEqual(result, "hello world")
        
        # Test with a more complex pattern
        result = pyroid.text_impl.regex_replace("hello world", r"(hello) (world)", r"\2 \1")
        self.assertEqual(result, "world hello")

    def test_to_uppercase(self):
        """Test the to_uppercase function."""
        # Test with a simple string
        result = pyroid.text_impl.to_uppercase("hello")
        self.assertEqual(result, "HELLO")
        
        # Test with an empty string
        result = pyroid.text_impl.to_uppercase("")
        self.assertEqual(result, "")
        
        # Test with a mixed case string
        result = pyroid.text_impl.to_uppercase("Hello World")
        self.assertEqual(result, "HELLO WORLD")

    def test_to_lowercase(self):
        """Test the to_lowercase function."""
        # Test with a simple string
        result = pyroid.text_impl.to_lowercase("HELLO")
        self.assertEqual(result, "hello")
        
        # Test with an empty string
        result = pyroid.text_impl.to_lowercase("")
        self.assertEqual(result, "")
        
        # Test with a mixed case string
        result = pyroid.text_impl.to_lowercase("Hello World")
        self.assertEqual(result, "hello world")

    def test_tokenize(self):
        """Test the tokenize function."""
        # Test with a simple string
        result = pyroid.text_impl.tokenize("hello world")
        self.assertEqual(result, ["hello", "world"])
        
        # Test with lowercase=False
        result = pyroid.text_impl.tokenize("Hello World", lowercase=False)
        self.assertEqual(result, ["Hello", "World"])
        
        # Test with remove_punct=False
        result = pyroid.text_impl.tokenize("hello, world!", remove_punct=False)
        self.assertEqual(result, ["hello,", "world!"])
        
        # Test with an empty string
        result = pyroid.text_impl.tokenize("")
        self.assertEqual(result, [])

    def test_ngrams(self):
        """Test the ngrams function."""
        # Test with a string
        result = pyroid.text_impl.ngrams("hello world", 2)
        self.assertEqual(result, [["hello", "world"]])
        
        # Test with a list of tokens
        result = pyroid.text_impl.ngrams(["a", "b", "c", "d"], 2)
        self.assertEqual(result, [["a", "b"], ["b", "c"], ["c", "d"]])
        
        # Test with n > len(tokens)
        result = pyroid.text_impl.ngrams(["a", "b"], 3)
        self.assertEqual(result, [])
        
        # Test with an empty list
        result = pyroid.text_impl.ngrams([], 2)
        self.assertEqual(result, [])

if __name__ == "__main__":
    unittest.main()