#!/usr/bin/env python3
"""
Test suite for Python implementations of Pyroid data module.
"""

import unittest
import sys
import os
import random
from typing import List, Dict, Any

# Add the parent directory to the path so we can import pyroid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Python implementations directly
import pyroid.data_impl

class TestDataImpl(unittest.TestCase):
    """Test the data_impl module."""

    def test_dataframe(self):
        """Test the DataFrame class."""
        # Create a DataFrame
        df = pyroid.data_impl.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6]
        })
        
        # Test __getitem__
        self.assertEqual(df["a"], [1, 2, 3])
        self.assertEqual(df["b"], [4, 5, 6])
        
        # Test __setitem__
        df["c"] = [7, 8, 9]
        self.assertEqual(df["c"], [7, 8, 9])
        
        # Test __len__
        self.assertEqual(len(df), 3)
        
        # Test columns
        self.assertEqual(set(df.columns()), {"a", "b", "c"})
        
        # Test to_dict
        self.assertEqual(df.to_dict(), {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9]
        })
        
        # Test validation
        with self.assertRaises(ValueError):
            pyroid.data_impl.DataFrame({
                "a": [1, 2, 3],
                "b": [4, 5]  # Different length
            })

    def test_filter(self):
        """Test the filter function."""
        # Test with integers
        result = pyroid.data_impl.filter([1, 2, 3, 4, 5], lambda x: x % 2 == 0)
        self.assertEqual(result, [2, 4])
        
        # Test with strings
        result = pyroid.data_impl.filter(["apple", "banana", "cherry"], lambda x: "a" in x)
        self.assertEqual(result, ["apple", "banana"])
        
        # Test with empty list
        result = pyroid.data_impl.filter([], lambda x: x > 0)
        self.assertEqual(result, [])

    def test_map(self):
        """Test the map function."""
        # Test with integers
        result = pyroid.data_impl.map([1, 2, 3, 4, 5], lambda x: x * 2)
        self.assertEqual(result, [2, 4, 6, 8, 10])
        
        # Test with strings
        result = pyroid.data_impl.map(["a", "b", "c"], lambda x: x.upper())
        self.assertEqual(result, ["A", "B", "C"])
        
        # Test with empty list
        result = pyroid.data_impl.map([], lambda x: x * 2)
        self.assertEqual(result, [])

    def test_reduce(self):
        """Test the reduce function."""
        # Test with integers
        result = pyroid.data_impl.reduce([1, 2, 3, 4, 5], lambda x, y: x + y)
        self.assertEqual(result, 15)
        
        # Test with initial value
        result = pyroid.data_impl.reduce([1, 2, 3, 4, 5], lambda x, y: x + y, 10)
        self.assertEqual(result, 25)
        
        # Test with single value
        result = pyroid.data_impl.reduce([5], lambda x, y: x + y)
        self.assertEqual(result, 5)
        
        # Test with empty list and initial value
        result = pyroid.data_impl.reduce([], lambda x, y: x + y, 0)
        self.assertEqual(result, 0)
        
        # Test with empty list and no initial value
        with self.assertRaises(ValueError):
            pyroid.data_impl.reduce([], lambda x, y: x + y)

    def test_sort(self):
        """Test the sort function."""
        # Test with integers
        result = pyroid.data_impl.sort([5, 3, 1, 4, 2])
        self.assertEqual(result, [1, 2, 3, 4, 5])
        
        # Test with reverse
        result = pyroid.data_impl.sort([5, 3, 1, 4, 2], reverse=True)
        self.assertEqual(result, [5, 4, 3, 2, 1])
        
        # Test with key function
        result = pyroid.data_impl.sort(["apple", "banana", "cherry"], key=len)
        self.assertEqual(result, ["apple", "cherry", "banana"])
        
        # Test with empty list
        result = pyroid.data_impl.sort([])
        self.assertEqual(result, [])

    def test_apply(self):
        """Test the apply function."""
        # Create a DataFrame
        df = pyroid.data_impl.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6]
        })
        
        # Apply a function
        result = pyroid.data_impl.apply(df, lambda x: [i * 2 for i in x])
        
        # Check the result
        self.assertEqual(result.to_dict(), {
            "a": [2, 4, 6],
            "b": [8, 10, 12]
        })

    def test_groupby_aggregate(self):
        """Test the groupby_aggregate function."""
        # Create a DataFrame
        df = pyroid.data_impl.DataFrame({
            "category": ["A", "B", "A", "B", "C"],
            "value": [1, 2, 3, 4, 5]
        })
        
        # Group by category and calculate mean
        result = pyroid.data_impl.groupby_aggregate(df, "category", {"value": "mean"})
        
        # Check the result
        self.assertEqual(set(result["category"]), {"A", "B", "C"})
        
        # Create a dictionary for easier checking
        result_dict = {}
        for i, cat in enumerate(result["category"]):
            result_dict[cat] = result["value_mean"][i]
        
        # Check the means
        self.assertEqual(result_dict["A"], 2)  # (1 + 3) / 2
        self.assertEqual(result_dict["B"], 3)  # (2 + 4) / 2
        self.assertEqual(result_dict["C"], 5)  # 5 / 1

if __name__ == "__main__":
    unittest.main()