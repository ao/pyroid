#!/usr/bin/env python3
"""
Integration tests for Pyroid data functionality.

This script contains tests to verify that the data Rust extensions are working correctly.
"""

import unittest
import pyroid

class TestCollections(unittest.TestCase):
    """Test the collection operations."""
    
    def test_filter(self):
        """Test filter operation."""
        items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = pyroid.data.filter(items, lambda x: x % 2 == 0)
        self.assertEqual(result, [2, 4, 6, 8, 10])
    
    def test_map(self):
        """Test map operation."""
        items = [1, 2, 3, 4, 5]
        result = pyroid.data.map(items, lambda x: x * 2)
        self.assertEqual(result, [2, 4, 6, 8, 10])
    
    def test_reduce(self):
        """Test reduce operation."""
        items = [1, 2, 3, 4, 5]
        result = pyroid.data.reduce(items, lambda x, y: x + y)
        self.assertEqual(result, 15)
        
        # Test with initial value
        result = pyroid.data.reduce(items, lambda x, y: x + y, 10)
        self.assertEqual(result, 25)
    
    def test_sort(self):
        """Test sort operation."""
        items = [5, 3, 8, 1, 2]
        result = pyroid.data.sort(items)
        self.assertEqual(result, [1, 2, 3, 5, 8])
        
        # Test with reverse
        result = pyroid.data.sort(items, reverse=True)
        self.assertEqual(result, [8, 5, 3, 2, 1])
        
        # Test with key function
        items = [(1, 5), (2, 3), (3, 8), (4, 1), (5, 2)]
        result = pyroid.data.sort(items, key=lambda x: x[1])
        self.assertEqual(result, [(4, 1), (5, 2), (2, 3), (1, 5), (3, 8)])

class TestDataFrame(unittest.TestCase):
    """Test the DataFrame class and related functionality."""
    
    def test_dataframe_creation(self):
        """Test creating a DataFrame object."""
        df = pyroid.data.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35]
        })
        # This test just verifies that DataFrame can be created
        self.assertIsNotNone(df)
    
    def test_apply(self):
        """Test apply operation on DataFrame."""
        df = pyroid.data.DataFrame({
            "id": [1, 2, 3],
            "value": [10, 20, 30]
        })
        result = pyroid.data.apply(df, lambda x: x * 2)
        # Check that the result is a DataFrame
        self.assertIsInstance(result, pyroid.data.DataFrame)
        
        # Implementation-dependent - adjust based on actual API
        # This is just a placeholder
        self.assertEqual(result["value"][0], 20)
        self.assertEqual(result["value"][1], 40)
        self.assertEqual(result["value"][2], 60)
    
    def test_groupby_aggregate(self):
        """Test groupby_aggregate operation on DataFrame."""
        df = pyroid.data.DataFrame({
            "category": ["A", "B", "A", "B", "A"],
            "value": [10, 20, 15, 25, 30]
        })
        result = pyroid.data.groupby_aggregate(df, "category", {"value": "sum"})
        # Check that the result is a DataFrame
        self.assertIsInstance(result, pyroid.data.DataFrame)
        
        # Implementation-dependent - adjust based on actual API
        # This is just a placeholder
        categories = set(result["category"])
        self.assertEqual(categories, {"A", "B"})
        
        # Find the row for category A and check its sum
        for i in range(len(result["category"])):
            if result["category"][i] == "A":
                self.assertEqual(result["value_sum"][i], 55)  # 10 + 15 + 30
            elif result["category"][i] == "B":
                self.assertEqual(result["value_sum"][i], 45)  # 20 + 25

if __name__ == "__main__":
    unittest.main()