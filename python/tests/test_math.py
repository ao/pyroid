#!/usr/bin/env python3
"""
Integration tests for Pyroid math functionality.

This script contains tests to verify that the math Rust extensions are working correctly.
"""

import unittest
import pyroid
import math

class TestVector(unittest.TestCase):
    """Test the Vector class and related functionality."""
    
    def test_vector_creation(self):
        """Test creating a Vector object."""
        v = pyroid.math.Vector([1, 2, 3])
        # This test just verifies that Vector can be created
        self.assertIsNotNone(v)
    
    def test_vector_addition(self):
        """Test vector addition."""
        v1 = pyroid.math.Vector([1, 2, 3])
        v2 = pyroid.math.Vector([4, 5, 6])
        v3 = v1 + v2
        # Convert to list for comparison (implementation-dependent)
        # This is just a placeholder - adjust based on actual API
        result = [v3[i] for i in range(len(v3))]
        self.assertEqual(result, [5, 7, 9])
    
    def test_vector_dot_product(self):
        """Test vector dot product."""
        v1 = pyroid.math.Vector([1, 2, 3])
        v2 = pyroid.math.Vector([4, 5, 6])
        result = v1.dot(v2)
        self.assertEqual(result, 32)  # 1*4 + 2*5 + 3*6 = 32

class TestMatrix(unittest.TestCase):
    """Test the Matrix class and related functionality."""
    
    def test_matrix_creation(self):
        """Test creating a Matrix object."""
        m = pyroid.math.Matrix([[1, 2], [3, 4]])
        # This test just verifies that Matrix can be created
        self.assertIsNotNone(m)
    
    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        m1 = pyroid.math.Matrix([[1, 2], [3, 4]])
        m2 = pyroid.math.Matrix([[5, 6], [7, 8]])
        m3 = m1 * m2
        # Convert to nested lists for comparison (implementation-dependent)
        # This is just a placeholder - adjust based on actual API
        result = [[m3[i][j] for j in range(2)] for i in range(2)]
        self.assertEqual(result, [[19, 22], [43, 50]])

class TestStatistics(unittest.TestCase):
    """Test the statistical functions."""
    
    def test_mean(self):
        """Test mean calculation."""
        numbers = [1, 2, 3, 4, 5]
        result = pyroid.math.mean(numbers)
        self.assertEqual(result, 3.0)
    
    def test_median(self):
        """Test median calculation."""
        numbers = [1, 2, 3, 4, 5]
        result = pyroid.math.median(numbers)
        self.assertEqual(result, 3.0)
        
        # Test even number of elements
        numbers = [1, 2, 3, 4]
        result = pyroid.math.median(numbers)
        self.assertEqual(result, 2.5)
    
    def test_std(self):
        """Test standard deviation calculation."""
        numbers = [1, 2, 3, 4, 5]
        result = pyroid.math.std(numbers)
        self.assertAlmostEqual(result, math.sqrt(2), places=10)
    
    def test_variance(self):
        """Test variance calculation."""
        numbers = [1, 2, 3, 4, 5]
        result = pyroid.math.variance(numbers)
        self.assertEqual(result, 2.0)
    
    def test_correlation(self):
        """Test correlation calculation."""
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        result = pyroid.math.correlation(x, y)
        self.assertEqual(result, -1.0)
    
    def test_describe(self):
        """Test descriptive statistics."""
        numbers = [1, 2, 3, 4, 5]
        result = pyroid.math.describe(numbers)
        self.assertEqual(result["mean"], 3.0)
        self.assertEqual(result["median"], 3.0)
        self.assertEqual(result["count"], 5)

class TestStatsNamespace(unittest.TestCase):
    """Test the stats namespace."""
    
    def test_stats_mean(self):
        """Test stats.mean function."""
        numbers = [1, 2, 3, 4, 5]
        result = pyroid.math.stats.mean(numbers)
        self.assertEqual(result, 3.0)
    
    def test_stats_calc_std(self):
        """Test stats.calc_std function."""
        numbers = [1, 2, 3, 4, 5]
        result = pyroid.math.stats.calc_std(numbers)
        self.assertAlmostEqual(result, math.sqrt(2), places=10)

if __name__ == "__main__":
    unittest.main()