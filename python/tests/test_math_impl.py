#!/usr/bin/env python3
"""
Test suite for Python implementations of Pyroid math module.
"""

import unittest
import sys
import os
import random
import math
from typing import List, Dict, Any

# Add the parent directory to the path so we can import pyroid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Python implementations directly
import pyroid.math_impl

class TestMathImpl(unittest.TestCase):
    """Test the math_impl module."""

    def test_vector(self):
        """Test the Vector class."""
        # Create a Vector
        v1 = pyroid.math_impl.Vector([1, 2, 3])
        
        # Test __getitem__
        self.assertEqual(v1[0], 1)
        self.assertEqual(v1[1], 2)
        self.assertEqual(v1[2], 3)
        
        # Test __len__
        self.assertEqual(len(v1), 3)
        
        # Test __add__
        v2 = pyroid.math_impl.Vector([4, 5, 6])
        v3 = v1 + v2
        self.assertEqual(v3.values, [5, 7, 9])
        
        # Test __sub__
        v4 = v2 - v1
        self.assertEqual(v4.values, [3, 3, 3])
        
        # Test __mul__
        v5 = v1 * 2
        self.assertEqual(v5.values, [2, 4, 6])
        
        # Test __truediv__
        v6 = v2 / 2
        self.assertEqual(v6.values, [2, 2.5, 3])
        
        # Test dot
        dot_product = v1.dot(v2)
        self.assertEqual(dot_product, 1*4 + 2*5 + 3*6)
        
        # Test norm
        norm = v1.norm()
        self.assertAlmostEqual(norm, (1*1 + 2*2 + 3*3)**0.5)

    def test_matrix(self):
        """Test the Matrix class."""
        # Create a Matrix
        m1 = pyroid.math_impl.Matrix([[1, 2], [3, 4]])
        
        # Test __getitem__
        self.assertEqual(m1[0], [1, 2])
        self.assertEqual(m1[1], [3, 4])
        
        # Test __len__
        self.assertEqual(len(m1), 2)
        
        # Test __add__
        m2 = pyroid.math_impl.Matrix([[5, 6], [7, 8]])
        m3 = m1 + m2
        self.assertEqual(m3.values, [[6, 8], [10, 12]])
        
        # Test __sub__
        m4 = m2 - m1
        self.assertEqual(m4.values, [[4, 4], [4, 4]])
        
        # Test __mul__ with scalar
        m5 = m1 * 2
        self.assertEqual(m5.values, [[2, 4], [6, 8]])
        
        # Test __mul__ with matrix
        m6 = m1 * m2
        self.assertEqual(m6.values, [[19, 22], [43, 50]])
        
        # Test transpose
        m7 = m1.transpose()
        self.assertEqual(m7.values, [[1, 3], [2, 4]])

    def test_sum(self):
        """Test the sum function."""
        # Test with integers
        result = pyroid.math_impl.sum([1, 2, 3, 4, 5])
        self.assertEqual(result, 15)
        
        # Test with floats
        result = pyroid.math_impl.sum([1.1, 2.2, 3.3, 4.4, 5.5])
        self.assertAlmostEqual(result, 16.5)
        
        # Test with empty list
        result = pyroid.math_impl.sum([])
        self.assertEqual(result, 0)

    def test_mean(self):
        """Test the mean function."""
        # Test with integers
        result = pyroid.math_impl.mean([1, 2, 3, 4, 5])
        self.assertEqual(result, 3)
        
        # Test with floats
        result = pyroid.math_impl.mean([1.1, 2.2, 3.3, 4.4, 5.5])
        self.assertAlmostEqual(result, 3.3)
        
        # Test with empty list
        result = pyroid.math_impl.mean([])
        self.assertEqual(result, 0)

    def test_std(self):
        """Test the std function."""
        # Test with integers
        result = pyroid.math_impl.std([1, 2, 3, 4, 5])
        self.assertAlmostEqual(result, 1.5811, places=4)
        
        # Test with single value
        result = pyroid.math_impl.std([5])
        self.assertEqual(result, 0)
        
        # Test with empty list
        result = pyroid.math_impl.std([])
        self.assertEqual(result, 0)

    def test_variance(self):
        """Test the variance function."""
        # Test with integers
        result = pyroid.math_impl.variance([1, 2, 3, 4, 5])
        self.assertAlmostEqual(result, 2.5, places=4)
        
        # Test with single value
        result = pyroid.math_impl.variance([5])
        self.assertEqual(result, 0)
        
        # Test with empty list
        result = pyroid.math_impl.variance([])
        self.assertEqual(result, 0)

    def test_correlation(self):
        """Test the correlation function."""
        # Test with correlated data
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]  # Perfect positive correlation
        result = pyroid.math_impl.correlation(x, y)
        self.assertAlmostEqual(result, 1.0, places=4)
        
        # Test with anti-correlated data
        y = [10, 8, 6, 4, 2]  # Perfect negative correlation
        result = pyroid.math_impl.correlation(x, y)
        self.assertAlmostEqual(result, -1.0, places=4)
        
        # Test with uncorrelated data
        y = [5, 5, 5, 5, 5]  # No correlation
        result = pyroid.math_impl.correlation(x, y)
        self.assertEqual(result, 0)
        
        # Test with empty lists
        result = pyroid.math_impl.correlation([], [])
        self.assertEqual(result, 0)

if __name__ == "__main__":
    unittest.main()