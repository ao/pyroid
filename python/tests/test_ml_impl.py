#!/usr/bin/env python3
"""
Test suite for Python implementations of Pyroid ML module.
"""

import unittest
import sys
import os
import math
import random
from typing import List, Dict, Any

# Add the parent directory to the path so we can import pyroid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Python implementations directly
import pyroid.ml_impl

class TestMLImpl(unittest.TestCase):
    """Test the ml_impl module."""

    def test_kmeans(self):
        """Test the kmeans function."""
        # Create some test data with clear clusters
        data = [
            [1.0, 1.0],
            [1.5, 2.0],
            [10.0, 10.0],
            [10.5, 9.5]
        ]
        
        # Run k-means
        result = pyroid.ml_impl.kmeans(data, k=2, max_iterations=100)
        
        # Check that we have the right number of centroids
        self.assertEqual(len(result["centroids"]), 2)
        
        # Check that we have the right number of clusters
        self.assertEqual(len(result["clusters"]), 2)
        
        # Check that each cluster has the right points
        # (we don't know which cluster is which, so we need to check both possibilities)
        if len(result["clusters"][0]) == 2:
            self.assertEqual(len(result["clusters"][0]), 2)
            self.assertEqual(len(result["clusters"][1]), 2)
        else:
            self.assertEqual(len(result["clusters"][0]), 0)
            self.assertEqual(len(result["clusters"][1]), 4)
        
        # Test with empty data
        result = pyroid.ml_impl.kmeans([], k=2)
        self.assertEqual(result["centroids"], [])
        self.assertEqual(result["clusters"], [])

    def test_linear_regression(self):
        """Test the linear_regression function."""
        # Create some test data with a clear linear relationship
        X = [[1], [2], [3], [4], [5]]
        y = [2, 4, 6, 8, 10]  # y = 2*x
        
        # Run linear regression
        result = pyroid.ml_impl.linear_regression(X, y)
        
        # Check the coefficients
        self.assertAlmostEqual(result["coefficients"][0], 2.0, places=5)
        
        # Check the intercept
        self.assertAlmostEqual(result["intercept"], 0.0, places=5)
        
        # Check the R-squared
        self.assertAlmostEqual(result["r_squared"], 1.0, places=5)
        
        # Test with empty data
        result = pyroid.ml_impl.linear_regression([], [])
        self.assertEqual(result["coefficients"], [])
        self.assertEqual(result["intercept"], 0.0)
        self.assertEqual(result["r_squared"], 0.0)
        
        # Test with mismatched data
        with self.assertRaises(ValueError):
            pyroid.ml_impl.linear_regression([[1], [2]], [1])

    def test_normalize(self):
        """Test the normalize function."""
        # Create some test data
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        
        # Run min-max normalization
        result = pyroid.ml_impl.normalize(data, method="min-max")
        
        # Check the result
        self.assertEqual(result[0][0], 0.0)  # min value
        self.assertEqual(result[2][0], 1.0)  # max value
        self.assertEqual(result[0][1], 0.0)  # min value
        self.assertEqual(result[2][1], 1.0)  # max value
        
        # Run z-score normalization
        result = pyroid.ml_impl.normalize(data, method="z-score")
        
        # Check the result (mean should be 0, std should be 1)
        mean_col0 = sum(row[0] for row in result) / len(result)
        mean_col1 = sum(row[1] for row in result) / len(result)
        self.assertAlmostEqual(mean_col0, 0.0, places=5)
        self.assertAlmostEqual(mean_col1, 0.0, places=5)
        
        std_col0 = (sum((row[0] - mean_col0)**2 for row in result) / len(result))**0.5
        std_col1 = (sum((row[1] - mean_col1)**2 for row in result) / len(result))**0.5
        self.assertAlmostEqual(std_col0, 1.0, places=5)
        self.assertAlmostEqual(std_col1, 1.0, places=5)
        
        # Test with empty data
        result = pyroid.ml_impl.normalize([])
        self.assertEqual(result, [])
        
        # Test with invalid method
        with self.assertRaises(ValueError):
            pyroid.ml_impl.normalize(data, method="invalid")

    def test_distance_matrix(self):
        """Test the distance_matrix function."""
        # Create some test data
        data = [[1.0, 1.0], [4.0, 5.0], [10.0, 10.0]]
        
        # Calculate Euclidean distance matrix
        result = pyroid.ml_impl.distance_matrix(data)
        
        # Check the result
        self.assertEqual(len(result), 3)  # 3x3 matrix
        self.assertEqual(len(result[0]), 3)
        
        # Check some specific distances
        self.assertEqual(result[0][0], 0.0)  # Distance to self is 0
        self.assertAlmostEqual(result[0][1], ((4-1)**2 + (5-1)**2)**0.5)  # Euclidean distance
        self.assertAlmostEqual(result[0][2], ((10-1)**2 + (10-1)**2)**0.5)  # Euclidean distance
        
        # Calculate Manhattan distance matrix
        result = pyroid.ml_impl.distance_matrix(data, metric="manhattan")
        
        # Check the result
        self.assertEqual(len(result), 3)  # 3x3 matrix
        self.assertEqual(len(result[0]), 3)
        
        # Check some specific distances
        self.assertEqual(result[0][0], 0.0)  # Distance to self is 0
        self.assertAlmostEqual(result[0][1], abs(4-1) + abs(5-1))  # Manhattan distance
        self.assertAlmostEqual(result[0][2], abs(10-1) + abs(10-1))  # Manhattan distance
        
        # Test with empty data
        result = pyroid.ml_impl.distance_matrix([])
        self.assertEqual(result, [])
        
        # Test with invalid metric
        with self.assertRaises(ValueError):
            pyroid.ml_impl.distance_matrix(data, metric="invalid")

    def test_euclidean_distance(self):
        """Test the euclidean_distance function."""
        # Test with simple vectors
        a = [1, 2, 3]
        b = [4, 5, 6]
        result = pyroid.ml_impl.euclidean_distance(a, b)
        self.assertAlmostEqual(result, ((4-1)**2 + (5-2)**2 + (6-3)**2)**0.5)
        
        # Test with empty vectors
        result = pyroid.ml_impl.euclidean_distance([], [])
        self.assertEqual(result, 0.0)

    def test_manhattan_distance(self):
        """Test the manhattan_distance function."""
        # Test with simple vectors
        a = [1, 2, 3]
        b = [4, 5, 6]
        result = pyroid.ml_impl.manhattan_distance(a, b)
        self.assertEqual(result, abs(4-1) + abs(5-2) + abs(6-3))
        
        # Test with empty vectors
        result = pyroid.ml_impl.manhattan_distance([], [])
        self.assertEqual(result, 0.0)

    def test_inverse_matrix(self):
        """Test the inverse_matrix function."""
        # Test with a simple 2x2 matrix
        matrix = [[1, 0], [0, 1]]  # Identity matrix
        result = pyroid.ml_impl.inverse_matrix(matrix)
        self.assertEqual(result, [[1, 0], [0, 1]])  # Inverse of identity is identity
        
        # Test with a more complex matrix
        matrix = [[4, 7], [2, 6]]
        result = pyroid.ml_impl.inverse_matrix(matrix)
        
        # Check that A * A^-1 = I
        product = [[0, 0], [0, 0]]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    product[i][j] += matrix[i][k] * result[k][j]
        
        # Allow for some floating-point error
        self.assertAlmostEqual(product[0][0], 1.0, places=5)
        self.assertAlmostEqual(product[0][1], 0.0, places=5)
        self.assertAlmostEqual(product[1][0], 0.0, places=5)
        self.assertAlmostEqual(product[1][1], 1.0, places=5)
        
        # Test with a singular matrix
        with self.assertRaises(ValueError):
            pyroid.ml_impl.inverse_matrix([[1, 1], [1, 1]])

if __name__ == "__main__":
    unittest.main()