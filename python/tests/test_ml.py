#!/usr/bin/env python3
"""
Integration tests for Pyroid machine learning functionality.

This script contains tests to verify that the ML Rust extensions are working correctly.
"""

import unittest
import pyroid
import math

class TestClustering(unittest.TestCase):
    """Test the clustering operations."""
    
    def test_kmeans(self):
        """Test K-means clustering."""
        # Create a simple dataset with two clear clusters
        data = [
            [1.0, 1.0],
            [1.5, 1.5],
            [1.2, 1.3],
            [5.0, 5.0],
            [5.5, 5.5],
            [5.2, 5.3]
        ]
        
        # Run K-means with k=2
        result = pyroid.ml.kmeans(data, k=2, max_iterations=100)
        
        # Check that we have the expected keys in the result
        self.assertIn("centroids", result)
        self.assertIn("clusters", result)
        
        # Check that we have 2 centroids
        self.assertEqual(len(result["centroids"]), 2)
        
        # Check that we have 2 clusters
        self.assertEqual(len(result["clusters"]), 2)
        
        # Check that all data points are assigned to clusters
        total_points = sum(len(cluster) for cluster in result["clusters"])
        self.assertEqual(total_points, len(data))
        
        # Check that the centroids are reasonable
        # One centroid should be close to [1.2, 1.3]
        # The other should be close to [5.2, 5.3]
        centroids = result["centroids"]
        
        # Calculate distances from expected centroids
        dist1_to_low = math.sqrt((centroids[0][0] - 1.2)**2 + (centroids[0][1] - 1.3)**2)
        dist1_to_high = math.sqrt((centroids[0][0] - 5.2)**2 + (centroids[0][1] - 5.3)**2)
        dist2_to_low = math.sqrt((centroids[1][0] - 1.2)**2 + (centroids[1][1] - 1.3)**2)
        dist2_to_high = math.sqrt((centroids[1][0] - 5.2)**2 + (centroids[1][1] - 5.3)**2)
        
        # Either centroid 1 is close to the low cluster and centroid 2 is close to the high cluster,
        # or vice versa
        self.assertTrue(
            (dist1_to_low < 1.0 and dist2_to_high < 1.0) or
            (dist1_to_high < 1.0 and dist2_to_low < 1.0)
        )

class TestRegression(unittest.TestCase):
    """Test the regression operations."""
    
    def test_linear_regression(self):
        """Test linear regression."""
        # Create a simple dataset with a clear linear relationship: y = 2x + 1
        X = [[1], [2], [3], [4], [5]]
        y = [3, 5, 7, 9, 11]  # 2*x + 1
        
        # Run linear regression
        result = pyroid.ml.linear_regression(X, y)
        
        # Check that we have the expected keys in the result
        self.assertIn("coefficients", result)
        self.assertIn("intercept", result)
        self.assertIn("r_squared", result)
        
        # Check that the coefficients and intercept are close to the expected values
        self.assertAlmostEqual(result["intercept"], 1.0, delta=0.1)
        self.assertEqual(len(result["coefficients"]), 1)
        self.assertAlmostEqual(result["coefficients"][0], 2.0, delta=0.1)
        
        # Check that R-squared is close to 1.0 (perfect fit)
        self.assertAlmostEqual(result["r_squared"], 1.0, delta=0.01)

class TestDataPreprocessing(unittest.TestCase):
    """Test the data preprocessing operations."""
    
    def test_normalize_min_max(self):
        """Test min-max normalization."""
        # Create a simple dataset
        data = [
            [1, 10, 100],
            [2, 20, 200],
            [3, 30, 300],
            [4, 40, 400],
            [5, 50, 500]
        ]
        
        # Run min-max normalization
        result = pyroid.ml.normalize(data, method="min-max")
        
        # Check dimensions
        self.assertEqual(len(result), len(data))
        self.assertEqual(len(result[0]), len(data[0]))
        
        # Check that values are in the range [0, 1]
        for row in result:
            for value in row:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
        
        # Check that the first column is normalized correctly
        # First column: [1, 2, 3, 4, 5] -> [0, 0.25, 0.5, 0.75, 1.0]
        expected_col1 = [0.0, 0.25, 0.5, 0.75, 1.0]
        for i in range(len(data)):
            self.assertAlmostEqual(result[i][0], expected_col1[i], delta=0.01)
    
    def test_normalize_z_score(self):
        """Test z-score normalization."""
        # Create a simple dataset
        data = [
            [1, 10, 100],
            [2, 20, 200],
            [3, 30, 300],
            [4, 40, 400],
            [5, 50, 500]
        ]
        
        # Run z-score normalization
        result = pyroid.ml.normalize(data, method="z-score")
        
        # Check dimensions
        self.assertEqual(len(result), len(data))
        self.assertEqual(len(result[0]), len(data[0]))
        
        # Check that the first column is normalized correctly
        # First column: [1, 2, 3, 4, 5]
        # Mean = 3, Std = sqrt(2)
        # Z-scores: [-1.414, -0.707, 0, 0.707, 1.414]
        expected_col1 = [-1.414, -0.707, 0, 0.707, 1.414]
        for i in range(len(data)):
            self.assertAlmostEqual(result[i][0], expected_col1[i], delta=0.01)

class TestDistanceCalculations(unittest.TestCase):
    """Test the distance calculation operations."""
    
    def test_distance_matrix_euclidean(self):
        """Test Euclidean distance matrix calculation."""
        # Create a simple dataset
        data = [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1]
        ]
        
        # Run distance matrix calculation
        result = pyroid.ml.distance_matrix(data, metric="euclidean")
        
        # Check dimensions
        self.assertEqual(len(result), len(data))
        self.assertEqual(len(result[0]), len(data))
        
        # Check that the diagonal is all zeros
        for i in range(len(data)):
            self.assertEqual(result[i][i], 0.0)
        
        # Check specific distances
        # Distance from [0,0] to [1,0] should be 1
        self.assertAlmostEqual(result[0][1], 1.0, delta=0.01)
        
        # Distance from [0,0] to [0,1] should be 1
        self.assertAlmostEqual(result[0][2], 1.0, delta=0.01)
        
        # Distance from [0,0] to [1,1] should be sqrt(2)
        self.assertAlmostEqual(result[0][3], math.sqrt(2), delta=0.01)
        
        # Distance from [1,0] to [0,1] should be sqrt(2)
        self.assertAlmostEqual(result[1][2], math.sqrt(2), delta=0.01)
    
    def test_distance_matrix_manhattan(self):
        """Test Manhattan distance matrix calculation."""
        # Create a simple dataset
        data = [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1]
        ]
        
        # Run distance matrix calculation
        result = pyroid.ml.distance_matrix(data, metric="manhattan")
        
        # Check dimensions
        self.assertEqual(len(result), len(data))
        self.assertEqual(len(result[0]), len(data))
        
        # Check that the diagonal is all zeros
        for i in range(len(data)):
            self.assertEqual(result[i][i], 0.0)
        
        # Check specific distances
        # Distance from [0,0] to [1,0] should be 1
        self.assertAlmostEqual(result[0][1], 1.0, delta=0.01)
        
        # Distance from [0,0] to [0,1] should be 1
        self.assertAlmostEqual(result[0][2], 1.0, delta=0.01)
        
        # Distance from [0,0] to [1,1] should be 2
        self.assertAlmostEqual(result[0][3], 2.0, delta=0.01)
        
        # Distance from [1,0] to [0,1] should be 2
        self.assertAlmostEqual(result[1][2], 2.0, delta=0.01)

if __name__ == "__main__":
    unittest.main()