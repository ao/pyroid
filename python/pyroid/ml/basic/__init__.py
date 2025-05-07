"""
Pyroid Machine Learning Basic Module
================================

This module provides basic machine learning operations.

Functions:
    kmeans: K-means clustering
    linear_regression: Linear regression
    normalize: Normalize data
    distance_matrix: Calculate distance matrix
"""

# Import from our Python implementation
from ...ml_impl import (
    # Clustering
    kmeans,
    
    # Regression
    linear_regression,
    
    # Data preprocessing
    normalize,
    
    # Distance calculations
    distance_matrix,
)

__all__ = [
    'kmeans',
    'linear_regression',
    'normalize',
    'distance_matrix',
]