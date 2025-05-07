"""
Pyroid Machine Learning Module
===========================

This module provides high-performance machine learning operations.

Functions:
    kmeans: K-means clustering
    linear_regression: Linear regression
    normalize: Normalize data
    distance_matrix: Calculate distance matrix
"""

# Import directly from Rust extension
try:
    from .pyroid import (
        # Clustering
        kmeans,
        
        # Regression
        linear_regression,
        
        # Data preprocessing
        normalize,
        
        # Distance calculations
        distance_matrix,
    )
except ImportError as e:
    error_message = f"""
    ERROR: Pyroid machine learning operations could not be loaded!
    
    Pyroid requires the ML Rust extensions to be properly built and installed.
    
    Error: {str(e)}
    
    To fix this:
    1. Make sure you've installed pyroid with the Rust components:
       python build_and_install.py
    2. Check that the Rust toolchain is properly installed
    3. Verify that the compiled extensions (.so/.pyd files) exist in the package directory
    
    For more help, visit: https://github.com/ao/pyroid/issues
    """
    raise ImportError(error_message)

__all__ = [
    # Clustering
    'kmeans',
    
    # Regression
    'linear_regression',
    
    # Data preprocessing
    'normalize',
    
    # Distance calculations
    'distance_matrix',
]