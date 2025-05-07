"""
Pyroid Math Module
================

This module provides high-performance mathematical operations.

Classes:
    Vector: Vector class for mathematical operations
    Matrix: Matrix class for mathematical operations

Functions:
    sum: Sum a list of numbers in parallel
    multiply: Matrix multiplication function
    mean: Calculate the mean of a list of numbers
    median: Calculate the median of a list of numbers
    std: Calculate the standard deviation of a list of numbers
    variance: Calculate the variance of a list of numbers
    correlation: Calculate the correlation coefficient between two lists of numbers
    describe: Calculate descriptive statistics for a list of numbers
"""

# Import directly from Rust extension
try:
    from .pyroid import (
        # Vector operations
        Vector,
        sum,
        
        # Matrix operations
        Matrix,
        multiply,
        
        # Statistical operations
        mean,
        median,
        std,
        variance,
        correlation,
        describe,
    )
except ImportError as e:
    error_message = f"""
    ERROR: Pyroid math operations could not be loaded!
    
    Pyroid requires the math Rust extensions to be properly built and installed.
    
    Error: {str(e)}
    
    To fix this:
    1. Make sure you've installed pyroid with the Rust components:
       python build_and_install.py
    2. Check that the Rust toolchain is properly installed
    3. Verify that the compiled extensions (.so/.pyd files) exist in the package directory
    
    For more help, visit: https://github.com/ao/pyroid/issues
    """
    raise ImportError(error_message)

# Create stats namespace
class stats:
    """Statistical functions namespace."""
    
    # Import statistical functions
    mean = mean
    median = median
    calc_std = std
    variance = variance
    correlation = correlation
    describe = describe
    
    __all__ = [
        'mean',
        'median',
        'calc_std',
        'variance',
        'correlation',
        'describe',
    ]

__all__ = [
    # Vector operations
    'Vector',
    'sum',
    
    # Matrix operations
    'Matrix',
    'multiply',
    
    # Statistical operations
    'mean',
    'median',
    'std',
    'variance',
    'correlation',
    'describe',
    
    # Namespaces
    'stats',
]