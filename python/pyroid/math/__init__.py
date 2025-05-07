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

# Import from our Python implementation
from ..math_impl import (
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

# Import stats from our Python implementation
from ..math_impl import stats

__all__ = [
    'Vector',
    'sum',
    'Matrix',
    'multiply',
    'mean',
    'median',
    'std',
    'variance',
    'correlation',
    'describe',
    'stats',
]