"""
Pyroid Math Implementation
=======================

This module provides Python implementations of the math functions.

Classes:
    Vector: Vector class for mathematical operations
    Matrix: Matrix class for mathematical operations

Functions:
    sum: Sum a list of numbers
    multiply: Matrix multiplication function
    mean: Calculate the mean of a list of numbers
    median: Calculate the median of a list of numbers
    std: Calculate the standard deviation of a list of numbers
    variance: Calculate the variance of a list of numbers
    correlation: Calculate the correlation coefficient between two lists of numbers
    describe: Calculate descriptive statistics for a list of numbers
"""

import math
import statistics
from typing import List, Dict, Any, Union, Optional

class Vector:
    """Vector class for mathematical operations."""
    
    def __init__(self, values: List[float]):
        """
        Create a new vector.
        
        Args:
            values: The vector values
        """
        self.values = list(values)
    
    def __getitem__(self, index: int) -> float:
        """Get a value by index."""
        return self.values[index]
    
    def __len__(self) -> int:
        """Get the length of the vector."""
        return len(self.values)
    
    def __add__(self, other: 'Vector') -> 'Vector':
        """Add two vectors."""
        if len(self) != len(other):
            raise ValueError("Vectors must have the same length")
        return Vector([a + b for a, b in zip(self.values, other.values)])
    
    def __sub__(self, other: 'Vector') -> 'Vector':
        """Subtract two vectors."""
        if len(self) != len(other):
            raise ValueError("Vectors must have the same length")
        return Vector([a - b for a, b in zip(self.values, other.values)])
    
    def __mul__(self, scalar: float) -> 'Vector':
        """Multiply a vector by a scalar."""
        return Vector([a * scalar for a in self.values])
    
    def __truediv__(self, scalar: float) -> 'Vector':
        """Divide a vector by a scalar."""
        return Vector([a / scalar for a in self.values])
    
    def dot(self, other: 'Vector') -> float:
        """Calculate the dot product of two vectors."""
        if len(self) != len(other):
            raise ValueError("Vectors must have the same length")
        return sum(a * b for a, b in zip(self.values, other.values))
    
    def norm(self) -> float:
        """Calculate the norm of the vector."""
        return math.sqrt(sum(a * a for a in self.values))

class Matrix:
    """Matrix class for mathematical operations."""
    
    def __init__(self, values: List[List[float]]):
        """
        Create a new matrix.
        
        Args:
            values: The matrix values
        """
        self.values = [list(row) for row in values]
        self.rows = len(values)
        self.cols = len(values[0]) if values else 0
    
    def __getitem__(self, index: int) -> List[float]:
        """Get a row by index."""
        return self.values[index]
    
    def __len__(self) -> int:
        """Get the number of rows in the matrix."""
        return self.rows
    
    def __add__(self, other: 'Matrix') -> 'Matrix':
        """Add two matrices."""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions")
        return Matrix([[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(self.values, other.values)])
    
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        """Subtract two matrices."""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions")
        return Matrix([[a - b for a, b in zip(row1, row2)] for row1, row2 in zip(self.values, other.values)])
    
    def __mul__(self, other: Union['Matrix', float]) -> 'Matrix':
        """Multiply a matrix by another matrix or a scalar."""
        if isinstance(other, Matrix):
            return multiply(self, other)
        else:
            return Matrix([[a * other for a in row] for row in self.values])
    
    def transpose(self) -> 'Matrix':
        """Transpose the matrix."""
        return Matrix([[self.values[j][i] for j in range(self.rows)] for i in range(self.cols)])

def sum(values: List[float]) -> float:
    """
    Sum a list of numbers.
    
    Args:
        values: The values to sum
        
    Returns:
        The sum of the values
    """
    return builtins_sum(values)

def multiply(a: Matrix, b: Matrix) -> Matrix:
    """
    Multiply two matrices.
    
    Args:
        a: The first matrix
        b: The second matrix
        
    Returns:
        The product of the matrices
    """
    if a.cols != b.rows:
        raise ValueError("Matrix dimensions do not match")
    
    result = [[0 for _ in range(b.cols)] for _ in range(a.rows)]
    for i in range(a.rows):
        for j in range(b.cols):
            for k in range(a.cols):
                result[i][j] += a.values[i][k] * b.values[k][j]
    
    return Matrix(result)

def mean(values: List[float]) -> float:
    """
    Calculate the mean of a list of numbers.
    
    Args:
        values: The values to calculate the mean of
        
    Returns:
        The mean of the values
    """
    if not values:
        return 0.0
    return statistics.mean(values)

def median(values: List[float]) -> float:
    """
    Calculate the median of a list of numbers.
    
    Args:
        values: The values to calculate the median of
        
    Returns:
        The median of the values
    """
    if not values:
        return 0.0
    return statistics.median(values)

def std(values: List[float]) -> float:
    """
    Calculate the standard deviation of a list of numbers.
    
    Args:
        values: The values to calculate the standard deviation of
        
    Returns:
        The standard deviation of the values
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return 0.0  # Standard deviation of a single value is 0
    return statistics.stdev(values)

def variance(values: List[float]) -> float:
    """
    Calculate the variance of a list of numbers.
    
    Args:
        values: The values to calculate the variance of
        
    Returns:
        The variance of the values
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return 0.0  # Variance of a single value is 0
    return statistics.variance(values)

def correlation(x: List[float], y: List[float]) -> float:
    """
    Calculate the correlation coefficient between two lists of numbers.
    
    Args:
        x: The first list of values
        y: The second list of values
        
    Returns:
        The correlation coefficient
    """
    if not x or not y or len(x) != len(y):
        return 0.0
    
    n = len(x)
    if n == 1:
        return 0.0  # Correlation requires at least two points
    
    mean_x = mean(x)
    mean_y = mean(y)
    
    # Calculate standard deviations manually to ensure consistency
    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)
    
    if sum_sq_x == 0 or sum_sq_y == 0:
        return 0.0
    
    std_x = (sum_sq_x / n) ** 0.5
    std_y = (sum_sq_y / n) ** 0.5
    
    # Calculate covariance
    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
    
    # Calculate correlation coefficient
    return covariance / (std_x * std_y)

def describe(values: List[float]) -> Dict[str, Any]:
    """
    Calculate descriptive statistics for a list of numbers.
    
    Args:
        values: The values to calculate statistics for
        
    Returns:
        A dictionary of statistics
    """
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "25%": 0.0,
            "50%": 0.0,
            "75%": 0.0,
            "max": 0.0,
        }
    
    n = len(values)
    sorted_values = sorted(values)
    
    return {
        "count": n,
        "mean": mean(values),
        "std": std(values),
        "min": sorted_values[0],
        "25%": sorted_values[n // 4],
        "50%": median(values),
        "75%": sorted_values[3 * n // 4],
        "max": sorted_values[-1],
    }

# Import the built-in sum function to avoid name conflicts
import builtins
builtins_sum = builtins.sum

# Create a stats namespace
class stats:
    """Statistical functions namespace."""
    
    # Import statistical functions
    mean = mean
    median = median
    calc_std = std
    variance = variance
    correlation = correlation
    describe = describe