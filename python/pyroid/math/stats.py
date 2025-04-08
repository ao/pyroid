"""
Pyroid Math Stats Module
======================

This module provides statistical functions for the Pyroid math module.
"""

# Try to import directly from the pyroid module
try:
    from ..pyroid import (
        mean,
        median,
        std as calc_std,
        variance,
        correlation,
        describe,
    )
except ImportError:
    # Fallback to importing from the math module
    try:
        from ..pyroid.math import (
            mean,
            median,
            std as calc_std,
            variance,
            correlation,
            describe,
        )
    except ImportError:
        # If all else fails, create dummy functions for documentation purposes
        def mean(values):
            """Calculate the mean of a list of numbers."""
            if not values:
                return 0
            return sum(values) / len(values)
            
        def median(values):
            """Calculate the median of a list of numbers."""
            if not values:
                return 0
            sorted_values = sorted(values)
            n = len(sorted_values)
            if n % 2 == 0:
                return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
            else:
                return sorted_values[n//2]
            
        def calc_std(values):
            """Calculate the standard deviation of a list of numbers."""
            if not values:
                return 0
            mean_val = mean(values)
            return (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
            
        def variance(values):
            """Calculate the variance of a list of numbers."""
            if not values:
                return 0
            mean_val = mean(values)
            return sum((x - mean_val) ** 2 for x in values) / len(values)
            
        def correlation(x, y):
            """Calculate the correlation coefficient between two lists of numbers."""
            if not x or not y or len(x) != len(y):
                return 0
            n = len(x)
            mean_x = mean(x)
            mean_y = mean(y)
            std_x = calc_std(x)
            std_y = calc_std(y)
            if std_x == 0 or std_y == 0:
                return 0
            return sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / (n * std_x * std_y)
            
        def describe(values):
            """Calculate descriptive statistics for a list of numbers."""
            if not values:
                return {
                    "count": 0,
                    "mean": 0,
                    "std": 0,
                    "min": 0,
                    "25%": 0,
                    "50%": 0,
                    "75%": 0,
                    "max": 0,
                }
            n = len(values)
            sorted_values = sorted(values)
            return {
                "count": n,
                "mean": mean(values),
                "std": calc_std(values),
                "min": sorted_values[0],
                "25%": sorted_values[n//4],
                "50%": median(values),
                "75%": sorted_values[3*n//4],
                "max": sorted_values[-1],
            }

__all__ = [
    'mean',
    'median',
    'calc_std',
    'variance',
    'correlation',
    'describe',
]