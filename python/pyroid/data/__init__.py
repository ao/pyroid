"""
Pyroid Data Module
================

This module provides high-performance data operations.

Classes:
    DataFrame: DataFrame class for data operations

Functions:
    filter: Filter a list using a predicate function in parallel
    map: Map a function over a list in parallel
    reduce: Reduce a list using a binary function
    sort: Sort a list in parallel
    apply: Apply a function to a DataFrame
    groupby_aggregate: Group by and aggregate a DataFrame
"""

# Import from our Python implementation
from ..data_impl import (
    # DataFrame operations
    DataFrame,
    apply,
    groupby_aggregate,
    
    # Collection operations
    filter,
    map,
    reduce,
    sort,
)

__all__ = [
    'DataFrame',
    'apply',
    'groupby_aggregate',
    'filter',
    'map',
    'reduce',
    'sort',
]