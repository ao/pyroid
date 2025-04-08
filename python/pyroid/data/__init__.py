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

# Try to import directly from the pyroid module
try:
    from ..pyroid import (
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
except ImportError:
    # Fallback to importing from the data module
    try:
        from ..pyroid.data import (
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
    except ImportError:
        # If all else fails, create dummy classes and functions for documentation purposes
        class DataFrame:
            """DataFrame class for data operations (not available)."""
            def __init__(self, data=None):
                self.data = data or {}
                
            def __repr__(self):
                return f"DataFrame({self.data})"
                
        def apply(df, func, axis=0):
            """Apply a function to a DataFrame (not available)."""
            return df
            
        def groupby_aggregate(df, by, agg_dict):
            """Group by and aggregate a DataFrame (not available)."""
            return df
            
        def filter(items, predicate):
            """Filter a list using a predicate function (not available)."""
            return [item for item in items if predicate(item)]
            
        def map(items, func):
            """Map a function over a list (not available)."""
            return [func(item) for item in items]
            
        def reduce(items, func, initial=None):
            """Reduce a list using a binary function (not available)."""
            if not items:
                return initial
            if initial is None:
                result = items[0]
                items = items[1:]
            else:
                result = initial
            for item in items:
                result = func(result, item)
            return result
            
        def sort(items, key=None, reverse=False):
            """Sort a list (not available)."""
            return sorted(items, key=key, reverse=reverse)

__all__ = [
    'DataFrame',
    'apply',
    'groupby_aggregate',
    'filter',
    'map',
    'reduce',
    'sort',
]