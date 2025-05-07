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

# Import directly from Rust extension
try:
    from .pyroid import (
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
except ImportError as e:
    error_message = f"""
    ERROR: Pyroid data operations could not be loaded!
    
    Pyroid requires the data Rust extensions to be properly built and installed.
    
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
    # DataFrame operations
    'DataFrame',
    'apply',
    'groupby_aggregate',
    
    # Collection operations
    'filter',
    'map',
    'reduce',
    'sort',
]