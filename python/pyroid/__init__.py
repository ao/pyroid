"""
pyroid: High-performance Rust functions for Python
==================================================

This package provides high-performance Rust implementations of common
operations that are typically slow in pure Python.

Main modules:
------------
- string_ops: Fast string processing operations
- math_ops: Accelerated mathematical operations
- data_ops: Efficient data processing functions
- async_ops: Non-blocking operations using Tokio

Examples:
---------
>>> import pyroid
>>> # Parallel sum of a large list
>>> numbers = list(range(1_000_000))
>>> result = pyroid.parallel_sum(numbers)
"""

from .pyroid import (
    # String operations
    parallel_regex_replace,
    parallel_text_cleanup,
    parallel_base64_encode,
    parallel_base64_decode,
    
    # Math operations
    parallel_sum,
    parallel_product,
    parallel_mean,
    parallel_std,
    parallel_apply,
    matrix_multiply,
    
    # Data operations
    parallel_filter,
    parallel_map,
    parallel_reduce,
    parallel_sort,
    
    # Async operations
    AsyncClient,
    AsyncChannel,
    AsyncFileReader,
    async_sleep,
    gather,
)

# Version information
__version__ = "0.1.0"