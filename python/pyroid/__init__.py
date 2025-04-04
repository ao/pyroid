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
- dataframe_ops: Pandas-like operations
- ml_ops: Machine learning operations
- text_nlp_ops: Text and NLP processing
- io_ops: File I/O operations
- image_ops: Image processing operations

Examples:
---------
>>> import pyroid
>>> # Parallel sum of a large list
>>> numbers = list(range(1_000_000))
>>> result = pyroid.parallel_sum(numbers)
"""

# Core functionality
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

# DataFrame operations
try:
    from .pyroid import (
        dataframe_apply,
        dataframe_groupby_aggregate,
        parallel_transform,
        parallel_join,
    )
except ImportError:
    pass

# Machine learning operations
try:
    from .pyroid import (
        parallel_distance_matrix,
        parallel_feature_scaling,
        parallel_cross_validation,
    )
except ImportError:
    pass

# Text and NLP operations
try:
    from .pyroid import (
        parallel_tokenize,
        parallel_ngrams,
        parallel_tfidf,
        parallel_document_similarity,
    )
except ImportError:
    pass

# File I/O operations
try:
    from .pyroid import (
        parallel_read_csv,
        parallel_json_parse,
        parallel_compress,
        parallel_decompress,
    )
except ImportError:
    pass

# Image processing operations
try:
    from .pyroid import (
        parallel_resize,
        parallel_convert,
        parallel_extract_metadata,
    )
    # Rename to avoid conflict with parallel_filter from data_ops
    from .pyroid import parallel_filter as parallel_image_filter
except ImportError:
    pass

# Version information
__version__ = "0.1.0"