"""
Pyroid: Python on Rust-Powered Steroids
======================================

High-performance Rust-powered utilities to eliminate Python's performance bottlenecks.

Modules:
    core: Core functionality and shared utilities
    math: Mathematical operations
    text: Text processing and NLP
    data: Data structures and operations
    io: File I/O and networking
    image: Image processing
    ml: Machine learning operations
    async_ops: Asynchronous operations

Examples:
    >>> import pyroid
    >>> # Create a configuration
    >>> config = pyroid.core.Config({"parallel": True, "chunk_size": 1000})
    >>> # Use the configuration with a context manager
    >>> with pyroid.config(parallel=True, chunk_size=1000):
    ...     # Perform operations with this configuration
    ...     result = pyroid.math.sum([1, 2, 3, 4, 5])
"""

import sys

# Import directly from Rust extension
# Import core classes from our Python implementation
from .core_impl import Config, ConfigContext, SharedData
from .core_impl import PyroidError, InputError, ComputationError, MemoryError, ConversionError, IoError

# Try to import async classes from the Rust extension
try:
    from .pyroid import AsyncClient, AsyncFileReader
except ImportError:
    # Create dummy classes if the Rust extension is not available
    class AsyncClient:
        """Dummy AsyncClient class."""
        def __init__(self, *args, **kwargs):
            pass
    
    class AsyncFileReader:
        """Dummy AsyncFileReader class."""
        def __init__(self, *args, **kwargs):
            pass
    
    import warnings
    warnings.warn("Pyroid Rust extensions could not be loaded. Using dummy implementations.")
# Import submodules
from . import core
from . import math
from . import text
from . import data
from . import io
from . import image
from . import ml
from . import async_ops

# Import convenience function from core_impl
from .core_impl import config

# Version information
__version__ = "0.7.0"

__all__ = [
    # Core classes
    'Config',
    'ConfigContext',
    'SharedData',
    
    # Error classes
    'PyroidError',
    'InputError',
    'ComputationError',
    'MemoryError',
    'ConversionError',
    'IoError',
    
    # Async classes
    'AsyncClient',
    'AsyncFileReader',
    
    # Submodules
    'core',
    'math',
    'text',
    'data',
    'io',
    'image',
    'ml',
    'async_ops',
    
    # Convenience functions
    'config',
]