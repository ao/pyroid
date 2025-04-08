"""
Pyroid: Python on Rust-Powered Steroids
======================================

Pyroid is a high-performance library that uses Rust to accelerate common
operations that are typically slow in pure Python.

Modules:
    core: Core functionality and shared utilities
    math: Mathematical operations
    text: Text processing and NLP
    data: Data structures and operations
    io: File I/O and networking
    image: Image processing
    ml: Machine learning operations

Examples:
    >>> import pyroid
    >>> # Create a configuration
    >>> config = pyroid.core.Config({"parallel": True, "chunk_size": 1000})
    >>> # Use the configuration with a context manager
    >>> with pyroid.config(parallel=True, chunk_size=1000):
    ...     # Perform operations with this configuration
    ...     result = pyroid.math.sum([1, 2, 3, 4, 5])
"""

# Import core functionality
from .core import (
    Config,
    ConfigContext,
    SharedData,
    PyroidError,
    InputError,
    ComputationError,
    MemoryError,
    ConversionError,
    IoError,
)

# Import submodules
from . import core
from . import math
from . import text
from . import data
from . import io
from . import image
from . import ml

# Convenience function for creating a configuration context
def config(**kwargs):
    """
    Create a configuration context with the specified options.
    
    Args:
        **kwargs: Configuration options as keyword arguments
        
    Returns:
        A context manager for the configuration
        
    Example:
        >>> with pyroid.config(parallel=True, chunk_size=1000):
        ...     result = pyroid.math.sum([1, 2, 3, 4, 5])
    """
    return ConfigContext(Config(kwargs))

# Version information
__version__ = "0.2.1"

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
    
    # Submodules
    'core',
    'math',
    'text',
    'data',
    'io',
    'image',
    'ml',
    
    # Convenience functions
    'config',
]