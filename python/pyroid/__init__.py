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
try:
    # Try to import directly from the pyroid module
    from .pyroid import Config, ConfigContext, SharedData
    from .pyroid import PyroidError, InputError, ComputationError, MemoryError, ConversionError, IoError
except ImportError:
    # Fallback to importing from the core module
    from .core import Config, ConfigContext, SharedData
    from .core import PyroidError, InputError, ComputationError, MemoryError, ConversionError, IoError

# Import submodules
from . import core
from . import math
from . import text
from . import string
from . import data
from . import io
from . import image
from . import ml

# Import async functionality
# Use Python implementation for now
if False:  # Force using the Python implementation
    from .pyroid import AsyncClient, AsyncFileReader
else:
    # Fallback to importing from async_bridge
    from .async_bridge import fetch_url, fetch_many, download_file, read_file, read_file_lines, write_file, http_post
    
    # Create AsyncClient class if not available from Rust
    class AsyncClient:
        def __init__(self, timeout=None, concurrency=10, adaptive_concurrency=True):
            self.timeout = timeout
            self.concurrency = concurrency
            self.adaptive_concurrency = adaptive_concurrency
            
        def fetch(self, url):
            from .async_helpers import fetch_url_optimized
            # Return the coroutine directly so it can be awaited
            return fetch_url_optimized(url)
            
        def fetch_many(self, urls, concurrency=None):
            from .async_helpers import fetch_many_optimized
            # Return the coroutine directly so it can be awaited
            return fetch_many_optimized(urls, concurrency=concurrency or self.concurrency)
            
        async def download_file(self, url, path):
            from .async_helpers import download_file as async_download_file
            # Return the coroutine directly so it can be awaited
            return await async_download_file(url, path)
            
        def connection_pool_stats(self):
            from .async_helpers import _PERFORMANCE_METRICS
            return {
                "optimal_concurrency": dict(_PERFORMANCE_METRICS.get("optimal_concurrency", {})),
                "response_times_count": {host: len(times) for host, times in _PERFORMANCE_METRICS.get("response_times", {}).items()}
            }
    
    class AsyncFileReader:
        def __init__(self, path):
            self.path = path
            
        def read_all(self):
            from .async_helpers import read_file as async_read_file
            # Return the coroutine directly so it can be awaited
            return async_read_file(self.path)
            
        def read_lines(self):
            from .async_helpers import read_file_lines as async_read_file_lines
            # Return the coroutine directly so it can be awaited
            return async_read_file_lines(self.path)

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
__version__ = "0.6.0"
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
    'string',
    'data',
    'io',
    'image',
    'ml',
    
    # Convenience functions
    'config',
]