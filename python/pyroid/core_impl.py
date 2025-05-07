"""
Pyroid Core Implementation
========================

This module provides core functionality and shared utilities for Pyroid.

Classes:
    Config: Configuration management
    ConfigContext: Context manager for temporary configuration
    SharedData: Wrapper for shared data

Exceptions:
    PyroidError: Base exception for all Pyroid errors
    InputError: Input validation error
    ComputationError: Computation error
    MemoryError: Memory error
    ConversionError: Type conversion error
    IoError: I/O error
"""

import threading
from contextlib import contextmanager

# Global configuration store
_global_config = {}

# Thread-local configuration store
_thread_local = threading.local()

class Config:
    """Configuration management for Pyroid."""
    
    def __init__(self, options=None):
        """
        Create a new configuration.
        
        Args:
            options: A dictionary of configuration options
        """
        self.options = options or {}
    
    def get(self, key, default=None):
        """
        Get a configuration value.
        
        Args:
            key: The key to get
            default: The default value if the key is not found
            
        Returns:
            The configuration value
        """
        return self.options.get(key, default)
    
    def set(self, key, value):
        """
        Set a configuration value.
        
        Args:
            key: The key to set
            value: The value to set
        """
        self.options[key] = value

class ConfigContext:
    """Context manager for temporary configuration."""
    
    def __init__(self, config):
        """
        Create a new configuration context.
        
        Args:
            config: The configuration to use
        """
        self.config = config
        self.previous_config = None
    
    def __enter__(self):
        """Enter the context manager."""
        # Save the current thread-local config
        self.previous_config = getattr(_thread_local, 'config', None)
        
        # Set the new config
        _thread_local.config = self.config
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        # Restore the previous config
        if self.previous_config is None:
            # If there was no previous config, delete the attribute
            delattr(_thread_local, 'config')
        else:
            _thread_local.config = self.previous_config
        
        # Don't suppress exceptions
        return False

class SharedData:
    """Wrapper for shared data."""
    
    def __init__(self, data):
        """
        Create a new shared data object.
        
        Args:
            data: The data to share
        """
        self.data = data
    
    def get(self):
        """
        Get the shared data.
        
        Returns:
            The shared data
        """
        return self.data

# Exception classes
class PyroidError(Exception):
    """Base exception for all Pyroid errors."""
    pass

class InputError(PyroidError):
    """Input validation error."""
    pass

class ComputationError(PyroidError):
    """Computation error."""
    pass

class MemoryError(PyroidError):
    """Memory error."""
    pass

class ConversionError(PyroidError):
    """Type conversion error."""
    pass

class IoError(PyroidError):
    """I/O error."""
    pass

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

# Get the current configuration
def get_config():
    """
    Get the current configuration.
    
    Returns:
        The current configuration
    """
    # Try to get the thread-local config
    config = getattr(_thread_local, 'config', None)
    
    # Fall back to the global config
    if config is None:
        config = _global_config
    
    return config