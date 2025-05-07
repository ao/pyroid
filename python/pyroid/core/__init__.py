"""
Pyroid Core Module
=================

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

# Import from our Python implementation
from ..core_impl import (
    Config, ConfigContext, SharedData,
    PyroidError, InputError, ComputationError, MemoryError, ConversionError, IoError,
    config, get_config
)

# Create dummy submodules
class runtime:
    """Dummy runtime module."""
    pass

class buffer:
    """Dummy buffer module."""
    pass

class parallel:
    """Dummy parallel module."""
    pass

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
    'runtime',
    'buffer',
    'parallel',
]