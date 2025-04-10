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

# Import submodules
try:
    from . import runtime
    from . import buffer
    from . import parallel
except ImportError:
    # Create dummy modules if imports fail
    class DummyModule:
        def __init__(self, name):
            self.__name__ = name
        def __getattr__(self, name):
            raise AttributeError(f"{self.__name__}.{name} is not available")
    
    runtime = DummyModule("runtime")
    buffer = DummyModule("buffer")
    parallel = DummyModule("parallel")

# Try to import directly from the pyroid module
try:
    from ..pyroid import Config, ConfigContext, SharedData
    from ..pyroid import PyroidError, InputError, ComputationError, MemoryError, ConversionError, IoError
except ImportError:
    # Fallback to importing from the core module
    try:
        from ..pyroid.core import Config, ConfigContext, SharedData
        from ..pyroid.core.error import PyroidError, InputError, ComputationError, MemoryError, ConversionError, IoError
    except ImportError:
        # If all else fails, create dummy classes for documentation purposes
        class Config:
            """Configuration management class (not available)."""
            pass
            
        class ConfigContext:
            """Context manager for temporary configuration (not available)."""
            pass
            
        class SharedData:
            """Wrapper for shared data (not available)."""
            pass
            
        class PyroidError(Exception):
            """Base exception for all Pyroid errors (not available)."""
            pass
            
        class InputError(PyroidError):
            """Input validation error (not available)."""
            pass
            
        class ComputationError(PyroidError):
            """Computation error (not available)."""
            pass
            
        class MemoryError(PyroidError):
            """Memory error (not available)."""
            pass
            
        class ConversionError(PyroidError):
            """Type conversion error (not available)."""
            pass
            
        class IoError(PyroidError):
            """I/O error (not available)."""
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