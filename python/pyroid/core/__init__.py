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

from ..pyroid import (
    # Config classes
    Config,
    ConfigContext,
    
    # Shared data
    SharedData,
    
    # Error classes
    PyroidError,
    InputError,
    ComputationError,
    MemoryError,
    ConversionError,
    IoError,
)

__all__ = [
    'Config',
    'ConfigContext',
    'SharedData',
    'PyroidError',
    'InputError',
    'ComputationError',
    'MemoryError',
    'ConversionError',
    'IoError',
]