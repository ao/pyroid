"""
Zero-copy buffer protocol implementation for Pyroid.

This module provides efficient zero-copy buffer implementations for data transfer
between Python and Rust.
"""

import array
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

class ZeroCopyBuffer:
    """A buffer that can be shared between Python and Rust without copying."""
    
    def __init__(self, size: int, readonly: bool = False):
        """Create a new zero-copy buffer with the specified size.
        
        Args:
            size: The size of the buffer in bytes
            readonly: Whether the buffer is readonly
        """
        self._data = bytearray(size)
        self._readonly = readonly
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'ZeroCopyBuffer':
        """Create a new zero-copy buffer from existing data.
        
        Args:
            data: The data to copy into the buffer
            
        Returns:
            A new ZeroCopyBuffer containing the data
        """
        buffer = cls(len(data), readonly=True)
        buffer._data[:] = data
        return buffer
    
    @classmethod
    def from_numpy_array(cls, array: np.ndarray) -> 'ZeroCopyBuffer':
        """Create a zero-copy buffer from a NumPy array.
        
        Args:
            array: The NumPy array to create a buffer from
            
        Returns:
            A new ZeroCopyBuffer containing the data
        """
        # Convert to bytes using numpy's tobytes() method
        data = array.tobytes()
        return cls.from_bytes(data)
    
    @property
    def size(self) -> int:
        """Get the size of the buffer."""
        return len(self._data)
    
    def as_bytes(self) -> bytes:
        """Get a copy of the buffer as bytes."""
        return bytes(self._data)
    
    def get_data(self) -> bytearray:
        """Get a reference to the underlying data."""
        return self._data
    
    def get_data_ptr(self) -> int:
        """Get a reference to the underlying data pointer as an integer."""
        # This is a placeholder. In the Rust implementation, this would return
        # the actual pointer to the data.
        return id(self._data)
    
    def to_numpy_array(self, dtype=None, shape=None) -> np.ndarray:
        """Get the data as a numpy array.
        
        Args:
            dtype: The data type of the array (default: uint8)
            shape: The shape of the array (default: 1D array)
            
        Returns:
            A NumPy array view of the data
        """
        if dtype is None:
            dtype = np.uint8
        
        # Create a NumPy array from the buffer
        arr = np.frombuffer(self._data, dtype=dtype)
        
        # Reshape if needed
        if shape is not None:
            arr = arr.reshape(shape)
        
        return arr
    
    def set_data(self, data: bytes) -> None:
        """Set data in the buffer.
        
        Args:
            data: The data to copy into the buffer
            
        Raises:
            ValueError: If the buffer is readonly or the data size doesn't match
        """
        if self._readonly:
            raise ValueError("Buffer is readonly")
        
        if len(data) != len(self._data):
            raise ValueError("Data size mismatch")
        
        self._data[:] = data

class MemoryView:
    """A memory view that provides efficient access to memory."""
    
    def __init__(self, size: int, readonly: bool = False):
        """Create a new memory view with the specified size.
        
        Args:
            size: The size of the memory view in bytes
            readonly: Whether the memory view is readonly
        """
        self._data = bytearray(size)
        self._readonly = readonly
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'MemoryView':
        """Create a memory view from existing data.
        
        Args:
            data: The data to copy into the memory view
            
        Returns:
            A new MemoryView containing the data
        """
        view = cls(len(data), readonly=True)
        view._data[:] = data
        return view
    
    @property
    def size(self) -> int:
        """Get the size of the memory view."""
        return len(self._data)
    
    def as_bytes(self) -> bytes:
        """Get a copy of the memory as bytes."""
        return bytes(self._data)
    
    def get_data(self) -> bytearray:
        """Get a reference to the underlying data."""
        return self._data
    
    def set_data(self, data: bytes) -> None:
        """Set data in the memory view.
        
        Args:
            data: The data to copy into the memory view
            
        Raises:
            ValueError: If the memory view is readonly or the data size doesn't match
        """
        if self._readonly:
            raise ValueError("Memory view is readonly")
        
        if len(data) != len(self._data):
            raise ValueError("Data size mismatch")
        
        self._data[:] = data