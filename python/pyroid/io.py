"""
Pyroid I/O Module
===============

This module provides high-performance I/O operations.

Functions:
    read_file: Read a file
    write_file: Write a file
    read_files: Read multiple files in parallel
    get: HTTP GET request
    post: HTTP POST request
    sleep: Async sleep
    read_file_async: Async read file
    write_file_async: Async write file
"""

# Import directly from Rust extension
try:
    from .pyroid import (
        # File operations
        read_file,
        write_file,
        read_files,
        
        # Network operations
        get,
        post,
        
        # Async operations
        sleep,
        read_file_async,
        write_file_async,
    )
except ImportError as e:
    error_message = f"""
    ERROR: Pyroid I/O operations could not be loaded!
    
    Pyroid requires the I/O Rust extensions to be properly built and installed.
    
    Error: {str(e)}
    
    To fix this:
    1. Make sure you've installed pyroid with the Rust components:
       python build_and_install.py
    2. Check that the Rust toolchain is properly installed
    3. Verify that the compiled extensions (.so/.pyd files) exist in the package directory
    
    For more help, visit: https://github.com/ao/pyroid/issues
    """
    raise ImportError(error_message)

__all__ = [
    # File operations
    'read_file',
    'write_file',
    'read_files',
    
    # Network operations
    'get',
    'post',
    
    # Async operations
    'sleep',
    'read_file_async',
    'write_file_async',
]