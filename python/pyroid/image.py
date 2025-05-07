"""
Pyroid Image Module
================

This module provides high-performance image processing operations.

Classes:
    Image: Image class for image processing operations

Functions:
    create_image: Create a new image
    from_bytes: Create an image from raw bytes
    to_grayscale: Convert an image to grayscale
    resize: Resize an image
    blur: Apply a blur filter to an image
    adjust_brightness: Adjust the brightness of an image
"""

# Import directly from Rust extension
try:
    from .pyroid import (
        # Image class
        Image,
        
        # Image creation
        create_image,
        from_bytes,
    )
except ImportError as e:
    error_message = f"""
    ERROR: Pyroid image operations could not be loaded!
    
    Pyroid requires the image Rust extensions to be properly built and installed.
    
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
    # Image class
    'Image',
    
    # Image creation
    'create_image',
    'from_bytes',
]