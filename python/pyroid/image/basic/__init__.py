"""
Pyroid Image Basic Module
======================

This module provides basic image processing operations.

Functions:
    create_image: Create a new image
    from_bytes: Create an image from raw bytes
    to_grayscale: Convert an image to grayscale
    resize: Resize an image
    blur: Apply a blur filter to an image
    adjust_brightness: Adjust the brightness of an image
"""

# Import from our Python implementation
from ...image_impl import (
    # Image creation
    create_image,
    from_bytes,
    
    # Image operations
    Image,
)

# Export the Image class and functions
__all__ = [
    'Image',
    'create_image',
    'from_bytes',
]