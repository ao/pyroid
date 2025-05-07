"""
Pyroid Image Implementation
=======================

This module provides Python implementations of the image functions.

Classes:
    Image: Image class for image processing operations

Functions:
    create_image: Create a new image
    from_bytes: Create an image from raw bytes
"""

from typing import List, Tuple, Union, Optional

class Image:
    """Image class for image processing operations."""
    
    def __init__(self, width: int, height: int, channels: int):
        """
        Create a new image.
        
        Args:
            width: The width of the image
            height: The height of the image
            channels: The number of channels (1 for grayscale, 3 for RGB, 4 for RGBA)
        """
        self.width = width
        self.height = height
        self.channels = channels
        self.data = [[0] * (channels * width) for _ in range(height)]
    
    def get_pixel(self, x: int, y: int) -> List[int]:
        """
        Get a pixel value.
        
        Args:
            x: The x coordinate
            y: The y coordinate
            
        Returns:
            The pixel value as a list of channel values
        """
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            raise IndexError("Pixel coordinates out of bounds")
        
        pixel = []
        for c in range(self.channels):
            pixel.append(self.data[y][x * self.channels + c])
        
        return pixel
    
    def set_pixel(self, x: int, y: int, value: List[int]):
        """
        Set a pixel value.
        
        Args:
            x: The x coordinate
            y: The y coordinate
            value: The pixel value as a list of channel values
        """
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            raise IndexError("Pixel coordinates out of bounds")
        
        if len(value) != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {len(value)}")
        
        for c in range(self.channels):
            self.data[y][x * self.channels + c] = value[c]
    
    def to_grayscale(self) -> 'Image':
        """
        Convert the image to grayscale.
        
        Returns:
            A new grayscale image
        """
        if self.channels == 1:
            return self
        
        result = Image(self.width, self.height, 1)
        
        for y in range(self.height):
            for x in range(self.width):
                pixel = self.get_pixel(x, y)
                
                # Use standard RGB to grayscale conversion
                if self.channels >= 3:
                    gray = int(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])
                else:
                    gray = pixel[0]  # Just use the first channel
                
                result.set_pixel(x, y, [gray])
        
        return result
    
    def resize(self, width: int, height: int) -> 'Image':
        """
        Resize the image.
        
        Args:
            width: The new width
            height: The new height
            
        Returns:
            A new resized image
        """
        result = Image(width, height, self.channels)
        
        # Simple nearest-neighbor scaling
        x_ratio = self.width / width
        y_ratio = self.height / height
        
        for y in range(height):
            for x in range(width):
                src_x = min(self.width - 1, int(x * x_ratio))
                src_y = min(self.height - 1, int(y * y_ratio))
                result.set_pixel(x, y, self.get_pixel(src_x, src_y))
        
        return result
    
    def blur(self, radius: int) -> 'Image':
        """
        Apply a blur filter to the image.
        
        Args:
            radius: The blur radius
            
        Returns:
            A new blurred image
        """
        result = Image(self.width, self.height, self.channels)
        
        # Simple box blur
        for y in range(self.height):
            for x in range(self.width):
                for c in range(self.channels):
                    total = 0
                    count = 0
                    
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            nx, ny = x + dx, y + dy
                            
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                total += self.data[ny][nx * self.channels + c]
                                count += 1
                    
                    if count > 0:
                        result.data[y][x * self.channels + c] = total // count
        
        return result
    
    def adjust_brightness(self, factor: float) -> 'Image':
        """
        Adjust the brightness of the image.
        
        Args:
            factor: The brightness factor (1.0 = no change)
            
        Returns:
            A new brightness-adjusted image
        """
        result = Image(self.width, self.height, self.channels)
        
        for y in range(self.height):
            for x in range(self.width):
                pixel = self.get_pixel(x, y)
                adjusted = [min(255, max(0, int(v * factor))) for v in pixel]
                result.set_pixel(x, y, adjusted)
        
        return result

def create_image(width: int, height: int, channels: int) -> Image:
    """
    Create a new image.
    
    Args:
        width: The width of the image
        height: The height of the image
        channels: The number of channels (1 for grayscale, 3 for RGB, 4 for RGBA)
        
    Returns:
        A new image
    """
    return Image(width, height, channels)

def from_bytes(data: bytes, width: int, height: int, channels: int) -> Image:
    """
    Create an image from raw bytes.
    
    Args:
        data: The raw image data
        width: The width of the image
        height: The height of the image
        channels: The number of channels (1 for grayscale, 3 for RGB, 4 for RGBA)
        
    Returns:
        A new image
    """
    if len(data) != width * height * channels:
        raise ValueError(f"Expected {width * height * channels} bytes, got {len(data)}")
    
    image = Image(width, height, channels)
    
    for y in range(height):
        for x in range(width):
            pixel = []
            for c in range(channels):
                index = (y * width + x) * channels + c
                pixel.append(data[index])
            image.set_pixel(x, y, pixel)
    
    return image