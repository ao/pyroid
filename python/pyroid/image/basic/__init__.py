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

# Try to import directly from the pyroid module
try:
    from ...pyroid import (
        # Image creation
        create_image,
        from_bytes,
        
        # Image operations
        Image,
    )
except ImportError:
    # Fallback to importing from the image module
    try:
        from ...pyroid.image import (
            # Image creation
            create_image,
            from_bytes,
            
            # Image operations
            Image,
        )
    except ImportError:
        # If all else fails, create dummy classes and functions for documentation purposes
        class Image:
            """Image class for image processing operations (not available)."""
            def __init__(self, width, height, channels):
                self.width = width
                self.height = height
                self.channels = channels
                self.data = bytearray(width * height * channels)
                
            def __repr__(self):
                return f"Image({self.width}, {self.height}, {self.channels})"
                
            def set_pixel(self, x, y, color):
                """Set a pixel color."""
                if 0 <= x < self.width and 0 <= y < self.height:
                    idx = (y * self.width + x) * self.channels
                    for i, c in enumerate(color[:self.channels]):
                        self.data[idx + i] = c
                
            def get_pixel(self, x, y):
                """Get a pixel color."""
                if 0 <= x < self.width and 0 <= y < self.height:
                    idx = (y * self.width + x) * self.channels
                    return list(self.data[idx:idx + self.channels])
                return [0] * self.channels
                
            def to_grayscale(self):
                """Convert to grayscale."""
                if self.channels == 1:
                    return self
                result = Image(self.width, self.height, 1)
                for y in range(self.height):
                    for x in range(self.width):
                        color = self.get_pixel(x, y)
                        gray = int(0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
                        result.set_pixel(x, y, [gray])
                return result
                
            def resize(self, width, height):
                """Resize the image."""
                result = Image(width, height, self.channels)
                x_ratio = self.width / width
                y_ratio = self.height / height
                for y in range(height):
                    for x in range(width):
                        src_x = int(x * x_ratio)
                        src_y = int(y * y_ratio)
                        result.set_pixel(x, y, self.get_pixel(src_x, src_y))
                return result
                
            def blur(self, radius):
                """Apply a blur filter."""
                result = Image(self.width, self.height, self.channels)
                for y in range(self.height):
                    for x in range(self.width):
                        r = min(radius, min(x, self.width - x - 1, y, self.height - y - 1))
                        count = 0
                        color_sum = [0] * self.channels
                        for dy in range(-r, r + 1):
                            for dx in range(-r, r + 1):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < self.width and 0 <= ny < self.height:
                                    pixel = self.get_pixel(nx, ny)
                                    for i in range(self.channels):
                                        color_sum[i] += pixel[i]
                                    count += 1
                        avg_color = [int(c / count) for c in color_sum]
                        result.set_pixel(x, y, avg_color)
                return result
                
            def adjust_brightness(self, factor):
                """Adjust the brightness."""
                result = Image(self.width, self.height, self.channels)
                for y in range(self.height):
                    for x in range(self.width):
                        color = self.get_pixel(x, y)
                        new_color = [min(255, int(c * factor)) for c in color]
                        result.set_pixel(x, y, new_color)
                return result
                
        def create_image(width, height, channels):
            """Create a new image."""
            return Image(width, height, channels)
            
        def from_bytes(data, width, height, channels):
            """Create an image from raw bytes."""
            img = Image(width, height, channels)
            img.data = bytearray(data[:width * height * channels])
            return img

# Export the Image class and functions
__all__ = [
    'Image',
    'create_image',
    'from_bytes',
]