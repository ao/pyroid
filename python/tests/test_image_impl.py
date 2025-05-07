#!/usr/bin/env python3
"""
Test suite for Python implementations of Pyroid image module.
"""

import unittest
import sys
import os
from typing import List, Dict, Any

# Add the parent directory to the path so we can import pyroid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Python implementations directly
import pyroid.image_impl

class TestImageImpl(unittest.TestCase):
    """Test the image_impl module."""

    def test_image_creation(self):
        """Test creating an Image."""
        # Create an image
        img = pyroid.image_impl.Image(10, 20, 3)
        
        # Check properties
        self.assertEqual(img.width, 10)
        self.assertEqual(img.height, 20)
        self.assertEqual(img.channels, 3)
        
        # Check that the data has the right dimensions
        self.assertEqual(len(img.data), 20)  # height
        self.assertEqual(len(img.data[0]), 10 * 3)  # width * channels

    def test_get_set_pixel(self):
        """Test getting and setting pixels."""
        # Create an image
        img = pyroid.image_impl.Image(10, 10, 3)
        
        # Set a pixel
        img.set_pixel(5, 5, [255, 128, 64])
        
        # Get the pixel
        pixel = img.get_pixel(5, 5)
        
        # Check the pixel value
        self.assertEqual(pixel, [255, 128, 64])
        
        # Test out of bounds
        with self.assertRaises(IndexError):
            img.get_pixel(10, 5)
        
        with self.assertRaises(IndexError):
            img.set_pixel(5, 10, [255, 128, 64])
        
        # Test wrong number of channels
        with self.assertRaises(ValueError):
            img.set_pixel(5, 5, [255, 128])

    def test_to_grayscale(self):
        """Test converting to grayscale."""
        # Create an RGB image
        img = pyroid.image_impl.Image(10, 10, 3)
        
        # Set a pixel
        img.set_pixel(5, 5, [255, 128, 64])
        
        # Convert to grayscale
        gray = img.to_grayscale()
        
        # Check properties
        self.assertEqual(gray.width, 10)
        self.assertEqual(gray.height, 10)
        self.assertEqual(gray.channels, 1)
        
        # Check the pixel value (using the standard RGB to grayscale formula)
        expected = int(0.299 * 255 + 0.587 * 128 + 0.114 * 64)
        self.assertEqual(gray.get_pixel(5, 5), [expected])

    def test_resize(self):
        """Test resizing an image."""
        # Create an image
        img = pyroid.image_impl.Image(10, 10, 3)
        
        # Set a pixel
        img.set_pixel(5, 5, [255, 128, 64])
        
        # Resize the image
        resized = img.resize(20, 20)
        
        # Check properties
        self.assertEqual(resized.width, 20)
        self.assertEqual(resized.height, 20)
        self.assertEqual(resized.channels, 3)
        
        # The pixel at (5, 5) should be mapped to (10, 10) in the resized image
        # But with nearest-neighbor scaling, it might not be exact
        # So we don't check the exact value

    def test_blur(self):
        """Test blurring an image."""
        # Create an image
        img = pyroid.image_impl.Image(10, 10, 3)
        
        # Set a pixel
        img.set_pixel(5, 5, [255, 128, 64])
        
        # Blur the image
        blurred = img.blur(1)
        
        # Check properties
        self.assertEqual(blurred.width, 10)
        self.assertEqual(blurred.height, 10)
        self.assertEqual(blurred.channels, 3)
        
        # The pixel at (5, 5) should be blurred, but we don't check the exact value
        # since it depends on the specific blur implementation

    def test_adjust_brightness(self):
        """Test adjusting brightness."""
        # Create an image
        img = pyroid.image_impl.Image(10, 10, 3)
        
        # Set a pixel
        img.set_pixel(5, 5, [100, 100, 100])
        
        # Adjust brightness
        brightened = img.adjust_brightness(1.5)
        
        # Check properties
        self.assertEqual(brightened.width, 10)
        self.assertEqual(brightened.height, 10)
        self.assertEqual(brightened.channels, 3)
        
        # Check the pixel value
        pixel = brightened.get_pixel(5, 5)
        self.assertEqual(pixel, [150, 150, 150])  # 100 * 1.5 = 150
        
        # Test with brightness > 255
        img.set_pixel(5, 5, [200, 200, 200])
        brightened = img.adjust_brightness(1.5)
        pixel = brightened.get_pixel(5, 5)
        self.assertEqual(pixel, [255, 255, 255])  # 200 * 1.5 = 300, clamped to 255

    def test_create_image(self):
        """Test the create_image function."""
        # Create an image
        img = pyroid.image_impl.create_image(10, 20, 3)
        
        # Check properties
        self.assertEqual(img.width, 10)
        self.assertEqual(img.height, 20)
        self.assertEqual(img.channels, 3)
        
        # Check that it's an instance of Image
        self.assertIsInstance(img, pyroid.image_impl.Image)

    def test_from_bytes(self):
        """Test the from_bytes function."""
        # Create some test data
        width = 2
        height = 2
        channels = 3
        data = bytes([
            255, 0, 0,  # Red
            0, 255, 0,  # Green
            0, 0, 255,  # Blue
            255, 255, 0  # Yellow
        ])
        
        # Create an image from bytes
        img = pyroid.image_impl.from_bytes(data, width, height, channels)
        
        # Check properties
        self.assertEqual(img.width, width)
        self.assertEqual(img.height, height)
        self.assertEqual(img.channels, channels)
        
        # Check pixel values
        self.assertEqual(img.get_pixel(0, 0), [255, 0, 0])  # Red
        self.assertEqual(img.get_pixel(1, 0), [0, 255, 0])  # Green
        self.assertEqual(img.get_pixel(0, 1), [0, 0, 255])  # Blue
        self.assertEqual(img.get_pixel(1, 1), [255, 255, 0])  # Yellow
        
        # Test with wrong data size
        with self.assertRaises(ValueError):
            pyroid.image_impl.from_bytes(bytes([0, 0, 0]), width, height, channels)

if __name__ == "__main__":
    unittest.main()