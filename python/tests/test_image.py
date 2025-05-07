#!/usr/bin/env python3
"""
Integration tests for Pyroid image functionality.

This script contains tests to verify that the image Rust extensions are working correctly.
"""

import unittest
import pyroid

class TestImage(unittest.TestCase):
    """Test the Image class and related functionality."""
    
    def test_image_creation(self):
        """Test creating an Image object."""
        # Create a small 2x2 RGB image
        img = pyroid.image.create_image(2, 2, 3)
        self.assertIsNotNone(img)
        self.assertEqual(img.width, 2)
        self.assertEqual(img.height, 2)
        self.assertEqual(img.channels, 3)
    
    def test_from_bytes(self):
        """Test creating an Image from bytes."""
        # Create a 2x2 RGB image from bytes
        # Each pixel has 3 bytes (R, G, B)
        data = bytes([
            255, 0, 0,    # Red
            0, 255, 0,    # Green
            0, 0, 255,    # Blue
            255, 255, 0   # Yellow
        ])
        img = pyroid.image.from_bytes(data, 2, 2, 3)
        self.assertIsNotNone(img)
        self.assertEqual(img.width, 2)
        self.assertEqual(img.height, 2)
        self.assertEqual(img.channels, 3)
        
        # Check pixel values
        self.assertEqual(img.get_pixel(0, 0), [255, 0, 0])      # Red
        self.assertEqual(img.get_pixel(1, 0), [0, 255, 0])      # Green
        self.assertEqual(img.get_pixel(0, 1), [0, 0, 255])      # Blue
        self.assertEqual(img.get_pixel(1, 1), [255, 255, 0])    # Yellow
    
    def test_set_get_pixel(self):
        """Test setting and getting pixel values."""
        img = pyroid.image.create_image(2, 2, 3)
        
        # Set pixel values
        img.set_pixel(0, 0, [255, 0, 0])      # Red
        img.set_pixel(1, 0, [0, 255, 0])      # Green
        img.set_pixel(0, 1, [0, 0, 255])      # Blue
        img.set_pixel(1, 1, [255, 255, 0])    # Yellow
        
        # Check pixel values
        self.assertEqual(img.get_pixel(0, 0), [255, 0, 0])      # Red
        self.assertEqual(img.get_pixel(1, 0), [0, 255, 0])      # Green
        self.assertEqual(img.get_pixel(0, 1), [0, 0, 255])      # Blue
        self.assertEqual(img.get_pixel(1, 1), [255, 255, 0])    # Yellow
    
    def test_to_grayscale(self):
        """Test converting an image to grayscale."""
        img = pyroid.image.create_image(2, 2, 3)
        
        # Set pixel values
        img.set_pixel(0, 0, [255, 0, 0])      # Red
        img.set_pixel(1, 0, [0, 255, 0])      # Green
        img.set_pixel(0, 1, [0, 0, 255])      # Blue
        img.set_pixel(1, 1, [255, 255, 255])  # White
        
        # Convert to grayscale
        gray = img.to_grayscale()
        
        # Check dimensions
        self.assertEqual(gray.width, 2)
        self.assertEqual(gray.height, 2)
        self.assertEqual(gray.channels, 1)
        
        # Check pixel values (using standard RGB to grayscale conversion)
        # Red: 0.299 * 255 = 76.245
        # Green: 0.587 * 255 = 149.685
        # Blue: 0.114 * 255 = 29.07
        # White: 0.299 * 255 + 0.587 * 255 + 0.114 * 255 = 255
        self.assertAlmostEqual(gray.get_pixel(0, 0)[0], 76, delta=1)    # Red
        self.assertAlmostEqual(gray.get_pixel(1, 0)[0], 150, delta=1)   # Green
        self.assertAlmostEqual(gray.get_pixel(0, 1)[0], 29, delta=1)    # Blue
        self.assertAlmostEqual(gray.get_pixel(1, 1)[0], 255, delta=1)   # White
    
    def test_resize(self):
        """Test resizing an image."""
        img = pyroid.image.create_image(2, 2, 3)
        
        # Set pixel values
        img.set_pixel(0, 0, [255, 0, 0])      # Red
        img.set_pixel(1, 0, [0, 255, 0])      # Green
        img.set_pixel(0, 1, [0, 0, 255])      # Blue
        img.set_pixel(1, 1, [255, 255, 0])    # Yellow
        
        # Resize to 4x4
        resized = img.resize(4, 4)
        
        # Check dimensions
        self.assertEqual(resized.width, 4)
        self.assertEqual(resized.height, 4)
        self.assertEqual(resized.channels, 3)
        
        # Check corner pixels (should match original)
        self.assertEqual(resized.get_pixel(0, 0), [255, 0, 0])      # Red
        self.assertEqual(resized.get_pixel(3, 0), [0, 255, 0])      # Green
        self.assertEqual(resized.get_pixel(0, 3), [0, 0, 255])      # Blue
        self.assertEqual(resized.get_pixel(3, 3), [255, 255, 0])    # Yellow
    
    def test_blur(self):
        """Test applying a blur filter."""
        img = pyroid.image.create_image(3, 3, 3)
        
        # Set all pixels to white except the center
        for y in range(3):
            for x in range(3):
                if x == 1 and y == 1:
                    img.set_pixel(x, y, [0, 0, 0])  # Black center
                else:
                    img.set_pixel(x, y, [255, 255, 255])  # White surroundings
        
        # Apply blur with radius 1
        blurred = img.blur(1)
        
        # Check dimensions
        self.assertEqual(blurred.width, 3)
        self.assertEqual(blurred.height, 3)
        self.assertEqual(blurred.channels, 3)
        
        # Center pixel should no longer be black
        center = blurred.get_pixel(1, 1)
        self.assertGreater(center[0], 0)
        self.assertGreater(center[1], 0)
        self.assertGreater(center[2], 0)
    
    def test_adjust_brightness(self):
        """Test adjusting brightness."""
        img = pyroid.image.create_image(2, 2, 3)
        
        # Set pixel values
        img.set_pixel(0, 0, [100, 100, 100])  # Gray
        img.set_pixel(1, 0, [100, 100, 100])  # Gray
        img.set_pixel(0, 1, [100, 100, 100])  # Gray
        img.set_pixel(1, 1, [100, 100, 100])  # Gray
        
        # Increase brightness by 50%
        brightened = img.adjust_brightness(1.5)
        
        # Check dimensions
        self.assertEqual(brightened.width, 2)
        self.assertEqual(brightened.height, 2)
        self.assertEqual(brightened.channels, 3)
        
        # Check pixel values
        for y in range(2):
            for x in range(2):
                pixel = brightened.get_pixel(x, y)
                self.assertEqual(pixel, [150, 150, 150])  # 100 * 1.5 = 150

if __name__ == "__main__":
    unittest.main()