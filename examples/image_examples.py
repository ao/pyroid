#!/usr/bin/env python3
"""
Image processing operation examples for pyroid.

This script demonstrates the image processing capabilities of pyroid.
"""

import time
import os
import io
import numpy as np
from PIL import Image, ImageFilter
import pyroid

def benchmark(func, *args, **kwargs):
    """Simple benchmarking function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")
    return result

def main():
    print("pyroid Image Processing Operations Examples")
    print("=========================================")
    
    # Example 1: Creating and manipulating images
    print("\n1. Creating and Manipulating Images")
    
    # Create a new image
    print("\nCreating a new 100x100 RGB image:")
    img = benchmark(lambda: pyroid.image.basic.create_image(100, 100, 3))
    
    # Set pixels to create a pattern
    print("\nSetting pixels to create a pattern:")
    def set_pixels(img):
        # Red square in top-left quadrant
        for x in range(50):
            for y in range(50):
                img.set_pixel(x, y, [255, 0, 0])
        
        # Green square in top-right quadrant
        for x in range(50, 100):
            for y in range(50):
                img.set_pixel(x, y, [0, 255, 0])
        
        # Blue square in bottom-left quadrant
        for x in range(50):
            for y in range(50, 100):
                img.set_pixel(x, y, [0, 0, 255])
        
        # Yellow square in bottom-right quadrant
        for x in range(50, 100):
            for y in range(50, 100):
                img.set_pixel(x, y, [255, 255, 0])
        
        return img
    
    img = benchmark(lambda: set_pixels(img))
    
    # Example 2: Image transformations
    print("\n2. Image Transformations")
    
    # Convert to grayscale
    print("\nConverting to grayscale:")
    grayscale_img = benchmark(lambda: img.to_grayscale())
    
    # Resize the image
    print("\nResizing to 200x200:")
    resized_img = benchmark(lambda: img.resize(200, 200))
    
    # Apply blur
    print("\nApplying blur with radius 2:")
    blurred_img = benchmark(lambda: img.blur(2))
    
    # Adjust brightness
    print("\nAdjusting brightness (1.5x):")
    brightened_img = benchmark(lambda: img.adjust_brightness(1.5))
    
    # Example 3: Creating an image from raw bytes
    print("\n3. Creating an Image from Raw Bytes")
    
    # Create raw data (all red pixels for a 10x10 RGB image)
    print("\nCreating a 10x10 red image from raw bytes:")
    raw_data = bytes([255, 0, 0] * (10 * 10))
    red_img = benchmark(lambda: pyroid.image.basic.from_bytes(raw_data, 10, 10, 3))
    
    # Example 4: Comparing with PIL
    print("\n4. Comparing with PIL")
    
    # Create a gradient image with PIL
    print("\nCreating a gradient image with PIL:")
    def create_pil_gradient(width, height):
        img = Image.new('RGB', (width, height))
        pixels = img.load()
        
        for x in range(width):
            for y in range(height):
                r = int(255 * x / width)
                g = int(255 * y / height)
                b = int(255 * ((x + y) / (width + height)))
                pixels[x, y] = (r, g, b)
        
        return img
    
    pil_img = benchmark(lambda: create_pil_gradient(100, 100))
    
    # Create a gradient image with pyroid
    print("\nCreating a gradient image with pyroid:")
    def create_pyroid_gradient(width, height):
        img = pyroid.image.basic.create_image(width, height, 3)
        
        for x in range(width):
            for y in range(height):
                r = int(255 * x / width)
                g = int(255 * y / height)
                b = int(255 * ((x + y) / (width + height)))
                img.set_pixel(x, y, [r, g, b])
        
        return img
    
    pyroid_img = benchmark(lambda: create_pyroid_gradient(100, 100))
    
    # Save images for inspection
    os.makedirs("output_images", exist_ok=True)
    
    # Convert pyroid images to PIL for saving
    def pyroid_to_pil(img):
        width = img.width
        height = img.height
        channels = img.channels
        data = img.data
        
        if channels == 1:
            # Grayscale
            pil_img = Image.new('L', (width, height))
            for y in range(height):
                for x in range(width):
                    idx = (y * width + x) * channels
                    pil_img.putpixel((x, y), data[idx])
        else:
            # RGB or RGBA
            mode = 'RGB' if channels == 3 else 'RGBA'
            pil_img = Image.new(mode, (width, height))
            for y in range(height):
                for x in range(width):
                    idx = (y * width + x) * channels
                    pixel = tuple(data[idx:idx+channels])
                    pil_img.putpixel((x, y), pixel)
        
        return pil_img
    
    # Save original image
    pil_from_pyroid = pyroid_to_pil(img)
    pil_from_pyroid.save("output_images/original.png")
    
    # Save grayscale image
    pil_from_grayscale = pyroid_to_pil(grayscale_img)
    pil_from_grayscale.save("output_images/grayscale.png")
    
    # Save resized image
    pil_from_resized = pyroid_to_pil(resized_img)
    pil_from_resized.save("output_images/resized.png")
    
    # Save blurred image
    pil_from_blurred = pyroid_to_pil(blurred_img)
    pil_from_blurred.save("output_images/blurred.png")
    
    # Save brightened image
    pil_from_brightened = pyroid_to_pil(brightened_img)
    pil_from_brightened.save("output_images/brightened.png")
    
    # Save gradient image
    pil_from_gradient = pyroid_to_pil(pyroid_img)
    pil_from_gradient.save("output_images/gradient.png")
    
    print("\nSample images saved to 'output_images' directory for inspection")

if __name__ == "__main__":
    main()