#!/usr/bin/env python3
"""
Image processing operation examples for pyroid.

This script demonstrates the image processing capabilities of pyroid.
"""

import time
import os
import pyroid

def benchmark(func, *args, **kwargs):
    """Simple benchmarking function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")
    return result

def main():
    print("Pyroid Image Processing Examples")
    print("=============================")
    
    # Example 1: Creating and manipulating images
    print("\n1. Creating and Manipulating Images")
    
    # Create a new image
    print("\nCreating a new 100x100 RGB image:")
    img = benchmark(lambda: pyroid.image.basic.create_image(100, 100, 3))
    print(f"Image created: {img.width}x{img.height} with {img.channels} channels")
    
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
    
    # Get a pixel
    pixel = img.get_pixel(25, 25)
    print(f"Pixel at (25, 25): {pixel}")
    
    # Example 2: Image transformations
    print("\n2. Image Transformations")
    
    # Convert to grayscale
    print("\nConverting to grayscale:")
    grayscale_img = benchmark(lambda: img.to_grayscale())
    print(f"Grayscale image: {grayscale_img.width}x{grayscale_img.height} with {grayscale_img.channels} channels")
    
    # Resize the image
    print("\nResizing to 50x50:")
    resized_img = benchmark(lambda: img.resize(50, 50))
    print(f"Resized image: {resized_img.width}x{resized_img.height}")
    
    # Apply blur
    print("\nApplying blur with radius 2:")
    blurred_img = benchmark(lambda: img.blur(2))
    print(f"Blurred image: {blurred_img.width}x{blurred_img.height}")
    
    # Adjust brightness
    print("\nAdjusting brightness (1.5x):")
    brightened_img = benchmark(lambda: img.adjust_brightness(1.5))
    print(f"Brightened image: {brightened_img.width}x{brightened_img.height}")
    
    # Example 3: Creating an image from raw bytes
    print("\n3. Creating an Image from Raw Bytes")
    
    # Create raw data (all red pixels for a 10x10 RGB image)
    print("\nCreating a 10x10 red image from raw bytes:")
    raw_data = bytes([255, 0, 0] * (10 * 10))
    red_img = benchmark(lambda: pyroid.image.basic.from_bytes(raw_data, 10, 10, 3))
    print(f"Image created from bytes: {red_img.width}x{red_img.height} with {red_img.channels} channels")
    
    # Example 4: Saving and loading images
    print("\n4. Saving and Loading Images")
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Try to save the image
    try:
        print("\nSaving the image to output/test_image.png:")
        img.save("output/test_image.png")
        print("Image saved successfully")
    except Exception as e:
        print(f"Image save not implemented: {e}")
    
    # Try to load the image
    try:
        print("\nLoading the image from output/test_image.png:")
        loaded_img = pyroid.image.basic.load("output/test_image.png")
        print(f"Image loaded: {loaded_img.width}x{loaded_img.height} with {loaded_img.channels} channels")
    except Exception as e:
        print(f"Image load not implemented: {e}")
    
    # Example 5: Advanced transformations
    print("\n5. Advanced Transformations")
    
    # Try to crop the image
    try:
        print("\nCropping the image to the top-left quadrant:")
        cropped_img = img.crop(0, 0, 50, 50)
        print(f"Cropped image: {cropped_img.width}x{cropped_img.height}")
    except Exception as e:
        print(f"Image crop not implemented: {e}")
    
    # Try to rotate the image
    try:
        print("\nRotating the image by 45 degrees:")
        rotated_img = img.rotate(45)
        print(f"Rotated image: {rotated_img.width}x{rotated_img.height}")
    except Exception as e:
        print(f"Image rotation not implemented: {e}")
    
    # Try to flip the image
    try:
        print("\nFlipping the image horizontally:")
        flipped_h = img.flip_horizontal()
        print(f"Horizontally flipped image: {flipped_h.width}x{flipped_h.height}")
        
        print("\nFlipping the image vertically:")
        flipped_v = img.flip_vertical()
        print(f"Vertically flipped image: {flipped_v.width}x{flipped_v.height}")
    except Exception as e:
        print(f"Image flip not implemented: {e}")
    
    print("\nImage processing examples completed.")

if __name__ == "__main__":
    main()