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
import requests
from concurrent.futures import ThreadPoolExecutor
import pyroid

def benchmark(func, *args, **kwargs):
    """Simple benchmarking function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")
    return result

def download_sample_images(num_images):
    """Download sample images for benchmarking."""
    os.makedirs("test_images", exist_ok=True)
    
    # Use Unsplash API for random images
    image_urls = [
        f"https://source.unsplash.com/random/800x600?sig={i}" 
        for i in range(num_images)
    ]
    
    image_data = []
    
    def download_image(url):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.content
        except Exception as e:
            print(f"Error downloading image: {e}")
        return None
    
    print(f"Downloading {num_images} sample images...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        image_data = list(executor.map(download_image, image_urls))
    
    # Filter out any failed downloads
    image_data = [img for img in image_data if img is not None]
    
    print(f"Successfully downloaded {len(image_data)} images")
    return image_data

def create_sample_images(num_images, width=800, height=600):
    """Create sample images for benchmarking if download fails."""
    image_data = []
    
    for i in range(num_images):
        # Create a gradient image
        img = Image.new('RGB', (width, height))
        pixels = img.load()
        
        for x in range(width):
            for y in range(height):
                r = int(255 * x / width)
                g = int(255 * y / height)
                b = int(255 * ((x + y) / (width + height)))
                pixels[x, y] = (r, g, b)
        
        # Convert to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        image_data.append(buffer.getvalue())
    
    return image_data

def main():
    print("pyroid Image Processing Operations Examples")
    print("=========================================")
    
    # Get sample images
    num_images = 10
    try:
        image_data = download_sample_images(num_images)
        if not image_data:
            raise Exception("No images downloaded")
    except Exception as e:
        print(f"Failed to download images: {e}")
        print("Creating sample images instead...")
        image_data = create_sample_images(num_images)
    
    # Example 1: Image Resizing
    print("\n1. Image Resizing")
    
    print(f"\nResizing {len(image_data)} images to 400x300:")
    
    print("\nPIL resize:")
    def pil_resize(images, size):
        results = []
        for img_data in images:
            img = Image.open(io.BytesIO(img_data))
            resized = img.resize(size)
            buffer = io.BytesIO()
            resized.save(buffer, format='JPEG')
            results.append(buffer.getvalue())
        return results
    
    pil_result = benchmark(lambda: pil_resize(image_data, (400, 300)))
    
    print("\npyroid parallel_resize:")
    pyroid_result = benchmark(lambda: pyroid.parallel_resize(image_data, (400, 300), "lanczos3"))
    
    print("\nResults (sizes):")
    print(f"PIL: {len(pil_result)} images, first image size: {len(pil_result[0])} bytes")
    print(f"pyroid: {len(pyroid_result)} images, first image size: {len(pyroid_result[0])} bytes")
    
    # Example 2: Image Filtering
    print("\n2. Image Filtering")
    
    print(f"\nApplying blur filter to {len(image_data)} images:")
    
    print("\nPIL filter:")
    def pil_filter(images, filter_type):
        results = []
        for img_data in images:
            img = Image.open(io.BytesIO(img_data))
            if filter_type == "blur":
                filtered = img.filter(ImageFilter.GaussianBlur(radius=2))
            elif filter_type == "sharpen":
                filtered = img.filter(ImageFilter.SHARPEN)
            elif filter_type == "edge":
                filtered = img.filter(ImageFilter.FIND_EDGES)
            else:
                filtered = img
            buffer = io.BytesIO()
            filtered.save(buffer, format='JPEG')
            results.append(buffer.getvalue())
        return results
    
    pil_result = benchmark(lambda: pil_filter(image_data, "blur"))
    
    print("\npyroid parallel_image_filter:")
    params = {"sigma": 2.0}
    pyroid_result = benchmark(lambda: pyroid.parallel_image_filter(image_data, "blur", params))
    
    print("\nResults (sizes):")
    print(f"PIL: {len(pil_result)} images, first image size: {len(pil_result[0])} bytes")
    print(f"pyroid: {len(pyroid_result)} images, first image size: {len(pyroid_result[0])} bytes")
    
    # Example 3: Format Conversion
    print("\n3. Format Conversion")
    
    print(f"\nConverting {len(image_data)} images from JPEG to PNG:")
    
    print("\nPIL convert:")
    def pil_convert(images, format):
        results = []
        for img_data in images:
            img = Image.open(io.BytesIO(img_data))
            buffer = io.BytesIO()
            img.save(buffer, format=format)
            results.append(buffer.getvalue())
        return results
    
    pil_result = benchmark(lambda: pil_convert(image_data, "PNG"))
    
    print("\npyroid parallel_convert:")
    pyroid_result = benchmark(lambda: pyroid.parallel_convert(image_data, None, "png", 90))
    
    print("\nResults (sizes):")
    print(f"PIL: {len(pil_result)} images, first image size: {len(pil_result[0])} bytes")
    print(f"pyroid: {len(pyroid_result)} images, first image size: {len(pyroid_result[0])} bytes")
    
    # Example 4: Metadata Extraction
    print("\n4. Metadata Extraction")
    
    print(f"\nExtracting metadata from {len(image_data)} images:")
    
    print("\nPIL metadata extraction:")
    def pil_metadata(images):
        results = []
        for img_data in images:
            img = Image.open(io.BytesIO(img_data))
            metadata = {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode
            }
            results.append(metadata)
        return results
    
    pil_result = benchmark(lambda: pil_metadata(image_data))
    
    print("\npyroid parallel_extract_metadata:")
    pyroid_result = benchmark(lambda: pyroid.parallel_extract_metadata(image_data))
    
    print("\nResults (first image metadata):")
    print(f"PIL: {pil_result[0]}")
    print(f"pyroid: {pyroid_result[0]}")
    
    # Save a sample of processed images for visual inspection
    os.makedirs("output_images", exist_ok=True)
    
    # Save original
    with open("output_images/original.jpg", "wb") as f:
        f.write(image_data[0])
    
    # Save resized
    with open("output_images/resized_pyroid.jpg", "wb") as f:
        f.write(pyroid_result[0])
    
    print("\nSample images saved to 'output_images' directory for inspection")

if __name__ == "__main__":
    main()