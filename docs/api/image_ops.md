# Image Processing Operations

The Image Processing operations module provides basic image manipulation capabilities implemented in pure Rust without external dependencies. These operations are designed to be simple, reliable, and easy to use.

## Overview

The Image Processing operations module provides the following key functions:

- `create_image`: Create a new image with specified dimensions
- `from_bytes`: Create an image from raw bytes
- Image manipulation methods: `to_grayscale`, `resize`, `blur`, `adjust_brightness`

## API Reference

### Image Class

The `Image` class represents an image with pixel-level access.

```python
pyroid.image.basic.Image(width, height, channels)
```

#### Parameters

- `width`: Width of the image in pixels
- `height`: Height of the image in pixels
- `channels`: Number of color channels (1 for grayscale, 3 for RGB, 4 for RGBA)

#### Properties

- `width`: Width of the image in pixels
- `height`: Height of the image in pixels
- `channels`: Number of color channels
- `data`: Raw image data as bytes

#### Methods

##### get_pixel

Get the color value of a pixel.

```python
image.get_pixel(x, y)
```

**Parameters**
- `x`: X-coordinate of the pixel
- `y`: Y-coordinate of the pixel

**Returns**
A list of color channel values (e.g., [R, G, B] for RGB images).

##### set_pixel

Set the color value of a pixel.

```python
image.set_pixel(x, y, pixel)
```

**Parameters**
- `x`: X-coordinate of the pixel
- `y`: Y-coordinate of the pixel
- `pixel`: A list of color channel values (e.g., [R, G, B] for RGB images)

##### to_grayscale

Convert the image to grayscale.

```python
image.to_grayscale()
```

**Returns**
A new grayscale image.

##### resize

Resize the image using nearest neighbor interpolation.

```python
image.resize(new_width, new_height)
```

**Parameters**
- `new_width`: New width in pixels
- `new_height`: New height in pixels

**Returns**
A new resized image.

##### blur

Apply a simple blur filter to the image.

```python
image.blur(radius)
```

**Parameters**
- `radius`: Blur radius (larger values create more blur)

**Returns**
A new blurred image.

##### adjust_brightness

Adjust the brightness of the image.

```python
image.adjust_brightness(factor)
```

**Parameters**
- `factor`: Brightness factor (1.0 is original, >1.0 is brighter, <1.0 is darker)

**Returns**
A new image with adjusted brightness.

### create_image

Create a new image with the specified dimensions.

```python
pyroid.image.basic.create_image(width, height, channels)
```

#### Parameters

- `width`: Width of the image in pixels
- `height`: Height of the image in pixels
- `channels`: Number of color channels (1 for grayscale, 3 for RGB, 4 for RGBA)

#### Returns

A new `Image` object.

#### Example

```python
import pyroid

# Create a new RGB image (100x100 pixels)
img = pyroid.image.basic.create_image(100, 100, 3)

# Set some pixels
for x in range(50):
    for y in range(50):
        img.set_pixel(x, y, [255, 0, 0])  # Red square

for x in range(50, 100):
    for y in range(50, 100):
        img.set_pixel(x, y, [0, 0, 255])  # Blue square

# Apply operations
grayscale_img = img.to_grayscale()
resized_img = img.resize(200, 200)
blurred_img = img.blur(2)
brightened_img = img.adjust_brightness(1.5)
```

### from_bytes

Create a new image from raw bytes.

```python
pyroid.image.basic.from_bytes(data, width, height, channels)
```

#### Parameters

- `data`: Raw image data as bytes
- `width`: Width of the image in pixels
- `height`: Height of the image in pixels
- `channels`: Number of color channels (1 for grayscale, 3 for RGB, 4 for RGBA)

#### Returns

A new `Image` object.

#### Example

```python
import pyroid

# Create raw image data (all red pixels for a 2x2 RGB image)
data = bytes([255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0])
img = pyroid.image.basic.from_bytes(data, 2, 2, 3)

# Get a pixel value
pixel = img.get_pixel(0, 0)
print(f"Pixel at (0,0): {pixel}")  # Should be [255, 0, 0]
```

## Performance Considerations

- The current implementation is focused on simplicity and reliability rather than maximum performance.
- Operations are performed in pure Rust without external dependencies, which provides good baseline performance.
- For very large images or when processing many images, memory usage can be a concern.
- The implementation does not currently support parallel processing of multiple images.

## Best Practices

1. **Optimize image dimensions**: Resize images to the dimensions you need before applying filters or other operations to improve performance.

2. **Use appropriate color channels**: Use grayscale (1 channel) when color is not needed to reduce memory usage and improve performance.

3. **Consider memory usage**: When processing very large images, be mindful of memory usage.

## Limitations

1. **Limited format support**: The current implementation does not include loading or saving images in specific formats like JPEG or PNG.

2. **Basic operations only**: The current implementation provides only basic image operations and does not include advanced features like color correction, histogram equalization, or perspective transformation.

3. **No parallel processing**: The current implementation does not support parallel processing of multiple images.

4. **Simple interpolation**: The resize operation uses nearest neighbor interpolation, which is fast but may result in lower quality for some images.

## Examples

### Example 1: Creating a Gradient Image

```python
import pyroid

# Create a new RGB image
width, height = 256, 256
img = pyroid.image.basic.create_image(width, height, 3)

# Create a gradient
for y in range(height):
    for x in range(width):
        r = int(x / width * 255)
        g = int(y / height * 255)
        b = int((x + y) / (width + height) * 255)
        img.set_pixel(x, y, [r, g, b])

# Convert to grayscale
gray_img = img.to_grayscale()

# Save the image data for external use
with open("gradient.raw", "wb") as f:
    f.write(img.data)
with open("gradient_gray.raw", "wb") as f:
    f.write(gray_img.data)
```

### Example 2: Image Processing Pipeline

```python
import pyroid

# Create a new image
img = pyroid.image.basic.create_image(100, 100, 3)

# Fill with a pattern
for y in range(100):
    for x in range(100):
        if (x // 10 + y // 10) % 2 == 0:
            img.set_pixel(x, y, [255, 255, 255])  # White
        else:
            img.set_pixel(x, y, [0, 0, 0])  # Black

# Apply a series of transformations
img = img.resize(200, 200)
img = img.blur(1)
img = img.adjust_brightness(1.2)
img = img.to_grayscale()

# Print image properties
print(f"Image dimensions: {img.width}x{img.height}")
print(f"Channels: {img.channels}")
print(f"Data size: {len(img.data)} bytes")