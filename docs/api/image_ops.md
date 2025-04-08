# Image Operations

This document provides detailed information about the image operations available in Pyroid.

## Image Creation

### Create Image

```python
pyroid.image.basic.create_image(width, height, channels)
```

Creates a new image with the specified dimensions and number of channels.

**Parameters:**
- `width` (int): The width of the image in pixels.
- `height` (int): The height of the image in pixels.
- `channels` (int): The number of channels (1 for grayscale, 3 for RGB, 4 for RGBA).

**Returns:**
- A new Image object.

**Example:**
```python
import pyroid

# Create a new RGB image (100x100)
img = pyroid.image.basic.create_image(100, 100, 3)
```

### From Bytes

```python
pyroid.image.basic.from_bytes(data, width, height, channels)
```

Creates a new image from raw byte data.

**Parameters:**
- `data` (bytes): The raw image data.
- `width` (int): The width of the image in pixels.
- `height` (int): The height of the image in pixels.
- `channels` (int): The number of channels (1 for grayscale, 3 for RGB, 4 for RGBA).

**Returns:**
- A new Image object.

**Example:**
```python
import pyroid

# Create a new image from raw bytes (all red pixels for a 10x10 RGB image)
raw_data = bytes([255, 0, 0] * (10 * 10))
img = pyroid.image.basic.from_bytes(raw_data, 10, 10, 3)
```

## Image Properties

### Width

```python
img.width
```

Gets the width of the image in pixels.

**Returns:**
- The width as an integer.

**Example:**
```python
import pyroid

img = pyroid.image.basic.create_image(100, 100, 3)
print(f"Image width: {img.width}")
```

### Height

```python
img.height
```

Gets the height of the image in pixels.

**Returns:**
- The height as an integer.

**Example:**
```python
import pyroid

img = pyroid.image.basic.create_image(100, 100, 3)
print(f"Image height: {img.height}")
```

### Channels

```python
img.channels
```

Gets the number of channels in the image.

**Returns:**
- The number of channels as an integer (1 for grayscale, 3 for RGB, 4 for RGBA).

**Example:**
```python
import pyroid

img = pyroid.image.basic.create_image(100, 100, 3)
print(f"Image channels: {img.channels}")
```

### Data

```python
img.data
```

Gets the raw image data as bytes.

**Returns:**
- The image data as bytes.

**Example:**
```python
import pyroid

img = pyroid.image.basic.create_image(100, 100, 3)
data = img.data
print(f"Image data length: {len(data)}")
```

## Pixel Operations

### Set Pixel

```python
img.set_pixel(x, y, color)
```

Sets the color of a pixel at the specified coordinates.

**Parameters:**
- `x` (int): The x-coordinate of the pixel.
- `y` (int): The y-coordinate of the pixel.
- `color` (list): The color values as a list (e.g., [255, 0, 0] for red in RGB).

**Returns:**
- None

**Example:**
```python
import pyroid

img = pyroid.image.basic.create_image(100, 100, 3)
# Set pixel at (10, 10) to red
img.set_pixel(10, 10, [255, 0, 0])
```

### Get Pixel

```python
img.get_pixel(x, y)
```

Gets the color of a pixel at the specified coordinates.

**Parameters:**
- `x` (int): The x-coordinate of the pixel.
- `y` (int): The y-coordinate of the pixel.

**Returns:**
- The color values as a list.

**Example:**
```python
import pyroid

img = pyroid.image.basic.create_image(100, 100, 3)
img.set_pixel(10, 10, [255, 0, 0])
color = img.get_pixel(10, 10)
print(f"Pixel color: {color}")
```

## Image Transformations

### To Grayscale

```python
img.to_grayscale()
```

Converts the image to grayscale.

**Returns:**
- A new Image object in grayscale.

**Example:**
```python
import pyroid

img = pyroid.image.basic.create_image(100, 100, 3)
# Set some pixels
for x in range(50):
    for y in range(50):
        img.set_pixel(x, y, [255, 0, 0])  # Red square

grayscale_img = img.to_grayscale()
print(f"Grayscale image channels: {grayscale_img.channels}")
```

### Resize

```python
img.resize(width, height)
```

Resizes the image to the specified dimensions.

**Parameters:**
- `width` (int): The new width of the image in pixels.
- `height` (int): The new height of the image in pixels.

**Returns:**
- A new Image object with the specified dimensions.

**Example:**
```python
import pyroid

img = pyroid.image.basic.create_image(100, 100, 3)
resized_img = img.resize(200, 200)
print(f"Resized image dimensions: {resized_img.width}x{resized_img.height}")
```

### Blur

```python
img.blur(radius)
```

Applies a Gaussian blur to the image.

**Parameters:**
- `radius` (float): The blur radius.

**Returns:**
- A new Image object with the blur applied.

**Example:**
```python
import pyroid

img = pyroid.image.basic.create_image(100, 100, 3)
# Set some pixels
for x in range(50):
    for y in range(50):
        img.set_pixel(x, y, [255, 0, 0])  # Red square

blurred_img = img.blur(2.0)
```

### Adjust Brightness

```python
img.adjust_brightness(factor)
```

Adjusts the brightness of the image.

**Parameters:**
- `factor` (float): The brightness adjustment factor. Values greater than 1.0 increase brightness, values less than 1.0 decrease brightness.

**Returns:**
- A new Image object with adjusted brightness.

**Example:**
```python
import pyroid

img = pyroid.image.basic.create_image(100, 100, 3)
# Set some pixels
for x in range(50):
    for y in range(50):
        img.set_pixel(x, y, [255, 0, 0])  # Red square

brightened_img = img.adjust_brightness(1.5)
darkened_img = img.adjust_brightness(0.5)
```

### Crop

```python
img.crop(x, y, width, height)
```

Crops the image to the specified rectangle.

**Parameters:**
- `x` (int): The x-coordinate of the top-left corner of the crop rectangle.
- `y` (int): The y-coordinate of the top-left corner of the crop rectangle.
- `width` (int): The width of the crop rectangle.
- `height` (int): The height of the crop rectangle.

**Returns:**
- A new Image object containing the cropped region.

**Example:**
```python
import pyroid

img = pyroid.image.basic.create_image(100, 100, 3)
# Set some pixels
for x in range(50):
    for y in range(50):
        img.set_pixel(x, y, [255, 0, 0])  # Red square

# Crop the red square
cropped_img = img.crop(0, 0, 50, 50)
print(f"Cropped image dimensions: {cropped_img.width}x{cropped_img.height}")
```

### Rotate

```python
img.rotate(angle)
```

Rotates the image by the specified angle in degrees.

**Parameters:**
- `angle` (float): The rotation angle in degrees.

**Returns:**
- A new Image object with the rotation applied.

**Example:**
```python
import pyroid

img = pyroid.image.basic.create_image(100, 100, 3)
# Set some pixels
for x in range(50):
    for y in range(50):
        img.set_pixel(x, y, [255, 0, 0])  # Red square

rotated_img = img.rotate(45.0)
```

### Flip

```python
img.flip_horizontal()
img.flip_vertical()
```

Flips the image horizontally or vertically.

**Returns:**
- A new Image object with the flip applied.

**Example:**
```python
import pyroid

img = pyroid.image.basic.create_image(100, 100, 3)
# Set some pixels
for x in range(50):
    for y in range(50):
        img.set_pixel(x, y, [255, 0, 0])  # Red square

flipped_h = img.flip_horizontal()
flipped_v = img.flip_vertical()
```

## Image Conversion

### To Bytes

```python
img.to_bytes()
```

Converts the image to raw byte data.

**Returns:**
- The image data as bytes.

**Example:**
```python
import pyroid

img = pyroid.image.basic.create_image(100, 100, 3)
data = img.to_bytes()
print(f"Image data length: {len(data)}")
```

### To Base64

```python
img.to_base64(format="png")
```

Converts the image to a base64-encoded string.

**Parameters:**
- `format` (str, optional): The image format. One of "png", "jpeg", "bmp". Default is "png".

**Returns:**
- The base64-encoded image as a string.

**Example:**
```python
import pyroid

img = pyroid.image.basic.create_image(100, 100, 3)
base64_str = img.to_base64(format="png")
print(f"Base64 string length: {len(base64_str)}")
```

## Image I/O

### Save

```python
img.save(path, format=None)
```

Saves the image to a file.

**Parameters:**
- `path` (str): The path to save the image to.
- `format` (str, optional): The image format. If None, it will be inferred from the file extension.

**Returns:**
- None

**Example:**
```python
import pyroid

img = pyroid.image.basic.create_image(100, 100, 3)
# Set some pixels
for x in range(50):
    for y in range(50):
        img.set_pixel(x, y, [255, 0, 0])  # Red square

img.save("red_square.png")
```

### Load

```python
pyroid.image.basic.load(path)
```

Loads an image from a file.

**Parameters:**
- `path` (str): The path to the image file.

**Returns:**
- A new Image object.

**Example:**
```python
import pyroid

img = pyroid.image.basic.load("image.png")
print(f"Loaded image dimensions: {img.width}x{img.height}")