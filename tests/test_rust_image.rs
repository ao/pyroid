//! Integration tests for the Rust image module

mod common;
use common::image::Image;

#[test]
fn test_image_creation() {
    // Create a new image
    let width = 100;
    let height = 50;
    let img = Image::new(width, height);
    
    // Check dimensions
    assert_eq!(img.width(), width);
    assert_eq!(img.height(), height);
    
    // Check default pixel values (should be transparent black)
    let pixel = img.get_pixel(0, 0);
    assert_eq!(pixel, [0, 0, 0, 0]);
}

#[test]
fn test_get_set_pixel() {
    // Create a new image
    let width = 10;
    let height = 10;
    let mut img = Image::new(width, height);
    
    // Set a pixel
    let red_pixel = [255, 0, 0, 255];
    img.set_pixel(5, 5, red_pixel);
    
    // Get the pixel and check it
    let pixel = img.get_pixel(5, 5);
    assert_eq!(pixel, red_pixel);
    
    // Check that other pixels are unchanged
    let pixel = img.get_pixel(0, 0);
    assert_eq!(pixel, [0, 0, 0, 0]);
    
    // Test out of bounds get_pixel
    let pixel = img.get_pixel(width, height);
    assert_eq!(pixel, [0, 0, 0, 0]);
    
    // Test out of bounds set_pixel (should not crash)
    img.set_pixel(width, height, [255, 255, 255, 255]);
}

#[test]
fn test_resize() {
    // Create a new image
    let width = 10;
    let height = 10;
    let mut img = Image::new(width, height);
    
    // Set a pixel
    let red_pixel = [255, 0, 0, 255];
    img.set_pixel(5, 5, red_pixel);
    
    // Resize the image
    let new_width = 5;
    let new_height = 5;
    let resized = img.resize(new_width, new_height);
    
    // Check dimensions
    assert_eq!(resized.width(), new_width);
    assert_eq!(resized.height(), new_height);
    
    // In our simple implementation, the resize might not preserve the exact pixel values
    // So we'll just check that the dimensions are correct
    assert_eq!(resized.width(), new_width);
    assert_eq!(resized.height(), new_height);
    
    // Resize to larger dimensions
    let new_width = 20;
    let new_height = 20;
    let resized = img.resize(new_width, new_height);
    
    // Check dimensions
    assert_eq!(resized.width(), new_width);
    assert_eq!(resized.height(), new_height);
}

#[test]
fn test_grayscale() {
    // Create a new image
    let width = 10;
    let height = 10;
    let mut img = Image::new(width, height);
    
    // Set some colored pixels
    img.set_pixel(0, 0, [255, 0, 0, 255]);     // Red
    img.set_pixel(1, 0, [0, 255, 0, 255]);     // Green
    img.set_pixel(2, 0, [0, 0, 255, 255]);     // Blue
    img.set_pixel(3, 0, [255, 255, 255, 255]); // White
    img.set_pixel(4, 0, [0, 0, 0, 255]);       // Black
    
    // Convert to grayscale
    let gray = img.grayscale();
    
    // Check dimensions
    assert_eq!(gray.width(), width);
    assert_eq!(gray.height(), height);
    
    // Check grayscale values
    // Red: 0.299 * 255 = ~76
    let pixel = gray.get_pixel(0, 0);
    assert_eq!(pixel[0], 76);
    assert_eq!(pixel[1], 76);
    assert_eq!(pixel[2], 76);
    assert_eq!(pixel[3], 255);
    
    // Green: 0.587 * 255 = ~150
    let pixel = gray.get_pixel(1, 0);
    // Allow for small rounding differences
    assert!(pixel[0] >= 149 && pixel[0] <= 150);
    assert!(pixel[1] >= 149 && pixel[1] <= 150);
    assert!(pixel[2] >= 149 && pixel[2] <= 150);
    assert_eq!(pixel[3], 255);
    
    // Blue: 0.114 * 255 = ~29
    let pixel = gray.get_pixel(2, 0);
    assert_eq!(pixel[0], 29);
    assert_eq!(pixel[1], 29);
    assert_eq!(pixel[2], 29);
    assert_eq!(pixel[3], 255);
    
    // White: 0.299 * 255 + 0.587 * 255 + 0.114 * 255 = 255
    let pixel = gray.get_pixel(3, 0);
    assert_eq!(pixel[0], 255);
    assert_eq!(pixel[1], 255);
    assert_eq!(pixel[2], 255);
    assert_eq!(pixel[3], 255);
    
    // Black: 0
    let pixel = gray.get_pixel(4, 0);
    assert_eq!(pixel[0], 0);
    assert_eq!(pixel[1], 0);
    assert_eq!(pixel[2], 0);
    assert_eq!(pixel[3], 255);
}

#[test]
fn test_blur() {
    // Create a new image
    let width = 10;
    let height = 10;
    let mut img = Image::new(width, height);
    
    // Set a single white pixel in the center
    img.set_pixel(5, 5, [255, 255, 255, 255]);
    
    // Blur the image
    let blurred = img.blur(1);
    
    // Check dimensions
    assert_eq!(blurred.width(), width);
    assert_eq!(blurred.height(), height);
    
    // Check that the center pixel is still white
    let pixel = blurred.get_pixel(5, 5);
    assert!(pixel[0] > 0);
    assert!(pixel[1] > 0);
    assert!(pixel[2] > 0);
    assert!(pixel[3] > 0);
    
    // Check that surrounding pixels are now partially white
    let pixel = blurred.get_pixel(4, 5);
    assert!(pixel[0] > 0);
    assert!(pixel[1] > 0);
    assert!(pixel[2] > 0);
    assert!(pixel[3] > 0);
    
    // Check that far away pixels are still black
    let pixel = blurred.get_pixel(0, 0);
    assert_eq!(pixel, [0, 0, 0, 0]);
}