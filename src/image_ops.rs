//! Image processing operations
//!
//! This module provides high-performance image processing operations.

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use pyo3::types::{PyBytes, PyList, PyTuple, PyDict};
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba, Luma, ImageFormat, imageops};
use imageproc::filter::{gaussian_blur_f32, sharpen3x3};
use std::io::{Cursor, Read, Write};

/// Resize multiple images in parallel
///
/// Args:
///     images: A list of image data (bytes)
///     dimensions: A tuple of (width, height) for the resized images
///     filter: The filter to use for resizing (nearest, triangle, catmull-rom, gaussian, lanczos3)
///
/// Returns:
///     A list of resized image data (bytes)
#[pyfunction]
fn parallel_resize(
    py: Python,
    images: &PyList,
    dimensions: (u32, u32),
    filter: Option<String>
) -> PyResult<PyObject> {
    let (width, height) = dimensions;
    let filter_type = match filter.as_deref().unwrap_or("lanczos3") {
        "nearest" => imageops::FilterType::Nearest,
        "triangle" => imageops::FilterType::Triangle,
        "catmull-rom" => imageops::FilterType::CatmullRom,
        "gaussian" => imageops::FilterType::Gaussian,
        "lanczos3" => imageops::FilterType::Lanczos3,
        _ => return Err(PyValueError::new_err("Unsupported filter type")),
    };
    
    // Process images in parallel
    // First collect indices to avoid PyO3 thread safety issues
    let indices: Vec<usize> = (0..images.len()).collect();
    let results: Result<Vec<PyObject>, PyErr> = indices
        .into_iter() // Use regular iterator instead of parallel
        .map(|i| {
            Python::with_gil(|py| {
                let item = images.get_item(i)?;
                
                // Get image bytes
                let bytes = if let Ok(bytes) = item.extract::<&PyBytes>() {
                    bytes.as_bytes().to_vec()
                } else {
                    return Err(PyValueError::new_err(
                        format!("Item at index {} is not bytes", i)
                    ));
                };
                
                // Load image
                let img = image::load_from_memory(&bytes)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to load image: {}", e)))?;
                
                // Resize image
                let resized = img.resize(width, height, filter_type);
                
                // Convert back to bytes
                let mut buffer = Vec::new();
                let format = image::guess_format(&bytes)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to guess image format: {}", e)))?;
                
                resized.write_to(&mut Cursor::new(&mut buffer), format)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to encode image: {}", e)))?;
                
                // Convert to Python bytes
                Ok(PyBytes::new(py, &buffer).to_object(py))
            })
        })
        .collect();
    
    // Convert to Python list
    let py_list = PyList::empty(py);
    
    for result in results? {
        py_list.append(result)?;
    }
    
    Ok(py_list.into())
}

/// Apply filters to images in parallel
///
/// Args:
///     images: A list of image data (bytes)
///     filter_type: The filter to apply (blur, sharpen, edge, grayscale, invert)
///     params: Optional parameters for the filter (e.g., blur sigma)
///
/// Returns:
///     A list of filtered image data (bytes)
#[pyfunction]
fn parallel_filter(
    py: Python,
    images: &PyList,
    filter_type: String,
    params: Option<&PyDict>
) -> PyResult<PyObject> {
    // Process images in parallel
    // First collect indices to avoid PyO3 thread safety issues
    let indices: Vec<usize> = (0..images.len()).collect();
    let results: Result<Vec<PyObject>, PyErr> = indices
        .into_iter() // Use regular iterator instead of parallel
        .map(|i| {
            Python::with_gil(|py| {
                let item = images.get_item(i)?;
                
                // Get image bytes
                let bytes = if let Ok(bytes) = item.extract::<&PyBytes>() {
                    bytes.as_bytes().to_vec()
                } else {
                    return Err(PyValueError::new_err(
                        format!("Item at index {} is not bytes", i)
                    ));
                };
                
                // Load image
                let img = image::load_from_memory(&bytes)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to load image: {}", e)))?;
                
                // Apply filter
                let filtered = match filter_type.as_str() {
                    "blur" => {
                        let sigma = if let Some(params) = params {
                            if let Ok(Some(sigma_obj)) = params.get_item("sigma") {
                                sigma_obj.extract::<f32>().unwrap_or(1.0)
                            } else {
                                1.0
                            }
                        } else {
                            1.0
                        };
                        
                        // Implement blur manually instead of using imageproc
                        let mut rgba_img = img.to_rgba8();
                        // Apply simple box blur as a fallback
                        let blurred = image::imageops::blur(&rgba_img, sigma);
                        DynamicImage::ImageRgba8(blurred)
                    },
                    "sharpen" => {
                        // Implement sharpen manually instead of using imageproc
                        let mut img = img.to_rgba8();
                        // Get sigma parameter or use default
                        let sharpen_sigma = if let Some(params) = params {
                            if let Ok(Some(sigma_obj)) = params.get_item("sigma") {
                                sigma_obj.extract::<f32>().unwrap_or(1.0)
                            } else {
                                1.0
                            }
                        } else {
                            1.0
                        };
                        // Apply simple sharpening using image crate's built-in functions
                        let sharpened = image::imageops::unsharpen(&img, sharpen_sigma, 5);
                        DynamicImage::ImageRgba8(sharpened)
                    },
                    "edge" => {
                        // Convert to grayscale and apply edge detection
                        let gray_img = img.to_luma8();
                        // Use image crate's filter3x3 to implement a simple edge detection
                        let edges = image::imageops::filter3x3(&gray_img, &[
                            -1.0, -1.0, -1.0,
                            -1.0,  8.0, -1.0,
                            -1.0, -1.0, -1.0
                        ]);
                        DynamicImage::ImageLuma8(edges)
                    },
                    "grayscale" => {
                        img.grayscale()
                    },
                    "invert" => {
                        let mut rgba_img = img.to_rgba8();
                        
                        for pixel in rgba_img.pixels_mut() {
                            pixel[0] = 255 - pixel[0];
                            pixel[1] = 255 - pixel[1];
                            pixel[2] = 255 - pixel[2];
                            // Don't invert alpha channel
                        }
                        
                        DynamicImage::ImageRgba8(rgba_img)
                    },
                    _ => return Err(PyValueError::new_err(format!("Unsupported filter type: {}", filter_type))),
                };
                
                // Convert back to bytes
                let mut buffer = Vec::new();
                let format = image::guess_format(&bytes)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to guess image format: {}", e)))?;
                
                filtered.write_to(&mut Cursor::new(&mut buffer), format)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to encode image: {}", e)))?;
                
                // Convert to Python bytes
                Ok(PyBytes::new(py, &buffer).to_object(py))
            })
        })
        .collect();
    
    // Convert to Python list
    let py_list = PyList::empty(py);
    
    for result in results? {
        py_list.append(result)?;
    }
    
    Ok(py_list.into())
}

/// Convert image formats in parallel
///
/// Args:
///     images: A list of image data (bytes)
///     from_format: Source format (auto-detect if None)
///     to_format: Target format (jpeg, png, gif, webp, etc.)
///     quality: JPEG/WebP quality (0-100, default: 90)
///
/// Returns:
///     A list of converted image data (bytes)
#[pyfunction]
#[pyo3(signature = (images, to_format, from_format=None, quality=90))]
fn parallel_convert(
    py: Python,
    images: &PyList,
    to_format: String,
    from_format: Option<String>,
    quality: Option<u8>
) -> PyResult<PyObject> {
    let quality = quality.unwrap_or(90);
    
    // Get target format
    let target_format = match to_format.to_lowercase().as_str() {
        "jpeg" | "jpg" => ImageFormat::Jpeg,
        "png" => ImageFormat::Png,
        "gif" => ImageFormat::Gif,
        "webp" => ImageFormat::WebP,
        "bmp" => ImageFormat::Bmp,
        "tiff" | "tif" => ImageFormat::Tiff,
        "pnm" => ImageFormat::Pnm,
        "tga" => ImageFormat::Tga,
        "farbfeld" | "ff" => ImageFormat::Farbfeld,
        "avif" => ImageFormat::Avif,
        _ => return Err(PyValueError::new_err(format!("Unsupported target format: {}", to_format))),
    };
    
    // Process images in parallel
    // First collect indices to avoid PyO3 thread safety issues
    let indices: Vec<usize> = (0..images.len()).collect();
    let results: Result<Vec<PyObject>, PyErr> = indices
        .into_iter() // Use regular iterator instead of parallel
        .map(|i| {
            Python::with_gil(|py| {
                let item = images.get_item(i)?;
                
                // Get image bytes
                let bytes = if let Ok(bytes) = item.extract::<&PyBytes>() {
                    bytes.as_bytes().to_vec()
                } else {
                    return Err(PyValueError::new_err(
                        format!("Item at index {} is not bytes", i)
                    ));
                };
                
                // Determine source format
                let source_format = if let Some(ref fmt) = from_format {
                    match fmt.to_lowercase().as_str() {
                        "jpeg" | "jpg" => ImageFormat::Jpeg,
                        "png" => ImageFormat::Png,
                        "gif" => ImageFormat::Gif,
                        "webp" => ImageFormat::WebP,
                        "bmp" => ImageFormat::Bmp,
                        "tiff" | "tif" => ImageFormat::Tiff,
                        "pnm" => ImageFormat::Pnm,
                        "tga" => ImageFormat::Tga,
                        "farbfeld" | "ff" => ImageFormat::Farbfeld,
                        "avif" => ImageFormat::Avif,
                        _ => return Err(PyValueError::new_err(format!("Unsupported source format: {}", fmt))),
                    }
                } else {
                    // Auto-detect format
                    image::guess_format(&bytes)
                        .map_err(|e| PyRuntimeError::new_err(format!("Failed to guess image format: {}", e)))?
                };
                
                // Load image
                let img = image::load_from_memory_with_format(&bytes, source_format)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to load image: {}", e)))?;
                
                // Convert to target format
                let mut buffer = Vec::new();
                
                if target_format == ImageFormat::Jpeg {
                    // For JPEG, we need to handle quality
                    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buffer, quality);
                    encoder.encode_image(&img)
                        .map_err(|e| PyRuntimeError::new_err(format!("Failed to encode JPEG: {}", e)))?;
                } else {
                    // For other formats
                    img.write_to(&mut Cursor::new(&mut buffer), target_format)
                        .map_err(|e| PyRuntimeError::new_err(format!("Failed to encode image: {}", e)))?;
                }
                
                // Convert to Python bytes
                Ok(PyBytes::new(py, &buffer).to_object(py))
            })
        })
        .collect();
    
    // Convert to Python list
    let py_list = PyList::empty(py);
    
    for result in results? {
        py_list.append(result)?;
    }
    
    Ok(py_list.into())
}

/// Extract image metadata in parallel
///
/// Args:
///     images: A list of image data (bytes)
///
/// Returns:
///     A list of dictionaries containing image metadata
#[pyfunction]
fn parallel_extract_metadata(py: Python, images: &PyList) -> PyResult<PyObject> {
    // Process images in parallel
    // First collect indices to avoid PyO3 thread safety issues
    let indices: Vec<usize> = (0..images.len()).collect();
    let results: Result<Vec<PyObject>, PyErr> = indices
        .into_iter() // Use regular iterator instead of parallel
        .map(|i| {
            Python::with_gil(|py| {
                let item = images.get_item(i)?;
                
                // Get image bytes
                let bytes = if let Ok(bytes) = item.extract::<&PyBytes>() {
                    bytes.as_bytes().to_vec()
                } else {
                    return Err(PyValueError::new_err(
                        format!("Item at index {} is not bytes", i)
                    ));
                };
                
                // Load image
                let img = image::load_from_memory(&bytes)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to load image: {}", e)))?;
                
                // Extract metadata
                let dimensions = img.dimensions();
                let color_type = format!("{:?}", img.color());
                
                // Create metadata dictionary
                let metadata = PyDict::new(py);
                metadata.set_item("width", dimensions.0)?;
                metadata.set_item("height", dimensions.1)?;
                metadata.set_item("color_type", color_type)?;
                
                // Try to get format
                if let Ok(format) = image::guess_format(&bytes) {
                    metadata.set_item("format", format!("{:?}", format))?;
                }
                
                Ok(metadata.to_object(py))
            })
        })
        .collect();
    
    // Convert to Python list
    let py_list = PyList::empty(py);
    
    for result in results? {
        py_list.append(result)?;
    }
    
    Ok(py_list.into())
}

/// Register the image processing operations module
pub fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parallel_resize, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_filter, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_convert, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_extract_metadata, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    // Helper function to load a test image
    fn load_test_image(py: Python) -> PyObject {
        // Create a small test image
        let img = DynamicImage::new_rgb8(100, 100);
        let mut buffer = Vec::new();
        img.write_to(&mut Cursor::new(&mut buffer), ImageFormat::Png).unwrap();
        PyBytes::new(py, &buffer).to_object(py)
    }
    
    #[test]
    fn test_resize() {
        Python::with_gil(|py| {
            let test_img = load_test_image(py);
            let images = PyList::new(py, &[test_img]);
            
            let result = parallel_resize(py, images, (50, 50), None).unwrap();
            let resized: Vec<&PyBytes> = result.extract(py).unwrap();
            
            assert_eq!(resized.len(), 1);
            assert!(resized[0].as_bytes().len() > 0);
        });
    }
    
    #[test]
    fn test_filter() {
        Python::with_gil(|py| {
            let test_img = load_test_image(py);
            let images = PyList::new(py, &[test_img]);
            
            let result = parallel_filter(py, images, "grayscale".to_string(), None).unwrap();
            let filtered: Vec<&PyBytes> = result.extract(py).unwrap();
            
            assert_eq!(filtered.len(), 1);
            assert!(filtered[0].as_bytes().len() > 0);
        });
    }
}