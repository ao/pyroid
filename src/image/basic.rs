//! Basic image operations for Pyroid
//!
//! This module provides basic image operations without external dependencies.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyDict};
use crate::core::error::PyroidError;
use std::io::Cursor;

/// Image format
#[derive(Debug, Clone, Copy)]
pub enum ImageFormat {
    PNG,
    JPEG,
    BMP,
    GIF,
    Unknown,
}

/// Image struct
#[pyclass]
#[derive(Clone)]
pub struct Image {
    width: usize,
    height: usize,
    channels: usize,
    data: Vec<u8>,
    format: ImageFormat,
}

#[pymethods]
impl Image {
    /// Create a new image
    #[new]
    fn new(width: usize, height: usize, channels: usize) -> Self {
        let data = vec![0; width * height * channels];
        Self {
            width,
            height,
            channels,
            data,
            format: ImageFormat::Unknown,
        }
    }
    
    /// Get the width of the image
    #[getter]
    fn width(&self) -> usize {
        self.width
    }
    
    /// Get the height of the image
    #[getter]
    fn height(&self) -> usize {
        self.height
    }
    
    /// Get the number of channels
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }
    
    /// Get the image data
    #[getter]
    fn data(&self, py: Python) -> PyResult<Py<PyBytes>> {
        Ok(PyBytes::new(py, &self.data).into())
    }
    
    /// Get a pixel value
    fn get_pixel(&self, x: usize, y: usize) -> PyResult<Vec<u8>> {
        if x >= self.width || y >= self.height {
            return Err(PyroidError::InputError(format!(
                "Pixel coordinates ({}, {}) out of bounds for image of size {}x{}",
                x, y, self.width, self.height
            )).into());
        }
        
        let idx = (y * self.width + x) * self.channels;
        let mut pixel = Vec::with_capacity(self.channels);
        for i in 0..self.channels {
            pixel.push(self.data[idx + i]);
        }
        
        Ok(pixel)
    }
    
    /// Set a pixel value
    fn set_pixel(&mut self, x: usize, y: usize, pixel: Vec<u8>) -> PyResult<()> {
        if x >= self.width || y >= self.height {
            return Err(PyroidError::InputError(format!(
                "Pixel coordinates ({}, {}) out of bounds for image of size {}x{}",
                x, y, self.width, self.height
            )).into());
        }
        
        if pixel.len() != self.channels {
            return Err(PyroidError::InputError(format!(
                "Pixel data must have {} channels, got {}",
                self.channels, pixel.len()
            )).into());
        }
        
        let idx = (y * self.width + x) * self.channels;
        for i in 0..self.channels {
            self.data[idx + i] = pixel[i];
        }
        
        Ok(())
    }
    
    /// Create a grayscale version of the image
    fn to_grayscale(&self, py: Python) -> PyResult<PyObject> {
        if self.channels == 1 {
            // Already grayscale
            return Ok(self.clone().into_py(py));
        }
        
        let mut result = Self::new(self.width, self.height, 1);
        
        for y in 0..self.height {
            for x in 0..self.width {
                let pixel = self.get_pixel(x, y)?;
                
                // Simple grayscale conversion: average of RGB channels
                let gray = if self.channels >= 3 {
                    ((pixel[0] as u32 + pixel[1] as u32 + pixel[2] as u32) / 3) as u8
                } else {
                    pixel[0]
                };
                
                result.set_pixel(x, y, vec![gray])?;
            }
        }
        
        Ok(result.into_py(py))
    }
    
    /// Resize the image using nearest neighbor interpolation
    fn resize(&self, py: Python, new_width: usize, new_height: usize) -> PyResult<PyObject> {
        let mut result = Self::new(new_width, new_height, self.channels);
        
        let x_ratio = self.width as f64 / new_width as f64;
        let y_ratio = self.height as f64 / new_height as f64;
        
        for y in 0..new_height {
            for x in 0..new_width {
                let px = (x as f64 * x_ratio) as usize;
                let py = (y as f64 * y_ratio) as usize;
                
                let pixel = self.get_pixel(px, py)?;
                result.set_pixel(x, y, pixel)?;
            }
        }
        
        Ok(result.into_py(py))
    }
    
    /// Apply a simple blur filter
    fn blur(&self, py: Python, radius: usize) -> PyResult<PyObject> {
        let mut result = Self::new(self.width, self.height, self.channels);
        
        for y in 0..self.height {
            for x in 0..self.width {
                let mut pixel = vec![0u32; self.channels];
                let mut count = 0;
                
                // Average pixels in the radius
                for dy in 0..=radius*2 {
                    let ny = y as isize + dy as isize - radius as isize;
                    if ny < 0 || ny >= self.height as isize {
                        continue;
                    }
                    
                    for dx in 0..=radius*2 {
                        let nx = x as isize + dx as isize - radius as isize;
                        if nx < 0 || nx >= self.width as isize {
                            continue;
                        }
                        
                        let neighbor = self.get_pixel(nx as usize, ny as usize)?;
                        for c in 0..self.channels {
                            pixel[c] += neighbor[c] as u32;
                        }
                        count += 1;
                    }
                }
                
                // Calculate average
                let mut final_pixel = Vec::with_capacity(self.channels);
                for c in 0..self.channels {
                    final_pixel.push((pixel[c] / count) as u8);
                }
                
                result.set_pixel(x, y, final_pixel)?;
            }
        }
        
        Ok(result.into_py(py))
    }
    
    /// Adjust the brightness of the image
    fn adjust_brightness(&self, py: Python, factor: f64) -> PyResult<PyObject> {
        let mut result = Self::new(self.width, self.height, self.channels);
        
        for y in 0..self.height {
            for x in 0..self.width {
                let pixel = self.get_pixel(x, y)?;
                let mut new_pixel = Vec::with_capacity(self.channels);
                
                for &value in &pixel {
                    let new_value = (value as f64 * factor).min(255.0).max(0.0) as u8;
                    new_pixel.push(new_value);
                }
                
                result.set_pixel(x, y, new_pixel)?;
            }
        }
        
        Ok(result.into_py(py))
    }
    
    /// Convert the image to a string representation
    fn __str__(&self) -> String {
        format!("Image({}x{}, {} channels)", self.width, self.height, self.channels)
    }
    
    /// Convert the image to a string representation
    fn __repr__(&self) -> String {
        format!("Image({}x{}, {} channels)", self.width, self.height, self.channels)
    }
}

/// Create a new image with the specified dimensions
#[pyfunction]
fn create_image(py: Python, width: usize, height: usize, channels: usize) -> PyResult<PyObject> {
    let image = Image::new(width, height, channels);
    Ok(image.into_py(py))
}

/// Create a new image from raw bytes
#[pyfunction]
fn from_bytes(py: Python, data: &PyBytes, width: usize, height: usize, channels: usize) -> PyResult<PyObject> {
    if data.as_bytes().len() != width * height * channels {
        return Err(PyroidError::InputError(format!(
            "Data length {} does not match dimensions {}x{}x{}",
            data.as_bytes().len(), width, height, channels
        )).into());
    }
    
    let mut image = Image::new(width, height, channels);
    image.data.copy_from_slice(data.as_bytes());
    
    Ok(image.into_py(py))
}

/// Detect image format from bytes
fn detect_format(data: &[u8]) -> ImageFormat {
    if data.len() < 8 {
        return ImageFormat::Unknown;
    }
    
    // Check for PNG signature
    if data.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
        return ImageFormat::PNG;
    }
    
    // Check for JPEG signature
    if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return ImageFormat::JPEG;
    }
    
    // Check for BMP signature
    if data.starts_with(&[0x42, 0x4D]) {
        return ImageFormat::BMP;
    }
    
    // Check for GIF signature
    if data.starts_with(&[0x47, 0x49, 0x46, 0x38]) {
        return ImageFormat::GIF;
    }
    
    ImageFormat::Unknown
}

/// Register the basic module
pub fn register(py: Python, module: &PyModule) -> PyResult<()> {
    let basic_module = PyModule::new(py, "basic")?;
    
    basic_module.add_class::<Image>()?;
    basic_module.add_function(wrap_pyfunction!(create_image, basic_module)?)?;
    basic_module.add_function(wrap_pyfunction!(from_bytes, basic_module)?)?;
    
    // Add the basic module to the parent module
    module.add_submodule(basic_module)?;
    
    Ok(())
}