//! String operations for Pyroid
//!
//! This module provides high-performance string operations.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use rayon::prelude::*;
use crate::core::error::PyroidError;
use crate::utils::split_into_chunks;

/// Reverse a string
#[pyfunction]
fn reverse(text: &str) -> String {
    text.chars().rev().collect()
}

/// Convert a string to uppercase
#[pyfunction]
fn to_uppercase(text: &str) -> String {
    text.to_uppercase()
}

/// Convert a string to lowercase
#[pyfunction]
fn to_lowercase(text: &str) -> String {
    text.to_lowercase()
}

/// Split a string by a delimiter
#[pyfunction]
fn split(py: Python, text: &str, delimiter: &str) -> PyResult<Py<PyList>> {
    let parts: Vec<&str> = text.split(delimiter).collect();
    let py_list = PyList::new(py, &parts);
    Ok(py_list.into())
}

/// Join a list of strings with a delimiter
#[pyfunction]
fn join(py: Python, strings: &PyList, delimiter: &str) -> PyResult<String> {
    let mut result = String::new();
    
    for i in 0..strings.len() {
        if i > 0 {
            result.push_str(delimiter);
        }
        
        let item = strings.get_item(i)?;
        let string = item.extract::<String>()?;
        result.push_str(&string);
    }
    
    Ok(result)
}

// Base64 encoding table
const BASE64_TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode a string to base64
#[pyfunction]
fn base64_encode(text: &str) -> String {
    // For small strings, use a simple implementation
    if text.len() < 1024 {
        return encode_base64(text.as_bytes());
    }
    
    // For larger strings, use parallel processing
    let chunks = split_into_chunks(text.as_bytes(), 1024);
    let encoded_chunks: Vec<String> = chunks.par_iter()
        .map(|&chunk| encode_base64(chunk))
        .collect();
    
    encoded_chunks.join("")
}

/// Decode a base64 string
#[pyfunction]
fn base64_decode(py: Python, encoded: &str) -> PyResult<Py<PyString>> {
    match decode_base64(encoded) {
        Ok(bytes) => {
            match String::from_utf8(bytes) {
                Ok(text) => Ok(PyString::new(py, &text).into()),
                Err(_) => Err(PyroidError::ConversionError("Invalid UTF-8 in decoded base64".to_string()).into()),
            }
        },
        Err(e) => Err(PyroidError::ConversionError(format!("Base64 decoding error: {}", e)).into()),
    }
}

/// Decode a base64 string to bytes
#[pyfunction]
fn base64_decode_to_bytes(py: Python, encoded: &str) -> PyResult<Vec<u8>> {
    match decode_base64(encoded) {
        Ok(bytes) => Ok(bytes),
        Err(e) => Err(PyroidError::ConversionError(format!("Base64 decoding error: {}", e)).into()),
    }
}

/// Encode bytes to base64
fn encode_base64(input: &[u8]) -> String {
    let mut output = String::with_capacity((input.len() + 2) / 3 * 4);
    
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        
        let triple = (b0 << 16) | (b1 << 8) | b2;
        
        output.push(BASE64_TABLE[((triple >> 18) & 0x3F) as usize] as char);
        output.push(BASE64_TABLE[((triple >> 12) & 0x3F) as usize] as char);
        
        if chunk.len() > 1 {
            output.push(BASE64_TABLE[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            output.push('=');
        }
        
        if chunk.len() > 2 {
            output.push(BASE64_TABLE[(triple & 0x3F) as usize] as char);
        } else {
            output.push('=');
        }
    }
    
    output
}

/// Decode a base64 string
fn decode_base64(input: &str) -> Result<Vec<u8>, String> {
    // Create a lookup table for base64 characters
    let mut lookup = [0u8; 256];
    for (i, &c) in BASE64_TABLE.iter().enumerate() {
        lookup[c as usize] = i as u8;
    }
    
    // Remove whitespace and validate input
    let clean_input: Vec<u8> = input.bytes().filter(|&c| !c.is_ascii_whitespace()).collect();
    
    // Check if the input is valid base64
    if clean_input.is_empty() {
        return Ok(Vec::new());
    }
    
    if clean_input.len() % 4 != 0 {
        return Err("Invalid base64 length".to_string());
    }
    
    // Calculate output length
    let padding = clean_input.iter().rev().take_while(|&&c| c == b'=').count();
    let output_len = clean_input.len() / 4 * 3 - padding;
    
    let mut output = Vec::with_capacity(output_len);
    
    // Decode
    for chunk in clean_input.chunks(4) {
        // Convert base64 characters to their values
        let mut values = [0u8; 4];
        for (i, &c) in chunk.iter().enumerate() {
            if c == b'=' && (i == 2 || i == 3) {
                values[i] = 0;
            } else if c >= 128 || BASE64_TABLE.iter().find(|&&x| x == c).is_none() {
                return Err(format!("Invalid base64 character: {}", c as char));
            } else {
                values[i] = lookup[c as usize];
            }
        }
        
        // Combine values and add to output
        let combined = (values[0] as u32) << 18 | (values[1] as u32) << 12 | (values[2] as u32) << 6 | values[3] as u32;
        
        output.push(((combined >> 16) & 0xFF) as u8);
        
        if chunk[2] != b'=' {
            output.push(((combined >> 8) & 0xFF) as u8);
        }
        
        if chunk[3] != b'=' {
            output.push((combined & 0xFF) as u8);
        }
    }
    
    Ok(output)
}

/// Register the string module
pub fn register(py: Python, m: &PyModule) -> PyResult<()> {
    let string_module = PyModule::new(py, "string")?;
    
    string_module.add_function(wrap_pyfunction!(reverse, string_module)?)?;
    string_module.add_function(wrap_pyfunction!(to_uppercase, string_module)?)?;
    string_module.add_function(wrap_pyfunction!(to_lowercase, string_module)?)?;
    string_module.add_function(wrap_pyfunction!(split, string_module)?)?;
    string_module.add_function(wrap_pyfunction!(join, string_module)?)?;
    string_module.add_function(wrap_pyfunction!(base64_encode, string_module)?)?;
    string_module.add_function(wrap_pyfunction!(base64_decode, string_module)?)?;
    string_module.add_function(wrap_pyfunction!(base64_decode_to_bytes, string_module)?)?;
    
    m.add_submodule(string_module)?;
    
    Ok(())
}