//! String processing operations
//!
//! This module provides high-performance string processing operations.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;
use regex::Regex;
use crate::utils::split_text_into_chunks;

/// Perform regex replacement in parallel
///
/// Args:
///     text: The input text to process
///     pattern: The regex pattern to search for
///     replacement: The replacement string
///
/// Returns:
///     The text with all matches replaced
#[pyfunction]
fn parallel_regex_replace(text: &str, pattern: &str, replacement: &str) -> PyResult<String> {
    // Compile the regex pattern
    let regex = match Regex::new(pattern) {
        Ok(r) => r,
        Err(e) => return Err(PyValueError::new_err(format!("Invalid regex pattern: {}", e))),
    };
    
    // Split the text into chunks for parallel processing
    let chunks = split_text_into_chunks(text, 10000); // 10KB chunks
    
    // Process each chunk in parallel
    let results: Vec<String> = chunks.par_iter()
        .map(|chunk| regex.replace_all(chunk, replacement).to_string())
        .collect();
    
    // Join the results
    Ok(results.join(""))
}

/// Clean up text in parallel (trim, lowercase, remove non-alphanumeric)
///
/// Args:
///     texts: A list of strings to clean up
///
/// Returns:
///     A list of cleaned strings
#[pyfunction]
fn parallel_text_cleanup(texts: Vec<&str>) -> Vec<String> {
    texts.par_iter()
        .map(|&text| {
            text.trim()
                .to_lowercase()
                .chars()
                .filter(|&c| c.is_alphanumeric() || c.is_whitespace())
                .collect()
        })
        .collect()
}

/// Encode a string to base64 in parallel
///
/// Args:
///     text: The input text to encode
///
/// Returns:
///     The base64 encoded string
#[pyfunction]
fn parallel_base64_encode(text: &str) -> String {
    // For small strings, just use the standard base64 encoding
    if text.len() < 10000 {
        return base64::encode(text);
    }
    
    // For larger strings, split into chunks and process in parallel
    let chunks = split_text_into_chunks(text, 10000);
    let encoded_chunks: Vec<String> = chunks.par_iter()
        .map(|&chunk| base64::encode(chunk))
        .collect();
    
    encoded_chunks.join("")
}

/// Decode a base64 string in parallel
///
/// Args:
///     encoded: The base64 encoded string
///
/// Returns:
///     The decoded string
#[pyfunction]
fn parallel_base64_decode(encoded: &str) -> PyResult<String> {
    // For small strings, just use the standard base64 decoding
    if encoded.len() < 10000 {
        match base64::decode(encoded) {
            Ok(bytes) => match String::from_utf8(bytes) {
                Ok(s) => return Ok(s),
                Err(e) => return Err(PyValueError::new_err(format!("Invalid UTF-8: {}", e))),
            },
            Err(e) => return Err(PyValueError::new_err(format!("Invalid base64: {}", e))),
        }
    }
    
    // For larger strings, we need to be careful with base64 padding
    // Base64 decoding requires complete 4-byte chunks, so we can't just split arbitrarily
    
    // Calculate chunk size that's a multiple of 4
    let chunk_size = 10000 - (10000 % 4);
    
    // Split the encoded string into chunks
    let mut chunks = Vec::new();
    let mut i = 0;
    while i < encoded.len() {
        let end = (i + chunk_size).min(encoded.len());
        chunks.push(&encoded[i..end]);
        i = end;
    }
    
    // Decode each chunk in parallel
    let decoded_chunks_result: Result<Vec<String>, String> = chunks.par_iter()
        .map(|&chunk| {
            match base64::decode(chunk) {
                Ok(bytes) => match String::from_utf8(bytes) {
                    Ok(s) => Ok(s),
                    Err(e) => Err(format!("Invalid UTF-8: {}", e)),
                },
                Err(e) => Err(format!("Invalid base64: {}", e)),
            }
        })
        .collect();
    
    // Check for errors and join the results
    match decoded_chunks_result {
        Ok(decoded_chunks) => Ok(decoded_chunks.join("")),
        Err(e) => Err(PyValueError::new_err(e)),
    }
}

/// Register the string operations module
pub fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parallel_regex_replace, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_text_cleanup, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_base64_encode, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_base64_decode, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parallel_regex_replace() {
        let text = "Hello world! Hello universe! Hello galaxy!";
        let result = parallel_regex_replace(text, r"Hello", "Hi").unwrap();
        assert_eq!(result, "Hi world! Hi universe! Hi galaxy!");
    }
    
    #[test]
    fn test_parallel_text_cleanup() {
        let texts = vec![
            "  Hello, World! ",
            "123 Testing 456",
            "Special @#$% Characters",
        ];
        let result = parallel_text_cleanup(texts);
        assert_eq!(result, vec![
            "hello world",
            "123 testing 456",
            "special  characters",
        ]);
    }
    
    #[test]
    fn test_parallel_base64_encode_decode() {
        let text = "Hello, world! This is a test of base64 encoding and decoding.";
        let encoded = parallel_base64_encode(text);
        let decoded = parallel_base64_decode(&encoded).unwrap();
        assert_eq!(decoded, text);
    }
}