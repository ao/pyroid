//! Utility functions for Pyroid
//!
//! This module contains internal utility functions used by other modules.

pub mod conversions;

/// Splits a slice into approximately equal chunks for parallel processing
pub fn split_into_chunks<T>(data: &[T], chunk_size: usize) -> Vec<&[T]> {
    if data.is_empty() {
        return vec![];
    }
    
    if data.len() <= chunk_size {
        return vec![data];
    }
    
    let chunk_count = (data.len() + chunk_size - 1) / chunk_size;
    let mut chunks = Vec::with_capacity(chunk_count);
    
    for i in 0..chunk_count {
        let start = i * chunk_size;
        let end = (start + chunk_size).min(data.len());
        chunks.push(&data[start..end]);
    }
    
    chunks
}

/// Splits a string into approximately equal chunks for parallel processing
pub fn split_text_into_chunks(text: &str, chunk_size: usize) -> Vec<&str> {
    if text.is_empty() {
        return vec![];
    }
    
    if text.len() <= chunk_size {
        return vec![text];
    }
    
    let chunk_count = (text.len() + chunk_size - 1) / chunk_size;
    let mut chunks = Vec::with_capacity(chunk_count);
    
    // Try to split at character boundaries
    let char_indices: Vec<_> = text.char_indices().collect();
    let total_chars = char_indices.len();
    let chars_per_chunk = total_chars / chunk_count;
    
    let mut start_idx = 0;
    for i in 0..chunk_count - 1 {
        let target_idx = (i + 1) * chars_per_chunk;
        if target_idx >= total_chars {
            break;
        }
        
        let end_idx = char_indices[target_idx].0;
        chunks.push(&text[start_idx..end_idx]);
        start_idx = end_idx;
    }
    
    // Add the last chunk
    chunks.push(&text[start_idx..]);
    
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_split_into_chunks() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        
        // Test with chunk size 3
        let chunks = split_into_chunks(&data, 3);
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0], &[1, 2, 3]);
        assert_eq!(chunks[1], &[4, 5, 6]);
        assert_eq!(chunks[2], &[7, 8, 9]);
        assert_eq!(chunks[3], &[10]);
        
        // Test with chunk size larger than data
        let chunks = split_into_chunks(&data, 20);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], &data[..]);
        
        // Test with empty data
        let empty: Vec<i32> = vec![];
        let chunks = split_into_chunks(&empty, 3);
        assert_eq!(chunks.len(), 0);
    }
    
    #[test]
    fn test_split_text_into_chunks() {
        let text = "Hello, world! This is a test.";
        
        // Test with chunk size 10
        let chunks = split_text_into_chunks(text, 10);
        assert!(chunks.len() > 1);
        assert_eq!(chunks.join(""), text);
        
        // Test with chunk size larger than text
        let chunks = split_text_into_chunks(text, 100);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
        
        // Test with empty text
        let empty = "";
        let chunks = split_text_into_chunks(empty, 10);
        assert_eq!(chunks.len(), 0);
    }
}