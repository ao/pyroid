//! File I/O operations
//!
//! This module provides high-performance file I/O operations.

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError, PyFileNotFoundError};
use pyo3::types::{PyDict, PyList, PyBytes};
use std::collections::HashMap;
use rayon::prelude::*;
use std::fs::File;
use std::io::{Read, Write, BufReader, BufWriter};
use std::path::Path;
use csv::{ReaderBuilder, WriterBuilder};
use serde_json::{Value, from_str, to_string};
use flate2::read::{GzDecoder, ZlibDecoder};
use flate2::write::{GzEncoder, ZlibEncoder};
use flate2::Compression;

/// Read multiple CSV files in parallel
///
/// Args:
///     files: A list of file paths to read
///     schema: An optional schema dictionary mapping column names to types
///     has_header: Whether the CSV files have headers (default: true)
///     delimiter: The delimiter character (default: ',')
///
/// Returns:
///     A list of dictionaries, each containing the data from one CSV file
#[pyfunction]
fn parallel_read_csv(
    py: Python,
    files: Vec<String>,
    schema: Option<&PyDict>,
    has_header: Option<bool>,
    delimiter: Option<char>
) -> PyResult<PyObject> {
    let has_header = has_header.unwrap_or(true);
    let delimiter = delimiter.unwrap_or(',');
    
    // Parse schema if provided
    let schema_map: Option<HashMap<String, String>> = if let Some(schema_dict) = schema {
        let mut map = HashMap::new();
        
        for (key, value) in schema_dict.iter() {
            let col_name = key.extract::<String>()?;
            let col_type = value.extract::<String>()?;
            map.insert(col_name, col_type);
        }
        
        Some(map)
    } else {
        None
    };
    
    // Read files in parallel
    let results: Result<Vec<PyObject>, PyErr> = files.par_iter()
        .map(|file_path| {
            Python::with_gil(|py| {
                // Check if file exists
                if !Path::new(file_path).exists() {
                    return Err(PyFileNotFoundError::new_err(format!("File not found: {}", file_path)));
                }
                
                // Open the file
                let file = File::open(file_path)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to open file {}: {}", file_path, e)))?;
                
                let reader = BufReader::new(file);
                
                // Create CSV reader
                let mut csv_reader = ReaderBuilder::new()
                    .has_headers(has_header)
                    .delimiter(delimiter as u8)
                    .from_reader(reader);
                
                // Read headers
                let headers: Vec<String> = if has_header {
                    csv_reader.headers()
                        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read CSV headers: {}", e)))?
                        .iter()
                        .map(|s| s.to_string())
                        .collect()
                } else {
                    // If no headers, use column indices as headers
                    let first_record = csv_reader.records().next();
                    
                    if let Some(Ok(record)) = first_record {
                        (0..record.len())
                            .map(|i| format!("column_{}", i))
                            .collect()
                    } else {
                        return Err(PyRuntimeError::new_err("Failed to read first CSV record"));
                    }
                };
                
                // Create a dictionary for each column
                let mut columns: HashMap<String, Vec<PyObject>> = headers.iter()
                    .map(|header| (header.clone(), Vec::new()))
                    .collect();
                
                // Read records
                for result in csv_reader.records() {
                    let record = result.map_err(|e| PyRuntimeError::new_err(format!("Failed to read CSV record: {}", e)))?;
                    
                    for (i, field) in record.iter().enumerate() {
                        if i < headers.len() {
                            let header = &headers[i];
                            
                            // Convert field based on schema
                            let py_value = if let Some(ref schema) = schema_map {
                                if let Some(col_type) = schema.get(header) as Option<&String> {
                                    match col_type.as_str() {
                                        "int" => {
                                            let value = field.parse::<i64>()
                                                .map_err(|_| PyValueError::new_err(format!("Failed to parse '{}' as int", field)))?;
                                            value.to_object(py)
                                        },
                                        "float" => {
                                            let value = field.parse::<f64>()
                                                .map_err(|_| PyValueError::new_err(format!("Failed to parse '{}' as float", field)))?;
                                            value.to_object(py)
                                        },
                                        "bool" => {
                                            let value = match field.to_lowercase().as_str() {
                                                "true" | "1" | "yes" | "y" => true,
                                                "false" | "0" | "no" | "n" => false,
                                                _ => return Err(PyValueError::new_err(format!("Failed to parse '{}' as bool", field))),
                                            };
                                            value.to_object(py)
                                        },
                                        _ => field.to_object(py), // Default to string
                                    }
                                } else {
                                    field.to_object(py) // No schema for this column, use string
                                }
                            } else {
                                field.to_object(py) // No schema, use string
                            };
                            
                            columns.get_mut(header).unwrap().push(py_value);
                        }
                    }
                }
                
                // Create a dictionary with the data
                let py_dict = PyDict::new(py);
                
                for (header, values) in columns {
                    let py_list = PyList::empty(py);
                    for value in values {
                        py_list.append(value)?;
                    }
                    py_dict.set_item(header, py_list)?;
                }
                
                Ok(py_dict.to_object(py))
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

/// Parse multiple JSON strings in parallel
///
/// Args:
///     json_strings: A list of JSON strings to parse
///
/// Returns:
///     A list of parsed JSON objects (as Python dictionaries)
#[pyfunction]
fn parallel_json_parse(py: Python, json_strings: Vec<String>) -> PyResult<PyObject> {
    // Parse JSON strings in parallel
    let results: Result<Vec<PyObject>, PyErr> = json_strings.par_iter()
        .map(|json_str| {
            Python::with_gil(|py| {
                // Parse JSON
                let json_value: Value = from_str(json_str)
                    .map_err(|e| PyValueError::new_err(format!("Failed to parse JSON: {}", e)))?;
                
                // Convert to Python object
                json_to_py_object(py, &json_value)
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

/// Helper function to convert a serde_json::Value to a Python object
fn json_to_py_object(py: Python, value: &Value) -> PyResult<PyObject> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => Ok(b.to_object(py)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.to_object(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.to_object(py))
            } else {
                Err(PyValueError::new_err("Unsupported JSON number"))
            }
        },
        Value::String(s) => Ok(s.to_object(py)),
        Value::Array(arr) => {
            let py_list = PyList::empty(py);
            
            for item in arr {
                py_list.append(json_to_py_object(py, item)?)?;
            }
            
            Ok(py_list.to_object(py))
        },
        Value::Object(obj) => {
            let py_dict = PyDict::new(py);
            
            for (key, value) in obj {
                py_dict.set_item(key, json_to_py_object(py, value)?)?;
            }
            
            Ok(py_dict.to_object(py))
        },
    }
}

/// Compress data in parallel
///
/// Args:
///     data: A list of bytes or strings to compress
///     method: Compression method (gzip, zlib, default: gzip)
///     level: Compression level (1-9, default: 6)
///
/// Returns:
///     A list of compressed data (as bytes)
#[pyfunction]
fn parallel_compress(py: Python, data: &PyList, method: Option<String>, level: Option<i32>) -> PyResult<PyObject> {
    let method = method.unwrap_or_else(|| "gzip".to_string());
    let level = level.unwrap_or(6);
    
    if level < 1 || level > 9 {
        return Err(PyValueError::new_err("Compression level must be between 1 and 9"));
    }
    
    // Compress data in parallel
    // First collect indices to avoid PyO3 thread safety issues
    let indices: Vec<usize> = (0..data.len()).collect();
    let results: Result<Vec<PyObject>, PyErr> = indices
        .into_iter() // Use regular iterator instead of parallel
        .map(|i| {
            Python::with_gil(|py| {
                let item = data.get_item(i)?;
                
                // Get bytes to compress
                let bytes = if let Ok(bytes) = item.extract::<&PyBytes>() {
                    bytes.as_bytes().to_vec()
                } else if let Ok(string) = item.extract::<String>() {
                    string.into_bytes()
                } else {
                    return Err(PyValueError::new_err(
                        format!("Item at index {} is not bytes or string", i)
                    ));
                };
                
                // Compress based on method
                let compressed = match method.as_str() {
                    "gzip" => {
                        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(level as u32));
                        encoder.write_all(&bytes)
                            .map_err(|e| PyRuntimeError::new_err(format!("Failed to compress data: {}", e)))?;
                        encoder.finish()
                            .map_err(|e| PyRuntimeError::new_err(format!("Failed to finish compression: {}", e)))?
                    },
                    "zlib" => {
                        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(level as u32));
                        encoder.write_all(&bytes)
                            .map_err(|e| PyRuntimeError::new_err(format!("Failed to compress data: {}", e)))?;
                        encoder.finish()
                            .map_err(|e| PyRuntimeError::new_err(format!("Failed to finish compression: {}", e)))?
                    },
                    _ => return Err(PyValueError::new_err(format!("Unsupported compression method: {}", method))),
                };
                
                // Convert to Python bytes
                Ok(PyBytes::new(py, &compressed).to_object(py))
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

/// Decompress data in parallel
///
/// Args:
///     data: A list of compressed bytes
///     method: Compression method (gzip, zlib, default: gzip)
///
/// Returns:
///     A list of decompressed data (as bytes)
#[pyfunction]
fn parallel_decompress(py: Python, data: &PyList, method: Option<String>) -> PyResult<PyObject> {
    let method = method.unwrap_or_else(|| "gzip".to_string());
    
    // Decompress data in parallel
    // First collect indices to avoid PyO3 thread safety issues
    let indices: Vec<usize> = (0..data.len()).collect();
    let results: Result<Vec<PyObject>, PyErr> = indices
        .into_iter() // Use regular iterator instead of parallel
        .map(|i| {
            Python::with_gil(|py| {
                let item = data.get_item(i)?;
                
                // Get compressed bytes
                let bytes = if let Ok(bytes) = item.extract::<&PyBytes>() {
                    bytes.as_bytes()
                } else {
                    return Err(PyValueError::new_err(
                        format!("Item at index {} is not bytes", i)
                    ));
                };
                
                // Decompress based on method
                let decompressed = match method.as_str() {
                    "gzip" => {
                        let mut decoder = GzDecoder::new(bytes);
                        let mut result = Vec::new();
                        decoder.read_to_end(&mut result)
                            .map_err(|e| PyRuntimeError::new_err(format!("Failed to decompress data: {}", e)))?;
                        result
                    },
                    "zlib" => {
                        let mut decoder = ZlibDecoder::new(bytes);
                        let mut result = Vec::new();
                        decoder.read_to_end(&mut result)
                            .map_err(|e| PyRuntimeError::new_err(format!("Failed to decompress data: {}", e)))?;
                        result
                    },
                    _ => return Err(PyValueError::new_err(format!("Unsupported compression method: {}", method))),
                };
                
                // Convert to Python bytes
                Ok(PyBytes::new(py, &decompressed).to_object(py))
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

/// Register the file I/O operations module
pub fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parallel_read_csv, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_json_parse, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_compress, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_decompress, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_json_parse() {
        Python::with_gil(|py| {
            let json_strings = vec![
                r#"{"name": "Alice", "age": 30}"#.to_string(),
                r#"{"name": "Bob", "age": 25}"#.to_string(),
            ];
            
            let result = parallel_json_parse(py, json_strings).unwrap();
            let parsed: Vec<HashMap<String, PyObject>> = result.extract(py).unwrap();
            
            assert_eq!(parsed.len(), 2);
            
            let alice = &parsed[0];
            let alice_name: String = alice.get("name").unwrap().extract(py).unwrap();
            let alice_age: i64 = alice.get("age").unwrap().extract(py).unwrap();
            
            assert_eq!(alice_name, "Alice");
            assert_eq!(alice_age, 30);
        });
    }
    
    #[test]
    fn test_compress_decompress() {
        Python::with_gil(|py| {
            let data = PyList::new(py, &[
                "Hello, world!".to_string(),
                "This is a test.".to_string(),
            ]);
            
            let compressed = parallel_compress(py, data, Some("gzip".to_string()), Some(6)).unwrap();
            let decompressed = parallel_decompress(py, compressed.extract(py).unwrap(), Some("gzip".to_string())).unwrap();
            
            let decompressed_strings: Vec<String> = decompressed.extract::<Vec<PyObject>>(py)
                .unwrap()
                .iter()
                .map(|bytes| String::from_utf8(bytes.extract::<Vec<u8>>(py).unwrap()).unwrap())
                .collect();
            
            assert_eq!(decompressed_strings, vec!["Hello, world!", "This is a test."]);
        });
    }
}