//! Network operations for Pyroid
//!
//! This module provides high-performance network operations.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyBytes};
use crate::core::error::PyroidError;
use crate::core::config::get_config;

use std::io::{Read, Write};
use std::net::TcpStream;
use std::collections::HashMap;

/// Simple HTTP client implementation
struct HttpClient;

impl HttpClient {
    /// Send an HTTP GET request
    fn get(url: &str) -> Result<HttpResponse, String> {
        Self::request("GET", url, None, None)
    }
    
    /// Send an HTTP POST request
    fn post(url: &str, body: Option<Vec<u8>>, content_type: Option<&str>) -> Result<HttpResponse, String> {
        let headers = if let Some(ct) = content_type {
            Some(HashMap::from([
                ("Content-Type".to_string(), ct.to_string()),
                ("Content-Length".to_string(), body.as_ref().map_or(0, |b| b.len()).to_string()),
            ]))
        } else if body.is_some() {
            Some(HashMap::from([
                ("Content-Type".to_string(), "application/octet-stream".to_string()),
                ("Content-Length".to_string(), body.as_ref().map_or(0, |b| b.len()).to_string()),
            ]))
        } else {
            None
        };
        
        Self::request("POST", url, body.as_deref(), headers)
    }
    
    /// Send an HTTP request
    fn request(method: &str, url: &str, body: Option<&[u8]>, headers: Option<HashMap<String, String>>) -> Result<HttpResponse, String> {
        // Parse URL
        let url = url.strip_prefix("http://").unwrap_or(url);
        let url = url.strip_prefix("https://").unwrap_or(url);
        
        let host_path: Vec<&str> = url.splitn(2, '/').collect();
        let host = host_path[0];
        let path = if host_path.len() > 1 { host_path[1] } else { "" };
        
        let host_port: Vec<&str> = host.splitn(2, ':').collect();
        let hostname = host_port[0];
        let port = if host_port.len() > 1 { host_port[1].parse::<u16>().unwrap_or(80) } else { 80 };
        
        // Connect to server
        let mut stream = match TcpStream::connect(format!("{}:{}", hostname, port)) {
            Ok(s) => s,
            Err(e) => return Err(format!("Failed to connect: {}", e)),
        };
        
        // Build request
        let mut request = format!("{} /{} HTTP/1.1\r\n", method, path);
        request.push_str(&format!("Host: {}\r\n", host));
        request.push_str("Connection: close\r\n");
        
        // Add headers
        if let Some(headers) = &headers {
            for (name, value) in headers {
                request.push_str(&format!("{}: {}\r\n", name, value));
            }
        }
        
        // Add body
        if let Some(body) = body {
            request.push_str("\r\n");
            
            // Send request headers
            if let Err(e) = stream.write_all(request.as_bytes()) {
                return Err(format!("Failed to send request: {}", e));
            }
            
            // Send body
            if let Err(e) = stream.write_all(body) {
                return Err(format!("Failed to send request body: {}", e));
            }
        } else {
            // End headers
            request.push_str("\r\n");
            
            // Send request
            if let Err(e) = stream.write_all(request.as_bytes()) {
                return Err(format!("Failed to send request: {}", e));
            }
        }
        
        // Read response
        let mut response = Vec::new();
        if let Err(e) = stream.read_to_end(&mut response) {
            return Err(format!("Failed to read response: {}", e));
        }
        
        // Parse response
        let response_str = String::from_utf8_lossy(&response);
        let parts: Vec<&str> = response_str.splitn(2, "\r\n\r\n").collect();
        
        if parts.len() < 2 {
            return Err("Invalid HTTP response".to_string());
        }
        
        let headers_str = parts[0];
        let body = parts[1].as_bytes().to_vec();
        
        // Parse status line
        let header_lines: Vec<&str> = headers_str.split("\r\n").collect();
        if header_lines.is_empty() {
            return Err("Invalid HTTP response".to_string());
        }
        
        let status_line = header_lines[0];
        let status_parts: Vec<&str> = status_line.split_whitespace().collect();
        if status_parts.len() < 3 {
            return Err("Invalid HTTP status line".to_string());
        }
        
        let status_code = match status_parts[1].parse::<u16>() {
            Ok(code) => code,
            Err(_) => return Err("Invalid HTTP status code".to_string()),
        };
        
        // Parse headers
        let mut headers = HashMap::new();
        for i in 1..header_lines.len() {
            let line = header_lines[i];
            let header_parts: Vec<&str> = line.splitn(2, ": ").collect();
            if header_parts.len() == 2 {
                headers.insert(header_parts[0].to_string(), header_parts[1].to_string());
            }
        }
        
        Ok(HttpResponse {
            status: status_code,
            headers,
            body,
        })
    }
}

/// HTTP response
struct HttpResponse {
    status: u16,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

/// HTTP GET request
#[pyfunction]
fn get(py: Python, url: &str) -> PyResult<Py<PyDict>> {
    match HttpClient::get(url) {
        Ok(response) => {
            let result = PyDict::new(py);
            result.set_item("status", response.status)?;
            
            let headers_dict = PyDict::new(py);
            for (name, value) in response.headers {
                headers_dict.set_item(name, value)?;
            }
            result.set_item("headers", headers_dict)?;
            result.set_item("body", PyBytes::new(py, &response.body))?;
            
            Ok(result.into())
        },
        Err(e) => Err(PyroidError::IoError(format!("HTTP request failed: {}", e)).into()),
    }
}

/// HTTP POST request
#[pyfunction]
fn post(py: Python, url: &str, data: Option<&PyBytes>, json: Option<&PyDict>) -> PyResult<Py<PyDict>> {
    let body_data = if let Some(data_bytes) = data {
        Some(data_bytes.as_bytes().to_vec())
    } else if let Some(json_dict) = json {
        // Convert PyDict to JSON string
        let mut map = std::collections::HashMap::new();
        for (key, value) in json_dict.iter() {
            let key_str = key.extract::<String>()?;
            
            if let Ok(val_str) = value.extract::<String>() {
                map.insert(key_str, serde_json::Value::String(val_str));
            } else if let Ok(val_int) = value.extract::<i64>() {
                map.insert(key_str, serde_json::Value::Number(serde_json::Number::from(val_int)));
            } else if let Ok(val_float) = value.extract::<f64>() {
                if let Some(num) = serde_json::Number::from_f64(val_float) {
                    map.insert(key_str, serde_json::Value::Number(num));
                }
            } else if let Ok(val_bool) = value.extract::<bool>() {
                map.insert(key_str, serde_json::Value::Bool(val_bool));
            } else {
                map.insert(key_str, serde_json::Value::Null);
            }
        }
        
        match serde_json::to_string(&map) {
            Ok(json_string) => Some(json_string.into_bytes()),
            Err(e) => return Err(PyroidError::IoError(format!("Failed to serialize JSON: {}", e)).into()),
        }
    } else {
        None
    };
    
    let content_type = if json.is_some() {
        Some("application/json")
    } else {
        None
    };
    
    match HttpClient::post(url, body_data, content_type) {
        Ok(response) => {
            let result = PyDict::new(py);
            result.set_item("status", response.status)?;
            
            let headers_dict = PyDict::new(py);
            for (name, value) in response.headers {
                headers_dict.set_item(name, value)?;
            }
            result.set_item("headers", headers_dict)?;
            result.set_item("body", PyBytes::new(py, &response.body))?;
            
            Ok(result.into())
        },
        Err(e) => Err(PyroidError::IoError(format!("HTTP request failed: {}", e)).into()),
    }
}

/// Register the network module
pub fn register(py: Python, module: &PyModule) -> PyResult<()> {
    let network_module = PyModule::new(py, "network")?;
    
    network_module.add_function(wrap_pyfunction!(get, network_module)?)?;
    network_module.add_function(wrap_pyfunction!(post, network_module)?)?;
    
    // Add the network module to the parent module
    module.add_submodule(network_module)?;
    
    Ok(())
}