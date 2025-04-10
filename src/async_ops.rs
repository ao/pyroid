//! Async operations
//!
//! This module provides high-performance async operations using Tokio.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Async HTTP client
///
/// This class provides methods for making HTTP requests asynchronously.
#[pyclass]
struct AsyncClient {
    #[pyo3(get)]
    timeout: Option<f64>,
    concurrency: usize,
    adaptive_concurrency: bool,
}

#[pymethods]
impl AsyncClient {
    /// Create a new AsyncClient
    #[new]
    fn new(timeout: Option<f64>, concurrency: Option<usize>, adaptive_concurrency: Option<bool>) -> PyResult<Self> {
        Ok(AsyncClient { 
            timeout,
            concurrency: concurrency.unwrap_or(10),
            adaptive_concurrency: adaptive_concurrency.unwrap_or(true),
        })
    }
    
    /// Fetch a URL asynchronously
    ///
    /// Args:
    ///     url: The URL to fetch
    ///
    /// Returns:
    ///     A dictionary with status and text
    fn fetch(&self, py: Python, url: String) -> PyResult<PyObject> {
        // Extract host from URL for connection pooling
        let host = extract_host(&url);
        
        // Call the fetch_url function with connection pooling
        let locals = PyDict::new(py);
        locals.set_item("url", url)?;
        locals.set_item("timeout", self.timeout.unwrap_or(30.0))?;
        locals.set_item("host", host)?;
        
        // Use optimized fetch with connection pooling
        let code = r#"
import asyncio
from pyroid.async_helpers import fetch_url_optimized

async def _fetch():
    return await fetch_url_optimized(url, timeout=timeout, host=host)

asyncio.run(_fetch())
"#;
        
        let result = py.eval(code, None, Some(locals))?;
        Ok(result.into())
    }
    
    /// Extract metrics about the connection pool
    fn connection_pool_stats(&self, py: Python) -> PyResult<PyObject> {
        // Create a dictionary with stats
        let stats = PyDict::new(py);
        stats.set_item("concurrency", self.concurrency)?;
        stats.set_item("adaptive", self.adaptive_concurrency)?;
        
        // Add more stats in the future
        
        Ok(stats.into())
    }
    
    /// Fetch multiple URLs concurrently
    ///
    /// Args:
    ///     urls: A list of URLs to fetch
    ///     concurrency: Maximum number of concurrent requests (default: 10)
    ///
    /// Returns:
    ///     A dictionary mapping URLs to their responses
    fn fetch_many(&self, py: Python, urls: Vec<String>, concurrency: Option<usize>) -> PyResult<PyObject> {
        // Use the provided concurrency or the default
        let concurrency = concurrency.unwrap_or(self.concurrency);
        
        // Call the optimized fetch_many function
        let locals = PyDict::new(py);
        locals.set_item("urls", urls)?;
        locals.set_item("concurrency", concurrency)?;
        locals.set_item("timeout", self.timeout.unwrap_or(30.0))?;
        
        // Use optimized fetch_many with connection pooling
        let code = r#"
import asyncio
from pyroid.async_helpers import fetch_many_optimized

async def _fetch_many():
    return await fetch_many_optimized(urls, concurrency=concurrency, timeout=timeout)

asyncio.run(_fetch_many())
"#;
        
        let result = py.eval(code, None, Some(locals))?;
        Ok(result.into())
    }
    
    /// Download a file asynchronously
    ///
    /// Args:
    ///     url: The URL to download from
    ///     path: The path to save the file to
    ///
    /// Returns:
    ///     A dictionary with success status and path
    fn download_file(&self, py: Python, url: String, path: String) -> PyResult<PyObject> {
        // Import the async_bridge module
        let async_bridge = PyModule::import(py, "pyroid.async_bridge")?;
        
        // Call the download_file function
        let result = async_bridge.getattr("download_file")?.call1((url, path))?;
        
        Ok(result.into())
    }
    
    /// Set a request timeout
    ///
    /// Args:
    ///     timeout_seconds: The timeout in seconds
    ///
    /// Returns:
    ///     A new AsyncClient with the specified timeout
    fn with_timeout(&self, timeout_seconds: f64) -> PyResult<Self> {
        Ok(AsyncClient {
            timeout: Some(timeout_seconds),
            concurrency: self.concurrency,
            adaptive_concurrency: self.adaptive_concurrency,
        })
    }
    
    /// Set the concurrency level
    fn with_concurrency(&self, concurrency: usize) -> PyResult<Self> {
        Ok(AsyncClient {
            timeout: self.timeout,
            concurrency,
            adaptive_concurrency: self.adaptive_concurrency,
        })
    }
    
    /// Enable or disable adaptive concurrency
    fn with_adaptive_concurrency(&self, adaptive: bool) -> PyResult<Self> {
        Ok(AsyncClient {
            timeout: self.timeout,
            concurrency: self.concurrency,
            adaptive_concurrency: adaptive,
        })
    }
}

/// Async file reader
///
/// This class provides methods for asynchronous file operations.
#[pyclass]
struct AsyncFileReader {
    path: String,
}

#[pymethods]
impl AsyncFileReader {
    /// Create a new AsyncFileReader
    ///
    /// Args:
    ///     path: The path to the file
    #[new]
    fn new(path: String) -> PyResult<Self> {
        Ok(AsyncFileReader { path })
    }
    
    /// Read the entire file asynchronously
    fn read_all(&self, py: Python) -> PyResult<PyObject> {
        // Import the async_bridge module
        let async_bridge = PyModule::import(py, "pyroid.async_bridge")?;
        
        // Call the read_file function
        let result = async_bridge.getattr("read_file")?.call1((self.path.clone(),))?;
        
        Ok(result.into())
    }
    
    /// Read the file line by line asynchronously
    fn read_lines(&self, py: Python) -> PyResult<PyObject> {
        // Import the async_bridge module
        let async_bridge = PyModule::import(py, "pyroid.async_bridge")?;
        
        // Call the read_file_lines function
        let result = async_bridge.getattr("read_file_lines")?.call1((self.path.clone(),))?;
        
        Ok(result.into())
    }
}

/// Sleep asynchronously
///
/// Args:
///     seconds: The number of seconds to sleep
#[pyfunction]
fn async_sleep(py: Python, seconds: f64) -> PyResult<()> {
    // Import the async_bridge module
    let async_bridge = PyModule::import(py, "pyroid.async_bridge")?;
    
    // Call the sleep function
    async_bridge.getattr("sleep")?.call1((seconds,))?;
    
    Ok(())
}

/// Extract host from URL
fn extract_host(url: &str) -> String {
    if let Some(start) = url.find("://") {
        let host_start = start + 3;
        if let Some(host_end) = url[host_start..].find('/') {
            return url[host_start..host_start + host_end].to_string();
        } else {
            return url[host_start..].to_string();
        }
    }
    
    // Default to the whole URL if we can't parse it
    url.to_string()
}

/// Register the async operations module
pub fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AsyncClient>()?;
    m.add_class::<AsyncFileReader>()?;
    m.add_function(wrap_pyfunction!(async_sleep, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_async_client_creation() {
        Python::with_gil(|py| {
            let client = AsyncClient::new(None, None, None);
            assert!(client.is_ok());
        });
    }
    
    #[test]
    fn test_extract_host() {
        assert_eq!(extract_host("https://example.com/path"), "example.com");
        assert_eq!(extract_host("http://test.org/foo/bar"), "test.org");
        assert_eq!(extract_host("https://api.example.net"), "api.example.net");
        assert_eq!(extract_host("invalid-url"), "invalid-url");
    }
}