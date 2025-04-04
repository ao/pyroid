//! Async operations
//!
//! This module provides high-performance async operations using Tokio.

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{PyDict, PyList, PyBytes, PyTuple};
use tokio::runtime::{Builder, Runtime};
use tokio::sync::mpsc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::fs::File;
use tokio::time::{sleep, Duration};
use std::sync::Arc;
use futures::stream::{self, StreamExt};

// Global runtime for reuse
thread_local! {
    static TOKIO_RUNTIME: Runtime = Builder::new_multi_thread()
        .worker_threads(num_cpus::get())
        .enable_all()
        .build()
        .expect("Failed to create Tokio runtime");
}

/// Async HTTP client
///
/// This class provides methods for making HTTP requests asynchronously.
#[pyclass]
struct AsyncClient {
    client: Arc<reqwest::Client>,
    runtime: Runtime,
}

#[pymethods]
impl AsyncClient {
    /// Create a new AsyncClient
    #[new]
    fn new() -> PyResult<Self> {
        let client = Arc::new(reqwest::Client::new());
        let runtime = Builder::new_multi_thread()
            .worker_threads(num_cpus::get())
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create Tokio runtime: {}", e)))?;
        
        Ok(AsyncClient { client, runtime })
    }
    
    /// Fetch a URL asynchronously
    ///
    /// Args:
    ///     url: The URL to fetch
    ///
    /// Returns:
    ///     A dictionary with status and text
    fn fetch<'py>(&self, py: Python<'py>, url: String) -> PyResult<&'py PyAny> {
        let client = self.client.clone();
        
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let response = client.get(&url).send().await
                .map_err(|e| PyRuntimeError::new_err(format!("Request failed: {}", e)))?;
                
            let status = response.status().as_u16();
            let text = response.text().await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to get response text: {}", e)))?;
                
            Ok(Python::with_gil(|py| {
                let dict = PyDict::new(py);
                dict.set_item("status", status).unwrap();
                dict.set_item("text", text).unwrap();
                dict.to_object(py)
            }))
        })
    }
    
    /// Fetch multiple URLs concurrently
    ///
    /// Args:
    ///     urls: A list of URLs to fetch
    ///     concurrency: Maximum number of concurrent requests (default: 10)
    ///
    /// Returns:
    ///     A dictionary mapping URLs to their responses
    fn fetch_many<'py>(&self, py: Python<'py>, urls: Vec<String>, concurrency: Option<usize>) -> PyResult<&'py PyAny> {
        let client = self.client.clone();
        let concurrency = concurrency.unwrap_or(10);
        
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let results = stream::iter(urls)
                .map(|url| {
                    let client = client.clone();
                    async move {
                        match client.get(&url).send().await {
                            Ok(resp) => {
                                let status = resp.status().as_u16();
                                match resp.text().await {
                                    Ok(text) => (url, Ok((status, text))),
                                    Err(e) => (url, Err(format!("Failed to get response text: {}", e))),
                                }
                            },
                            Err(e) => (url, Err(format!("Request failed: {}", e))),
                        }
                    }
                })
                .buffer_unordered(concurrency)
                .collect::<Vec<_>>()
                .await;
                
            Ok(Python::with_gil(|py| {
                let dict = PyDict::new(py);
                for (url, result) in results {
                    match result {
                        Ok((status, text)) => {
                            let response_dict = PyDict::new(py);
                            response_dict.set_item("status", status).unwrap();
                            response_dict.set_item("text", text).unwrap();
                            dict.set_item(url, response_dict).unwrap();
                        },
                        Err(e) => {
                            dict.set_item(url, e).unwrap();
                        }
                    }
                }
                dict.to_object(py)
            }))
        })
    }
    
    /// Download a file asynchronously
    ///
    /// Args:
    ///     url: The URL to download from
    ///     path: The path to save the file to
    ///
    /// Returns:
    ///     A dictionary with success status and path
    fn download_file<'py>(&self, py: Python<'py>, url: String, path: String) -> PyResult<&'py PyAny> {
        let client = self.client.clone();
        
        pyo3_asyncio::tokio::future_into_py(py, async move {
            // Create a stream from the URL
            let response = client.get(&url).send().await
                .map_err(|e| PyRuntimeError::new_err(format!("Request failed: {}", e)))?;
                
            if !response.status().is_success() {
                return Err(PyRuntimeError::new_err(format!(
                    "Failed to download file: HTTP {}", response.status()
                )));
            }
            
            // Create the file
            let mut file = File::create(&path).await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create file: {}", e)))?;
                
            // Stream the response body to the file
            let mut stream = response.bytes_stream();
            while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to download chunk: {}", e)))?;
                    
                file.write_all(&chunk).await
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to write to file: {}", e)))?;
            }
            
            file.flush().await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to flush file: {}", e)))?;
                
            Ok(Python::with_gil(|py| {
                let dict = PyDict::new(py);
                dict.set_item("success", true).unwrap();
                dict.set_item("path", path).unwrap();
                dict.to_object(py)
            }))
        })
    }
    
    /// Set a request timeout
    ///
    /// Args:
    ///     timeout_seconds: The timeout in seconds
    ///
    /// Returns:
    ///     A new AsyncClient with the specified timeout
    fn with_timeout(&self, timeout_seconds: f64) -> PyResult<Self> {
        let timeout = Duration::from_secs_f64(timeout_seconds);
        
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create client: {}", e)))?;
            
        let runtime = Builder::new_multi_thread()
            .worker_threads(num_cpus::get())
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create Tokio runtime: {}", e)))?;
            
        Ok(AsyncClient {
            client: Arc::new(client),
            runtime,
        })
    }
}

/// Async communication channel
///
/// This class provides methods for asynchronous communication between tasks.
#[pyclass]
struct AsyncChannel {
    sender: mpsc::Sender<PyObject>,
    receiver: Option<mpsc::Receiver<PyObject>>,
    runtime: Runtime,
}

#[pymethods]
impl AsyncChannel {
    /// Create a new AsyncChannel
    ///
    /// Args:
    ///     capacity: The channel capacity (default: 100)
    #[new]
    fn new(capacity: Option<usize>) -> PyResult<Self> {
        let capacity = capacity.unwrap_or(100);
        let (sender, receiver) = mpsc::channel(capacity);
        let runtime = Builder::new_multi_thread()
            .worker_threads(num_cpus::get())
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create Tokio runtime: {}", e)))?;
            
        Ok(AsyncChannel {
            sender,
            receiver: Some(receiver),
            runtime,
        })
    }
    
    /// Send a value to the channel
    ///
    /// Args:
    ///     value: The value to send
    fn send<'py>(&self, py: Python<'py>, value: PyObject) -> PyResult<&'py PyAny> {
        let sender = self.sender.clone();
        
        pyo3_asyncio::tokio::future_into_py(py, async move {
            sender.send(value).await
                .map_err(|_| PyRuntimeError::new_err("Failed to send value: channel closed"))?;
            Ok(Python::with_gil(|py| py.None()))
        })
    }
    
    /// Receive a value from the channel
    fn receive<'py>(&mut self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let mut receiver = self.receiver.take()
            .ok_or_else(|| PyRuntimeError::new_err("Receiver has been moved"))?;
            
        pyo3_asyncio::tokio::future_into_py(py, async move {
            match receiver.recv().await {
                Some(value) => {
                    // Return the value
                    Python::with_gil(|_py| {
                        // Store the receiver back in self in a real implementation
                        // For now, we'll just return the value
                        Ok(value)
                    })
                },
                None => {
                    // Channel is closed
                    Python::with_gil(|py| {
                        Ok(py.None())
                    })
                }
            }
        })
    }
    
    /// Close the channel
    fn close(&mut self) -> PyResult<()> {
        // Drop the receiver to close the channel
        self.receiver = None;
        Ok(())
    }
}

/// Async file reader
///
/// This class provides methods for asynchronous file operations.
#[pyclass]
struct AsyncFileReader {
    path: String,
    runtime: Runtime,
}

#[pymethods]
impl AsyncFileReader {
    /// Create a new AsyncFileReader
    ///
    /// Args:
    ///     path: The path to the file
    #[new]
    fn new(path: String) -> PyResult<Self> {
        let runtime = Builder::new_multi_thread()
            .worker_threads(num_cpus::get())
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create Tokio runtime: {}", e)))?;
            
        Ok(AsyncFileReader { path, runtime })
    }
    
    /// Read the entire file asynchronously
    fn read_all<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let path = self.path.clone();
        
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let mut file = File::open(&path).await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to open file: {}", e)))?;
                
            let mut contents = Vec::new();
            file.read_to_end(&mut contents).await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to read file: {}", e)))?;
                
            Ok(Python::with_gil(|py| {
                PyBytes::new(py, &contents).to_object(py)
            }))
        })
    }
    
    /// Read the file line by line asynchronously
    fn read_lines<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let path = self.path.clone();
        
        pyo3_asyncio::tokio::future_into_py(py, async move {
            use tokio::io::{BufReader, AsyncBufReadExt};
            
            let file = File::open(&path).await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to open file: {}", e)))?;
                
            let reader = BufReader::new(file);
            let mut lines = reader.lines();
            
            let mut result = Vec::new();
            while let Some(line) = lines.next_line().await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to read line: {}", e)))? {
                result.push(line);
            }
            
            Ok(Python::with_gil(|py| {
                let py_list = PyList::empty(py);
                for line in result {
                    py_list.append(line).unwrap();
                }
                py_list.to_object(py)
            }))
        })
    }
}

/// Sleep asynchronously
///
/// Args:
///     seconds: The number of seconds to sleep
#[pyfunction]
fn async_sleep<'py>(py: Python<'py>, seconds: f64) -> PyResult<&'py PyAny> {
    let duration = Duration::from_secs_f64(seconds);
    
    pyo3_asyncio::tokio::future_into_py(py, async move {
        sleep(duration).await;
        Ok(Python::with_gil(|py| py.None()))
    })
}

/// Run multiple async tasks concurrently and wait for all to complete
///
/// Args:
///     coroutines: A list of coroutines to run
///
/// Returns:
///     A list of results from the coroutines
#[pyfunction]
fn gather<'py>(py: Python<'py>, coroutines: &'py PyList) -> PyResult<&'py PyAny> {
    // Use Python's asyncio.gather directly
    let asyncio = py.import("asyncio")?;
    let gather_fn = asyncio.getattr("gather")?;
    
    // Create a Python code snippet to call gather with the list
    let locals = PyDict::new(py);
    locals.set_item("coroutines", coroutines)?;
    locals.set_item("gather", gather_fn)?;
    
    // Execute Python code to call gather with unpacked list
    py.eval("gather(*coroutines)", None, Some(locals))
}

/// Register the async operations module
pub fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AsyncClient>()?;
    m.add_class::<AsyncChannel>()?;
    m.add_class::<AsyncFileReader>()?;
    m.add_function(wrap_pyfunction!(async_sleep, m)?)?;
    m.add_function(wrap_pyfunction!(gather, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_async_client_creation() {
        Python::with_gil(|py| {
            let client = AsyncClient::new();
            assert!(client.is_ok());
        });
    }
    
    #[test]
    fn test_async_channel_creation() {
        Python::with_gil(|py| {
            let channel = AsyncChannel::new(Some(10));
            assert!(channel.is_ok());
        });
    }
}