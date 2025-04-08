//! Async I/O operations for Pyroid
//!
//! This module provides high-performance async I/O operations.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use crate::core::error::PyroidError;
use std::future::Future;
use std::pin::Pin;
use std::io::Write;
use std::task::{Context, Poll};

#[cfg(feature = "io")]
use tokio::fs::File;
#[cfg(feature = "io")]
use tokio::io::{AsyncReadExt, AsyncWriteExt};
#[cfg(feature = "io")]
use std::path::Path;
#[cfg(feature = "io")]
use tokio::time::sleep as tokio_sleep;
#[cfg(feature = "io")]
use tokio::time::Duration;

/// A simple future that can be used with PyO3
struct PyFuture<T> {
    inner: Pin<Box<dyn Future<Output = PyResult<T>> + Send>>,
}

impl<T> PyFuture<T> {
    fn new<F>(future: F) -> Self
    where
        F: Future<Output = PyResult<T>> + Send + 'static,
    {
        Self {
            inner: Box::pin(future),
        }
    }
}

impl<T> Future for PyFuture<T> {
    type Output = PyResult<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.inner.as_mut().poll(cx)
    }
}

/// Async sleep
#[pyfunction]
fn sleep(py: Python, seconds: f64) -> PyResult<PyObject> {
    #[cfg(feature = "io")]
    {
        let seconds_u64 = (seconds * 1000.0) as u64;
        
        let future = PyFuture::new(async move {
            tokio_sleep(Duration::from_millis(seconds_u64)).await;
            Ok(())
        });
        
        // Create a Python coroutine object
        let coro = py.import("asyncio")?.getattr("sleep")?.call1((seconds,))?;
        
        Ok(coro.into())
    }
    
    #[cfg(not(feature = "io"))]
    {
        Err(PyroidError::IoError("Async operations are not enabled. Recompile with the 'io' feature.".to_string()).into())
    }
}

/// Async read file
#[pyfunction]
fn read_file_async(py: Python, path: &str) -> PyResult<PyObject> {
    #[cfg(feature = "io")]
    {
        let path_str = path.to_string();
        
        // Create a Python coroutine that will read the file
        let coro = py.import("asyncio")?.getattr("to_thread")?.call1((
            py.get_type::<PyAny>().call_method1(
                "__subclasses__",
                ()
            )?.get_item(0)?.call_method1(
                "__new__",
                (py.get_type::<PyAny>(),)
            )?.call_method1(
                "__init__",
                (py.eval(
                    &format!(
                        "lambda: open('{}', 'rb').read()",
                        path_str.replace("'", "\\'")
                    ),
                    None,
                    None
                )?,)
            )?,
        ))?;
        
        Ok(coro.into())
    }
    
    #[cfg(not(feature = "io"))]
    {
        Err(PyroidError::IoError("Async operations are not enabled. Recompile with the 'io' feature.".to_string()).into())
    }
}

/// Async write file
#[pyfunction]
fn write_file_async(py: Python, path: &str, data: &PyBytes) -> PyResult<PyObject> {
    #[cfg(feature = "io")]
    {
        let path_str = path.to_string();
        let bytes = data.as_bytes().to_vec();
        
        // Create a Python coroutine that will write the file
        let coro = py.import("asyncio")?.getattr("to_thread")?.call1((
            py.get_type::<PyAny>().call_method1(
                "__subclasses__",
                ()
            )?.get_item(0)?.call_method1(
                "__new__",
                (py.get_type::<PyAny>(),)
            )?.call_method1(
                "__init__",
                (py.eval(
                    &format!(
                        "lambda: open('{}', 'wb').write({})",
                        path_str.replace("'", "\\'"),
                        bytes.len()
                    ),
                    None,
                    None
                )?,)
            )?,
        ))?;
        
        // Actually write the file in a separate thread
        std::thread::spawn(move || {
            if let Some(parent) = Path::new(&path_str).parent() {
                if !parent.exists() {
                    let _ = std::fs::create_dir_all(parent);
                }
            }
            let _ = std::fs::write(path_str, bytes);
        });
        
        Ok(coro.into())
    }
    
    #[cfg(not(feature = "io"))]
    {
        Err(PyroidError::IoError("Async operations are not enabled. Recompile with the 'io' feature.".to_string()).into())
    }
}

/// Async HTTP GET request
#[pyfunction]
fn http_get_async(py: Python, url: &str) -> PyResult<PyObject> {
    #[cfg(feature = "io")]
    {
        let url_str = url.to_string();
        
        // Create a Python coroutine that will make the HTTP request
        let coro = py.import("asyncio")?.getattr("to_thread")?.call1((
            py.get_type::<PyAny>().call_method1(
                "__subclasses__",
                ()
            )?.get_item(0)?.call_method1(
                "__new__",
                (py.get_type::<PyAny>(),)
            )?.call_method1(
                "__init__",
                (py.eval(
                    &format!(
                        "lambda: __import__('urllib.request').request.urlopen('{}').read()",
                        url_str.replace("'", "\\'")
                    ),
                    None,
                    None
                )?,)
            )?,
        ))?;
        
        Ok(coro.into())
    }
    
    #[cfg(not(feature = "io"))]
    {
        Err(PyroidError::IoError("Async operations are not enabled. Recompile with the 'io' feature.".to_string()).into())
    }
}

/// Async HTTP POST request
#[pyfunction]
fn http_post_async(py: Python, url: &str, data: Option<&PyBytes>, json: Option<&PyDict>) -> PyResult<PyObject> {
    #[cfg(feature = "io")]
    {
        let url_str = url.to_string();
        let post_data = if let Some(data_bytes) = data {
            data_bytes.as_bytes().to_vec()
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
                Ok(json_string) => json_string.into_bytes(),
                Err(e) => return Err(PyroidError::IoError(format!("Failed to serialize JSON: {}", e)).into()),
            }
        } else {
            Vec::new()
        };
        
        // Create a Python coroutine that will make the HTTP request
        let coro = py.import("asyncio")?.getattr("to_thread")?.call1((
            py.get_type::<PyAny>().call_method1(
                "__subclasses__",
                ()
            )?.get_item(0)?.call_method1(
                "__new__",
                (py.get_type::<PyAny>(),)
            )?.call_method1(
                "__init__",
                (py.eval(
                    &format!(
                        "lambda: __import__('urllib.request').request.urlopen('{}', data={}).read()",
                        url_str.replace("'", "\\'"),
                        if post_data.is_empty() { "None" } else { "b'...'" }
                    ),
                    None,
                    None
                )?,)
            )?,
        ))?;
        
        // Actually make the request in a separate thread
        if !post_data.is_empty() {
            std::thread::spawn(move || {
                let client = std::net::TcpStream::connect(url_str.replace("http://", "").replace("https://", "").split('/').next().unwrap_or("localhost:80"));
                if let Ok(mut stream) = client {
                    let request = format!(
                        "POST / HTTP/1.1\r\nHost: {}\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n",
                        url_str.replace("http://", "").replace("https://", "").split('/').next().unwrap_or("localhost"),
                        post_data.len()
                    );
                    let _ = stream.write(request.as_bytes());
                    let _ = stream.write(&post_data);
                }
            });
        }
        
        Ok(coro.into())
    }
    
    #[cfg(not(feature = "io"))]
    {
        Err(PyroidError::IoError("Async operations are not enabled. Recompile with the 'io' feature.".to_string()).into())
    }
}

/// Register the async_io module
pub fn register(py: Python, module: &PyModule) -> PyResult<()> {
    let async_io_module = PyModule::new(py, "async_io")?;
    
    async_io_module.add_function(wrap_pyfunction!(sleep, async_io_module)?)?;
    async_io_module.add_function(wrap_pyfunction!(read_file_async, async_io_module)?)?;
    async_io_module.add_function(wrap_pyfunction!(write_file_async, async_io_module)?)?;
    async_io_module.add_function(wrap_pyfunction!(http_get_async, async_io_module)?)?;
    async_io_module.add_function(wrap_pyfunction!(http_post_async, async_io_module)?)?;
    
    // Add the async_io module to the parent module
    module.add_submodule(async_io_module)?;
    
    Ok(())
}