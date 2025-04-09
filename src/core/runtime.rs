//! Runtime management for Pyroid
//!
//! This module provides a unified Tokio runtime for all async operations.

use once_cell::sync::Lazy;
use std::sync::Arc;
use tokio::runtime::{Builder, Runtime};
use std::sync::atomic::{AtomicBool, Ordering};
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::thread;

/// Global runtime for all async operations
static GLOBAL_RUNTIME: Lazy<Arc<Runtime>> = Lazy::new(|| {
    Arc::new(
        Builder::new_multi_thread()
            .worker_threads(num_cpus::get())
            .thread_name("pyroid-worker")
            .enable_all()
            .build()
            .expect("Failed to create global Tokio runtime")
    )
});

/// Flag to track if Python's GIL is currently held
static GIL_HELD: AtomicBool = AtomicBool::new(false);

/// Get a reference to the global runtime
pub fn get_runtime() -> Arc<Runtime> {
    GLOBAL_RUNTIME.clone()
}

/// Set the GIL state
pub fn set_gil_state(held: bool) {
    GIL_HELD.store(held, Ordering::SeqCst);
}

/// Check if the GIL is currently held
pub fn is_gil_held() -> bool {
    GIL_HELD.load(Ordering::SeqCst)
}

/// Execute a function on the runtime with GIL awareness
pub fn execute<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    get_runtime().block_on(async {
        // If we're on a Tokio thread already, just run the function
        if tokio::runtime::Handle::try_current().is_ok() {
            return f();
        }
        
        // Otherwise, spawn a new task
        let (tx, rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            let result = f();
            let _ = tx.send(result);
        });
        
        rx.await.expect("Task failed")
    })
}

/// Execute an async function on the runtime with GIL awareness
pub async fn execute_async<F, Fut, R>(f: F) -> R
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = R> + Send,
    R: Send + 'static,
{
    // If we're on a Tokio thread already, just run the function
    if tokio::runtime::Handle::try_current().is_ok() {
        return f().await;
    }
    
    // Otherwise, spawn a new task
    let (tx, rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let result = f().await;
        let _ = tx.send(result);
    });
    
    rx.await.expect("Task failed")
}

/// Python module for runtime management
#[pymodule]
fn runtime(_py: Python, m: &PyModule) -> PyResult<()> {
    /// Initialize the runtime
    #[pyfn(m)]
    fn init() -> PyResult<()> {
        // Force initialization of the global runtime
        let _ = get_runtime();
        Ok(())
    }
    
    /// Get the number of worker threads
    #[pyfn(m)]
    fn get_worker_threads() -> PyResult<usize> {
        Ok(num_cpus::get())
    }
    
    /// Set the GIL state for optimization
    #[pyfn(m)]
    fn set_gil_state_py(held: bool) -> PyResult<()> {
        set_gil_state(held);
        Ok(())
    }
    
    Ok(())
}

/// Register the runtime module
pub fn register(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let runtime_module = PyModule::new(py, "runtime")?;
    
    // Add the runtime module to the parent module
    parent_module.add_submodule(runtime_module)?;
    
    // Initialize the runtime
    let _ = get_runtime();
    
    Ok(())
}