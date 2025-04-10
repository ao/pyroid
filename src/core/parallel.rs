//! Parallel processing utilities for Pyroid
//!
//! This module provides high-performance parallel processing utilities.

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use pyo3::types::{PyList, PyDict};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use crate::core::runtime::get_runtime;

/// A batch processor for parallel operations
#[pyclass]
pub struct BatchProcessor {
    batch_size: usize,
    max_workers: usize,
    adaptive: bool,
    performance_metrics: Arc<Mutex<PerformanceMetrics>>,
}

/// Performance metrics for adaptive batch sizing
struct PerformanceMetrics {
    batch_times: Vec<(usize, f64)>,  // (data_size, processing_time_seconds)
    optimal_batch_size: usize,
}

#[pymethods]
impl BatchProcessor {
    /// Create a new batch processor
    #[new]
    fn new(batch_size: Option<usize>, max_workers: Option<usize>, adaptive: Option<bool>) -> Self {
        let batch_size = batch_size.unwrap_or(1000);
        Self {
            batch_size,
            max_workers: max_workers.unwrap_or(num_cpus::get()),
            adaptive: adaptive.unwrap_or(true),
            performance_metrics: Arc::new(Mutex::new(PerformanceMetrics {
                batch_times: Vec::new(),
                optimal_batch_size: batch_size,
            })),
        }
    }
    
    /// Optimize batch size based on performance metrics
    fn optimize_batch_size(&self, data_size: usize) -> usize {
        if !self.adaptive {
            return self.batch_size;
        }
        
        let metrics_lock = self.performance_metrics.lock().unwrap();
        
        // Use previously optimized batch size or calculate a new one
        let optimal_size = metrics_lock.optimal_batch_size;
        
        // Ensure we don't create too many or too few batches
        let cpus = num_cpus::get();
        let min_batches = cpus * 2; // At least 2 batches per CPU
        let max_batches = cpus * 8; // At most 8 batches per CPU
        
        let calculated_size = data_size / min_batches.max(1);
        let adjusted_size = if data_size / optimal_size > max_batches {
            data_size / max_batches.max(1)
        } else {
            optimal_size
        };
        
        std::cmp::max(1, std::cmp::min(calculated_size, adjusted_size))
    }
    
    /// Map a function over a list of items in parallel
    fn map<'py>(&self, py: Python<'py>, items: &'py PyList, func: PyObject) -> PyResult<&'py PyList> {
        // Convert Python list to Rust Vec
        let items_vec: Vec<PyObject> = items.iter().map(|item| item.into()).collect();
        let total_items = items_vec.len();
        
        // Calculate optimal batch size
        let batch_size = self.optimize_batch_size(total_items);
        
        // Create batches
        let batches: Vec<Vec<PyObject>> = items_vec
            .chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        
        // Process batches in parallel
        let counter = Arc::new(AtomicUsize::new(0));
        let results: Vec<PyResult<Vec<PyObject>>> = batches
            .par_iter()
            .map(|batch| {
                // Update progress counter
                let current = counter.fetch_add(1, Ordering::SeqCst) + 1;
                if total_items > 1000 && current % 10 == 0 {
                    Python::with_gil(|py| {
                        let locals = PyDict::new(py);
                        locals.set_item("current", current)?;
                        locals.set_item("total", batches.len())?;
                        locals.set_item("percent", (current as f64 / batches.len() as f64) * 100.0)?;
                        
                        py.eval(
                            "print(f'Processing batch {current}/{total} ({percent:.1f}%)', end='\\r')",
                            None,
                            Some(locals),
                        )?;
                        Ok::<_, PyErr>(())
                    }).ok();
                }
                
                // Process batch
                Python::with_gil(|py| {
                    let batch_list = PyList::new(py, batch);
                    let result = func.call1(py, (batch_list,))?;
                    
                    // Convert result back to Vec<PyObject>
                    let result_list = result.extract::<&PyList>(py)?;
                    let result_vec: Vec<PyObject> = result_list.iter().map(|item| item.into()).collect();
                    
                    Ok(result_vec)
                })
            })
            .collect();
        
        // Update batch size metrics
        if self.adaptive {
            let mut metrics = self.performance_metrics.lock().unwrap();
            metrics.optimal_batch_size = batch_size;
        }
        
        // Flatten results
        let mut all_results = Vec::new();
        for result in results {
            match result {
                Ok(batch_results) => all_results.extend(batch_results),
                Err(e) => return Err(e),
            }
        }
        
        // Convert back to Python list
        let result_list = PyList::new(py, &all_results);
        Ok(result_list)
    }
    
    /// Filter a list of items in parallel
    fn filter<'py>(&self, py: Python<'py>, items: &'py PyList, func: PyObject) -> PyResult<&'py PyList> {
        // Create a wrapper function that applies the filter to each item
        let locals = PyDict::new(py);
        locals.set_item("func", func)?;
        
        let filter_func = py.eval(
            "lambda batch: [item for item in batch if func(item)]",
            None,
            Some(locals),
        )?;
        
        // Process the list using the wrapper function
        self.map(py, items, filter_func.into())
    }
    
    /// Sort a list of items in parallel
    fn sort<'py>(&self, py: Python<'py>, items: &'py PyList, key: Option<PyObject>, reverse: Option<bool>) -> PyResult<&'py PyList> {
        let reverse = reverse.unwrap_or(false);
        
        // Convert Python list to Rust Vec
        let mut items_vec: Vec<PyObject> = items.iter().map(|item| item.into()).collect();
        
        // Sort the items
        if let Some(key_func) = key {
            // Sort with key function
            // First, compute keys for all items
            let keys: Result<Vec<PyObject>, PyErr> = items_vec
                .iter()
                .map(|item| {
                    Python::with_gil(|py| {
                        key_func.call1(py, (item.clone_ref(py),)).map(|k| k.into())
                    })
                })
                .collect();
            
            let keys = keys?;
            
            // Create pairs of (key, item)
            let mut pairs: Vec<(PyObject, PyObject)> = keys.into_iter().zip(items_vec).collect();
            
            // Sort the pairs based on the keys
            pairs.sort_by(|(k1, _), (k2, _)| {
                Python::with_gil(|py| {
                    let result = k1.as_ref(py).rich_compare(k2.as_ref(py), pyo3::basic::CompareOp::Lt);
                    match result {
                        Ok(result) => {
                            if result.is_true().unwrap_or(false) {
                                std::cmp::Ordering::Less
                            } else {
                                let result = k1.as_ref(py).rich_compare(k2.as_ref(py), pyo3::basic::CompareOp::Gt);
                                if result.unwrap().is_true().unwrap_or(false) {
                                    std::cmp::Ordering::Greater
                                } else {
                                    std::cmp::Ordering::Equal
                                }
                            }
                        }
                        Err(_) => std::cmp::Ordering::Equal,
                    }
                })
            });
            
            // Extract the sorted items
            items_vec = pairs.into_iter().map(|(_, item)| item).collect();
            
            // Reverse if needed
            if reverse {
                items_vec.reverse();
            }
        } else {
            // Sort without key function
            items_vec.sort_by(|a, b| {
                Python::with_gil(|py| {
                    let result = a.as_ref(py).rich_compare(b.as_ref(py), pyo3::basic::CompareOp::Lt);
                    match result {
                        Ok(result) => {
                            if result.is_true().unwrap_or(false) {
                                std::cmp::Ordering::Less
                            } else {
                                let result = a.as_ref(py).rich_compare(b.as_ref(py), pyo3::basic::CompareOp::Gt);
                                if result.unwrap().is_true().unwrap_or(false) {
                                    std::cmp::Ordering::Greater
                                } else {
                                    std::cmp::Ordering::Equal
                                }
                            }
                        }
                        Err(_) => std::cmp::Ordering::Equal,
                    }
                })
            });
            
            // Reverse if needed
            if reverse {
                items_vec.reverse();
            }
        }
        
        // Convert back to Python list
        let result_list = PyList::new(py, &items_vec);
        Ok(result_list)
    }
}

/// Process a list of items in parallel
#[pyfunction]
fn parallel_map<'py>(py: Python<'py>, items: &'py PyList, func: PyObject, batch_size: Option<usize>) -> PyResult<&'py PyList> {
    let processor = BatchProcessor::new(batch_size, None, Some(true));
    processor.map(py, items, func)
}

/// Filter a list of items in parallel
#[pyfunction]
fn parallel_filter<'py>(py: Python<'py>, items: &'py PyList, func: PyObject, batch_size: Option<usize>) -> PyResult<&'py PyList> {
    let processor = BatchProcessor::new(batch_size, None, Some(true));
    processor.filter(py, items, func)
}

/// Sort a list of items in parallel
#[pyfunction]
fn parallel_sort<'py>(py: Python<'py>, items: &'py PyList, key: Option<PyObject>, reverse: Option<bool>, batch_size: Option<usize>) -> PyResult<&'py PyList> {
    let processor = BatchProcessor::new(batch_size, None, Some(true));
    processor.sort(py, items, key, reverse)
}

/// Register the parallel module
pub fn register(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let parallel_module = PyModule::new(py, "parallel")?;
    
    parallel_module.add_class::<BatchProcessor>()?;
    parallel_module.add_function(wrap_pyfunction!(parallel_map, parallel_module)?)?;
    parallel_module.add_function(wrap_pyfunction!(parallel_filter, parallel_module)?)?;
    parallel_module.add_function(wrap_pyfunction!(parallel_sort, parallel_module)?)?;
    
    // Add the parallel module to the parent module
    parent_module.add_submodule(parallel_module)?;
    
    Ok(())
}