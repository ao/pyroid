//! Configuration system for Pyroid
//!
//! This module provides a unified configuration system for Pyroid.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyBool};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::cell::RefCell;
use crate::core::error::PyroidError;

/// Global configuration store
static GLOBAL_CONFIG: Mutex<Option<Arc<Config>>> = Mutex::new(None);

/// Thread-local configuration store
thread_local! {
    static THREAD_CONFIG: RefCell<Option<Arc<Config>>> = RefCell::new(None);
}

/// Configuration options
#[derive(Debug, Clone)]
pub struct Config {
    /// Configuration options
    options: HashMap<String, ConfigValue>,
}

/// Configuration value types
#[derive(Debug, Clone)]
pub enum ConfigValue {
    /// Boolean value
    Bool(bool),
    /// Integer value
    Int(i64),
    /// Float value
    Float(f64),
    /// String value
    String(String),
}

impl Config {
    /// Create a new configuration
    pub fn new() -> Self {
        Self {
            options: HashMap::new(),
        }
    }
    
    /// Set a boolean option
    pub fn set_bool(&mut self, key: &str, value: bool) {
        self.options.insert(key.to_string(), ConfigValue::Bool(value));
    }
    
    /// Set an integer option
    pub fn set_int(&mut self, key: &str, value: i64) {
        self.options.insert(key.to_string(), ConfigValue::Int(value));
    }
    
    /// Set a float option
    pub fn set_float(&mut self, key: &str, value: f64) {
        self.options.insert(key.to_string(), ConfigValue::Float(value));
    }
    
    /// Set a string option
    pub fn set_string(&mut self, key: &str, value: &str) {
        self.options.insert(key.to_string(), ConfigValue::String(value.to_string()));
    }
    
    /// Get a boolean option
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.options.get(key) {
            Some(ConfigValue::Bool(value)) => Some(*value),
            _ => None,
        }
    }
    
    /// Get an integer option
    pub fn get_int(&self, key: &str) -> Option<i64> {
        match self.options.get(key) {
            Some(ConfigValue::Int(value)) => Some(*value),
            _ => None,
        }
    }
    
    /// Get a float option
    pub fn get_float(&self, key: &str) -> Option<f64> {
        match self.options.get(key) {
            Some(ConfigValue::Float(value)) => Some(*value),
            _ => None,
        }
    }
    
    /// Get a string option
    pub fn get_string(&self, key: &str) -> Option<String> {
        match self.options.get(key) {
            Some(ConfigValue::String(value)) => Some(value.clone()),
            _ => None,
        }
    }
    
    /// Merge with another configuration
    pub fn merge(&mut self, other: &Config) {
        for (key, value) in &other.options {
            self.options.insert(key.clone(), value.clone());
        }
    }
    
    /// Create from Python dictionary
    pub fn from_pydict(dict: &PyDict) -> PyResult<Self> {
        let mut config = Config::new();
        
        for (key, value) in dict.iter() {
            let key_str = key.extract::<String>()?;
            
            if let Ok(bool_val) = value.extract::<bool>() {
                config.set_bool(&key_str, bool_val);
            } else if let Ok(int_val) = value.extract::<i64>() {
                config.set_int(&key_str, int_val);
            } else if let Ok(float_val) = value.extract::<f64>() {
                config.set_float(&key_str, float_val);
            } else if let Ok(str_val) = value.extract::<String>() {
                config.set_string(&key_str, &str_val);
            } else {
                return Err(PyroidError::ConversionError(format!(
                    "Unsupported configuration value type for key: {}", key_str
                )).into());
            }
        }
        
        Ok(config)
    }
    
    /// Convert to Python dictionary
    pub fn to_pydict(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        
        for (key, value) in &self.options {
            match value {
                ConfigValue::Bool(val) => dict.set_item(key, *val)?,
                ConfigValue::Int(val) => dict.set_item(key, *val)?,
                ConfigValue::Float(val) => dict.set_item(key, *val)?,
                ConfigValue::String(val) => dict.set_item(key, val)?,
            }
        }
        
        Ok(dict.into())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the current configuration
pub fn get_config() -> Arc<Config> {
    // First check thread-local config
    let thread_config = THREAD_CONFIG.with(|cell| cell.borrow().clone());
    
    if let Some(config) = thread_config {
        return config;
    }
    
    // Fall back to global config
    let global_config = GLOBAL_CONFIG.lock().unwrap().clone();
    
    if let Some(config) = global_config {
        return config;
    }
    
    // Create default config if none exists
    let default_config = Arc::new(Config::new());
    *GLOBAL_CONFIG.lock().unwrap() = Some(default_config.clone());
    
    default_config
}

/// Set the global configuration
pub fn set_global_config(config: Config) {
    *GLOBAL_CONFIG.lock().unwrap() = Some(Arc::new(config));
}

/// Set the thread-local configuration
pub fn set_thread_config(config: Config) {
    THREAD_CONFIG.with(|cell| {
        *cell.borrow_mut() = Some(Arc::new(config));
    });
}

/// Clear the thread-local configuration
pub fn clear_thread_config() {
    THREAD_CONFIG.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

/// Python Config class
#[pyclass(name = "Config")]
#[derive(Clone)]
pub struct PyConfig {
    config: Arc<Config>,
}

#[pymethods]
impl PyConfig {
    #[new]
    fn new(dict: Option<&PyDict>) -> PyResult<Self> {
        let config = if let Some(d) = dict {
            Config::from_pydict(d)?
        } else {
            Config::new()
        };
        
        Ok(Self {
            config: Arc::new(config),
        })
    }
    
    /// Get a configuration value
    fn get(&self, py: Python, key: &str) -> PyObject {
        if let Some(val) = self.config.get_bool(key) {
            val.into_py(py)
        } else if let Some(val) = self.config.get_int(key) {
            val.into_py(py)
        } else if let Some(val) = self.config.get_float(key) {
            val.into_py(py)
        } else if let Some(val) = self.config.get_string(key) {
            val.into_py(py)
        } else {
            py.None()
        }
    }
    
    /// Set a configuration value
    fn set(&mut self, key: &str, value: &PyAny) -> PyResult<()> {
        // We need to clone the config to modify it
        let mut new_config = (*self.config).clone();
        
        if let Ok(val) = value.extract::<bool>() {
            new_config.set_bool(key, val);
        } else if let Ok(val) = value.extract::<i64>() {
            new_config.set_int(key, val);
        } else if let Ok(val) = value.extract::<f64>() {
            new_config.set_float(key, val);
        } else if let Ok(val) = value.extract::<String>() {
            new_config.set_string(key, &val);
        } else {
            return Err(PyroidError::ConversionError(format!(
                "Unsupported configuration value type for key: {}", key
            )).into());
        }
        
        self.config = Arc::new(new_config);
        Ok(())
    }
    
    /// Convert to dictionary
    fn to_dict(&self, py: Python) -> PyResult<Py<PyDict>> {
        self.config.to_pydict(py)
    }
    
    /// Set as global configuration
    fn set_as_global(&self) -> PyResult<()> {
        set_global_config((*self.config).clone());
        Ok(())
    }
    
    /// Set as thread-local configuration
    fn set_as_thread_local(&self) -> PyResult<()> {
        set_thread_config((*self.config).clone());
        Ok(())
    }
    
    /// Clear thread-local configuration
    #[staticmethod]
    fn clear_thread_local() -> PyResult<()> {
        clear_thread_config();
        Ok(())
    }
}

/// Python context manager for configuration
#[pyclass(name = "ConfigContext")]
pub struct ConfigContext {
    previous_config: Option<Arc<Config>>,
    new_config: Arc<Config>,
}

#[pymethods]
impl ConfigContext {
    #[new]
    fn new(config: &PyConfig) -> Self {
        Self {
            previous_config: None,
            new_config: config.config.clone(),
        }
    }
    
    fn __enter__(&mut self, py: Python) -> PyResult<PyObject> {
        // Save the current thread-local config
        THREAD_CONFIG.with(|cell| {
            self.previous_config = cell.borrow().clone();
        });
        
        // Set the new config
        set_thread_config((*self.new_config).clone());
        
        Ok(py.None())
    }
    
    fn __exit__(
        &mut self,
        _exc_type: &PyAny,
        _exc_value: &PyAny,
        _traceback: &PyAny,
    ) -> PyResult<bool> {
        // Restore the previous config
        THREAD_CONFIG.with(|cell| {
            *cell.borrow_mut() = self.previous_config.clone();
        });
        
        Ok(false) // Don't suppress exceptions
    }
}

/// Register the config module
pub fn register(py: Python, module: &PyModule) -> PyResult<()> {
    let config_module = PyModule::new(py, "config")?;
    
    // Add the PyConfig class as "Config" to the module
    module.add_class::<PyConfig>()?;
    
    // Also add it to the config submodule
    config_module.add_class::<PyConfig>()?;
    config_module.add_class::<ConfigContext>()?;
    
    // Add the config module to the parent module
    module.add_submodule(config_module)?;
    
    Ok(())
}