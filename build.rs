//! Build script for Pyroid
//!
//! This script configures the build process for Pyroid.

use std::env;
extern crate num_cpus;
use std::path::Path;

fn main() {
    // Configure PyO3
    pyo3_build_config::add_extension_module_link_args();
    
    // Print cargo configuration
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=src/");
    
    // Set up feature-specific configurations
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir).join("config.rs");
    
    let mut config = String::new();
    
    // Add feature flags
    config.push_str("/// Feature flags\n");
    config.push_str("pub mod features {\n");
    
    #[cfg(feature = "math")]
    config.push_str("    pub const MATH_ENABLED: bool = true;\n");
    #[cfg(not(feature = "math"))]
    config.push_str("    pub const MATH_ENABLED: bool = false;\n");
    
    #[cfg(feature = "text")]
    config.push_str("    pub const TEXT_ENABLED: bool = true;\n");
    #[cfg(not(feature = "text"))]
    config.push_str("    pub const TEXT_ENABLED: bool = false;\n");
    
    #[cfg(feature = "data")]
    config.push_str("    pub const DATA_ENABLED: bool = true;\n");
    #[cfg(not(feature = "data"))]
    config.push_str("    pub const DATA_ENABLED: bool = false;\n");
    
    #[cfg(feature = "io")]
    config.push_str("    pub const IO_ENABLED: bool = true;\n");
    #[cfg(not(feature = "io"))]
    config.push_str("    pub const IO_ENABLED: bool = false;\n");
    
    #[cfg(feature = "image")]
    config.push_str("    pub const IMAGE_ENABLED: bool = true;\n");
    #[cfg(not(feature = "image"))]
    config.push_str("    pub const IMAGE_ENABLED: bool = false;\n");
    
    #[cfg(feature = "ml")]
    config.push_str("    pub const ML_ENABLED: bool = true;\n");
    #[cfg(not(feature = "ml"))]
    config.push_str("    pub const ML_ENABLED: bool = false;\n");
    
    config.push_str("}\n");
    
    // Add platform-specific configurations
    config.push_str("\n/// Platform information\n");
    config.push_str("pub mod platform {\n");
    
    #[cfg(target_os = "windows")]
    config.push_str("    pub const OS: &str = \"windows\";\n");
    #[cfg(target_os = "macos")]
    config.push_str("    pub const OS: &str = \"macos\";\n");
    #[cfg(target_os = "linux")]
    config.push_str("    pub const OS: &str = \"linux\";\n");
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    config.push_str("    pub const OS: &str = \"unknown\";\n");
    
    #[cfg(target_arch = "x86_64")]
    config.push_str("    pub const ARCH: &str = \"x86_64\";\n");
    #[cfg(target_arch = "aarch64")]
    config.push_str("    pub const ARCH: &str = \"aarch64\";\n");
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    config.push_str("    pub const ARCH: &str = \"unknown\";\n");
    
    // Add CPU information
    config.push_str("    pub const NUM_CPUS: usize = ");
    config.push_str(&num_cpus::get().to_string());
    config.push_str(";\n");
    
    config.push_str("}\n");
    
    // Write the configuration file
    std::fs::write(out_path, config).unwrap();
}