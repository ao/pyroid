use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search=/usr/local/lib");
    println!("cargo:rustc-link-search=/opt/homebrew/lib");
    println!("cargo:rustc-link-search=/opt/homebrew/anaconda3/lib");
    
    // Explicitly link against the found Python library
    println!("cargo:rustc-link-lib=dylib=python3.12");
    
    // Try to find Python library path using python3-config
    let output = Command::new("python3-config")
        .arg("--ldflags")
        .output();
    
    if let Ok(output) = output {
        if output.status.success() {
            let ldflags = String::from_utf8_lossy(&output.stdout);
            for flag in ldflags.split_whitespace() {
                if flag.starts_with("-L") {
                    let path = &flag[2..]; // Remove the -L prefix
                    println!("cargo:rustc-link-search={}", path);
                } else if flag.starts_with("-l") {
                    let lib = &flag[2..]; // Remove the -l prefix
                    println!("cargo:rustc-link-lib={}", lib);
                }
            }
        }
    }
    
    // Try to find Python library from environment
    if let Ok(python_path) = env::var("PYTHON_SYS_EXECUTABLE") {
        let python_path = PathBuf::from(python_path);
        let python_dir = python_path.parent().unwrap();
        let lib_dir = python_dir.join("lib");
        println!("cargo:rustc-link-search={}", lib_dir.display());
    }
    
    // Try to find Python library using python3 command
    let output = Command::new("python3")
        .args(["-c", "import sys; import os; print(os.path.join(sys.exec_prefix, 'lib'))"])
        .output();
    
    if let Ok(output) = output {
        if output.status.success() {
            let lib_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            println!("cargo:rustc-link-search={}", lib_path);
        }
    }
    
    // Tell cargo to invalidate the built crate whenever the build script changes
    println!("cargo:rerun-if-changed=build.rs");
}