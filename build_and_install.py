#!/usr/bin/env python3
"""
Build and install script for pyroid.

This script builds and installs the pyroid package.
"""

import os
import sys
import subprocess
import argparse
import shutil
import platform
from typing import List, Optional

def run_command(cmd: List[str], cwd: Optional[str] = None) -> int:
    """Run a command and return the exit code."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode

def build_rust_extension() -> int:
    """Build the Rust extension."""
    print("Building Rust extension...")
    
    # Determine the appropriate command based on the platform
    if platform.system() == "Windows":
        cmd = ["cargo", "build", "--release"]
    else:
        cmd = ["cargo", "build", "--release"]
    
    return run_command(cmd)

def install_python_package(develop: bool = False) -> int:
    """Install the Python package."""
    print("Installing Python package...")
    
    # Determine the appropriate command based on the arguments
    if develop:
        cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "."]
    
    return run_command(cmd)

def run_tests() -> int:
    """Run the tests."""
    print("Running tests...")
    
    # Run Python tests
    python_test_result = run_command([sys.executable, "-m", "unittest", "discover", "python/tests"])
    
    # Run Rust tests
    rust_test_result = run_command(["cargo", "test"])
    
    return python_test_result or rust_test_result

def run_benchmarks(size: str = "small", suite: Optional[str] = None) -> int:
    """Run the benchmarks."""
    print(f"Running benchmarks (size={size}, suite={suite or 'all'})...")
    
    cmd = [sys.executable, "-m", "benchmarks.run_benchmarks", f"--size={size}"]
    
    if suite:
        cmd.append(f"--suite={suite}")
    
    return run_command(cmd)

def clean() -> int:
    """Clean the build artifacts."""
    print("Cleaning build artifacts...")
    
    # Clean Rust artifacts
    rust_clean_result = run_command(["cargo", "clean"])
    
    # Clean Python artifacts
    dirs_to_clean = [
        "build",
        "dist",
        "*.egg-info",
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
    ]
    
    for dir_pattern in dirs_to_clean:
        for path in os.popen(f"find . -name '{dir_pattern}' -type d").read().splitlines():
            if os.path.exists(path):
                print(f"Removing {path}")
                shutil.rmtree(path)
    
    return rust_clean_result

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build and install pyroid")
    parser.add_argument("--develop", action="store_true", help="Install in development mode")
    parser.add_argument("--test", action="store_true", help="Run tests after building")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks after building")
    parser.add_argument("--benchmark-size", choices=["small", "medium", "large"], default="small", help="Size of benchmarks to run")
    parser.add_argument("--benchmark-suite", help="Specific benchmark suite to run")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    parser.add_argument("--skip-build", action="store_true", help="Skip building the Rust extension")
    parser.add_argument("--skip-install", action="store_true", help="Skip installing the Python package")
    
    args = parser.parse_args()
    
    if args.clean:
        return clean()
    
    result = 0
    
    if not args.skip_build:
        result = build_rust_extension()
        if result != 0:
            return result
    
    if not args.skip_install:
        result = install_python_package(args.develop)
        if result != 0:
            return result
    
    if args.test:
        result = run_tests()
        if result != 0:
            return result
    
    if args.benchmark:
        result = run_benchmarks(args.benchmark_size, args.benchmark_suite)
    
    return result

if __name__ == "__main__":
    sys.exit(main())