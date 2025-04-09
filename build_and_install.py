#!/usr/bin/env python3
"""
Build and install script for pyroid.

This script builds and installs the pyroid package.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def check_requirements():
    """Check if the required tools are installed."""
    print("Checking requirements...")
    
    # Check if Rust is installed
    try:
        subprocess.run(["rustc", "--version"], check=True, stdout=subprocess.PIPE)
        print("✓ Rust is installed")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("✗ Rust is not installed. Please install Rust from https://rustup.rs/")
        sys.exit(1)
    
    # Check if Cargo is installed
    try:
        subprocess.run(["cargo", "--version"], check=True, stdout=subprocess.PIPE)
        print("✓ Cargo is installed")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("✗ Cargo is not installed. Please install Rust from https://rustup.rs/")
        sys.exit(1)
    
    # Check if Python is installed
    try:
        subprocess.run([sys.executable, "--version"], check=True, stdout=subprocess.PIPE)
        print(f"✓ Python is installed: {sys.version}")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("✗ Python is not installed. Please install Python from https://www.python.org/")
        sys.exit(1)
    
    # Check if pip is installed
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, stdout=subprocess.PIPE)
        print("✓ pip is installed")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("✗ pip is not installed. Please install pip.")
        sys.exit(1)

def build_package():
    """Build the pyroid package."""
    print("\nBuilding pyroid package...")
    
    # Build the package
    try:
        subprocess.run(["cargo", "build", "--release"], check=True)
        print("✓ Cargo build successful")
    except subprocess.SubprocessError as e:
        print(f"✗ Cargo build failed: {e}")
        sys.exit(1)

def install_package():
    """Install the pyroid package."""
    print("\nInstalling pyroid package...")
    
    # Install the package
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        print("✓ Package installed successfully")
    except subprocess.SubprocessError as e:
        print(f"✗ Package installation failed: {e}")
        sys.exit(1)

def install_dependencies():
    """Install Python dependencies."""
    print("\nInstalling Python dependencies...")
    
    # Install the dependencies
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "aiohttp", "aiofiles"], check=True)
        print("✓ Dependencies installed successfully")
    except subprocess.SubprocessError as e:
        print(f"✗ Dependencies installation failed: {e}")
        sys.exit(1)

def main():
    """Main function."""
    print("Pyroid Build and Install Script")
    print("===============================\n")
    
    check_requirements()
    build_package()
    install_dependencies()
    install_package()
    
    print("\nPyroid has been built and installed successfully!")
    print("\nYou can now run the benchmarks:")
    print("  python -m benchmarks.run_benchmarks --size small --suite async --no-dashboard")
    print("  python -m benchmarks.run_benchmarks --size small --suite high-throughput --no-dashboard")
    print("\nOr try the examples:")
    print("  python examples/optimized_async_example.py")

if __name__ == "__main__":
    main()