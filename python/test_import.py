#!/usr/bin/env python3
"""
Test script to check if we can import the Config class directly.
"""

import sys
import os

# Add the parent directory to the path so we can import pyroid
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    # Try to import directly from our Python implementation
    from pyroid.core import Config, ConfigContext, SharedData
    print("Successfully imported Config, ConfigContext, and SharedData from pyroid.core")
    
    # Create a Config object
    config = Config({"parallel": True, "chunk_size": 1000})
    print(f"Created Config object: {config.options}")
    
    # Create a ConfigContext object
    context = ConfigContext(config)
    print(f"Created ConfigContext object: {context.config.options}")
    
    # Create a SharedData object
    data = SharedData([1, 2, 3, 4, 5])
    print(f"Created SharedData object: {data.get()}")
    
    # Try to import async classes
    try:
        from pyroid import AsyncClient, AsyncFileReader
        print("Successfully imported AsyncClient and AsyncFileReader")
    except ImportError as e:
        print(f"Failed to import AsyncClient and AsyncFileReader: {e}")
    
except ImportError as e:
    print(f"Failed to import from pyroid.core: {e}")

print("Done")