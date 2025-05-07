#!/usr/bin/env python3
"""
Inspect the pyroid.pyroid module.
"""

import sys
import os
import importlib.util

# Get the path to the .so file
so_path = os.path.join(os.path.dirname(__file__), "pyroid", "pyroid.cpython-312-darwin.so")
print(f"Looking for .so file at: {so_path}")
print(f"File exists: {os.path.exists(so_path)}")

# Try to load the module directly
try:
    spec = importlib.util.spec_from_file_location("pyroid.pyroid", so_path)
    if spec is not None:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("Successfully loaded pyroid.pyroid module")
        
        # Print all available attributes
        print("Available attributes:")
        for attr in dir(module):
            if not attr.startswith('__'):
                print(f"  - {attr}")
                
                # Try to get the attribute
                try:
                    value = getattr(module, attr)
                    print(f"    Type: {type(value)}")
                except Exception as e:
                    print(f"    Error getting attribute: {e}")
    else:
        print("Failed to create spec from file location")
except Exception as e:
    print(f"Error loading module: {e}")

print("Done")