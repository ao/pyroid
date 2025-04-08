"""
Pyroid I/O Module
===============

This module provides high-performance I/O operations.

Functions:
    read_file: Read a file
    write_file: Write a file
    read_files: Read multiple files in parallel
    get: HTTP GET request
    post: HTTP POST request
    sleep: Async sleep
    read_file_async: Async read file
    write_file_async: Async write file
"""

# Try to import directly from the pyroid module
try:
    from ..pyroid import (
        # File operations
        read_file,
        write_file,
        read_files,
        
        # Network operations
        get,
        post,
        
        # Async operations
        sleep,
        read_file_async,
        write_file_async,
    )
except ImportError:
    # Fallback to importing from the io module
    try:
        from ..pyroid.io import (
            # File operations
            read_file,
            write_file,
            read_files,
            
            # Network operations
            get,
            post,
            
            # Async operations
            sleep,
            read_file_async,
            write_file_async,
        )
    except ImportError:
        # If all else fails, create dummy functions for documentation purposes
        def read_file(path):
            """Read a file (not available)."""
            try:
                with open(path, 'r') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {e}"
            
        def write_file(path, content):
            """Write a file (not available)."""
            try:
                with open(path, 'w') as f:
                    f.write(content)
                return True
            except Exception as e:
                return f"Error writing file: {e}"
            
        def read_files(paths):
            """Read multiple files in parallel (not available)."""
            return {path: read_file(path) for path in paths}
            
        def get(url, headers=None):
            """HTTP GET request (not available)."""
            try:
                import urllib.request
                req = urllib.request.Request(url, headers=headers or {})
                with urllib.request.urlopen(req) as response:
                    return response.read().decode('utf-8')
            except Exception as e:
                return f"Error making GET request: {e}"
            
        def post(url, data=None, headers=None):
            """HTTP POST request (not available)."""
            try:
                import urllib.request
                import json
                if isinstance(data, dict):
                    data = json.dumps(data).encode('utf-8')
                req = urllib.request.Request(url, data=data, headers=headers or {}, method='POST')
                with urllib.request.urlopen(req) as response:
                    return response.read().decode('utf-8')
            except Exception as e:
                return f"Error making POST request: {e}"
            
        async def sleep(seconds):
            """Async sleep (not available)."""
            try:
                import asyncio
                await asyncio.sleep(seconds)
            except Exception as e:
                return f"Error sleeping: {e}"
            
        async def read_file_async(path):
            """Async read file (not available)."""
            try:
                import aiofiles
                async with aiofiles.open(path, 'r') as f:
                    return await f.read()
            except ImportError:
                return read_file(path)
            except Exception as e:
                return f"Error reading file asynchronously: {e}"
            
        async def write_file_async(path, content):
            """Async write file (not available)."""
            try:
                import aiofiles
                async with aiofiles.open(path, 'w') as f:
                    await f.write(content)
                return True
            except ImportError:
                return write_file(path, content)
            except Exception as e:
                return f"Error writing file asynchronously: {e}"

__all__ = [
    'read_file',
    'write_file',
    'read_files',
    'get',
    'post',
    'sleep',
    'read_file_async',
    'write_file_async',
]