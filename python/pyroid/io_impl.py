"""
Pyroid I/O Implementation
=====================

This module provides Python implementations of the I/O functions.

Functions:
    read_file: Read a file
    write_file: Write a file
    read_files: Read multiple files
    get: HTTP GET request
    post: HTTP POST request
    sleep: Async sleep
    read_file_async: Async read file
    write_file_async: Async write file
"""

import os
import asyncio
import urllib.request
import urllib.parse
import json
from typing import Dict, List, Any, Union, Optional

# Dummy aiofiles implementation
class aiofiles:
    @staticmethod
    async def open(file_path, mode):
        class AsyncFileContext:
            def __init__(self, file_path, mode):
                self.file_path = file_path
                self.mode = mode
                self.file = None
            
            async def __aenter__(self):
                self.file = open(self.file_path, self.mode)
                return self
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                if self.file:
                    self.file.close()
            
            async def read(self):
                return self.file.read()
            
            async def write(self, content):
                return self.file.write(content)
        
        return AsyncFileContext(file_path, mode)

def read_file(path: str) -> str:
    """
    Read a file.
    
    Args:
        path: The path to the file
        
    Returns:
        The file contents
    """
    with open(path, 'r') as f:
        return f.read()

def write_file(path: str, content: str) -> bool:
    """
    Write a file.
    
    Args:
        path: The path to the file
        content: The content to write
        
    Returns:
        True if successful
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    with open(path, 'w') as f:
        f.write(content)
    
    return True

def read_files(paths: List[str]) -> Dict[str, str]:
    """
    Read multiple files.
    
    Args:
        paths: The paths to the files
        
    Returns:
        A dictionary mapping paths to file contents
    """
    result = {}
    for path in paths:
        result[path] = read_file(path)
    return result

def get(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    HTTP GET request.
    
    Args:
        url: The URL to request
        params: The query parameters
        
    Returns:
        The response
    """
    if params:
        query_string = urllib.parse.urlencode(params)
        url = f"{url}?{query_string}"
    
    with urllib.request.urlopen(url) as response:
        response_text = response.read().decode('utf-8')
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {"text": response_text}

def post(url: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    HTTP POST request.
    
    Args:
        url: The URL to request
        data: The data to send
        
    Returns:
        The response
    """
    headers = {'Content-Type': 'application/json'}
    
    if data:
        data_bytes = json.dumps(data).encode('utf-8')
    else:
        data_bytes = None
    
    request = urllib.request.Request(url, data=data_bytes, headers=headers, method='POST')
    
    with urllib.request.urlopen(request) as response:
        response_text = response.read().decode('utf-8')
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {"text": response_text}

async def sleep(seconds: float) -> None:
    """
    Async sleep.
    
    Args:
        seconds: The number of seconds to sleep
    """
    await asyncio.sleep(seconds)

async def read_file_async(path: str) -> str:
    """
    Async read file.
    
    Args:
        path: The path to the file
        
    Returns:
        The file contents
    """
    async with aiofiles.open(path, 'r') as f:
        return await f.read()

async def write_file_async(path: str, content: str) -> bool:
    """
    Async write file.
    
    Args:
        path: The path to the file
        content: The content to write
        
    Returns:
        True if successful
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    async with aiofiles.open(path, 'w') as f:
        await f.write(content)
    
    return True