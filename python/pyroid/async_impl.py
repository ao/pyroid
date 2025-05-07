"""
Pyroid Async Implementation
=======================

This module provides Python implementations of the async functions.

Classes:
    AsyncClient: Client for asynchronous HTTP requests
    AsyncFileReader: Reader for asynchronous file operations

Functions:
    sleep: Async sleep
    read_file_async: Async read file
    write_file_async: Async write file
    fetch_url: Async HTTP GET request
    fetch_many: Async fetch multiple URLs
    download_file: Async download file
    http_post: Async HTTP POST request
"""

import os
import asyncio
import urllib.request
import urllib.parse
import json
from typing import Dict, List, Any, Union, Optional

class AsyncClient:
    """Client for asynchronous HTTP requests."""
    
    def __init__(self, base_url: str = "", timeout: float = 30.0):
        """
        Create a new AsyncClient.
        
        Args:
            base_url: The base URL for requests
            timeout: The timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
    
    async def get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Async HTTP GET request.
        
        Args:
            url: The URL to request
            params: The query parameters
            
        Returns:
            The response
        """
        return await fetch_url(self.base_url + url, params, self.timeout)
    
    async def post(self, url: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Async HTTP POST request.
        
        Args:
            url: The URL to request
            data: The data to send
            
        Returns:
            The response
        """
        return await http_post(self.base_url + url, data, self.timeout)
    
    async def download(self, url: str, path: str) -> bool:
        """
        Async download file.
        
        Args:
            url: The URL to download from
            path: The path to save to
            
        Returns:
            True if successful
        """
        return await download_file(self.base_url + url, path, self.timeout)

class AsyncFileReader:
    """Reader for asynchronous file operations."""
    
    def __init__(self, path: str):
        """
        Create a new AsyncFileReader.
        
        Args:
            path: The path to the file
        """
        self.path = path
        self.file = None
    
    async def open(self) -> bool:
        """
        Open the file.
        
        Returns:
            True if successful
        """
        try:
            self.file = await aiofiles_open(self.path, 'r')
            return True
        except Exception:
            return False
    
    async def read(self) -> str:
        """
        Read the file.
        
        Returns:
            The file contents
        """
        if not self.file:
            await self.open()
        
        return await self.file.read()
    
    async def read_lines(self) -> List[str]:
        """
        Read the file lines.
        
        Returns:
            The file lines
        """
        if not self.file:
            await self.open()
        
        content = await self.file.read()
        return content.splitlines()
    
    async def close(self) -> None:
        """Close the file."""
        if self.file:
            await self.file.close()
            self.file = None

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
    async with aiofiles_open(path, 'r') as f:
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
    
    async with aiofiles_open(path, 'w') as f:
        await f.write(content)
    
    return True

async def fetch_url(url: str, params: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Dict[str, Any]:
    """
    Async HTTP GET request.
    
    Args:
        url: The URL to request
        params: The query parameters
        timeout: The timeout in seconds
        
    Returns:
        The response
    """
    if params:
        query_string = urllib.parse.urlencode(params)
        url = f"{url}?{query_string}"
    
    # Use asyncio to run the blocking request in a thread pool
    loop = asyncio.get_event_loop()
    response_text = await loop.run_in_executor(
        None,
        lambda: urllib.request.urlopen(url, timeout=timeout).read().decode('utf-8')
    )
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {"text": response_text}

async def fetch_many(urls: List[str], timeout: float = 30.0) -> List[Dict[str, Any]]:
    """
    Async fetch multiple URLs.
    
    Args:
        urls: The URLs to request
        timeout: The timeout in seconds
        
    Returns:
        The responses
    """
    tasks = [fetch_url(url, None, timeout) for url in urls]
    return await asyncio.gather(*tasks)

async def download_file(url: str, path: str, timeout: float = 30.0) -> bool:
    """
    Async download file.
    
    Args:
        url: The URL to download from
        path: The path to save to
        timeout: The timeout in seconds
        
    Returns:
        True if successful
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # Use asyncio to run the blocking request in a thread pool
    loop = asyncio.get_event_loop()
    
    try:
        data = await loop.run_in_executor(
            None,
            lambda: urllib.request.urlopen(url, timeout=timeout).read()
        )
        
        async with aiofiles_open(path, 'wb') as f:
            await f.write(data)
        
        return True
    except Exception:
        return False

async def http_post(url: str, data: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Dict[str, Any]:
    """
    Async HTTP POST request.
    
    Args:
        url: The URL to request
        data: The data to send
        timeout: The timeout in seconds
        
    Returns:
        The response
    """
    headers = {'Content-Type': 'application/json'}
    
    if data:
        data_bytes = json.dumps(data).encode('utf-8')
    else:
        data_bytes = None
    
    request = urllib.request.Request(url, data=data_bytes, headers=headers, method='POST')
    
    # Use asyncio to run the blocking request in a thread pool
    loop = asyncio.get_event_loop()
    response_text = await loop.run_in_executor(
        None,
        lambda: urllib.request.urlopen(request, timeout=timeout).read().decode('utf-8')
    )
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {"text": response_text}

# Dummy aiofiles implementation
class AsyncFile:
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

def aiofiles_open(file_path, mode):
    """
    Open a file asynchronously.
    
    Args:
        file_path: The path to the file
        mode: The mode to open the file in
        
    Returns:
        An async file object
    """
    return AsyncFile(file_path, mode)