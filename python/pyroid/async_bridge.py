"""
Async bridge for pyroid.

This module provides a bridge between Rust and Python for async operations.
"""

import asyncio
import aiohttp
import aiofiles
from typing import List, Dict, Any, Optional, Union
import json
import time
import logging
from .async_helpers import (
    fetch_url_optimized, 
    fetch_many_optimized, 
    read_file, 
    read_file_lines, 
    write_file as async_write_file,
    download_file as async_download_file,
    http_post as async_http_post
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyroid.bridge")

def run_async(coro):
    """Run an async coroutine and return the result.
    
    Args:
        coro: The coroutine to run
        
    Returns:
        The result of the coroutine
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # Create a new event loop if one doesn't exist
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


def sleep(seconds: float) -> None:
    """Sleep for the specified number of seconds.
    
    Args:
        seconds: The number of seconds to sleep
    """
    return run_async(asyncio.sleep(seconds))

def read_file(path: str) -> bytes:
    """Read a file asynchronously.
    
    Args:
        path: The path to the file
        
    Returns:
        The file contents as bytes
    """
    from .async_helpers import read_file as async_read_file
    return run_async(async_read_file(path))
    return run_async(read_file(path))


def read_file_lines(path: str) -> List[str]:
    """Read a file line by line asynchronously.
    
    Args:
        path: The path to the file
        
    Returns:
        A list of lines from the file
    """
    from .async_helpers import read_file_lines as async_read_file_lines
    return run_async(async_read_file_lines(path))


def write_file(path: str, data: bytes) -> Dict[str, Any]:
    """Write data to a file asynchronously.
    
    Args:
        path: The path to the file
        data: The data to write
        
    Returns:
        A dictionary with success status and path
    """
    return run_async(async_write_file(path, data))


def fetch_url(url: str) -> Dict[str, Any]:
    """Fetch a URL asynchronously.
    
    Args:
        url: The URL to fetch
        
    Returns:
        A dictionary with status and text
    """
    # Extract host from URL for connection pooling
    if "://" in url:
        host = url.split("://")[1].split("/")[0]
    else:
        host = url
        
    return run_async(fetch_url_optimized(url, host=host))


def fetch_many(urls: List[str], concurrency: int = 10) -> Dict[str, Any]:
    """Fetch multiple URLs concurrently.
    
    Args:
        urls: A list of URLs to fetch
        concurrency: Maximum number of concurrent requests
        
    Returns:
        A dictionary mapping URLs to their responses
    """
    # Group URLs by host for connection pooling
    url_groups = {}
    for url in urls:
        if "://" in url:
            host = url.split("://")[1].split("/")[0]
        else:
            host = url
            
        if host not in url_groups:
            url_groups[host] = []
        url_groups[host].append(url)
    
    return run_async(fetch_many_optimized(urls, concurrency=concurrency, url_groups=url_groups))


def download_file(url: str, path: str) -> Dict[str, Any]:
    """Download a file asynchronously.
    
    Args:
        url: The URL to download from
        path: The path to save the file to
        
    Returns:
        A dictionary with success status and path
    """
    return run_async(async_download_file(url, path))


def http_post(url: str, data: Optional[bytes] = None, json_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Make an HTTP POST request asynchronously.
    
    Args:
        url: The URL to send the request to
        data: The data to send as the request body
        json_data: The JSON data to send as the request body
        
    Returns:
        A dictionary with status, content, and headers
    """
    return run_async(async_http_post(url, data, json_data))


# Performance monitoring
_PERFORMANCE_METRICS = {
    "requests": 0,
    "errors": 0,
    "total_time": 0,
    "start_time": time.time()
}

def get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics for the async bridge.
    
    Returns:
        A dictionary with performance metrics
    """
    uptime = time.time() - _PERFORMANCE_METRICS["start_time"]
    requests_per_second = _PERFORMANCE_METRICS["requests"] / uptime if uptime > 0 else 0
    
    return {
        "requests": _PERFORMANCE_METRICS["requests"],
        "errors": _PERFORMANCE_METRICS["errors"],
        "total_time": _PERFORMANCE_METRICS["total_time"],
        "uptime": uptime,
        "requests_per_second": requests_per_second
    }

def _update_metrics(success: bool, duration: float) -> None:
    """Update performance metrics.
    
    Args:
        success: Whether the request was successful
        duration: The duration of the request in seconds
    """
    _PERFORMANCE_METRICS["requests"] += 1
    _PERFORMANCE_METRICS["total_time"] += duration
    
    if not success:
        _PERFORMANCE_METRICS["errors"] += 1