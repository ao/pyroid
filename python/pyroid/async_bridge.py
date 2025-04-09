"""
Async bridge for pyroid.

This module provides a bridge between Rust and Python for async operations.
"""

import asyncio
import aiohttp
import aiofiles
from typing import List, Dict, Any, Optional, Union
import json


def run_async(coro):
    """Run an async coroutine and return the result.
    
    Args:
        coro: The coroutine to run
        
    Returns:
        The result of the coroutine
    """
    loop = asyncio.get_event_loop()
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
    async def _read_file():
        async with aiofiles.open(path, "rb") as f:
            return await f.read()
    
    return run_async(_read_file())


def read_file_lines(path: str) -> List[str]:
    """Read a file line by line asynchronously.
    
    Args:
        path: The path to the file
        
    Returns:
        A list of lines from the file
    """
    async def _read_file_lines():
        async with aiofiles.open(path, "r") as f:
            return await f.readlines()
    
    return run_async(_read_file_lines())


def write_file(path: str, data: bytes) -> Dict[str, Any]:
    """Write data to a file asynchronously.
    
    Args:
        path: The path to the file
        data: The data to write
        
    Returns:
        A dictionary with success status and path
    """
    async def _write_file():
        # Create parent directories if they don't exist
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Write the file
        async with aiofiles.open(path, "wb") as f:
            await f.write(data)
        
        return {
            "success": True,
            "path": path,
            "bytes_written": len(data)
        }
    
    return run_async(_write_file())


def fetch_url(url: str) -> Dict[str, Any]:
    """Fetch a URL asynchronously.
    
    Args:
        url: The URL to fetch
        
    Returns:
        A dictionary with status and text
    """
    async def _fetch_url():
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return {
                    "status": response.status,
                    "text": await response.text()
                }
    
    return run_async(_fetch_url())


def fetch_many(urls: List[str], concurrency: int = 10) -> Dict[str, Any]:
    """Fetch multiple URLs concurrently.
    
    Args:
        urls: A list of URLs to fetch
        concurrency: Maximum number of concurrent requests
        
    Returns:
        A dictionary mapping URLs to their responses
    """
    async def _fetch_many():
        semaphore = asyncio.Semaphore(concurrency)
        results = {}
        
        async def fetch_with_semaphore(url: str) -> None:
            async with semaphore:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url) as response:
                            results[url] = {
                                "status": response.status,
                                "text": await response.text()
                            }
                except Exception as e:
                    results[url] = str(e)
        
        await asyncio.gather(*(fetch_with_semaphore(url) for url in urls))
        return results
    
    return run_async(_fetch_many())


def download_file(url: str, path: str) -> Dict[str, Any]:
    """Download a file asynchronously.
    
    Args:
        url: The URL to download from
        path: The path to save the file to
        
    Returns:
        A dictionary with success status and path
    """
    async def _download_file():
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if not response.ok:
                    return {
                        "success": False,
                        "error": f"Failed to download file: HTTP {response.status}"
                    }
                
                # Create parent directories if they don't exist
                import os
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                
                # Write the file
                async with aiofiles.open(path, "wb") as f:
                    await f.write(await response.read())
                
                return {
                    "success": True,
                    "path": path
                }
    
    return run_async(_download_file())


def http_post(url: str, data: Optional[bytes] = None, json_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Make an HTTP POST request asynchronously.
    
    Args:
        url: The URL to send the request to
        data: The data to send as the request body
        json_data: The JSON data to send as the request body
        
    Returns:
        A dictionary with status, content, and headers
    """
    async def _http_post():
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data, json=json_data) as response:
                content = await response.read()
                headers = {k: v for k, v in response.headers.items()}
                
                return {
                    "status": response.status,
                    "content": content,
                    "headers": headers
                }
    
    return run_async(_http_post())