"""
Async helpers for pyroid.

This module provides async helper functions for pyroid.
"""

import asyncio
import aiohttp
import aiofiles
from typing import List, Dict, Any, Optional, Union


async def fetch_url(url: str) -> Dict[str, Any]:
    """Fetch a URL asynchronously.
    
    Args:
        url: The URL to fetch
        
    Returns:
        A dictionary with status and text
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return {
                "status": response.status,
                "text": await response.text()
            }


async def fetch_many(urls: List[str], concurrency: int = 10) -> Dict[str, Any]:
    """Fetch multiple URLs concurrently.
    
    Args:
        urls: A list of URLs to fetch
        concurrency: Maximum number of concurrent requests
        
    Returns:
        A dictionary mapping URLs to their responses
    """
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


async def download_file(url: str, path: str) -> Dict[str, Any]:
    """Download a file asynchronously.
    
    Args:
        url: The URL to download from
        path: The path to save the file to
        
    Returns:
        A dictionary with success status and path
    """
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


async def send_to_channel(value: Any) -> None:
    """Send a value to a channel.
    
    Args:
        value: The value to send
    """
    # This is a placeholder. In a real implementation, this would use
    # the actual channel from the Rust side.
    await asyncio.sleep(0)
    return None


async def receive_from_channel() -> Any:
    """Receive a value from a channel.
    
    Returns:
        The received value
    """
    # This is a placeholder. In a real implementation, this would use
    # the actual channel from the Rust side.
    await asyncio.sleep(0)
    return None


async def read_file(path: str) -> bytes:
    """Read a file asynchronously.
    
    Args:
        path: The path to the file
        
    Returns:
        The file contents as bytes
    """
    async with aiofiles.open(path, "rb") as f:
        return await f.read()


async def read_file_lines(path: str) -> List[str]:
    """Read a file line by line asynchronously.
    
    Args:
        path: The path to the file
        
    Returns:
        A list of lines from the file
    """
    async with aiofiles.open(path, "r") as f:
        return await f.readlines()


async def write_file(path: str, data: bytes) -> Dict[str, Any]:
    """Write data to a file asynchronously.
    
    Args:
        path: The path to the file
        data: The data to write
        
    Returns:
        A dictionary with success status and path
    """
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


async def http_post(url: str, data: Optional[bytes] = None, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Make an HTTP POST request asynchronously.
    
    Args:
        url: The URL to send the request to
        data: The data to send as the request body
        json: The JSON data to send as the request body
        
    Returns:
        A dictionary with status, content, and headers
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data, json=json) as response:
            content = await response.read()
            headers = {k: v for k, v in response.headers.items()}
            
            return {
                "status": response.status,
                "content": content,
                "headers": headers
            }