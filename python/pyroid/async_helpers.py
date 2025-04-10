"""
Async helpers for pyroid.

This module provides async helper functions for pyroid.
"""

import asyncio
import aiohttp
import aiofiles
from typing import List, Dict, Any, Optional, Union
import time
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyroid.async")

# Global connection pool for reuse
_CONNECTION_POOLS = {}

# Performance metrics for adaptive concurrency
_PERFORMANCE_METRICS = {
    "response_times": defaultdict(list),
    "optimal_concurrency": {},
    "last_adjustment": {},
}

# Maximum number of response times to keep per host
MAX_RESPONSE_TIMES = 50

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

async def fetch_url_optimized(url: str, timeout: float = 30.0, host: Optional[str] = None) -> Dict[str, Any]:
    """Fetch a URL asynchronously with optimized connection pooling.
    
    Args:
        url: The URL to fetch
        timeout: Request timeout in seconds
        host: The host for connection pooling (extracted from URL if not provided)
        
    Returns:
        A dictionary with status and text
    """
    if host is None:
        # Extract host from URL
        if "://" in url:
            host = url.split("://")[1].split("/")[0]
        else:
            host = url
    
    # Get or create a connection pool for this host
    if host not in _CONNECTION_POOLS:
        connector = aiohttp.TCPConnector(
            limit_per_host=20,
            ttl_dns_cache=300,
            use_dns_cache=True,
            ssl=False  # For better performance, but less secure
        )
        _CONNECTION_POOLS[host] = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=timeout)
        )
    
    session = _CONNECTION_POOLS[host]
    
    # Record start time for metrics
    start_time = time.time()
    
    try:
        async with session.get(url) as response:
            result = {
                "status": response.status,
                "text": await response.text(),
                "headers": dict(response.headers),
                "url": str(response.url)
            }
            
            # Record response time for adaptive concurrency
            response_time = time.time() - start_time
            _PERFORMANCE_METRICS["response_times"][host].append(response_time)
            
            # Keep only the last MAX_RESPONSE_TIMES response times
            if len(_PERFORMANCE_METRICS["response_times"][host]) > MAX_RESPONSE_TIMES:
                _PERFORMANCE_METRICS["response_times"][host] = _PERFORMANCE_METRICS["response_times"][host][-MAX_RESPONSE_TIMES:]
            
            return result
    except Exception as e:
        # Log the error and return an error response
        logger.error(f"Error fetching {url}: {str(e)}")
        return {
            "status": 0,
            "error": str(e),
            "url": url
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

async def fetch_many_optimized(
    urls: List[str], 
    concurrency: int = 10, 
    timeout: float = 30.0,
    url_groups: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Any]:
    """Fetch multiple URLs concurrently with optimized connection pooling.
    
    Args:
        urls: A list of URLs to fetch
        concurrency: Maximum number of concurrent requests
        timeout: Request timeout in seconds
        url_groups: Optional pre-grouped URLs by host
        
    Returns:
        A dictionary mapping URLs to their responses
    """
    results = {}
    
    # Group URLs by host if not already grouped
    if url_groups is None:
        url_groups = {}
        for url in urls:
            if "://" in url:
                host = url.split("://")[1].split("/")[0]
            else:
                host = url
                
            if host not in url_groups:
                url_groups[host] = []
            url_groups[host].append(url)
    
    # Process each host group with its own connection pool and semaphore
    async def process_host_group(host: str, host_urls: List[str]):
        # Get optimal concurrency for this host
        host_concurrency = _PERFORMANCE_METRICS.get("optimal_concurrency", {}).get(host, concurrency)
        
        # Create a semaphore for this host
        host_semaphore = asyncio.Semaphore(host_concurrency)
        
        # Get or create a connection pool for this host
        if host not in _CONNECTION_POOLS:
            connector = aiohttp.TCPConnector(
                limit_per_host=host_concurrency * 2,  # Allow more connections than concurrency
                ttl_dns_cache=300,
                use_dns_cache=True,
                ssl=False  # For better performance, but less secure
            )
            _CONNECTION_POOLS[host] = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=timeout)
            )
        
        session = _CONNECTION_POOLS[host]
        
        # Process URLs for this host
        async def fetch_url_with_semaphore(url: str):
            async with host_semaphore:
                start_time = time.time()
                try:
                    async with session.get(url) as response:
                        results[url] = {
                            "status": response.status,
                            "text": await response.text(),
                            "headers": dict(response.headers),
                            "url": str(response.url)
                        }
                        
                        # Record response time
                        response_time = time.time() - start_time
                        _PERFORMANCE_METRICS["response_times"][host].append(response_time)
                        
                except Exception as e:
                    results[url] = {
                        "status": 0,
                        "error": str(e),
                        "url": url
                    }
                    logger.error(f"Error fetching {url}: {str(e)}")
        
        # Process all URLs for this host
        await asyncio.gather(*(fetch_url_with_semaphore(url) for url in host_urls))
        
        # Adjust concurrency based on response times
        adjust_concurrency_for_host(host, host_concurrency)
    
    # Process all host groups concurrently
    await asyncio.gather(*(process_host_group(host, host_urls) 
                          for host, host_urls in url_groups.items()))
    
    return results

def adjust_concurrency_for_host(host: str, current_concurrency: int):
    """Adjust concurrency for a host based on response times."""
    # Only adjust every 10 seconds
    now = time.time()
    if host in _PERFORMANCE_METRICS["last_adjustment"] and \
       now - _PERFORMANCE_METRICS["last_adjustment"][host] < 10:
        return
    
    # Update last adjustment time
    _PERFORMANCE_METRICS["last_adjustment"][host] = now
    
    # Get response times for this host
    response_times = _PERFORMANCE_METRICS["response_times"].get(host, [])
    if len(response_times) < 10:
        return  # Not enough data
    
    # Calculate average response time
    avg_time = sum(response_times) / len(response_times)
    
    # Adjust concurrency based on average response time
    new_concurrency = current_concurrency
    if avg_time < 0.1:  # Very fast responses
        new_concurrency = min(current_concurrency * 2, 50)
    elif avg_time < 0.5:  # Fast responses
        new_concurrency = min(current_concurrency + 2, 30)
    elif avg_time > 2.0:  # Slow responses
        new_concurrency = max(current_concurrency - 2, 5)
    elif avg_time > 1.0:  # Somewhat slow responses
        new_concurrency = max(current_concurrency - 1, 5)
    
    # Update optimal concurrency if changed
    if new_concurrency != current_concurrency:
        _PERFORMANCE_METRICS["optimal_concurrency"][host] = new_concurrency
        logger.info(f"Adjusted concurrency for {host}: {current_concurrency} -> {new_concurrency} (avg time: {avg_time:.2f}s)")

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

# Clean up connection pools on module unload
import atexit

def cleanup_connection_pools():
    """Close all connection pools."""
    for session in _CONNECTION_POOLS.values():
        try:
            if not session.closed:
                asyncio.run(session.close())
        except Exception as e:
            logger.error(f"Error closing session: {e}")

atexit.register(cleanup_connection_pools)