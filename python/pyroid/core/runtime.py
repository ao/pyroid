"""
Runtime management for Pyroid.

This module provides a unified runtime for all async operations.
"""

import asyncio
import threading
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

# Global runtime state
_initialized = False
_worker_threads = 0
_max_connections_per_host = 0

def init(worker_threads: Optional[int] = None, max_connections_per_host: Optional[int] = None) -> None:
    """Initialize the runtime.
    
    Args:
        worker_threads: Number of worker threads to use (default: number of CPUs)
        max_connections_per_host: Maximum number of connections per host (default: 10)
    """
    global _initialized, _worker_threads, _max_connections_per_host
    
    import multiprocessing
    
    # Set default values if not provided
    if worker_threads is None:
        worker_threads = multiprocessing.cpu_count()
    
    if max_connections_per_host is None:
        max_connections_per_host = 10
    
    # Store configuration
    _worker_threads = worker_threads
    _max_connections_per_host = max_connections_per_host
    
    # Initialize the event loop policy
    if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy') and isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsSelectorEventLoopPolicy):
        # On Windows, use the selector event loop policy
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    _initialized = True

def init_with_settings(worker_threads: Optional[int] = None, max_connections_per_host: Optional[int] = None) -> None:
    """Initialize the runtime with specific settings.
    
    Args:
        worker_threads: Number of worker threads to use (default: number of CPUs)
        max_connections_per_host: Maximum number of connections per host (default: 10)
    """
    init(worker_threads, max_connections_per_host)

def get_worker_threads() -> int:
    """Get the number of worker threads.
    
    Returns:
        The number of worker threads
    """
    global _worker_threads
    
    if not _initialized:
        init()
    
    return _worker_threads

def get_max_connections_per_host() -> int:
    """Get the maximum number of connections per host.
    
    Returns:
        The maximum number of connections per host
    """
    global _max_connections_per_host
    
    if not _initialized:
        init()
    
    return _max_connections_per_host

def set_gil_state(held: bool) -> None:
    """Set the GIL state for optimization.
    
    Args:
        held: Whether the GIL is currently held
    """
    # This is a no-op in the Python implementation
    pass

def run_in_executor(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a function in the executor.
    
    Args:
        func: The function to run
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function
    """
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))
def run_async(coro: Any) -> Any:
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

# Initialize the runtime
init()