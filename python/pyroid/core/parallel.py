"""
Parallel processing utilities for Pyroid.

This module provides high-performance parallel processing utilities.
"""

import multiprocessing
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast
import numpy as np

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

class BatchProcessor:
    """A batch processor for parallel operations."""
    
    def __init__(self, batch_size: int = 1000, max_workers: Optional[int] = None, adaptive: bool = True):
        """Create a new batch processor.
        
        Args:
            batch_size: The size of each batch
            max_workers: Maximum number of worker processes/threads (default: number of CPUs)
            adaptive: Whether to adaptively adjust the batch size based on performance
        """
        self.batch_size = batch_size
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.adaptive = adaptive
        self.performance_metrics = {
            "batch_times": [],
            "optimal_batch_size": batch_size
        }
    
    def optimize_batch_size(self, data_size: int) -> int:
        """Optimize batch size based on performance metrics.
        
        Args:
            data_size: The size of the data to process
            
        Returns:
            The optimized batch size
        """
        if not self.adaptive:
            return self.batch_size
        
        # Use previously optimized batch size or calculate a new one
        optimal_size = self.performance_metrics["optimal_batch_size"]
        
        # Ensure we don't create too many or too few batches
        cpus = multiprocessing.cpu_count()
        min_batches = cpus * 2  # At least 2 batches per CPU
        max_batches = cpus * 8  # At most 8 batches per CPU
        
        calculated_size = data_size // min_batches if min_batches > 0 else data_size
        adjusted_size = data_size // max_batches if data_size // optimal_size > max_batches and max_batches > 0 else optimal_size
        
        return max(1, min(calculated_size, adjusted_size))
    
    def map(self, items: List[T], func: Callable[[T], R], use_processes: bool = False) -> List[R]:
        """Map a function over a list of items in parallel.
        
        Args:
            items: The items to process
            func: The function to apply to each item
            use_processes: Whether to use processes instead of threads
            
        Returns:
            A list of results
        """
        # Calculate optimal batch size
        total_items = len(items)
        batch_size = self.optimize_batch_size(total_items)
        
        # Create batches
        batches = [items[i:i + batch_size] for i in range(0, total_items, batch_size)]
        
        # Process batches in parallel
        executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        with executor_cls(max_workers=self.max_workers) as executor:
            # Process each batch
            batch_results = list(executor.map(
                lambda batch: [func(item) for item in batch],
                batches
            ))
        
        # Update batch size metrics
        if self.adaptive:
            self.performance_metrics["optimal_batch_size"] = batch_size
        
        # Flatten results
        return [result for batch_result in batch_results for result in batch_result]
    
    def filter(self, items: List[T], func: Callable[[T], bool], use_processes: bool = False) -> List[T]:
        """Filter a list of items in parallel.
        
        Args:
            items: The items to filter
            func: The function to apply to each item
            use_processes: Whether to use processes instead of threads
            
        Returns:
            A list of items for which func returns True
        """
        # Create a wrapper function that applies the filter to each item
        def filter_func(batch: List[T]) -> List[T]:
            return [item for item in batch if func(item)]
        
        # Calculate optimal batch size
        total_items = len(items)
        batch_size = self.optimize_batch_size(total_items)
        
        # Create batches
        batches = [items[i:i + batch_size] for i in range(0, total_items, batch_size)]
        
        # Process batches in parallel
        executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        with executor_cls(max_workers=self.max_workers) as executor:
            # Process each batch
            batch_results = list(executor.map(filter_func, batches))
        
        # Update batch size metrics
        if self.adaptive:
            self.performance_metrics["optimal_batch_size"] = batch_size
        
        # Flatten results
        return [item for batch_result in batch_results for item in batch_result]
    
    def sort(self, items: List[T], key: Optional[Callable[[T], Any]] = None, reverse: bool = False, use_processes: bool = False) -> List[T]:
        """Sort a list of items in parallel.
        
        Args:
            items: The items to sort
            key: A function that extracts a comparison key from each item
            reverse: Whether to sort in reverse order
            use_processes: Whether to use processes instead of threads
            
        Returns:
            A sorted list of items
        """
        # Make a copy of the items to avoid modifying the original list
        items_copy = list(items)
        
        # Sort the items
        if key is not None:
            # Sort with key function
            items_copy.sort(key=key, reverse=reverse)
        else:
            # Sort without key function
            items_copy.sort(reverse=reverse)
        
        return items_copy

def parallel_map(items: List[T], func: Callable[[T], R], batch_size: Optional[int] = None, use_processes: bool = False) -> List[R]:
    """Process a list of items in parallel.
    
    Args:
        items: The items to process
        func: The function to apply to each item
        batch_size: The size of each batch (default: auto)
        use_processes: Whether to use processes instead of threads
        
    Returns:
        A list of results
    """
    processor = BatchProcessor(batch_size or 1000, adaptive=True)
    return processor.map(items, func, use_processes)

def parallel_filter(items: List[T], func: Callable[[T], bool], batch_size: Optional[int] = None, use_processes: bool = False) -> List[T]:
    """Filter a list of items in parallel.
    
    Args:
        items: The items to filter
        func: The function to apply to each item
        batch_size: The size of each batch (default: auto)
        use_processes: Whether to use processes instead of threads
        
    Returns:
        A list of items for which func returns True
    """
    processor = BatchProcessor(batch_size or 1000, adaptive=True)
    return processor.filter(items, func, use_processes)

def parallel_sort(items: List[T], key: Optional[Callable[[T], Any]] = None, reverse: bool = False, batch_size: Optional[int] = None, use_processes: bool = False) -> List[T]:
    """Sort a list of items in parallel.
    
    Args:
        items: The items to sort
        key: A function that extracts a comparison key from each item
        reverse: Whether to sort in reverse order
        batch_size: The size of each batch (default: auto)
        use_processes: Whether to use processes instead of threads
        
    Returns:
        A sorted list of items
    """
    processor = BatchProcessor(batch_size or 1000, adaptive=True)
    return processor.sort(items, key, reverse, use_processes)