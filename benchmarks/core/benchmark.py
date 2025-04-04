"""
Core benchmarking engine for Pyroid.

This module provides the core functionality for running benchmarks with timeout support.
"""

import time
import signal
from contextlib import contextmanager
from typing import Any, Callable, List, Dict, Optional, Union, TypeVar

T = TypeVar('T')


class TimeoutException(Exception):
    """Exception raised when a benchmark times out."""
    pass


@contextmanager
def time_limit(seconds: int):
    """Context manager for setting a timeout on a block of code.
    
    Args:
        seconds: The timeout in seconds.
        
    Raises:
        TimeoutException: If the code block takes longer than the specified timeout.
    """
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class BenchmarkResult:
    """Class to store the result of a benchmark run."""
    
    def __init__(
        self, 
        name: str, 
        implementation: str, 
        duration_ms: Optional[float] = None, 
        timed_out: bool = False, 
        timeout_seconds: Optional[int] = None, 
        result: Any = None
    ):
        """Initialize a benchmark result.
        
        Args:
            name: The name of the benchmark.
            implementation: The implementation being benchmarked (e.g., "Python", "NumPy", "pyroid").
            duration_ms: The duration of the benchmark in milliseconds.
            timed_out: Whether the benchmark timed out.
            timeout_seconds: The timeout in seconds, if applicable.
            result: The result of the benchmark function, if applicable.
        """
        self.name = name
        self.implementation = implementation
        self.duration_ms = duration_ms
        self.timed_out = timed_out
        self.timeout_seconds = timeout_seconds
        self.result = result
    
    def speedup_vs(self, other_result: 'BenchmarkResult') -> Optional[float]:
        """Calculate the speedup factor compared to another benchmark result.
        
        Args:
            other_result: The benchmark result to compare against.
            
        Returns:
            The speedup factor, or None if either result timed out.
        """
        if self.timed_out or other_result.timed_out:
            return None
        return other_result.duration_ms / self.duration_ms if self.duration_ms > 0 else float('inf')
    
    @property
    def display_time(self) -> str:
        """Get a human-readable string representation of the benchmark time.
        
        Returns:
            A string representation of the time, or a timeout message.
        """
        if self.timed_out:
            return f"Timed out ({self.timeout_seconds}s)"
        return f"{self.duration_ms:.2f}ms"


class Benchmark:
    """Class for running benchmarks with timeout support."""
    
    def __init__(self, name: str, description: str):
        """Initialize a benchmark.
        
        Args:
            name: The name of the benchmark.
            description: A description of the benchmark.
        """
        self.name = name
        self.description = description
        self.results: List[BenchmarkResult] = []
    
    def run_test(
        self, 
        name: str, 
        implementation: str, 
        func: Callable[..., T], 
        timeout: int = 10, 
        *args: Any, 
        **kwargs: Any
    ) -> BenchmarkResult:
        """Run a benchmark test with timeout support.
        
        Args:
            name: The name of the test.
            implementation: The implementation being tested (e.g., "Python", "NumPy", "pyroid").
            func: The function to benchmark.
            timeout: The timeout in seconds.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
            
        Returns:
            A BenchmarkResult object containing the results of the test.
        """
        try:
            with time_limit(timeout):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                
                benchmark_result = BenchmarkResult(
                    name=name, 
                    implementation=implementation, 
                    duration_ms=duration_ms, 
                    result=result
                )
                self.results.append(benchmark_result)
                return benchmark_result
        except TimeoutException:
            benchmark_result = BenchmarkResult(
                name=name, 
                implementation=implementation, 
                timed_out=True, 
                timeout_seconds=timeout
            )
            self.results.append(benchmark_result)
            return benchmark_result
    
    def compare_results(self) -> Optional[List[Dict[str, Any]]]:
        """Compare the results of all tests in this benchmark.
        
        Returns:
            A list of dictionaries containing comparison data, or None if there are fewer than 2 results.
        """
        if len(self.results) < 2:
            return None
            
        baseline = next((r for r in self.results if r.implementation == "Python"), None)
        if not baseline:
            baseline = self.results[0]
            
        comparisons = []
        for result in self.results:
            speedup = None
            if not result.timed_out and not baseline.timed_out:
                speedup = result.speedup_vs(baseline)
                
            comparisons.append({
                "name": result.name,
                "implementation": result.implementation,
                "duration_ms": result.duration_ms,
                "timed_out": result.timed_out,
                "timeout_seconds": result.timeout_seconds,
                "display_time": result.display_time,
                "speedup": speedup
            })
            
        return comparisons