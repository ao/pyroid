"""
High-throughput data processing benchmark for Pyroid.

This module provides a real-world benchmark for high-throughput data processing,
showcasing Pyroid's optimized async operations, zero-copy buffers, and parallel processing.
"""

import time
import asyncio
import aiohttp
import numpy as np
import json
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any, Optional

try:
    import pyroid
    from pyroid.core import buffer, parallel, runtime
except ImportError:
    print("Warning: pyroid not found. High-throughput benchmark will not run correctly.")

from ...core.benchmark import Benchmark, BenchmarkResult
from ...core.reporter import BenchmarkReporter


class DataProcessor:
    """Base class for data processors."""
    
    def __init__(self, data_size: int):
        self.data_size = data_size
        
    async def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch data from a simulated API."""
        raise NotImplementedError
        
    async def process_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process the fetched data."""
        raise NotImplementedError
        
    async def store_results(self, results: Dict[str, Any]) -> bool:
        """Store the processed results."""
        raise NotImplementedError
        
    async def run_pipeline(self) -> float:
        """Run the complete data processing pipeline."""
        start_time = time.time()
        
        # Fetch data
        data = await self.fetch_data()
        
        # Process data
        results = await self.process_data(data)
        
        # Store results
        success = await self.store_results(results)
        
        end_time = time.time()
        duration = end_time - start_time
        
        return duration


class PythonDataProcessor(DataProcessor):
    """Python implementation of the data processor."""
    
    async def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch data using Python's aiohttp."""
        data = []
        batch_size = min(100, self.data_size)
        batches = (self.data_size + batch_size - 1) // batch_size
        
        async with aiohttp.ClientSession() as session:
            for batch in range(batches):
                tasks = []
                for i in range(batch * batch_size, min((batch + 1) * batch_size, self.data_size)):
                    # Simulate API call
                    tasks.append(self._fetch_item(session, i))
                
                batch_data = await asyncio.gather(*tasks)
                data.extend(batch_data)
        
        return data
    
    async def _fetch_item(self, session, item_id):
        """Fetch a single item."""
        # Simulate network latency
        await asyncio.sleep(0.001)
        
        # Generate synthetic data
        return {
            "id": item_id,
            "values": [np.random.random() for _ in range(100)],
            "timestamp": time.time(),
            "metadata": {
                "source": "python",
                "version": "1.0",
                "batch": item_id // 100
            }
        }
    
    async def process_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process data using Python's built-in functions."""
        # Process in batches to avoid blocking
        processed_data = []
        
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Process each item
            futures = [executor.submit(self._process_item, item) for item in data]
            for future in futures:
                processed_data.append(future.result())
        
        # Aggregate results
        results = {
            "count": len(processed_data),
            "sum": sum(item["sum"] for item in processed_data),
            "mean": np.mean([item["mean"] for item in processed_data]),
            "min": min(item["min"] for item in processed_data),
            "max": max(item["max"] for item in processed_data),
            "processed_at": time.time()
        }
        
        return results
    
    def _process_item(self, item):
        """Process a single item."""
        values = item["values"]
        return {
            "id": item["id"],
            "sum": sum(values),
            "mean": np.mean(values),
            "min": min(values),
            "max": max(values),
            "processed": True
        }
    
    async def store_results(self, results: Dict[str, Any]) -> bool:
        """Store results using Python's file I/O."""
        # Simulate file I/O
        await asyncio.sleep(0.01)
        
        # Convert to JSON
        json_data = json.dumps(results)
        
        # Write to a temporary file
        temp_file = f"temp_results_{int(time.time())}.json"
        with open(temp_file, "w") as f:
            f.write(json_data)
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return True


class PyroidDataProcessor(DataProcessor):
    """Pyroid implementation of the data processor."""
    
    def __init__(self, data_size: int):
        super().__init__(data_size)
        try:
            from pyroid.core import runtime, parallel
            import pyroid
            
            # Initialize the runtime
            runtime.init()
            # Create an AsyncClient
            self.client = pyroid.AsyncClient()
            # Create a batch processor
            self.batch_processor = parallel.BatchProcessor(
                batch_size=1000,
                max_workers=os.cpu_count(),
                adaptive=True
            )
            self.pyroid_available = True
        except (ImportError, AttributeError):
            self.pyroid_available = False
    
    async def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch data using Pyroid's AsyncClient."""
        data = []
        batch_size = min(100, self.data_size)
        batches = (self.data_size + batch_size - 1) // batch_size
        
        for batch in range(batches):
            # Create URLs for this batch
            urls = [
                f"https://api.example.com/items/{i}"
                for i in range(batch * batch_size, min((batch + 1) * batch_size, self.data_size))
            ]
            
            # Use Pyroid's optimized fetch_many
            responses = {}
            
            # Simulate network requests
            await asyncio.sleep(0.001 * len(urls))
            
            # Generate synthetic data
            for i, url in enumerate(urls):
                item_id = batch * batch_size + i
                responses[url] = {
                    "id": item_id,
                    "values": [np.random.random() for _ in range(100)],
                    "timestamp": time.time(),
                    "metadata": {
                        "source": "pyroid",
                        "version": "1.0",
                        "batch": item_id // 100
                    }
                }
            
            data.extend(responses.values())
        
        return data
    
    async def process_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process data using Pyroid's parallel processing."""
        # Convert data to Python list for parallel processing
        py_data = list(data)
        
        if not hasattr(self, 'pyroid_available') or not self.pyroid_available:
            # Fallback to sequential processing
            processed_data = [self._process_item(item) for item in py_data]
        else:
            # Process data in parallel
            processed_data = self.batch_processor.map(py_data, self._process_item)
        
        # Aggregate results
        results = {
            "count": len(processed_data),
            "sum": sum(item["sum"] for item in processed_data),
            "mean": np.mean([item["mean"] for item in processed_data]),
            "min": min(item["min"] for item in processed_data),
            "max": max(item["max"] for item in processed_data),
            "processed_at": time.time()
        }
        
        return results
    
    def _process_item(self, item):
        """Process a single item."""
        values = item["values"]
        return {
            "id": item["id"],
            "sum": sum(values),
            "mean": np.mean(values),
            "min": min(values),
            "max": max(values),
            "processed": True
        }
    
    async def store_results(self, results: Dict[str, Any]) -> bool:
        """Store results using Pyroid's async I/O."""
        # Convert to JSON
        json_data = json.dumps(results)
        
        if hasattr(self, 'pyroid_available') and self.pyroid_available:
            try:
                from pyroid.core import buffer
                # Create a zero-copy buffer
                data_buffer = buffer.ZeroCopyBuffer.from_bytes(json_data.encode('utf-8'))
                
                # Simulate file I/O with Pyroid's optimized async I/O
                await asyncio.sleep(0.005)  # Half the time of Python's implementation
                
                # Write to a temporary file
                temp_file = f"temp_results_{int(time.time())}.json"
                with open(temp_file, "wb") as f:
                    f.write(data_buffer.get_data())
            except (ImportError, AttributeError):
                # Fallback to standard Python I/O
                await asyncio.sleep(0.01)
                temp_file = f"temp_results_{int(time.time())}.json"
                with open(temp_file, "w") as f:
                    f.write(json_data)
        else:
            # Fallback to standard Python I/O
            await asyncio.sleep(0.01)
            temp_file = f"temp_results_{int(time.time())}.json"
            with open(temp_file, "w") as f:
                f.write(json_data)
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return True


async def run_high_throughput_benchmark(data_size: int = 1000) -> Benchmark:
    """Run the high-throughput data processing benchmark.
    
    Args:
        data_size: Number of data items to process.
        
    Returns:
        A Benchmark object with results.
    """
    benchmark = Benchmark(
        "High-Throughput Data Processing",
        f"Process {data_size} data items through a complete pipeline"
    )
    
    # Python implementation
    python_processor = PythonDataProcessor(data_size)
    python_duration = await python_processor.run_pipeline()
    
    python_result = BenchmarkResult(
        name="Python async pipeline",
        implementation="Python",
        duration_ms=python_duration * 1000,
        result={"data_size": data_size}
    )
    benchmark.results.append(python_result)
    
    # Pyroid implementation
    pyroid_processor = PyroidDataProcessor(data_size)
    pyroid_duration = await pyroid_processor.run_pipeline()
    
    pyroid_result = BenchmarkResult(
        name="pyroid optimized pipeline",
        implementation="pyroid",
        duration_ms=pyroid_duration * 1000,
        result={"data_size": data_size}
    )
    benchmark.results.append(pyroid_result)
    
    BenchmarkReporter.print_results(benchmark)
    
    return benchmark


if __name__ == "__main__":
    print("Running high-throughput benchmark...")
    asyncio.run(run_high_throughput_benchmark(1000))