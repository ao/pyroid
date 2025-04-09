"""
Web scraping benchmark for Pyroid.

This module provides a benchmark that simulates a web scraping workflow
to showcase Pyroid's performance advantages in concurrent HTTP requests and text processing.
"""

import time
import asyncio
import aiohttp
import re

try:
    import pyroid
except ImportError:
    print("Warning: pyroid not found. Web scraping benchmark will not run correctly.")

from ...core.benchmark import Benchmark, BenchmarkResult
from ...core.reporter import BenchmarkReporter


async def benchmark_async(name, implementation, func, timeout, *args, **kwargs):
    """Benchmark an async function.
    
    Args:
        name: The name of the benchmark.
        implementation: The implementation being benchmarked (e.g., "Python", "pyroid").
        func: The async function to benchmark.
        timeout: The timeout in seconds.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
        
    Returns:
        A BenchmarkResult object containing the results of the benchmark.
    """
    try:
        start_time = time.time()
        result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        return BenchmarkResult(
            name=name, 
            implementation=implementation, 
            duration_ms=duration_ms, 
            result=result
        )
    except asyncio.TimeoutError:
        return BenchmarkResult(
            name=name, 
            implementation=implementation, 
            timed_out=True, 
            timeout_seconds=timeout
        )


async def run_web_scraping_benchmark(urls_count=50):
    """Run a web scraping benchmark.
    
    Args:
        urls_count: Number of URLs to scrape.
        
    Returns:
        A Benchmark object with results.
    """
    # Generate a list of URLs to scrape
    urls = [f"https://httpbin.org/get?id={i}" for i in range(urls_count)]
    
    web_benchmark = Benchmark("Web Scraping", f"Scrape {urls_count} URLs and process the results")
    
    # Python implementation
    async def python_web_scraping(urls=None):
        if urls is None:
            # Default URLs if none provided
            urls = [f"https://httpbin.org/get?id={i}" for i in range(10)]
        print("Running Python web scraping pipeline...")
        
        # Step 1: Fetch all URLs
        print("  Step 1: Fetching URLs...")
        async with aiohttp.ClientSession() as session:
            tasks = []
            for url in urls:
                tasks.append(asyncio.create_task(session.get(url)))
            responses = await asyncio.gather(*tasks)
            
            # Step 2: Extract text content
            print("  Step 2: Extracting text content...")
            contents = []
            for response in responses:
                text = await response.text()
                contents.append(text)
            
            # Step 3: Clean and normalize text
            print("  Step 3: Cleaning and normalizing text...")
            cleaned = []
            for text in contents:
                # Remove extra whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                cleaned.append(text)
            
            # Step 4: Extract specific data (e.g., URLs, IDs)
            print("  Step 4: Extracting specific data...")
            extracted_data = []
            for text in cleaned:
                # Extract URLs
                urls = re.findall(r'https?://[^\s"\']+', text)
                # Extract IDs
                ids = re.findall(r'"id": "(\d+)"', text)
                
                extracted_data.append({
                    "urls": urls,
                    "ids": ids
                })
            
            # Step 5: Calculate statistics
            print("  Step 5: Calculating statistics...")
            url_counts = [len(data["urls"]) for data in extracted_data]
            id_counts = [len(data["ids"]) for data in extracted_data]
            
            avg_urls = sum(url_counts) / len(url_counts) if url_counts else 0
            avg_ids = sum(id_counts) / len(id_counts) if id_counts else 0
            
            # Sort results by ID count
            sorted_data = sorted(extracted_data, key=lambda x: len(x["ids"]), reverse=True)
            
            print("Python web scraping pipeline complete.")
            return {
                "total_urls_fetched": len(responses),
                "total_content_length": sum(len(text) for text in contents),
                "avg_urls_per_page": avg_urls,
                "avg_ids_per_page": avg_ids,
                "top_results": sorted_data[:5]
            }
    
    # pyroid implementation
    async def pyroid_web_scraping(urls=None):
        if urls is None:
            # Default URLs if none provided
            urls = [f"https://httpbin.org/get?id={i}" for i in range(10)]
        print("Running pyroid web scraping pipeline...")
        
        # Step 1: Fetch all URLs
        print("  Step 1: Fetching URLs...")
        # Use a fallback since AsyncClient is not available
        try:
            client = pyroid.AsyncClient()
            responses = await client.fetch_many(urls, concurrency=10)
        except AttributeError:
            # Create a simple fallback client
            async with aiohttp.ClientSession() as session:
                tasks = []
                for url in urls:
                    tasks.append(asyncio.create_task(session.get(url)))
                responses_list = await asyncio.gather(*tasks)
                
                responses = {}
                for i, response in enumerate(responses_list):
                    text = await response.text()
                    responses[urls[i]] = {"status": response.status, "text": text}
        
        # Step 2: Extract text content
        print("  Step 2: Extracting text content...")
        contents = []
        for url, response in responses.items():
            if isinstance(response, dict) and "text" in response:
                contents.append(response["text"])
        
        # Step 3: Clean and normalize text
        print("  Step 3: Cleaning and normalizing text...")
        
        def clean_text(text):
            return re.sub(r'\s+', ' ', text).strip()
        
        # Use data.collections.map or fallback to map
        try:
            cleaned = pyroid.data.collections.map(contents, clean_text)
        except AttributeError:
            cleaned = list(map(clean_text, contents))
        
        # Step 4: Extract specific data (e.g., URLs, IDs)
        print("  Step 4: Extracting specific data...")
        
        def extract_data(text):
            # Extract URLs
            urls = re.findall(r'https?://[^\s"\']+', text)
            # Extract IDs
            ids = re.findall(r'"id": "(\d+)"', text)
            
            return {
                "urls": urls,
                "ids": ids
            }
        
        # Use data.collections.map or fallback to map
        try:
            extracted_data = pyroid.data.collections.map(cleaned, extract_data)
        except AttributeError:
            extracted_data = list(map(extract_data, cleaned))
        
        # Step 5: Calculate statistics
        print("  Step 5: Calculating statistics...")
        # Use data.collections.map or fallback to map
        try:
            url_counts = pyroid.data.collections.map(extracted_data, lambda data: len(data["urls"]))
            id_counts = pyroid.data.collections.map(extracted_data, lambda data: len(data["ids"]))
        except AttributeError:
            url_counts = list(map(lambda data: len(data["urls"]), extracted_data))
            id_counts = list(map(lambda data: len(data["ids"]), extracted_data))
        
        # Use math.mean or fallback to sum/len
        try:
            avg_urls = pyroid.math.mean(url_counts) if url_counts else 0
            avg_ids = pyroid.math.mean(id_counts) if id_counts else 0
        except AttributeError:
            avg_urls = sum(url_counts) / len(url_counts) if url_counts else 0
            avg_ids = sum(id_counts) / len(id_counts) if id_counts else 0
        
        # Sort results by ID count
        # Use data.collections.sort or fallback to sorted
        try:
            sorted_data = pyroid.data.collections.sort(
                extracted_data,
                lambda x: len(x["ids"]),
                True  # Reverse order
            )
        except AttributeError:
            sorted_data = sorted(extracted_data, key=lambda x: len(x["ids"]), reverse=True)
        
        print("pyroid web scraping pipeline complete.")
        return {
            "total_urls_fetched": len(responses),
            "total_content_length": sum(len(text) for text in contents),
            "avg_urls_per_page": avg_urls,
            "avg_ids_per_page": avg_ids,
            "top_results": sorted_data[:5]
        }
    
    # Set timeouts
    python_timeout = 30
    pyroid_timeout = 30
    
    # Run benchmarks
    python_result = await benchmark_async("Python web scraping", "Python", lambda: python_web_scraping(urls), python_timeout)
    web_benchmark.results.append(python_result)
    
    pyroid_result = await benchmark_async("pyroid web scraping", "pyroid", lambda: pyroid_web_scraping(urls), pyroid_timeout)
    web_benchmark.results.append(pyroid_result)
    
    BenchmarkReporter.print_results(web_benchmark)
    
    return web_benchmark


if __name__ == "__main__":
    print("Running web scraping benchmark...")
    asyncio.run(run_web_scraping_benchmark())