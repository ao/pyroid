# Pyroid Benchmark Suite

This directory contains the benchmark suite for Pyroid, designed to showcase the performance advantages of Pyroid compared to pure Python implementations.

## Overview

The benchmark suite provides:

- Comparative benchmarks between Pyroid, pure Python, and NumPy implementations
- Timeout support for slow Python operations
- Comprehensive visualizations for marketing materials
- Real-world scenario benchmarks that simulate practical use cases

## Quick Start

### Jupyter Notebook

The easiest way to run benchmarks and visualize results is using the Jupyter notebook:

```bash
# Install Jupyter if you don't have it
pip install jupyter

# Launch the notebook
jupyter notebook benchmarks/pyroid_benchmarks.ipynb
```

The notebook includes:
- Interactive benchmarks for math, string, and data operations
- Real-world scenario benchmarks
- Visualizations of performance comparisons
- Summary of speedup factors

### Command-line Benchmarks

For more comprehensive benchmarks, you can use the command-line tools:

```bash
# Run all benchmarks
python benchmarks/run_benchmarks.py

# Run specific benchmark suites
python benchmarks/run_benchmarks.py --suite math
python benchmarks/run_benchmarks.py --suite string
python benchmarks/run_benchmarks.py --suite data
python benchmarks/run_benchmarks.py --suite async
python benchmarks/run_benchmarks.py --suite scenarios

# Run with different dataset sizes
python benchmarks/run_benchmarks.py --size small
python benchmarks/run_benchmarks.py --size medium
python benchmarks/run_benchmarks.py --size large
```

### Demo Script

For a quick demonstration of Pyroid's performance advantages:

```bash
python benchmarks/demo.py
```

## Directory Structure

```
benchmarks/
├── benchmark_plan.md     # Detailed implementation plan
├── README.md             # This file
├── README_update.md      # Template for updating the main README
├── pyroid_benchmarks.ipynb # Jupyter notebook for interactive benchmarks
├── run_benchmarks.py     # Main entry point for command-line benchmarks
├── demo.py               # Demo script for showcasing performance
├── core/                 # Core benchmarking engine
├── suites/               # Benchmark suites
│   ├── math_benchmarks.py
│   ├── string_benchmarks.py
│   ├── data_benchmarks.py
│   ├── async_benchmarks.py
│   └── scenarios/        # Real-world scenario benchmarks
├── dashboard/            # Generated dashboard (after running benchmarks)
├── charts/               # Generated charts (after running benchmarks)
├── results/              # Generated results (after running benchmarks)
```

## Benchmark Categories

1. **Math Operations**
   - Parallel sum vs. Python sum vs. NumPy sum
   - Parallel product vs. Python product
   - Statistical functions (mean, std) vs. NumPy equivalents
   - Matrix multiplication vs. NumPy matmul

2. **String Operations**
   - Regex replacement vs. Python re
   - Text cleanup vs. Python string methods
   - Base64 encoding/decoding vs. Python base64

3. **Data Operations**
   - Parallel filter vs. Python filter
   - Parallel map vs. Python map
   - Parallel reduce vs. Python reduce
   - Parallel sort vs. Python sorted

4. **Async Operations**
   - HTTP requests vs. aiohttp
   - File operations vs. Python async file I/O
   - Task concurrency vs. asyncio.gather

5. **Real-world Scenarios**
   - Data Processing Pipeline
   - Web Scraping and Processing
   - Text Processing Pipeline
   - Scientific Computing