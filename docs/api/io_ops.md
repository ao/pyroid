# I/O Operations

The I/O operations module provides high-performance implementations of common file input/output tasks. These operations are implemented in Rust and are designed to be significantly faster than their Python equivalents, especially for batch processing of multiple files or large datasets.

## Overview

The I/O operations module provides the following key functions:

- `parallel_read_csv`: Read multiple CSV files in parallel
- `parallel_json_parse`: Parse multiple JSON strings in parallel
- `parallel_compress`: Compress data in parallel
- `parallel_decompress`: Decompress data in parallel

## API Reference

### parallel_read_csv

Read multiple CSV files in parallel.

```python
pyroid.parallel_read_csv(files, schema=None, has_header=True, delimiter=',')
```

#### Parameters

- `files`: A list of file paths to read
- `schema`: An optional schema dictionary mapping column names to types
  - Supported types: 'int', 'float', 'bool', 'string'
- `has_header`: Whether the CSV files have headers (default: True)
- `delimiter`: The delimiter character (default: ',')

#### Returns

A list of dictionaries, each containing the data from one CSV file.

#### Example

```python
import pyroid
import time
import pandas as pd
import os

# Create sample CSV files
os.makedirs("test_data", exist_ok=True)
for i in range(5):
    df = pd.DataFrame({
        'id': range(10000),
        'name': [f"item_{j}" for j in range(10000)],
        'value': [i * j / 100 for j in range(10000)],
        'flag': [j % 2 == 0 for j in range(10000)]
    })
    df.to_csv(f"test_data/test_file_{i}.csv", index=False)

# Define file paths
files = [f"test_data/test_file_{i}.csv" for i in range(5)]

# Define schema
schema = {
    'id': 'int',
    'value': 'float',
    'flag': 'bool'
}

# Compare with pandas
start = time.time()
pandas_data = [pd.read_csv(file) for file in files]
pandas_time = time.time() - start

start = time.time()
pyroid_data = pyroid.parallel_read_csv(files, schema)
pyroid_time = time.time() - start

print(f"Pandas time: {pandas_time:.2f}s, Pyroid time: {pyroid_time:.2f}s")
print(f"Speedup: {pandas_time / pyroid_time:.1f}x")

# Print the first few rows of the first file
print("\nPandas first file head:")
print(pandas_data[0].head())

print("\nPyroid first file sample:")
for col in pyroid_data[0]:
    print(f"{col}: {pyroid_data[0][col][:5]}")

# Clean up
for file in files:
    os.remove(file)
os.rmdir("test_data")
```

#### Schema Types

- **'int'**: Integer values
- **'float'**: Floating-point values
- **'bool'**: Boolean values ('true'/'false', '1'/'0', 'yes'/'no', 'y'/'n')
- **'string'**: String values (default if no schema is provided)

#### Performance Considerations

- `parallel_read_csv` is particularly efficient for reading multiple CSV files or large CSV files.
- The implementation processes each file in parallel, which can lead to significant performance improvements on multi-core systems.
- Specifying a schema can improve performance by avoiding type inference.
- For very large CSV files, memory usage can be a concern. Consider processing files in batches if memory is limited.

### parallel_json_parse

Parse multiple JSON strings in parallel.

```python
pyroid.parallel_json_parse(json_strings)
```

#### Parameters

- `json_strings`: A list of JSON strings to parse

#### Returns

A list of parsed JSON objects (as Python dictionaries).

#### Example

```python
import pyroid
import time
import json
import random

# Generate sample JSON strings
json_strings = []
for i in range(10000):
    data = {
        "id": i,
        "name": f"record_{i}",
        "values": [random.random() for _ in range(100)],
        "metadata": {
            "created": "2025-04-04",
            "version": "1.0",
            "tags": ["test", "benchmark", f"tag_{i}"]
        }
    }
    json_strings.append(json.dumps(data))

# Compare with Python's json module
start = time.time()
python_parsed = [json.loads(s) for s in json_strings]
python_time = time.time() - start

start = time.time()
pyroid_parsed = pyroid.parallel_json_parse(json_strings)
pyroid_time = time.time() - start

print(f"Python json time: {python_time:.2f}s, Pyroid time: {pyroid_time:.2f}s")
print(f"Speedup: {python_time / pyroid_time:.1f}x")

# Verify results
print("\nPython parsed first item:")
print(python_parsed[0])
print("\nPyroid parsed first item:")
print(pyroid_parsed[0])
```

#### Performance Considerations

- `parallel_json_parse` is particularly efficient for parsing multiple JSON strings or large JSON strings.
- The implementation processes each JSON string in parallel, which can lead to significant performance improvements on multi-core systems.
- The function uses the serde_json crate, which is one of the fastest JSON parsers available for Rust.
- For very large JSON strings, memory usage can be a concern. Consider processing strings in batches if memory is limited.

### parallel_compress

Compress data in parallel.

```python
pyroid.parallel_compress(data, method='gzip', level=6)
```

#### Parameters

- `data`: A list of bytes or strings to compress
- `method`: Compression method (default: 'gzip')
  - Supported methods: 'gzip', 'zlib'
- `level`: Compression level (1-9, default: 6)

#### Returns

A list of compressed data (as bytes).

#### Example

```python
import pyroid
import time
import gzip
import random
import string

# Generate sample data
def generate_random_text(size):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(size))

data = [generate_random_text(10000).encode() for _ in range(1000)]

# Compare with Python's gzip module
start = time.time()
python_compressed = [gzip.compress(d) for d in data]
python_time = time.time() - start

start = time.time()
pyroid_compressed = pyroid.parallel_compress(data, "gzip", 6)
pyroid_time = time.time() - start

print(f"Python gzip time: {python_time:.2f}s, Pyroid time: {pyroid_time:.2f}s")
print(f"Speedup: {python_time / pyroid_time:.1f}x")

# Calculate compression ratios
original_size = sum(len(d) for d in data)
python_size = sum(len(d) for d in python_compressed)
pyroid_size = sum(len(d) for d in pyroid_compressed)

print(f"\nOriginal size: {original_size / 1024:.2f} KB")
print(f"Python compressed size: {python_size / 1024:.2f} KB (ratio: {original_size / python_size:.2f}x)")
print(f"Pyroid compressed size: {pyroid_size / 1024:.2f} KB (ratio: {original_size / pyroid_size:.2f}x)")
```

#### Compression Methods

- **'gzip'**: GZIP compression format, widely supported and good for general-purpose compression
- **'zlib'**: ZLIB compression format, similar to GZIP but with a different header and checksum

#### Compression Levels

- **1**: Fastest compression, lowest compression ratio
- **6**: Default level, good balance between speed and compression ratio
- **9**: Slowest compression, highest compression ratio

#### Performance Considerations

- `parallel_compress` is particularly efficient for compressing multiple data items or large data items.
- The implementation processes each data item in parallel, which can lead to significant performance improvements on multi-core systems.
- Higher compression levels result in better compression ratios but slower compression times.
- For very large data items, memory usage can be a concern. Consider processing data in batches if memory is limited.

### parallel_decompress

Decompress data in parallel.

```python
pyroid.parallel_decompress(data, method='gzip')
```

#### Parameters

- `data`: A list of compressed bytes
- `method`: Compression method (default: 'gzip')
  - Supported methods: 'gzip', 'zlib'

#### Returns

A list of decompressed data (as bytes).

#### Example

```python
import pyroid
import time
import gzip
import random
import string

# Generate sample data
def generate_random_text(size):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(size))

original_data = [generate_random_text(10000).encode() for _ in range(1000)]

# Compress data
compressed_data = pyroid.parallel_compress(original_data, "gzip", 6)

# Compare with Python's gzip module
start = time.time()
python_decompressed = [gzip.decompress(d) for d in compressed_data]
python_time = time.time() - start

start = time.time()
pyroid_decompressed = pyroid.parallel_decompress(compressed_data, "gzip")
pyroid_time = time.time() - start

print(f"Python gzip time: {python_time:.2f}s, Pyroid time: {pyroid_time:.2f}s")
print(f"Speedup: {python_time / pyroid_time:.1f}x")

# Verify results
all_match = all(o == p for o, p in zip(original_data, pyroid_decompressed))
print(f"\nAll decompressed data matches original: {all_match}")
```

#### Performance Considerations

- `parallel_decompress` is particularly efficient for decompressing multiple data items or large data items.
- The implementation processes each data item in parallel, which can lead to significant performance improvements on multi-core systems.
- Decompression is generally faster than compression, but still benefits from parallelization.
- For very large data items, memory usage can be a concern. Consider processing data in batches if memory is limited.

## Performance Comparison

The following table shows the performance comparison between Python standard library and pyroid for various I/O operations:

| Operation | Dataset Size | Python | pyroid | Speedup |
|-----------|-------------|--------|--------|---------|
| CSV Reading | 5 files, 10K rows each | 1500ms | 200ms | 7.5x |
| JSON Parsing | 10K strings, 100 values each | 2000ms | 250ms | 8.0x |
| Compression (gzip) | 1K strings, 10K chars each | 1200ms | 150ms | 8.0x |
| Decompression (gzip) | 1K compressed strings | 800ms | 100ms | 8.0x |

## Best Practices

1. **Specify schema for CSV files**: When reading CSV files, specify a schema to improve performance and ensure correct data types.

2. **Choose appropriate compression level**: For compression, choose a compression level that balances speed and compression ratio based on your needs.

3. **Batch process large files**: For very large files, consider processing them in batches to avoid memory issues.

4. **Reuse compressed data**: If you need to decompress the same data multiple times, store the compressed data and decompress it as needed.

5. **Use appropriate compression method**: Choose the compression method based on your needs. GZIP is widely supported, while ZLIB may be more efficient for certain use cases.

## Limitations

1. **Limited CSV parsing options**: The current implementation provides basic CSV parsing functionality, but lacks some advanced features like quoting and escaping.

2. **Limited compression methods**: The current implementation supports only GZIP and ZLIB compression methods.

3. **Memory usage**: For very large files or datasets, memory usage can be a concern. Consider processing data in batches if memory is limited.

4. **No streaming support**: The current implementation does not support streaming, which means that the entire file or data item must be loaded into memory.

## Examples

### Example 1: CSV Data Processing Pipeline

```python
import pyroid
import os
import matplotlib.pyplot as plt

def process_csv_data(input_dir, output_file):
    """Process multiple CSV files and generate a summary."""
    # Get all CSV files
    csv_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    # Define schema
    schema = {
        'id': 'int',
        'timestamp': 'string',
        'temperature': 'float',
        'humidity': 'float',
        'pressure': 'float',
        'status': 'string'
    }
    
    # Read all CSV files
    print(f"Reading {len(csv_files)} CSV files...")
    data = pyroid.parallel_read_csv(csv_files, schema)
    
    # Process data
    all_temps = []
    all_humidity = []
    all_pressure = []
    status_counts = {}
    
    for file_data in data:
        # Extract data
        temps = file_data['temperature']
        humidity = file_data['humidity']
        pressure = file_data['pressure']
        statuses = file_data['status']
        
        # Collect data for analysis
        all_temps.extend(temps)
        all_humidity.extend(humidity)
        all_pressure.extend(pressure)
        
        # Count statuses
        for status in statuses:
            status_counts[status] = status_counts.get(status, 0) + 1
    
    # Calculate statistics
    avg_temp = sum(all_temps) / len(all_temps)
    avg_humidity = sum(all_humidity) / len(all_humidity)
    avg_pressure = sum(all_pressure) / len(all_pressure)
    
    # Generate report
    with open(output_file, 'w') as f:
        f.write("# Sensor Data Analysis\n\n")
        f.write(f"Total records: {len(all_temps)}\n")
        f.write(f"Average temperature: {avg_temp:.2f}°C\n")
        f.write(f"Average humidity: {avg_humidity:.2f}%\n")
        f.write(f"Average pressure: {avg_pressure:.2f} hPa\n\n")
        
        f.write("## Status Counts\n\n")
        for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {status}: {count}\n")
    
    # Generate plots
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(all_temps, bins=20)
    plt.title('Temperature Distribution')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Count')
    
    plt.subplot(1, 3, 2)
    plt.hist(all_humidity, bins=20)
    plt.title('Humidity Distribution')
    plt.xlabel('Humidity (%)')
    plt.ylabel('Count')
    
    plt.subplot(1, 3, 3)
    plt.hist(all_pressure, bins=20)
    plt.title('Pressure Distribution')
    plt.xlabel('Pressure (hPa)')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.splitext(output_file)[0] + '.png')
    
    print(f"Analysis complete. Report saved to {output_file}")

# Example usage
process_csv_data("sensor_data", "sensor_analysis.md")
```

### Example 2: JSON Data Compression and Storage

```python
import pyroid
import os
import json
import time
import random

def compress_and_store_json(data_generator, num_records, records_per_file, output_dir):
    """Generate, compress, and store JSON data."""
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    
    # Generate and process data in batches
    for batch_idx in range(0, num_records, records_per_file):
        batch_end = min(batch_idx + records_per_file, num_records)
        batch_size = batch_end - batch_idx
        
        print(f"Processing batch {batch_idx // records_per_file + 1}: records {batch_idx} to {batch_end - 1}")
        
        # Generate batch of records
        records = [data_generator(i) for i in range(batch_idx, batch_end)]
        
        # Convert to JSON strings
        json_strings = [json.dumps(record) for record in records]
        
        # Compress JSON strings
        compressed_data = pyroid.parallel_compress(json_strings, "gzip", 9)
        
        # Save compressed data
        output_file = os.path.join(output_dir, f"batch_{batch_idx // records_per_file + 1}.gz")
        with open(output_file, 'wb') as f:
            # Simple format: each compressed item is preceded by its length as a 4-byte integer
            for item in compressed_data:
                f.write(len(item).to_bytes(4, byteorder='little'))
                f.write(item)
        
        # Calculate compression ratio
        original_size = sum(len(s.encode()) for s in json_strings)
        compressed_size = sum(len(d) for d in compressed_data)
        ratio = original_size / compressed_size
        
        print(f"  Original size: {original_size / 1024:.2f} KB")
        print(f"  Compressed size: {compressed_size / 1024:.2f} KB")
        print(f"  Compression ratio: {ratio:.2f}x")
    
    total_time = time.time() - start_time
    print(f"\nProcessed {num_records} records in {total_time:.2f} seconds")
    print(f"Average processing rate: {num_records / total_time:.2f} records/second")

def read_compressed_json(input_dir):
    """Read and decompress stored JSON data."""
    # Get all compressed files
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.gz')]
    files.sort()  # Ensure files are processed in order
    
    start_time = time.time()
    all_records = []
    
    for file_path in files:
        print(f"Processing {os.path.basename(file_path)}")
        
        # Read compressed data
        compressed_items = []
        with open(file_path, 'rb') as f:
            while True:
                # Read length (4 bytes)
                length_bytes = f.read(4)
                if not length_bytes:
                    break  # End of file
                
                length = int.from_bytes(length_bytes, byteorder='little')
                item = f.read(length)
                compressed_items.append(item)
        
        # Decompress data
        json_strings = pyroid.parallel_decompress(compressed_items, "gzip")
        
        # Parse JSON
        records = pyroid.parallel_json_parse(json_strings)
        all_records.extend(records)
    
    total_time = time.time() - start_time
    print(f"\nRead {len(all_records)} records in {total_time:.2f} seconds")
    print(f"Average reading rate: {len(all_records) / total_time:.2f} records/second")
    
    return all_records

# Example data generator
def generate_sensor_data(idx):
    return {
        "id": idx,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() - random.randint(0, 86400))),
        "device_id": f"DEVICE_{random.randint(1, 100):03d}",
        "location": {
            "latitude": random.uniform(-90, 90),
            "longitude": random.uniform(-180, 180),
            "altitude": random.uniform(0, 1000)
        },
        "measurements": {
            "temperature": random.uniform(-10, 40),
            "humidity": random.uniform(0, 100),
            "pressure": random.uniform(950, 1050),
            "light": random.uniform(0, 1000),
            "noise": random.uniform(30, 90)
        },
        "battery": random.uniform(0, 100),
        "status": random.choice(["OK", "WARNING", "ERROR", "CRITICAL"]),
        "tags": random.sample(["indoor", "outdoor", "mobile", "fixed", "prototype", "production"], k=random.randint(1, 3))
    }

# Example usage
compress_and_store_json(generate_sensor_data, 100000, 10000, "compressed_data")
records = read_compressed_json("compressed_data")
```

### Example 3: Parallel File Compression Utility

```python
import pyroid
import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

def compress_file(file_path, method, level, delete_original=False):
    """Compress a single file using pyroid."""
    # Read file
    with open(file_path, 'rb') as f:
        data = f.read()
    
    # Compress data
    compressed = pyroid.parallel_compress([data], method, level)[0]
    
    # Determine output file name
    if method == 'gzip':
        output_path = file_path + '.gz'
    elif method == 'zlib':
        output_path = file_path + '.zlib'
    else:
        output_path = file_path + '.compressed'
    
    # Write compressed data
    with open(output_path, 'wb') as f:
        f.write(compressed)
    
    # Delete original if requested
    if delete_original:
        os.remove(file_path)
    
    # Calculate compression ratio
    original_size = len(data)
    compressed_size = len(compressed)
    ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
    
    return {
        'file': file_path,
        'original_size': original_size,
        'compressed_size': compressed_size,
        'ratio': ratio
    }

def decompress_file(file_path, method, delete_original=False):
    """Decompress a single file using pyroid."""
    # Read file
    with open(file_path, 'rb') as f:
        data = f.read()
    
    # Decompress data
    decompressed = pyroid.parallel_decompress([data], method)[0]
    
    # Determine output file name
    if file_path.endswith('.gz') and method == 'gzip':
        output_path = file_path[:-3]
    elif file_path.endswith('.zlib') and method == 'zlib':
        output_path = file_path[:-5]
    else:
        output_path = file_path + '.decompressed'
    
    # Write decompressed data
    with open(output_path, 'wb') as f:
        f.write(decompressed)
    
    # Delete original if requested
    if delete_original:
        os.remove(file_path)
    
    return {
        'file': file_path,
        'compressed_size': len(data),
        'decompressed_size': len(decompressed)
    }

def process_directory(directory, action, method, level=6, delete_original=False, recursive=False):
    """Process all files in a directory."""
    # Get all files
    files = []
    if recursive:
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                files.append(os.path.join(root, filename))
    else:
        files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Filter files based on action
    if action == 'compress':
        # Exclude already compressed files
        files = [f for f in files if not (f.endswith('.gz') or f.endswith('.zlib') or f.endswith('.compressed'))]
    elif action == 'decompress':
        # Only include compressed files
        if method == 'gzip':
            files = [f for f in files if f.endswith('.gz')]
        elif method == 'zlib':
            files = [f for f in files if f.endswith('.zlib')]
        else:
            files = [f for f in files if f.endswith('.gz') or f.endswith('.zlib') or f.endswith('.compressed')]
    
    if not files:
        print(f"No {'un' if action == 'compress' else ''}compressed files found in {directory}")
        return
    
    # Process files in parallel using ThreadPoolExecutor
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        if action == 'compress':
            futures = [executor.submit(compress_file, f, method, level, delete_original) for f in files]
        else:  # decompress
            futures = [executor.submit(decompress_file, f, method, delete_original) for f in files]
        
        for future in futures:
            results.append(future.result())
    
    total_time = time.time() - start_time
    
    # Print summary
    if action == 'compress':
        total_original = sum(r['original_size'] for r in results)
        total_compressed = sum(r['compressed_size'] for r in results)
        avg_ratio = total_original / total_compressed if total_compressed > 0 else float('inf')
        
        print(f"\nCompressed {len(results)} files in {total_time:.2f} seconds")
        print(f"Total original size: {total_original / 1024 / 1024:.2f} MB")
        print(f"Total compressed size: {total_compressed / 1024 / 1024:.2f} MB")
        print(f"Average compression ratio: {avg_ratio:.2f}x")
    else:  # decompress
        total_compressed = sum(r['compressed_size'] for r in results)
        total_decompressed = sum(r['decompressed_size'] for r in results)
        
        print(f"\nDecompressed {len(results)} files in {total_time:.2f} seconds")
        print(f"Total compressed size: {total_compressed / 1024 / 1024:.2f} MB")
        print(f"Total decompressed size: {total_decompressed / 1024 / 1024:.2f} MB")

# Example command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel file compression utility using pyroid")
    parser.add_argument("action", choices=["compress", "decompress"], help="Action to perform")
    parser.add_argument("directory", help="Directory containing files to process")
    parser.add_argument("--method", choices=["gzip", "zlib"], default="gzip", help="Compression method")
    parser.add_argument("--level", type=int, choices=range(1, 10), default=6, help="Compression level (1-9)")
    parser.add_argument("--delete", action="store_true", help="Delete original files after processing")
    parser.add_argument("--recursive", action="store_true", help="Process directories recursively")
    
    args = parser.parse_args()
    
    process_directory(args.directory, args.action, args.method, args.level, args.delete, args.recursive)