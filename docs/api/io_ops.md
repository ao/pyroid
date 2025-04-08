# I/O Operations

This document provides detailed information about the I/O operations available in Pyroid.

## File Operations

### Read File

```python
pyroid.io.read_file(path)
```

Reads the contents of a file.

**Parameters:**
- `path` (str): The path to the file to read.

**Returns:**
- The contents of the file as a string.

**Example:**
```python
import pyroid

# Read a file
content = pyroid.io.read_file("example.txt")
print(f"File content: {content}")
```

### Write File

```python
pyroid.io.write_file(path, content)
```

Writes content to a file.

**Parameters:**
- `path` (str): The path to the file to write.
- `content` (str): The content to write to the file.

**Returns:**
- None

**Example:**
```python
import pyroid

# Write to a file
pyroid.io.write_file("output.txt", "Hello, world!")
```

### Read Multiple Files

```python
pyroid.io.read_files(paths)
```

Reads the contents of multiple files.

**Parameters:**
- `paths` (list): A list of file paths to read.

**Returns:**
- A dictionary mapping file paths to their contents.

**Example:**
```python
import pyroid

# Read multiple files
files = ["file1.txt", "file2.txt", "file3.txt"]
contents = pyroid.io.read_files(files)
for path, content in contents.items():
    print(f"{path}: {content}")
```

## Network Operations

### HTTP GET

```python
pyroid.io.get(url)
```

Performs an HTTP GET request.

**Parameters:**
- `url` (str): The URL to request.

**Returns:**
- The response body as a string.

**Example:**
```python
import pyroid

# Make a GET request
response = pyroid.io.get("https://example.com")
print(f"Response length: {len(response)}")
```

## Async Operations

### Sleep

```python
await pyroid.io.sleep(seconds)
```

Asynchronously sleeps for the specified number of seconds.

**Parameters:**
- `seconds` (float): The number of seconds to sleep.

**Returns:**
- None

**Example:**
```python
import asyncio
import pyroid

async def main():
    print("Sleeping for 1 second...")
    await pyroid.io.sleep(1.0)
    print("Awake!")

asyncio.run(main())
```

### Read File Async

```python
await pyroid.io.read_file_async(path)
```

Asynchronously reads the contents of a file.

**Parameters:**
- `path` (str): The path to the file to read.

**Returns:**
- The contents of the file as a string.

**Example:**
```python
import asyncio
import pyroid

async def main():
    content = await pyroid.io.read_file_async("example.txt")
    print(f"File content: {content}")

asyncio.run(main())
```

## Advanced I/O Operations

### Streaming File Read

```python
pyroid.io.stream_read_file(path, chunk_size=4096)
```

Reads a file in chunks, yielding each chunk.

**Parameters:**
- `path` (str): The path to the file to read.
- `chunk_size` (int, optional): The size of each chunk in bytes. Default is 4096.

**Returns:**
- A generator yielding chunks of the file.

**Example:**
```python
import pyroid

# Stream read a large file
for chunk in pyroid.io.stream_read_file("large_file.txt"):
    print(f"Chunk length: {len(chunk)}")
```

### Streaming File Write

```python
pyroid.io.stream_write_file(path, chunks)
```

Writes chunks of data to a file.

**Parameters:**
- `path` (str): The path to the file to write.
- `chunks` (iterable): An iterable of chunks to write.

**Returns:**
- None

**Example:**
```python
import pyroid

# Generate chunks
def generate_chunks():
    for i in range(10):
        yield f"Chunk {i}\n"

# Stream write to a file
pyroid.io.stream_write_file("output.txt", generate_chunks())
```

### File Copy

```python
pyroid.io.copy_file(src, dst)
```

Copies a file from source to destination.

**Parameters:**
- `src` (str): The path to the source file.
- `dst` (str): The path to the destination file.

**Returns:**
- None

**Example:**
```python
import pyroid

# Copy a file
pyroid.io.copy_file("source.txt", "destination.txt")
```

### File Move

```python
pyroid.io.move_file(src, dst)
```

Moves a file from source to destination.

**Parameters:**
- `src` (str): The path to the source file.
- `dst` (str): The path to the destination file.

**Returns:**
- None

**Example:**
```python
import pyroid

# Move a file
pyroid.io.move_file("source.txt", "destination.txt")
```

### File Delete

```python
pyroid.io.delete_file(path)
```

Deletes a file.

**Parameters:**
- `path` (str): The path to the file to delete.

**Returns:**
- None

**Example:**
```python
import pyroid

# Delete a file
pyroid.io.delete_file("file_to_delete.txt")
```

### Directory Operations

```python
pyroid.io.create_directory(path)
pyroid.io.list_directory(path)
pyroid.io.delete_directory(path, recursive=False)
```

Operations for working with directories.

**Parameters:**
- `path` (str): The path to the directory.
- `recursive` (bool, optional): Whether to recursively delete directories. Default is False.

**Returns:**
- `list_directory` returns a list of file and directory names.
- Other functions return None.

**Example:**
```python
import pyroid

# Create a directory
pyroid.io.create_directory("new_directory")

# List files in a directory
files = pyroid.io.list_directory(".")
print(f"Files: {files}")

# Delete a directory
pyroid.io.delete_directory("directory_to_delete")
```

### File Compression

```python
pyroid.io.compress_file(src, dst, format="gzip")
pyroid.io.decompress_file(src, dst, format="gzip")
```

Compresses or decompresses a file.

**Parameters:**
- `src` (str): The path to the source file.
- `dst` (str): The path to the destination file.
- `format` (str, optional): The compression format. One of "gzip", "bzip2", "xz". Default is "gzip".

**Returns:**
- None

**Example:**
```python
import pyroid

# Compress a file
pyroid.io.compress_file("large_file.txt", "large_file.txt.gz")

# Decompress a file
pyroid.io.decompress_file("large_file.txt.gz", "large_file.txt")