"""Streaming Parquet / JSONL readers and writers over Arrow batches (for the ``pyroid`` CLI).

Inputs are files, directories, globs or URIs (``s3://``, ``gs://``, ``hf://datasets/...``,
``memory://``; through fsspec when it has the protocol, else through pyarrow's filesystems),
resolved to :class:`Source` records by :func:`resolve_inputs` and read one file at a time, in
bounded memory, by :func:`iter_tables`. Outputs are written to a hidden sibling and renamed into
place once every handle on them is closed; renames and deletes are retried while Windows reports
the target as in use (a virus scanner or an indexer has it open for a moment).
"""

from __future__ import annotations

import contextlib
import fnmatch
import importlib
import json
import os
import re
import secrets
import time
import urllib.request
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa

PARQUET = (".parquet", ".pq")
JSONL = (".jsonl", ".ndjson", ".json")
COMPRESSED = {".gz": "gzip", ".bz2": "bz2", ".zst": "zstd", ".lz4": "lz4"}

# JSON Lines are parsed this many bytes at a time, cut at a line end (a longer line is read
# whole); pyarrow parses each chunk in blocks of JSON_BLOCK_SIZE bytes, in parallel.
JSON_CHUNK_SIZE = 8 << 20
JSON_BLOCK_SIZE = 1 << 20

_MAGIC = re.compile(r"[*?\[]")
_URI = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]+(?:://|::)")


def _codec(name: str | os.PathLike[str]) -> str | None:
    return COMPRESSED.get(Path(name).suffix.lower())


def file_format(path: str | os.PathLike[str]) -> str:
    """``"parquet"`` or ``"jsonl"``, from the file name (``.jsonl.gz`` etc. are JSONL)."""
    name = Path(path).name.lower()
    for ext in COMPRESSED:
        name = name.removesuffix(ext)
    if name.endswith(PARQUET):
        return "parquet"
    if name.endswith(JSONL):
        return "jsonl"
    raise ValueError(
        f"{path}: unknown format (expected .parquet/.pq or .jsonl/.ndjson/.json, "
        "optionally compressed)"
    )


def _known(name: str) -> bool:
    try:
        file_format(name)
    except ValueError:
        return False
    return True


# ------------------------------------------------------------------------------------- sources


@dataclass(frozen=True)
class Source:
    """One input file, as :func:`resolve_inputs` found it.

    Sources pickle (they are sent to worker processes); a remote one carries its filesystem's
    class and constructor arguments, and the worker re-creates the filesystem.

    Attributes:
        uri: The file as the user knows it: a local path (the spec joined with the path found
            under it) or the full URI (``s3://b/k/x.parquet``).
        name: The path relative to the spec's root, ``/``-separated
            (``sub/part-0001.parquet``): :func:`output_path` names outputs after it.
        size: The size in bytes, when the listing reports it.
        format: ``"parquet"`` or ``"jsonl"``.
    """

    uri: str
    name: str
    size: int | None
    format: str
    # None (a local file), an _Fsspec or a pyarrow FileSystem; and the path inside it.
    _fs: Any = field(default=None, repr=False, compare=False)
    _path: str = field(default="", repr=False, compare=False)


class _Fsspec:
    """An fsspec filesystem that pickles as its class and constructor arguments only.

    fsspec finds (in its instance cache) or re-creates the same filesystem when it is unpickled;
    some implementations' own pickles carry their whole listing cache, once per Source.
    """

    __slots__ = ("fs",)

    def __init__(self, fs: Any) -> None:
        self.fs = fs

    def __reduce__(self) -> tuple[Any, ...]:
        return _fsspec_instance, (type(self.fs), self.fs.storage_args, self.fs.storage_options)


def _fsspec_instance(cls: Any, args: Any, kwargs: Any) -> _Fsspec:
    return _Fsspec(cls(*args, **kwargs))


def resolve_inputs(
    specs: Sequence[str | os.PathLike[str]], *, exclude: Sequence[str | os.PathLike[str]] = ()
) -> list[Source]:
    """Return the Parquet and JSONL files that ``specs`` name.

    Each spec is a file, a directory (listed recursively) or a glob, as a local path or a URI
    (``s3://``, ``gs://``, ``hf://``, ``memory://``, ...; ``http(s)://`` names one file and
    ``file://`` is local). Only files with a known suffix (:func:`file_format`) are kept, and
    paths with a component below the spec's root that starts with ``.`` or ``_`` are skipped
    (``.pyroid/``, ``_SUCCESS``, ``_delta_log/``). A glob has ``*``, ``?`` and ``[...]`` within
    a path component and ``**`` across components; its root is its leading components without
    wildcards, and a directory it matches stands for every file below it (as when a shell
    expands it). Globs match the same way, case-sensitively, on every filesystem.

    Args:
        specs: The inputs, in order.
        exclude: Local paths: every local file at or under one of them is dropped (e.g. the
            command's own output and work directories).

    Returns:
        Each spec's files sorted by :attr:`Source.name`, the specs in the order given.

    Raises:
        FileNotFoundError: A spec matches no file.
        ValueError: The same file is listed twice, a single file has an unknown suffix, or no
            installed filesystem reads a URI's protocol.
    """
    excluded = [_canon(os.path.abspath(os.fspath(e))) for e in exclude]
    out: list[Source] = []
    seen: set[str] = set()
    for spec in specs:
        text = os.fspath(spec)
        if not os.path.exists(text) and _URI.match(text):
            if text[:7].lower() == "file://":
                found = _local_sources(_file_uri_path(text), excluded)
            else:
                found = _uri_sources(text)
        else:
            found = _local_sources(text, excluded)
        for key, src in found:
            if key in seen:
                raise ValueError(f"{src.uri}: listed twice")
            seen.add(key)
            out.append(src)
    return out


def input_files(path: str | os.PathLike[str]) -> list[Path]:
    """Return the local files that ``path`` (a file, directory or glob) names, in order."""
    text = os.fspath(path)
    if not os.path.exists(text) and _URI.match(text) and text[:7].lower() != "file://":
        raise ValueError(f"{text}: input_files reads local paths only (resolve_inputs reads URIs)")
    return [Path(s.uri) for s in resolve_inputs([text])]


def _canon(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _excluded(key: str, excluded: Sequence[str]) -> bool:
    return any(key == e or key.startswith(e.rstrip(os.sep) + os.sep) for e in excluded)


def _no_files(spec: str) -> FileNotFoundError:
    return FileNotFoundError(f"{spec}: no .parquet or .jsonl files")


def _file_uri_path(uri: str) -> str:
    host, slash, rest = uri[7:].partition("/")
    if host and host.lower() != "localhost":
        raise ValueError(f"{uri}: file:// URIs with a host name are not supported")
    return urllib.request.url2pathname(slash + rest)


def _local_sources(spec: str, excluded: Sequence[str]) -> list[tuple[str, Source]]:
    p = Path(spec)
    if p.exists() or not _MAGIC.search(spec):
        if p.is_dir():
            found = _walk(spec, None, excluded)
            if not found:
                raise _no_files(spec)
            return found
        if not p.exists():
            raise FileNotFoundError(f"{spec}: no such file or directory")
        fmt = file_format(spec)
        absolute = os.path.abspath(spec)
        key = os.path.join(_canon(os.path.dirname(absolute)), os.path.normcase(p.name))
        if _excluded(key, excluded):
            raise _no_files(spec)
        src = Source(spec, p.name, p.stat().st_size, fmt, None, absolute)
        return [(key, src)]
    parts = p.parts
    i = next(i for i, part in enumerate(parts) if _MAGIC.search(part))
    root = str(Path(*parts[:i])) if i else ""
    found = _walk(root, _Glob(parts[i:]), excluded)
    if not found:
        raise _no_files(spec)
    return found


def _walk(root: str, glob: _Glob | None, excluded: Sequence[str]) -> list[tuple[str, Source]]:
    """List the files below the local directory ``root`` (pruned by ``glob``), sorted by name."""
    top = root or os.curdir
    base = os.path.abspath(top)
    real = _canon(base)
    found: list[tuple[str, Source]] = []
    if _excluded(real, excluded):
        return found
    stack = [("", real, base)]
    visited = {real}  # symlinked directories: list each real directory once (no cycles)
    while stack:
        rel, real_dir, abs_dir = stack.pop()
        try:
            entries = os.scandir(abs_dir)
        except (FileNotFoundError, NotADirectoryError):
            if rel:
                raise
            break  # a glob's root that does not exist: no match
        with entries:
            for e in entries:
                if e.name.startswith((".", "_")):
                    continue
                r = f"{rel}/{e.name}" if rel else e.name
                if e.is_dir():
                    if glob is not None and not glob.may_match_below(r):
                        continue
                    sub = _canon(e.path) if e.is_symlink() else os.path.join(real_dir, e.name)
                    sub = os.path.normcase(sub)
                    if sub in visited or _excluded(sub, excluded):
                        continue
                    visited.add(sub)
                    stack.append((r, sub, e.path))
                elif e.is_file() and _known(e.name) and (glob is None or glob(r)):
                    key = os.path.join(real_dir, os.path.normcase(e.name))
                    if _excluded(key, excluded):
                        continue
                    uri = os.path.join(root, *r.split("/"))
                    src = Source(uri, r, e.stat().st_size, file_format(e.name), None, e.path)
                    found.append((key, src))
    found.sort(key=lambda kv: kv[1].name)
    return found


class _Glob:
    """A glob over root-relative ``/``-separated paths: ``*``, ``?``, ``[...]`` and ``**``."""

    def __init__(self, parts: Sequence[str]) -> None:
        parts = list(parts)
        while parts and not parts[-1]:  # "dir/*/" is "dir/*"
            parts.pop()
        # One regex per component (no component contains "/"); None is "**".
        self.parts = [None if p == "**" else re.compile(fnmatch.translate(p)) for p in parts]

    def __call__(self, rel: str) -> bool:
        """Whether ``rel`` matches, or is below a directory that matches."""
        comps = rel.split("/")
        return any(_match(self.parts, comps[:k]) for k in range(len(comps), 0, -1))

    def may_match_below(self, rel_dir: str) -> bool:
        """Whether a file below the directory ``rel_dir`` can match."""
        comps = rel_dir.split("/")
        for p, c in zip(self.parts, comps, strict=False):
            if p is None:
                return True
            if not p.match(c):
                return False
        return True


def _match(parts: Sequence[re.Pattern[str] | None], comps: Sequence[str]) -> bool:
    i = j = 0
    while i < len(parts):
        p = parts[i]
        if p is None:
            if i == len(parts) - 1:
                return j < len(comps)
            return any(_match(parts[i + 1 :], comps[k:]) for k in range(j, len(comps)))
        if j >= len(comps) or not p.match(comps[j]):
            return False
        i += 1
        j += 1
    return j == len(comps)


# The package that gives fsspec a protocol (pip name), and pyarrow's name for its own support.
_FSSPEC_PACKAGES = {
    "s3": "s3fs",
    "s3a": "s3fs",
    "gs": "gcsfs",
    "gcs": "gcsfs",
    "hf": "huggingface_hub",
    "az": "adlfs",
    "abfs": "adlfs",
    "abfss": "adlfs",
    "adl": "adlfs",
    "http": "aiohttp",
    "https": "aiohttp",
}
_ARROW_SUPPORT = {"s3": "S3", "gs": "GCS", "gcs": "GCS", "hdfs": "HDFS", "abfs": "Azure"}


def _needs(uri: str, protocol: str) -> str:
    package = _FSSPEC_PACKAGES.get(protocol)
    if package is None:
        msg = f"{uri}: reading {protocol}:// needs fsspec and a package that implements it"
    else:
        pip = "fsspec aiohttp requests" if package == "aiohttp" else package
        msg = f"{uri}: reading {protocol}:// needs fsspec with {package} (pip install {pip})"
    if protocol in _ARROW_SUPPORT:
        msg += f", or a pyarrow built with {_ARROW_SUPPORT[protocol]}"
    return msg


def _arrow_from_uri(uri: str) -> tuple[Any, str]:
    import pyarrow.fs as pafs

    fs, path = pafs.FileSystem.from_uri(uri)
    return fs, path


def _filesystem(uri: str, spec: str) -> tuple[Any, str]:
    """Return the filesystem handle (an _Fsspec or a pyarrow FileSystem) and the path in it."""
    protocols = [part.split("://", 1)[0].lower() for part in uri.split("::")]
    try:
        fsspec: Any = importlib.import_module("fsspec")
    except ImportError:
        fsspec = None
    if fsspec is not None:
        try:
            for protocol in protocols:
                fsspec.get_filesystem_class(protocol)
        except (ImportError, ValueError):
            pass  # not installed (s3fs, ...) or unknown to fsspec: try pyarrow
        else:
            fs, path = fsspec.core.url_to_fs(uri)
            return _Fsspec(fs), path
    try:
        return _arrow_from_uri(uri)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
        raise ValueError(_needs(spec, protocols[-1])) from e


def _info(fs: Any, path: str) -> tuple[str | None, int | None, str]:
    """Return ("file" | "directory" | None when missing, the size, the canonical path)."""
    if isinstance(fs, _Fsspec):
        try:
            info = fs.fs.info(path)
        except FileNotFoundError:
            return None, None, path
        size = info.get("size")
        kind = "file" if info.get("type") == "file" else "directory"
        return kind, None if size is None else int(size), str(info.get("name", path))
    import pyarrow.fs as pafs

    fi = fs.get_file_info(path)
    if fi.type == pafs.FileType.NotFound:
        return None, None, path
    if fi.type == pafs.FileType.Directory:
        return "directory", None, fi.path
    return "file", fi.size, fi.path


def _find(fs: Any, path: str) -> Iterator[tuple[str, int | None]]:
    """Yield every file below the directory ``path``, with its size."""
    if isinstance(fs, _Fsspec):
        for p, info in fs.fs.find(path, withdirs=False, detail=True).items():
            if info.get("type", "file") == "file":
                size = info.get("size")
                yield p, None if size is None else int(size)
        return
    import pyarrow.fs as pafs

    for fi in fs.get_file_info(pafs.FileSelector(path, recursive=True, allow_not_found=True)):
        if fi.type == pafs.FileType.File:
            yield fi.path, fi.size


def _key(fs: Any, path: str) -> str:
    if isinstance(fs, _Fsspec):
        return str(fs.fs.unstrip_protocol(path))
    return f"{fs.type_name}://{path}"


def _uri_sources(spec: str) -> list[tuple[str, Source]]:
    glob: _Glob | None = None
    root = spec
    head, sep, rest = spec.rpartition("://")
    if sep and head.lower() not in ("http", "https"):  # an http(s) URL is one file
        parts = rest.split("/")
        for i, part in enumerate(parts):
            if _MAGIC.search(part):
                root, glob = head + sep + "/".join(parts[:i]), _Glob(parts[i:])
                break
    fs, path = _filesystem(root, spec)
    kind, size, canon = _info(fs, path)
    if glob is None:
        if kind is None:
            raise FileNotFoundError(f"{spec}: no such file or directory")
        if kind == "file":
            name = spec.split("?", 1)[0] if head.lower() in ("http", "https") else spec
            name = name.rstrip("/").rsplit("/", 1)[-1]
            try:
                fmt = file_format(name)
            except ValueError:
                raise ValueError(
                    f"{spec}: unknown format (expected .parquet/.pq or "
                    ".jsonl/.ndjson/.json, optionally compressed)"
                ) from None
            return [(_key(fs, path), Source(spec, name, size, fmt, fs, path))]
    elif kind != "directory":
        raise _no_files(spec)
    prefix = canon if not canon or canon.endswith("/") else canon + "/"
    base = root if root.endswith("/") else root + "/"
    found: list[tuple[str, Source]] = []
    for p, n in _find(fs, canon):
        if not p.startswith(prefix):
            raise ValueError(f"{spec}: the listing returned {p}, which is not under {canon}")
        rel = p[len(prefix) :]
        comps = rel.split("/")
        if any(c.startswith((".", "_")) or not c for c in comps) or not _known(comps[-1]):
            continue
        if glob is not None and not glob(rel):
            continue
        found.append((_key(fs, p), Source(base + rel, rel, n, file_format(rel), fs, p)))
    if not found:
        raise _no_files(spec)
    found.sort(key=lambda kv: kv[1].name)
    return found


# ------------------------------------------------------------------------------------- reading


@contextlib.contextmanager
def _open_file(source: Source) -> Iterator[Any]:
    """Open ``source`` for random access (Parquet)."""
    fs = source._fs
    if fs is None:
        f = pa.OSFile(source._path, "rb")  # not a memory map: Windows can't delete a mapped file
    elif isinstance(fs, _Fsspec):
        f = fs.fs.open(source._path, "rb")
    else:
        f = fs.open_input_file(source._path)
    with f:
        yield f


@contextlib.contextmanager
def _open_stream(source: Source) -> Iterator[Any]:
    """Open ``source`` as a byte stream, decompressed after its suffix (JSON Lines)."""
    codec = _codec(source.name)
    fs = source._fs
    with contextlib.ExitStack() as stack:
        if fs is None:
            stream = stack.enter_context(pa.input_stream(source._path, compression=codec))
        elif isinstance(fs, _Fsspec):
            f = stack.enter_context(fs.fs.open(source._path, "rb"))
            stream = stack.enter_context(pa.PythonFile(f, mode="r"))  # does not close f
            if codec is not None:
                stream = stack.enter_context(pa.CompressedInputStream(stream, codec))
        else:
            stream = stack.enter_context(fs.open_input_stream(source._path, compression=codec))
        yield stream


@contextlib.contextmanager
def _errors(source: Source) -> Iterator[None]:
    """Name the file in pyarrow's parse errors."""
    try:
        yield
    except pa.ArrowInvalid as e:
        raise ValueError(f"{source.uri}: {e}") from e


def _check_columns(source: Source, names: Sequence[str], columns: Sequence[str]) -> None:
    for c in columns:
        if c not in names:
            raise ValueError(f"{source.uri}: no column {c!r} (columns: {', '.join(names)})")


def _json_chunks(stream: Any) -> Iterator[bytes]:
    """Yield the stream in pieces of about JSON_CHUNK_SIZE bytes that end at a line end."""
    parts: list[bytes | memoryview] = []
    while True:
        data = stream.read(JSON_CHUNK_SIZE)
        if not data:
            break
        cut = data.rfind(b"\n") + 1
        if not cut:  # inside a long line
            parts.append(data)
            continue
        parts.append(memoryview(data)[:cut])
        chunk = b"".join(parts)
        parts = [data[cut:]] if cut < len(data) else []
        if not chunk.isspace():
            yield chunk
    chunk = b"".join(parts)
    if chunk and not chunk.isspace():
        yield chunk


def _parse_json(chunk: bytes, schema: Any) -> Any:
    """Parse whole JSON lines, later chunks with the first chunk's ``schema``.

    As in pyarrow's streaming JSON reader, a field the schema lacks is an error.
    """
    import pyarrow.json as pj

    parse = (
        None
        if schema is None
        else pj.ParseOptions(explicit_schema=schema, unexpected_field_behavior="error")
    )
    try:
        return pj.read_json(
            pa.BufferReader(chunk),
            read_options=pj.ReadOptions(block_size=JSON_BLOCK_SIZE),
            parse_options=parse,
        )
    except pa.ArrowInvalid as e:
        if "straddl" not in str(e):
            raise
    # A line longer than a block: parse the chunk as one block.
    return pj.read_json(
        pa.BufferReader(chunk),
        read_options=pj.ReadOptions(block_size=len(chunk) + 1, use_threads=False),
        parse_options=parse,
    )


def _json_batches(
    stream: Any, source: Source, *, schema: Any = None, columns: Sequence[str] | None = None
) -> Iterator[Any]:
    for chunk in _json_chunks(stream):
        t = _parse_json(chunk, schema)
        if schema is None:
            schema = t.schema
            if columns is not None:
                _check_columns(source, schema.names, columns)
        if columns is not None:
            t = t.select(list(columns))
        yield from t.to_batches()


def _row_groups(pf: Any, columns: list[str] | None) -> Iterator[Any]:
    """The record batches of a Parquet file, one row group at a time.

    ``ParquetFile.iter_batches`` buffers ahead across row groups: on a 2 GB FineWeb file with a
    thousand row groups, a process held over 2 GiB. Reading each row group whole holds one.
    Decoding on one thread keeps the allocator's per-thread caches from growing the RSS to
    several times what Arrow allocates; the text column dominates, so threads would gain little.
    """
    for i in range(pf.num_row_groups):
        yield from pf.read_row_group(i, columns=columns, use_threads=False).to_batches()


def _rebatched(batches: Iterable[Any], batch_size: int) -> Generator[Any, None, int]:
    """Yield tables of exactly ``batch_size`` rows (the last may be shorter); return how many.

    Closing this generator closes ``batches`` too (and so the files a generator there holds).
    """
    pending: list[Any] = []
    rows = n = 0
    try:
        for b in batches:
            if not b.num_rows:
                continue
            pending.append(b)
            rows += b.num_rows
            while rows >= batch_size:
                t = pa.Table.from_batches(pending)
                yield t.slice(0, batch_size)
                n += 1
                rest = t.slice(batch_size)
                pending, rows = rest.to_batches(), rest.num_rows
    finally:
        close = getattr(batches, "close", None)
        if close is not None:
            close()
    if rows:
        yield pa.Table.from_batches(pending)
        n += 1
    return n


def iter_tables(
    source: Source, batch_size: int, *, columns: Sequence[str] | None = None
) -> Generator[pa.Table, None, None]:
    """Yield ``source`` as tables of exactly ``batch_size`` rows (the last may be shorter).

    A Parquet file without rows yields one empty table (its schema); an empty JSONL file yields
    nothing. Memory stays at about one row group or JSON chunk plus one batch. The file is
    closed when the iteration ends, fails or is closed early (Windows can't replace or delete
    a file that is still open).

    Args:
        source: A file from :func:`resolve_inputs`.
        batch_size: The rows per table.
        columns: Read only these columns (Parquet reads only them; JSONL parses every field).

    Raises:
        ValueError: A column in ``columns`` is missing, or the file does not parse.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, not {batch_size}")
    cols = None if columns is None else list(columns)
    if source.format == "parquet":
        import pyarrow.parquet as pq

        with _errors(source), _open_file(source) as f, pq.ParquetFile(f) as pf:
            schema = pf.schema_arrow
            if cols is not None:
                _check_columns(source, schema.names, cols)
                schema = pa.schema([schema.field(c) for c in cols], metadata=schema.metadata)
            if not (yield from _rebatched(_row_groups(pf, cols), batch_size)):
                yield schema.empty_table()
        return
    with _errors(source), _open_stream(source) as stream:
        yield from _rebatched(_json_batches(stream, source, columns=cols), batch_size)


def read_schema(source: Source) -> pa.Schema | None:
    """Return the file's Arrow schema: the Parquet footer, or the first JSON chunk's.

    None for a JSONL file without rows.
    """
    if source.format == "parquet":
        import pyarrow.parquet as pq

        with _errors(source), _open_file(source) as f, pq.ParquetFile(f) as pf:
            return pf.schema_arrow
    with _errors(source), _open_stream(source) as stream:
        for chunk in _json_chunks(stream):
            return _parse_json(chunk, None).schema
    return None


def read_tables(paths: Sequence[Path], batch_size: int) -> Generator[pa.Table, None, None]:
    """Yield tables of exactly ``batch_size`` rows (the last may be shorter) across ``paths``.

    JSONL files after the first are parsed with the first file's schema, and every batch is
    cast to it. No rows at all yields one empty table when the schema is known.
    """
    sources = [
        Source(str(p), Path(p).name, None, file_format(p), None, os.path.abspath(p)) for p in paths
    ]
    schema: Any = None

    def batches() -> Iterator[Any]:
        nonlocal schema
        for s in sources:
            with contextlib.ExitStack() as stack:
                stack.enter_context(_errors(s))
                if s.format == "parquet":
                    import pyarrow.parquet as pq

                    f = stack.enter_context(_open_file(s))
                    it = _row_groups(stack.enter_context(pq.ParquetFile(f)), None)
                else:
                    stream = stack.enter_context(_open_stream(s))
                    it = _json_batches(stream, s, schema=schema)
                for b in it:
                    if not b.num_rows:
                        continue
                    if schema is None:
                        schema = b.schema
                    elif b.schema != schema:
                        b = pa.Table.from_batches([b]).cast(schema).to_batches()[0]
                    yield b

    if not (yield from _rebatched(batches(), batch_size)):
        if schema is None and sources and sources[0].format == "parquet":
            schema = read_schema(sources[0])
        if schema is not None:
            yield schema.empty_table()


# ------------------------------------------------------------------------------------- outputs

_UNSAFE = re.compile(r'[<>:"\\|?*\x00-\x1f\x7f]')
_DEVICES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"{d}{i}" for d in ("COM", "LPT") for i in "0123456789\u00b9\u00b2\u00b3"}
)


def safe_relpath(name: str) -> str:
    r"""Return ``name`` (a relative ``/``-separated path) as a file name valid on Windows and POSIX.

    The result is the same on every platform: ``<>:"\\|?*`` and ASCII control characters become
    ``_``, trailing dots and spaces are dropped from each component, and Windows device names
    (``CON``, ``NUL``, ``COM1``, ``LPT9``, also as ``nul.txt``, any case) get a ``_`` prefix.

    Raises:
        ValueError: ``name`` is empty or absolute, or has an empty, ``.`` or ``..`` component.
    """
    if not name:
        raise ValueError("an empty path is not a relative file name")
    if name.startswith(("/", "\\")) or re.match(r"[A-Za-z]:[/\\]", name):
        raise ValueError(f"{name}: an absolute path is not a relative file name")
    out = []
    for part in name.split("/"):
        if not part:
            raise ValueError(f"{name}: an empty path component is not a file name")
        if part in (".", ".."):
            raise ValueError(f"{name}: a {part!r} path component is not a file name")
        part = _UNSAFE.sub("_", part).rstrip(". ") or "_"
        if part.split(".", 1)[0].rstrip(" ").upper() in _DEVICES:
            part = "_" + part
        out.append(part)
    return "/".join(out)


def _strip_suffix(name: str) -> str:
    low = name.lower()
    for ext in COMPRESSED:
        if low.endswith(ext):
            name, low = name[: -len(ext)], low[: -len(ext)]
            break
    for ext in (*PARQUET, *JSONL):
        if low.endswith(ext):
            return name[: -len(ext)]
    return name


def output_path(root: str | os.PathLike[str], source: Source, suffix: str | None = None) -> Path:
    """Return where the output for ``source`` goes under ``root``: ``root / safe_relpath(name)``.

    With ``suffix`` (``".parquet"``, ``".jsonl.gz"``), it replaces the source's format and
    compression suffix: ``a/x.jsonl.gz`` with ``".parquet"`` is ``a/x.parquet``.
    """
    name = source.name
    if suffix is not None:
        if not suffix.startswith("."):
            raise ValueError(f"{suffix!r}: a file suffix starts with '.'")
        name = _strip_suffix(name) + suffix
    return Path(root) / safe_relpath(name)


# Waits between tries while Windows reports a file as in use: 10 tries over about 2.6 s.
_BACKOFF = (0.02, 0.04, 0.08, 0.16, 0.32, 0.5, 0.5, 0.5, 0.5)
_sleep = time.sleep


def _retry(fn: Callable[..., None], *args: Any) -> None:
    for delay in (*_BACKOFF, None):
        try:
            fn(*args)
        except PermissionError:
            if delay is None:
                raise
            _sleep(delay)
        else:
            return


def replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
    """Rename ``src`` to ``dst`` (replacing it), retrying while Windows reports either in use."""
    _retry(os.replace, src, dst)


def remove(path: str | os.PathLike[str], *, missing_ok: bool = True) -> None:
    """Delete the file ``path``, retrying while Windows reports it in use."""
    try:
        _retry(os.unlink, path)
    except FileNotFoundError:
        if not missing_ok:
            raise


_O_BINARY = getattr(os, "O_BINARY", 0)  # Windows: no newline translation


def _sibling(path: Path, ext: str) -> tuple[Path, int]:
    """Create a hidden, uniquely named file next to ``path``; return it and its open descriptor."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_BINARY
    for _ in range(100):
        tmp = path.with_name(f".{path.name[:200]}.{secrets.token_hex(4)}{ext}")
        try:
            return tmp, os.open(tmp, flags, 0o666)  # the umask applies, as for any new file
        except FileExistsError:
            continue
    raise FileExistsError(f"{path}: no free name for a temporary file next to it")


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDWR | _O_BINARY)  # Windows flushes only a writable handle
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    """Write ``data`` to ``path`` atomically: a synced hidden sibling, then a rename."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp, fd = _sibling(p, ".tmp")
    try:
        with open(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            remove(tmp)
        raise


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    """Write ``text`` to ``path`` as UTF-8, atomically (see :func:`atomic_write_bytes`)."""
    atomic_write_bytes(path, text.encode("utf-8"))


class Writer:
    """Writes tables to a Parquet or JSONL file, atomically.

    The rows go to a hidden sibling, ``.NAME.<random>.partial``; :meth:`close` closes every
    handle on it, syncs it and renames it to ``path``, and :meth:`abort` deletes it. Nothing
    appears at ``path`` before :meth:`close` returns.
    """

    def __init__(self, path: str | os.PathLike[str], *, compression: str | None = None) -> None:
        self.path = Path(path)
        self.format = file_format(self.path)
        self.compression = compression
        self.tmp: Path | None = None
        self.rows = 0
        self._schema: Any = None
        self._sink: Any = None  # the file self.tmp
        self._pq: Any = None  # a ParquetWriter over _sink
        self._fh: Any = None  # the JSONL stream: _sink, or a compressed stream over it
        self._state = "open"

    def _open(self, schema: Any) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp, fd = _sibling(self.path, ".partial")
        os.close(fd)
        self.tmp = tmp
        self._schema = schema
        try:
            self._sink = pa.OSFile(str(tmp), "wb")
            if self.format == "parquet":
                import pyarrow.parquet as pq

                kw = {} if self.compression is None else {"compression": self.compression}
                self._pq = pq.ParquetWriter(self._sink, schema, **kw)
            else:
                codec = _codec(self.path)
                self._fh = (
                    self._sink if codec is None else pa.CompressedOutputStream(self._sink, codec)
                )
        except BaseException:
            self.abort()
            raise
        return tmp

    def write(self, table: pa.Table) -> None:
        """Append ``table`` (cast to the first table's schema)."""
        if self._state != "open":
            raise ValueError(f"{self.path}: the writer is {self._state}")
        if self._schema is None:
            self._open(table.schema)
        elif table.schema != self._schema:
            table = table.cast(self._schema)
        if not table.num_rows:
            return
        if self._pq is not None:
            self._pq.write_table(table)
        else:
            lines = [json.dumps(r, ensure_ascii=False, default=str) for r in table.to_pylist()]
            self._fh.write(("\n".join(lines) + "\n").encode("utf-8", "surrogatepass"))
        self.rows += table.num_rows

    def _close_handles(self) -> None:
        pqw, fh, sink = self._pq, self._fh, self._sink
        self._pq = self._fh = self._sink = None
        try:
            if pqw is not None:
                pqw.close()  # writes the footer; does not close the sink
            if fh is not None and fh is not sink:
                fh.close()
        finally:
            if sink is not None:
                sink.close()

    def close(self, schema: pa.Schema | None = None) -> None:
        """Finish the file and move it to ``path``.

        An empty output gets ``schema`` (Parquet; without one, a file with no columns) or zero
        lines (JSONL). On failure the partial file is deleted and the error re-raised.
        """
        if self._state == "closed":
            return
        if self._state == "aborted":
            raise ValueError(f"{self.path}: the writer was aborted")
        try:
            if self._schema is None:
                self._open(schema if schema is not None else pa.schema([]))
            self._close_handles()
            tmp = self.tmp
            assert tmp is not None
            _fsync(tmp)
            replace(tmp, self.path)
        except BaseException:
            self.abort()
            raise
        self.tmp = None
        self._state = "closed"

    def abort(self) -> None:
        """Drop the partial file (again, or after :meth:`close` failed: a no-op)."""
        if self._state == "closed":
            return
        self._state = "aborted"
        with contextlib.suppress(Exception):  # best effort: we are already failing
            self._close_handles()
        tmp, self.tmp = self.tmp, None
        if tmp is not None:
            with contextlib.suppress(OSError):
                remove(tmp)


def set_column(table: pa.Table, name: str, values: Any) -> pa.Table:
    """``table`` with column ``name`` replaced in place (or appended)."""
    i = table.schema.get_field_index(name)
    if i < 0:
        return table.append_column(name, values)
    return table.set_column(i, name, values)
