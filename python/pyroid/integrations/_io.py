"""Streaming Parquet / JSONL readers and writers over Arrow batches (for the ``pyroid`` CLI)."""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa

PARQUET = (".parquet", ".pq")
JSONL = (".jsonl", ".ndjson", ".json")
COMPRESSED = {".gz": "gzip", ".bz2": "bz2", ".zst": "zstd", ".lz4": "lz4"}


def _codec(path: Path) -> str | None:
    return COMPRESSED.get(path.suffix.lower())


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


def input_files(path: str | os.PathLike[str]) -> list[Path]:
    """``path`` itself, or every Parquet/JSONL file in the directory ``path``, sorted by name."""
    p = Path(path)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.is_file() and _known(f))
        if not files:
            raise FileNotFoundError(f"{p}: no .parquet or .jsonl files in this directory")
        kinds = {file_format(f) for f in files}
        if len(kinds) > 1:
            raise ValueError(f"{p}: mixes Parquet and JSONL files")
        return files
    if not p.exists():
        raise FileNotFoundError(f"{p}: no such file or directory")
    file_format(p)
    return [p]


def _known(f: Path) -> bool:
    try:
        file_format(f)
    except ValueError:
        return False
    return True


def _batches(path: Path, schema: pa.Schema | None, batch_size: int) -> Iterator[pa.RecordBatch]:
    if file_format(path) == "parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        yield from pf.iter_batches(batch_size=batch_size)
        return
    import pyarrow.json as pj

    parse = pj.ParseOptions(explicit_schema=schema) if schema is not None else None
    stream = pa.input_stream(str(path), compression=_codec(path))
    with pj.open_json(stream, parse_options=parse) as reader:
        yield from reader


def read_tables(paths: Sequence[Path], batch_size: int) -> Iterator[pa.Table]:
    """Tables of exactly ``batch_size`` rows (the last may be shorter), across all files in order.

    JSONL files after the first are parsed with the first file's schema.
    """
    schema: pa.Schema | None = None
    pending: list[pa.RecordBatch] = []
    rows = 0
    yielded = False
    for path in paths:
        for b in _batches(path, schema, batch_size):
            if schema is None:
                schema = b.schema
            elif b.schema != schema:
                b = pa.Table.from_batches([b]).cast(schema).to_batches()[0]
            pending.append(b)
            rows += b.num_rows
            while rows >= batch_size:
                t = pa.Table.from_batches(pending, schema)
                yield t.slice(0, batch_size)
                yielded = True
                rest = t.slice(batch_size)
                pending, rows = rest.to_batches(), rest.num_rows
    if rows:
        yield pa.Table.from_batches(pending, schema)
    elif not yielded:
        # No rows at all: one empty table, so that callers still see (and write) the schema.
        if schema is None and paths and file_format(paths[0]) == "parquet":
            import pyarrow.parquet as pq

            schema = pq.read_schema(paths[0])
        if schema is not None:
            yield schema.empty_table()


class Writer:
    """Writes tables to a Parquet or JSONL file, atomically (``PATH.partial``, renamed on close)."""

    def __init__(self, path: str | os.PathLike[str], *, compression: str | None = None) -> None:
        self.path = Path(path)
        self.format = file_format(self.path)
        self.compression = compression
        self.tmp = self.path.with_name(self.path.name + ".partial")
        self.rows = 0
        self._schema: pa.Schema | None = None
        self._pq: Any = None
        self._fh: Any = None

    def write(self, table: pa.Table) -> None:
        """Append ``table`` (cast to the first table's schema)."""
        if self._schema is None:
            self._schema = table.schema
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.format == "parquet":
                import pyarrow.parquet as pq

                kw = {} if self.compression is None else {"compression": self.compression}
                self._pq = pq.ParquetWriter(self.tmp, table.schema, **kw)
            else:
                self._fh = pa.output_stream(str(self.tmp), compression=_codec(self.path))
        elif table.schema != self._schema:
            table = table.cast(self._schema)
        if not table.num_rows:
            return
        self.rows += table.num_rows
        if self._pq is not None:
            self._pq.write_table(table)
        else:
            lines = [json.dumps(r, ensure_ascii=False, default=str) for r in table.to_pylist()]
            self._fh.write(("\n".join(lines) + "\n").encode("utf-8", "surrogatepass"))

    def close(self, schema: pa.Schema | None = None) -> None:
        """Finish the file; an empty output gets ``schema`` (Parquet) or zero lines (JSONL)."""
        if self._schema is None and schema is not None:
            self.write(schema.empty_table())
        if self._pq is not None:
            self._pq.close()
        elif self._fh is not None:
            self._fh.close()
        else:
            self.tmp.write_bytes(b"")
        os.replace(self.tmp, self.path)

    def abort(self) -> None:
        """Drop the partial file."""
        for h in (self._pq, self._fh):
            if h is not None:
                with contextlib.suppress(Exception):  # best effort: we are already failing
                    h.close()
        if self.tmp.exists():
            self.tmp.unlink()


def set_column(table: pa.Table, name: str, values: Any) -> pa.Table:
    """``table`` with column ``name`` replaced in place (or appended)."""
    i = table.schema.get_field_index(name)
    if i < 0:
        return table.append_column(name, values)
    return table.set_column(i, name, values)
