"""The ``pyroid`` command line (also ``python -m pyroid``).

Reads Parquet or JSON Lines in Arrow batches: files, directories (recursively), globs, and
``s3://``, ``gs://`` or ``hf://`` URIs when fsspec or pyarrow can open them. A file's format
follows its name: ``.parquet``/``.pq`` is Parquet; ``.jsonl``/``.ndjson``/``.json`` is JSON
Lines, optionally compressed as ``.gz``, ``.bz2``, ``.zst`` or ``.lz4``.

    pyroid clean IN... OUT --recipe fineweb [--shard I/N] [--workers N] [--excluded DROPPED]
    pyroid clean IN.warc.gz... OUT --recipe fineweb-warc [--dump NAME] [--limit N]
    pyroid langid IN... OUT [--model lid.176.bin] [--keep en,fr --min-score 0.65] [--top-k K]
    pyroid dedup IN... OUT [--method minhash|exact] [--normalize] [--excluded DROPPED]
    pyroid dedup signatures IN... SIGS  ->  buckets SIGS DUPS  ->  cluster DUPS REMOVED
                                        ->  filter IN... OUT --removed REMOVED
    pyroid translit IN... OUT [--mode anyascii|unidecode] [--output-column COL]
    pyroid stats OUT
    pyroid run [-p ftfy,fasttext] (SCRIPT | -m MODULE) [ARGS...]

OUT is a file when its name has a data suffix, else a directory that gets one output file per
input file (``--merge``: one per shard). Outputs are written under temporary names and renamed
when complete; finished files are recorded in ``OUT/.pyroid``, so rerunning an interrupted
command redoes only what did not finish. ``--shard I/N`` takes every N-th input file from the
I-th (datatrove's split); ``--workers N`` runs N processes on one machine.

``clean --recipe fineweb`` runs the FineWeb chain the benchmarks use
(``pyroid.integrations.recipes.FineWeb``): ftfy -> fastText LID (en > 0.65) -> the Gopher
repetition and quality, C4 and FineWeb filters (datatrove's decisions) -> MinHash dedup over
the survivors. Kept rows get the cleaned text and datatrove's ``language`` and
``language_score`` columns, replacing any columns of those names. ``--excluded`` gets the
dropped rows, with a ``filter_reason`` of ``lid``, ``<stage>:<datatrove reason>``, ``null`` or
``minhash``. Needs pyarrow and numpy: ``pip install 'pyroid[cli]'``.

``clean --recipe fineweb-warc`` runs datatrove's ``examples/fineweb.py`` on WARC files
(``pyroid.integrations.recipes.FineWebWarc``): the pages as datatrove's ``WarcReader`` reads them
(``pyroid.warc``) -> URL filter -> Trafilatura (trafilatura 1.11.0, as datatrove installs it) ->
fastText LID -> the Gopher, C4 and FineWeb filters -> MinHash dedup over every file in order ->
PII replacement, one rotation per file. IN are WARC, ARC or WET files (``.gz`` or plain); the
output has ``text``, ``id``, ``dump`` (with ``--dump``), ``url``, ``date``, ``file_path``,
``language`` and ``language_score``, one row per kept document. Also needs datatrove (its URL
block lists), tldextract and fasteners: ``pip install 'pyroid[cli,warc,datatrove]' tldextract
fasteners``.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import itertools
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pyroid

RECIPES = ("fineweb", "fineweb-warc")
WARC_RECIPE = "fineweb-warc"
# What --recipe fineweb-warc finds in a directory or glob: WARC, ARC and WET files, plain or gzip.
_WARC_SUFFIXES = (".warc", ".arc", ".wet")
# fineweb-warc gathers its kept and dropped rows into tables of about this size before writing.
_WARC_FLUSH_ROWS = 10_000
_WARC_FLUSH_BYTES = 64 << 20
# The suffixes pyroid.integrations._io reads and writes (only JSON Lines can be compressed).
_FORMATS = ".parquet/.pq or .jsonl/.ndjson/.json[.gz|.bz2|.zst|.lz4]"
_OUT_FORMATS = ("parquet", "jsonl", "jsonl.gz", "jsonl.bz2", "jsonl.zst", "jsonl.lz4")
_STAGES = ("signatures", "buckets", "cluster", "filter")


def _need_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise SystemExit(
            "pyroid: this command needs pyarrow (and numpy): pip install 'pyroid[cli]'"
        ) from None


# ========================================================================================= layout


def _data_suffix(name: str) -> str | None:
    """The format suffix of ``name`` as written (``.jsonl.gz``, ``.parquet``), or None."""
    from pyroid.integrations import _io

    low = name.lower()
    end = len(name)
    for ext in _io.COMPRESSED:
        if low.endswith(ext):
            end -= len(ext)
            low = low[: -len(ext)]
            break
    for ext in (*_io.PARQUET, *_io.JSONL):
        if low.endswith(ext):
            if end != len(name) and ext in _io.PARQUET:
                return None  # compressed Parquet is not a format
            return name[end - len(ext) :]
    return None


def _is_file_out(out: str) -> bool:
    """OUT names a file (it has a data suffix and is not a directory)."""
    p = Path(out)
    if out.endswith(("/", os.sep)) or p.is_dir():
        return False
    if _data_suffix(p.name) is not None:
        return True
    if p.suffix:
        from pyroid.integrations import _io

        try:
            _io.file_format(p)  # raises "unknown format" for e.g. out.csv
        except ValueError as e:
            raise ValueError(f"{e} (end OUT with / to make it a directory)") from None
    return False


def _tagged(path: Path, tag: str) -> Path:
    """``kept.parquet`` -> ``kept-00003-of-00008.parquet``."""
    suffix = _data_suffix(path.name) or ""
    stem = path.name[: len(path.name) - len(suffix)]
    return path.with_name(f"{stem}-{tag}{suffix}")


@dataclass(frozen=True)
class Layout:
    """Where one run writes: per-file outputs under a directory, or one merged file."""

    merge: bool
    out_dir: Path | None  # directory OUT (per-file outputs, or the part-*.EXT files with --merge)
    final: Path | None  # the merged output of this shard
    excluded_dir: Path | None
    excluded_final: Path | None
    suffix: str | None  # per-file output suffix (None: keep each input's)
    state_root: Path
    work: Path
    shard_tag: str
    # Merged Parquet from several files: parts wait as Arrow IPC, so Parquet is encoded once.
    ipc_parts: bool = False

    def outputs(self, src: Any) -> tuple[Path, Path | None]:
        """Per-file mode: the output and excluded paths of input ``src``."""
        from pyroid.integrations import _io

        assert self.out_dir is not None
        if src.format == "warc":  # a/x.warc.gz -> a/x.parquet
            assert self.suffix is not None
            name = _io.safe_relpath(_warc_stem(src.name) + self.suffix)
            out = self.out_dir / name
            return out, self.excluded_dir / name if self.excluded_dir is not None else None
        out = _io.output_path(self.out_dir, src, self.suffix)
        excl = (
            _io.output_path(self.excluded_dir, src, self.suffix)
            if self.excluded_dir is not None
            else None
        )
        return out, excl

    def part(self, index: int, excluded: bool = False) -> Path:
        """Merged mode: the part file of input ``index`` (the final format, or Arrow IPC)."""
        final = self.excluded_final if excluded else self.final
        assert final is not None
        sfx = _data_suffix(final.name) or ".parquet"
        if self.ipc_parts and sfx.lower().endswith((".parquet", ".pq")):
            sfx = ".arrow"
        return self.work / ("excluded" if excluded else "parts") / f"{index:06d}{sfx}"


def _layout(a: argparse.Namespace, sources: Sequence[Any], shard: Any, *, excluded: bool) -> Layout:
    out = a.output
    file_out = _is_file_out(out)
    excl = getattr(a, "excluded", None) if excluded else None
    fmt = getattr(a, "format", None)
    if file_out:
        if fmt:
            raise SystemExit("pyroid: --format applies to a directory OUT; name the file instead")
        final: Path | None = _tagged(Path(out), shard.tag) if shard.count > 1 else Path(out)
        assert final is not None
        excl_final: Path | None = None
        if excl:
            if not _is_file_out(excl):
                raise SystemExit(f"pyroid: --excluded {excl}: OUT is a file, so this must be too")
            excl_final = _tagged(Path(excl), shard.tag) if shard.count > 1 else Path(excl)
        state_root = final.parent / f".{final.name}.pyroid"
        return Layout(
            merge=True,
            out_dir=None,
            final=final,
            excluded_dir=None,
            excluded_final=excl_final,
            suffix=None,
            state_root=state_root,
            work=_work(a, state_root, shard),
            shard_tag=shard.tag,
        )
    out_dir = Path(out)
    if excl and _is_file_out(excl):
        raise SystemExit(f"pyroid: --excluded {excl}: OUT is a directory, so this must be too")
    if excl and Path(excl).resolve().is_relative_to(out_dir.resolve()):
        raise SystemExit("pyroid: --excluded must not be inside OUT (OUT is read by later stages)")
    suffix = f".{fmt}" if fmt else None
    final = excl_final = None
    if getattr(a, "merge", False):
        sfx = suffix or (_data_suffix(sources[0].name) if sources else None) or ".parquet"
        final = out_dir / f"part-{shard.tag}{sfx}"
        excl_final = Path(excl) / f"part-{shard.tag}{sfx}" if excl else None
    state_root = out_dir / ".pyroid"
    return Layout(
        merge=final is not None,
        out_dir=out_dir,
        final=final,
        excluded_dir=Path(excl) if excl and final is None else None,
        excluded_final=excl_final,
        suffix=suffix,
        state_root=state_root,
        work=_work(a, state_root, shard),
        shard_tag=shard.tag,
    )


def _work(a: argparse.Namespace, state_root: Path, shard: Any) -> Path:
    tmp = getattr(a, "tmp_dir", None)
    if tmp:
        import hashlib

        h = hashlib.sha256(str(state_root.resolve()).encode()).hexdigest()[:12]
        return Path(tmp) / f".pyroid-work-{h}-{shard.tag}"
    return state_root / f"work-{shard.tag}"


def _private_work(path: Path) -> None:
    """Create the work directory ``path`` under ``--tmp-dir``, or check the one a run left.

    Its name is predictable and ``--tmp-dir`` may be shared, so a directory or link that someone
    else put there could make pyroid write through planted links. The directory is made with
    mode 0700; a reused one must be a real directory (not a link or junction) and, on POSIX,
    belong to the effective user and not be writable by others. On Windows the directory
    inherits the ACL of ``--tmp-dir`` (``%TEMP%`` is per-user).
    """
    import stat

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(path, 0o700)
        return
    except FileExistsError:
        pass
    hint = "remove it, or pass another --tmp-dir"
    st = os.lstat(path)
    # Windows: a reparse tag with the name-surrogate bit (a symlink, a junction) points elsewhere.
    if not stat.S_ISDIR(st.st_mode) or getattr(st, "st_reparse_tag", 0) & 0x20000000:
        raise SystemExit(f"pyroid: {path} is not a directory pyroid made ({hint})")
    if hasattr(os, "geteuid"):
        if st.st_uid != os.geteuid():
            raise SystemExit(f"pyroid: {path} belongs to another user ({hint})")
        if st.st_mode & 0o022:
            raise SystemExit(f"pyroid: {path} is writable by other users ({hint})")
        if st.st_mode & 0o077:
            os.chmod(path, 0o700)


def _check_out(layout: Layout, overwrite: bool, sources: Sequence[Any]) -> None:
    """Refuse to replace files a previous, unrelated command wrote."""
    if layout.merge and layout.final is not None and layout.out_dir is None:
        if layout.final.exists() and not overwrite and not layout.state_root.exists():
            raise SystemExit(f"pyroid: {layout.final} exists (pass --overwrite to replace it)")
        excl = layout.excluded_final
        if excl is not None and excl.exists() and not overwrite and not layout.state_root.exists():
            raise SystemExit(f"pyroid: {excl} exists (pass --overwrite to replace it)")
    if layout.out_dir is not None and layout.out_dir.is_dir() and not overwrite:
        # Another shard may be creating .pyroid right now: a directory with it is ours.
        ours = layout.state_root.exists()
        if not ours and any(p.name != ".pyroid" for p in layout.out_dir.iterdir()):
            raise SystemExit(
                f"pyroid: {layout.out_dir} is not empty and was not written by pyroid "
                "(pass --overwrite to write into it anyway)"
            )
    outs = [p for p in (layout.final, layout.excluded_final) if p is not None]
    for s in sources:
        if "://" not in s.uri and any(Path(s.uri).resolve() == p.resolve() for p in outs):
            raise SystemExit(f"pyroid: {s.uri} is both an input and an output")


# ======================================================================================== options


@dataclass(frozen=True)
class Options:
    """What a unit of work needs (picklable: sent to worker processes)."""

    command: str
    text_column: str
    batch_size: int
    threads: int | None
    compression: str | None
    excluded: bool
    extra: dict[str, Any] = field(default_factory=dict)


def _options(a: argparse.Namespace, command: str, **extra: Any) -> Options:
    return Options(
        command=command,
        text_column=a.text_column,
        batch_size=a.batch_size or _default_batch(a),
        threads=a.threads,
        compression=getattr(a, "compression", None),
        excluded=bool(getattr(a, "excluded", None)),
        extra=extra,
    )


def _default_batch(a: argparse.Namespace) -> int:
    """--batch-size when not given: 10000 rows, or 256 pages for --recipe fineweb-warc."""
    return 256 if getattr(a, "recipe", None) == WARC_RECIPE else 10_000


def _fingerprint_options(o: Options) -> dict[str, Any]:
    """The options that decide the output (not threads, batch size, sharding or scheduling)."""
    return {
        "text_column": o.text_column,
        "compression": o.compression,
        "excluded": o.excluded,
        **o.extra,
    }


def _text(table: Any, column: str) -> Any:
    import pyarrow as pa

    if column not in table.column_names:
        raise ValueError(f"no column {column!r} (columns: {', '.join(table.column_names)})")
    col = table.column(column)
    # An all-null or empty JSON file, or an empty table, infers the column as type null.
    return col.cast(pa.string()) if pa.types.is_null(col.type) else col


def _check_text_column(sources: Sequence[Any], column: str) -> None:
    """Fail early (exit 1, with the message) when the first input has no text column."""
    from pyroid.integrations import _io

    if not sources:
        return
    schema = _io.read_schema(sources[0])
    if schema is not None and column not in schema.names:
        raise SystemExit(
            f"pyroid: {sources[0].uri}: no column {column!r} (columns: {', '.join(schema.names)})"
        )


# ================================================================================ per-file steps
# A step turns one input table into (kept, dropped-with-filter_reason or None, removed counts).
# Steps are cached per process: workers load the LID model once.

_CACHE: dict[Any, Any] = {}


def _recipe(o: Options) -> Any:
    from pyroid.integrations import recipes

    key = (o.command, json.dumps(o.extra, sort_keys=True, default=str), o.threads)
    r = _CACHE.get(key)
    if r is None:
        x = o.extra
        if o.command == "langid":
            keep_langs = x["keep"]
            r = recipes.FineWeb(
                lid_model=x["model"],
                languages=keep_langs,
                language_threshold=x["min_score"] if keep_langs else -math.inf,
                ftfy_config=False,
                filters=[],
                minhash=None,
                threads=o.threads,
            )
        else:
            r = recipes.FineWeb(lid_model=x["lid_model"], threads=o.threads, minhash=None)
        _CACHE[key] = r
    return r


def _counts(reasons: Any) -> Counter[str]:
    import pyarrow.compute as pc

    out: Counter[str] = Counter()
    if len(reasons):
        for row in pc.value_counts(reasons).to_pylist():
            out[str(row["values"])] += int(row["counts"])
    return out


def _step_clean(t: Any, o: Options) -> tuple[Any, Any, Counter[str]]:
    import pyarrow.compute as pc

    from pyroid.integrations import _io

    ann = _recipe(o).annotate(_text(t, o.text_column))
    t = _io.set_column(t, o.text_column, ann.text)
    t = _io.set_column(t, "language", ann.language)
    t = _io.set_column(t, "language_score", ann.language_score)
    drop = pc.invert(ann.keep)
    reasons = ann.reason.filter(drop)
    dropped = t.filter(drop).append_column("filter_reason", reasons) if o.excluded else None
    return t.filter(ann.keep), dropped, _counts(reasons)


def _step_langid(t: Any, o: Options) -> tuple[Any, Any, Counter[str]]:
    import pyarrow as pa
    import pyarrow.compute as pc

    from pyroid.integrations import _io

    recipe = _recipe(o)
    text = _text(t, o.text_column)
    ann = recipe.annotate(text)
    t = _io.set_column(t, "language", ann.language)
    t = _io.set_column(t, "language_score", ann.language_score)
    top_k = o.extra["top_k"]
    if top_k > 1:
        labels, probs = recipe.model.predict(
            pc.replace_substring(text, "\n", " "), k=top_k, threads=o.threads
        )
        t = _io.set_column(
            t,
            "top_languages",
            pa.array(
                [
                    None if lb is None else [x.removeprefix("__label__") for x in lb]
                    for lb in labels
                ],
                pa.list_(pa.string()),
            ),
        )
        t = _io.set_column(
            t,
            "top_language_scores",
            pa.array(
                [None if p is None else [float(x) for x in p] for p in probs],
                pa.list_(pa.float64()),
            ),
        )
    drop = pc.invert(ann.keep)
    reasons = ann.reason.filter(drop)
    dropped = t.filter(drop).append_column("filter_reason", reasons) if o.excluded else None
    return t.filter(ann.keep), dropped, _counts(reasons)


def _step_translit(t: Any, o: Options) -> tuple[Any, Any, Counter[str]]:
    import pyroid.translit
    from pyroid.integrations import _io

    x = o.extra
    text = _text(t, o.text_column)
    if x["mode"] == "anyascii":
        col = pyroid.translit.anyascii(text, threads=o.threads)
    else:
        col = pyroid.translit.unidecode(text, x["errors"], x["replace_str"], threads=o.threads)
    return _io.set_column(t, x["output_column"] or o.text_column, col), None, Counter()


_STEPS: dict[str, Callable[[Any, Options], tuple[Any, Any, Counter[str]]]] = {
    "clean": _step_clean,
    "langid": _step_langid,
    "translit": _step_translit,
}


# ========================================================================================= units


@dataclass(frozen=True)
class FileJob:
    """One input file's work, and where its results go."""

    source: Any
    rank: int
    key: str
    options: Options
    state: Any
    out: str | None = None
    excluded: str | None = None
    # clean with MinHash, phase 1: the kept and dropped rows wait in Arrow IPC spools and the
    # kept rows' signatures go to `sigs`.
    spool: str | None = None
    excluded_spool: str | None = None
    sigs: str | None = None
    minhash: dict[str, Any] | None = None
    sort_buffer: int = 64 << 20
    # stage 4: read the rows from `spool` (else from `source`) and drop those in `removed`
    removed: str | None = None
    cluster_ids: bool = False
    cluster_sizes: bool = False


def _job_outputs(job: FileJob) -> list[str]:
    """Every file a job writes: recorded when it is done, and checked when a run resumes."""
    if job.removed is not None:  # stage 4 reads the spools; it writes only the outputs
        return [p for p in (job.out, job.excluded) if p]
    out = [p for p in (job.out, job.excluded, job.spool, job.excluded_spool) if p]
    if job.sigs is not None:
        assert job.minhash is not None
        out += [
            os.path.join(job.sigs, f"bucket_{b:03d}", f"{job.rank:05d}.minhash.sig")
            for b in range(job.minhash["num_buckets"])
        ]
    return out


class _Ipc:
    """An Arrow IPC file (LZ4) that rows wait in.

    It is read back with plain reads, never mapped: Windows can't delete a mapped file.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._sink: Any = None
        self._writer: Any = None

    def write(self, table: Any) -> None:
        import pyarrow as pa

        from pyroid.integrations import _io

        if self._writer is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            # Written through the descriptor of a file made here, never through a planted link.
            fh = open(_io.create_new(self.path + ".partial"), "wb", buffering=1 << 20)  # noqa: SIM115
            self._sink = pa.PythonFile(fh, mode="w")
            opts = pa.ipc.IpcWriteOptions(compression="lz4")
            self._writer = pa.ipc.new_file(self._sink, table.schema, options=opts)
        self._writer.write_table(table)

    def close(self, schema: Any = None) -> None:
        """Finish the file (an empty one with ``schema`` if nothing was written)."""
        from pyroid.integrations import _io

        if self._writer is None:
            if schema is None:  # no rows and no schema (an empty JSONL input): an empty file
                _io.atomic_write_bytes(self.path, b"")
                return
            self.write(schema.empty_table())
        writer, sink, self._writer, self._sink = self._writer, self._sink, None, None
        try:
            writer.close()
        finally:
            sink.close()
        _io.replace(self.path + ".partial", self.path)

    def abort(self) -> None:
        from pyroid.integrations import _io

        writer, sink, self._writer, self._sink = self._writer, self._sink, None, None
        with contextlib.suppress(Exception):
            if writer is not None:
                writer.close()
        with contextlib.suppress(Exception):
            if sink is not None:
                sink.close()
        _io.remove(self.path + ".partial")


def _read_ipc(path: str) -> Iterator[Any]:
    """The tables of an IPC spool (nothing if it does not exist); the file is closed after."""
    import pyarrow as pa

    if not os.path.exists(path) or not os.path.getsize(path):
        return
    with pa.OSFile(path, "rb") as f:
        reader = pa.ipc.open_file(f)
        for i in range(reader.num_record_batches):
            yield pa.Table.from_batches([reader.get_batch(i)])


def _writer(path: str | None, compression: str | None) -> Any:
    from pyroid.integrations import _io

    if path is None:
        return None
    if path.endswith(".arrow"):
        return _Ipc(path)  # a part of a merged Parquet output
    return _io.Writer(path, compression=compression)


def _finish(writers: Sequence[tuple[Any, Any]]) -> None:
    for w, schema in writers:
        if w is not None:
            w.close(schema)


def _abort(*objs: Any) -> None:
    for w in objs:
        if w is not None:
            with contextlib.suppress(Exception):
                w.abort()


def _unit_rows(job: FileJob, report: Callable[[int], None]) -> dict[str, Any]:
    """Clean (without MinHash, or its phase 1), langid, translit on one file."""
    from pyroid.integrations import _io

    o = job.options
    step = _STEPS[o.command]
    stats: Counter[str] = Counter()
    n_in = n_out = 0
    t0 = time.perf_counter()
    if job.spool is not None:
        out: Any = _Ipc(job.spool)
        excl: Any = _Ipc(job.excluded_spool) if job.excluded_spool else None
    else:
        out, excl = _writer(job.out, o.compression), _writer(job.excluded, o.compression)
    sig = None
    if job.sigs is not None:
        from pyroid._pyroid import dedup as _native
        from pyroid.integrations import _minhash

        assert job.minhash is not None
        sig = _native._SigFileWriter(
            job.sigs,
            job.rank,
            _minhash.native_config(job.minhash),
            memory_budget=job.sort_buffer,
            threads=o.threads,
        )
    schema = excl_schema = None
    try:
        for t in _io.iter_tables(job.source, o.batch_size):
            kept, dropped, removed = step(t, o)
            n_in += t.num_rows
            n_out += kept.num_rows
            stats.update(removed)
            schema = kept.schema
            out.write(kept)
            if excl is not None and dropped is not None:
                excl_schema = dropped.schema
                excl.write(dropped)
            if sig is not None:
                sig.add(kept.column(o.text_column), threads=o.threads)
            report(t.num_rows)
        if sig is not None:
            sig.finish()
        if excl is not None and excl_schema is None and schema is not None:
            import pyarrow as pa

            excl_schema = schema.append(pa.field("filter_reason", pa.string()))
        out.close(schema)
        if excl is not None:
            excl.close(excl_schema)
    except BaseException:
        _abort(out, excl, sig)
        raise
    result = {
        "input": job.source.uri,
        "size": job.source.size,
        "rank": job.rank,
        "outputs": _job_outputs(job),
        "wall_s": round(time.perf_counter() - t0, 3),
        "stats": {"docs_in": n_in, "docs_out": n_out, "removed": dict(stats)},
    }
    job.state.mark(job.key, result)
    return result


def _unit_filter(job: FileJob, report: Callable[[int], None]) -> dict[str, Any]:
    """MinHash stage 4 on one file: drop the rows in ``{rank:06d}.remove``."""
    import pyarrow as pa
    import pyarrow.compute as pc

    from pyroid.integrations import _io, _minhash

    o = job.options
    t0 = time.perf_counter()
    assert job.removed is not None
    removal = _minhash.Removal(
        job.removed, job.rank, cluster_ids=job.cluster_ids, cluster_sizes=job.cluster_sizes
    )
    # fineweb-warc: PIIFormatter after the MinHash filter, a fresh rotation per file (task)
    pii = _warc_pii() if o.extra.get("recipe") == WARC_RECIPE else None
    out, excl = _writer(job.out, o.compression), _writer(job.excluded, o.compression)
    if job.spool is not None and not os.path.exists(job.spool):
        raise FileNotFoundError(
            f"{job.spool}: the survivors of this file are gone; rerun with --overwrite"
        )
    tables = (
        _read_ipc(job.spool) if job.spool is not None else _io.iter_tables(job.source, o.batch_size)
    )
    n_in = n_out = 0
    schema = excl_schema = None
    try:
        if excl is not None and job.excluded_spool is not None:
            for t in _read_ipc(job.excluded_spool):  # the rows the filters dropped, first
                excl_schema = t.schema
                excl.write(t)
        for t, keep, _ in _minhash.iter_kept(tables, removal):
            n = t.num_rows
            kept = t.filter(keep)
            if pii is not None and kept.num_rows:
                kept = _pii_text(kept, pii, o.threads)
            n_in += n
            n_out += kept.num_rows
            schema = kept.schema
            out.write(kept)
            if excl is not None:
                dups = t.filter(pc.invert(keep))
                dups = dups.append_column(
                    "filter_reason", pa.array(["minhash"] * dups.num_rows, pa.string())
                )
                if excl_schema is None:
                    excl_schema = dups.schema
                excl.write(dups)
            report(n)
        if schema is None and job.spool is not None and os.path.getsize(job.spool):
            schema = _ipc_schema(Path(job.spool))  # no survivor: still the spool's columns
        out.close(schema)
        if excl is not None:
            if excl_schema is None and schema is not None:
                excl_schema = schema.append(pa.field("filter_reason", pa.string()))
            excl.close(excl_schema)
    except BaseException:
        _abort(out, excl)
        raise
    stats: dict[str, Any] = {"docs_in": n_in, "docs_out": n_out, "removed": {}}
    if n_in - n_out:
        stats["removed"] = {"minhash": n_in - n_out}
    if job.spool is not None:
        # clean: this file's counts are its filter counts plus the MinHash removals
        first = job.state.record(f"p1-{job.rank:06d}") or {}
        s1 = first.get("stats", {})
        stats = {
            "docs_in": s1.get("docs_in", n_in),
            "docs_out": n_out,
            "removed": {**s1.get("removed", {}), **stats["removed"]},
        }
    result = {
        "input": job.source.uri,
        "size": job.source.size,
        "rank": job.rank,
        "outputs": _job_outputs(job),
        "wall_s": round(time.perf_counter() - t0, 3),
        "stats": stats,
    }
    job.state.mark(job.key, result)
    return result


def _unit_signatures(job: FileJob, report: Callable[[int], None]) -> dict[str, Any]:
    """MinHash stage 1 on one file."""
    from pyroid._pyroid import dedup as _native
    from pyroid.integrations import _io, _minhash

    o = job.options
    assert job.sigs is not None
    assert job.minhash is not None
    t0 = time.perf_counter()
    w = _native._SigFileWriter(
        job.sigs,
        job.rank,
        _minhash.native_config(job.minhash),
        memory_budget=job.sort_buffer,
        threads=o.threads,
    )
    n = 0
    try:
        for t in _io.iter_tables(job.source, o.batch_size, columns=[o.text_column]):
            w.add(_text(t, o.text_column), threads=o.threads)
            n += t.num_rows
            report(t.num_rows)
        w.finish()
    except BaseException:
        w.abort()
        raise
    result = {
        "input": job.source.uri,
        "size": job.source.size,
        "rank": job.rank,
        "outputs": _job_outputs(job),
        "wall_s": round(time.perf_counter() - t0, 3),
        "stats": {"docs_in": n, "docs_out": n, "removed": {}},
    }
    job.state.mark(job.key, result)
    return result


# ================================================================================ fineweb-warc


def _is_warc(name: str) -> bool:
    return name.lower().removesuffix(".gz").endswith(_WARC_SUFFIXES)


def _warc_stem(name: str) -> str:
    """``a/x.warc.gz`` -> ``a/x`` (output names keep the rest of the input's name)."""
    low = name.lower()
    if low.endswith(".gz"):
        name, low = name[:-3], low[:-3]
    for ext in _WARC_SUFFIXES:
        if low.endswith(ext):
            return name[: -len(ext)]
    return name


def _no_warc(spec: str) -> FileNotFoundError:
    return FileNotFoundError(f"{spec}: no WARC files (.warc, .arc or .wet, optionally .gz)")


def _warc_sources(specs: Sequence[str], exclude: Sequence[Path]) -> list[Any]:
    """The WARC files ``specs`` name, found as ``_io.resolve_inputs`` finds Parquet and JSONL.

    Files, directories (recursively), globs and URIs; each spec's files sorted by name. A file
    named directly is taken whatever its suffix.
    """
    from pyroid.integrations import _io

    excluded = [_io._canon(os.path.abspath(os.fspath(e))) for e in exclude]
    out: list[Any] = []
    seen: set[str] = set()
    for spec in specs:
        if not os.path.exists(spec) and _io._URI.match(spec) and spec[:7].lower() != "file://":
            found = _warc_uri_sources(spec)
        else:
            path = _io._file_uri_path(spec) if spec[:7].lower() == "file://" else spec
            found = _warc_local_sources(path, excluded)
        for key, src in found:
            if key in seen:
                raise ValueError(f"{src.uri}: listed twice")
            seen.add(key)
            out.append(src)
    return out


def _warc_local_sources(spec: str, excluded: Sequence[str]) -> list[tuple[str, Any]]:
    from pyroid.integrations import _io

    p = Path(spec)
    if p.exists() or not _io._MAGIC.search(spec):
        if not p.exists():
            raise FileNotFoundError(f"{spec}: no such file or directory")
        if not p.is_dir():
            absolute = os.path.abspath(spec)
            key = os.path.join(_io._canon(os.path.dirname(absolute)), os.path.normcase(p.name))
            if _io._excluded(key, excluded):
                raise _no_warc(spec)
            return [(key, _io.Source(spec, p.name, p.stat().st_size, "warc", None, absolute))]
        found = _warc_walk(spec, None, excluded)
    else:
        parts = p.parts
        i = next(i for i, part in enumerate(parts) if _io._MAGIC.search(part))
        found = _warc_walk(str(Path(*parts[:i])) if i else "", _io._Glob(parts[i:]), excluded)
    if not found:
        raise _no_warc(spec)
    return found


def _warc_walk(root: str, glob: Any, excluded: Sequence[str]) -> list[tuple[str, Any]]:
    """The WARC files below the local directory ``root`` (``_io._walk``'s rules), by name."""
    from pyroid.integrations import _io

    base = os.path.abspath(root or os.curdir)
    real = _io._canon(base)
    found: list[tuple[str, Any]] = []
    if _io._excluded(real, excluded):
        return found
    stack = [("", real, base)]
    visited = {real}
    while stack:
        rel, real_dir, abs_dir = stack.pop()
        try:
            entries = os.scandir(abs_dir)
        except (FileNotFoundError, NotADirectoryError):
            if rel:
                raise
            break
        with entries:
            for e in entries:
                if e.name.startswith((".", "_")):
                    continue
                r = f"{rel}/{e.name}" if rel else e.name
                if e.is_dir():
                    if glob is not None and not glob.may_match_below(r):
                        continue
                    sub = _io._canon(e.path) if e.is_symlink() else os.path.join(real_dir, e.name)
                    sub = os.path.normcase(sub)
                    if sub in visited or _io._excluded(sub, excluded):
                        continue
                    visited.add(sub)
                    stack.append((r, sub, e.path))
                elif e.is_file() and _is_warc(e.name) and (glob is None or glob(r)):
                    key = os.path.join(real_dir, os.path.normcase(e.name))
                    if not _io._excluded(key, excluded):
                        uri = os.path.join(root, *r.split("/"))
                        src = _io.Source(uri, r, e.stat().st_size, "warc", None, e.path)
                        found.append((key, src))
    found.sort(key=lambda kv: kv[1].name)
    return found


def _warc_uri_sources(spec: str) -> list[tuple[str, Any]]:
    from pyroid.integrations import _io

    glob: Any = None
    root = spec
    head, sep, rest = spec.rpartition("://")
    web = head.lower() in ("http", "https")
    if sep and not web:
        parts = rest.split("/")
        for i, part in enumerate(parts):
            if _io._MAGIC.search(part):
                root, glob = head + sep + "/".join(parts[:i]), _io._Glob(parts[i:])
                break
    fs, path = _io._filesystem(root, spec)
    kind, size, canon = _io._info(fs, path)
    if glob is None:
        if kind is None:
            raise FileNotFoundError(f"{spec}: no such file or directory")
        if kind == "file":
            name = (spec.split("?", 1)[0] if web else spec).rstrip("/").rsplit("/", 1)[-1]
            return [(_io._key(fs, path), _io.Source(spec, name, size, "warc", fs, path))]
    elif kind != "directory":
        raise _no_warc(spec)
    prefix = canon if not canon or canon.endswith("/") else canon + "/"
    base = root if root.endswith("/") else root + "/"
    found: list[tuple[str, Any]] = []
    for p, n in _io._find(fs, canon):
        if not p.startswith(prefix):
            raise ValueError(f"{spec}: the listing returned {p}, which is not under {canon}")
        rel = p[len(prefix) :]
        comps = rel.split("/")
        if any(c.startswith((".", "_")) or not c for c in comps) or not _is_warc(comps[-1]):
            continue
        if glob is not None and not glob(rel):
            continue
        found.append((_io._key(fs, p), _io.Source(base + rel, rel, n, "warc", fs, p)))
    if not found:
        raise _no_warc(spec)
    found.sort(key=lambda kv: kv[1].name)
    return found


@contextlib.contextmanager
def _warc_open(src: Any) -> Iterator[tuple[Any, str | None]]:
    """What :func:`pyroid.warc.iter_documents` reads for ``src``, and its compression.

    A local file is read by path (in Rust, without the GIL; ``.gz`` through the emulation of
    ``gzip.GzipFile``, as datatrove reads it); a remote one through a file object.
    """
    from pyroid.integrations import _io

    fs = src._fs
    if fs is None:
        yield src._path, "infer"
        return
    gz = "gzip" if src.name.lower().endswith(".gz") else None
    f = (
        fs.fs.open(src._path, "rb")
        if isinstance(fs, _io._Fsspec)
        else fs.open_input_stream(src._path)
    )
    with f:
        yield f, gz


def _warc_recipe(o: Options) -> Any:
    from pyroid.integrations import recipes

    key = (WARC_RECIPE, o.extra.get("lid_model"), o.threads)
    r = _CACHE.get(key)
    if r is None:
        r = _CACHE[key] = recipes.FineWebWarc(lid_model=o.extra.get("lid_model"), threads=o.threads)
    return r


def _warc_pii() -> Any:
    from pyroid.integrations import recipes

    return recipes.FineWebWarc.new_pii()


def _pii_text(t: Any, pii: Any, threads: int | None) -> Any:
    import pyarrow as pa

    from pyroid.integrations import _io

    return _io.set_column(t, "text", pa.chunked_array(pii(t.column("text"), threads=threads)))


def _warc_schema(dump: bool, excluded: bool) -> Any:
    """The columns of fineweb-warc's output (datatrove's metadata order), and ``filter_reason``."""
    import pyarrow as pa

    s = pa.string()
    names = ["text", "id", *(["dump"] if dump else []), "url", "date", "file_path", "language"]
    fields = [pa.field(n, s) for n in names] + [pa.field("language_score", pa.float64())]
    if excluded:
        fields.append(pa.field("filter_reason", s))
    return pa.schema(fields)


def _warc_table(
    docs: Sequence[dict[str, Any]],
    ann: Any,
    rows: Sequence[int],
    file_path: str,
    dump: str | None,
    schema: Any,
) -> Any:
    """Rows ``rows`` of one annotated batch, as a table of ``schema``."""
    import pyarrow as pa

    cols: dict[str, list[Any]] = {
        "text": [ann.text[i] for i in rows],
        "id": [docs[i]["id"] for i in rows],
        "url": [docs[i]["url"] for i in rows],
        "date": [docs[i]["date"] for i in rows],
        "file_path": [file_path] * len(rows),
        "language": [ann.language[i] for i in rows],
        "language_score": [ann.language_score[i] for i in rows],
        "dump": [dump] * len(rows),
        "filter_reason": [ann.reason[i] for i in rows],
    }
    return pa.table({n: pa.array(cols[n], schema.field(n).type) for n in schema.names}, schema)


class _Gather:
    """Tables gathered into one of about ``_WARC_FLUSH_ROWS`` rows (or bytes) before ``sink``.

    A WARC batch keeps few rows: written one by one, they would make tiny Parquet row groups.
    """

    def __init__(self, sink: Callable[[Any], None]) -> None:
        self.sink = sink
        self.parts: list[Any] = []
        self.rows = self.nbytes = 0

    def add(self, t: Any) -> None:
        if not t.num_rows:
            return
        self.parts.append(t)
        self.rows += t.num_rows
        self.nbytes += t.nbytes
        if self.rows >= _WARC_FLUSH_ROWS or self.nbytes >= _WARC_FLUSH_BYTES:
            self.flush()

    def flush(self) -> None:
        if self.parts:
            import pyarrow as pa

            t = pa.concat_tables(self.parts)
            self.parts, self.rows, self.nbytes = [], 0, 0
            self.sink(t)


def _unit_warc(job: FileJob, report: Callable[[int], None]) -> dict[str, Any]:
    """fineweb-warc on one WARC file: up to the filters (MinHash phase 1), or to PII (no dedup)."""
    o = job.options
    x = o.extra
    recipe = _warc_recipe(o)
    task = recipe.new_task()  # the file's Trafilatura dedup cache starts empty, as in datatrove
    dump = x.get("dump")
    schema = _warc_schema(dump is not None, False)
    excl_schema = _warc_schema(dump is not None, True)
    stats: Counter[str] = Counter()
    n_in = n_out = 0
    t0 = time.perf_counter()
    if job.spool is not None:
        out: Any = _Ipc(job.spool)
        excl: Any = _Ipc(job.excluded_spool) if job.excluded_spool else None
    else:
        out, excl = _writer(job.out, o.compression), _writer(job.excluded, o.compression)
    sig = None
    if job.sigs is not None:
        from pyroid._pyroid import dedup as _native
        from pyroid.integrations import _minhash

        assert job.minhash is not None
        sig = _native._SigFileWriter(
            job.sigs,
            job.rank,
            _minhash.native_config(job.minhash),
            memory_budget=job.sort_buffer,
            threads=o.threads,
        )
    pii = _warc_pii() if sig is None else None  # with MinHash, PII comes after it (stage 4)

    def write_kept(t: Any) -> None:
        if sig is not None:
            sig.add(t.column("text"), threads=o.threads)
        out.write(_pii_text(t, pii, o.threads) if pii is not None else t)

    kept = _Gather(write_kept)
    dropped = _Gather(excl.write) if excl is not None else None
    file_path = job.source._path if job.source._fs is None else job.source.uri
    try:
        with _warc_open(job.source) as (source, compression):
            docs = recipe.documents(source, compression=compression, limit=x.get("limit", -1))
            while batch := list(itertools.islice(docs, o.batch_size)):
                ann = recipe.annotate(batch, task)
                keep = [i for i, k in enumerate(ann.keep) if k]
                kept.add(_warc_table(batch, ann, keep, file_path, dump, schema))
                drop = [i for i, k in enumerate(ann.keep) if not k]
                if dropped is not None:
                    dropped.add(_warc_table(batch, ann, drop, file_path, dump, excl_schema))
                stats.update(ann.reason[i] for i in drop)
                n_in += len(batch)
                n_out += len(keep)
                report(len(batch))
        kept.flush()
        if dropped is not None:
            dropped.flush()
        if sig is not None:
            sig.finish()
        out.close(schema)
        if excl is not None:
            excl.close(excl_schema)
    except BaseException:
        _abort(out, excl, sig)
        raise
    result = {
        "input": job.source.uri,
        "size": job.source.size,
        "rank": job.rank,
        "outputs": _job_outputs(job),
        "wall_s": round(time.perf_counter() - t0, 3),
        "stats": {"docs_in": n_in, "docs_out": n_out, "removed": dict(stats)},
    }
    job.state.mark(job.key, result)
    return result


# ====================================================================================== the runs


@dataclass
class Run:
    """Everything one invocation of a per-file command works with."""

    a: argparse.Namespace
    command: str
    sources: list[Any]  # all inputs, in rank order
    shard: Any
    mine: list[tuple[int, Any]]  # this shard's (rank, source)
    layout: Layout
    state: Any
    options: Options
    progress: Any
    t0: float = field(default_factory=time.perf_counter)


def _sources(a: argparse.Namespace, exclude: Sequence[Path]) -> list[Any]:
    from pyroid.integrations import _io

    specs = list(a.input)
    if getattr(a, "files_from", None):
        text = Path(a.files_from).read_text(encoding="utf-8")
        specs += [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    if not specs:
        raise SystemExit("pyroid: no input (give IN paths or --files-from)")
    if getattr(a, "recipe", None) == WARC_RECIPE:
        return _warc_sources(specs, [p for p in exclude if p is not None])
    return _io.resolve_inputs(specs, exclude=[p for p in exclude if p is not None])


def _start(a: argparse.Namespace, command: str, options: Options, *, excluded: bool = True) -> Run:
    from pyroid.integrations import _shards

    shard = _shards.Shard.parse(a.shard) if getattr(a, "shard", None) else _shards.Shard()
    out = Path(a.output)
    excl = Path(a.excluded) if getattr(a, "excluded", None) else None
    tmp = Path(a.tmp_dir) if getattr(a, "tmp_dir", None) else None
    exclude = [p for p in (out, excl, tmp) if p is not None]
    for spec in getattr(a, "input", []):
        if "://" not in spec and any(Path(spec).resolve() == p.resolve() for p in exclude):
            raise SystemExit(f"pyroid: {spec} is both an input and an output (or --tmp-dir)")
    sources = _sources(a, exclude)
    layout = _layout(a, sources, shard, excluded=excluded)
    _check_out(layout, a.overwrite, sources)
    if tmp is not None:
        _private_work(layout.work)
    if layout.out_dir is not None and not layout.merge:
        seen: dict[Path, str] = {}
        for s in sources:
            p, _ = layout.outputs(s)
            if p in seen:
                raise SystemExit(f"pyroid: {seen[p]} and {s.uri} would both be written to {p}")
            seen[p] = s.uri
    fp = _shards.fingerprint(command, _fingerprint_options(options), sources)
    state = _shards.State(layout.state_root, fp)
    mine = shard.take(list(enumerate(sources)))
    if layout.merge and len(mine) > 1:
        layout = replace(layout, ipc_parts=True)
    total = sum(s.size or 0 for _, s in mine) or None
    progress = _shards.Progress(
        f"pyroid {command}", len(mine), total, enabled=getattr(a, "progress", False)
    )
    return Run(a, command, sources, shard, mine, layout, state, options, progress)


def _record_run(run: Run, **info: Any) -> None:
    from pyroid.integrations import _minhash

    run.state.write_run(
        {
            "command": run.command,
            "options": _fingerprint_options(run.options),
            "inputs": _minhash.inputs_digest(run.sources),
            **info,
        }
    )


def _file_units(
    run: Run, keys: Callable[[int], str], make: Callable[[int, Any], FileJob]
) -> list[Any]:
    from pyroid.integrations import _shards

    units = []
    for rank, src in run.mine:
        job = make(rank, src)
        outputs = tuple(_job_outputs(job))
        units.append(_shards.Unit(keys(rank), src.uri, src.size, job, outputs))
    return units


def _merged_outputs(run: Run) -> list[str]:
    lay = run.layout
    return [str(p) for p in (lay.final, lay.excluded_final) if p is not None]


def _merged_done(run: Run) -> bool:
    """A merged output of this shard is complete: its parts are gone and need not be redone."""
    lay = run.layout
    key = f"merged-{lay.shard_tag}"
    return (
        lay.merge and not run.a.overwrite and run.state.is_done(key, outputs=_merged_outputs(run))
    )


def _execute(run: Run, units: Sequence[Any], fn: Any) -> None:
    from pyroid.integrations import _shards

    if _merged_done(run):
        return
    todo = _shards.pending(run.state, units, overwrite=run.a.overwrite)
    run.progress.done_units = len(units) - len(todo)
    for _ in _shards.run_units(
        todo, fn, workers=run.a.workers, threads=run.a.threads, progress=run.progress
    ):
        pass


def _out_job(run: Run, rank: int, src: Any, **kw: Any) -> FileJob:
    """A per-file job whose outputs are its final per-file paths, or its parts when merging."""
    lay = run.layout
    if lay.merge:
        out = lay.part(rank)
        excl = lay.part(rank, excluded=True) if lay.excluded_final is not None else None
    else:
        out, excl = lay.outputs(src)
    key = f"part-{rank:06d}" if lay.merge else f"{rank:06d}"
    return FileJob(
        source=src,
        rank=rank,
        key=key,
        options=run.options,
        state=run.state,
        out=str(out),
        excluded=str(excl) if excl else None,
        **kw,
    )


def _merge_parts(run: Run) -> None:
    """Merged mode: concatenate this shard's parts, in input order, into the final file(s)."""
    lay = run.layout
    assert lay.final is not None
    key = f"merged-{lay.shard_tag}"
    if _merged_done(run):
        return
    ranks = [r for r, _ in run.mine]
    _concat([lay.part(r) for r in ranks], lay.final, run.options.compression)
    outputs = [str(lay.final)]
    if lay.excluded_final is not None:
        _concat([lay.part(r, excluded=True) for r in ranks], lay.excluded_final, None)
        outputs.append(str(lay.excluded_final))
    run.state.mark(key, {"outputs": outputs})


def _move(src: Path, dest: Path) -> None:
    """Rename ``src`` to ``dest``; across filesystems (``--tmp-dir``), copy then rename."""
    import errno

    from pyroid.integrations import _io

    try:
        _io.replace(src, dest)
    except OSError as e:
        if e.errno != errno.EXDEV:
            raise
        # An O_EXCL sibling with a random name: a planted symlink is never written through.
        tmp, fd = _io._sibling(dest, ".copy.partial")
        try:
            with open(fd, "wb") as f, open(src, "rb") as g:
                shutil.copyfileobj(g, f, 1 << 20)
                f.flush()
                os.fsync(f.fileno())
            _io.commit(tmp, dest)
        except BaseException:
            _io.remove(tmp)
            raise
        _io.remove(src)


def _ipc_schema(path: Path) -> Any:
    import pyarrow as pa

    with pa.OSFile(str(path), "rb") as f:
        return pa.ipc.open_file(f).schema


def _parquet_row_groups(path: Path) -> Iterator[Any]:
    """A Parquet file's row groups, one at a time (``iter_batches`` reads far ahead)."""
    import pyarrow.parquet as pq

    with open(path, "rb") as fh, pq.ParquetFile(fh) as pf:
        for i in range(pf.num_row_groups):
            yield pf.read_row_group(i, use_threads=False)


def _concat(parts: Sequence[Path], dest: Path, compression: str | None) -> None:
    """Concatenate part files (all in ``dest``'s format) into ``dest``, atomically."""
    from pyroid.integrations import _io

    dest.parent.mkdir(parents=True, exist_ok=True)
    present = [p for p in parts if p.exists()]
    ipc = any(p.suffix == ".arrow" for p in present)
    if len(present) == 1 and not ipc:
        _move(present[0], dest)
        return
    if _io.file_format(dest) == "jsonl":
        # JSON Lines, plain or as gzip, bz2, zstd or lz4 frames, concatenates as bytes.
        tmp, fd = _io._sibling(dest, ".concat.partial")  # O_EXCL, as in _move
        try:
            with open(fd, "wb") as f:
                for p in present:
                    with open(p, "rb") as g:
                        shutil.copyfileobj(g, f, 1 << 20)
                f.flush()
                os.fsync(f.fileno())
            _io.commit(tmp, dest)
        except BaseException:
            _io.remove(tmp)
            raise
        return
    import pyarrow as pa
    import pyarrow.parquet as pq

    present = [p for p in present if p.stat().st_size]  # an empty JSONL input leaves 0 bytes
    schemas = [_ipc_schema(p) if ipc else pq.read_schema(p) for p in present]
    schema = pa.unify_schemas(schemas, promote_options="permissive") if schemas else None
    w = _io.Writer(dest, compression=compression)
    try:
        for p in present:
            for t in _read_ipc(str(p)) if ipc else _parquet_row_groups(p):
                w.write(t.cast(schema) if schema is not None else t)
        w.close(schema)
    except BaseException:
        w.abort()
        raise


def _finalize(run: Run, *, keep_state: bool) -> dict[str, Any]:
    """Merge parts, sum this shard's stats, write --stats-json, and clean up."""
    from pyroid.integrations import _io, _shards

    lay = run.layout
    if lay.merge and run.mine:
        _merge_parts(run)
    elif lay.merge:
        sys.stderr.write(f"pyroid {run.command}: shard {run.shard} has no input files\n")
    stats = _shards.Stats()
    for rank, _ in run.mine:
        r = run.state.record(_final_key(run, rank))
        if r is not None and "stats" in r:
            stats.add(r["stats"])
    extra: dict[str, Any] = {"files": len(run.mine)}
    if run.shard.count > 1:
        extra["shard"] = str(run.shard)
    d = _shards.summary(run.command, stats, wall=time.perf_counter() - run.t0, extra=extra)
    run.progress.close()
    if lay.out_dir is not None:
        name = f"stats-{lay.shard_tag}.json"
        _io.atomic_write_text(lay.state_root / name, json.dumps(d, indent=1) + "\n")
    if run.a.stats_json:
        _io.atomic_write_text(run.a.stats_json, json.dumps(d, indent=1) + "\n")
    if run.a.stats:
        _shards.print_summary(d)
    _shards.rmtree(lay.work)
    if not keep_state and lay.out_dir is None:
        _shards.rmtree(lay.state_root)
    return d


def _final_key(run: Run, rank: int) -> str:
    return f"part-{rank:06d}" if run.layout.merge else f"{rank:06d}"


# ------------------------------------------------------------------- clean, langid, translit


def _rowwise(a: argparse.Namespace, command: str, **extra: Any) -> int:
    o = _options(a, command, **extra)
    run = _start(a, command, o, excluded=command != "translit")
    warc = extra.get("recipe") == WARC_RECIPE
    if not warc:
        _check_text_column([s for _, s in run.mine], o.text_column)
    _record_run(run)
    units = _file_units(run, lambda r: _final_key(run, r), lambda r, s: _out_job(run, r, s))
    _execute(run, units, _unit_warc if warc else _unit_rows)
    _finalize(run, keep_state=False)
    return 0


def _staged(
    run: Run,
    first: Callable[[int, Any], FileJob],
    first_fn: Any,
    last: Callable[[int, Any], FileJob],
    cfg: dict[str, Any],
) -> None:
    """Run MinHash inside one run, each step resumable.

    ``first`` runs per file and writes the signatures, stages 2-3 run in this process, then
    stage 4 runs per file (``last``).
    """
    from pyroid.integrations import _shards

    a = run.a
    if _merged_done(run):
        return
    finals = _file_units(run, lambda r: _final_key(run, r), last)
    fresh = not a.overwrite and not run.state.stale(u.key for u in finals)
    if fresh and all(run.state.is_done(u.key, u.size) for u in finals):
        return  # finished earlier; only the merge and the stats are left
    p1 = _file_units(run, lambda r: f"p1-{r:06d}", first)
    todo = _shards.pending(run.state, p1, overwrite=a.overwrite)
    if todo:
        # The MinHash decision covers every file: redo it, and stage 4, after any change.
        run.state.forget(["minhash", *(u.key for u in finals)])
    label = run.progress.label
    run.progress.label = f"{label} (1/2)"
    run.progress.done_units = len(p1) - len(todo)
    for _ in _shards.run_units(
        todo, first_fn, workers=a.workers, threads=a.threads, progress=run.progress
    ):
        pass
    run.progress.close()
    work = run.layout.work
    _minhash_middle(run, work / "sigs", work / "dups", work / "removed", cfg, threads=a.threads)
    run.progress = _shards.Progress(
        f"{label} (2/2)", len(run.mine), None, enabled=getattr(a, "progress", False)
    )
    _execute(run, finals, _unit_filter)


def _cmd_clean(a: argparse.Namespace) -> int:
    from pyroid.integrations import _minhash, recipes

    extra: dict[str, Any] = {"recipe": a.recipe, "lid_model": a.lid_model, "dedup": not a.no_dedup}
    warc = a.recipe == WARC_RECIPE
    if warc:
        _check_warc(a)
        extra.update(dump=a.dump, limit=a.limit)
    elif a.dump is not None or a.limit != -1:
        raise SystemExit("pyroid clean: --dump and --limit apply to --recipe fineweb-warc")
    if a.no_dedup:
        return _rowwise(a, "clean", **extra)
    if getattr(a, "shard", None) and not a.shard.endswith("/1"):
        raise SystemExit(
            "pyroid clean: MinHash dedup needs every shard's survivors, so it can't run "
            "inside one --shard. Pass --no-dedup, then run 'pyroid dedup signatures', "
            "'buckets', 'cluster' and 'filter' on the output (see the large datasets guide)."
        )
    cfg = _minhash.config_dict(recipes.MINHASH_CONFIG)
    o = _options(a, "clean", **extra, minhash=cfg)
    run = _start(a, "clean", o)
    if not warc:
        _check_text_column([s for _, s in run.mine], o.text_column)
    _record_run(run, minhash=cfg)
    work = run.layout.work

    def spools(rank: int) -> dict[str, Any]:
        dropped = str(work / "dropped" / f"{rank:06d}.arrow") if o.excluded else None
        return {"spool": str(work / "kept" / f"{rank:06d}.arrow"), "excluded_spool": dropped}

    def first(rank: int, src: Any) -> FileJob:
        # The chain up to the filters; survivors wait in a spool, their signatures in sigs/.
        return FileJob(
            source=src,
            rank=rank,
            key=f"p1-{rank:06d}",
            options=o,
            state=run.state,
            sigs=str(work / "sigs"),
            minhash=cfg,
            sort_buffer=a.sort_buffer_mb << 20,
            **spools(rank),
        )

    def last(rank: int, src: Any) -> FileJob:
        return replace(_out_job(run, rank, src), removed=str(work / "removed"), **spools(rank))

    _staged(run, first, _unit_warc if warc else _unit_rows, last, cfg)
    _finalize(run, keep_state=False)
    return 0


def _check_warc(a: argparse.Namespace) -> None:
    """fineweb-warc's dependencies (exit 1 with an install hint), and its output format."""
    import importlib.util

    need = ("datatrove", "tldextract", "fasteners")
    missing = ", ".join(m for m in need if importlib.util.find_spec(m) is None)
    if missing:
        raise SystemExit(
            f"pyroid clean --recipe {WARC_RECIPE}: needs {missing} (the URL filter "
            "reads datatrove's block lists): pip install 'pyroid[cli,warc,datatrove]' "
            "tldextract fasteners"
        )
    if not getattr(a, "format", None) and not _is_file_out(a.output):
        a.format = "parquet"  # per-file outputs: a WARC input has no table format to keep


def _minhash_middle(
    run: Run, sigs: Path, dups: Path, removed: Path, cfg: dict[str, Any], *, threads: int | None
) -> None:
    """Stages 2 and 3 in this process (all buckets, in parallel threads)."""
    from pyroid._pyroid import dedup as _native
    from pyroid.integrations import _io, _minhash

    if not run.a.overwrite and run.state.is_done("minhash", outputs=[str(removed)]):
        return
    units = _minhash.bucket_units(cfg["num_buckets"], 0, 1)
    conc = min(len(units), threads or os.cpu_count() or 1)
    max_open = _minhash.open_files_limit(conc)
    n = len(run.sources)

    def bucket(u: Any) -> int:
        return _minhash.run_bucket(sigs, dups, u, cfg, max_open=max_open, expected=n)

    with cf.ThreadPoolExecutor(conc) as pool:
        list(pool.map(bucket, units))
    removed.mkdir(parents=True, exist_ok=True)
    for old in removed.iterdir():  # lists of an earlier, unfinished attempt
        _io.remove(old)
    res = _native._minhash_cluster(_minhash.dups_files(dups), str(removed))
    run.state.mark("minhash", {"outputs": [str(removed)], "cluster": res})


def _cmd_langid(a: argparse.Namespace) -> int:
    keep = [x.strip() for x in a.keep.split(",") if x.strip()] if a.keep else None
    return _rowwise(a, "langid", model=a.model, keep=keep, min_score=a.min_score, top_k=a.top_k)


def _cmd_translit(a: argparse.Namespace) -> int:
    return _rowwise(
        a,
        "translit",
        mode=a.mode,
        errors=a.errors,
        replace_str=a.replace_str,
        output_column=a.output_column,
    )


# ------------------------------------------------------------------------------------------ dedup


def _minhash_cfg_from_args(a: argparse.Namespace) -> dict[str, Any]:
    from pyroid.integrations import _minhash

    return _minhash.config_dict(
        None,
        n_grams=a.n_grams,
        num_buckets=a.num_buckets,
        hashes_per_bucket=a.hashes_per_bucket,
        seed=a.seed,
        hash_fc=a.hash_fc,
        precision=a.precision,
    )


def _cmd_dedup(a: argparse.Namespace) -> int:
    if a.method == "exact":
        return _dedup_exact(a)
    if getattr(a, "shard", None) and not a.shard.endswith("/1"):
        raise SystemExit(
            "pyroid dedup: --shard needs the staged flow: 'pyroid dedup signatures', "
            "'buckets', 'cluster', then 'filter'"
        )
    cfg = _minhash_cfg_from_args(a)
    o = _options(a, "dedup", method="minhash", minhash=cfg)
    run = _start(a, "dedup", o)
    _check_text_column([s for _, s in run.mine], o.text_column)
    _record_run(run, minhash=cfg)
    work = run.layout.work

    def first(rank: int, src: Any) -> FileJob:
        return FileJob(
            source=src,
            rank=rank,
            key=f"p1-{rank:06d}",
            options=o,
            state=run.state,
            sigs=str(work / "sigs"),
            minhash=cfg,
            sort_buffer=a.sort_buffer_mb << 20,
        )

    def last(rank: int, src: Any) -> FileJob:
        return replace(_out_job(run, rank, src), removed=str(work / "removed"))

    _staged(run, first, _unit_signatures, last, cfg)
    _finalize(run, keep_state=False)
    return 0


def _dedup_exact(a: argparse.Namespace) -> int:
    """Exact dedup keeps the first copy over all inputs in order: one process, one pass."""
    import pyarrow as pa
    import pyarrow.compute as pc

    import pyroid.dedup
    from pyroid.integrations import _io, _shards

    if (getattr(a, "shard", None) and not a.shard.endswith("/1")) or a.workers > 1:
        raise SystemExit("pyroid dedup --method exact: runs in one process (no --shard/--workers)")
    o = _options(a, "dedup", method="exact", normalize=a.normalize)
    run = _start(a, "dedup", o)
    lay = run.layout
    if not lay.merge or lay.final is None:
        raise SystemExit("pyroid dedup --method exact: OUT must be a file")
    _check_text_column([s for _, s in run.mine], o.text_column)
    state = pyroid.dedup.ExactDedup(normalize=a.normalize)
    out = _io.Writer(lay.final, compression=o.compression)
    excl = _io.Writer(lay.excluded_final, compression=o.compression) if lay.excluded_final else None
    stats = _shards.Stats()
    schema = excl_schema = None
    try:
        for _, src in run.mine:
            for t in _io.iter_tables(src, o.batch_size):
                mask = pa.array(state.add(_text(t, o.text_column), threads=a.threads), pa.bool_())
                kept = t.filter(mask)
                schema = t.schema
                stats.docs_in += t.num_rows
                stats.docs_out += kept.num_rows
                stats.removed["exact"] += t.num_rows - kept.num_rows
                out.write(kept)
                if excl is not None:
                    d = t.filter(pc.invert(mask))
                    reason = pa.array(["exact"] * d.num_rows, pa.string())
                    excl_schema = (d := d.append_column("filter_reason", reason)).schema
                    excl.write(d)
                run.progress.docs_seen(t.num_rows)
            run.progress.unit_done(src.size)
        out.close(schema)
        if excl is not None:
            if excl_schema is None and schema is not None:
                excl_schema = schema.append(pa.field("filter_reason", pa.string()))
            excl.close(excl_schema)
    except BaseException:
        _abort(out, excl)
        raise
    wall = time.perf_counter() - run.t0
    d = _shards.summary("dedup", stats, wall=wall, extra={"files": len(run.mine)})
    run.progress.close()
    if a.stats_json:
        _io.atomic_write_text(a.stats_json, json.dumps(d, indent=1) + "\n")
    if a.stats:
        _shards.print_summary(d)
    return 0


# ---------------------------------------------------------------------------- dedup: the 4 stages


def _stage_signatures(a: argparse.Namespace) -> int:

    cfg = _minhash_cfg_from_args(a)
    o = _options(a, "dedup signatures", minhash=cfg)
    run = _start(a, "dedup signatures", o, excluded=False)
    if run.layout.merge:
        raise SystemExit("pyroid dedup signatures: SIGS is a directory")
    _check_text_column([s for _, s in run.mine], o.text_column)
    _record_run(run, minhash=cfg, stage="signatures")
    sigs = run.layout.out_dir
    assert sigs is not None

    def make(rank: int, src: Any) -> FileJob:
        return FileJob(
            source=src,
            rank=rank,
            key=f"{rank:06d}",
            options=o,
            state=run.state,
            sigs=str(sigs),
            minhash=cfg,
            sort_buffer=a.sort_buffer_mb << 20,
        )

    _execute(run, _file_units(run, lambda r: f"{r:06d}", make), _unit_signatures)
    _finalize(run, keep_state=True)
    return 0


def _stage_meta(folder: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((folder / ".pyroid" / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _stage_cfg(a: argparse.Namespace, meta: dict[str, Any] | None, where: Path) -> dict[str, Any]:
    """The MinHash settings recorded upstream, checked against any flags given."""
    from pyroid.integrations import _minhash

    flags = {k: getattr(a, k) for k in _minhash.CONFIG_FIELDS if getattr(a, k, None) is not None}
    if meta is None or "minhash" not in meta:
        return _minhash.config_dict(None, **flags)
    cfg = dict(meta["minhash"])
    for k, v in flags.items():
        if cfg.get(k) != v:
            raise SystemExit(
                f"pyroid dedup: --{k.replace('_', '-')} {v} differs from {cfg.get(k)} "
                f"recorded in {where}"
            )
    return cfg


def _check_complete(folder: Path, meta: dict[str, Any] | None, what: str, hint: str) -> None:
    if meta is None:
        return
    want = meta.get("inputs", {}).get("count")
    done = folder / ".pyroid" / "done"
    have = len(list(done.glob("*.json"))) if done.is_dir() else 0
    if want is not None and have < want:
        raise SystemExit(
            f"pyroid dedup: {folder} is incomplete: {have} of {want} {what} are done ({hint})"
        )


def _simple_state(folder: Path, command: str, info: dict[str, Any]) -> Any:
    from pyroid.integrations import _shards

    fp = _shards.fingerprint(command, info, [])
    return _shards.State(folder / ".pyroid", fp)


def _stage_buckets(a: argparse.Namespace) -> int:
    from pyroid.integrations import _minhash, _shards

    sigs, out = Path(a.sigs), Path(a.output)
    if not sigs.is_dir():
        raise SystemExit(f"pyroid dedup buckets: {sigs}: no such directory")
    meta = _stage_meta(sigs)
    _check_complete(sigs, meta, "input files", "run every 'pyroid dedup signatures' shard first")
    cfg = _stage_cfg(a, meta, sigs)
    shard = _shards.Shard.parse(a.shard) if a.shard else _shards.Shard()
    units = _minhash.bucket_units(cfg["num_buckets"], shard.index, shard.count)
    # The signatures, by their own run's fingerprint (or by file names and sizes, for
    # datatrove's), never by path: a machine with another mount point resumes the same run.
    ident = (meta or {}).get("fingerprint") or _shards.folder_digest(sigs, "bucket_*/*.minhash.sig")
    inputs = (meta or {}).get("inputs")
    info = {"minhash": cfg, "inputs": inputs, "sigs": ident}
    state = _simple_state(out, "dedup buckets", info)

    def dest(u: Any) -> str:
        return str(out / _minhash.dups_name(u.bucket, u.worker))

    todo = [u for u in units if a.overwrite or not state.is_done(u.key, outputs=[dest(u)])]
    if state.stale(u.key for u in units) and not a.overwrite:
        raise SystemExit(
            f"pyroid dedup buckets: {out} holds buckets of other signatures; pass --overwrite"
        )
    t0 = time.perf_counter()
    conc = max(1, min(len(todo), a.threads or os.cpu_count() or 1))
    max_open = _minhash.open_files_limit(conc)
    state.write_run({"stage": "buckets", **info})

    expected = inputs.get("count") if inputs else None

    def one(u: Any) -> int:
        n = _minhash.run_bucket(sigs, out, u, cfg, max_open=max_open, expected=expected)
        record = {"pairs": n, "workers_per_bucket": u.workers_per_bucket, "outputs": [dest(u)]}
        state.mark(u.key, record)
        return n

    with cf.ThreadPoolExecutor(conc) as pool:
        pairs = sum(pool.map(one, todo))
    if a.stats or a.stats_json:
        d = {
            "command": "dedup buckets",
            "buckets": [u.key for u in units],
            "pairs": pairs,
            "wall_s": round(time.perf_counter() - t0, 3),
            "pyroid": pyroid.__version__,
        }
        _emit(a, d)
    return 0


def _stage_cluster(a: argparse.Namespace) -> int:
    from pyroid._pyroid import dedup as _native
    from pyroid.integrations import _io, _minhash, _shards

    dups, out = Path(a.dups), Path(a.output)
    if not dups.is_dir():
        raise SystemExit(f"pyroid dedup cluster: {dups}: no such directory")
    meta = _stage_meta(dups)
    cfg = _stage_cfg(a, meta, dups)
    files = _minhash.dups_files(dups)
    if meta is not None:
        # Only the bucket tasks of the run that wrote DUPS/.pyroid/run.json count.
        recs = [
            r
            for r in _shards.State(dups / ".pyroid", "").records()
            if r.get("fingerprint") == meta.get("fingerprint")
        ]
        wpb = {int(r.get("workers_per_bucket", 1)) for r in recs}
        if len(wpb) > 1:
            raise SystemExit(
                f"pyroid dedup cluster: {dups} mixes buckets split {sorted(wpb)} ways; rerun "
                "'pyroid dedup buckets' into an empty directory"
            )
        want = cfg["num_buckets"] * (wpb.pop() if wpb else 1)
        if len(recs) < want:
            raise SystemExit(
                f"pyroid dedup cluster: {dups} is incomplete: {len(recs)} of {want} bucket "
                "tasks are done (run every 'pyroid dedup buckets' shard first)"
            )
        files = sorted(str(dups / f"{r['key']}.dups") for r in recs)
    if len(files) % cfg["num_buckets"]:
        raise SystemExit(
            f"pyroid dedup cluster: {len(files)} .dups files is not a multiple of "
            f"{cfg['num_buckets']} buckets"
        )
    t0 = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    _io.remove(out / ".pyroid" / "run.json")  # written again only once this run is complete
    for old in out.iterdir():  # an earlier cluster run's lists must not survive this one
        if old.suffix in (".remove", ".clusters", ".sizes") and old.stem.isdigit():
            _io.remove(old)
    res = _native._minhash_cluster(
        files,
        str(out),
        save_cluster_id=a.save_cluster_id,
        save_cluster_size=a.save_cluster_size,
        ignore_index_matches=a.ignore_index_matches,
    )
    ident = (meta or {}).get("fingerprint") or _shards.folder_digest(dups, "*.dups")
    info = {"minhash": cfg, "inputs": (meta or {}).get("inputs"), "dups": ident}
    state = _simple_state(out, "dedup cluster", info)
    state.write_run({"stage": "cluster", **info, "cluster": res})
    state.mark("cluster", {"cluster": res})
    if a.stats or a.stats_json:
        _emit(
            a,
            {
                "command": "dedup cluster",
                **res,
                "wall_s": round(time.perf_counter() - t0, 3),
                "pyroid": pyroid.__version__,
            },
        )
    return 0


def _stage_filter(a: argparse.Namespace) -> int:
    from pyroid.integrations import _minhash, _shards

    removed = Path(a.removed)
    if not removed.is_dir():
        raise SystemExit(f"pyroid dedup filter: --removed {removed}: no such directory")
    meta = _stage_meta(removed)
    if meta is None and (removed / ".pyroid").is_dir():
        raise SystemExit(
            f"pyroid dedup filter: {removed} is incomplete (the cluster stage did not finish); "
            "run 'pyroid dedup cluster' again"
        )
    lists = ("*.remove", "*.clusters", "*.sizes")
    if meta is None and not any(removed.glob("*.remove")):
        sys.stderr.write(
            f"pyroid dedup filter: {removed} has no .remove files: no document is removed\n"
        )
    # REMOVED by its cluster run's fingerprint (or its files, for datatrove's), never by path
    ident = (meta or {}).get("fingerprint") or _shards.folder_digest(removed, *lists)
    o = _options(
        a,
        "dedup filter",
        removed=ident,
        cluster_ids=a.load_cluster_ids,
        cluster_sizes=a.load_cluster_sizes,
    )
    run = _start(a, "dedup filter", o)
    want = (meta or {}).get("inputs")
    if want and want != _minhash.inputs_digest(run.sources):
        raise SystemExit(
            f"pyroid dedup filter: the inputs ({len(run.sources)} files) are not the ones the "
            f"signatures were computed from ({want.get('count')} files, recorded in {removed})"
        )
    _record_run(run, stage="filter")
    units = _file_units(
        run,
        lambda r: _final_key(run, r),
        lambda r, s: replace(
            _out_job(run, r, s),
            removed=str(removed),
            cluster_ids=a.load_cluster_ids,
            cluster_sizes=a.load_cluster_sizes,
        ),
    )
    _execute(run, units, _unit_filter)
    _finalize(run, keep_state=False)
    return 0


def _emit(a: argparse.Namespace, d: dict[str, Any]) -> None:
    from pyroid.integrations import _io

    if a.stats_json:
        _io.atomic_write_text(a.stats_json, json.dumps(d, indent=1) + "\n")
    if a.stats:
        sys.stderr.write(" ".join(f"{k}={v}" for k, v in d.items()) + "\n")


# ------------------------------------------------------------------------------------------ stats


def _cmd_stats(a: argparse.Namespace) -> int:
    from pyroid.integrations import _io, _shards

    root = Path(a.output) / ".pyroid"
    if not root.is_dir():
        raise SystemExit(f"pyroid stats: {a.output} has no pyroid run records (.pyroid/)")
    state = _shards.State(root, "")
    run = state.read_run() or {}
    stats = _shards.Stats()
    files = 0
    for r in state.records():
        if "stats" in r and not str(r["key"]).startswith("p1-"):
            stats.add(r["stats"])
            files += 1
    total = (run.get("inputs") or {}).get("count")
    d = _shards.summary(
        str(run.get("command", "?")),
        stats,
        wall=0.0,
        extra={"files": files, "files_total": total},
    )
    d.pop("wall_s")
    if a.json:
        _io.atomic_write_text(a.json, json.dumps(d, indent=1) + "\n")
    else:
        sys.stdout.write(json.dumps(d, indent=1) + "\n")
    return 0


# ========================================================================================== parser


def _positive(s: str) -> int:
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return v


def _threads(s: str) -> int | None:
    v = int(s)
    if v < 0:
        raise argparse.ArgumentTypeError("must be >= 0 (0 = the whole pool)")
    return v or None


def _shard(s: str) -> str:
    from pyroid.integrations._shards import Shard

    try:
        Shard.parse(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e).removeprefix(f"--shard {s!r}: ")) from None
    return s


def _io_args(
    p: argparse.ArgumentParser, *, out: str = "OUT", out_help: str | None = None, warc: bool = False
) -> None:
    formats = f"{_FORMATS}; fineweb-warc: .warc, .arc, .wet[.gz]" if warc else _FORMATS
    p.add_argument(
        "input",
        nargs="*",
        metavar="IN",
        help=f"input files ({formats}), directories, globs or URIs",
    )
    p.add_argument(
        "output",
        metavar=out,
        help=out_help or "output file, or a directory for one output per input file (see --merge)",
    )
    p.add_argument("--files-from", metavar="PATH", help="read more inputs from PATH, one per line")


def _run_args(p: argparse.ArgumentParser, *, excluded: bool = True, merge: bool = True) -> None:
    p.add_argument("--text-column", default="text", help="the text column (default: text)")
    p.add_argument(
        "--threads",
        type=_threads,
        default=None,
        help="threads per process (default: 0, the whole pool, sized by $PYROID_NUM_THREADS; "
        "with --workers, an equal share of the CPUs)",
    )
    p.add_argument(
        "--batch-size",
        type=_positive,
        default=None,
        help="rows per Arrow batch (default: 10000; --recipe fineweb-warc: pages per batch, 256)",
    )
    p.add_argument(
        "--shard", type=_shard, metavar="I/N", help="process input files I, I+N, I+2N, ..."
    )
    p.add_argument(
        "--workers",
        type=_positive,
        default=1,
        metavar="N",
        help="worker processes, one input file each (default: 1)",
    )
    if merge:
        p.add_argument(
            "--merge",
            action="store_true",
            help="with a directory OUT, write one file per shard (part-I-of-N) instead of one "
            "per input file",
        )
        p.add_argument(
            "--format",
            choices=_OUT_FORMATS,
            metavar="FMT",
            help="with a directory OUT, the output format: "
            + ", ".join(_OUT_FORMATS)
            + " (default: each input's; parquet for WARC inputs)",
        )
    if excluded:
        p.add_argument(
            "--excluded",
            metavar="PATH",
            help="also write the dropped rows here (a file, or a directory like OUT)",
        )
    p.add_argument(
        "--stats", action="store_true", help="print row counts and drops by reason to stderr"
    )
    p.add_argument("--stats-json", metavar="PATH", help="write the stats to PATH as JSON")
    p.add_argument("--progress", action="store_true", help="show a progress line on stderr")
    p.add_argument("--compression", help="Parquet compression (default: pyarrow's, snappy)")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="redo finished work and replace existing outputs",
    )


def _minhash_args(p: argparse.ArgumentParser, *, sort_buffer: bool = True) -> None:
    g = p.add_argument_group("MinHash options (datatrove's MinhashConfig fields and defaults)")
    g.add_argument("--n-grams", type=_positive, help="words per shingle (default: 5)")
    g.add_argument("--num-buckets", type=_positive, help="LSH bands (default: 14)")
    g.add_argument("--hashes-per-bucket", type=_positive, help="hashes per band (default: 8)")
    g.add_argument("--seed", type=int, help="seed of the permutations (default: 1)")
    g.add_argument(
        "--hash-fc", choices=("xxhash", "sha1"), help="shingle hash function (default: xxhash)"
    )
    g.add_argument(
        "--precision", type=int, choices=(32, 64), help="shingle hash bits (default: 64)"
    )
    if sort_buffer:
        _sort_buffer_arg(g)


def _sort_buffer_arg(p: Any) -> None:
    p.add_argument(
        "--sort-buffer-mb",
        type=_positive,
        default=64,
        metavar="MB",
        help="signatures sorted in memory per process before spilling to disk (default: 64)",
    )


def build_parser() -> argparse.ArgumentParser:
    """The argument parser (``pyroid run`` and the ``dedup`` stages are dispatched before it)."""
    ap = argparse.ArgumentParser(
        prog="pyroid",
        description="Clean text datasets for LLM training, in Parquet or JSON Lines files.",
        epilog="Across machines, 'pyroid dedup signatures', 'buckets', 'cluster' and 'filter' "
        "run datatrove's four MinHash stages; see 'pyroid dedup --help'.",
    )
    ap.add_argument("--version", action="version", version=f"pyroid {pyroid.__version__}")
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser(
        "clean",
        help="run a cleaning recipe (ftfy, LID, quality filters, MinHash; or FineWeb from WARC)",
        description="Run a cleaning recipe and write the kept rows. The fineweb recipe is "
        "datatrove's FineWeb chain on text: ftfy, fastText LID, the Gopher, C4 and FineWeb "
        "quality filters, then MinHash dedup. The fineweb-warc recipe is datatrove's "
        "examples/fineweb.py on WARC files: datatrove's WarcReader, URL filter, Trafilatura, "
        "fastText LID, the same filters, MinHash dedup, then PII replacement.",
    )
    _io_args(p, warc=True)
    _run_args(p)
    p.add_argument(
        "--recipe",
        choices=RECIPES,
        required=True,
        help="fineweb (Parquet or JSONL text) or fineweb-warc (WARC, ARC or WET files)",
    )
    p.add_argument(
        "--dump",
        metavar="NAME",
        help="fineweb-warc: add a dump column holding NAME (datatrove's default_metadata)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=-1,
        metavar="N",
        help="fineweb-warc: read at most N documents per WARC file (WarcReader's limit)",
    )
    p.add_argument(
        "--lid-model",
        help="fastText's lid.176.bin (default: $PYROID_LID_MODEL, else a cached download)",
    )
    p.add_argument(
        "--no-dedup",
        action="store_true",
        help="skip the MinHash stage (needed with --shard; run the 'pyroid dedup' stages after)",
    )
    p.add_argument(
        "--tmp-dir",
        help="where rows and signatures wait for the MinHash decision (default: next to OUT)",
    )
    _sort_buffer_arg(p)
    p.set_defaults(body=_cmd_clean)

    p = sub.add_parser(
        "langid",
        help="add fastText language columns; optionally keep some languages",
        description="Add datatrove's language and language_score columns from fastText "
        "language ID, and optionally keep only some languages.",
    )
    _io_args(p)
    _run_args(p)
    p.add_argument("--model", help="fastText model file (default: lid.176.bin, found as for clean)")
    p.add_argument("--keep", metavar="LANGS", help="comma-separated labels to keep, e.g. en,fr")
    p.add_argument(
        "--min-score",
        type=float,
        default=0.65,
        help="with --keep: datatrove's language_threshold (default: 0.65)",
    )
    p.add_argument(
        "--top-k",
        type=_positive,
        default=1,
        help="if above 1, also add top_languages and top_language_scores (default: 1)",
    )
    p.set_defaults(body=_cmd_langid)

    p = sub.add_parser(
        "dedup",
        help="drop near-duplicates (datatrove's MinHash) or exact duplicates",
        description="Drop near-duplicates with datatrove's MinHash dedup, or exact "
        "duplicates (keeping the first copy).",
        epilog="Across machines, run datatrove's four stages instead: "
        "'pyroid dedup signatures IN... SIGS', 'pyroid dedup buckets SIGS DUPS', "
        "'pyroid dedup cluster DUPS REMOVED', 'pyroid dedup filter IN... OUT --removed REMOVED'. "
        "An input named like a stage needs a path prefix (./signatures).",
    )
    _io_args(p)
    _run_args(p)
    p.add_argument(
        "--method",
        choices=("minhash", "exact"),
        default="minhash",
        help="dedup method (default: minhash)",
    )
    p.add_argument(
        "--normalize",
        action="store_true",
        help="--method exact: compare texts after datatrove's simplify_text()",
    )
    p.add_argument("--tmp-dir", help="where signatures wait for the MinHash decision")
    _minhash_args(p)
    p.set_defaults(body=_cmd_dedup)

    p = sub.add_parser(
        "translit",
        help="transliterate to ASCII (anyascii data, ISC)",
        description="Transliterate the text column to ASCII with anyascii's data (ISC licence).",
    )
    _io_args(p)
    _run_args(p, excluded=False)
    p.add_argument(
        "--mode",
        choices=("anyascii", "unidecode"),
        default="anyascii",
        help="anyascii (default), or unidecode for Unidecode's errors= handling",
    )
    p.add_argument(
        "--errors",
        default="ignore",
        help="--mode unidecode: ignore (default), replace, strict or preserve",
    )
    p.add_argument(
        "--replace-str",
        default="?",
        help="--mode unidecode: replacement for --errors replace (default: ?)",
    )
    p.add_argument("--output-column", help="write here instead of replacing the text column")
    p.set_defaults(body=_cmd_translit)

    p = sub.add_parser(
        "stats",
        help="sum the stats of every finished file of a directory OUT (all shards)",
        description="Sum the per-file stats recorded in OUT/.pyroid by every shard that wrote "
        "to OUT, and print them as JSON.",
    )
    p.add_argument("output", metavar="OUT", help="a directory OUT of clean, langid, dedup, ...")
    p.add_argument("--json", metavar="PATH", help="write the JSON to PATH instead of stdout")
    p.set_defaults(body=_cmd_stats)
    # Listed for --help only: main() hands `pyroid run ...` to pyroid.integrations.run first.
    sub.add_parser(
        "run",
        help="run a Python program with pyroid.patch() applied (see 'pyroid run --help')",
        add_help=False,
    )
    return ap


def build_stage_parser(stage: str) -> argparse.ArgumentParser:
    """The parser of ``pyroid dedup STAGE``."""
    prog = f"pyroid dedup {stage}"
    if stage == "signatures":
        p = argparse.ArgumentParser(
            prog=prog,
            description="MinHash stage 1 (datatrove's MinhashDedupSignature): write each input "
            "file's signatures to SIGS/bucket_XXX/RANK.minhash.sig, RANK being the file's "
            "position in the sorted input list.",
        )
        _io_args(p, out="SIGS", out_help="the signatures directory")
        _run_args(p, excluded=False, merge=False)
        _minhash_args(p)
        p.set_defaults(body=_stage_signatures)
    elif stage == "buckets":
        p = argparse.ArgumentParser(
            prog=prog,
            description="MinHash stage 2 (datatrove's MinhashDedupBuckets): merge each bucket's "
            "signatures and write the duplicate pairs to DUPS/BUCKET_WORKER.dups. --shard "
            "splits the buckets (at most num_buckets shards, or a multiple of it to split "
            "buckets by hash range); --threads runs that many buckets at once.",
        )
        p.add_argument("sigs", metavar="SIGS", help="the signatures directory")
        p.add_argument("output", metavar="DUPS", help="the duplicate-pairs directory")
        p.add_argument("--shard", type=_shard, metavar="I/N", help="this shard of the buckets")
        p.add_argument(
            "--threads", type=_threads, default=None, help="buckets at once (default: CPUs)"
        )
        p.add_argument("--stats", action="store_true", help="print a summary to stderr")
        p.add_argument("--stats-json", metavar="PATH", help="write the summary to PATH as JSON")
        p.add_argument("--overwrite", action="store_true", help="redo finished buckets")
        _minhash_args(p, sort_buffer=False)
        p.set_defaults(body=_stage_buckets)
    elif stage == "cluster":
        p = argparse.ArgumentParser(
            prog=prog,
            description="MinHash stage 3 (datatrove's MinhashDedupCluster): join the duplicate "
            "pairs into clusters and write the documents to remove to REMOVED/RANK.remove. "
            "One process; memory grows with the number of documents in clusters.",
        )
        p.add_argument("dups", metavar="DUPS", help="the duplicate-pairs directory")
        p.add_argument("output", metavar="REMOVED", help="the removal-lists directory")
        p.add_argument("--save-cluster-id", action="store_true", help="also write RANK.clusters")
        p.add_argument("--save-cluster-size", action="store_true", help="also write RANK.sizes")
        p.add_argument(
            "--ignore-index-matches", action="store_true", help="datatrove's option of that name"
        )
        p.add_argument("--stats", action="store_true", help="print a summary to stderr")
        p.add_argument("--stats-json", metavar="PATH", help="write the summary to PATH as JSON")
        _minhash_args(p, sort_buffer=False)
        p.set_defaults(body=_stage_cluster)
    else:
        p = argparse.ArgumentParser(
            prog=prog,
            description="MinHash stage 4 (datatrove's MinhashDedupFilter): copy each input file "
            "without the rows listed in REMOVED. IN must be the inputs of the signatures stage.",
        )
        _io_args(p)
        _run_args(p)
        p.add_argument(
            "--removed", required=True, metavar="REMOVED", help="the cluster stage's output"
        )
        p.add_argument(
            "--load-cluster-ids", action="store_true", help="add a minhash_cluster_id column"
        )
        p.add_argument(
            "--load-cluster-sizes", action="store_true", help="add a minhash_cluster_size column"
        )
        p.set_defaults(body=_stage_filter)
    p.set_defaults(command=f"dedup {stage}")
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``pyroid`` console script."""
    from pyroid.integrations import _shards

    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "run":
        from pyroid.integrations import run

        return run(args[1:])
    if len(args) > 1 and args[0] == "dedup" and args[1] in _STAGES:
        a = build_stage_parser(args[1]).parse_args(args[2:])
    else:
        a = build_parser().parse_args(args)
    _need_pyarrow()
    try:
        return int(a.body(a))
    except _shards.StateError as e:
        raise SystemExit(f"pyroid {a.command}: {e}") from None
    except (FileNotFoundError, ValueError, ImportError) as e:
        if os.environ.get("PYROID_DEBUG"):
            raise
        msg = _import_hint(e) if isinstance(e, ImportError) else str(e)
        notes = "".join(f"\n  ({n})" for n in getattr(e, "__notes__", ()))
        print(f"pyroid {a.command}: {msg}{notes}", file=sys.stderr)  # noqa: T201
        return 1


# What installs a module the commands import only when they need it (top-level name: pip args).
_INSTALL = {
    "numpy": "'pyroid[cli]'",
    "pyarrow": "'pyroid[cli]'",
    "cchardet": "'pyroid[warc]'",
    "magic": "'pyroid[warc]'",
    "datatrove": "'pyroid[datatrove]'",
    "orjson": "'pyroid[datatrove]'",
    "regex": "'pyroid[datatrove]'",
    "tokenizers": "'pyroid[datatrove]'",
    "datasets": "'pyroid[hf]'",
    "tldextract": "tldextract",
    "fasteners": "fasteners",
}


def _import_hint(e: ImportError) -> str:
    """The message of a missing optional dependency, with what installs it."""
    msg = str(e)
    if "pip install" in msg:
        return msg
    if "libmagic" in msg:
        return f"{msg} (python-magic needs the libmagic library, e.g. apt install libmagic1)"
    pip = _INSTALL.get((e.name or "").partition(".")[0])
    return f"{msg} (pip install {pip})" if pip else msg


if __name__ == "__main__":
    sys.exit(main())
