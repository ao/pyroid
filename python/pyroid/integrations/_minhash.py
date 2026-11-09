"""datatrove's four MinHash stages on disk, for ``pyroid dedup signatures|buckets|cluster|filter``.

The folders have datatrove 0.10.1's layout and byte format, so either tool can run any stage on
the other's files:

- signatures: ``bucket_{b:03d}/{rank:05d}.minhash.sig``, one per bucket per input file;
- buckets: ``{bucket:05d}_{worker:02d}.dups`` (pairs of ``(rank, doc)``);
- cluster: ``{rank:06d}.remove`` (doc indices), optionally ``.clusters`` and ``.sizes``.

A rank is an input file's position in the sorted input list, and a doc index is a row's position
in that file. This is what datatrove does with one task per input file (or one task in total),
and it numbers documents in input order, so the decisions equal one in-memory pass over all
files in order (:class:`pyroid.dedup.MinHashDedup`).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyroid.dedup
from pyroid._pyroid import dedup as _native

from . import _io

_dedup: Any = pyroid.dedup  # its private helpers are not in dedup.pyi

SIG_SUFFIX = ".minhash.sig"

#: The MinhashConfig fields the CLI exposes (the text normalisation is datatrove's default).
CONFIG_FIELDS = ("n_grams", "num_buckets", "hashes_per_bucket", "seed", "hash_fc", "precision")

#: Signature records sorted in memory before they spill to disk (per writer).
SORT_BUFFER = 256 << 20
#: Read buffers of one bucket merge.
MERGE_BUFFER = 64 << 20


def config_dict(cfg: Any = None, **overrides: Any) -> dict[str, Any]:
    """The CLI's MinHash settings as a plain dict (validated through ``pyroid.dedup``)."""
    kw = {k: v for k, v in overrides.items() if v is not None}
    c = _dedup._minhash_config(cfg, kw)
    return {
        "n_grams": c.n_grams,
        "num_buckets": c.num_buckets,
        "hashes_per_bucket": c.hashes_per_bucket,
        "seed": c.seed,
        "hash_fc": c.hash_config.hash_fc,
        "precision": c.hash_config.precision,
    }


def native_config(cfg: Mapping[str, Any]) -> Any:
    """The ``_NativeConfig`` for ``cfg`` (English word tokenizer, as datatrove's default)."""
    c = _dedup._minhash_config(None, dict(cfg))
    src = _dedup._words_source("eng", False, "pyroid dedup")
    return _dedup._native_config(c, src)


def inputs_digest(sources: Sequence[_io.Source]) -> dict[str, Any]:
    """What identifies an input list across machines: names and sizes, not absolute paths."""
    h = hashlib.sha256()
    for s in sources:
        h.update(json.dumps([s.name, s.size]).encode() + b"\n")
    return {"count": len(sources), "digest": h.hexdigest()[:16]}


# ----------------------------------------------------------------------------------------- layout


def sig_files(folder: str | os.PathLike[str], bucket: int) -> list[str]:
    """One bucket's signature files in datatrove's reader order (file names as strings)."""
    d = Path(folder) / f"bucket_{bucket:03d}"
    if not d.is_dir():
        return []
    names = sorted(
        p.name
        for p in d.iterdir()
        if p.name.endswith(SIG_SUFFIX) and not p.name.startswith(".") and p.is_file()
    )
    return [str(d / n) for n in names]


def dups_name(bucket: int, worker: int) -> str:
    return f"{bucket:05d}_{worker:02d}.dups"


def dups_files(folder: str | os.PathLike[str]) -> list[str]:
    """The ``.dups`` files in datatrove's (name) order."""
    d = Path(folder)
    return [
        str(d / n)
        for n in sorted(
            p.name for p in d.iterdir() if p.name.endswith(".dups") and not p.name.startswith(".")
        )
    ]


@dataclass(frozen=True)
class BucketUnit:
    """One ``MinhashDedupBuckets`` task: a bucket, or one hash range of it."""

    bucket: int
    worker: int
    workers_per_bucket: int

    @property
    def key(self) -> str:
        return f"{self.bucket:05d}_{self.worker:02d}"


def bucket_units(num_buckets: int, index: int, count: int) -> list[BucketUnit]:
    """The bucket tasks of shard ``index/count``.

    Up to ``num_buckets`` shards split the buckets (``index, index + count, ...``); a multiple
    of ``num_buckets`` splits each bucket into ``count // num_buckets`` hash ranges, as
    datatrove does with that many tasks.
    """
    if count <= num_buckets:
        return [BucketUnit(b, 0, 1) for b in range(index, num_buckets, count)]
    if count % num_buckets:
        raise ValueError(
            f"--shard {index}/{count}: with {num_buckets} buckets, use at most {num_buckets} "
            f"shards or a multiple of {num_buckets}"
        )
    wpb = count // num_buckets
    b, w = divmod(index, wpb)
    return [BucketUnit(b, w, wpb)]


def run_bucket(
    sigs: str | os.PathLike[str],
    out: str | os.PathLike[str],
    unit: BucketUnit,
    cfg: Mapping[str, Any],
    *,
    max_open: int,
    memory: int = MERGE_BUFFER,
) -> int:
    """Stage 2 for one unit; returns the pairs written."""
    native = native_config(cfg)
    files = sig_files(sigs, unit.bucket)
    first = files[0] if files else None
    lo, hi = _native._minhash_hash_range(first, native, unit.worker, unit.workers_per_bucket)
    dest = Path(out) / dups_name(unit.bucket, unit.worker)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pairs: int = _native._minhash_buckets(
        files, str(dest), native, hash_min=lo, hash_max=hi, memory_budget=memory, max_open=max_open
    )
    return pairs


def open_files_limit(concurrent: int) -> int:
    """How many files one bucket merge may open, after raising the soft limit if allowed."""
    try:
        import resource
    except ImportError:  # Windows: the C runtime's default is 512 stdio streams, far more handles
        return 4096
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard == resource.RLIM_INFINITY or hard > soft:
        target = hard if hard != resource.RLIM_INFINITY else max(soft, 65536)
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            soft = target
        except (ValueError, OSError):
            pass
    return max(16, (soft - 64) // max(1, concurrent))


# ----------------------------------------------------------------------------------------- filter


def _read_u32(path: Path) -> Any:
    import numpy as np

    if not path.exists():
        return np.empty(0, dtype="<u4")
    data = path.read_bytes()
    if len(data) % 4:
        raise ValueError(f"{path}: {len(data)} bytes is not a whole number of 4-byte records")
    return np.frombuffer(data, dtype="<u4")


def _read_pairs(path: Path) -> tuple[Any, Any]:
    import numpy as np

    data = path.read_bytes()
    if len(data) % 8:
        raise ValueError(f"{path}: {len(data)} bytes is not a whole number of 8-byte records")
    a = np.frombuffer(data, dtype="<u4").reshape(-1, 2)
    return a[:, 0], a[:, 1]


class Removal:
    """Stage 4 for one rank: which rows go, and optionally their cluster ids and sizes."""

    def __init__(
        self,
        folder: str | os.PathLike[str],
        rank: int,
        *,
        cluster_ids: bool = False,
        cluster_sizes: bool = False,
    ) -> None:
        d = Path(folder)
        self.path = d / f"{rank:06d}.remove"
        self.removed = _read_u32(self.path)
        self._ids = self._sizes = None
        if cluster_ids:
            p = d / f"{rank:06d}.clusters"
            if not p.exists() and self.path.exists():
                raise FileNotFoundError(
                    f"{p}: missing (run the cluster stage with --save-cluster-id)"
                )
            self._ids = _read_pairs(p) if p.exists() else None
        if cluster_sizes:
            p = d / f"{rank:06d}.sizes"
            if not p.exists() and self.path.exists():
                raise FileNotFoundError(
                    f"{p}: missing (run the cluster stage with --save-cluster-size)"
                )
            self._sizes = _read_pairs(p) if p.exists() else None
        self.cluster_ids = cluster_ids
        self.cluster_sizes = cluster_sizes

    def mask(self, start: int, n: int) -> Any:
        """Keep mask (numpy bool) for rows ``start .. start + n``."""
        import numpy as np

        r = self.removed
        lo, hi = np.searchsorted(r, [start, start + n])
        keep = np.ones(n, dtype=bool)
        keep[(r[lo:hi] - start).astype(np.intp)] = False
        return keep

    def _lookup(self, pairs: tuple[Any, Any] | None, start: int, n: int, default: int) -> Any:
        import numpy as np

        out = np.full(n, default, dtype=np.int64)
        if pairs is None:
            return out
        docs, vals = pairs
        lo, hi = np.searchsorted(docs, [start, start + n])
        out[(docs[lo:hi] - start).astype(np.intp)] = vals[lo:hi]
        return out

    def ids(self, start: int, n: int) -> Any:
        """Datatrove's ``minhash_cluster_id`` for the rows (-1: in no cluster)."""
        return self._lookup(self._ids, start, n, -1)

    def sizes(self, start: int, n: int) -> Any:
        """Datatrove's ``minhash_cluster_size`` for the rows (1: in no cluster)."""
        return self._lookup(self._sizes, start, n, 1)

    def check(self, rows: int) -> None:
        """Raise if the removal list names a row past the end of the file."""
        if len(self.removed) and int(self.removed[-1]) >= rows:
            raise ValueError(
                f"{self.path} removes document {int(self.removed[-1])}, but the input has only "
                f"{rows} rows: the input is not the one the signatures were computed from"
            )


def iter_kept(tables: Iterator[Any], removal: Removal) -> Iterator[tuple[Any, Any, Any]]:
    """``(table, keep mask, start row)`` for each table, with the cluster columns added."""
    import pyarrow as pa

    start = 0
    for t in tables:
        n = t.num_rows
        keep = removal.mask(start, n)
        if removal.cluster_ids:
            t = _io.set_column(t, "minhash_cluster_id", pa.array(removal.ids(start, n)))
        if removal.cluster_sizes:
            t = _io.set_column(t, "minhash_cluster_size", pa.array(removal.sizes(start, n)))
        yield t, pa.array(keep, pa.bool_()), start
        start += n
    removal.check(start)
