"""Pinned datasets: manifest lookup, sha256 verification and loaders.

Every file the harness reads lives under ``$PYROID_DATA`` (default ``/data``, the shared Docker volume) and is
pinned in the committed ``bench/manifest.json``. Loaders refuse to return data whose sha256 does not match.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
MANIFEST = Path(os.environ.get("PYROID_MANIFEST", REPO / "bench" / "manifest.json"))
DATA = Path(os.environ.get("PYROID_DATA", "/data"))


class DataError(RuntimeError):
    """A pinned file is missing, or its bytes do not match the manifest."""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def content_sha256(ids: Iterable[str], texts: Iterable[str]) -> str:
    """R2 §5 content hash: sha256 over ``id\\0text\\0`` for each doc, in order. Format-independent."""
    h = hashlib.sha256()
    for i, t in zip(ids, texts, strict=True):
        h.update(i.encode("utf-8", "surrogatepass"))
        h.update(b"\0")
        h.update(t.encode("utf-8", "surrogatepass"))
        h.update(b"\0")
    return h.hexdigest()


def load_manifest() -> dict[str, Any]:
    if not MANIFEST.exists():
        return {"schema": 1, "files": {}}
    return json.loads(MANIFEST.read_text("utf-8"))


def save_manifest(manifest: dict[str, Any]) -> None:
    manifest["files"] = dict(sorted(manifest["files"].items()))
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", "utf-8")


def pin(rel: str, **extra: Any) -> dict[str, Any]:
    """Record ``rel``'s sha256 and size (plus ``extra``) in the manifest; returns the entry."""
    path = DATA / rel
    manifest = load_manifest()
    entry = {"sha256": sha256_file(path), "bytes": path.stat().st_size, **extra}
    manifest["files"][rel] = entry
    save_manifest(manifest)
    return entry


@cache
def _verified(rel: str) -> Path:
    entry = load_manifest()["files"].get(rel)
    path = DATA / rel
    if entry is None:
        raise DataError(f"{rel} is not pinned in {MANIFEST}")
    if not path.exists():
        raise DataError(
            f"{path} is missing: run `bench/fetch.py` (and `python -m pyroid_parity.refs`) in Docker"
        )
    if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
        raise DataError(f"{path} does not match the sha256 pinned in {MANIFEST}")
    return path


def path(rel: str, verify: bool = True) -> Path:
    """Absolute path of a pinned file, verified against the manifest (cached per process)."""
    return _verified(rel) if verify else DATA / rel


def available(rel: str) -> bool:
    try:
        path(rel)
    except DataError:
        return False
    return True


@dataclass(frozen=True)
class Corpus:
    """An ordered document set. ``ids`` are unique; ``meta`` holds per-doc provenance (e.g. adversarial kind)."""

    name: str
    ids: list[str]
    texts: list[str]
    meta: list[str]

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return zip(self.ids, self.texts, strict=True)

    def head(self, n: int) -> Corpus:
        return Corpus(f"{self.name}[:{n}]", self.ids[:n], self.texts[:n], self.meta[:n])


def read_parquet_columns(p: Path, columns: list[str]) -> dict[str, list[Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(p, columns=columns)
    return {c: table.column(c).to_pylist() for c in columns}


# Logical corpora -> pinned files. "dev" is the 10k-doc FineWeb slice, "headline" the 100k slice (a superset),
# "adversarial" the generated mojibake / non-English / near-duplicate gate set (see pyroid_parity.refs).
CORPORA = {
    "dev": "fineweb/fineweb-dev-10k.parquet",
    "headline": "fineweb/fineweb-100k.parquet",
    "adversarial": "parity/adversarial.parquet",
}


@cache
def corpus(name: str) -> Corpus:
    rel = CORPORA[name]
    p = path(rel)
    import pyarrow.parquet as pq

    cols = pq.read_schema(p).names
    data = read_parquet_columns(p, ["id", "text"] + (["kind"] if "kind" in cols else []))
    meta = data.get("kind") or [name] * len(data["id"])
    entry = load_manifest()["files"][rel]
    if "content_sha256" in entry and content_sha256(data["id"], data["text"]) != entry["content_sha256"]:
        raise DataError(f"{rel}: content sha256 differs from the manifest")
    return Corpus(name, data["id"], data["text"], meta)
