#!/usr/bin/env python3
"""Fetch the PGO training inputs into one folder: pinned, and disjoint from every slice that is timed or gated.

    cd bench && uv run --frozen --no-default-groups python pgo_corpus.py --out ../pgo/corpus

Writes, into ``--out``:

    fineweb.jsonl          FineWeb headline rows 10000:20000 (``id``, ``url``, ``text``). The benchmarks time
                           rows 0:10000 (the dev slice) and the parity suites gate them.
    fineweb2-<lang>.jsonl  rows 250:500 of each FineWeb-2 test split in fetch.FINEWEB2. The pinned slices,
                           which multilingual.py, jieba_bench.py and the parity suites use, are rows 0:250.
    lid.176.bin, lid.176.ftz   the fastText models, checked against bench/manifest.json.
    MANIFEST.json          the sha256 and size of each file above, and where its rows came from.

The FineWeb rows come from ``$PYROID_DATA`` when the headline slice is there, else from HTTP range reads of
the pinned Hugging Face revisions (as fetch.py does); the rows before each training slice are checked
against the manifest's content hashes, so the slices are the documented ones. JSONL keeps the trainer
(pgo_train.py) free of pyarrow, which has no Windows ARM64 wheels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from pyroid_parity import data as D

from fetch import FINEWEB, FINEWEB2, MODELS, download, log, range_read

FINEWEB_ROWS = (10_000, 20_000)
FINEWEB2_ROWS = (FINEWEB2["rows"], 2 * FINEWEB2["rows"])


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_jsonl(out: Path, columns: dict[str, list], start: int, stop: int) -> int:
    names = list(columns)
    n = 0
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for i in range(start, stop):
            f.write(json.dumps({k: columns[k][i] for k in names}, ensure_ascii=False) + "\n")
            n += 1
    return n


def check_prefix(rel: str, ids: list[str], texts: list[str]) -> None:
    """The rows before a training slice must be exactly the pinned slice ``rel``."""
    entry = D.load_manifest()["files"][rel]
    n = entry["rows"]
    if D.content_sha256(ids[:n], texts[:n]) != entry["content_sha256"]:
        raise SystemExit(f"{rel}: the first {n} rows read do not match the manifest's content sha256")


def fineweb(out: Path) -> dict:
    lo, hi = FINEWEB_ROWS
    local = D.DATA / D.CORPORA["headline"]
    if local.exists():
        c = D.corpus("headline")  # verifies the content sha256 of all 100k rows
        import pyarrow.parquet as pq

        urls = pq.read_table(local, columns=["url"]).column("url").to_pylist()
        cols = {"id": c.ids, "url": urls, "text": c.texts}
        source = f"{local} (pinned headline slice)"
    else:
        log(f"FineWeb: range-reading the first {hi} rows")
        t = range_read(FINEWEB["repo"], FINEWEB["revision"], FINEWEB["file"], hi)
        cols = {k: t.column(k).to_pylist() for k in ("id", "url", "text")}
        check_prefix(D.CORPORA["dev"], cols["id"], cols["text"])
        source = f"{FINEWEB['repo']}@{FINEWEB['revision']}/{FINEWEB['file']}"
    n = write_jsonl(out / "fineweb.jsonl", cols, lo, hi)
    return {"rows": f"{lo}:{hi}", "docs": n, "source": source, "license": FINEWEB["license"]}


def fineweb2(out: Path, lang: str) -> dict:
    lo, hi = FINEWEB2_ROWS
    file = f"data/{lang}/test/000_00000.parquet"
    log(f"FineWeb-2 {lang}: range-reading the first {hi} rows")
    t = range_read(FINEWEB2["repo"], FINEWEB2["revision"], file, hi)
    cols = {k: t.column(k).to_pylist() for k in ("id", "text")}
    check_prefix(f"fineweb2/{lang}.parquet", cols["id"], cols["text"])
    n = write_jsonl(out / f"fineweb2-{lang}.jsonl", cols, lo, min(hi, len(cols["id"])))
    source = f"{FINEWEB2['repo']}@{FINEWEB2['revision']}/{file}"
    return {"rows": f"{lo}:{lo + n}", "docs": n, "source": source, "license": FINEWEB2["license"]}


def model(out: Path, rel: str) -> dict:
    want = D.load_manifest()["files"][rel]["sha256"]
    dest = out / Path(rel).name
    local = D.DATA / rel
    if local.exists() and sha256(local) == want:
        shutil.copyfile(local, dest)
    else:
        log(f"downloading {MODELS[rel]}")
        download(MODELS[rel], dest)
    if sha256(dest) != want:
        raise SystemExit(f"{dest.name}: sha256 differs from bench/manifest.json")
    return {"source": MODELS[rel], "license": "CC BY-SA 3.0"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out: Path = a.out
    out.mkdir(parents=True, exist_ok=True)
    meta = {"fineweb.jsonl": fineweb(out)}
    for lang in FINEWEB2["langs"]:
        meta[f"fineweb2-{lang}.jsonl"] = fineweb2(out, lang)
    for rel in MODELS:
        meta[Path(rel).name] = model(out, rel)
    for name, m in meta.items():
        p = out / name
        m.update(bytes=p.stat().st_size, sha256=sha256(p))
        log(f"{name}: {m['bytes']:,} bytes, sha256 {m['sha256'][:16]}")
    (out / "MANIFEST.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
