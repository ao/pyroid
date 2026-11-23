#!/usr/bin/env python3
"""Rebuild the pyroid.yaml real-file corpus from its manifest and verify every sha256.

    python -I bench/yaml/fetch_corpus.py /target/data/corpus
    python -I bench/yaml/fetch_corpus.py DEST --family gha --family k8s     # some families only
    python -I bench/yaml/fetch_corpus.py DEST --verify                      # no network: check DEST only
    python -I bench/yaml/fetch_corpus.py DEST --no-synthetic --jobs 4

Stdlib only. Reads ``parity/data/yaml-corpus/manifest.json`` (``--manifest`` to override). Each file
entry has a raw URL at a pinned commit; the downloaded bytes must match the recorded sha256 and size.
Hugging Face card entries (``"extract": "front-matter"``) store only the YAML front matter of the
README at the pinned revision (:func:`front_matter`). The synthetic entries are rebuilt from corpus
files (:func:`build_synthetic`), so their parts are fetched first even under ``--family synthetic``.

Files already present with the right sha256 are not downloaded again. HTTP 429/403/5xx answers are
retried with backoff (honouring ``Retry-After`` and GitHub's ``X-RateLimit-Reset``). Exits 1 if any
file is missing or does not match after the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "parity" / "data" / "yaml-corpus" / "manifest.json"
USER_AGENT = "pyroid-yaml-corpus (https://github.com/ao/pyroid)"


def front_matter(data: bytes) -> bytes | None:
    """Return the YAML front matter of a Markdown file, or None if it has none.

    The first line must be ``---`` (an optional trailing CR is ignored); the block ends at the next
    line that is ``---``. The lines in between are returned joined with ``\\n`` plus a final ``\\n``,
    so a card written with CRLF keeps its CRs at the line ends.
    """
    lines = data.split(b"\n")
    if not lines or lines[0].rstrip(b"\r") != b"---":
        return None
    for i in range(1, len(lines)):
        if lines[i].rstrip(b"\r") == b"---":
            return b"\n".join(lines[1:i]) + b"\n"
    return None


def synthetic_part(data: bytes) -> bytes:
    """Return one corpus file as a part of a synthetic multi-document stream.

    A part gets a ``---`` line in front unless its first line that is neither blank nor a comment
    already starts a document, and a final newline if it lacks one.
    """
    if not data.endswith(b"\n"):
        data += b"\n"
    for line in data.split(b"\n"):
        s = line.strip()
        if not s or s.startswith(b"#"):
            continue
        if line.startswith(b"---") and (len(line) == 3 or line[3:4] in b" \t\r"):
            return data
        break
    return b"---\n" + data


def build_synthetic(root: Path, parts: list[str], target_bytes: int) -> bytes:
    """Concatenate ``parts`` (corpus paths) in order, cycling, until ``target_bytes`` is reached."""
    chunks = [synthetic_part((root / p).read_bytes()) for p in parts]
    out = bytearray()
    i = 0
    while len(out) < target_bytes:
        out += chunks[i % len(chunks)]
        i += 1
    return bytes(out)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str, tries: int = 8) -> bytes:
    delay = 2.0
    for attempt in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code not in (403, 408, 429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
            wait = delay
            if e.headers.get("Retry-After", "").isdigit():
                wait = max(wait, float(e.headers["Retry-After"]))
            elif e.headers.get("X-RateLimit-Remaining") == "0" and e.headers.get("X-RateLimit-Reset"):
                wait = max(wait, float(e.headers["X-RateLimit-Reset"]) - time.time() + 1)
            time.sleep(min(wait, 900))
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == tries - 1:
                raise
            time.sleep(delay)
        delay = min(delay * 2, 120)
    raise RuntimeError("unreachable")


def fetch_one(dest: Path, entry: dict, verify_only: bool) -> str | None:
    """Fetch one manifest file entry into ``dest``; return an error message or None."""
    path = dest / entry["path"]
    if path.is_file():
        data = path.read_bytes()
        if sha256(data) == entry["sha256"]:
            return None
        if verify_only:
            return f"{entry['path']}: sha256 mismatch"
    elif verify_only:
        return f"{entry['path']}: missing"
    try:
        data = download(entry["url"])
    except Exception as e:  # noqa: BLE001 - reported; the other files go on
        return f"{entry['path']}: {type(e).__name__}: {e}"
    if entry.get("extract") == "front-matter":
        fm = front_matter(data)
        if fm is None:
            return f"{entry['path']}: no front matter in {entry['url']}"
        data = fm
    if len(data) != entry["bytes"] or sha256(data) != entry["sha256"]:
        return f"{entry['path']}: got {len(data)} bytes, sha256 {sha256(data)}, expected {entry['sha256']}"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dest", type=Path, help="corpus root to fill")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--family", action="append", help="only these families (repeatable)")
    ap.add_argument("--jobs", type=int, default=8, help="parallel downloads (default 8)")
    ap.add_argument("--verify", action="store_true", help="check DEST against the manifest, no network")
    ap.add_argument("--no-synthetic", action="store_true", help="skip the synthetic large files")
    a = ap.parse_args()

    man = json.loads(a.manifest.read_text())
    families = set(a.family) if a.family else None
    synth = [] if a.no_synthetic else man.get("synthetic", [])
    if families is not None:
        synth = [s for s in synth if "synthetic" in families]
    needed = {p for s in synth for p in s["parts"]}
    files = [f for f in man["files"] if families is None or f["family"] in families or f["path"] in needed]

    errors: list[str] = []
    done = 0
    with ThreadPoolExecutor(max(1, a.jobs)) as pool:
        for err in pool.map(lambda f: fetch_one(a.dest, f, a.verify), files):
            done += 1
            if err:
                errors.append(err)
                print(err, file=sys.stderr)
            if done % 500 == 0:
                print(f"{done}/{len(files)} files", file=sys.stderr)

    for s in synth:
        path = a.dest / s["path"]
        if path.is_file() and path.stat().st_size == s["bytes"] and sha256(path.read_bytes()) == s["sha256"]:
            continue
        if a.verify:
            errors.append(f"{s['path']}: missing or sha256 mismatch")
            continue
        try:
            data = build_synthetic(a.dest, s["parts"], s["target_bytes"])
        except OSError as e:
            errors.append(f"{s['path']}: {e}")
            continue
        if sha256(data) != s["sha256"]:
            errors.append(f"{s['path']}: rebuilt sha256 {sha256(data)} != {s['sha256']}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    total = len(files) + len(synth)
    print(f"{total - len(errors)}/{total} files verified in {a.dest}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
