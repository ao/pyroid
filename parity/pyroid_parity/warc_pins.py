"""The warc suite's jobs and pinned reference digests (``parity/data/warcio-1.8.2/ref-digests.json``).

    python -m pyroid_parity.warc_pins write    # run warcio 1.8.2 + datatrove 0.10.1 over every job, write the pins
    python -m pyroid_parity.warc_pins check    # recompute and compare (exit 1 on any difference)

A *job* is one input file under one ``warc_compare.Config``. The jobs are grouped: ``cc`` (the pinned Common
Crawl slices, ``CC_CONFIGS``), ``warcio-tests`` (warcio's test files, ``SYNTH_CONFIGS``), ``synth-<family>``
(``warc_synth`` cases, ``SYNTH_CONFIGS``) and ``derived`` (plain / recompressed variants of the CC slices). The
suite and this module walk the same jobs.

What is pinned, per job of ``cc`` and ``warcio-tests``: the sha256 (first 12 hex digits) of the trace's main
events and of its ``io`` events, for the first ``HEAD`` records (the default run) and for all of them. For the
other groups one digest pair per (group, config), over the group's inputs in order. The datatrove level pins
``reference_documents`` per input the same way. The pins hold the versions they were made with (CPython minor,
zlib, brotlicffi, libmagic, cchardet); a check whose inputs depend on a version that differs here skips.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib.metadata as md
import importlib.util
import json
import shutil
import sys
import tempfile
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from pyroid_parity import warc_compare as C
from pyroid_parity import warc_corpus as W
from pyroid_parity import warc_datatrove as T
from pyroid_parity import warc_synth as S

PINS = Path(__file__).resolve().parents[1] / "data" / f"warcio-{W.WARCIO_VERSION}" / "ref-digests.json"
HEAD = 100
DIGITS = 12
# Derived variants of the CC slices (plain, one stream, pairs, level 1) and what runs on them.
DERIVED_CONFIGS = tuple(
    c
    for c in C.CC_CONFIGS
    if c.name in ("content/file", "random/file", "content/gzipfile", "offsets/bytesio")
)


@dataclass(frozen=True)
class Job:
    group: str
    name: str  # input name
    path: Path
    gz: bool
    config: C.Config


# The "datatrove" opener reads through datatrove's DataFolder.open: without datatrove (the free-threaded legs) those
# configs are left out.
HAVE_DATATROVE = importlib.util.find_spec("datatrove") is not None


def configs_for(gz: bool, configs: tuple[C.Config, ...], big: bool = False) -> list[C.Config]:
    return [
        c
        for c in configs
        if (gz or not c.needs_gzip)
        and not (big and c.name in C.SLOW)
        and (HAVE_DATATROVE or c.opener != "datatrove")
    ]


def env() -> dict[str, Any]:
    def v(dist: str) -> str | None:
        try:
            return md.version(dist)
        except md.PackageNotFoundError:
            return None

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "zlib": zlib.ZLIB_RUNTIME_VERSION,
        "brotlicffi": v("brotlicffi"),
        "libmagic": T.ensure_magic(),
        "cchardet": v("faust-cchardet"),
        "warcio": v("warcio"),
        "datatrove": v("datatrove"),
    }


@cache
def synth_dir() -> Path:
    """Every synthetic case written once per process (a temporary directory)."""
    d = Path(tempfile.mkdtemp(prefix="pyroid-warc-synth-"))
    atexit.register(shutil.rmtree, d, True)
    for c in S.cases():
        (d / c.filename).write_bytes(c.data)
    return d


@cache
def derived_dir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="pyroid-warc-derived-"))
    atexit.register(shutil.rmtree, d, True)
    return d


def families() -> list[str]:
    return list(dict.fromkeys(c.name.split("/")[0] for c in S.cases()))


def synth_inputs(family: str) -> list[tuple[S.Case, Path]]:
    d = synth_dir()
    return [(c, d / c.filename) for c in S.cases() if c.name.split("/")[0] == family]


def derived_inputs() -> list[W.Input]:
    out = []
    for inp in W.pinned_inputs("warc/cc/"):
        out += W.derived(inp, derived_dir())
    return out


def jobs(group: str, other_digests: bool = False) -> Iterator[Job]:
    """The jobs of a group. Synthetic cases tagged ``digest-other`` (algorithms hashlib knows and pyroid does not
    port) are left out unless ``other_digests``: the suite checks them on their own."""
    if group == "cc":
        for inp in W.pinned_inputs("warc/cc/"):
            for c in configs_for(inp.gz, C.CC_CONFIGS):
                yield Job(group, inp.name, inp.path, inp.gz, c)
    elif group == "warcio-tests":
        for inp in W.pinned_inputs(W.WARCIO_DIR):
            for c in configs_for(inp.gz, C.SYNTH_CONFIGS):
                yield Job(group, inp.name, inp.path, inp.gz, c)
    elif group == "derived":
        for inp in derived_inputs():
            for c in configs_for(inp.gz, DERIVED_CONFIGS):
                yield Job(group, inp.name, inp.path, inp.gz, c)
    elif group.startswith("synth-"):
        for case, path in synth_inputs(group.removeprefix("synth-")):
            if ("digest-other" in case.tags) != other_digests:
                continue
            for c in configs_for(case.gz, C.SYNTH_CONFIGS, big="big" in case.tags):
                yield Job(group, case.name, path, case.gz, c)
    else:
        raise ValueError(group)


def groups() -> list[str]:
    return ["cc", "warcio-tests", "derived", *(f"synth-{f}" for f in families())]


def per_input(group: str) -> bool:
    return group in ("cc", "warcio-tests")


def limited(group: str) -> bool:
    """Groups of whole Common Crawl slices: the default run reads the first ``HEAD`` records of each."""
    return group in ("cc", "derived")


def short(events: list[C.Event]) -> str:
    return C.digest(events)[:DIGITS]


def pair(trace: list[C.Event]) -> str:
    main, io_ = C.split(trace)
    return f"{short(main)},{short(io_)}"


def fold(parts: list[str]) -> str:
    """One digest pair for a sequence of per-input pairs."""
    m = hashlib.sha256("\n".join(p.split(",")[0] for p in parts).encode()).hexdigest()[:DIGITS]
    i = hashlib.sha256("\n".join(p.split(",")[1] for p in parts).encode()).hexdigest()[:DIGITS]
    return f"{m},{i}"


def pack(entry: dict[str, str]) -> str:
    """{"all": "m,i", "head": "m,i"} -> ``"all:m,i,head:m,i"``."""
    return ",".join(f"{k}:{v}" for k, v in sorted(entry.items()))


def record_pins(mod: Any, exc_name: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for g in groups():
        limits = {"all": 0, "head": HEAD} if limited(g) else {"all": 0}
        if per_input(g):
            got: dict[str, dict[str, str]] = {}
            for j in jobs(g):
                entry = {
                    k: pair(C.iter_events(mod, j.path, j.config, n, exc_name)) for k, n in limits.items()
                }
                got.setdefault(j.name, {})[j.config.name] = pack(entry)
            out[g] = got
        else:
            parts: dict[str, dict[str, list[str]]] = {}
            for j in jobs(g):
                for k, n in limits.items():
                    trace = C.iter_events(mod, j.path, j.config, n, exc_name)
                    parts.setdefault(j.config.name, {}).setdefault(k, []).append(pair(trace))
            out[g] = {name: pack({k: fold(v) for k, v in by.items()}) for name, by in parts.items()}
    return out


def unpack(entry: str) -> dict[str, str]:
    """``"all:m,i,head:m,i"`` -> {"all": "m,i", "head": "m,i"}."""
    bits = entry.split(",")
    out = {}
    for k in range(0, len(bits), 2):
        name, m = bits[k].split(":")
        out[name] = f"{m},{bits[k + 1]}"
    return out


def pin_key(group: str, limit: int) -> str | None:
    """Which pinned digest a run with ``limit`` records per slice compares with (None: neither)."""
    if not limited(group) or limit == 0:
        return "all"
    return "head" if limit == HEAD else None


def document_inputs(group: str) -> list[tuple[str, Path]]:
    if group == "cc":
        return [(i.name, i.path) for i in W.pinned_inputs("warc/cc/")]
    if group == "warcio-tests":
        return [(i.name, i.path) for i in W.pinned_inputs(W.WARCIO_DIR)]
    if group == "derived":
        return [(i.name, i.path) for i in derived_inputs()]
    return [(c.name, p) for c, p in synth_inputs(group.removeprefix("synth-"))]


def document_digest(exc_name: Any, path: Path, limit: int = 0) -> str:
    return short(T.events_of(lambda: T.reference_documents(path), T.doc_view, exc_name, limit))


def document_pins(exc_name: Any) -> dict[str, Any]:
    """Per input (``cc``, ``warcio-tests``) or per group: ``"<all>,<head>"`` (``"<all>"`` for unlimited groups)."""
    out: dict[str, Any] = {}
    for g in groups():
        heads = per_input(g) or limited(g)
        if per_input(g):
            out[g] = {
                name: f"{document_digest(exc_name, path)},{document_digest(exc_name, path, HEAD)}"
                for name, path in document_inputs(g)
            }
            continue
        inputs = document_inputs(g)
        full = fold_one([document_digest(exc_name, p) for _, p in inputs])
        out[g] = (
            f"{full},{fold_one([document_digest(exc_name, p, HEAD) for _, p in inputs])}" if heads else full
        )
    return out


def fold_one(parts: list[str]) -> str:
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:DIGITS]


def document_pin(entry: str, key: str) -> str | None:
    """The ``all`` or ``head`` digest of a document pin."""
    parts = entry.split(",")
    return parts[0] if key == "all" else (parts[1] if len(parts) > 1 else None)


def compute() -> dict[str, Any]:
    from pyroid_parity.adapters import warc_warcio

    ref = warc_warcio.make()
    e = env()
    return {
        "about": "warcio 1.8.2 / datatrove 0.10.1 trace digests; python -m pyroid_parity.warc_pins write",
        "env": e,
        "head": HEAD,
        "synth_sha256": S.inputs_sha256(S.cases()),
        "inputs": sorted(i.name for i in W.pinned_inputs()),
        "records": record_pins(ref.mod, ref.exc_name),
        "documents": document_pins(ref.exc_name),
    }


@cache
def load() -> dict[str, Any] | None:
    if not PINS.exists():
        return None
    return json.loads(PINS.read_text("utf-8"))


def mismatch(pins: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """Why the pins do not apply here (an input version in ``keys`` differs), or None."""
    here = env()
    for k in keys:
        if pins["env"].get(k) != here.get(k):
            return f"the pins were made with {k} {pins['env'].get(k)}, this interpreter has {here.get(k)}"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pyroid_parity.warc_pins")
    ap.add_argument("cmd", choices=("write", "check"))
    a = ap.parse_args(argv)
    if W.missing():
        print(f"pinned inputs missing: {W.missing()[:5]} ... (bench/fetch.py --only warc)", file=sys.stderr)
        return 2
    got = compute()
    if a.cmd == "write":
        PINS.parent.mkdir(parents=True, exist_ok=True)
        PINS.write_text(json.dumps(got, indent=1, ensure_ascii=False, sort_keys=False) + "\n", "utf-8")
        print(f"wrote {PINS}")
        return 0
    want = load()
    if want is None:
        print(f"{PINS} is missing", file=sys.stderr)
        return 1
    bad = 0
    for part in ("records", "documents"):
        for g, entries in want[part].items():
            if got[part].get(g) != entries:
                bad += 1
                print(f"{part}/{g}: differs")
    print("same" if not bad else f"{bad} groups differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
