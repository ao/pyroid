"""Record Trafilatura 2.3.1's own test suite as golden public calls (``extract`` and ``bare_extraction``).

    python -m pyroid_parity.refs trafilatura_upstream [--pin]   # record, write, check against the manifest

The output goes to ``$PYROID_DATA`` (``parity/trafilatura-upstream.parquet`` and its blob table
``parity/trafilatura-upstream-blobs.parquet``), not to the repository: the test pages under ``tests/cache/`` are
third-party web pages. The upstream tests themselves are fetched by ``bench/fetch.py --only trafilatura`` from
GitHub at the commit tagged v2.3.1 (``trafilatura/upstream-tests-2.3.1.parquet``: path, git blob sha1, bytes).

How it works:

1. The upstream test files are written to a temporary ``tests/`` package and run by pytest in a subprocess with
   this module as a plugin. Before any test module is imported the plugin replaces ``extract`` and
   ``bare_extraction`` in ``trafilatura`` and ``trafilatura.core`` with recording wrappers.
2. Only outermost calls are recorded (``extract`` calls ``bare_extraction``; ``extract_with_metadata`` is
   wrapped so that its inner call is not recorded either): every argument as given, encoded with
   ``trafilatura_compare.encode_arg`` before the call (so an ``LRUCache`` is stored with its state at that point),
   the pytest node id, and the outcome the test observed. Calls whose arguments cannot be stored are dropped and
   counted.
3. The golden outcome is a clean replay, not what the test observed: in a fresh subprocess with
   ``PYTHONHASHSEED=0``, each test's calls run in their recorded order after ``trafilatura.meta.reset_caches()``
   (the module-level dedup cache is part of the state, so a test is the unit of replay). Tag
   ``differs-from-test`` marks calls whose clean outcome is not what the test saw (tests that swap
   ``LRU_TEST``, monkeypatch internals, or depend on calls made by earlier tests).
4. A second replay with ``PYTHONHASHSEED=1`` tags calls whose outcome depends on the hash seed (``hash-seed``).

Not run (``NOT_RUN``): the CLI tests (subprocesses and the CLI's worker pools), downloads (HTTP), feeds, sitemaps
and spider (no extraction calls) and the evaluation harness (it needs ``tests/eval/``, 923 pages, not fetched).
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import inspect
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pyroid_parity import trafilatura_compare as T

TRAFILATURA_VERSION = "2.3.1"
COMMIT = "e2f85c965662c7df26b73290ef175b2c2a5fdca6"  # tag v2.3.1 of https://github.com/adbar/trafilatura
UPSTREAM_TESTS = f"trafilatura/upstream-tests-{TRAFILATURA_VERSION}.parquet"
CALLS = "parity/trafilatura-upstream.parquet"
BLOBS = "parity/trafilatura-upstream-blobs.parquet"
PUBLIC = ("extract", "bare_extraction")
NOT_RUN = {
    "cli_tests.py": "the command-line interface: subprocesses and the CLI's worker pools, not direct calls",
    "downloads_tests.py": "HTTP downloads",
    "feeds_tests.py": "feed discovery; no extract or bare_extraction calls",
    "sitemaps_tests.py": "sitemaps; no extract or bare_extraction calls",
    "spider_tests.py": "crawling; no extract or bare_extraction calls",
    "eval_common_tests.py": "the evaluation harness; it needs tests/eval/ (923 pages, 135 MB), not fetched",
    "eval_gate_tests.py": "the evaluation harness; it needs tests/eval/ (923 pages, 135 MB), not fetched",
}
TAGS = {
    "tree-input": "the input is an lxml tree, rebuilt node by node for the replay",
    "test-file": "an argument names a file of the upstream tests (settingsfile=), rebuilt for the replay",
    "unreplayable": "an argument could not be stored (dropped from the recording)",
    "extractor-subclass": "options= was a subclass of Extractor; replayed as a plain Extractor",
    "differs-from-test": "the clean replay differs from what the test observed (see step 3)",
    "hash-seed": "the outcome changes with PYTHONHASHSEED",
}
SKIP_TAGS = {"hash-seed": TAGS["hash-seed"]}

# ================================================================================ pytest plugin (subprocess side)

_OUT_ENV = "PYROID_TRAF_RECORD_OUT"
_ROOT_ENV = "PYROID_TRAF_RECORD_ROOT"


class _State(threading.local):
    depth = 0


_state = _State()
_records: list[dict[str, Any]] = []
_blobs = T.Blobs()
_current = {"test": "<collection>"}
_tally: Counter[str] = Counter()
_failed: list[str] = []


def _append(name: str, args: list[Any], kwargs: dict[str, Any], tags: set[str], observed: T.Outcome) -> None:
    # The test process cannot record warnings without hiding them from pytest.warns: observed has none.
    _records.append(
        {
            "fn": name,
            "args": args,
            "kwargs": kwargs,
            "tags": sorted(tags),
            "test": _current["test"],
            "observed": T.outcome_json(observed),
        }
    )


def _wrap(name: str, orig: Any, record: bool) -> Any:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _state.depth or not record:
            _state.depth += 1
            try:
                return orig(*args, **kwargs)
            finally:
                _state.depth -= 1
        tags: set[str] = set()
        enc_a = [T.encode_arg(a, _blobs, tags) for a in args]
        enc_k = {k: T.encode_arg(v, _blobs, tags) for k, v in kwargs.items()}
        _state.depth += 1
        try:
            result = orig(*args, **kwargs)
        except Exception as e:
            _append(name, enc_a, enc_k, tags, T.Outcome("raise", (type(e).__name__, str(e))))
            raise
        finally:
            _state.depth -= 1
        _append(name, enc_a, enc_k, tags, T.Outcome("ok", T.normalize(result)))
        return result

    wrapper.__wrapped__ = orig  # type: ignore[attr-defined]
    wrapper.__signature__ = inspect.signature(orig)  # type: ignore[attr-defined]
    wrapper.__name__ = orig.__name__
    wrapper.__doc__ = orig.__doc__
    return wrapper


def _install() -> None:
    import trafilatura
    from trafilatura import core

    for name, record in (("extract", True), ("bare_extraction", True), ("extract_with_metadata", False)):
        orig = getattr(core, name)
        w = _wrap(name, orig, record)
        setattr(core, name, w)
        if getattr(trafilatura, name, None) is orig:
            setattr(trafilatura, name, w)


def pytest_configure(config: Any) -> None:
    if os.environ.get(_OUT_ENV):
        _blobs.root = os.environ[_ROOT_ENV]
        _install()


def pytest_runtest_logstart(nodeid: str, location: Any) -> None:
    _current["test"] = nodeid


def pytest_runtest_logreport(report: Any) -> None:
    if report.when == "call" and report.passed:
        _tally["passed"] += 1
    elif report.failed:
        _tally["failed" if report.when == "call" else "errors"] += 1
        _failed.append(report.nodeid)
    elif report.skipped:
        _tally["xfailed" if hasattr(report, "wasxfail") else "skipped"] += 1


def pytest_sessionfinish(session: Any) -> None:
    out = os.environ.get(_OUT_ENV)
    if not out:
        return
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({"tests": dict(_tally), "failed": _failed}), "utf-8")
    with (d / "calls.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=True) + "\n" for r in _records)
    blob_dir = d / "blobs"
    blob_dir.mkdir(exist_ok=True)
    for key, raw in _blobs.data.items():
        (blob_dir / key).write_bytes(raw)


# ======================================================================================= replay (both sides)


def replay_groups(target: Any, reset: Any, calls: list[dict[str, Any]], blobs: T.Blobs) -> list[T.Outcome]:
    """Outcomes of ``calls`` on ``target``, test by test: ``reset()`` at the start of each test's calls."""
    out: list[T.Outcome] = []
    group = None
    for c in calls:
        if c["test"] != group:
            group = c["test"]
            reset()
        args = [T.decode_arg(a, blobs) for a in c["args"]]
        kwargs = {k: T.decode_arg(v, blobs) for k, v in c["kwargs"].items()}
        out.append(T.capture_call(getattr(target, c["fn"]), *args, **kwargs))
    return out


def _oracle_replay(calls_path: Path, blob_dir: Path, out_path: Path, root: Path) -> None:
    import trafilatura
    from trafilatura.meta import reset_caches

    calls = [json.loads(line) for line in calls_path.read_text("utf-8").splitlines()]
    blobs = T.Blobs(((p.name, p.read_bytes()) for p in blob_dir.iterdir()), root=str(root))
    outs = replay_groups(trafilatura, reset_caches, calls, blobs)
    out_path.write_text(json.dumps([T.outcome_json(o) for o in outs]), "utf-8")


def _replay_subprocess(calls_path: Path, blob_dir: Path, seed: str, tmp: Path) -> list[str]:
    out = tmp / f"outcomes-{seed}.json"
    env = {**os.environ, "PYTHONHASHSEED": seed}
    cmd = [
        sys.executable,
        "-m",
        "pyroid_parity.trafilatura_record",
        "--replay",
        str(calls_path),
        str(blob_dir),
    ]
    subprocess.run([*cmd, str(out), str(tmp)], check=True, env=env)
    return json.loads(out.read_text("utf-8"))


# ===================================================================================== recorder (CLI side)


def materialize_tests(dst: Path) -> int:
    """Write the pinned upstream test files under ``dst`` (``dst/tests/...``); returns the number of files."""
    from pyroid_parity import data as D

    cols = D.read_parquet_columns(D.path(UPSTREAM_TESTS), ["path", "data"])
    for rel, raw in zip(cols["path"], cols["data"], strict=True):
        p = dst / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    return len(cols["path"])


def run_upstream_suite(tmp: Path, pytest_args: list[str]) -> Path:
    materialize_tests(tmp)
    (tmp / "pytest.ini").write_text("[pytest]\npython_files = *_tests.py\npythonpath = .\n", "utf-8")
    raw = tmp / "raw"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-c",
        str(tmp / "pytest.ini"),
        "--rootdir",
        str(tmp),
        "-p",
        "pyroid_parity.trafilatura_record",
        "-p",
        "no:cacheprovider",
        "-q",
        *[f"--ignore=tests/{name}" for name in NOT_RUN],
        *pytest_args,
    ]
    env = {**os.environ, _OUT_ENV: str(raw), _ROOT_ENV: str(tmp), "PYTHONDONTWRITEBYTECODE": "1"}
    env["PYTHONHASHSEED"] = "0"
    proc = subprocess.run(cmd, cwd=tmp, env=env, check=False, stdout=subprocess.DEVNULL)
    if proc.returncode not in (0, 1):
        raise SystemExit(f"the upstream pytest run did not complete (exit {proc.returncode})")
    return raw


def _without_warnings(outcome_json: str) -> str:
    j = json.loads(outcome_json)
    j.pop("warnings", None)
    return json.dumps(j, ensure_ascii=True, separators=(",", ":"))


def record(pytest_args: list[str] | None = None) -> tuple[dict[str, Any], Any, Any]:
    """Run, replay and tag; returns (header, calls table, blobs table) as pyarrow tables."""
    import pyarrow as pa

    installed = md.version("trafilatura")
    if installed != TRAFILATURA_VERSION:
        raise SystemExit(f"needs trafilatura=={TRAFILATURA_VERSION}, found {installed}")
    with tempfile.TemporaryDirectory(prefix="pyroid-traf-record-") as d:
        tmp = Path(d)
        raw = run_upstream_suite(tmp, pytest_args or [])
        summary = json.loads((raw / "summary.json").read_text("utf-8"))
        recorded = [json.loads(line) for line in (raw / "calls.jsonl").read_text("utf-8").splitlines()]
        dropped = sum(1 for c in recorded if "unreplayable" in c["tags"])
        calls = [c for c in recorded if "unreplayable" not in c["tags"]]
        kept_path = tmp / "kept.jsonl"
        kept_path.write_text("".join(json.dumps(c, ensure_ascii=True) + "\n" for c in calls), "utf-8")
        with ThreadPoolExecutor(2) as ex:  # two independent processes
            golden, seed1 = ex.map(lambda s: _replay_subprocess(kept_path, raw / "blobs", s, tmp), ("0", "1"))
        used: set[str] = set()

        def walk(x: Any) -> None:  # iterative: recorded trees can be nested thousands of levels deep
            stack = [x]
            while stack:
                y = stack.pop()
                if isinstance(y, dict):
                    if set(y) == {"$blob"}:
                        used.add(y["$blob"][0])
                    stack.extend(y.values())
                elif isinstance(y, list):
                    stack.extend(y)

        for c, want, probe in zip(calls, golden, seed1, strict=True):
            tags = set(c["tags"])
            if probe != want:
                tags.add("hash-seed")
            if c["observed"] != _without_warnings(want):
                tags.add("differs-from-test")
            c["tags"], c["outcome"] = sorted(tags), want
            walk(c["args"])
            walk(c["kwargs"])
        blob_keys = sorted(used)
        blobs = pa.table(
            {"sha256": blob_keys, "data": [(raw / "blobs" / k).read_bytes() for k in blob_keys]},
            schema=pa.schema([("sha256", pa.string()), ("data", pa.binary())]),
        )
    table = pa.table(
        {
            "index": list(range(len(calls))),
            "test": [c["test"] for c in calls],
            "fn": [c["fn"] for c in calls],
            "args": [json.dumps(c["args"], ensure_ascii=True) for c in calls],
            "kwargs": [json.dumps(c["kwargs"], ensure_ascii=True) for c in calls],
            "tags": [c["tags"] for c in calls],
            "outcome": [c["outcome"] for c in calls],
        }
    )
    tag_counts = Counter(t for c in calls for t in c["tags"])
    header = {
        "source": f"trafilatura v{TRAFILATURA_VERSION} tests/ at {COMMIT}",
        "python": platform.python_version(),
        "upstream_tests": {**summary["tests"], "failed_ids": summary["failed"], "not_run": NOT_RUN},
        "calls": len(calls),
        "dropped_unreplayable": dropped,
        "tests_with_calls": len({c["test"] for c in calls}),
        "fn_counts": dict(Counter(c["fn"] for c in calls)),
        "tag_counts": dict(sorted(tag_counts.items())),
        "tags": TAGS,
    }
    return header, table, blobs


def load(root: Path | None = None) -> tuple[list[dict[str, Any]], T.Blobs]:
    """The pinned recorded calls (``args``/``kwargs``/``outcome`` decoded from JSON) and their blobs.

    ``root``: where ``materialize_tests`` rebuilt the upstream tests (calls that name a test file need it).
    """
    from pyroid_parity import data as D

    cols = D.read_parquet_columns(D.path(CALLS), ["index", "test", "fn", "args", "kwargs", "tags", "outcome"])
    calls = [
        {
            "index": i,
            "test": t,
            "fn": fn,
            "args": json.loads(a),
            "kwargs": json.loads(k),
            "tags": list(tags),
            "outcome": o,
        }
        for i, t, fn, a, k, tags, o in zip(*cols.values(), strict=True)
    ]
    b = D.read_parquet_columns(D.path(BLOBS), ["sha256", "data"])
    return calls, T.Blobs(zip(b["sha256"], b["data"], strict=True), root=None if root is None else str(root))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pyroid_parity.trafilatura_record")
    ap.add_argument(
        "--replay", nargs=4, type=Path, metavar=("CALLS", "BLOBS", "OUT", "ROOT"), help=argparse.SUPPRESS
    )
    a = ap.parse_args(argv)
    if a.replay:
        _oracle_replay(*a.replay)
        return 0
    ap.error("run it through `python -m pyroid_parity.refs trafilatura_upstream`")
    return 2


if __name__ == "__main__":
    sys.exit(main())
