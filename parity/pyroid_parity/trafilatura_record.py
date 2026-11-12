"""Record Trafilatura's own test suite as golden public calls (``extract`` and ``bare_extraction``), per mode.

    python -m pyroid_parity.refs trafilatura_upstream [--pin]      # 2.3.1, in the default env
    python -m pyroid_parity.refs trafilatura111_upstream [--pin]   # 1.11.0, with the trafilatura111 group

The release recorded is the one installed, and it must be one of ``specs.TRAFILATURA_MODES``. The output goes to
``$PYROID_DATA`` (``specs.trafilatura_files(mode)``: ``calls`` and its blob table ``blobs``), not to the repository:
the test pages under ``tests/cache/`` are third-party web pages. The upstream tests themselves are fetched by
``bench/fetch.py --only trafilatura`` from GitHub at the commit of each release tag (``tests``: path, git blob
sha1, bytes).

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

Not run (``NOT_RUN_BY_MODE``): the CLI tests (subprocesses and the CLI's worker pools), downloads (HTTP), feeds,
sitemaps and spider (no extraction calls; in 1.11.0 they go to the network unmocked) and, in 2.3.1, the evaluation
harness (it needs ``tests/eval/``, 923 pages, not fetched).
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
from datetime import date
from pathlib import Path
from typing import Any

from pyroid_parity import specs
from pyroid_parity import trafilatura_compare as T

TRAFILATURA_VERSION = specs.TRAFILATURA_VERSION  # the default mode
#: The commits of the release tags in https://github.com/adbar/trafilatura.
COMMITS = {
    "2.3.1": "e2f85c965662c7df26b73290ef175b2c2a5fdca6",
    "1.11.0": "60647e5eeff77df9ef4b7432a936f580736b3075",
}
COMMIT = COMMITS[TRAFILATURA_VERSION]

#: Stands for the day of the run in the stored calls. Trafilatura stamps each document with the day it was
#: extracted (``filedate``), and its tests pass ``date_params`` whose ``max_date`` is that day, so a recording
#: that kept the date would reproduce, and replay, only on the day it was made. ``record`` writes this token
#: for the day (or days, past midnight) it ran; ``load`` puts back the day it runs on.
TODAY = "$today"


def _undate(s: str, days: set[str]) -> str:
    for d in days:
        s = s.replace(d, TODAY)
    return s


UPSTREAM_TESTS = specs.trafilatura_files(TRAFILATURA_VERSION)["tests"]
CALLS = specs.trafilatura_files(TRAFILATURA_VERSION)["calls"]
BLOBS = specs.trafilatura_files(TRAFILATURA_VERSION)["blobs"]
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
NOT_RUN_BY_MODE = {
    "2.3.1": NOT_RUN,
    "1.11.0": {
        "cli_tests.py": "the command-line interface: subprocesses and the CLI's worker pools, not direct calls",
        "downloads_tests.py": "HTTP downloads (to the network: 1.11.0's tests mock nothing)",
        "feeds_tests.py": "feed discovery over the network; no extract or bare_extraction calls",
        "sitemaps_tests.py": "sitemaps over the network; no extract or bare_extraction calls",
        "spider_tests.py": "crawling over the network; no extract or bare_extraction calls",
    },
}
TAGS = {
    "tree-input": "the input is an lxml tree, rebuilt node by node for the replay",
    "test-file": "an argument names a file of the upstream tests (settingsfile=), rebuilt for the replay",
    "unreplayable": "an argument could not be stored (dropped from the recording)",
    "extractor-subclass": "options= was a subclass of Extractor; replayed as a plain Extractor",
    "default-config-mutated": "the tests had changed trafilatura's DEFAULT_CONFIG in place (1.11.0's set "
    "MIN_OUTPUT_SIZE and MIN_EXTRACTED_SIZE to 0 at import): the call is recorded with the config= it saw",
    "config-added": "with default-config-mutated: the test passed no config=, the recording adds the one in effect",
    "pristine-config": "a default-config-mutated call again with trafilatura's own DEFAULT_CONFIG (its own group)",
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
        _blobs.default_config_mutated = _default_config_mutated()
        enc_a = [T.encode_arg(a, _blobs, tags) for a in args]
        enc_k = {k: T.encode_arg(v, _blobs, tags) for k, v in kwargs.items()}
        if _blobs.default_config_mutated:
            _add_effective_config(orig, args, kwargs, enc_k, tags)
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


def _config_state(config: Any) -> dict[str, dict[str, str]]:
    return {name: dict(section) for name, section in config.items()}


_pristine: dict[str, Any] = {}


def _default_config_mutated() -> bool:
    from trafilatura.settings import DEFAULT_CONFIG

    return _config_state(DEFAULT_CONFIG) != _pristine["config"]


def _add_effective_config(
    orig: Any, args: tuple[Any, ...], kwargs: dict[str, Any], enc_k: dict[str, Any], tags: set[str]
) -> None:
    """A test changed DEFAULT_CONFIG in place (1.11.0's do, at import): record the config a call that relies on
    the default actually used, so the replay sees the same thresholds."""
    from trafilatura.settings import DEFAULT_CONFIG

    sig = inspect.signature(orig)
    bound = sig.bind_partial(*args, **kwargs).arguments
    param = sig.parameters.get("config")
    if param is None or param.default is not DEFAULT_CONFIG:
        return
    if not ({"config", "options", "settingsfile"} & set(bound)):
        enc_k["config"] = T.encode_arg(DEFAULT_CONFIG, _blobs, tags)
        tags.update(("default-config-mutated", "config-added"))
    elif "config" in kwargs and kwargs["config"] is DEFAULT_CONFIG:
        tags.add("default-config-mutated")


def _install() -> None:
    import trafilatura
    from trafilatura import core
    from trafilatura.settings import DEFAULT_CONFIG

    _pristine["config"] = _config_state(DEFAULT_CONFIG)  # before any test module is imported

    for name, record in (("extract", True), ("bare_extraction", True), ("extract_with_metadata", False)):
        orig = getattr(core, name, None)
        if orig is None:  # extract_with_metadata is newer than 1.11.0
            continue
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


def materialize_tests(dst: Path, mode: str = TRAFILATURA_VERSION) -> int:
    """Write a mode's pinned upstream test files under ``dst`` (``dst/tests/...``); returns the number of files."""
    from pyroid_parity import data as D

    cols = D.read_parquet_columns(D.path(specs.trafilatura_files(mode)["tests"]), ["path", "data"])
    for rel, raw in zip(cols["path"], cols["data"], strict=True):
        p = dst / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    return len(cols["path"])


def run_upstream_suite(tmp: Path, pytest_args: list[str], mode: str = TRAFILATURA_VERSION) -> Path:
    materialize_tests(tmp, mode)
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
        *[f"--ignore=tests/{name}" for name in NOT_RUN_BY_MODE[mode]],
        *pytest_args,
    ]
    env = {**os.environ, _OUT_ENV: str(raw), _ROOT_ENV: str(tmp), "PYTHONDONTWRITEBYTECODE": "1"}
    env["PYTHONHASHSEED"] = "0"
    proc = subprocess.run(cmd, cwd=tmp, env=env, check=False, stdout=subprocess.DEVNULL)
    if proc.returncode not in (0, 1):
        raise SystemExit(f"the upstream pytest run did not complete (exit {proc.returncode})")
    return raw


def _pristine_variants(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The calls a test made with DEFAULT_CONFIG changed in place, again with trafilatura's own default.

    Each test's variants form a group of their own (test id + " [pristine config]"), appended after every
    original call, so the originals' dedup sequences are untouched. They have no observed outcome.
    """
    out = []
    for c in calls:
        if "default-config-mutated" not in c["tags"]:
            continue
        kwargs = dict(c["kwargs"])
        if "config-added" in c["tags"]:
            del kwargs["config"]
        else:
            kwargs["config"] = {"$config": "default"}
        tags = sorted((set(c["tags"]) - {"default-config-mutated", "config-added"}) | {"pristine-config"})
        out.append(
            {**c, "kwargs": kwargs, "tags": tags, "test": c["test"] + " [pristine config]", "observed": None}
        )
    return out


def _without_warnings(outcome_json: str) -> str:
    j = json.loads(outcome_json)
    j.pop("warnings", None)
    return json.dumps(j, ensure_ascii=True, separators=(",", ":"))


def record(pytest_args: list[str] | None = None, mode: str | None = None) -> tuple[dict[str, Any], Any, Any]:
    """Run, replay and tag; returns (header, calls table, blobs table) as pyarrow tables."""
    import pyarrow as pa

    installed = md.version("trafilatura")
    mode = mode or installed
    if installed != mode or mode not in COMMITS:
        raise SystemExit(f"recording mode {mode} needs trafilatura=={mode} installed, found {installed}")
    days = {date.today().isoformat()}
    with tempfile.TemporaryDirectory(prefix="pyroid-traf-record-") as d:
        tmp = Path(d)
        raw = run_upstream_suite(tmp, pytest_args or [], mode)
        summary = json.loads((raw / "summary.json").read_text("utf-8"))
        recorded = [json.loads(line) for line in (raw / "calls.jsonl").read_text("utf-8").splitlines()]
        dropped = sum(1 for c in recorded if "unreplayable" in c["tags"])
        calls = [c for c in recorded if "unreplayable" not in c["tags"]]
        calls += _pristine_variants(calls)
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
            if c["observed"] is not None and c["observed"] != _without_warnings(want):
                tags.add("differs-from-test")
            c["tags"], c["outcome"] = sorted(tags), want
            walk(c["args"])
            walk(c["kwargs"])
        days.add(date.today().isoformat())
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
            "args": [_undate(json.dumps(c["args"], ensure_ascii=True), days) for c in calls],
            "kwargs": [_undate(json.dumps(c["kwargs"], ensure_ascii=True), days) for c in calls],
            "tags": [c["tags"] for c in calls],
            "outcome": [_undate(c["outcome"], days) for c in calls],
        }
    )
    tag_counts = Counter(t for c in calls for t in c["tags"])
    header = {
        "source": f"trafilatura v{mode} tests/ at {COMMITS[mode]}",
        "python": platform.python_version(),
        "upstream_tests": {
            **summary["tests"],
            "failed_ids": summary["failed"],
            "not_run": NOT_RUN_BY_MODE[mode],
        },
        "calls": len(calls),
        "dropped_unreplayable": dropped,
        "tests_with_calls": len({c["test"] for c in calls}),
        "fn_counts": dict(Counter(c["fn"] for c in calls)),
        "tag_counts": dict(sorted(tag_counts.items())),
        "tags": TAGS,
    }
    return header, table, blobs


def load(root: Path | None = None, mode: str = TRAFILATURA_VERSION) -> tuple[list[dict[str, Any]], T.Blobs]:
    """The pinned recorded calls (``args``/``kwargs``/``outcome`` decoded from JSON) and their blobs.

    ``root``: where ``materialize_tests`` rebuilt the upstream tests (calls that name a test file need it).
    The recording day's token (``TODAY``) becomes today's date, as a fresh run of the tests would see it.
    """
    from pyroid_parity import data as D

    today = date.today().isoformat()

    files = specs.trafilatura_files(mode)
    names = ["index", "test", "fn", "args", "kwargs", "tags", "outcome"]
    cols = D.read_parquet_columns(D.path(files["calls"]), names)
    calls = [
        {
            "index": i,
            "test": t,
            "fn": fn,
            "args": json.loads(a.replace(TODAY, today)),
            "kwargs": json.loads(k.replace(TODAY, today)),
            "tags": list(tags),
            "outcome": o.replace(TODAY, today),
        }
        for i, t, fn, a, k, tags, o in zip(*cols.values(), strict=True)
    ]
    b = D.read_parquet_columns(D.path(files["blobs"]), ["sha256", "data"])
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
