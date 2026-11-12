"""Record json_repair 0.64.5's own test suite as golden public calls (``upstream-calls.jsonl``).

    python -m pyroid_parity.json_repair_record           # rewrite data/json_repair-0.64.5/upstream-calls.jsonl
    python -m pyroid_parity.json_repair_record --check   # re-record and compare with the committed file

Run it in Docker, in the parity env plus pydantic (the schema tests need it)::

    cd parity && UV_PROJECT_ENVIRONMENT=/target/venv-parity uv run --frozen -p 3.12 --with pydantic \\
        python -m pyroid_parity.json_repair_record

How it works:

1. The vendored upstream tests (``upstream_tests/``, tag v0.64.5) are copied to a temporary ``tests/`` package
   and run by pytest in a subprocess with this module as a plugin (``-p pyroid_parity.json_repair_record``).
   The upstream tests import ``src.json_repair...`` (the source tree): the plugin aliases every
   ``src.json_repair.*`` module to the installed ``json_repair.*`` module object, then wraps ``repair_json``,
   ``loads``, ``load`` and ``from_file`` (module and package attributes) before any test module is imported.
2. Only *outermost* calls are recorded (``loads`` calls ``repair_json``, ``from_file`` calls ``load``): every
   argument as given, the stream content and position for ``load(fd)`` / ``json_fd=``, the file bytes for
   ``from_file``, and the outcome the test observed. A test's ``JSONParser(text, None, logging, chunk,
   stream_stable, strict).parse()`` on a plain string maps to the public call ``repair_json(text,
   return_objects=True, skip_json_loads=True, ...)`` and is recorded as that (tag ``via-JSONParser``); other
   ``JSONParser`` use (internal methods, a swapped ``json_str``) is not a public call and is not recorded.
   Calls whose arguments cannot be stored (a schema nested past the stack limit) are dropped and counted.
3. The calls are deduplicated, then **replayed in a clean subprocess with** ``PYTHONHASHSEED=0`` against the
   unpatched json_repair: the golden outcome is that replay, not what the test observed (some tests monkeypatch
   json_repair internals; tag ``patched-in-test`` if the two differ).
4. Each call is probed for reference nondeterminism: replayed under 400 extra stack frames (outcomes that depend
   on Python's recursion limit and the caller's stack depth get the tag ``recursion``), and with
   ``PYTHONHASHSEED=1`` and a random seed (tag ``hash-seed`` if the outcome changes).

Not run (and why): ``test_performance.py`` (pytest-benchmark timing tests on multi-MB inputs),
``test_type_inference.py`` (runs mypy on a snippet; no runtime calls), ``test_docs_app_schema.py`` (imports the
Flask demo app ``docs/app.py``, which is not part of the package).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata as md
import inspect
import json
import os
import pkgutil
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import types
from collections import Counter
from pathlib import Path
from typing import Any

from pyroid_parity import json_repair_compare as C

JSON_REPAIR_VERSION = "0.64.5"
DATA = Path(__file__).resolve().parents[1] / "data" / f"json_repair-{JSON_REPAIR_VERSION}"
UPSTREAM_TESTS = DATA / "upstream_tests"
GOLDEN = DATA / "upstream-calls.jsonl"
NOT_RUN = {
    "test_performance.py": "pytest-benchmark timing tests on multi-MB inputs",
    "test_type_inference.py": "runs mypy on a code snippet; makes no runtime calls",
    "test_docs_app_schema.py": "imports the Flask demo app docs/app.py, which is not part of the package",
}
TAGS = {
    "schema": "uses schema= (pyroid delegates schema-guided repair to json_repair itself)",
    "schema-pydantic": "the test passed a pydantic model; recorded as the JSON Schema dict json_repair derives "
    "from it (schema_from_input), which is what the repair uses",
    "via-JSONParser": "the test called JSONParser(text, ...).parse(); recorded as the equivalent public call "
    "repair_json(text, return_objects=True, skip_json_loads=True, ...)",
    "patched-in-test": "the test monkeypatched json_repair internals during this call, so what it observed "
    "differs from the clean replay stored here",
    "recursion": "the outcome depends on Python's recursion limit and the caller's stack depth (deep nesting)",
    "hash-seed": "the outcome changes with PYTHONHASHSEED",
}
#: Tags whose calls are not compared by default, with the reason shown in the skip.
SKIP_TAGS = {
    "recursion": TAGS["recursion"],
    "hash-seed": TAGS["hash-seed"],
}
EXTRA_FRAMES = 400
MAX_TEST_ID = 160  # upstream parametrize ids embed whole inputs
FILE_PARAMS = {"repair_json": ("json_fd", 4), "load": ("fd", 0), "from_file": ("filename", 0)}
SCHEMA_POS = {"repair_json": 8, "loads": 5, "load": 5, "from_file": 5}
PUBLIC = ("repair_json", "loads", "load", "from_file")

# ============================================================================== pytest plugin (subprocess side)

_OUT_ENV = "PYROID_JR_RECORD_OUT"


class _State(threading.local):
    depth = 0


_state = _State()
_records: list[dict[str, Any]] = []
_tally: Counter[str] = Counter()
_current = {"test": "<collection>"}


def _file_spec(name: str, value: Any, tags: list[str]) -> Any:
    if value is None:
        return None
    if name == "from_file":
        p = Path(value) if isinstance(value, (str, os.PathLike)) else None
        if p is None or not p.is_file():
            tags.append("unreplayable")
            return value
        return C.PathSpec(p.read_bytes(), "Path" if isinstance(value, Path) else "str")
    if hasattr(value, "getvalue") and hasattr(value, "tell"):
        return C.TextIOSpec(value.getvalue(), value.tell())
    tags.append("unreplayable")
    return value


def _schema_spec(value: Any, tags: list[str]) -> Any:
    if value is None:
        return None
    tags.append("schema")
    if isinstance(value, (dict, bool)):
        return value
    tags.append("schema-pydantic")
    try:
        from json_repair.schema_repair import schema_from_input

        return schema_from_input(value)
    except Exception:  # noqa: BLE001  e.g. the test patched pydantic.VERSION to provoke an error
        tags.append("unreplayable")
        return value


def _inputs(
    name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any], list[str]]:
    tags: list[str] = []
    a, k = list(args), dict(kwargs)
    if name in FILE_PARAMS:
        param, pos = FILE_PARAMS[name]
        if param in k:
            k[param] = _file_spec(name, k[param], tags)
        elif len(a) > pos:
            a[pos] = _file_spec(name, a[pos], tags)
    if "schema" in k:
        k["schema"] = _schema_spec(k["schema"], tags)
    elif len(a) > SCHEMA_POS[name]:
        a[SCHEMA_POS[name]] = _schema_spec(a[SCHEMA_POS[name]], tags)
    try:
        enc_a, enc_k = C.encode(a), C.encode(k)
        if "$object" in json.dumps([enc_a, enc_k]):
            tags.append("unreplayable")
    except RecursionError:  # a schema nested deeper than the stack (the test provokes json_repair's guard)
        enc_a, enc_k = [C.encode(Ellipsis)], {}
        tags.append("unreplayable")
    return enc_a, enc_k, tags


def _record(name: str, args: list[Any], kwargs: dict[str, Any], tags: list[str], observed: Any) -> None:
    _records.append(
        {
            "fn": name,
            "args": args,
            "kwargs": kwargs,
            "tags": tags,
            "observed": observed,
            "test": _current["test"],
        }
    )


def _wrap(name: str, orig: Any) -> Any:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _state.depth:
            return orig(*args, **kwargs)
        enc_a, enc_k, tags = _inputs(name, args, kwargs)
        _state.depth += 1
        try:
            result = orig(*args, **kwargs)
        except Exception as e:
            _record(name, enc_a, enc_k, tags, {"raise": [type(e).__name__, str(e)]})
            raise
        finally:
            _state.depth -= 1
        try:
            observed: Any = C.Outcome("ok", result).to_json()
        except RecursionError:
            observed = None
        _record(name, enc_a, enc_k, tags, observed)
        return result

    wrapper.__wrapped__ = orig  # type: ignore[attr-defined]
    wrapper.__signature__ = inspect.signature(orig)  # type: ignore[attr-defined]
    wrapper.__name__ = orig.__name__
    wrapper.__doc__ = orig.__doc__
    return wrapper


def _wrap_parser(jp: Any) -> None:
    orig_init, orig_parse = jp.__init__, jp.parse
    init_sig = inspect.signature(orig_init)

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        self._pyroid_record = None if _state.depth else (args, kwargs)

    def parse(self: Any) -> Any:
        rec = getattr(self, "_pyroid_record", None)
        if _state.depth or rec is None:
            return orig_parse(self)
        b = init_sig.bind(self, *rec[0], **rec[1])
        b.apply_defaults()
        p = b.arguments
        if (
            p["json_fd"] is None
            and type(p["json_str"]) is str
            and self.json_str is p["json_str"]
            and not self.index
        ):
            kwargs: dict[str, Any] = {"return_objects": True, "skip_json_loads": True}
            for opt in ("logging", "stream_stable", "strict"):
                if p[opt]:
                    kwargs[opt] = True
            _record("repair_json", [C.encode(p["json_str"])], C.encode(kwargs), ["via-JSONParser"], None)
        _state.depth += 1
        try:
            return orig_parse(self)
        finally:
            _state.depth -= 1

    jp.__init__, jp.parse = init, parse


def _install() -> None:
    import json_repair

    for m in pkgutil.walk_packages(json_repair.__path__, "json_repair."):
        if not m.name.endswith("__main__"):
            importlib.import_module(m.name)
    src = types.ModuleType("src")
    src.__path__ = []
    src.json_repair = json_repair  # type: ignore[attr-defined]
    sys.modules["src"] = src
    for name, mod in list(sys.modules.items()):
        if name == "json_repair" or name.startswith("json_repair."):
            sys.modules["src." + name] = mod
    jr = sys.modules["json_repair.json_repair"]
    for name in PUBLIC:
        orig = getattr(jr, name)
        w = _wrap(name, orig)
        setattr(jr, name, w)
        if getattr(json_repair, name, None) is orig:
            setattr(json_repair, name, w)
    _wrap_parser(sys.modules["json_repair.json_parser"].JSONParser)


def pytest_configure(config: Any) -> None:
    if os.environ.get(_OUT_ENV):
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


_failed: list[str] = []


def pytest_sessionfinish(session: Any) -> None:
    out = os.environ.get(_OUT_ENV)
    if not out:
        return
    tally = dict(_tally)
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"tests": tally, "failed": _failed}) + "\n")
        f.writelines(json.dumps(r, ensure_ascii=True) + "\n" for r in _records)


# ===================================================================================== recorder (CLI side)


def _deep(n: int, fn: Any) -> Any:
    return fn() if n <= 0 else _deep(n - 1, fn)


def _replay_all(calls: list[C.Call], extra_frames: int = 0) -> list[dict[str, Any]]:
    import json_repair

    return [_deep(extra_frames, lambda c=c: C.replay(json_repair, c)).to_json() for c in calls]


def _replay_subprocess(calls: list[C.Call], seed: str, tmp: Path) -> list[dict[str, Any]]:
    src, dst = tmp / f"calls-{seed}.jsonl", tmp / f"outcomes-{seed}.json"
    C.write_calls(src, {}, calls)
    env = {**os.environ, "PYTHONHASHSEED": seed}
    subprocess.run(
        [sys.executable, "-m", "pyroid_parity.json_repair_record", "--replay", str(src), str(dst)],
        check=True,
        env=env,
    )
    return json.loads(dst.read_text("utf-8"))


def _is_recursion_outcome(o: dict[str, Any]) -> bool:
    return "raise" in o and ("recursion" in o["raise"][1].lower() or o["raise"][0] == "RecursionError")


def run_upstream_suite(tmp: Path, pytest_args: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tests = tmp / "tests"
    shutil.copytree(UPSTREAM_TESTS, tests, ignore=shutil.ignore_patterns("__pycache__"))
    (tmp / "pytest.ini").write_text("[pytest]\npythonpath = .\n", "utf-8")
    raw = tmp / "raw.jsonl"
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
        "pyroid_parity.json_repair_record",
        "-p",
        "no:cacheprovider",
        "-q",
        *[f"--ignore=tests/{name}" for name in NOT_RUN],
        *pytest_args,
    ]
    env = {**os.environ, _OUT_ENV: str(raw), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}
    proc = subprocess.run(cmd, cwd=tmp, env=env, check=False)
    if proc.returncode not in (0, 1):
        raise SystemExit(f"upstream pytest run failed to complete (exit {proc.returncode})")
    lines = raw.read_text("utf-8").splitlines()
    return json.loads(lines[0]), [json.loads(line) for line in lines[1:]]


def record(pytest_args: list[str]) -> tuple[dict[str, Any], list[C.Call]]:
    with tempfile.TemporaryDirectory(prefix="pyroid-jr-record-") as d:
        tmp = Path(d)
        summary, raw = run_upstream_suite(tmp, pytest_args)
        uniq: dict[str, C.Call] = {}
        observed: dict[str, list[Any]] = {}
        for r in raw:
            call = C.Call(
                r["fn"], r["args"], r["kwargs"], {}, sorted(set(r["tags"])), r["test"][:MAX_TEST_ID]
            )
            k = call.key()
            if k not in uniq:
                uniq[k] = call
            else:
                uniq[k].tags = sorted(set(uniq[k].tags) | set(call.tags))
            observed.setdefault(k, []).append(r["observed"])
        unreplayable = sum(1 for c in uniq.values() if "unreplayable" in c.tags)
        calls = [c for c in uniq.values() if "unreplayable" not in c.tags]
        # The golden outcome is a clean replay with PYTHONHASHSEED=0 (deterministic); the others are probes.
        golden, seed1 = (_replay_subprocess(calls, s, tmp) for s in ("0", "1"))
        here, deep = _replay_all(calls), _replay_all(calls, EXTRA_FRAMES)
        for i, (call, out) in enumerate(zip(calls, golden, strict=True)):
            call.outcome, call.index = out, i
            tags = set(call.tags)
            want = C.Outcome.from_json(out)
            o_here, o_deep = C.Outcome.from_json(here[i]), C.Outcome.from_json(deep[i])
            if _is_recursion_outcome(out) or not C.same_outcome(o_deep, o_here):
                tags.add("recursion")
            if not (C.same_outcome(C.Outcome.from_json(seed1[i]), want) and C.same_outcome(o_here, want)):
                tags.add("hash-seed")
            seen = [C.Outcome.from_json(o) for o in observed[call.key()] if o is not None]
            if "hash-seed" not in tags and any(not C.same_outcome(o, want) for o in seen):
                tags.add("patched-in-test")
            call.tags = sorted(tags)
        tag_counts = Counter(t for c in calls for t in c.tags)
        header = {
            "json_repair": md.version("json-repair"),
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "free_threaded": bool(getattr(sys, "_is_gil_enabled", lambda: True)() is False),
            "platform": f"{sys.platform}-{platform.machine()}",
            "source": f"json_repair v{JSON_REPAIR_VERSION} tests/ (vendored in upstream_tests/)",
            "upstream_tests": {**summary["tests"], "failed_ids": summary["failed"], "not_run": NOT_RUN},
            "calls": len(calls),
            "recorded_calls": len(raw),
            "dropped_unreplayable": unreplayable,
            "fn_counts": dict(Counter(c.fn for c in calls)),
            "tag_counts": dict(sorted(tag_counts.items())),
            "tags": TAGS,
            "skip_tags": SKIP_TAGS,
        }
        return header, calls


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pyroid_parity.json_repair_record")
    ap.add_argument("--out", type=Path, default=GOLDEN)
    ap.add_argument("--check", action="store_true", help="compare with --out instead of writing it")
    ap.add_argument("--replay", nargs=2, type=Path, metavar=("CALLS", "OUT"), help=argparse.SUPPRESS)
    ap.add_argument("pytest_args", nargs="*", help="extra arguments for the upstream pytest run")
    a = ap.parse_args(argv)
    if a.replay:
        _, calls = C.read_calls(a.replay[0])
        a.replay[1].write_text(json.dumps(_replay_all(calls)), "utf-8")
        return 0
    installed = md.version("json-repair")
    if installed != JSON_REPAIR_VERSION:
        raise SystemExit(f"needs json-repair=={JSON_REPAIR_VERSION}, found {installed}")
    header, calls = record(a.pytest_args)
    t = header["upstream_tests"]
    print(
        f"upstream tests: {t.get('passed', 0)} passed, {t.get('failed', 0)} failed, {t.get('errors', 0)} errors, "
        f"{t.get('skipped', 0)} skipped; {header['recorded_calls']} public calls recorded, "
        f"{header['calls']} unique; tags: {header['tag_counts']}"
    )
    if a.check:
        _, pinned = C.read_calls(a.out)
        got = [json.dumps(c.to_json(), sort_keys=True) for c in calls]
        want = [json.dumps(c.to_json(), sort_keys=True) for c in pinned]
        if got != want:
            diff = sum(1 for x, y in zip(got, want, strict=False) if x != y) + abs(len(got) - len(want))
            print(f"{a.out}: {diff} calls differ ({len(got)} recorded, {len(want)} pinned)")
            return 1
        print(f"{a.out}: identical ({len(got)} calls)")
        return 0
    with contextlib.suppress(FileNotFoundError):
        a.out.unlink()
    C.write_calls(a.out, header, calls)
    print(f"wrote {a.out} ({a.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
