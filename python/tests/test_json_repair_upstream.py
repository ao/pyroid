# Replays calls recorded from json_repair 0.64.5's test suite (MIT), (c) Stefano Baccianella.
"""json_repair's own test suite, replayed against pyroid.json_repair.

``parity/data/json_repair-0.64.5/upstream-calls.jsonl`` holds every outermost public call
(``repair_json``, ``loads``, ``load``, ``from_file``) that json_repair 0.64.5's tests make, with
json_repair's outcome on CPython 3.12 (recorded by ``python -m pyroid_parity.json_repair_record``
in ``parity/``). Each call is replayed and must give the same outcome: ``str`` results byte for
byte, objects and ``(object, log)`` tuples type-exact (dict key order, int vs float, NaN), the
same exception type and message. The decoder and comparison come from
``parity/pyroid_parity/json_repair_compare.py``, loaded by path; without ``parity/`` (an sdist)
the module is skipped.

``PYROID_JR_IMPL=json_repair`` replays against json_repair itself (checks the recording).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "parity" / "data" / "json_repair-0.64.5" / "upstream-calls.jsonl"
COMPARE_PY = REPO / "parity" / "pyroid_parity" / "json_repair_compare.py"
IMPL = os.environ.get("PYROID_JR_IMPL", "pyroid.json_repair")

if not (DATA.is_file() and COMPARE_PY.is_file()):
    pytest.skip(f"{DATA.relative_to(REPO)} is not in this checkout", allow_module_level=True)
try:
    candidate = importlib.import_module(IMPL)
except ImportError as e:  # not built yet, or not in this wheel
    pytest.skip(f"{IMPL} is not importable: {e}", allow_module_level=True)


def _load_compare() -> Any:
    name = "pyroid_tests_json_repair_compare"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, COMPARE_PY)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses look their module up while the module executes
    spec.loader.exec_module(mod)
    return mod


C = _load_compare()
HEADER, CALLS = C.read_calls(DATA)
SKIP_TAGS: dict[str, str] = HEADER["skip_tags"]
# schema= calls: pyroid delegates them to json_repair, which needs jsonschema.
HAS_SCHEMA_DEPS = all(
    importlib.util.find_spec(m) is not None for m in ("json_repair", "jsonschema")
)


def test_recording_is_json_repair_0_64_5() -> None:
    assert HEADER["json_repair"] == "0.64.5"
    assert len(CALLS) == HEADER["calls"]


@pytest.mark.thread_unsafe  # warnings.catch_warnings inside C.replay
@pytest.mark.parametrize("call", CALLS, ids=[f"{c.index:04d}-{c.fn}" for c in CALLS])
def test_upstream_call(call: Any) -> None:
    for tag in call.tags:
        if tag in SKIP_TAGS:
            pytest.skip(f"{tag}: {SKIP_TAGS[tag]}")
    if "schema" in call.tags and not HAS_SCHEMA_DEPS:
        pytest.skip("schema= calls need json_repair and jsonschema installed")
    got, want = C.replay(candidate, call), call.want
    assert C.same_outcome(got, want), (
        f"{call.test}\n{call.fn}(*{call.args!r}, **{call.kwargs!r})\n"
        f"  got: {got.short(2000)}\n want: {want.short(2000)}\n"
        f"where: {C.describe_difference(got, want)}"
    )
