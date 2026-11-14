"""pyroid.tabulate against tabulate 0.10.0: the same text (and type), or the same exception.

Random tables of mixed cells (ints of any size, floats with inf/nan/-0.0, numeric strings in
every form tabulate's parser meets, bools, None, bytes, Decimal, dates, numpy scalars,
multiline, wide and ANSI-coloured text), in every input shape and format, with random option
combinations and module settings. Warnings must match too (``pyroid.FallbackWarning`` aside).

``PYROID_TABULATE_EXAMPLES`` sets the number of Hypothesis examples (default 300).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, assume, given, settings


def _helpers() -> Any:
    """python/tests/_tabulate_util.py (CI imports tests with --import-mode=importlib)."""
    name = "_tabulate_util"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("_tabulate_util.py")
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_u = _helpers()
ref: Any = pytest.importorskip("tabulate")
pytest.importorskip("numpy")
pytest.importorskip("pandas")
import pyroid  # noqa: E402
import pyroid.tabulate as pt  # noqa: E402

if getattr(ref, "__pyroid__", False) or ref.__version__ != pt.TABULATE_VERSION:
    pytest.skip(f"needs the real tabulate {pt.TABULATE_VERSION}", allow_module_level=True)

EXAMPLES = int(os.environ.get("PYROID_TABULATE_EXAMPLES", "300"))


def _compare(case: dict[str, Any]) -> None:
    def call(mod: Any) -> Any:
        data, kwargs = _u.build_args(case, mod)
        return mod.tabulate(data, **kwargs)

    try:
        want = _u.outcome(_u.with_settings(ref, case["settings"], lambda: call(ref)), timeout=5)
    except _u.HangError:
        assume(False)  # tabulate itself never returns; pyroid runs the same wrapping code
    got = _u.outcome(
        _u.with_settings(pt, case["settings"], lambda: call(pt)),
        ignore=(pyroid.FallbackWarning,),
    )
    assert got.kind == want.kind, f"\n got: {got}\nwant: {want}"
    assert got.value == want.value, f"\n got: {got}\nwant: {want}"
    assert got.warnings == want.warnings


@pytest.mark.thread_unsafe  # module settings, warnings.catch_warnings
@settings(
    max_examples=EXAMPLES,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(_u.tables())
def test_same_as_tabulate(case: dict[str, Any]) -> None:
    _compare(case)
