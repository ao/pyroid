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
from hypothesis import strategies as st


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


# Focused properties: number formats and the strings tabulate takes for numbers.

_FILL = st.sampled_from(["", "*", "0", " ", "é"])
_ALIGN = st.sampled_from(["", "<", ">", "^", "="])


@st.composite
def float_specs(draw: st.DrawFn) -> str:
    """Format specs from Python's mini-language (some invalid, which must fail alike)."""
    align = draw(_ALIGN)
    fill = draw(_FILL) if align else ""
    parts = [
        fill + align,
        draw(st.sampled_from(["", "", "+", "-", " "])),
        draw(st.sampled_from(["", "", "z"])),
        draw(st.sampled_from(["", "", "#"])),
        draw(st.sampled_from(["", "", "0"])),
        draw(st.sampled_from(["", "", "", "1", "8", "12"])),
        draw(st.sampled_from(["", "", "", ",", "_"])),
        draw(st.sampled_from(["", "", ".0", ".1", ".2", ".3", ".6", ".10", ".17", ".25"])),
        draw(st.sampled_from(["", "e", "E", "f", "F", "g", "G", "%", "n", "d", "x"])),
    ]
    return "".join(parts)


@pytest.mark.thread_unsafe  # warnings.catch_warnings
@settings(max_examples=EXAMPLES * 5, deadline=None)
@given(
    st.lists(
        st.one_of(
            st.floats(allow_nan=True, allow_infinity=True),
            st.integers(-(10**20), 10**20),
            st.sampled_from([0.0, -0.0, 0.5, 2.5, 1e16, 1e-5, 123456.789, 0.125]),
        ),
        min_size=1,
        max_size=4,
    ),
    float_specs(),
    st.sampled_from(["", ",", "_", "d", "+", "08d", ">6", "x", ".2f", "e"]),
    st.sampled_from(["simple", "pipe", "plain"]),
)
def test_number_formats(values: list[Any], floatfmt: str, intfmt: str, fmt: str) -> None:
    rows = [[v, float(v) if isinstance(v, int) else v] for v in values]
    want = _u.outcome(lambda: ref.tabulate(rows, floatfmt=floatfmt, intfmt=intfmt, tablefmt=fmt))
    got = _u.outcome(lambda: pt.tabulate(rows, floatfmt=floatfmt, intfmt=intfmt, tablefmt=fmt))
    assert (got.kind, got.value) == (want.kind, want.value), (floatfmt, intfmt)


_NUMBERISH = st.lists(
    st.sampled_from(
        [
            *"0123456789+-.,_eEinfatyINFATY \t\n\x0b\x0c\r\x1c",
            *["\u0661", "\u0662", "\uff10", "\uff15", "\xa0", "\u2028", "\u3000", "\xb2"],
            *["\xbd", "\u65e5", "\x1b[1m", "\x1b[0m"],
        ]
    ),
    max_size=10,
).map("".join)


@pytest.mark.thread_unsafe  # warnings.catch_warnings
@settings(max_examples=EXAMPLES * 5, deadline=None)
@given(
    st.lists(st.lists(_NUMBERISH, min_size=2, max_size=2), min_size=1, max_size=4),
    st.sampled_from(["simple", "grid", "pipe"]),
    st.booleans(),
)
def test_number_like_strings(rows: list[list[str]], fmt: str, wide: bool) -> None:
    def run(mod: Any) -> Any:
        return _u.with_settings(
            mod, {"WIDE_CHARS_MODE": wide}, lambda: mod.tabulate(rows, tablefmt=fmt)
        )()

    want = _u.outcome(lambda: run(ref))
    got = _u.outcome(lambda: run(pt))
    assert (got.kind, got.value) == (want.kind, want.value)
