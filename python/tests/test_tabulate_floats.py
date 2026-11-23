"""pyroid.tabulate's float text against CPython's own ``repr`` and ``format(x, spec)``.

tabulate prints a float cell as ``f"{x}"`` (text columns, ``disable_numparse``, the ``pretty``
format) or as ``format(float(x), floatfmt)``, and pyroid.tabulate makes that text natively.
The values here are the hard cases for the digits: exact ties at the last digit of the
shortest repr (common in float32 values widened to float64), each side of every power of ten,
subnormals and random bit patterns. The reference is the running interpreter, so tabulate
need not be installed.
"""

from __future__ import annotations

import math
import random
import struct
import warnings
from typing import Any

import pytest

import pyroid.tabulate as pt

# Found by an outside check (each printed with another last digit before the fix).
TIES = [0.9790115356445312, 0.6272964477539062, 1331752170181752.2, 272721534007552.62]


def _f32(bits: int) -> float:
    return float(struct.unpack("<f", struct.pack("<I", bits))[0])


def _f64(bits: int) -> float:
    return float(struct.unpack("<d", struct.pack("<Q", bits))[0])


def _values(n: int, seed: int) -> list[float]:
    rnd = random.Random(seed)
    vals = [*TIES, 1e23, 1e-06, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308]
    vals += [0.0, -0.0, 0.5, 2.5, 0.125, 1e16, 9007199254740993.0, math.inf, -math.inf, math.nan]
    for k in range(-323, 309):
        p = float(f"1e{k}")
        vals += [p, math.nextafter(p, 0.0), math.nextafter(p, math.inf)]
    vals += [_f32(0x3F800000 | rnd.getrandbits(23)) - 1.0 for _ in range(n)]  # float32 in [0, 1)
    vals += [_f32(rnd.getrandbits(32)) for _ in range(n)]  # any float32
    vals += [_f64(rnd.getrandbits(64)) for _ in range(n)]  # any float64
    vals += [_f64(rnd.getrandbits(52)) for _ in range(n // 10)]  # subnormals
    return [-v if rnd.random() < 0.25 else v for v in vals]


REPR_VALUES = _values(20_000, 1)
FORMAT_VALUES = _values(1_000, 2)

SPECS = [
    *["", ",", "_", "g", "G", "e", "E", "f", "F", "%", "n", "#", "#g", "#e", "#f", "#%"],
    *[f".{p}{t}" for t in ("g", "e", "f") for p in range(32)],
    *[f".{p}{t}" for t in ("", "%", "G", "E") for p in (0, 1, 2, 6, 14, 15, 16, 17, 18, 25)],
    *["#.0f", "#.0e", "#.0%", "#.3g", "#.16g", "#.17g", "#.16", "+g", " .6g", "+.17g", "-.16e"],
    *["z.2f", "z.0g", ",.2f", "_.3f", ",.17g", ",.16", ",.0f", ",f", ",e", ",%", ",.1%", ".3n"],
]


def _lines(rows: list[list[Any]], **kw: Any) -> list[str]:
    """One-column table, cells as tabulate formats them (no alignment or padding)."""
    return pt.tabulate(rows, tablefmt="plain", numalign=None, stralign=None, **kw).split("\n")


def _first_difference(got: list[str], want: list[str], values: list[Any]) -> str:
    for v, g, w in zip(values, got, want, strict=True):
        if g != w:
            return f"{v!r} ({float(v).hex()}): got {g!r}, want {w!r}"
    return ""


def test_the_reported_values() -> None:
    reprs = ["0.9790115356445312", "0.6272964477539062", "1331752170181752.2", "272721534007552.62"]
    assert [repr(x) for x in TIES] == reprs
    for x, want in zip(TIES, reprs, strict=True):
        assert pt.tabulate([[x]], tablefmt="pretty").split("\n")[1] == f"| {want} |"
        assert pt.tabulate([[x]], disable_numparse=True).split("\n")[1] == want
        assert _lines([[x]], floatfmt="") == [want]
        assert _lines([[x]], floatfmt=",") == [format(x, ",")]
        assert _lines([[x], ["n/a"]]) == [want, "n/a"]
    assert _lines([[1e23]], floatfmt=".16g") == ["9.999999999999999e+22"]
    assert _lines([[1e-06]], floatfmt=".17g") == ["9.9999999999999995e-07"]


def test_repr_like_python() -> None:
    vals = REPR_VALUES
    want = [repr(v) for v in vals]
    rows = [[v] for v in vals]
    options: list[dict[str, Any]] = [{"disable_numparse": True}, {"floatfmt": ""}]
    for kw in options:
        got = _lines(rows, **kw)
        assert got == want, _first_difference(got, want, vals)
    # A text cell makes the column a text column.
    got = _lines([*rows, ["n/a"]])[:-1]
    assert got == want, _first_difference(got, want, vals)
    # tablefmt="pretty" turns number parsing off.
    got = [line[2:-2].strip() for line in pt.tabulate(rows, tablefmt="pretty").split("\n")[1:-1]]
    assert got == want, _first_difference(got, want, vals)


@pytest.mark.parametrize("spec", SPECS)
def test_floatfmt_like_format(spec: str) -> None:
    vals = FORMAT_VALUES
    want = [format(v, spec) for v in vals]
    got = _lines([[v] for v in vals], floatfmt=spec)
    assert got == want, _first_difference(got, want, vals)


@pytest.mark.parametrize("ty", ["f", "F", "e", "E", "%", "g", "G", ""])
def test_precisions_and_widths_at_rusts_limit(ty: str) -> None:
    # Rust's formatter takes a width or precision of at most 65535 and panics above it; P67
    # found `.65536f` raising PanicException. Longer specs are Python's.
    vals = [0.5, -2.25, 1e300, 5e-324, math.inf]
    specs = [f"{w}.{p}{ty}" for w in ("", "65535", "65536") for p in (65535, 65536, 65537)]
    specs += [f"{w}.1{ty}" for w in (65535, 65536, 65537)]
    for spec in specs:
        want = [format(v, spec) for v in vals]
        assert _lines([[v] for v in vals], floatfmt=spec) == want, spec
        # Number strings in a float column take the same path.
        assert _lines([["0.5"], ["1e300"]], floatfmt=spec) == want[:1] + want[2:3], spec
    for spec in ("65535", "65536", "65537", "65536,", "65536_d"):
        assert _lines([[5], [-12345]], intfmt=spec) == [format(5, spec), format(-12345, spec)]


@pytest.mark.parametrize("spec", ["g", "", ".16g", ".17g", ".20e", ".3f", ",.2f"])
def test_numeric_strings_like_format(spec: str) -> None:
    # A numeric str cell in a float column is format(float(s), spec); float(s) rounds correctly.
    vals = [v for v in FORMAT_VALUES if math.isfinite(v)]
    for text in (repr, lambda v: f"{v:.25e}"):
        strings = [text(v) for v in vals]
        want = [format(float(s), spec) for s in strings]
        got = _lines([[s] for s in strings], floatfmt=spec)
        assert got == want, _first_difference(got, want, strings)


def test_numpy_scalars_like_tabulate() -> None:
    np = pytest.importorskip("numpy")
    vals = FORMAT_VALUES
    for dtype in (np.float64, np.float32, np.float16):
        with np.errstate(over="ignore"):
            scalars = [dtype(v) for v in vals]
        for spec in ("", ".16g", ".17g", ".3e"):
            want = [format(float(x), spec) for x in scalars]
            got = _lines([[x] for x in scalars], floatfmt=spec)
            assert got == want, _first_difference(got, want, scalars)
        # In a text column a numpy scalar prints its own str.
        want = [f"{x}" for x in scalars]
        got = _lines([[x] for x in scalars], disable_numparse=True)
        assert got == want, _first_difference(got, want, scalars)


@pytest.mark.thread_unsafe  # sys.modules
def test_dataframe_to_markdown_patched() -> None:
    from pyroid import integrations

    pd = pytest.importorskip("pandas")
    pytest.importorskip("tabulate")
    vals = [v for v in REPR_VALUES[:5000] if not math.isnan(v)]
    df = pd.DataFrame({"score": vals})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", integrations.PatchWarning)
        integrations.patch("tabulate")
    try:
        got = df.to_markdown(tablefmt="pretty", index=False).split("\n")[3:-1]
    finally:
        integrations.unpatch("tabulate")
    want = [repr(v) for v in vals]
    got = [line[2:-2].strip() for line in got]
    assert got == want, _first_difference(got, want, vals)
