"""pyroid.packaging's batch helpers give what the one-at-a-time calls give."""

from __future__ import annotations

import random
from typing import Any

import pytest

import pyroid.packaging as rp
from pyroid.packaging.specifiers import Specifier, SpecifierSet
from pyroid.packaging.version import InvalidVersion, Version

RNG = random.Random(7)
SPELLINGS = [
    "",
    "",
    "",
    "rc1",
    "a2",
    "b0",
    ".post1",
    ".dev3",
    "+local.7",
    ".post2.dev1",
    "-1",
    "alpha",
    "c1",
]
VERSIONS = [
    f"{RNG.randint(0, 3)}.{RNG.randint(0, 12)}"
    + RNG.choice(["", f".{RNG.randint(0, 9)}"])
    + RNG.choice(SPELLINGS)
    for _ in range(20_000)
] + ["foo", "1..0", "", " 1.0 ", "v1.0", "1!2.0", "1.0\ud800"]
SPECS = [
    "", ">=1.0", "<2,>=1", "~=1.4.2", "==1.*", "!=1.5.*,>=1", ">1.0.post1", "<1.0", "==1.0+local",
    ">1.0a1",
    ">=1.0,<1.0", "===1.0", "==2.0.0,===2.0",
]  # fmt: skip


def outcome(f: Any, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
    try:
        return ("ok", f(*args, **kwargs))
    except Exception as e:
        return ("err", type(e).__name__, str(e))


def test_parse_batch() -> None:
    got = rp.parse_batch(VERSIONS, errors="none")
    for text, v in zip(VERSIONS, got, strict=True):
        try:
            want = Version(text)
        except InvalidVersion:
            assert v is None
        else:
            assert type(v) is Version
            assert str(v) == str(want)
            assert v == want
    with pytest.raises(InvalidVersion, match="Invalid version: 'foo'") as info:
        rp.parse_batch(VERSIONS)
    assert any("index 20000" in n for n in info.value.__notes__)


def test_sort_versions() -> None:
    valid = [v for v in VERSIONS if outcome(Version, v)[0] == "ok"]
    mixed = [Version(v) if i % 3 == 0 else v for i, v in enumerate(valid)]
    for threads in (None, 1, 4):
        for reverse in (False, True):
            got = rp.sort_versions(mixed, reverse=reverse, threads=threads)
            want = sorted(
                mixed, key=lambda x: x if isinstance(x, Version) else Version(x), reverse=reverse
            )
            assert all(a is b for a, b in zip(got, want, strict=True))
    with pytest.raises(InvalidVersion, match="Invalid version: 'foo'"):
        rp.sort_versions(VERSIONS)
    assert rp.sort_versions(tuple(valid[:5])) == sorted(valid[:5], key=Version)


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("pre", [None, True, False])
def test_contains_and_filter_batch(spec: str, pre: bool | None) -> None:
    ss = SpecifierSet(spec)
    items = VERSIONS[:3000] + VERSIONS[-7:] + [Version("1.0"), Version("1.5a1")]
    first = next(iter(ss), None)
    targets: list[Any] = [ss, spec, *([first] if first is not None else [])]
    for target in targets:
        obj = SpecifierSet(target) if isinstance(target, str) else target
        want = [obj.contains(x, prereleases=pre) for x in items]
        assert rp.contains_batch(target, items, prereleases=pre) == want
        assert rp.contains_batch(target, tuple(items), prereleases=pre) == tuple(want)
        assert rp.filter_batch(target, items, prereleases=pre) == list(
            obj.filter(items, prereleases=pre)
        )
    want = [ss.contains(x, prereleases=pre, installed=True) for x in items]
    assert rp.contains_batch(ss, items, prereleases=pre, installed=True) == want


def test_single_specifier() -> None:
    s = Specifier(">=1.2")
    assert rp.contains_batch(s, VERSIONS[:2000]) == [s.contains(v) for v in VERSIONS[:2000]]
    assert rp.filter_batch(s, VERSIONS[:2000]) == list(s.filter(VERSIONS[:2000]))


def test_arrow() -> None:
    pa = pytest.importorskip("pyarrow")
    arr = pa.array(["1.0", None, "2.0a1", "foo", "3.0"])
    assert rp.parse_batch(arr, errors="none") == [
        Version("1.0"),
        None,
        Version("2.0a1"),
        None,
        Version("3.0"),
    ]
    col = rp.contains_batch(">=1.5", arr)
    assert col.to_pylist() == [False, None, True, False, True]
    assert rp.filter_batch(">=1.5", arr) == ["3.0"]
    assert rp.sort_versions(pa.array(["2", "1.0", "1.5"])) == ["1.0", "1.5", "2"]


def test_errors() -> None:
    with pytest.raises(TypeError):
        rp.parse_batch("1.0")
    with pytest.raises(ValueError, match="errors must be"):
        rp.parse_batch(["1.0"], errors="ignore")  # type: ignore[call-overload]
    with pytest.raises(ValueError, match="threads"):
        rp.parse_batch(["1.0"], threads=0)
