"""pyroid.difflib: batches, fallbacks, threads, pyroid.patch("difflib") and pyroid run.

Parity with the standard library is in test_difflib_parity.py (differential) and
test_difflib_cpython.py (CPython's own tests).
"""

from __future__ import annotations

import _thread
import collections
import difflib
import functools
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
import unittest
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import pyroid
import pyroid.difflib as pd
from pyroid import integrations
from pyroid._pyroid import _testing
from pyroid._pyroid import difflib as native

WORDS = ["apple", "ape", "peach", "puppy", "applet", "maple", "Apple", "appel", "", "ÄPfel"]


def ratios(a: list[str], b: list[str], **kw: Any) -> list[float]:
    junk = kw.pop("isjunk", None)
    return [difflib.SequenceMatcher(junk, x, y, **kw).ratio() for x, y in zip(a, b, strict=True)]


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------


def test_ratio_batch_kinds() -> None:
    a = WORDS
    b = list(reversed(WORDS))
    assert pd.ratio_batch(a, b) == ratios(a, b)
    assert pd.ratio_batch(tuple(a), tuple(b)) == ratios(a, b)
    assert pd.ratio_batch("appel", b) == ratios(["appel"] * len(b), b)
    assert pd.ratio_batch(a, "appel") == ratios(a, ["appel"] * len(a))
    assert pd.ratio_batch("ab", "abc") == [difflib.SequenceMatcher(None, "ab", "abc").ratio()]
    assert pd.ratio_batch([], []) == []
    space = " ".__eq__
    long_b = ["a b " * 80 + w for w in b]  # past 200 characters: autojunk's popular characters
    for autojunk in (True, False):
        assert pd.ratio_batch(a, long_b, isjunk=space, autojunk=autojunk) == ratios(
            a, long_b, isjunk=space, autojunk=autojunk
        )


def test_ratio_batch_arrow() -> None:
    a = _testing.column(["apple", None, "peach", "x"], chunks=2)
    b = _testing.column(["appel", "ape", None, "y"], large=True)
    out = pd.ratio_batch(a, b)
    assert out == [ratios(["apple"], ["appel"])[0], None, None, 0.0]
    assert pd.ratio_batch(a, "apple")[1] is None


def test_ratio_batch_pyarrow(pa: Any) -> None:
    a = pa.chunked_array([["apple", "ape"], [None, "peach"]])
    b = pa.array(["appel", "apex", "x", None], type=pa.string_view())
    assert pd.ratio_batch(a, b) == [*ratios(["apple", "ape"], ["appel", "apex"]), None, None]


def test_ratio_batch_errors() -> None:
    with pytest.raises(ValueError, match="same length"):
        pd.ratio_batch(["a", "b"], ["a"])
    with pytest.raises(pyroid.InputError):
        pd.ratio_batch([1], ["a"])  # type: ignore[list-item]
    with pytest.raises(UnicodeEncodeError):
        pd.ratio_batch(["\ud800"], ["a"])
    with pytest.raises(ZeroDivisionError):
        pd.ratio_batch(["a"], ["b"], isjunk=lambda c: 1 / 0)


def test_ratio_batch_threads() -> None:
    a = [f"name {i} street {i % 97}" for i in range(20_000)]
    b = [f"Name {i * 7 % 1000} Street {i % 89}" for i in range(20_000)]
    expected = pd.ratio_batch(a, b, threads=1)
    assert expected[:500] == ratios(a[:500], b[:500])
    for threads in (2, 7, None):
        assert pd.ratio_batch(a, b, threads=threads) == expected
    with pytest.raises(ValueError, match="threads"):
        pd.ratio_batch(a, b, threads=0)


def test_close_matches_batch() -> None:
    words = ["appel", "pech", "", "zzz", "Apple"]
    for kind in (list, tuple):
        possibilities = kind(WORDS)
        out = pd.close_matches_batch(words, possibilities)
        assert out == [difflib.get_close_matches(w, WORDS) for w in words]
        flat = [x for found in out if found for x in found]
        assert all(any(x is p for p in possibilities) for x in flat)  # the caller's own objects
    for n, cutoff in ((1, 0.0), (5, 0.3), (10**30, 0.9), (2, 1.0), (True, 0)):
        assert pd.close_matches_batch(words, WORDS, n, cutoff) == [
            difflib.get_close_matches(w, WORDS, n, cutoff) for w in words
        ]
    many = [f"{w}{i}" for i in range(200) for w in ("appel", "peach")]
    assert pd.close_matches_batch(many, WORDS, threads=3) == [
        difflib.get_close_matches(w, WORDS) for w in many
    ]
    col = _testing.column(["appel", None])
    poss = _testing.column(["apple", None, "ape"], chunks=2)
    assert pd.close_matches_batch(col, poss) == [["apple", "ape"], None]
    assert pd.close_matches_batch("appel", WORDS) == [difflib.get_close_matches("appel", WORDS)]


def test_close_matches_batch_errors() -> None:
    with pytest.raises(ValueError, match=r"n must be > 0: 0"):
        pd.close_matches_batch(["a"], ["a"], 0)
    with pytest.raises(ValueError, match=r"cutoff must be in \[0.0, 1.0\]: 1.5"):
        pd.close_matches_batch(["a"], ["a"], 3, 1.5)
    with pytest.raises(pyroid.InputError, match="possibilities"):
        pd.close_matches_batch(["a"], "abc")
    with pytest.raises(pyroid.InputError):
        pd.close_matches_batch(["a"], [b"a"])  # type: ignore[list-item]


def test_get_close_matches_threads() -> None:
    possibilities = [f"{w}{i % 13}" for i in range(60_000) for w in ("apple", "maple")]
    expected = difflib.get_close_matches("apple7", possibilities, 5, 0.5)
    for threads in (1, 4, None):
        assert pd.get_close_matches("apple7", possibilities, 5, 0.5, threads=threads) == expected


@pytest.mark.thread_unsafe(reason="interrupts the main thread")
def test_batches_stop_on_ctrl_c() -> None:
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("KeyboardInterrupt goes to the main thread")
    a = ["a b c d e f g " * 40] * 400_000
    timer = threading.Timer(0.3, _thread.interrupt_main)
    start = time.monotonic()
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            pd.ratio_batch(a, "a b c d e f g h " * 40, threads=1)
    finally:
        timer.cancel()
    assert time.monotonic() - start < 30


# ---------------------------------------------------------------------------
# The module
# ---------------------------------------------------------------------------


def test_same_objects_and_subclasses() -> None:
    for name in ("Match", "IS_LINE_JUNK", "IS_CHARACTER_JUNK", "restore", "diff_bytes"):
        assert getattr(pd, name) is getattr(difflib, name)
    for name in ("SequenceMatcher", "Differ", "HtmlDiff"):
        cls = getattr(pd, name)
        assert issubclass(cls, getattr(difflib, name))
        assert cls.__module__ == "pyroid.difflib"
    assert pd.SUPPORTED_VERSIONS[0] == (3, 11)
    assert set(difflib.__all__) <= set(pd.__all__)
    assert pd.SequenceMatcher.__doc__ == difflib.SequenceMatcher.__doc__
    assert pd.SequenceMatcher[str] is not None  # __class_getitem__, as difflib's


@pytest.mark.thread_unsafe(reason="FallbackWarning is once per process")
def test_fallback_warns_once() -> None:
    warned: set[str] = pd._WARNED  # type: ignore[attr-defined]
    warned.discard("SequenceMatcher")
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            s = pd.SequenceMatcher(None, [(1, 2), (3,)], [(1, 2)])
            assert s.ratio() == difflib.SequenceMatcher(None, [(1, 2), (3,)], [(1, 2)]).ratio()
            pd.SequenceMatcher(None, [1.5], [1.5]).ratio()
        assert [type(w.message) for w in rec] == [pyroid.FallbackWarning]
        assert "standard library's difflib" in str(rec[0].message)
        assert rec[0].filename == __file__
    finally:
        warned.add("SequenceMatcher")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pd.SequenceMatcher(None, "abc", ["a", "b"]).ratio()  # native: no warning
        list(pd.ndiff(["a\n"], ["b\n"], charjunk=lambda c: c == " "))  # Python code, native matcher


class _Name(str):
    __slots__ = ()


class _Loud(str):
    def __eq__(self, other: object) -> bool:
        return str.__eq__(self.lower(), str(other).lower())

    __hash__ = str.__hash__


class _Row(tuple):  # type: ignore[type-arg]
    __slots__ = ()


def _matcher_state(m: Any) -> tuple[Any, ...]:
    return (
        m.get_matching_blocks(),
        m.get_opcodes(),
        m.ratio(),
        m.quick_ratio(),
        list(m.b2j.items()),
        m.bjunk,
        m.bpopular,
    )


def _native_pairs() -> list[tuple[Any, Any]]:
    import enum

    class Color(enum.IntEnum):
        RED = 1
        BLUE = 2

    return [
        # ints outside the signed 64-bit range, next to ones inside it
        ([2**64, 1, 2, -(2**63) - 1, 2**63], [1, 2**64, 2**63, 5]),
        ([10**40, -(10**40), 3], [3, 10**40]),
        # str with bytes (no python -b)
        (["a", b"a", "b"], ["b", b"a"]),
        ("ab", [b"a", "a", "b"]),
        ([b"x", "x"], "xx"),
        # subclasses that keep their base's comparisons and indexing
        (_Name("private Thread currentThread;"), _Name("private volatile Thread currentThread;")),
        ([_Name("a"), "b", _Name("c")], ["a", _Name("b")]),
        ([Color.RED, 2, Color.BLUE], [1, 2]),
        (_Row((1, 2, 3)), _Row((2, 3))),
        (bytearray(b"abcd"), b"bcd"),
    ]


def test_native_elements_and_subclasses() -> None:
    # Big ints, str mixed with bytes and plain subclasses run natively, with difflib's results.
    for a, b in _native_pairs():
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = _matcher_state(pd.SequenceMatcher(None, a, b))
        assert got == _matcher_state(difflib.SequenceMatcher(None, a, b)), (a, b)
        assert "_pyroid_b" in vars(pd.SequenceMatcher(None, a, b))


@pytest.mark.thread_unsafe(reason="FallbackWarning is once per process")
@pytest.mark.parametrize(
    ("a", "b", "why"),
    [
        ([1.5, 2], [2], "a list with an element of type 'float'"),
        ([(1, 2)], ((1, 2),), "a tuple with an element of type 'tuple'"),
        ("abc", [_Loud("A")], "a list with an element of type '_Loud' (it overrides __eq__)"),
        ([1], collections.deque([1]), "a sequence of type 'deque'"),
    ],
)
def test_fallback_warning_names_the_trigger(a: Any, b: Any, why: str) -> None:
    warned: set[str] = pd._WARNED  # type: ignore[attr-defined]
    warned.discard("SequenceMatcher")
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            got = pd.SequenceMatcher(None, a, b).ratio()
        assert got == difflib.SequenceMatcher(None, a, b).ratio()
    finally:
        warned.add("SequenceMatcher")
    assert [type(w.message) for w in rec] == [pyroid.FallbackWarning]
    message = str(rec[0].message)
    assert message.startswith(f"pyroid.difflib.SequenceMatcher: {why} runs the standard"), message
    assert "int (bool too) and bytes" in message
    assert rec[0].filename == __file__


def test_str_and_bytes_under_python_b(run_python: Any) -> None:
    # Under -b, comparing str with bytes warns: difflib's code runs, so the warnings are its own.
    code = """
        import difflib, warnings
        import pyroid.difflib as pd

        def run(module):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                r = module.SequenceMatcher(None, ["a", b"a", "b"], ["b", b"a"]).ratio()
            return r, rec

        (rp, wp), (rd, wd) = run(pd), run(difflib)
        assert rp == rd
        bw = lambda rec: [str(w.message) for w in rec if w.category is BytesWarning]
        assert bw(wp) == bw(wd) and bw(wd), (wp, wd)
        print([str(w.message) for w in wp if w.category.__name__ == "FallbackWarning"])
    """
    proc = run_python(code, args=["-b"])
    assert proc.returncode == 0, proc.stderr
    assert "a mix of str and bytes elements under python -b" in proc.stdout, proc.stdout


@pytest.mark.thread_unsafe(reason="FallbackWarning is once per process")
def test_get_close_matches_over_str_subclasses() -> None:
    # Native (no FallbackWarning), returning the possibilities' own objects.
    names = [_Name(n) for n in ("apple", "ape", "peach", "puppy", "applet", "maple")]
    warned: set[str] = pd._WARNED  # type: ignore[attr-defined]
    warned.discard("SequenceMatcher")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = pd.get_close_matches(_Name("appel"), names, n=4, cutoff=0.3)
            assert pd.get_close_matches("appel", names, threads=2) == difflib.get_close_matches(
                "appel", names
            )
    finally:
        warned.add("SequenceMatcher")
    want = difflib.get_close_matches(_Name("appel"), names, n=4, cutoff=0.3)
    assert got == want
    assert [id(x) for x in got] == [id(x) for x in want]
    loud = [_Loud("APPLE"), _Loud("apple")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.FallbackWarning)
        assert pd.get_close_matches("apple", loud) == difflib.get_close_matches("apple", loud)


def _vowels_are_junk(base: type) -> type:
    class VowelsAreJunk(base):  # type: ignore[misc]
        calls = 0

        def _SequenceMatcher__chain_b(self) -> None:  # noqa: N802 - the hook's mangled name
            type(self).calls += 1
            super()._SequenceMatcher__chain_b()
            for v in "aeiou":
                if self.b2j.pop(v, None) is not None:
                    self.bjunk.add(v)

    return VowelsAreJunk


def test_chain_b_override_is_called() -> None:
    # A subclass's __chain_b post-processes b2j: called once per set_seq2, as by difflib.
    a, b = "private Thread currentThread;", "private volatile Thread currentThread;"

    def run(base: type) -> tuple[Any, ...]:
        cls: Any = _vowels_are_junk(base)
        m = cls(None, a, b)
        first = (sorted(m.bjunk), m.get_matching_blocks(), m.ratio(), m.get_opcodes())
        m.set_seq2(b.upper() + "aeiou")
        return cls.calls, first, sorted(m.bjunk), m.ratio(), dict(m.b2j)

    want = run(difflib.SequenceMatcher)
    assert want[0] == 2
    assert run(pd.SequenceMatcher) == want
    ctx = _quiet_patch()
    try:
        with integrations.patching("difflib"):
            assert run(difflib.SequenceMatcher) == want
    finally:
        ctx.__exit__(None, None, None)


@pytest.mark.thread_unsafe(reason="HtmlDiff's anchor counter is global")
def test_numeric_arguments_as_difflib() -> None:
    def outcome(f: Any) -> Any:
        difflib.HtmlDiff._default_prefix = 0  # type: ignore[attr-defined]
        try:
            return f()
        except Exception as e:
            return type(e), str(e)

    cases: list[Callable[[Any], Any]] = [
        lambda m: m.HtmlDiff(wrapcolumn=10**20).make_table(["abcdef\n"], ["abcxef\n"]),
        lambda m: m.HtmlDiff(wrapcolumn=2**63).make_file(["a\n"], ["b\n"]),
        lambda m: m.HtmlDiff(tabsize=10**20).make_table(["a\tb\n"], ["a\tc\n"]),
        lambda m: m.HtmlDiff(tabsize=2**31).make_table(["a\tb\n"], ["a\tc\n"]),
        lambda m: m.HtmlDiff(tabsize=-(2**31)).make_table(["a\tb\n"], ["a\tc\n"]),
        lambda m: m.HtmlDiff().make_table(["a\n"], ["b\n"], context=True, numlines=10**20),
        lambda m: m.HtmlDiff().make_table(["a\n"], ["b\n"], numlines=10**20),
        lambda m: list(m._mdiff(["a\n"], ["b\n"], context=10**20)),
        lambda m: list(m._mdiff(["a\n"], ["b\n"], context=sys.maxsize)),
    ]
    for i, case in enumerate(cases):
        assert outcome(functools.partial(case, pd)) == outcome(functools.partial(case, difflib)), i
    # close_matches_batch gives what its loop gives (for a non-integer n, heapq decides, and 3.15
    # takes some that 3.14 rejects).
    poss = ["apple", "ape", "peach"]
    ns: list[Any] = [2.5, 3.0, 1.0, 0.5, True, 2, 10**30]
    for qs in (["appel", "pech"], ["appel"], ["zzz"], [], "appel"):
        for n in ns:
            words = [qs] if isinstance(qs, str) else qs
            want = outcome(lambda: [difflib.get_close_matches(q, poss, n, 0.6) for q in words])  # noqa: B023
            assert outcome(lambda: pd.close_matches_batch(qs, poss, n, 0.6)) == want, (qs, n)  # noqa: B023
    with pytest.raises(TypeError):
        pd.close_matches_batch(["appel"], _testing.column(poss, chunks=1), 2.5)  # type: ignore[arg-type]


@pytest.mark.thread_unsafe(reason="HtmlDiff's anchor counter is global")
def test_html_tables_of_both_modules_get_distinct_anchors() -> None:
    import re

    def ids(table: str) -> set[str]:
        return set(re.findall(r'id="([^"]+)"', table))

    tables = [m.HtmlDiff().make_table(["a\n"], ["b\n"]) for m in (pd, difflib, pd, difflib)]
    seen: set[str] = set()
    for t in tables:
        assert not ids(t) & seen
        seen |= ids(t)
    for owner in (pd, difflib):
        owner.HtmlDiff._default_prefix = 0  # as CPython's tests reset it
        assert 'id="difflib_chg_to0__0"' in pd.HtmlDiff().make_table(["a\n"], ["b\n"])
        assert 'id="difflib_chg_to1__0"' in difflib.HtmlDiff().make_table(["a\n"], ["b\n"])


def test_runtime_docstrings_match_the_stub() -> None:
    # help() shows the stub's documentation for pyroid's own functions.
    import ast
    import inspect

    stub = ast.parse(Path(pd.__file__).with_suffix(".pyi").read_text(encoding="utf-8"))
    docs: dict[str, str] = {}
    for node in stub.body:
        if isinstance(node, ast.FunctionDef) and (doc := ast.get_docstring(node)):
            docs.setdefault(node.name, doc)
    for name in ("ratio_batch", "close_matches_batch"):
        assert inspect.getdoc(getattr(pd, name)) == docs[name], name


def test_racing_set_seq_errors_name_no_internals() -> None:
    # Racing set_seq* on one matcher is unsupported (as in difflib), but nothing pyroid keeps
    # inside the matcher shows up in what it raises.
    import random

    m: Any = pd.SequenceMatcher(None, "abc" * 50, "abe" * 50)
    errors: list[str] = []
    stop = time.monotonic() + 1.5

    def race(seed: int) -> None:
        r = random.Random(seed)
        while time.monotonic() < stop:
            k = r.randrange(5)
            text = "".join(r.choice("abcd") for _ in range(r.randint(0, 200)))
            try:
                if k == 0:
                    m.set_seq1(text)
                elif k == 1:
                    m.set_seq2(text)
                elif k == 2:
                    m.ratio()
                elif k == 3:
                    m.get_opcodes()
                else:
                    assert isinstance(m.b2j, dict)
            except Exception as e:  # collected
                errors.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=race, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    lazy = ("b2j", "bjunk", "bpopular", "fullbcount", "matching_blocks")
    internal = [e for e in errors if "_pyroid" in e or any(f"'{n}'" in e for n in lazy)]
    assert not internal, internal[:5]


@pytest.mark.thread_unsafe(reason="tracemalloc is process-wide")
def test_diff_lines_are_made_as_they_are_read() -> None:
    # ndiff, unified_diff and context_diff keep the diff as small ops and make each line at
    # next(): the first line does not cost a list of every line (13 MB here).
    import tracemalloc

    a = [f"{i:08d} " + "x" * 70 + "\n" for i in range(100_000)]
    b = [*a[:10], "new\n", *a[10:]]
    for name, kw in (("ndiff", {}), ("unified_diff", {"n": 10**6}), ("context_diff", {"n": 10**6})):
        tracemalloc.start()
        try:
            lines = getattr(pd, name)(a, b, **kw)
            first = next(lines)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 5_000_000, (name, peak)
        assert [first, *lines] == list(getattr(difflib, name)(a, b, **kw)), name


def test_large_inputs() -> None:
    a = [f"line {i}\n" for i in range(100_000)]
    b = [*a[:5000], "new\n", *a[5000:90000], *a[90001:]]
    start = time.monotonic()
    out = list(pd.unified_diff(a, b))
    assert out == list(difflib.unified_diff(a, b))
    assert time.monotonic() - start < 60
    popular = "ab" * 5000
    assert (
        pd.SequenceMatcher(None, popular, popular[1:]).ratio()
        == difflib.SequenceMatcher(None, popular, popular[1:]).ratio()
    )


def test_unknown_python_version() -> None:
    code = textwrap.dedent("""
        import sys, warnings, difflib
        sys.version_info = (3, 99, 0, "final", 0)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            import pyroid.difflib as pd
        assert pd.SequenceMatcher is difflib.SequenceMatcher
        assert pd.unified_diff is difflib.unified_diff
        assert [type(w.message).__name__ for w in rec] == ["FallbackWarning"], rec
        assert "3.11 to 3.15" in str(rec[0].message)
        # The stub's threads= is accepted; patch() patches nothing and says so.
        words = ["apple", "ape", "peach"]
        assert pd.get_close_matches("appel", words, threads=4) == ["apple", "ape"]
        import pyroid
        before = dict(vars(difflib.SequenceMatcher))
        functions = difflib.unified_diff, difflib.get_close_matches
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            pyroid.patch("difflib")
        assert dict(vars(difflib.SequenceMatcher)) == before
        assert (difflib.unified_diff, difflib.get_close_matches) == functions
        messages = [str(w.message) for w in rec]
        assert len(messages) == 1 and "nothing was patched" in messages[0], messages
        assert "drop-in" not in messages[0], messages
        pyroid.unpatch("difflib")
        print("ok")
    """)
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert proc.stdout.strip() == "ok", proc.stderr


# ---------------------------------------------------------------------------
# Threads (free-threaded builds run these without the GIL)
# ---------------------------------------------------------------------------


def test_one_matcher_shared_by_threads() -> None:
    a, b = "private Thread currentThread;" * 20, "private volatile Thread currentThread;" * 20
    expected = difflib.SequenceMatcher(None, a, b)
    want = (expected.ratio(), expected.get_opcodes(), expected.quick_ratio())
    s = pd.SequenceMatcher(None, a, b)
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def work() -> None:
        try:
            barrier.wait(timeout=60)
            for _ in range(200):
                assert (s.ratio(), s.get_opcodes(), s.quick_ratio()) == want
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not errors, errors[0]


def test_lists_changed_by_another_thread() -> None:
    # The lists are read in one step: whatever state is read, no crash and a valid diff.
    lines = [f"{i}\n" for i in range(2000)]
    stop = threading.Event()

    def mutate() -> None:
        while not stop.is_set():
            lines.append("x\n")
            del lines[-1]
            lines.insert(0, "y\n")
            del lines[0]

    t = threading.Thread(target=mutate)
    t.start()
    try:
        for _ in range(50):
            out = list(pd.unified_diff(lines, lines[::2]))
            assert all(isinstance(x, str) for x in out)
            assert pd.SequenceMatcher(None, lines, lines[:100]).ratio() >= 0.0
            pd.get_close_matches("10\n", lines)
    finally:
        stop.set()
        t.join(timeout=60)


# ---------------------------------------------------------------------------
# pyroid.patch("difflib")
# ---------------------------------------------------------------------------

pytestmark_patch = pytest.mark.thread_unsafe(reason="patches the standard library")


def _quiet_patch() -> Any:
    ctx = warnings.catch_warnings()
    ctx.__enter__()
    warnings.simplefilter("ignore", integrations.PatchWarning)
    return ctx


@pytestmark_patch
def test_patch_in_place_and_undo() -> None:
    before = {cls: dict(vars(cls)) for cls in (difflib.SequenceMatcher, difflib.Differ)}
    functions = {n: getattr(difflib, n) for n in ("unified_diff", "context_diff", "_mdiff")}
    original_class = difflib.SequenceMatcher
    made_before = difflib.SequenceMatcher(None, "abcd", "bcde")
    ctx = _quiet_patch()
    try:
        with integrations.patching("difflib"):
            assert integrations.patched() == ("difflib",)
            assert difflib.SequenceMatcher is original_class  # in place: the same class
            assert difflib.unified_diff is pd.unified_diff
            s = difflib.SequenceMatcher(None, "abcd", "bcde")
            assert "_pyroid_b" in vars(s)
            assert s.ratio() == 0.75
            assert made_before.ratio() == 0.75  # made unpatched: the standard library's state
            assert made_before.get_opcodes() == s.get_opcodes()
    finally:
        ctx.__exit__(None, None, None)
    assert {cls: dict(vars(cls)) for cls in before} == before
    assert {n: getattr(difflib, n) for n in functions} == functions


@pytestmark_patch
def test_matchers_from_under_the_patch_work_after_unpatch() -> None:
    # A matcher made or used under the patch gets CPython's state when the patch ends.
    class Sub(difflib.SequenceMatcher):  # type: ignore[type-arg]
        pass

    def steps(sm: Any, step: str) -> Any:
        if step == "ratio":
            sm.ratio()
        elif step == "quick_ratio":
            sm.quick_ratio()
        elif step == "blocks":
            sm.get_matching_blocks()
        elif step == "b2j":
            assert isinstance(sm.b2j, dict)
        return sm

    kinds = [
        (cls, step, a, b)
        for cls in (difflib.SequenceMatcher, Sub)
        for step in ("", "ratio", "quick_ratio", "blocks", "b2j")
        for a, b in (("abcd", "bcde"), ([1, 2, 3, 4], [2, 3, 4, 5]), ("x" * 300, "xy" * 150))
    ]
    ctx = _quiet_patch()
    try:
        with integrations.patching("difflib"):
            made = [steps(cls(None, a, b), step) for cls, step, a, b in kinds]
            own = pd.SequenceMatcher(None, "abcd", "bcde")
            assert all("_pyroid_b" in vars(m) for m in made)
    finally:
        ctx.__exit__(None, None, None)
    assert "_pyroid_b" in vars(own)  # pyroid.difflib's class keeps its methods and state
    assert own.ratio() == 0.75
    for m, (cls, step, a, b) in zip(made, kinds, strict=True):
        twin = steps(cls(None, a, b), step)  # unpatched: the standard library's state
        assert vars(m) == vars(twin), (step, a)
        assert m.ratio() == twin.ratio()
        assert m.get_opcodes() == twin.get_opcodes()
        m.set_seq1(b)
        twin.set_seq1(b)
        assert m.ratio() == twin.ratio() == 1.0
        assert m.quick_ratio() == twin.quick_ratio()


@pytestmark_patch
def test_matchers_of_ended_threads_work_after_unpatch() -> None:
    # Each thread records the matchers it gives native state; a thread that ends hands its
    # records to the next one that starts, and unpatch still finds them.
    def make(out: list[Any], n: int) -> None:
        out.extend(difflib.SequenceMatcher(None, f"abcd{i}", "bcde") for i in range(n))

    first: list[Any] = []
    second: list[Any] = []
    ctx = _quiet_patch()
    try:
        with integrations.patching("difflib"):
            for out, n in ((first, 3), (second, 200)):  # 200: past the first pruning
                t = threading.Thread(target=make, args=(out, n))
                t.start()
                t.join()
            del second[:190]
            assert all("_pyroid_b" in vars(m) for m in first + second)
    finally:
        ctx.__exit__(None, None, None)
    for m in first + second:
        twin = difflib.SequenceMatcher(None, m.a, m.b)
        assert vars(m) == vars(twin)
        assert m.ratio() == twin.ratio()


@pytestmark_patch
def test_unpatch_while_another_thread_sets_up_a_matcher() -> None:
    # unpatch does not wait for a matcher another thread is setting up (here, blocked in
    # isjunk): that call gives it difflib's own state when it ends.
    called = threading.Event()
    release = threading.Event()
    made: list[Any] = []

    def junk(ch: str) -> bool:
        called.set()
        assert release.wait(60)
        return ch == " "

    def build() -> None:
        made.append(difflib.SequenceMatcher(junk, "a b c d", "a  b x d"))

    ctx = _quiet_patch()
    try:
        integrations.patch("difflib")
        t = threading.Thread(target=build)
        t.start()
        try:
            assert called.wait(60)
            start = time.monotonic()
            integrations.unpatch("difflib")
            assert time.monotonic() - start < 5
        finally:
            release.set()
            t.join(60)
    finally:
        integrations.unpatch()
        ctx.__exit__(None, None, None)
    (m,) = made
    twin = difflib.SequenceMatcher(lambda ch: ch == " ", "a b c d", "a  b x d")
    assert {k: v for k, v in vars(m).items() if k != "isjunk"} == {
        k: v for k, v in vars(twin).items() if k != "isjunk"
    }
    assert (m.ratio(), m.get_opcodes()) == (twin.ratio(), twin.get_opcodes())


_TOGGLE_STRESS = r"""
import difflib, faulthandler, json, os, sys, threading, time, warnings

warnings.simplefilter("ignore")
import pyroid

togglers, users, toggles, limit = (int(x) for x in sys.argv[1:5])
faulthandler.dump_traceback_later(limit + 60, exit=True)
sys.setswitchinterval(1e-4)  # with the GIL: hand it to the togglers often


def rss_mb():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (OSError, ValueError, AttributeError):
        return None


class Sub(difflib.SequenceMatcher):
    pass


a, b = "private Thread currentThread;" * 4, "private volatile Thread currentThread;" * 4
la = [f"line {i}\n" for i in range(60)]
lb = [*la[:20], "new\n", *la[21:]]
want = {}
for cls in (difflib.SequenceMatcher, Sub):
    m = cls(None, a, b)
    want[cls] = (m.ratio(), m.get_opcodes(), m.quick_ratio())
want_diff = list(difflib.unified_diff(la, lb))
want_ab, want_aa = want[difflib.SequenceMatcher][0], 1.0
before = dict(vars(difflib.SequenceMatcher))

stop = threading.Event()
errors, calls, done = [], [0] * users, [0] * togglers


def user(k):
    keep = difflib.SequenceMatcher(None, a, b)  # made once, used across every toggle
    i = 0
    while not stop.is_set():
        i += 1
        try:
            cls = Sub if k % 2 else difflib.SequenceMatcher
            m = cls(None, a, b)
            got = (m.ratio(), m.get_opcodes(), m.quick_ratio())
            if got != want[cls]:
                errors.append(f"wrong {cls.__name__}: {got[0]}")
            keep.set_seq2(a if i % 2 else b)
            if keep.ratio() != (want_aa if i % 2 else want_ab):
                errors.append("wrong ratio after set_seq2")
            if i % 8 == 0 and list(difflib.unified_diff(la, lb)) != want_diff:
                errors.append("wrong unified_diff")
            # Garbage only the cycle collector frees, as PyYAML's loaders leave.
            node = {"pad": [[j] for j in range(20)]}
            node["self"] = node
            calls[k] += 1
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")


def toggler(k):
    try:
        for _ in range(toggles):
            pyroid.patch("difflib")
            pyroid.unpatch("difflib")
            done[k] += 1
    except Exception as e:
        errors.append(f"toggler: {type(e).__name__}: {e}")


threads = [threading.Thread(target=user, args=(k,)) for k in range(users)]
threads += [threading.Thread(target=toggler, args=(k,)) for k in range(togglers)]
start = time.monotonic()
base = rss_mb()
peak = base
for t in threads:
    t.start()
while any(t.is_alive() for t in threads[users:]) and time.monotonic() - start < limit:
    time.sleep(0.05)
    now = rss_mb()
    if now is not None:
        peak = max(peak, now)
elapsed = time.monotonic() - start
stop.set()
finished = not any(t.is_alive() for t in threads[users:])
for t in threads[:users]:
    t.join()
restored = dict(vars(difflib.SequenceMatcher)) == before
print(json.dumps({
    "finished": finished, "elapsed": elapsed, "toggles": sum(done), "calls": sum(calls),
    "errors": len(errors), "first_errors": sorted(set(errors))[:5], "restored": restored,
    "rss_base": base, "rss_peak": peak,
}))
sys.stdout.flush()
os._exit(0)  # a toggler still running (the old bug) must not keep the process alive
"""


@pytestmark_patch
def test_toggling_the_patch_while_threads_diff() -> None:
    # 4 threads patch and unpatch difflib 300 times each (100 with the GIL) while 8 threads use
    # it (matchers made fresh, one kept across toggles and re-pointed with set_seq2, a
    # subclass, unified_diff) and leave cyclic garbage. Every result is difflib's, no call
    # fails, and time and memory stay bounded: unpatch neither walks the heap (which kept every
    # object alive and starved the cycle collector) nor misses a matcher another thread uses.
    toggles = 100 if getattr(sys, "_is_gil_enabled", lambda: True)() else 300
    proc = subprocess.run(
        [sys.executable, "-c", _TOGGLE_STRESS, "4", "8", str(toggles), "60"],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["finished"], json.dumps(out)
    assert out["toggles"] == 4 * toggles, json.dumps(out)
    assert out["errors"] == 0, json.dumps(out)
    assert out["calls"] > 0, json.dumps(out)
    assert out["restored"], json.dumps(out)
    if out["rss_base"] is not None:
        assert out["rss_peak"] - out["rss_base"] < 400, json.dumps(out)


@pytestmark_patch
def test_unpatch_puts_difflib_back_when_a_matcher_cannot_be_settled() -> None:
    # unpatch gives the matchers made under the patch difflib's own state before it puts
    # difflib's methods back. If one cannot be settled (here a subclass's `b` raises), the
    # others still are, every attribute still goes back, and then the error is raised.
    class Flaky(difflib.SequenceMatcher):  # type: ignore[type-arg]
        broken = False

        @property
        def b(self) -> Any:
            if Flaky.broken:
                raise RuntimeError("b is broken")
            return self.__dict__["_b"]

        @b.setter
        def b(self, value: Any) -> None:
            self.__dict__["_b"] = value

    names = ("SequenceMatcher", "Differ", "HtmlDiff")
    before = {n: dict(vars(getattr(difflib, n))) for n in names}
    functions = {n: getattr(difflib, n) for n in ("unified_diff", "context_diff", "_mdiff")}
    ctx = _quiet_patch()
    try:
        integrations.patch("difflib")
        flaky = Flaky(None, "abcd", "bcde")
        plain = difflib.SequenceMatcher(None, "abcd", "bcde")
        assert "_pyroid_b" in vars(flaky)
        assert "_pyroid_b" in vars(plain)
        Flaky.broken = True
        with pytest.raises(RuntimeError, match="b is broken"):
            integrations.unpatch("difflib")
    finally:
        Flaky.broken = False
        integrations.unpatch()
        ctx.__exit__(None, None, None)
    assert {n: dict(vars(getattr(difflib, n))) for n in names} == before
    assert {n: getattr(difflib, n) for n in functions} == functions
    assert integrations.patched() == ()
    assert vars(plain) == vars(difflib.SequenceMatcher(None, "abcd", "bcde"))
    assert plain.ratio() == 0.75


def test_a_failed_patch_step_undoes_the_others() -> None:
    class Owner:
        pass

    missing = object()
    swaps = ((Owner, "x", 1), (Owner, "y", 2), (int, "z", 3))  # int's attributes are read-only
    with pytest.raises(TypeError, match="immutable type"):
        native._sm_patch(swaps, missing)
    assert not hasattr(Owner, "x")
    assert not hasattr(Owner, "y")


_CTRL_C = r"""
import _thread, difflib, json, random, signal, sys, threading, time, warnings

warnings.simplefilter("ignore")
from pyroid import integrations
from pyroid.integrations import _patch

seconds = float(sys.argv[1])
NAMES = ("get_close_matches", "unified_diff", "context_diff", "_mdiff")
CLASSES = ("SequenceMatcher", "Differ", "HtmlDiff")


def snapshot():
    classes = {n: dict(vars(getattr(difflib, n))) for n in CLASSES}
    return classes, {n: getattr(difflib, n) for n in NAMES}


plain = snapshot()
integrations.patch("difflib")
patched = snapshot()
integrations.unpatch("difflib")
assert snapshot() == plain


def put_back():
    for n, d in plain[0].items():
        cls = getattr(difflib, n)
        for k in set(vars(cls)) - set(d):
            delattr(cls, k)
        for k, v in d.items():
            if vars(cls).get(k, None) is not v:
                setattr(cls, k, v)
    for k, v in plain[1].items():
        setattr(difflib, k, v)
    _patch._ACTIVE.pop("difflib", None)


class CtrlC(KeyboardInterrupt):
    pass


armed = False


def on_sigint(signum, frame):
    # Runs on the main thread, between two bytecodes: raises only while it toggles.
    if armed:
        raise CtrlC


signal.signal(signal.SIGINT, on_sigint)
stop = threading.Event()


def interrupter():
    rng = random.Random(1)
    while not stop.is_set():
        _thread.interrupt_main()
        time.sleep(rng.uniform(0, 3e-4))


counts = {"toggles": 0, "interrupted": 0, "half": 0, "untracked": 0, "unrepaired": 0}
counts["first_half"] = None
t = threading.Thread(target=interrupter, daemon=True)
t.start()
end = time.monotonic() + seconds
while time.monotonic() < end:
    try:
        armed = True
        integrations.patch("difflib")
        integrations.unpatch("difflib")
        armed = False
    except CtrlC:
        armed = False
        counts["interrupted"] += 1
    counts["toggles"] += 1
    now = snapshot()
    tracked = "difflib" in integrations.patched()
    if now == patched and tracked:  # interrupted between patch() and unpatch()
        integrations.unpatch("difflib")
    elif now != plain or tracked:
        if now in (patched, plain):
            # Interrupted in patch()'s or unpatch()'s own bookkeeping, around difflib's steps:
            # the next patch() takes over what is in place, and unpatch() puts difflib back.
            counts["untracked"] += 1
        else:
            counts["half"] += 1
            if counts["first_half"] is None:
                counts["first_half"] = sorted(
                    f"{n}.{k}" for n, d in now[0].items() for k in set(d) ^ set(plain[0][n])
                )
            put_back()
        integrations.patch("difflib")
        integrations.unpatch("difflib")
        if snapshot() != plain:
            counts["unrepaired"] += 1
            put_back()
stop.set()
t.join()
integrations.patch("difflib")
integrations.unpatch("difflib")
counts["restored"] = snapshot() == plain
print(json.dumps(counts))
"""


@pytestmark_patch
def test_ctrl_c_never_leaves_difflib_half_patched() -> None:
    # Ctrl-C (an interrupt sent to the main thread every 0-0.3 ms) while the main thread
    # patches and unpatches difflib: difflib's classes and functions are always either all
    # pyroid's or all difflib's, never a mix. One cut off from patch()'s bookkeeping is
    # taken over by the next patch(), and unpatch() then puts difflib back.
    proc = subprocess.run(
        [sys.executable, "-c", _CTRL_C, "3"],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["half"] == 0, json.dumps(out)
    assert out["unrepaired"] == 0, json.dumps(out)
    assert out["interrupted"] > 10, json.dumps(out)
    assert out["restored"], json.dumps(out)


def _unittest_messages() -> list[str]:
    class Report(unittest.TestCase):
        maxDiff = None

        def test_text(self) -> None:
            old = "".join(f"line {i}\n" for i in range(300))
            new = old.replace("line 7\n", "line seven\n").replace("line 250\n", "")
            self.assertMultiLineEqual(old, new)  # noqa: PT009 - unittest's diff is the point

        def test_list(self) -> None:
            self.assertListEqual([str(i) for i in range(50)], [str(i) for i in range(49)])  # noqa: PT009

    result = unittest.TestResult()
    unittest.defaultTestLoader.loadTestsFromTestCase(Report).run(result)
    return [text[text.index("AssertionError") :] for _, text in result.failures]


@pytestmark_patch
def test_patched_unittest_diffs_are_the_same() -> None:
    expected = _unittest_messages()
    assert len(expected) == 2
    ctx = _quiet_patch()
    try:
        with integrations.patching("difflib"):
            assert _unittest_messages() == expected
    finally:
        ctx.__exit__(None, None, None)


RUN = "import sys; from pyroid.integrations._patch import run; sys.exit(run(sys.argv[1:]))"

FAILING = """
def test_text():
    old = "".join(f"line {i}\\n" for i in range(400))
    assert old.replace("line 7", "line seven").replace("line 300\\n", "") == old

def test_list():
    assert [str(i) for i in range(50)] == [str(i) for i in range(50) if i != 20]
"""


@pytestmark_patch
def test_pyroid_run_pytest_diffs(tmp_path: Path) -> None:
    pytest.importorskip("pytest")
    (tmp_path / "test_failing.py").write_text(FAILING)
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import difflib, sys\nprint('_pyroid_b' in vars(difflib.SequenceMatcher(None, 'a', 'b')))\n"
    )
    env = {**os.environ, "PYTHONWARNINGS": "ignore"}
    args = ["-m", "pytest", "-q", "-p", "no:cacheprovider", "-vv", "test_failing.py"]

    def run(*prefix: str) -> str:
        proc = subprocess.run(
            [sys.executable, *prefix, *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            env=env,
        )
        assert proc.returncode == 1, proc.stdout + proc.stderr
        return "\n".join(line for line in proc.stdout.splitlines() if " in " not in line)

    plain = run()
    patched = run("-c", RUN, "-p", "difflib")
    assert patched == plain
    assert "line seven" in plain
    out = subprocess.run(
        [sys.executable, "-c", RUN, "-p", "difflib", str(probe)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=env,
    )
    assert out.stdout.strip() == "True", out.stderr
