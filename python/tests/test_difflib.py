"""pyroid.difflib: batches, fallbacks, threads, pyroid.patch("difflib") and pyroid run.

Parity with the standard library is in test_difflib_parity.py (differential) and
test_difflib_cpython.py (CPython's own tests).
"""

from __future__ import annotations

import _thread
import collections
import difflib
import os
import subprocess
import sys
import textwrap
import threading
import time
import unittest
import warnings
from pathlib import Path
from typing import Any

import pytest

import pyroid
import pyroid.difflib as pd
from pyroid import integrations
from pyroid._pyroid import _testing

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
    class VowelsAreJunk(base):  # type: ignore[valid-type, misc]
        calls = 0

        def _SequenceMatcher__chain_b(self) -> None:
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
        cls = _vowels_are_junk(base)
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
    class Sub(difflib.SequenceMatcher):
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
