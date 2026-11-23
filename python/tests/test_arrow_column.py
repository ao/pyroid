"""``pyroid.Column`` as a read-only sequence: ``len``, indexing, slices,
iteration, ``reversed``, ``in``/``index``/``count``, each checked against the
same operation on the list of the column's values; threads sharing a column
or one iterator; ``fork``.

Columns are built with ``_testing.column`` (no Arrow library), of every type
pyroid returns and in one or several chunks. The numpy, pyarrow and polars
checks at the end skip when those are not installed.
"""

import collections.abc as abc
import datetime as dt
import gc
import operator
import os
import subprocess
import sys
import textwrap
import threading
import tracemalloc
from collections import Counter
from collections.abc import Callable
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import pyroid
import pyroid.dateutil.parser as pp
import pyroid.text as t
from pyroid._pyroid import _testing

STRINGS: list[Any] = [
    "  a  b ",
    None,
    "",
    "ok",
    "\xe9\t\xfc",
    "a much longer document than twelve bytes",
    None,
    "  \U0001f600  ",
    "x" * 100,
    "nul\x00 inside",
    "ok",
]
#: 19 rows, so the bitmaps run past two byte boundaries.
BOOLS: list[Any] = [True, None, False, True, True, None, False, False, True, None]
BOOLS += [True, True, False, None, False, True, False, True, None]
BINARY: list[Any] = [b"<html>", None, b"", b"\xff\xfe", b"\x00", None, "\xe9".encode(), b"x" * 50]

RunPython = Callable[..., subprocess.CompletedProcess[str]]

UTC = dt.UTC
#: Before and after 1970, with microseconds, repeated values and nulls.
NAIVE: list[Any] = [
    dt.datetime(2003, 9, 25, 10, 49, 41, 502),
    None,
    dt.datetime(1969, 12, 31, 23, 59, 59, 999999),
    dt.datetime(1970, 1, 1),
    dt.datetime(1, 1, 1),
    dt.datetime(9999, 12, 31, 23, 59, 59, 999999),
    None,
    dt.datetime(2003, 9, 25, 10, 49, 41, 502),
    dt.datetime(2024, 2, 29, 12),
]
AWARE: list[Any] = [None if v is None else v.replace(tzinfo=UTC) for v in NAIVE]

VALUES = {
    "string": STRINGS,
    "large_string": STRINGS,
    "bool": BOOLS,
    "large_binary": BINARY,
    "timestamp[us]": NAIVE,
    "timestamp[us, tz=UTC]": AWARE,
}
KINDS = list(VALUES)
KIND_IDS = ["string", "large_string", "bool", "large_binary", "timestamp", "timestamp-utc"]
EPOCHS = {
    "timestamp[us]": dt.datetime(1970, 1, 1),
    "timestamp[us, tz=UTC]": dt.datetime(1970, 1, 1, tzinfo=UTC),
}


def hook_values(kind: str, values: list[Any]) -> list[Any]:
    """``values`` as ``_testing.column`` takes them: microseconds for the timestamp types."""
    epoch = EPOCHS.get(kind)
    if epoch is None:
        return values
    return [None if v is None else (v - epoch) // dt.timedelta(microseconds=1) for v in values]


#: One chunk, three (nearly) equal chunks, and uneven chunks with a one-row chunk.
LAYOUTS = ["one", "three", "uneven"]


def make(kind: str, layout: str, values: list[Any] | None = None, name: str = "text") -> Any:
    values = hook_values(kind, VALUES[kind] if values is None else values)
    if layout == "one":
        return _testing.column(values, type=kind, name=name)
    if layout == "three":
        return _testing.column(values, type=kind, chunks=3, name=name)
    n = len(values)
    sizes = [min(1, n), min(4, max(n - 1, 0))]
    sizes.append(n - sum(sizes))
    return _testing.column(values, type=kind, sizes=sizes, name=name)


@pytest.fixture(params=KINDS, ids=KIND_IDS)
def kind(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(params=LAYOUTS)
def layout(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def test_testing_columns(kind: str, layout: str) -> None:
    col = make(kind, layout)
    assert col.to_pylist() == VALUES[kind]
    assert col.type == kind
    assert col.num_chunks == {"one": 1, "three": 3, "uneven": 3}[layout]


# ---- iteration ----------------------------------------------------------------


def test_iteration(kind: str, layout: str) -> None:
    col = make(kind, layout)
    ref = VALUES[kind]
    got = list(col)
    assert got == ref
    assert [type(x) for x in got] == [type(x) for x in ref]
    assert list(reversed(col)) == ref[::-1]
    assert [type(x) for x in reversed(col)] == [type(x) for x in ref[::-1]]
    assert tuple(col) == tuple(ref)
    # Each call starts a new iterator.
    assert list(col) == list(col)


def test_iterator_protocol() -> None:
    col = make("string", "three")
    n = len(col)
    it = iter(col)
    assert iter(it) is it
    assert type(it) is type(reversed(col))
    assert operator.length_hint(it) == n
    next(it)
    next(it)
    assert operator.length_hint(it) == n - 2
    assert list(it) == STRINGS[2:]
    assert operator.length_hint(it) == 0
    for _ in range(3):  # stays exhausted
        with pytest.raises(StopIteration):
            next(it)
    back = reversed(col)
    assert operator.length_hint(back) == n
    assert next(back) == STRINGS[-1]
    assert operator.length_hint(back) == n - 1
    assert list(back) == STRINGS[-2::-1]
    with pytest.raises(StopIteration):
        next(back)


def test_an_iterator_keeps_its_column_alive() -> None:
    it = iter(_testing.column(["a", None, "b"], chunks=2))
    back = reversed(_testing.column(["a", None, "b"], chunks=2))
    gc.collect()
    assert list(it) == ["a", None, "b"]
    assert list(back) == ["b", None, "a"]


@pytest.mark.thread_unsafe(reason="tracemalloc is process-wide")
def test_iteration_is_lazy() -> None:
    n = 200_000
    col = _testing.column([f"r{i}" for i in range(n)], chunks=4)
    tracemalloc.start()
    try:
        it = iter(col)
        first = [next(it) for _ in range(10)]
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert first == [f"r{i}" for i in range(10)]
    # A list of 200,000 values would take more than 1.6 MB for its pointers alone.
    assert peak < 100_000


# ---- indexing -----------------------------------------------------------------


class Index:
    def __init__(self, i: int) -> None:
        self.i = i

    def __index__(self) -> int:
        return self.i


def test_getitem(kind: str, layout: str) -> None:
    col = make(kind, layout)
    ref = VALUES[kind]
    n = len(ref)
    for i in range(-n, n):
        assert col[i] == ref[i], i
        assert type(col[i]) is type(ref[i]), i
    assert col[Index(2)] == ref[2]
    assert col[True] == ref[1]
    for i in [n, n + 1, -n - 1, -n - 7, 2**62, -(2**62), sys.maxsize, -sys.maxsize - 1]:
        with pytest.raises(IndexError, match="Column index out of range"):
            col[i]
        with pytest.raises(IndexError):
            ref[i]
    bigs: list[Any] = [10**30, -(10**30), Index(2**80)]
    for big in bigs:
        # As for a list: an index past the C range is an IndexError too.
        with pytest.raises(IndexError, match="into an index-sized integer"):
            col[big]
        with pytest.raises(IndexError, match="into an index-sized integer"):
            ref[big]


@pytest.mark.parametrize(
    "key", ["1", 1.0, None, (1,), [1], b"1"], ids=["str", "float", "None", "tuple", "list", "bytes"]
)
def test_getitem_rejects_what_list_rejects(key: Any) -> None:
    col: Any = make("string", "one")
    with pytest.raises(TypeError, match="Column indices must be integers or slices, not "):
        col[key]
    with pytest.raises(TypeError):
        STRINGS[key]


def test_an_index_method_error_propagates() -> None:
    class Bad:
        def __index__(self) -> int:
            raise ZeroDivisionError

    col: Any = make("string", "one")
    with pytest.raises(ZeroDivisionError):
        col[Bad()]


# ---- slices -------------------------------------------------------------------

BOUNDS = [None, 0, 1, 3, 5, -1, -4, 9, 25, -25, 10**30, -(10**30)]
STEPS = [None, 1, 2, 3, -1, -2, -5, 10**30, -(10**30)]


def check_slice(col: Any, ref: list[Any], s: slice) -> Any:
    got = col[s]
    assert isinstance(got, pyroid.Column), s
    want = ref[s]
    assert got.to_pylist() == want, s
    assert list(got) == want, s
    assert len(got) == len(want), s
    assert got.null_count == want.count(None), s
    assert (got.type, got.name) == (col.type, col.name), s
    return got


def test_slices(kind: str, layout: str) -> None:
    col = make(kind, layout)
    ref = VALUES[kind]
    for start in BOUNDS:
        for stop in BOUNDS:
            for step in STEPS:
                got = check_slice(col, ref, slice(start, stop, step))
                if step in (None, 1):
                    assert got.num_chunks <= col.num_chunks
                if len(got) == 0:
                    assert got.num_chunks == 0


def test_slices_of_slices(kind: str, layout: str) -> None:
    col = make(kind, layout)
    ref = VALUES[kind]
    for outer in [slice(1, None), slice(2, -1), slice(None, None, -1), slice(1, None, 2)]:
        for inner in [slice(None), slice(1, 4), slice(None, None, -2), slice(-3, None)]:
            check_slice(col[outer], ref[outer], inner)
    assert col[2:][3] == ref[2:][3]
    assert col[::-1][0] == ref[-1]


def test_slice_step_zero() -> None:
    col: Any = make("string", "one")
    with pytest.raises(ValueError, match="slice step cannot be zero"):
        col[::0]


def test_contiguous_slices_keep_the_chunks() -> None:
    col = _testing.column([str(i) for i in range(10)], sizes=[4, 4, 2])
    assert col[:].num_chunks == 3
    assert col[1:3].num_chunks == 1
    assert col[3:5].num_chunks == 2
    assert col[3:9].num_chunks == 3
    assert col[4:8].num_chunks == 1
    assert col[2:2].num_chunks == 0
    assert col[7:7].to_pylist() == []
    # A step copies the rows into one new chunk.
    assert col[::2].num_chunks == 1
    assert col[::-1].num_chunks == 1
    assert col[5:6:3].num_chunks == 1  # one row: no copy needed
    assert col[3:9][1:4].to_pylist() == ["4", "5", "6"]


def test_slices_are_columns_like_any_other(kind: str) -> None:
    col = make(kind, "three")
    ref = VALUES[kind]
    part = col[1:-1]
    assert repr(part) == (
        f"pyroid.Column<{kind}>(len={len(ref) - 2}, null_count={ref[1:-1].count(None)}, "
        f"num_chunks={part.num_chunks})"
    )
    assert list(reversed(part)) == ref[-2:0:-1]
    if kind in ("string", "large_string"):
        out = t.normalize_whitespace(part)
        assert isinstance(out, pyroid.Column)
        assert out.to_pylist() == [None if s is None else " ".join(s.split()) for s in ref[1:-1]]


# ---- in, index, count ---------------------------------------------------------


class StrSub(str):
    pass


class EqAll:
    def __eq__(self, other: object) -> bool:
        return True

    __hash__ = object.__hash__


class EqNone:
    def __eq__(self, other: object) -> bool:
        return False

    __hash__ = object.__hash__


class DateTimeSub(dt.datetime):
    pass


EXTRA: dict[str, list[Any]] = {
    "string": ["missing", "\ud800", "a\ud800", b"ok", 1, StrSub("ok"), StrSub("no"), 0.0],
    "large_string": ["missing", "\ud800", b"ok", StrSub("ok")],
    "bool": [1, 0, 1.0, 0.0, 2, "True", b"", Index(1)],
    "large_binary": [b"missing", "<html>", bytearray(b"<html>"), memoryview(b"<html>"), 0],
    "timestamp[us]": [
        dt.datetime(2003, 9, 25, 10, 49, 41),
        dt.datetime(2003, 9, 25, 10, 49, 41, 502, tzinfo=UTC),  # aware: equals no naive row
        dt.datetime(2003, 9, 25, 10, 49, 41, 502, fold=1),  # fold is ignored without a zone
        dt.date(2003, 9, 25),
        DateTimeSub(2003, 9, 25, 10, 49, 41, 502),
        0,
        "2003-09-25 10:49:41.000502",
    ],
    "timestamp[us, tz=UTC]": [
        dt.datetime(2003, 9, 25, 10, 49, 41, 502),  # naive: equals no aware row
        dt.datetime(2003, 9, 25, 12, 49, 41, 502, tzinfo=dt.timezone(dt.timedelta(hours=2))),
        dt.datetime(2003, 9, 25, 10, 49, 41, 502, tzinfo=dt.timezone(dt.timedelta(0), "Z")),
        dt.datetime(2003, 9, 25, 10, 49, 41, 503, tzinfo=UTC),
        DateTimeSub(2003, 9, 25, 10, 49, 41, 502, tzinfo=UTC),
        dt.date(2003, 9, 25),
    ],
}


def test_contains_index_count(kind: str, layout: str) -> None:
    col = make(kind, layout)
    ref = VALUES[kind]
    n = len(ref)
    needles = [*ref, None, EqAll(), EqNone(), *EXTRA[kind]]
    for x in needles:
        assert (x in col) == (x in ref), x
        assert col.count(x) == ref.count(x), x
        for bounds in [(), (2,), (-3,), (1, -1), (3, 3), (5, 2), (-100, 10**30), (10**30,)]:
            try:
                want: int | None = ref.index(x, *bounds)
            except ValueError:
                want = None
            if want is None:
                with pytest.raises(ValueError, match="is not in Column"):
                    col.index(x, *bounds)
            else:
                assert col.index(x, *bounds) == want, (x, bounds)
    assert col.index(ref[-1], n - 1) == n - 1


def test_index_arguments() -> None:
    col: Any = make("string", "three")
    assert col.index("ok", Index(4)) == STRINGS.index("ok", 4)
    with pytest.raises(TypeError, match="slice indices must be integers"):
        col.index("ok", "1")
    with pytest.raises(TypeError):
        col.index("ok", start=1)  # positional only, as for list
    with pytest.raises(TypeError):
        col.index()


def test_comparison_errors_propagate() -> None:
    class Boom:
        def __eq__(self, other: object) -> bool:
            raise ZeroDivisionError

        __hash__ = object.__hash__

    col = make("string", "three")
    with pytest.raises(ZeroDivisionError):
        Boom() in col  # noqa: B015
    with pytest.raises(ZeroDivisionError):
        col.count(Boom())
    with pytest.raises(ZeroDivisionError):
        col.index(Boom())


def test_large_scans() -> None:
    n = 300_000  # several Ctrl-C check intervals
    rows = [None if i % 11 == 0 else f"doc {i % 1000}" for i in range(n)]
    col = _testing.column(rows, chunks=7)
    assert col.count("doc 5") == rows.count("doc 5")
    assert col.count(None) == rows.count(None)
    assert col.index("doc 999", 150_000) == rows.index("doc 999", 150_000)
    assert "doc 1000" not in col
    assert col[::3].to_pylist() == rows[::3]


# ---- empty columns ------------------------------------------------------------


def test_empty_column(kind: str) -> None:
    for col in [_testing.column([], type=kind), make(kind, "three")[3:3]]:
        assert len(col) == 0
        assert not col
        assert list(col) == []
        assert list(reversed(col)) == []
        for i in [0, -1, 1]:
            with pytest.raises(IndexError):
                col[i]
        assert col[:].to_pylist() == []
        assert col[::-1].to_pylist() == []
        assert None not in col
        assert col.count(None) == 0
        with pytest.raises(ValueError, match="None is not in Column"):
            col.index(None)


# ---- the sequence ABCs --------------------------------------------------------


def test_collection_abcs() -> None:
    col = make("string", "one")
    for cls in [abc.Iterable, abc.Reversible, abc.Collection, abc.Container, abc.Sized]:
        assert isinstance(col, cls), cls
    assert isinstance(iter(col), abc.Iterator)
    # Deliberately not a registered Sequence: polars.Series(x) checks
    # isinstance(x, Sequence) before the Arrow PyCapsule interface, and would
    # then copy the values through Python objects instead of the Arrow buffers.
    assert not isinstance(col, abc.Sequence)


# ---- threads ------------------------------------------------------------------

THREADS = 8


def run_threads(target: Any, n: int = THREADS) -> None:
    barrier = threading.Barrier(n)
    errors: list[BaseException] = []

    def run(k: int) -> None:
        try:
            barrier.wait()
            target(k)
        except BaseException as e:  # re-raised in the caller
            errors.append(e)

    workers = [threading.Thread(target=run, args=(k,)) for k in range(n)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    if errors:
        raise errors[0]


def test_threads_read_one_column() -> None:
    n = 20_000
    rows = [None if i % 13 == 0 else f"row {i}" for i in range(n)]
    col = _testing.column(rows, chunks=5)

    def read(k: int) -> None:
        assert list(col) == rows
        assert list(reversed(col)) == rows[::-1]
        for i in range(k, n, 97):
            assert col[i] == rows[i]
        assert col[k::7].to_pylist() == rows[k::7]
        assert col[k : n - k].to_pylist() == rows[k : n - k]
        assert col.count(None) == rows.count(None)
        assert ("row 19999" in col) is True

    run_threads(read)


@pytest.mark.parametrize("backwards", [False, True], ids=["iter", "reversed"])
def test_threads_share_one_iterator(backwards: bool) -> None:
    n = 50_000
    col = _testing.column([str(i) for i in range(n)], chunks=3)
    it = reversed(col) if backwards else iter(col)
    taken: list[list[int]] = [[] for _ in range(THREADS)]

    def drain(k: int) -> None:
        mine = taken[k]
        for value in it:
            mine.append(int(value))

    run_threads(drain)
    # Every row was taken exactly once, and each thread saw its rows in order.
    assert Counter(i for mine in taken for i in mine) == Counter(range(n))
    for mine in taken:
        assert mine == sorted(mine, reverse=backwards)
    with pytest.raises(StopIteration):
        next(it)


# ---- fork ---------------------------------------------------------------------

FORK_SCRIPT = textwrap.dedent(
    """
    import os, threading
    import pyroid.text as t
    from pyroid._pyroid import _testing

    rows = [None if i % 7 == 0 else f"row  {i}" for i in range(5000)]
    col = _testing.column(rows, chunks=4)
    it = iter(col)
    head = [next(it) for _ in range(10)]
    # Another thread reads the column while the main thread forks.
    stop = threading.Event()
    def spin():
        while not stop.is_set():
            assert list(col[100:300]) == rows[100:300]
    reader = threading.Thread(target=spin)
    reader.start()
    pid = os.fork()
    if pid == 0:
        ok = (
            list(it) == rows[10:]
            and col[1234] == rows[1234]
            and list(reversed(col[50:60])) == rows[59:49:-1]
            and col.count(None) == rows.count(None)
            and t.normalize_whitespace(col[5:50]).to_pylist()
            == [None if s is None else " ".join(s.split()) for s in rows[5:50]]
        )
        os._exit(0 if ok else 3)
    stop.set()
    reader.join()
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0, status
    assert head == rows[:10] and list(it) == rows[10:]
    print("ok")
    """
)


@pytest.mark.thread_unsafe(reason="a subprocess that forks")
@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_fork(run_python: RunPython) -> None:
    # Python warns that fork() with threads running may deadlock; the child
    # here only reads Rust memory and runs one small batch.
    out = run_python(FORK_SCRIPT, args=["-W", "ignore::DeprecationWarning"])
    assert (out.returncode, out.stdout.strip()) == (0, "ok"), out.stderr


# ---- Ctrl-C -------------------------------------------------------------------

CTRL_C_SCRIPT = textwrap.dedent(
    """
    import signal, time
    from pyroid._pyroid import _testing

    armed, fired = [False], []

    def stop(*args):
        if not armed[0]:
            return  # a timer that came after its call: ignored
        armed[0] = False
        fired.append(time.perf_counter())
        raise KeyboardInterrupt

    # SIGALRM comes from the kernel, so it trips CPython's signal flag while the call runs
    # attached (a Python timer thread could not take the GIL on a GIL build). The handler
    # runs where the call checks for signals, so its time is where the call stopped.
    signal.signal(signal.SIGALRM, stop)
    col = _testing.column(["d"] * 10_000_000, chunks=3)
    # Each call makes or compares a Python object per row, with no Python code in between.
    calls = {"count": lambda: col.count(0), "to_pylist": col.to_pylist, "in": lambda: 0 in col}
    for name, call in calls.items():
        t0 = time.perf_counter()
        call()
        full = time.perf_counter() - t0
        result = "finished"
        # A loaded macOS runner can deliver the timer after the whole call: try again then.
        for _ in range(5):
            armed[0] = True
            signal.setitimer(signal.ITIMER_REAL, full / 5)
            t0 = time.perf_counter()
            try:
                try:
                    call()
                finally:
                    armed[0] = False
                    signal.setitimer(signal.ITIMER_REAL, 0)
            except KeyboardInterrupt:
                at, done = fired[-1] - t0, time.perf_counter() - fired[-1]
                if at < 0.9 * full:
                    result = f"{at} {done}"
                    break
        print(name, result, full)
    """
)


@pytest.mark.thread_unsafe(reason="a subprocess that signals itself")
@pytest.mark.skipif(sys.platform == "win32", reason="needs SIGALRM")
def test_ctrl_c_stops_a_long_call(run_python: RunPython) -> None:
    out = run_python(CTRL_C_SCRIPT, timeout=300.0)
    assert out.returncode == 0, out.stderr
    for line in out.stdout.splitlines():
        _, *result, full = line.split()
        if float(full) < 0.1:
            continue  # too fast to tell a check during the call from one after it
        # Interrupted a fifth of the way in, in one of five tries: the call saw the signal
        # before its end (it checks during the call, not only after it), and then stopped
        # promptly. Where the signal lands depends on the timer, which a loaded macOS runner
        # delivers tens of milliseconds late, so a try that saw it only at the end is retried.
        assert result != ["finished"], line
        at, done = map(float, result)
        assert at < 0.9 * float(full), line
        assert done < 0.25, line


STOPPED_FILL_SCRIPT = textwrap.dedent(
    """
    import gc, signal, time, tracemalloc
    from pyroid._pyroid import _testing

    n = 2_000_000
    col = _testing.column([f"row {i}" for i in range(n)], chunks=3)
    armed, half_built = [False], []

    def stop(*args):
        if not armed[0]:
            return  # the call ended first
        armed[0] = False
        # to_pylist fills its list in place. A list with slots still empty (NULL) has fewer
        # referents than slots, and the cycle GC must not show it to Python code. Only len and
        # gc.get_referents here: reading an empty slot would crash.
        half_built.append(sum(
            1 for o in gc.get_objects()
            if type(o) is list and len(o) == n and len(gc.get_referents(o)) < n
        ))
        raise KeyboardInterrupt

    signal.signal(signal.SIGALRM, stop)
    t0 = time.perf_counter()
    col.to_pylist()
    full = time.perf_counter() - t0
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    stopped = 0
    for i in range(10):
        armed[0] = True
        signal.setitimer(signal.ITIMER_REAL, full * (i + 1) / 12)
        try:
            col.to_pylist()
        except KeyboardInterrupt:
            stopped += 1
        armed[0] = False
        signal.setitimer(signal.ITIMER_REAL, 0)
    gc.collect()
    print(stopped, sum(half_built), tracemalloc.get_traced_memory()[0] - before)
    """
)


@pytest.mark.thread_unsafe(reason="a subprocess that signals itself")
@pytest.mark.skipif(sys.platform == "win32", reason="needs SIGALRM")
def test_a_stopped_to_pylist_frees_its_list_and_hides_it(run_python: RunPython) -> None:
    out = run_python(STOPPED_FILL_SCRIPT, timeout=300.0)
    assert out.returncode == 0, out.stderr
    stopped, half_built, leaked = map(int, out.stdout.split())
    assert stopped > 0, out.stdout  # else nothing was checked
    assert half_built == 0, out.stdout
    # Each stopped call made up to 2 million strs (about 60 MB) and the list (16 MB).
    assert leaked < 2**20, out.stdout


@pytest.mark.thread_unsafe(reason="tracemalloc is process-wide")
@pytest.mark.parametrize("ts_kind", list(EPOCHS), ids=["timestamp", "timestamp-utc"])
def test_a_to_pylist_that_fails_frees_its_list(ts_kind: str) -> None:
    # A row past year 9999 fails in datetime's own arithmetic, with the rows of two Ctrl-C
    # blocks already in the list: it is freed with them, as when a Ctrl-C stops the call.
    n = 2 * 65_536
    col = _testing.column([*range(n), 2**62, 0], type=ts_kind, chunks=2)
    tracemalloc.start()
    try:
        for _ in range(3):
            with pytest.raises(OverflowError):
                col.to_pylist()
        gc.collect()
        left, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Each call made 131,072 datetimes (about 6 MB) and a list of 1 MB.
    assert left < 2**20
    with pytest.raises(OverflowError):
        col[n]
    assert col[n + 1] == EPOCHS[ts_kind]
    assert col[:n].to_pylist()[-1] == EPOCHS[ts_kind] + dt.timedelta(microseconds=n - 1)


# ---- generated columns ---------------------------------------------------------

VALUE_STRATEGIES = {
    "string": st.text(max_size=12),
    "large_string": st.text(max_size=12),
    "bool": st.booleans(),
    "large_binary": st.binary(max_size=12),
    "timestamp[us]": st.datetimes(),
    "timestamp[us, tz=UTC]": st.datetimes(timezones=st.just(UTC)),
}


@st.composite
def columns(draw: st.DrawFn) -> tuple[Any, list[Any]]:
    kind = draw(st.sampled_from(KINDS))
    values = draw(st.lists(st.none() | VALUE_STRATEGIES[kind], max_size=40))
    sizes: list[int] = []
    left = len(values)
    while left:
        # At most 8 draws: Hypothesis can draw 0 thousands of times in a row, which used up the
        # example's data and failed the data_too_large health check for some seeds.
        size = draw(st.integers(0, left)) if len(sizes) < 8 else left
        sizes.append(size)
        left -= size
    return _testing.column(hook_values(kind, values), type=kind, sizes=sizes), values


bounds = st.none() | st.integers(-45, 45)


@settings(max_examples=300, deadline=None)
@given(
    columns(),
    st.lists(st.tuples(bounds, bounds, st.none() | st.integers(-6, 6).filter(bool)), max_size=3),
    st.integers(-45, 45),
)
def test_generated(column: tuple[Any, list[Any]], slices: list[Any], i: int) -> None:
    col, ref = column
    assert list(col) == ref
    assert list(reversed(col)) == ref[::-1]
    for start, stop, step in slices:
        s = slice(start, stop, step)
        col, ref = col[s], ref[s]
        assert col.to_pylist() == ref
        assert col.null_count == ref.count(None)
    if -len(ref) <= i < len(ref):
        assert col[i] == ref[i]
    else:
        with pytest.raises(IndexError):
            col[i]
    for x in ref[:3]:
        assert col.index(x) == ref.index(x)
        assert col.count(x) == ref.count(x)


# ---- numpy, pyarrow, polars ----------------------------------------------------


@pytest.fixture(scope="session")
def np() -> Any:
    """numpy, or skip the test."""
    return pytest.importorskip("numpy")


def test_numpy_reads_the_sequence(np: Any) -> None:
    col = _testing.column(["a", "bc", "d"], chunks=2)
    assert np.array(col).tolist() == ["a", "bc", "d"]
    assert np.array(_testing.column([True, False], type="bool")).dtype == np.bool_
    assert np.array(make("string", "three")).tolist() == STRINGS


def test_pyarrow_slices_share_the_buffers(pa: Any, kind: str) -> None:
    col = make(kind, "one")
    ref = VALUES[kind]
    whole = pa.array(col)
    part = pa.array(col[2:7])
    assert part.offset == 2
    assert part.to_pylist() == ref[2:7]
    part.validate(full=True)
    # The same data buffer (strings: the offsets and data; bool: the bits).
    assert [b.address for b in part.buffers()[1:]] == [b.address for b in whole.buffers()[1:]]


def test_pyarrow_takes_any_slice(pa: Any, kind: str, layout: str) -> None:
    col = make(kind, layout)
    ref = VALUES[kind]
    want_type = {
        "string": pa.string(),
        "large_string": pa.large_string(),
        "bool": pa.bool_(),
        "large_binary": pa.large_binary(),
        "timestamp[us]": pa.timestamp("us"),
        "timestamp[us, tz=UTC]": pa.timestamp("us", tz="UTC"),
    }[kind]
    for s in [slice(1, 6), slice(3, None), slice(None, -2), slice(4, 5), slice(None, None, -3)]:
        chunked = pa.chunked_array(col[s])
        chunked.validate(full=True)
        assert chunked.type == want_type
        assert chunked.to_pylist() == ref[s], s
        array = pa.array(col[s])
        array.validate(full=True)
        assert array.to_pylist() == ref[s], s


def test_polars_takes_slices_through_arrow(pl: Any, kind: str) -> None:
    col = make(kind, "three", name="field")
    ref = VALUES[kind]
    for s in [slice(None), slice(1, 6), slice(None, None, -2)]:
        series = pl.Series(col[s])
        # The Arrow path: the field name comes along (a Python sequence would give "").
        assert series.name == "field"
        if kind in EPOCHS:
            # The microseconds: polars' datetimes with a zone need a tz database (tzdata on
            # Windows), which the test environment does not promise.
            assert series.to_physical().to_list() == hook_values(kind, ref[s])
        else:
            assert series.to_list() == ref[s]


def test_pyarrow_example_from_inputs_md(pa: Any) -> None:
    col = t.normalize_whitespace(pa.chunked_array([["a  b", None], ["c", "d  e"]]))
    assert len(col) == 4
    assert (col[0], col[-1]) == ("a b", "d e")
    assert list(col) == ["a b", None, "c", "d e"]
    assert ("c" in col, col.index("c"), col.count(None)) == (True, 2, 1)
    part = col[1:3]
    assert (part.to_pylist(), part.num_chunks) == ([None, "c"], 2)


# ---- pyroid.dateutil's timestamp columns ------------------------------------------

DATES = ["2003-09-25 10:49:41", None, "Sep 26 2003 10:00:00.5", "not a date", "1969-07-20 20:17"]
DATES += ["2003-09-25 10:49:41", "1 Jan 2038", None]
ZONED = ["2003-09-25 10:49:41Z", None, "2003-09-25 10:49:41+05:30", "x", "1969-07-20 20:17-04:00"]


def parsed(texts: list[str | None], tz: bool) -> list[Any]:
    out: list[Any] = []
    for s in texts:
        try:
            v = None if s is None else pp.parse(s)
        except (ValueError, OverflowError):
            v = None
        out.append(v.astimezone(UTC) if tz and v is not None else v)
    return out


def check_sequence(col: Any, ref: list[Any]) -> None:
    assert list(col) == ref
    assert list(reversed(col)) == ref[::-1]
    assert [col[i] for i in range(-len(ref), len(ref))] == ref + ref
    for s in [slice(1, 4), slice(2, None), slice(None, None, -2), slice(5, 1, -1), slice(3, 3)]:
        check_slice(col, ref, s)
    for x in [*ref, dt.datetime(2000, 1, 1)]:
        assert (x in col) == (x in ref)
        assert col.count(x) == ref.count(x)
    assert col.null_count == ref.count(None)


@pytest.mark.parametrize("chunks", [1, 3])
def test_dateutil_columns(chunks: int) -> None:
    cases = [(DATES, False, "timestamp[us]"), (ZONED, True, "timestamp[us, tz=UTC]")]
    for texts, tz, kind in cases:
        col = pp.parse_batch(_testing.column(texts, chunks=chunks), errors="coerce")
        assert col.type == kind
        ref = parsed(texts, tz)
        assert col.to_pylist() == ref
        check_sequence(col, ref)
        # Each row compares with an equal instant in another zone, in Python.
        if tz:
            other = dt.timezone(dt.timedelta(hours=-3))
            for v in ref:
                if v is not None:
                    assert col.index(v.astimezone(other)) == ref.index(v)
    col = pp.isoparse_batch(_testing.column(["2003-09-25T10:49:41", None, "2003-09-26"]))
    check_sequence(col, [dt.datetime(2003, 9, 25, 10, 49, 41), None, dt.datetime(2003, 9, 26)])


def test_dateutil_columns_with_pyarrow(pa: Any) -> None:
    for texts, tz, want_type in [
        (DATES, False, pa.timestamp("us")),
        (ZONED, True, pa.timestamp("us", tz="UTC")),
    ]:
        col = pp.parse_batch(pa.chunked_array([texts[:3], texts[3:]]), errors="coerce")
        ref = parsed(texts, tz)
        check_sequence(col, ref)
        whole = pa.array(col)
        for s in [slice(1, 4), slice(2, None), slice(None, -1), slice(None, None, -2)]:
            for out in (pa.array(col[s]), pa.chunked_array(col[s])):
                out.validate(full=True)
                assert out.type == want_type
                assert out.to_pylist() == ref[s], s
        part = pa.array(col[2:5])
        assert part.offset == 2
        assert part.buffers()[1].address == whole.buffers()[1].address
