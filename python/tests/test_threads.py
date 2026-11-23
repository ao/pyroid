"""Threads: threads= equivalence, large batches, deadlock and fork regressions,
the free-threaded GIL state, and the PYROID_NUM_THREADS pool size.

A1 found a GIL deadlock in the 0.x code. Every scenario that could hang runs
under a timeout; the heavy ones run in a child interpreter (``run_python``),
so a hang fails the test instead of freezing pytest.
"""

import subprocess
import sys
import sysconfig
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any

import pytest

import pyroid.text as t
from pyroid._pyroid import _testing

FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
RunPython = Callable[..., subprocess.CompletedProcess[str]]
THREADS = [None, 1, 2, 3, 64]


def ref(s: str | None) -> str | None:
    return None if s is None else " ".join(s.split())


def _docs(n: int) -> list[str]:
    """``n`` short documents; one in three is already normalized."""
    shapes = ("  doc {}  ", "doc\t{}", "doc {}")
    return [shapes[i % 3].format(i) for i in range(n)]


DOCS = _docs(5_000)
WANT = [ref(d) for d in DOCS]


# ---- threads= gives identical results -----------------------------------------


@pytest.mark.parametrize("threads", THREADS)
@pytest.mark.parametrize("kind", ["list", "tuple", "column", "column-4-chunks"])
def test_threads_equivalence(kind: str, threads: int | None) -> None:
    x: Any
    if kind == "list":
        x = DOCS
    elif kind == "tuple":
        x = tuple(DOCS)
    else:
        x = _testing.column(DOCS, chunks=4 if kind.endswith("chunks") else 1)
    out = t.normalize_whitespace(x, threads=threads)
    mask = t.min_chars(x, 7, threads=threads)
    lengths = _testing.char_lengths(x, threads=threads)
    if kind.startswith("column"):
        out, mask = out.to_pylist(), mask.to_pylist()
    assert list(out) == WANT
    assert list(mask) == [len(d) >= 7 for d in DOCS]
    assert lengths == [len(d) for d in DOCS]
    if kind in {"list", "tuple"}:
        # Identity survives every threads setting.
        assert all((o is d) == (w == d) for o, d, w in zip(out, x, WANT, strict=True))


# ---- large batches never hang ---------------------------------------------------


@pytest.mark.timeout(60)
@pytest.mark.parametrize("n", [20_000, 100_000])
def test_large_list_and_tuple(n: int) -> None:
    docs = _docs(n)
    want = [ref(d) for d in docs]
    assert t.normalize_whitespace(docs) == want
    assert t.normalize_whitespace(tuple(docs)) == tuple(want)
    assert t.min_chars(docs, 8) == [len(d) >= 8 for d in docs]


@pytest.mark.timeout(60)
def test_large_column() -> None:
    values: list[str | None] = [None if i % 11 == 0 else d for i, d in enumerate(_docs(20_000))]
    col = _testing.column(values, chunks=3)
    assert t.normalize_whitespace(col).to_pylist() == [ref(v) for v in values]
    assert t.min_chars(col, 8).to_pylist() == [None if v is None else len(v) >= 8 for v in values]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("type_name", ["string", "large_string", "string_view"])
def test_large_pyarrow(pa: ModuleType, type_name: str) -> None:
    values: list[str | None] = [None if i % 11 == 0 else d for i, d in enumerate(_docs(20_000))]
    arr = pa.chunked_array([values[:15_000], values[15_000:]], type=getattr(pa, type_name)())
    out = t.normalize_whitespace(arr)
    assert out.to_pylist() == [ref(v) for v in values]
    assert pa.chunked_array(out).num_chunks == 2


@pytest.mark.timeout(60)
def test_concurrent_calls_in_process() -> None:
    docs = _docs(10_000)
    want = [ref(d) for d in docs]
    col = _testing.column(docs, chunks=2)

    def work(i: int) -> bool:
        threads = [None, 1, 2][i % 3]
        ok = t.normalize_whitespace(docs, threads=threads) == want
        return ok and t.normalize_whitespace(col, threads=threads).to_pylist() == want

    with ThreadPoolExecutor(max_workers=4) as ex:
        assert all(ex.map(work, range(24)))


def _race_on(docs: list[str], n_threads: int = 8) -> bool:
    """Start ``n_threads`` pyroid calls on the same ``docs`` at the same instant."""
    want = [ref(d) for d in docs]  # str.split() and len() don't build the UTF-8 form
    want_mask = [len(d) >= 5 for d in docs]
    barrier = threading.Barrier(n_threads)

    def work(k: int) -> bool:
        barrier.wait()
        if k % 2:
            return t.min_chars(docs, 5) == want_mask
        return t.normalize_whitespace(docs) == want

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        return all(ex.map(work, range(n_threads)))


@pytest.mark.timeout(60)
def test_threads_share_fresh_non_ascii_strs() -> None:
    # A non-ASCII str gets its UTF-8 form cached on first use; here several
    # threads race to do that for the same, freshly built objects.
    for _ in range(3):
        assert _race_on(["".join(("\xe9", " ", str(i), "  \U0001f600")) for i in range(5_000)])


def test_pool_size_is_positive() -> None:
    assert _testing.pool_size() >= 1


# ---- a producer that hands out one cached capsule -----------------------------


class _CachedStream:
    """Returns the *same* stream capsule from every ``__arrow_c_stream__`` call."""

    def __init__(self, docs: list[str]) -> None:
        self.capsule: object = _testing.column(docs, chunks=2).__arrow_c_stream__()

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
        return self.capsule


class _CachedArray:
    """Returns the *same* (schema, array) capsules from every ``__arrow_c_array__`` call."""

    def __init__(self, docs: list[str]) -> None:
        self.capsules: tuple[object, object] = _testing.column(docs).__arrow_c_array__()

    def __arrow_c_array__(self, requested_schema: object | None = None) -> tuple[object, object]:
        return self.capsules


@pytest.mark.thread_unsafe  # starts its own threads
@pytest.mark.timeout(120)
@pytest.mark.parametrize("producer", [_CachedStream, _CachedArray])
def test_cached_capsule_is_consumed_exactly_once(producer: Callable[[list[str]], Any]) -> None:
    # Regression: on free-threaded builds two threads could both move the
    # struct out of one shared capsule and both release it (a double free).
    docs = _docs(500)
    want = [ref(d) for d in docs]
    n_threads = 8
    for _ in range(40):
        source = producer(docs)
        barrier = threading.Barrier(n_threads)

        def work(_: int, source: Any = source, barrier: threading.Barrier = barrier) -> object:
            barrier.wait()
            try:
                return t.normalize_whitespace(source).to_pylist()
            except ValueError as err:
                return err

        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            results = list(ex.map(work, range(n_threads)))
        ok = [r for r in results if not isinstance(r, ValueError)]
        errors = [r for r in results if isinstance(r, ValueError)]
        assert ok == [want]
        assert len(errors) == n_threads - 1
        assert all("already consumed" in str(e) for e in errors), errors


# ---- heavy concurrency in a child interpreter -------------------------------------

_STRESS = """
import gc, sys, threading
import pyroid, pyroid.text as t
from pyroid._pyroid import _testing

try:
    import pyarrow as pa
except ImportError:
    pa = None

sys.setswitchinterval(1e-6)  # maximise GIL hand-offs on GIL builds
N = 10_000
docs = [f"  doc {i}  " if i % 2 else f"d\xf6c {i}" for i in range(N)]
want = [" ".join(d.split()) for d in docs]
want_mask = [len(d) >= 8 for d in docs]
col = _testing.column(docs, chunks=3)
arr = pa.chunked_array([docs[:4000], docs[4000:]]) if pa else None
errors = []
stop = threading.Event()

def check(cond, what):
    if not cond:
        errors.append(what)

def worker(k):
    try:
        for it in range(ROUNDS):
            threads = [None, 1, 2, 3][(k + it) % 4]
            check(t.normalize_whitespace(docs, threads=threads) == want, "list")
            check(t.normalize_whitespace(tuple(docs), threads=threads) == tuple(want), "tuple")
            check(t.min_chars(docs, 8, threads=threads) == want_mask, "mask")
            check(t.normalize_whitespace(col, threads=threads).to_pylist() == want, "column")
            check(_testing.char_lengths(col, threads=threads) == [len(d) for d in docs], "map")
            if arr is not None:
                out = t.normalize_whitespace(arr, threads=threads)
                check(pa.chunked_array(out).to_pylist() == want, "pyarrow")
            try:  # error paths under contention
                t.normalize_whitespace(docs[:100] + [chr(0xD800)] + docs[:100])
                errors.append("no surrogate error")
            except UnicodeEncodeError:
                pass
            try:
                t.normalize_whitespace(docs[:100] + [1])
                errors.append("no input error")
            except pyroid.InputError:
                pass
    except BaseException as e:  # report everything, panics included
        errors.append(f"worker {k}: {type(e).__name__}: {e}")

def collector():
    # On free-threaded builds each collection stops the world (twice). Back to
    # back, they leave the other threads almost no time attached: a worker that
    # detached (every batch call does) waits for each pause to end, and each
    # pause waits for every attached thread to be scheduled. On a busy 4-CPU
    # runner that slows the run 20x or more, past the timeout, with no
    # deadlock (a pure-Python workload slows the same way). With the 1 ms gap
    # there are still about a hundred collections a second, landing at
    # arbitrary points of the workers' calls.
    while not stop.wait(0.001):
        gc.collect()

g = threading.Thread(target=collector)
g.start()
ws = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
for w in ws:
    w.start()
for w in ws:
    w.join()
stop.set()
g.join()
print("errors:", errors[:5])
assert not errors, errors
print("ok")
"""


@pytest.mark.thread_unsafe(reason="runs a subprocess with 8 threads of its own")
@pytest.mark.timeout(240)
def test_concurrent_batches_with_gc_do_not_deadlock(run_python: RunPython) -> None:
    proc = run_python("ROUNDS = 30\n" + _STRESS, timeout=200)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip().endswith("ok")


@pytest.mark.thread_unsafe(reason="runs a subprocess")
@pytest.mark.timeout(120)
def test_worker_panics_under_concurrency(run_python: RunPython) -> None:
    proc = run_python(
        """
        import threading
        import pyroid.text as t
        from pyroid._pyroid import _testing
        docs = ["a  b"] * 10_000
        bad = []
        def panicker():
            for _ in range(5):
                try:
                    _testing.panic(in_worker=True)
                except BaseException as e:
                    if type(e).__name__ != "PanicException":
                        bad.append(repr(e))
        def worker():
            for _ in range(10):
                if t.normalize_whitespace(docs) != ["a b"] * 10_000:
                    bad.append("wrong result")
        ts = [threading.Thread(target=panicker) for _ in range(2)]
        ts += [threading.Thread(target=worker) for _ in range(4)]
        for th in ts: th.start()
        for th in ts: th.join()
        assert not bad, bad
        assert t.normalize_whitespace(docs) == ["a b"] * 10_000
        print("ok")
        """,
        timeout=100,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == "ok"


# ---- fork ------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="no fork() on Windows")
@pytest.mark.thread_unsafe(reason="runs a subprocess that forks")
@pytest.mark.timeout(120)
def test_fork_child_rebuilds_the_pool(run_python: RunPython) -> None:
    proc = run_python(
        """
        import multiprocessing as mp, warnings
        warnings.simplefilter("ignore")  # "multi-threaded process, fork() may deadlock"
        import pyroid.text as t
        from pyroid._pyroid import _testing

        docs = [f"  doc  {i} " for i in range(10_000)]
        want = [" ".join(d.split()) for d in docs]
        assert t.normalize_whitespace(docs) == want  # builds the pool in the parent
        parent_size = _testing.pool_size()

        def child(q):
            ok = t.normalize_whitespace(docs) == want
            ok = ok and t.normalize_whitespace(_testing.column(docs, chunks=2)).to_pylist() == want
            ok = ok and t.min_chars(docs, 5, threads=2) == [True] * len(docs)
            q.put((ok, _testing.pool_size()))

        ctx = mp.get_context("fork")
        q = ctx.Queue()
        p = ctx.Process(target=child, args=(q,))
        p.start()
        ok, child_size = q.get(timeout=60)
        p.join(60)
        assert p.exitcode == 0, p.exitcode
        assert ok
        assert child_size == parent_size, (child_size, parent_size)
        assert t.normalize_whitespace(docs) == want  # the parent's pool still works
        print("ok")
        """,
        timeout=100,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == "ok"


# ---- free-threaded builds keep the GIL off -------------------------------------------


@pytest.mark.skipif(not FREE_THREADED, reason="needs a free-threaded CPython build")
@pytest.mark.thread_unsafe(reason="runs a subprocess")
def test_gil_stays_disabled_after_use(run_python: RunPython) -> None:
    proc = run_python(
        """
        import sys
        import pyroid.text as t
        from pyroid._pyroid import _testing
        docs = ["a  b"] * 10_000
        assert t.normalize_whitespace(docs) == ["a b"] * 10_000
        assert t.min_chars(tuple(docs), 3) == (True,) * 10_000
        col = t.normalize_whitespace(_testing.column(docs, chunks=2))
        assert col.to_pylist() == ["a b"] * 10_000
        assert _testing.char_lengths(col) == [3] * 10_000
        _testing.pool_size()
        print(sys._is_gil_enabled())
        """,
        args=["-W", "error::RuntimeWarning"],
        env={"PYTHON_GIL": None},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", proc.stderr


# ---- PYROID_NUM_THREADS ------------------------------------------------------------


@pytest.mark.thread_unsafe(reason="runs a subprocess")
def test_num_threads_env_sets_the_pool_size(run_python: RunPython) -> None:
    proc = run_python(
        """
        import os, pyroid.text as t
        from pyroid._pyroid import _testing
        print(_testing.pool_size())
        os.environ["PYROID_NUM_THREADS"] = "5"  # read once per process: no effect now
        assert t.normalize_whitespace(["a  b"] * 1000) == ["a b"] * 1000
        print(_testing.pool_size())
        """,
        env={"PYROID_NUM_THREADS": "3"},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["3", "3"]


@pytest.mark.thread_unsafe(reason="runs a subprocess")
def test_num_threads_env_set_before_first_use(run_python: RunPython) -> None:
    proc = run_python(
        """
        import os
        os.environ["PYROID_NUM_THREADS"] = "2"
        from pyroid._pyroid import _testing
        print(_testing.pool_size())
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "2"


@pytest.mark.thread_unsafe(reason="runs a subprocess")
@pytest.mark.parametrize("value", ["abc", "0", "-1", "1.5"])
def test_invalid_num_threads_env(run_python: RunPython, value: str) -> None:
    proc = run_python(
        """
        import pyroid, pyroid.text as t
        from pyroid._pyroid import _testing
        # Work that never touches the pool still runs (one task's worth: < 128 documents).
        assert t.normalize_whitespace(["a  b"] * 127) == ["a b"] * 127
        assert t.normalize_whitespace(["a  b"] * 1000, threads=1) == ["a b"] * 1000
        long = " a  b " * 2000  # >= 4096 bytes: detached, but not on the pool
        assert t.normalize_whitespace(long) == " ".join(long.split())
        for call in (
            lambda: t.normalize_whitespace(["a  b"] * 128),
            lambda: t.min_chars(["a"] * 1000, 1),
            lambda: t.normalize_whitespace(_testing.column(["a  b"] * 1000)),
            _testing.pool_size,
        ):
            try:
                call()
            except pyroid.PyroidError as e:
                assert "PYROID_NUM_THREADS" in str(e), e
                assert not isinstance(e, pyroid.InputError)
            else:
                raise AssertionError("no PyroidError")
        print("ok")
        """,
        env={"PYROID_NUM_THREADS": value},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
