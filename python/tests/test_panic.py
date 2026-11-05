"""Rust panics become ``pyo3_runtime.PanicException``, never an abort.

A panic means a pyroid bug, so the exception is a ``BaseException`` (not an
``Exception``): ``except Exception`` handlers in user code must not swallow it.
The process, and the shared thread pool, keep working afterwards.
"""

import subprocess
from collections.abc import Callable

import pytest

import pyroid.text as t
from pyroid._pyroid import _testing

RunPython = Callable[..., subprocess.CompletedProcess[str]]
WHERE = pytest.mark.parametrize("in_worker", [False, True], ids=["caller", "worker"])


def _panic(in_worker: bool) -> BaseException:
    """Run the panic hook; return what it raised."""
    with pytest.raises(BaseException, match="pyroid test panic") as info:
        _testing.panic(in_worker=in_worker)
    return info.value


@WHERE
def test_panic_is_a_panic_exception(in_worker: bool) -> None:
    exc = _panic(in_worker)
    cls = type(exc)
    assert cls.__name__ == "PanicException"
    assert cls.__module__ == "pyo3_runtime"
    assert not isinstance(exc, Exception)
    where = "in a worker thread" if in_worker else "on the calling thread"
    assert str(exc) == f"pyroid test panic {where}"


@WHERE
def test_except_exception_does_not_catch_a_panic(in_worker: bool) -> None:
    swallowed: list[Exception] = []

    def careless() -> None:
        try:
            _testing.panic(in_worker=in_worker)
        except Exception as exc:  # what a careless user would write
            swallowed.append(exc)

    with pytest.raises(BaseException, match="pyroid test panic"):
        careless()
    assert swallowed == []


@WHERE
def test_pyroid_works_after_a_panic(in_worker: bool) -> None:
    for _ in range(3):
        _panic(in_worker)
    docs = ["a  b"] * 10_000  # large enough for the thread pool
    assert t.normalize_whitespace(docs) == ["a b"] * 10_000
    assert t.min_chars(docs, 3, threads=2) == [True] * 10_000
    assert _testing.pool_size() >= 1


@pytest.mark.thread_unsafe(reason="runs a subprocess")
def test_panic_does_not_abort_the_process(run_python: RunPython) -> None:
    proc = run_python(
        """
        import pyroid.text as t
        from pyroid._pyroid import _testing
        for in_worker in (False, True, True):
            try:
                _testing.panic(in_worker=in_worker)
            except BaseException as e:
                assert type(e).__name__ == "PanicException", repr(e)
            else:
                raise AssertionError("no panic")
        assert t.normalize_whitespace(["a  b"] * 10_000) == ["a b"] * 10_000
        print("ok")
        """,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
    assert "panicked" in proc.stderr  # Rust's panic message, not a crash report
