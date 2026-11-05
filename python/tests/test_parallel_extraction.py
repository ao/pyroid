"""Large list/tuple inputs: on a free-threaded build without the GIL, their strs' UTF-8 forms are built on the
pool (``TextInput::extract_with`` in crates/pyroid-py/src/input.rs). Results and errors must be exactly those
of the serial path (``threads=1``), which every other test exercises. Batches here are past the parallel
thresholds (>= 128 docs and >= 1 Mi code points); on GIL builds the same tests check the serial path."""

from __future__ import annotations

import sys
from typing import Any

import pytest

import pyroid
import pyroid.dedup as D
import pyroid.filters as F
import pyroid.ftfy as ftfy
import pyroid.translit as T

FREE_THREADED = not getattr(sys, "_is_gil_enabled", lambda: True)()


def fresh(n: int = 400, size: int = 4000) -> list[str]:
    """New non-ASCII strs (no cached UTF-8): about n * size code points, all different objects."""
    return [f"{i} " + ("é ü ж 中 " * (size // 8)) + f" café {i}" for i in range(n)]


def notes(err: BaseException) -> list[str]:
    return list(getattr(err, "__notes__", []))


def test_free_threaded_flag_is_known() -> None:
    assert isinstance(FREE_THREADED, bool)


@pytest.mark.parametrize("kind", [list, tuple], ids=["list", "tuple"])
@pytest.mark.parametrize("threads", [None, 2, 8])
def test_same_results_and_objects_as_serial(kind: Any, threads: int | None) -> None:
    docs = kind(fresh())
    want = T.anyascii(kind(fresh()), threads=1)
    got = T.anyascii(docs, threads=threads)
    assert type(got) is type(want)
    assert got == want
    ascii_docs = kind(["plain ascii text " * 300 for _ in range(300)])
    out = D.simplify_text(ascii_docs, threads=threads)
    assert all(o is d for o, d in zip(ftfy.fix_text_batch(ascii_docs, threads=threads), ascii_docs, strict=True))
    assert out == D.simplify_text(ascii_docs, threads=1)


def test_every_module_matches_serial_on_fresh_strs() -> None:
    docs = fresh(300)
    assert ftfy.fix_text_batch(fresh(300)) == ftfy.fix_text_batch(docs, threads=1)
    assert D.minhash_signatures(fresh(300)) == D.minhash_signatures(docs, threads=1)
    assert D.exact_dedup(fresh(300)) == D.exact_dedup(docs, threads=1)
    assert F.gopher_quality(fresh(300)) == F.gopher_quality(docs, threads=1)
    assert T.unidecode(fresh(300)) == T.unidecode(docs, threads=1)


def test_repeated_and_subclassed_strs() -> None:
    class S(str):
        def __len__(self) -> int:  # never consulted: pyroid weighs only exact strs
            raise AssertionError("__len__ called")

    one = "é" * 5000
    docs = [one] * 300 + [S("ü" * 5000)] * 10 + fresh(50)
    assert T.anyascii(docs) == T.anyascii(docs, threads=1)
    assert D.exact_dedup(docs) == D.exact_dedup(docs, threads=1)


@pytest.mark.parametrize("bad", [[0], [7], [399], [5, 200, 390], [390, 5]], ids=str)
def test_lowest_surrogate_index_is_reported(bad: list[int]) -> None:
    docs = fresh()
    for i in bad:
        docs[i] = docs[i] + chr(0xD800)
    for threads in (None, 1, 3):
        with pytest.raises(UnicodeEncodeError) as info:
            D.exact_dedup(list(docs), threads=threads)
        assert any(f"the str at index {min(bad)} contains a lone surrogate" in n for n in notes(info.value))
        assert "while processing 'texts'" in notes(info.value)


def test_error_raised_inside_except_block_keeps_its_context() -> None:
    docs = fresh()
    docs[123] += chr(0xDC00)
    try:
        raise KeyError("outer")
    except KeyError:
        with pytest.raises(UnicodeEncodeError) as info:
            D.exact_dedup(docs)
    assert isinstance(info.value.__context__, KeyError)


def test_surrogate_before_a_non_str_wins_and_after_loses() -> None:
    docs: list[Any] = fresh()
    docs[100] += chr(0xD800)
    docs[300] = 42
    with pytest.raises(UnicodeEncodeError, match="surrogate"):
        D.exact_dedup(docs)
    mixed: list[Any] = fresh()
    mixed[300] += chr(0xD800)
    mixed[100] = None
    with pytest.raises(pyroid.InputError, match="expected str at index 100, got NoneType") as info:
        D.exact_dedup(mixed)
    assert "while processing 'texts'" in notes(info.value)


def test_bad_texts_reported_before_bad_threads() -> None:
    with pytest.raises(pyroid.InputError):
        D.exact_dedup(42, threads=0)  # type: ignore[call-overload]
    with pytest.raises(ValueError, match="threads must be None or a positive int") as info:
        D.exact_dedup(fresh(), threads=0)
    assert "while processing 'threads'" in notes(info.value)


def test_surrogates_still_take_the_fallback_paths() -> None:
    docs = fresh()
    docs[250] += chr(0xD800)
    assert T.anyascii(docs) == T.anyascii(docs, threads=1)  # anyascii drops lone surrogates
    assert ftfy.fix_text_batch(docs) == ftfy.fix_text_batch(docs, threads=1)  # ftfy repairs them
