"""pyroid.text: examples, container kinds and object identity.

The contract (docs/dev/MODULES.md): transforms return the caller's kind and
hand back unchanged documents as the *original* objects; predicates return a
mask of the same kind. A single str of >= 4096 UTF-8 bytes and batches of more
than 64 documents take different code paths (detached / thread pool) from
small inputs, so both sides of each threshold are covered.
"""

import subprocess
from collections.abc import Callable
from typing import Any

import pytest

import pyroid
import pyroid.text as t
from pyroid._pyroid import _testing

#: Thresholds in the bindings: a single str this long (in UTF-8 bytes) is
#: processed detached; a batch longer than this uses the thread pool.
DETACH_BYTES = 4096
INLINE_BATCH = 64


def ref(s: str) -> str:
    """The reference for normalize_whitespace."""
    return " ".join(s.split())


class MyStr(str):
    """A str subclass: accepted as a document, returned as itself if unchanged."""

    __slots__ = ()


# Every character Python's str.split() splits on (str.isspace()).
WHITESPACE = (
    "\t\n\x0b\x0c\r\x1c\x1d\x1e\x1f \x85\xa0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)


def test_whitespace_table_is_python_isspace() -> None:
    assert [c for c in map(chr, range(0x3001)) if c.isspace()] == list(WHITESPACE)


@pytest.mark.parametrize(
    ("doc", "want"),
    [
        ("  a \t b ", "a b"),
        ("", ""),
        (" ", ""),
        ("\n\n\t", ""),
        ("a", "a"),
        ("a b", "a b"),
        (" a", "a"),
        ("a ", "a"),
        ("a  b", "a b"),
        ("a\tb", "a b"),
        ("x\r\ny", "x y"),
        ("a\xa0b", "a b"),
        ("a\x1cb\x1fc", "a b c"),
        ("\u3000\u65e5\u672c\u2028\u8a9e\x85", "\u65e5\u672c \u8a9e"),
        ("a\u200bb", "a\u200bb"),  # zero-width space is not whitespace
        ("\ufeffa", "\ufeffa"),  # neither is a BOM
        ("a\x00 b", "a\x00 b"),  # NUL is a normal character
        ("\xe9  \xfc", "\xe9 \xfc"),
        ("\U0001f600 \t\U0001f601", "\U0001f600 \U0001f601"),
    ],
)
def test_normalize_whitespace_examples(doc: str, want: str) -> None:
    assert ref(doc) == want
    assert t.normalize_whitespace(doc) == want


@pytest.mark.parametrize("c", list(WHITESPACE), ids=[f"U+{ord(c):04X}" for c in WHITESPACE])
def test_every_whitespace_char_collapses(c: str) -> None:
    assert t.normalize_whitespace(f"{c}a{c}{c}b{c}") == "a b"


def test_modules_md_examples() -> None:
    docs = ["a  b", "ok"]
    out = t.normalize_whitespace(docs)
    assert out == ["a b", "ok"]
    assert out[1] is docs[1]
    assert t.min_chars(("abc", "x"), 2) == (True, False)
    assert t.normalize_whitespace("  a \t b ") == "a b"


# ---- container kinds -------------------------------------------------------


def test_str_in_str_out() -> None:
    out = t.normalize_whitespace(" a  b ")
    assert type(out) is str
    assert out == "a b"


@pytest.mark.parametrize("n", [0, 1, INLINE_BATCH, INLINE_BATCH + 1, 1000])
def test_list_in_list_out(n: int) -> None:
    docs = [f" doc  {i} " if i % 2 else f"doc {i}" for i in range(n)]
    out = t.normalize_whitespace(docs)
    assert type(out) is list
    assert out == [ref(d) for d in docs]
    mask = t.min_chars(docs, 7)
    assert type(mask) is list
    assert mask == [len(d) >= 7 for d in docs]
    assert all(type(m) is bool for m in mask)


@pytest.mark.parametrize("n", [0, 1, INLINE_BATCH, INLINE_BATCH + 1, 1000])
def test_tuple_in_tuple_out(n: int) -> None:
    docs = tuple(f" doc  {i} " if i % 3 else f"doc {i}" for i in range(n))
    out = t.normalize_whitespace(docs)
    assert type(out) is tuple
    assert out == tuple(ref(d) for d in docs)
    mask = t.min_chars(docs, 7)
    assert type(mask) is tuple
    assert mask == tuple(len(d) >= 7 for d in docs)


def test_input_is_not_modified() -> None:
    docs = ["a  b", " c", "d"]
    t.normalize_whitespace(docs)
    assert docs == ["a  b", " c", "d"]


def test_list_subclass_is_accepted() -> None:
    class MyList(list[str]):
        pass

    out = t.normalize_whitespace(MyList(["a  b", "c"]))
    assert isinstance(out, list)
    assert list(out) == ["a b", "c"]


# ---- identity of unchanged documents ---------------------------------------


@pytest.mark.parametrize(
    "doc",
    [
        "",
        "a",
        "a b c",
        "\u65e5\u672c \u8a9e",
        "x\u200by",
        "word " * 1000 + "end",  # >= 4096 bytes: the detached path
        ("\xe9 " * 2047) + "z",  # >= 4096 bytes, non-ASCII
    ],
    ids=["empty", "a", "words", "cjk", "zwsp", "long-ascii", "long-non-ascii"],
)
def test_unchanged_str_is_the_same_object(doc: str) -> None:
    assert ref(doc) == doc
    assert t.normalize_whitespace(doc) is doc


@pytest.mark.parametrize(
    "doc",
    [" ", "a  b", " a", "a\nb", "\xa0a", " word" * 1000, "\xe9  " * 2000],
    ids=["space", "double", "leading", "newline", "nbsp", "long-ascii", "long-non-ascii"],
)
def test_changed_str_is_a_new_object(doc: str) -> None:
    out = t.normalize_whitespace(doc)
    assert out == ref(doc)
    assert out is not doc
    assert type(out) is str


@pytest.mark.parametrize("nbytes", [DETACH_BYTES - 1, DETACH_BYTES, DETACH_BYTES + 1, 100_000])
def test_single_str_on_both_sides_of_the_detach_threshold(nbytes: int) -> None:
    unchanged = "a" * nbytes
    assert len(unchanged.encode()) == nbytes
    assert t.normalize_whitespace(unchanged) is unchanged
    changed = " " + "a" * (nbytes - 2) + " "
    assert t.normalize_whitespace(changed) == "a" * (nbytes - 2)
    assert t.min_chars(unchanged, nbytes) is True
    assert t.min_chars(unchanged, nbytes + 1) is False
    assert _testing.char_lengths(unchanged) == [nbytes]


@pytest.mark.parametrize("nchars", [DETACH_BYTES // 2 - 1, DETACH_BYTES // 2, DETACH_BYTES])
def test_threshold_counts_bytes_not_chars(nchars: int) -> None:
    # 2 UTF-8 bytes per char: min_chars must still count code points.
    doc = "\xe9" * nchars
    assert t.normalize_whitespace(doc) is doc
    assert t.min_chars(doc, nchars) is True
    assert t.min_chars(doc, nchars + 1) is False
    assert _testing.char_lengths(doc) == [nchars]


@pytest.mark.parametrize("n", [1, INLINE_BATCH, INLINE_BATCH + 1, 5000])
@pytest.mark.parametrize("kind", [list, tuple], ids=["list", "tuple"])
def test_batch_identity(kind: Callable[[list[str]], Any], n: int) -> None:
    raw = [f"doc {i}" if i % 2 == 0 else f"doc  {i}" for i in range(n)]
    docs = kind(raw)
    out = t.normalize_whitespace(docs)
    assert len(out) == n
    for i, (o, d) in enumerate(zip(out, docs, strict=True)):
        if i % 2 == 0:
            assert o is d, i
        else:
            assert o is not d, i
            assert type(o) is str
            assert o == ref(d)


def test_same_object_repeated_in_a_batch() -> None:
    s, u = "keep me", "change  me"
    out = t.normalize_whitespace([s, u, s, u] * 50)
    assert all(o is s for o in out[::2])
    assert out[1::2] == ["change me"] * 100


@pytest.mark.parametrize("n", [1, INLINE_BATCH + 1])
def test_str_subclass_documents(n: int) -> None:
    keep = MyStr("a b")
    change = MyStr("a  b")
    assert t.normalize_whitespace(keep) is keep
    out = t.normalize_whitespace(change)
    assert type(out) is str
    assert out == "a b"
    batch = t.normalize_whitespace([keep, change] * n)
    assert all(o is keep for o in batch[::2])
    assert all(type(o) is str for o in batch[1::2])
    assert batch[1::2] == ["a b"] * n
    assert t.min_chars([keep, change] * n, 4) == [False, True] * n


# ---- min_chars -------------------------------------------------------------


@pytest.mark.parametrize(
    ("doc", "n", "want"),
    [
        ("", 0, True),
        ("", 1, False),
        ("abc", 0, True),
        ("abc", 3, True),
        ("abc", 4, False),
        ("\U0001f600\U0001f600", 2, True),  # 8 bytes, 2 chars
        ("\U0001f600\U0001f600", 3, False),
        ("\U0001f600" * 4, 2, True),
        ("\xe9" * 100, 100, True),
        ("\xe9" * 100, 101, False),
        ("a\u0301", 2, True),  # code points, not grapheme clusters
        ("ab", 10**9, False),
        ("ab", 2**62, False),
    ],
)
def test_min_chars_examples(doc: str, n: int, want: bool) -> None:
    assert (len(doc) >= n) is want
    assert t.min_chars(doc, n) is want
    assert t.min_chars([doc], n) == [want]
    assert t.min_chars(doc, n=n) is want


def test_min_chars_keeps_order_in_large_batches() -> None:
    docs = ["x" * (i % 17) for i in range(3000)]
    assert t.min_chars(docs, 9) == [len(d) >= 9 for d in docs]
    assert t.min_chars(tuple(docs), 9) == tuple(len(d) >= 9 for d in docs)


# ---- map_values (via the private char_lengths hook) ------------------------


def test_char_lengths_str_list_tuple() -> None:
    assert _testing.char_lengths("h\xe9llo") == [5]
    assert _testing.char_lengths(["", "ab", "\U0001f600"]) == [0, 2, 1]
    assert _testing.char_lengths(("ab", "c")) == [2, 1]
    assert _testing.char_lengths([]) == []
    docs = [str(i) * (i % 13) for i in range(2000)]
    assert _testing.char_lengths(docs) == [len(d) for d in docs]


def test_char_lengths_arrow_nulls() -> None:
    col = _testing.column(["ab", None, "", "\xe9"])
    assert _testing.char_lengths(col) == [2, None, 0, 1]


# ---- module surface --------------------------------------------------------


def test_module_surface() -> None:
    assert t.__all__ == ["min_chars", "normalize_whitespace"]
    for f in (t.normalize_whitespace, t.min_chars):
        assert f.__module__ == "pyroid.text"
        assert f.__doc__


def test_results_are_plain_python_objects() -> None:
    # Nothing pyroid-specific leaks out of str/list/tuple calls.
    out = t.normalize_whitespace(["a"])
    assert not isinstance(out, pyroid.Column)


@pytest.mark.thread_unsafe(reason="runs a subprocess")
def test_import_pyroid_does_not_import_submodules(
    run_python: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    proc = run_python(
        """
        import sys, pyroid
        assert "pyroid.text" not in sys.modules, sorted(sys.modules)
        import pyroid.text
        assert pyroid.text.normalize_whitespace(" a ") == "a"
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
