"""pyroid.difflib against the running interpreter's difflib, on Hypothesis inputs.

Every comparison is exact: ratios bit for bit (``float.hex``), blocks with their type, opcodes,
the attributes (``b2j``, ``bjunk``, ``bpopular``, ``fullbcount``) with their key order, the calls
made to ``isjunk``/``linejunk``, every line of text output, and exceptions (type and message).
"""

from __future__ import annotations

import copy
import difflib
import os
import pickle
import sys
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import pyroid.difflib as pd
from pyroid import integrations

pytestmark = pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning")

COLOR = sys.version_info >= (3, 15)
# PYROID_DIFFLIB_EXAMPLES=n runs n examples per fuzz test (the parity counts in the docs).
_EXAMPLES = os.environ.get("PYROID_DIFFLIB_EXAMPLES")
_HEALTH = [HealthCheck.too_slow, HealthCheck.data_too_large]
FUZZ = (
    settings(suppress_health_check=_HEALTH, max_examples=int(_EXAMPLES))
    if _EXAMPLES
    else settings(suppress_health_check=_HEALTH)
)


def outcome(f: Callable[[], Any]) -> Any:
    """`f()`, or the exception it raised as (type name, message)."""
    try:
        return ("ok", f())
    except Exception as exc:
        return ("raised", type(exc).__name__, str(exc))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

any_char = st.characters(exclude_categories=())  # surrogates included
small_text = st.text(alphabet=st.sampled_from("ab c\t\n#"), max_size=40)
unicode_text = st.text(alphabet=any_char, max_size=30)
# Past 200 elements with few distinct ones: autojunk's popular elements.
long_text = st.text(alphabet=st.sampled_from("abcde "), min_size=190, max_size=400)
texts = st.one_of(small_text, small_text, unicode_text, long_text)

vocab = st.sampled_from(
    ["a\n", "b\n", "c\n", "\n", "# x\n", "  \n", "aaa\n", "abc", "\ud800\n", "é\n"]
)
line_lists = st.one_of(
    st.lists(vocab, max_size=25),
    st.lists(vocab, min_size=195, max_size=260),
    st.lists(small_text.map(lambda s: s + "\n"), max_size=12),
    st.lists(unicode_text, max_size=8),
)
ints = st.lists(st.one_of(st.integers(-3, 3), st.booleans(), st.just(2**70)), max_size=30)
bytes_items = st.lists(st.sampled_from([b"a", b"b", b"", b"ab"]), max_size=20)
mixed = st.lists(
    st.one_of(
        st.sampled_from(["a", "b"]), st.integers(0, 2), st.just(1.0), st.just((1,)), st.none()
    ),
    max_size=12,
)


def sequences() -> st.SearchStrategy[Any]:
    return st.one_of(
        texts,
        line_lists,
        line_lists.map(tuple),
        ints,
        st.binary(max_size=30),
        st.binary(max_size=30).map(bytearray),
        bytes_items,
        mixed,
        st.lists(st.sampled_from(["a", "b", " "]), max_size=20),  # characters as items
        st.lists(st.lists(st.integers(0, 1), max_size=1), max_size=3),  # unhashable
        st.sampled_from([range(5), "", [], ()]),
    )


class Junk:
    """An isjunk that records its calls (both libraries must make the same ones)."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.calls: list[Any] = []

    def __call__(self, x: Any) -> Any:
        self.calls.append(x)
        if self.kind == "space":
            return x == " "
        if self.kind == "vowel":
            return x in ("a", "e", 1)
        if self.kind == "raise" and x == "c":
            raise KeyError(x)
        if self.kind == "truthy":
            return [x] if x == "b" else []
        return False

    def __repr__(self) -> str:
        return f"Junk({self.kind!r})"


class Falsy(Junk):
    def __bool__(self) -> bool:  # `if isjunk:` skips it
        return False


def junk_factory() -> st.SearchStrategy[Callable[[], Any]]:
    return st.sampled_from(
        [
            lambda: None,
            lambda: Junk("space"),
            lambda: Junk("vowel"),
            lambda: Junk("raise"),
            lambda: Junk("truthy"),
            lambda: Falsy("space"),
            lambda: difflib.IS_CHARACTER_JUNK,
        ]
    )


# ---------------------------------------------------------------------------
# SequenceMatcher
# ---------------------------------------------------------------------------

OPS = st.lists(
    st.one_of(
        st.sampled_from(
            [
                ("ratio",),
                ("quick_ratio",),
                ("real_quick_ratio",),
                ("get_matching_blocks",),
                ("get_opcodes",),
                ("attr", "b2j"),
                ("attr", "bjunk"),
                ("attr", "bpopular"),
                ("attr", "fullbcount"),
                ("attr", "matching_blocks"),
                ("attr", "opcodes"),
                ("pickle",),
                ("copy",),
            ]
        ),
        st.tuples(st.just("grouped"), st.integers(-3, 6)),
        st.tuples(
            st.just("flm"),
            st.lists(
                st.one_of(st.none(), st.integers(-2, 40), st.just(True)), min_size=0, max_size=4
            ),
        ),
        st.tuples(st.just("seq1"), sequences()),
        st.tuples(st.just("seq2"), sequences()),
        st.tuples(st.just("same"), st.sampled_from(["a", "b"])),
    ),
    max_size=8,
)


def view(value: Any) -> Any:
    """A comparable form: Matches keep their type, sets their iteration order, dicts their order."""
    if isinstance(value, difflib.Match):
        return ("Match", tuple(value))
    if isinstance(value, (set, frozenset)):
        return ("set", [view(v) for v in value])
    if isinstance(value, dict):
        return ("dict", [(view(k), view(v)) for k, v in value.items()])
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [view(v) for v in value])
    if isinstance(value, float):
        return ("float", value.hex())
    return (type(value).__name__, value)


def step(f: Callable[..., Any], *args: Any) -> Any:
    """`outcome` of `view(f(*args))`."""
    return outcome(lambda: view(f(*args)))


def run_ops(cls: type, isjunk: Any, a: Any, b: Any, autojunk: Any, ops: list[Any]) -> list[Any]:
    log: list[Any] = []
    sm = outcome(lambda: cls(isjunk, a, b, autojunk))
    if sm[0] != "ok":
        return [sm]
    s = sm[1]
    for op in ops:
        kind = op[0]
        if kind == "attr":
            log.append(step(getattr, s, op[1]))
        elif kind == "grouped":
            log.append(step(lambda s, n: list(s.get_grouped_opcodes(n)), s, op[1]))
            log.append(step(lambda s: s.get_opcodes(), s))
        elif kind == "flm":
            log.append(step(lambda s, bounds: s.find_longest_match(*bounds), s, op[1]))
        elif kind == "seq1":
            log.append(step(lambda s, x: s.set_seq1(x), s, op[1]))
        elif kind == "seq2":
            log.append(step(lambda s, x: s.set_seq2(x), s, op[1]))
        elif kind == "same":  # set_seq1/set_seq2 with the current object: a no-op
            log.append(
                step(lambda s, w: s.set_seq1(s.a) if w == "a" else s.set_seq2(s.b), s, op[1])
            )
        elif kind == "pickle":  # a round trip of the matcher this test just made
            if isjunk is None:
                s = pickle.loads(pickle.dumps(s))
                log.append(type(s).__name__)
                log.append(sorted(k for k in vars(s) if not k.startswith("_pyroid_")))
        elif kind == "copy":
            s = copy.copy(s)
            log.append(step(lambda s: s.get_opcodes(), s))
        else:
            log.append(step(lambda s, name: getattr(s, name)(), s, kind))
    log.append(step(lambda s: s.ratio(), s))
    log.append(step(lambda s: s.get_opcodes(), s))
    return log


def compare_matchers(
    make_junk: Callable[[], Any], a: Any, b: Any, autojunk: Any, ops: list[Any]
) -> None:
    j1, j2 = make_junk(), make_junk()
    expected = run_ops(difflib.SequenceMatcher, j1, a, b, autojunk, ops)
    actual = run_ops(pd.SequenceMatcher, j2, a, b, autojunk, ops)
    assert actual == expected
    if isinstance(j1, Junk):
        assert view(j2.calls) == view(j1.calls)


@FUZZ
@given(junk_factory(), sequences(), sequences(), st.sampled_from([True, False, 0, 1, ""]), OPS)
def test_sequence_matcher(make_junk: Any, a: Any, b: Any, autojunk: Any, ops: list[Any]) -> None:
    compare_matchers(make_junk, a, b, autojunk, ops)


@FUZZ
@given(junk_factory(), texts, texts)
def test_string_pairs(make_junk: Any, a: str, b: str) -> None:
    compare_matchers(make_junk, a, b, True, [("quick_ratio",), ("get_matching_blocks",)])
    compare_matchers(make_junk, b, a, False, [("flm", [0, None, 0, None])])


@FUZZ
@given(line_lists, line_lists)
def test_line_lists(a: list[str], b: list[str]) -> None:
    compare_matchers(lambda: None, a, b, True, [("grouped", 3), ("attr", "bpopular")])


@FUZZ
@given(texts, texts, st.integers(0, 50), st.integers(0, 50), st.integers(0, 50), st.integers(0, 50))
def test_find_longest_match_bounds(a: str, b: str, alo: int, ahi: int, blo: int, bhi: int) -> None:
    for junk in (None, " ".__eq__):
        s1, s2 = difflib.SequenceMatcher(junk, a, b), pd.SequenceMatcher(junk, a, b)
        expected = step(s1.find_longest_match, alo, ahi, blo, bhi)
        assert step(s2.find_longest_match, alo, ahi, blo, bhi) == expected


def test_documented_examples() -> None:
    s = pd.SequenceMatcher(lambda x: x == " ", "private Thread currentThread;",
                           "private volatile Thread currentThread;")  # fmt: skip
    assert s.ratio() == 58 / 67
    assert s.get_matching_blocks() == [(0, 0, 8), (8, 17, 21), (29, 38, 0)]
    assert isinstance(s, difflib.SequenceMatcher)
    assert pd.SequenceMatcher(None, "abcd", "bcde").ratio() == 0.75


class Overriding(difflib.SequenceMatcher[str]):
    def find_longest_match(
        self, alo: int = 0, ahi: Any = None, blo: int = 0, bhi: Any = None
    ) -> Any:
        self.calls = getattr(self, "calls", 0) + 1
        return super().find_longest_match(alo, ahi, blo, bhi)


class PyroidOverriding(pd.SequenceMatcher[str]):
    def find_longest_match(
        self, alo: int = 0, ahi: Any = None, blo: int = 0, bhi: Any = None
    ) -> Any:
        self.calls = getattr(self, "calls", 0) + 1
        return super().find_longest_match(alo, ahi, blo, bhi)


@FUZZ
@given(texts, texts)
def test_subclass_overrides_are_called(a: str, b: str) -> None:
    s1, s2 = Overriding(None, a, b), PyroidOverriding(None, a, b)
    assert view(s2.get_opcodes()) == view(s1.get_opcodes())
    assert s2.ratio() == s1.ratio()
    assert getattr(s2, "calls", 0) == getattr(s1, "calls", 0)


# ---------------------------------------------------------------------------
# get_close_matches
# ---------------------------------------------------------------------------

words = st.text(alphabet=st.sampled_from("abcdé\ud800"), max_size=8)


@FUZZ
@given(
    st.one_of(words, st.lists(st.sampled_from(["a", "b"]), max_size=4)),
    st.lists(st.one_of(words, words, st.sampled_from(["apple", "ape", "peach"])), max_size=40),
    st.sampled_from([1, 2, 3, 5, 100, True, 0, -1, 2.5, 10**30]),
    st.sampled_from([0.0, 0.3, 0.6, 0.75, 1.0, 0, 1, -0.1, 1.5, float("nan")]),
    st.sampled_from(["list", "tuple", "iter", "dict", "set"]),
)
def test_get_close_matches(
    word: Any, possibilities: list[str], n: Any, cutoff: Any, kind: str
) -> None:
    def make() -> Any:
        if kind == "tuple":
            return tuple(possibilities)
        if kind == "iter":
            return iter(possibilities)
        if kind == "dict":
            return dict.fromkeys(possibilities)
        if kind == "set":
            return set_
        return possibilities

    set_ = set(possibilities)
    expected = outcome(lambda: difflib.get_close_matches(word, make(), n, cutoff))
    actual = outcome(lambda: pd.get_close_matches(word, make(), n, cutoff))
    assert actual == expected
    if actual[0] == "ok" and kind == "list":  # the possibilities' own objects come back
        ids = {id(p) for p in possibilities}
        assert all(id(x) in ids for x in actual[1])
        assert pd.get_close_matches(word, possibilities, n, cutoff, threads=1) == actual[1]


def test_get_close_matches_ties() -> None:
    # Equal scores: the larger string first; equal strings: the earlier first (heapq.nlargest).
    possibilities = ["ab", "ba", "ab", "aa", "bb"] * 30
    for n in (1, 2, 3, 7, 1000):
        assert pd.get_close_matches("ab", possibilities, n, 0.0) == difflib.get_close_matches(
            "ab", possibilities, n, 0.0
        )


# ---------------------------------------------------------------------------
# Differ, ndiff, restore, _mdiff, HtmlDiff
# ---------------------------------------------------------------------------


def line_junk_factory() -> st.SearchStrategy[Callable[[], Any]]:
    return st.sampled_from(
        [lambda: None, lambda: difflib.IS_LINE_JUNK, lambda: Junk("vowel"), lambda: Falsy("x")]
    )


char_junk = st.sampled_from([None, difflib.IS_CHARACTER_JUNK, lambda c: c in "ae", " ".__eq__])
fancy_lines = st.lists(
    st.one_of(
        vocab,
        st.text(alphabet=st.sampled_from("abcd \t"), min_size=1, max_size=12).map(
            lambda s: s + "\n"
        ),
    ),
    max_size=16,
)
diff_inputs = st.one_of(fancy_lines, line_lists, fancy_lines.map(tuple))


@FUZZ
@given(diff_inputs, diff_inputs, line_junk_factory(), char_junk)
def test_ndiff(a: list[str], b: list[str], make_junk: Any, charjunk: Any) -> None:
    j1, j2 = make_junk(), make_junk()
    expected = outcome(lambda: list(difflib.ndiff(a, b, j1, charjunk)))
    actual = outcome(lambda: list(pd.ndiff(a, b, j2, charjunk)))
    assert actual == expected
    if isinstance(j1, Junk):
        assert j2.calls == j1.calls
    if expected[0] == "ok":
        for which in (1, 2):
            assert list(pd.restore(actual[1], which)) == list(difflib.restore(expected[1], which))


@FUZZ
@given(diff_inputs, diff_inputs, st.one_of(st.none(), st.integers(0, 4)), char_junk)
def test_mdiff(a: list[str], b: list[str], context: Any, charjunk: Any) -> None:
    expected = outcome(lambda: list(difflib._mdiff(a, b, context, None, charjunk)))  # type: ignore[attr-defined]
    assert outcome(lambda: list(pd._mdiff(a, b, context, None, charjunk))) == expected  # type: ignore[attr-defined]


class Quiet(difflib.Differ):
    def _qformat(self, aline: str, bline: str, atags: str, btags: str) -> Iterator[str]:
        yield f"~ {aline}"


class PyroidQuiet(pd.Differ):
    def _qformat(self, aline: str, bline: str, atags: str, btags: str) -> Iterator[str]:
        yield f"~ {aline}"


@FUZZ
@given(fancy_lines, fancy_lines)
def test_differ_subclass(a: list[str], b: list[str]) -> None:
    assert list(PyroidQuiet().compare(a, b)) == list(Quiet().compare(a, b))
    assert isinstance(pd.Differ(), difflib.Differ)


html_lines = st.lists(
    st.one_of(
        vocab,
        st.text(alphabet=st.sampled_from("ab \t<&>\x00\x01é\ud800"), max_size=30).map(
            lambda s: s + "\n"
        ),
    ),
    max_size=14,
)


@pytest.mark.thread_unsafe(reason="HtmlDiff's table prefix counter is global")
@FUZZ
@given(
    st.one_of(fancy_lines, html_lines),
    st.one_of(fancy_lines, html_lines),
    st.sampled_from([8, 2, 1, 0, -1]),
    st.sampled_from([None, 14, 5, 1, 0]),
    st.booleans(),
    st.integers(0, 3),
    char_junk,
    line_junk_factory(),
    st.sampled_from([("from", "to"), ("", ""), ("a\tb<", ""), ("", "x\x00+y")]),
)
def test_html_diff(
    a: list[str],
    b: list[str],
    tabsize: int,
    wrap: Any,
    context: bool,
    numlines: int,
    cj: Any,
    make_junk: Any,
    descs: tuple[str, str],
) -> None:
    def html(module: Any) -> Any:
        module.HtmlDiff._default_prefix = 0
        h = module.HtmlDiff(tabsize=tabsize, wrapcolumn=wrap, linejunk=make_junk(), charjunk=cj)
        return (
            h.make_table(a, b, *descs, context=context, numlines=numlines),
            h.make_file(a, b, context=context, numlines=numlines, charset="latin-1"),
            h.make_table(a, a, context=True),
            h.make_table([], [], context=context),
        )

    assert outcome(lambda: html(pd)) == outcome(lambda: html(difflib))


# ---------------------------------------------------------------------------
# unified_diff, context_diff, diff_bytes
# ---------------------------------------------------------------------------

names = st.one_of(st.sampled_from(["", "a.txt", "b\ud800"]), st.text(max_size=5))


@FUZZ
@given(
    st.one_of(
        diff_inputs, st.lists(st.one_of(vocab, st.just(b"x\n"), st.just(1)), max_size=3), small_text
    ),
    diff_inputs,
    names,
    names,
    st.integers(-3, 6),
    st.sampled_from(["\n", "", "\r\n", "\ud800"]),
    st.booleans(),
)
def test_unified_and_context(
    a: Any, b: Any, f1: str, f2: str, n: int, lineterm: str, color: bool
) -> None:
    for name in ("unified_diff", "context_diff"):
        kw: dict[str, Any] = {"color": color} if COLOR and name == "unified_diff" else {}
        args = (a, b, f1, f2, "d1", "", n, lineterm)
        expected = step(lambda f, args, kw: list(f(*args, **kw)), getattr(difflib, name), args, kw)
        assert (
            step(lambda f, args, kw: list(f(*args, **kw)), getattr(pd, name), args, kw) == expected
        )


def test_lazy_errors() -> None:
    # As generators, both raise at the first next(), not at the call.
    g = pd.unified_diff([1], ["a"])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="lines to compare must be str"):
        next(g)
    g2 = pd.ndiff([["unhashable"]], [["x"]])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="unhashable"):
        next(g2)


@FUZZ
@given(
    st.lists(st.binary(max_size=6).map(lambda s: s + b"\n"), max_size=10),
    st.lists(st.sampled_from([b"a\n", b"\xff\n"]), max_size=10),
)
def test_diff_bytes(a: list[bytes], b: list[bytes]) -> None:
    for d1, d2 in (
        (difflib.unified_diff, pd.unified_diff),
        (difflib.context_diff, pd.context_diff),
    ):
        expected = list(difflib.diff_bytes(d1, a, b, b"x", b"y\xff"))
        assert list(pd.diff_bytes(d2, a, b, b"x", b"y\xff")) == expected


# ---------------------------------------------------------------------------
# Under pyroid.patch("difflib")
# ---------------------------------------------------------------------------


@pytest.mark.thread_unsafe(reason="patches the standard library")
@FUZZ
@given(texts, texts, fancy_lines, fancy_lines)
def test_patched(a: str, b: str, la: list[str], lb: list[str]) -> None:
    before = Overriding(None, a, b)  # a subclass made before the patch
    expected = (
        view(
            run_ops(difflib.SequenceMatcher, None, a, b, True, [("get_opcodes",), ("quick_ratio",)])
        ),
        list(difflib.ndiff(la, lb)),
        list(difflib.unified_diff(la, lb)),
        view(before.get_opcodes()),
        difflib.get_close_matches(a, [b, a[:3], "x"]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", integrations.PatchWarning)
        with integrations.patching("difflib"):
            assert difflib.SequenceMatcher.ratio is pd.SequenceMatcher.ratio
            after = Overriding(None, a, b)
            actual = (
                view(
                    run_ops(
                        difflib.SequenceMatcher,
                        None,
                        a,
                        b,
                        True,
                        [("get_opcodes",), ("quick_ratio",)],
                    )
                ),
                list(difflib.ndiff(la, lb)),
                list(difflib.unified_diff(la, lb)),
                view(after.get_opcodes()),
                difflib.get_close_matches(a, [b, a[:3], "x"]),
            )
            assert isinstance(after, difflib.SequenceMatcher)
    assert actual == expected
    assert "b2j" not in difflib.SequenceMatcher.__dict__  # unpatched again
