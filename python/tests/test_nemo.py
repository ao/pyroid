"""pyroid.nemo's filter scores and fastText helpers against NeMo Curator 1.4.0.

- **Reference parity**: every score function against ``_nemo_reference.py`` (NeMo's own
  ``score_document`` code, vendored), with Hypothesis: the same ``repr`` of every score, the same
  exception type for the first failing document. Runs on every interpreter.
- **Live parity**: the same against NeMo Curator itself where it is installed. Without its Ray
  dependencies, NeMo's filter modules are run from the installed package with stand-ins for the
  parent packages (their ``__init__`` imports cosmos-xenna and Ray; the filters need neither).
- ``repeating_top_ngrams``' tie path, in child interpreters with random ``PYTHONHASHSEED``s.
- Container kinds, Arrow nulls, errors and their notes, ``threads``, a 10k-document batch, the
  ``str.lower()`` / ``str.isspace()`` / ``[[:alpha:]]`` facts the kernels rely on, and the
  fastText helpers on a synthetic model (and ``lid.176.ftz`` when ``$PYROID_DATA`` has it).
"""

from __future__ import annotations

import ast
import contextlib
import functools
import importlib
import importlib.util
import json
import math
import os
import random
import struct
import subprocess
import sys
import sysconfig
import types
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is needed for batch results
    pytest.skip("pyroid.nemo returns numpy arrays for batches", allow_module_level=True)

import pyroid
import pyroid.fasttext as ft
import pyroid.nemo as nemo
from pyroid._pyroid import _testing

with warnings.catch_warnings():
    # jieba 0.42.1 (the reference for lang="zh") has invalid escape sequences: a SyntaxWarning
    # when first compiled. Import it once here, not in a test.
    warnings.simplefilter("ignore")
    with contextlib.suppress(ImportError):
        importlib.import_module("jieba")

#: pyroid.nemo, for its private helpers (not in the stub).
_nemo: Any = importlib.import_module("pyroid.nemo")
FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
DATA = Path(os.environ.get("PYROID_DATA", "/data"))
HERE = Path(__file__).resolve()


def _load_reference() -> Any:
    name = "_pyroid_test_nemo_reference"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE.with_name("_nemo_reference.py"))
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def reference() -> Any:
    """The vendored copy of NeMo's filters (needs the ``regex`` module)."""
    return _load_reference()


#: NeMo's package and the parents of the filter modules.
_NEMO_PACKAGES = (
    "nemo_curator",
    "nemo_curator.stages",
    "nemo_curator.stages.text",
    "nemo_curator.stages.text.utils",
    "nemo_curator.stages.text.filters",
    "nemo_curator.stages.text.filters.heuristic",
    "nemo_curator.stages.text.filters.heuristic.repetition",
    "nemo_curator.stages.text.filters.fasttext",
)
_FILTER_MODULES = (
    "nemo_curator.stages.text.filters.heuristic.string",
    "nemo_curator.stages.text.filters.heuristic.repetition.repetition",
)
_FASTTEXT_MODULE = "nemo_curator.stages.text.filters.fasttext.fasttext_filters"


@functools.cache
def live_nemo() -> Any | None:
    """NeMo Curator's own filter classes, or None where it is not installed.

    ``PYROID_NEMO_LIVE=0`` leaves NeMo out even where it is installed.
    """
    if os.environ.get("PYROID_NEMO_LIVE") == "0":
        return None
    modules: list[types.ModuleType] = []
    try:
        modules = [importlib.import_module(m) for m in _FILTER_MODULES]
    except ImportError:
        spec = importlib.util.find_spec("nemo_curator")
        if spec is None or not spec.submodule_search_locations:
            return None
        root = Path(next(iter(spec.submodule_search_locations)))
        for name in _NEMO_PACKAGES:
            if name not in sys.modules:
                pkg = types.ModuleType(name)
                pkg.__path__ = [str(root.joinpath(*name.split(".")[1:]))]
                sys.modules[name] = pkg
        modules = [importlib.import_module(m) for m in _FILTER_MODULES]
    with contextlib.suppress(ImportError):  # fastText is not installed
        modules.append(importlib.import_module(_FASTTEXT_MODULE))
    ns = types.SimpleNamespace()
    for module in modules:
        for key, value in vars(module).items():
            if isinstance(value, type) and key[:1].isupper():
                setattr(ns, key, value)
    return ns


@pytest.fixture(scope="session")
def ref() -> Any:
    pytest.importorskip("regex")
    return reference()


@pytest.fixture(scope="session")
def live() -> Any:
    ns = live_nemo()
    if ns is None:
        pytest.skip("NeMo Curator is not installed")
    return ns


# ---- inputs ----------------------------------------------------------------------------

#: Every character ``str.isspace()`` accepts.
WHITESPACE = (
    "\t\n\x0b\x0c\r\x1c\x1d\x1e\x1f \x85\xa0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)

#: Pieces that reach the filters' branches.
PIECES = (
    "",
    "\n",
    "\n\n",
    "\n\n\n",
    "\r\n",
    "\b",
    "\u200b",
    "#",
    "##",
    "###",
    "a#b",
    "#a#",
    "...",
    "[...]",
    "…",
    "(...)",
    "[…]",
    "-»",
    "read more",
    "read more..",
    "READ MORE",
    "Read More..",
    "•",
    "‣",
    "\u2043",
    "⁌",
    "⁍",
    "∙",
    "○",
    "●",
    "◘",
    "◦",
    "⦾",
    "⦿",
    "-",
    "*",
    "\u0130",
    "\u212a",  # KELVIN SIGN
    "\u017f",
    "Σ",
    "ß",
    "ǅ",
    "\U0001f600",
    "\U0001d518",
    "٣",
    "²",
    "Ⅻ",
    "http://",
    "https://",
    "http://a.b/c?d=e&f#g",
    "https://x.org/%2F%zz%4",
    "http://x/%41%4",
    "httpss://y",
    "HTTP://Z",
    "porn",
    "PORN",
    "http://porn.example",
    "https://x.org/(p)orn!,*",
    "{",
    "}",
    "⟨",
    "⟩",
    "⟪",
    "[",
    "]",
    "(",
    ")",
    ".",
    "?",
    "!",
    '"',
    "'",
    ",",
    "the",
    "be",
    "to",
    "of",
    "and",
    "that",
    "have",
    "with",
    "The",
    "terms of use",
    "Privacy Policy",
    "cookie policy",
    "uses cookies",
    "use of cookies",
    "use cookies",
    "privacy & cookies policy",
    "privacy and cookies policy",
    "lorem ipsum",
    "Lorem Ipsum",
    "privacy overview",
    "a",
    "b",
    "ab",
    "abc",
    "x y",
    "中文",
    "我来到北京清华大学",
    "日本語",
    "0",
    "123",
    "1.5",
    "**a**",
    "***a**",
    "**a*b**",
    "[a](b)(c)",
    "_x_",
)

#: Characters with a UTF-8 form (pyroid rejects lone surrogates: see test_errors).
CHARS = st.characters(codec="utf-8")
_piece = st.one_of(
    st.sampled_from(PIECES),
    st.sampled_from(WHITESPACE),
    CHARS,
    st.text(CHARS, max_size=3),
)
#: Arbitrary documents built from the pieces above (no lone surrogates).
TEXT = st.lists(_piece, max_size=24).map("".join)

_WORDS = ("a", "b", "ab", "the", "of", "#", "...", "é", "中文", "x", "a b")
_SEPS = (" ", "\xa0", "\u2003", "  ", "\n", "\n\n", "\t", "\u3000", "\r\n", " \n ")


@st.composite
def _repetitive(draw: st.DrawFn) -> str:
    """Documents from a small vocabulary: repeated lines, paragraphs and n-grams, and ties."""
    words = draw(st.lists(st.sampled_from(_WORDS), max_size=40))
    lead = draw(st.sampled_from(("", " ", "\n")))
    return lead + "".join(w + draw(st.sampled_from(_SEPS)) for w in words)


DOC = st.one_of(TEXT, _repetitive())
DOCS = st.lists(DOC, max_size=8)


def tie_corpus(seed: int, n_docs: int = 300) -> list[str]:
    """Documents whose most frequent n-grams are mostly tied (several words of each length)."""
    rng = random.Random(seed)
    vocab = [
        "a",
        "bb",
        "ccc",
        "dddd",
        "e",
        "ff",
        "the",
        "of",
        "and",
        "x y",
        "été",
        "z",
        "q",
        "rr",
    ]
    seps = [" ", " ", " ", "  ", "\n", "\t"]
    docs = []
    for _ in range(n_docs):
        k = rng.randrange(1, 60)
        docs.append("".join(rng.choice(vocab) + rng.choice(seps) for _ in range(k)))
    return docs


#: Characters of every PEP 393 width (1, 2 and 4 bytes per character), for the hash checks.
_WIDTH_CHARS = "ab~\x00\x7f\x80\xe9\xff\u0100\u4e2d\ud7ff\uffff\U0001f600\U0010ffff"


def random_word(rng: random.Random) -> str:
    """A short word whose widest character is of a random PEP 393 width."""
    pool = _WIDTH_CHARS[: rng.choice((3, 8, 12, len(_WIDTH_CHARS)))]
    return "".join(rng.choices(pool, k=rng.randrange(1, 12)))


def set_order_mismatch(words: list[str], n: int, key: bytes) -> int:
    """1 if pyroid's simulated ``list(set(ngrams))`` differs from this process's, else 0."""
    ngrams = list(zip(*[words[i:] for i in range(n)], strict=False))
    first = dict(zip(reversed(ngrams), reversed(range(len(ngrams))), strict=True))
    want = [first[g] for g in set(ngrams)]
    return int(_nemo._native._ngram_set_order(words, n, key) != want)


# ---- the score functions and their NeMo classes ----------------------------------------


@dataclass(frozen=True)
class Case:
    """A pyroid score function and the NeMo filter it reproduces."""

    name: str
    #: Build the NeMo filter from a module namespace (the reference, or NeMo itself).
    nemo_filter: Callable[[Any], Any]
    #: ``pyroid.nemo``'s function: ``(texts, threads=...) -> score(s)``.
    pyroid: Callable[..., Any]
    integer: bool = False


def _filter(name: str, *args: Any, **kwargs: Any) -> Callable[[Any], Any]:
    """Build NeMo's filter ``name(*args, **kwargs)`` from a module namespace."""
    return lambda module: getattr(module, name)(*args, **kwargs)


def _cases() -> list[Case]:
    n = nemo
    cases = [
        Case("non_alpha_numeric", _filter("NonAlphaNumericFilter"), n.non_alpha_numeric),
        Case("symbols_to_words", _filter("SymbolsToWordsFilter"), n.symbols_to_words),
        Case("numbers", _filter("NumbersFilter"), n.numbers),
        Case("urls", _filter("UrlsFilter"), n.urls),
        Case("white_space", _filter("WhiteSpaceFilter"), n.white_space),
        Case("parentheses", _filter("ParenthesesFilter"), n.parentheses),
        Case("boilerplate_string", _filter("BoilerPlateStringFilter"), n.boilerplate_string),
        Case("repeated_lines", _filter("RepeatedLinesFilter"), n.repeated_lines),
        Case("repeated_paragraphs", _filter("RepeatedParagraphsFilter"), n.repeated_paragraphs),
        Case(
            "repeated_lines_by_char",
            _filter("RepeatedLinesByCharFilter"),
            n.repeated_lines_by_char,
        ),
        Case(
            "repeated_paragraphs_by_char",
            _filter("RepeatedParagraphsByCharFilter"),
            n.repeated_paragraphs_by_char,
        ),
        Case("word_count", _filter("WordCountFilter"), n.word_count, integer=True),
        Case("punctuation", _filter("PunctuationFilter"), n.punctuation),
        Case(
            "words_without_alphabets",
            _filter("WordsWithoutAlphabetsFilter"),
            n.words_without_alphabets,
        ),
        Case("mean_word_length", _filter("MeanWordLengthFilter"), n.mean_word_length),
        Case("long_word", _filter("LongWordFilter"), n.long_word, integer=True),
        Case("ellipsis", _filter("EllipsisFilter"), n.ellipsis),
        Case("bullets", _filter("BulletsFilter"), n.bullets),
        Case(
            "pornographic_urls",
            _filter("PornographicUrlsFilter"),
            n.pornographic_urls,
            integer=True,
        ),
    ]
    for cutoff, stop in ((2, True), (2, False), (0, True), (-1, True), (5, True)):
        cases.append(
            Case(
                f"common_english_words[{cutoff}-{stop}]",
                _filter("CommonEnglishWordsFilter", cutoff, stop),
                functools.partial(
                    n.common_english_words, min_num_common_words=cutoff, stop_at_false=stop
                ),
                integer=True,
            )
        )
    cases.extend(
        Case(
            f"repeating_top_ngrams[{k}]",
            _filter("RepeatingTopNGramsFilter", k),
            functools.partial(n.repeating_top_ngrams, n=k),
        )
        for k in (1, 2, 3, 4)
    )
    cases.extend(
        Case(
            f"repeating_duplicate_ngrams[{k}]",
            _filter("RepeatingDuplicateNGramsFilter", k),
            functools.partial(n.repeating_duplicate_ngrams, n=k),
        )
        for k in (0, -1, 1, 2, 3, 5)
    )
    cases.append(
        Case(
            "repeating_top_ngrams[0]",
            _filter("RepeatingTopNGramsFilter", 0),
            functools.partial(n.repeating_top_ngrams, n=0),
        )
    )
    for sub, pos in (
        ("a", "prefix"),
        ("", "suffix"),
        ("the", "any"),
        ("\n", "any"),
        ("é", "suffix"),
    ):
        cases.append(
            Case(
                f"substring[{sub!r}-{pos}]",
                _filter("SubstringFilter", sub, pos),
                functools.partial(n.substring, substring=sub, position=pos),
                integer=True,
            )
        )
    return cases


def _zh_cases() -> list[Case]:
    n = nemo
    zh = {"lang": "zh"}
    return [
        Case(
            "zh-symbols_to_words",
            _filter("SymbolsToWordsFilter", lang="zh"),
            functools.partial(n.symbols_to_words, **zh),
        ),
        Case(
            "zh-word_count",
            _filter("WordCountFilter", lang="ZH"),
            functools.partial(n.word_count, lang="ZH"),
            integer=True,
        ),
        Case(
            "zh-words_without_alphabets",
            _filter("WordsWithoutAlphabetsFilter", lang="zh"),
            functools.partial(n.words_without_alphabets, **zh),
        ),
        Case(
            "zh-mean_word_length",
            _filter("MeanWordLengthFilter", lang="zh"),
            functools.partial(n.mean_word_length, **zh),
        ),
        Case(
            "zh-long_word",
            _filter("LongWordFilter", lang="zh"),
            functools.partial(n.long_word, **zh),
            integer=True,
        ),
        Case(
            "zh-repeating_top_ngrams[2]",
            _filter("RepeatingTopNGramsFilter", 2, lang="zh"),
            functools.partial(n.repeating_top_ngrams, n=2, **zh),
        ),
        Case(
            "zh-repeating_duplicate_ngrams[2]",
            _filter("RepeatingDuplicateNGramsFilter", 2, lang="zh"),
            functools.partial(n.repeating_duplicate_ngrams, n=2, **zh),
        ),
    ]


CASES = _cases()
ZH_CASES = _zh_cases()
_ids: Callable[[Case], str] = lambda c: c.name  # noqa: E731

#: An outcome: ("ok", repr of the score) or ("raise", the exception type's name).
Outcome = tuple[str, str]


def nemo_outcome(filt: Any, text: str) -> Outcome:
    try:
        value = filt.score_document(text)
    except Exception as exc:
        return ("raise", type(exc).__name__)
    return ("ok", repr(value))


def scalar_outcome(case: Case, text: str) -> Outcome:
    try:
        value = case.pyroid(text)
    except Exception as exc:
        return ("raise", type(exc).__name__)
    assert type(value) is (int if case.integer else float), (case.name, value)
    return ("ok", repr(value))


def as_reprs(case: Case, result: Any) -> list[str]:
    """A batch result's scores, checked to be a 1-D array of the documented dtype."""
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert result.dtype == (np.int64 if case.integer else np.float64)
    if case.integer:
        return [repr(int(x)) for x in result]
    return [repr(float(x)) for x in result]


def check_batch(case: Case, filt: Any, docs: Sequence[str], got: Callable[[], Any]) -> None:
    """``got()`` gives NeMo's scores for ``docs``, or raises for the first failing one."""
    want = [nemo_outcome(filt, d) for d in docs]
    failing = [i for i, w in enumerate(want) if w[0] == "raise"]
    if failing:
        i = failing[0]
        with pytest.raises(Exception) as info:  # noqa: PT011 - the type is compared below
            got()
        assert type(info.value).__name__ == want[i][1], case.name
        assert f"pyroid: raised for the document at index {i}" in getattr(
            info.value, "__notes__", []
        )
        return
    assert as_reprs(case, got()) == [w[1] for w in want], case.name


# ---- reference parity ------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=_ids)
@given(text=DOC)
def test_reference_parity_str(ref: Any, case: Case, text: str) -> None:
    assert scalar_outcome(case, text) == nemo_outcome(case.nemo_filter(ref), text)


@pytest.mark.parametrize("case", CASES, ids=_ids)
@given(docs=DOCS)
def test_reference_parity_batches(ref: Any, case: Case, docs: list[str]) -> None:
    filt = case.nemo_filter(ref)
    check_batch(case, filt, docs, lambda: case.pyroid(docs))
    check_batch(case, filt, docs, lambda: case.pyroid(tuple(docs)))
    check_batch(case, filt, docs, lambda: case.pyroid(_testing.column(docs, chunks=2)))


@pytest.mark.parametrize("case", ZH_CASES, ids=_ids)
@given(text=DOC)
def test_reference_parity_zh(ref: Any, case: Case, text: str) -> None:
    assert scalar_outcome(case, text) == nemo_outcome(case.nemo_filter(ref), text)


@pytest.mark.parametrize("case", CASES + ZH_CASES, ids=_ids)
def test_reference_parity_fixed(ref: Any, case: Case) -> None:
    """Every piece alone, after a word, before a newline; and the tie corpus."""
    filt = case.nemo_filter(ref)
    docs = [*PIECES, *(f"x {p}" for p in PIECES), *(f"{p}\nok." for p in PIECES)]
    docs += [*WHITESPACE, *tie_corpus(1, 60)]
    for d in docs:
        assert scalar_outcome(case, d) == nemo_outcome(filt, d), (case.name, d)
    ok = [d for d in docs if nemo_outcome(filt, d)[0] == "ok"]
    check_batch(case, filt, ok, lambda: case.pyroid(ok))


def test_constants_are_nemos(ref: Any) -> None:
    """Each ellipsis mark, bullet, end mark and policy substring scores as in NeMo."""
    probes: list[tuple[Case, str]] = []
    by_name = {c.name: c for c in CASES}
    for mark in ref.ellipsis_marks:
        probes += [(by_name["ellipsis"], f"x{mark}"), (by_name["symbols_to_words"], mark)]
        probes.append((by_name["ellipsis"], f"x{mark.upper()} "))
    for bullet in ref.bullet_list:
        probes.append((by_name["bullets"], f" {bullet}x\ny"))
    for end in ref.end_marks:
        probes.append((by_name["punctuation"], f"a{end} \nb"))
    for policy in ref.policy_substrings:
        probes += [
            (by_name["boilerplate_string"], policy.upper()),
            (by_name["boilerplate_string"], f"ok\n\n{policy[1:]}"),
        ]
    for word in ref.common_english_words:
        probes.append((by_name["common_english_words[2-False]"], f"{word} {word.title()} x"))
    for case, text in probes:
        assert scalar_outcome(case, text) == nemo_outcome(case.nemo_filter(ref), text), (
            case.name,
            text,
        )


# ---- live NeMo -------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES + ZH_CASES, ids=_ids)
@given(text=DOC)
def test_nemo_parity_str(live: Any, case: Case, text: str) -> None:
    assert scalar_outcome(case, text) == nemo_outcome(case.nemo_filter(live), text)


@pytest.mark.parametrize("case", CASES, ids=_ids)
@given(docs=DOCS)
def test_nemo_parity_batches(live: Any, case: Case, docs: list[str]) -> None:
    check_batch(case, case.nemo_filter(live), docs, lambda: case.pyroid(docs))


def test_reference_is_nemos_code(live: Any) -> None:
    """The vendored constants are NeMo's."""
    pytest.importorskip("regex")
    ref = reference()
    nemo_constants = importlib.import_module("nemo_curator.stages.text.utils.constants")
    for name in (
        "end_marks",
        "ellipsis_marks",
        "policy_substrings",
        "white_space_list",
        "common_english_words",
        "bullet_list",
    ):
        assert getattr(ref, name) == getattr(nemo_constants, name), name
    for name in ("regex_alpha", "regex_digit", "regex_alphanum", "regex_url", "regex_paren"):
        assert getattr(ref, name).pattern == getattr(nemo_constants, name).pattern, name


# ---- RepeatingTopNGramsFilter's ties ---------------------------------------------------

_SEED_SCRIPT = """
import importlib.util, json, random, sys
spec = importlib.util.spec_from_file_location("_pyroid_test_nemo_child", sys.argv[1])
t = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = t
spec.loader.exec_module(t)
import pyroid.nemo as nemo

docs = t.tie_corpus(int(sys.argv[2]))
impls = {"reference": t.reference()}
live = t.live_nemo()
if live is not None:
    impls["nemo"] = live
key = nemo._hash_key()
out = {"impls": sorted(impls), "key": key is not None, "scores": {}, "mismatches": {}}
for n in (1, 2, 3, 4):
    batch = [float(x) for x in nemo.repeating_top_ngrams(docs, n)]
    one = [nemo.repeating_top_ngrams(d, n) for d in docs]
    out["scores"][n] = batch
    for name, m in impls.items():
        f = m.RepeatingTopNGramsFilter(n)
        want = [f.score_document(d) for d in docs]
        bad = sum(a != w for a, w in zip(batch, want)) + sum(a != w for a, w in zip(one, want))
        out["mismatches"][f"{name}-{n}"] = bad
    # The Python path (no key) gives the same.
    scores, ties = nemo._native._repeating_top_ngrams(docs, n)
    for i, text, starts, words in ties:
        scores[i] = nemo._tied_score(text, n, starts, words)
    out["mismatches"][f"fallback-{n}"] = sum(a != float(b) for a, b in zip(batch, scores))
# The simulated set order against this process's own sets, on random n-gram lists.
if key is not None:
    rng = random.Random(sys.argv[3])
    bad = 0
    for _ in range(int(sys.argv[4])):
        n = rng.randint(1, 4)
        vocab = [t.random_word(rng) for _ in range(rng.randint(1, 80))]
        words = rng.choices(vocab, k=rng.randrange(0, 3000) + n - 1)
        bad += t.set_order_mismatch(words, n, key)
    out["mismatches"]["set-order"] = bad
print(json.dumps(out))
"""


@pytest.mark.thread_unsafe(reason="runs subprocesses")
def test_ties_follow_the_hash_seed(ref: Any) -> None:
    """Tied documents get NeMo's score whatever the process's hash seed is, on the Rust path
    (the process's key) and on the Python one; the simulated set order is the process's."""
    seeds = ["0", *(str(random.SystemRandom().randrange(1, 2**32)) for _ in range(4))]
    runs = []
    for seed in seeds:
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run(
            [sys.executable, "-P", "-c", _SEED_SCRIPT, str(HERE), "7", seed, "60"],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        out = json.loads(proc.stdout)
        assert set(out["mismatches"].values()) == {0}, (seed, out["mismatches"])
        assert out["key"] == CPYTHON_SIPHASH, seed
        if live_nemo() is not None:
            assert out["impls"] == ["nemo", "reference"]
        runs.append(out["scores"])
    # The corpus really is hash-dependent: some document's score changes with the seed.
    assert any(len({json.dumps(r[n]) for r in runs}) > 1 for n in ("2", "3", "4"))


def test_tied_documents_in_every_container(pa: Any, ref: Any) -> None:
    docs = tie_corpus(3, 40)
    for n in (1, 2, 3):
        want = [repr(ref.RepeatingTopNGramsFilter(n).score_document(d)) for d in docs]
        case = Case(
            "t",
            _filter("RepeatingTopNGramsFilter", n),
            functools.partial(nemo.repeating_top_ngrams, n=n),
        )
        for kind in (
            docs,
            tuple(docs),
            pa.array(docs),
            pa.chunked_array([docs[:7], docs[7:]]),
            pa.array(docs, pa.large_string()),
            pa.array(["x", *docs]).slice(1),
        ):
            assert as_reprs(case, nemo.repeating_top_ngrams(kind, n)) == want


# ---- the ties in Rust: CPython's hashes and set order ---------------------------------

#: Interpreters on which pyroid reads the hash key and resolves ties in Rust.
CPYTHON_SIPHASH = (
    sys.implementation.name == "cpython"
    and sys.hash_info.algorithm == "siphash13"
    and sys.hash_info.cutoff == 0
    and sys.hash_info.hash_bits == 64
)


def _key() -> bytes:
    key: bytes | None = _nemo._hash_key()
    if key is None:
        pytest.skip("pyroid resolves ties in Python on this interpreter")
    return key


def test_hash_key_is_used_where_it_can_be() -> None:
    assert (_nemo._hash_key() is not None) == CPYTHON_SIPHASH


@given(
    strings=st.lists(st.one_of(st.text(CHARS), st.text(st.sampled_from(_WIDTH_CHARS))), max_size=8)
)
def test_str_hashes_are_pythons(strings: list[str]) -> None:
    assert _nemo._native._str_hashes(strings, _key()) == [hash(s) for s in strings]


_vocab_words = st.one_of(
    st.sampled_from(("a", "b", "ab", "é", "中", "\U0001f600", "")), st.text(CHARS, max_size=4)
)


@given(words=st.lists(_vocab_words, max_size=400), n=st.integers(1, 4))
def test_ngram_set_order_is_pythons(words: list[str], n: int) -> None:
    assert set_order_mismatch(words, n, _key()) == 0


_MODULUS = (1 << 61) - 1
_colliding_ints = st.one_of(
    st.integers(-(2**62), 2**62),
    st.sampled_from((0, 1, -1, -2, 8, 16, _MODULUS, _MODULUS + 1, -_MODULUS, 2 * _MODULUS)),
    st.integers(0, 2**20).map(lambda x: x << 40),
    st.integers(0, 2**20).map(lambda x: x + _MODULUS),
)


@given(xs=st.lists(_colliding_ints, max_size=500))
def test_set_order_with_colliding_hashes(xs: list[int]) -> None:
    """Items sharing slot bits or whole hashes (``hash(x + 2**61 - 1) == hash(x)``)."""
    order = _nemo._native._set_order([hash(x) for x in xs], xs)
    assert [xs[i] for i in order] == list(set(xs))


@pytest.mark.thread_unsafe(reason="large sets")
@pytest.mark.parametrize("size", [0, 4, 5, 19, 20, 76, 77, 4915, 4916, 78_642, 78_643, 100_000])
def test_set_order_across_resizes(size: int) -> None:
    """Sizes on both sides of each table growth (5 of 8 slots, ..., and the 2x rule)."""
    key = _key()
    rng = random.Random(size)
    xs = [rng.getrandbits(62) - (1 << 61) for _ in range(size)]
    order = _nemo._native._set_order([hash(x) for x in xs], xs)
    assert [xs[i] for i in order] == list(set(xs))
    words = [f"w{i}" for i in range(size)] + ["w0"] * 3
    assert set_order_mismatch(words, 1, key) == 0


@pytest.mark.thread_unsafe(reason="monkeypatches pyroid.nemo")
def test_python_path_gives_the_same_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the key (it cannot be read, or the check fails) ties are resolved in Python,
    with the same scores."""
    docs = tie_corpus(11, 200)
    want = {n: nemo.repeating_top_ngrams(docs, n) for n in (1, 2, 3, 4)}
    key = _nemo._hash_key()
    if key is not None:
        assert _nemo._self_check(key)
        assert not _nemo._self_check(bytes([key[0] ^ 1]) + key[1:])

    def unreadable() -> bytes:
        raise ValueError("symbol not found")

    monkeypatch.setattr(_nemo, "_secret", unreadable)
    assert _nemo._hash_key.__wrapped__() is None
    monkeypatch.setattr(_nemo, "_hash_key", lambda: None)
    for n, scores in want.items():
        assert np.array_equal(nemo.repeating_top_ngrams(docs, n), scores), n
        assert [nemo.repeating_top_ngrams(d, n) for d in docs[:30]] == scores[:30].tolist()


def test_bad_hash_key_is_refused() -> None:
    with pytest.raises(ValueError, match="16 bytes"):
        _nemo._native._repeating_top_ngrams(["a b a b"], 2, hash_key=b"short")


# ---- containers, nulls, errors, threads ------------------------------------------------


def test_result_types(pa: Any) -> None:
    assert nemo.word_count("a b") == 2
    assert type(nemo.word_count("a b")) is int
    assert nemo.white_space("a b") == 1 / 3
    for kind in (
        ["a b", ""],
        ("a b", ""),
        pa.array(["a b", ""]),
        pa.chunked_array([["a b"], [""]]),
    ):
        counts = nemo.word_count(kind)
        assert counts.dtype == np.int64
        assert counts.tolist() == [2, 0]
        ratios = nemo.white_space(kind)
        assert ratios.dtype == np.float64
        assert ratios.tolist() == [1 / 3, 1.0]
    assert nemo.word_count([]).dtype == np.int64
    assert nemo.repeating_top_ngrams([]).dtype == np.float64
    assert nemo.repeating_top_ngrams(()).shape == (0,)
    assert nemo.lang_id_strings([], []) == []


def test_arrow_types(pa: Any) -> None:
    docs = ["a a b", "ok.", "", "é é"]
    want = nemo.repeated_paragraphs(docs).tolist()
    for arr in (
        pa.array(docs, pa.large_string()),
        pa.array(docs, pa.string_view()),
        pa.array(["drop", *docs]).slice(1),
        pa.chunked_array([docs[:1], [], docs[1:]]),
        _testing.column(docs, large=True, chunks=3, name="text"),
    ):
        assert nemo.repeated_paragraphs(arr).tolist() == want


def test_arrow_nulls_raise(pa: Any) -> None:
    with pytest.raises(ValueError, match="index 2 is null"):
        nemo.word_count(pa.chunked_array([["a", "b"], [None, "c"]]))
    with pytest.raises(ValueError, match="index 1 is null"):
        nemo.repeating_top_ngrams(pa.array(["a b a b", None]))
    with pytest.raises(ValueError, match="index 0 is null"):
        nemo.long_word(_testing.column([None]))
    # The first failing document wins: a ZeroDivisionError before a null, a null before one.
    with pytest.raises(ZeroDivisionError) as info:
        nemo.repeated_lines(pa.array(["", None]))
    assert info.value.__notes__ == ["pyroid: raised for the document at index 0"]
    with pytest.raises(ValueError, match="index 1 is null"):
        nemo.repeated_lines(pa.array(["a", None, ""]))


def test_errors() -> None:
    with pytest.raises(ZeroDivisionError, match=r"^division by zero$") as zd:
        nemo.repeated_lines("")
    assert not getattr(zd.value, "__notes__", None)
    with pytest.raises(ValueError, match=r"^max\(\) iterable argument is empty$"):
        nemo.long_word(" ")
    with pytest.raises(ValueError, match="max"):
        nemo.repeating_top_ngrams("a b", 0)
    with pytest.raises(ZeroDivisionError) as info:
        nemo.mean_word_length(["a", "b", " ", ""])
    assert info.value.__notes__ == ["pyroid: raised for the document at index 2"]
    with pytest.raises(NotImplementedError, match="MeCab"):
        nemo.word_count("a", lang="JA")
    with pytest.raises(
        ValueError, match=r"^Invalid position: middle\. Must be one of: prefix, suffix, any\.$"
    ):
        nemo.substring([], "a", "middle")  # type: ignore[call-overload]
    with pytest.raises(ValueError, match="Invalid position: 3"):
        nemo.substring("a", "a", 3)  # type: ignore[call-overload]
    with pytest.raises(pyroid.InputError, match="index 1"):
        nemo.numbers(["a", 1])  # type: ignore[list-item]
    with pytest.raises(pyroid.InputError):
        nemo.numbers(b"a")  # type: ignore[call-overload]
    with pytest.raises(UnicodeEncodeError):
        nemo.numbers(["a", "\ud800"])
    with pytest.raises(ValueError, match="threads"):
        nemo.numbers(["a"], threads=0)


def test_lang_is_compared_like_nemo(ref: Any) -> None:
    """``get_word_splitter`` lowercases ``lang``: "ZH" is Chinese, other codes split on spaces."""
    text = "我来到北京清华大学 ok"
    for lang in ("zh", "ZH", "Zh", "en", "fr", "", "zh-cn", "\u212aa"):
        want = ref.WordCountFilter(lang=lang).score_document(text)
        assert nemo.word_count(text, lang=lang) == want, lang


def test_threads_give_the_same_results() -> None:
    docs = tie_corpus(5, 500) + [d * 3 for d in PIECES]
    for case in CASES:
        try:
            base = case.pyroid(docs)
        except Exception as exc:
            for setting in (1, 2, 3):
                with pytest.raises(type(exc)):
                    case.pyroid(docs, threads=setting)
            continue
        for t in (None, 1, 2, 3):
            assert np.array_equal(case.pyroid(docs, threads=t), base), (case.name, t)


@pytest.mark.thread_unsafe(reason="10k documents")
@pytest.mark.timeout(240)
def test_10k_documents(ref: Any) -> None:
    rng = random.Random(10)
    docs = [
        "\n".join(
            " ".join(rng.choice(_WORDS) for _ in range(rng.randrange(1, 30))) for _ in range(5)
        )
        for _ in range(10_000)
    ]
    sample = rng.sample(range(len(docs)), 100)
    for case in CASES:
        filt = case.nemo_filter(ref)
        if nemo_outcome(filt, docs[0])[0] == "raise":
            continue
        got = as_reprs(case, case.pyroid(docs))
        assert len(got) == len(docs)
        for i in sample:
            assert got[i] == nemo_outcome(filt, docs[i])[1], (case.name, i)


# ---- the Python facts the kernels rely on ----------------------------------------------


@pytest.mark.thread_unsafe(reason="~1.1M code points")
def test_lower_projection_guard() -> None:
    """Only U+0130 and U+212A lowercase to ASCII; only "…" and "»" to "…" or "»".

    The kernels match NeMo's lowercase needles (ASCII, "…", "»") against a projection of
    ``str.lower()`` that relies on exactly this, on this interpreter.
    """
    found = []
    for cp in range(0x80, 0x110000):
        if 0xD800 <= cp < 0xE000:
            continue
        low = chr(cp).lower()
        if any(c.isascii() or c in "…»" for c in low):
            found.append(cp)
    assert found == [0xBB, 0x130, 0x2026, 0x212A]
    assert chr(0x130).lower() == "i\u0307"
    assert chr(0x212A).lower() == "k"
    assert all(c.lower() == c for c in WHITESPACE)


@pytest.mark.thread_unsafe(reason="~1.1M documents")
def test_split_is_isspace() -> None:
    chars = [chr(cp) for cp in range(0x110000) if not 0xD800 <= cp < 0xE000]
    docs = [f"a{c}b" for c in chars]
    got = nemo.word_count(docs).tolist()
    assert got == [len(d.split()) for d in docs]
    assert "".join(c for c in chars if c.isspace()) == WHITESPACE


@pytest.mark.thread_unsafe(reason="~1.1M documents")
def test_alpha_table_is_regex() -> None:
    """``[[:alpha:]]`` as the pinned ``regex`` releases read it (Unicode 18.0.0)."""
    regex = pytest.importorskip("regex")
    if regex.__version__ not in ("2026.9.29", "2026.11.9"):
        pytest.skip(f"the table records regex 2026.9.29, not {regex.__version__}")
    alpha = regex.compile("[[:alpha:]]")
    chars = [
        chr(cp) for cp in range(0x110000) if not 0xD800 <= cp < 0xE000 and not chr(cp).isspace()
    ]
    got = nemo.words_without_alphabets(chars).tolist()
    assert got == [1.0 if alpha.search(c) else 0.0 for c in chars]


# ---- fastText stages -------------------------------------------------------------------

_FT_WORDS = ("the", "cat", "sat", "hq", "spam", "été", "中文", "__label__x")
_FT_LABELS = ("__label__hq", "__label__lq", "__label__o'k", "xx", "__label__é")


def write_model(path: Path, *, seed: int = 0, dim: int = 4) -> Path:
    """A supervised fastText model (softmax, no subwords, no ``</s>``): out-of-vocabulary
    lines have no tokens, so fastText predicts no label for them."""
    rng = np.random.default_rng(seed)
    out = struct.pack("<ii", 793_712_314, 12)
    # Args: dim ws epoch minCount neg wordNgrams loss(softmax) model(sup) bucket minn maxn
    # lrUpdateRate t.
    out += struct.pack("<12id", dim, 5, 5, 1, 5, 1, 3, 3, 0, 0, 0, 100, 1e-4)
    words = [w.encode() for w in _FT_WORDS]
    labels = [x.encode() for x in _FT_LABELS]
    out += struct.pack("<iiiqq", len(words) + len(labels), len(words), len(labels), 12_345, -1)
    for i, w in enumerate(words):
        out += w + b"\0" + struct.pack("<qb", 1000 - i, 0)
    for i, x in enumerate(labels):
        out += x + b"\0" + struct.pack("<qb", 500 - i, 1)
    for rows in (len(words), len(labels)):
        out += struct.pack("<?", False)
        weights = rng.uniform(-2, 2, (rows, dim)).astype("<f4")
        out += struct.pack("<qq", rows, dim) + weights.tobytes()
    path.write_bytes(out)
    return path


@pytest.fixture(scope="session")
def ft_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_model(tmp_path_factory.mktemp("nemo-fasttext") / "model.bin")


@pytest.fixture(scope="session")
def ft_model(ft_path: Path) -> ft.Model:
    return ft.load_model(ft_path)


_ft_piece = st.one_of(
    st.sampled_from(_FT_WORDS),
    st.sampled_from(("\n", " ", "\t", "\r", "__label__", "__lab", "el__", "zzz", "\u3000")),
    st.text(CHARS, max_size=2),
)
FT_TEXT = st.lists(_ft_piece, max_size=12).map("".join)


def lang_id_outcome(filt: Any, text: str) -> Outcome:
    return nemo_outcome(filt, text)


def pyroid_lang_id(model: ft.Model, text: str) -> Outcome:
    try:
        score, lang = nemo.fasttext_lang_id(model, text)
    except Exception as exc:
        return ("raise", type(exc).__name__)
    assert type(score) is float
    assert type(lang) is str
    return ("ok", repr(nemo.lang_id_strings([score], [lang])[0]))


def pyroid_quality(model: ft.Model, text: str, label: str) -> Outcome:
    try:
        score = nemo.fasttext_quality(model, text, label)
    except Exception as exc:
        return ("raise", type(exc).__name__)
    assert type(score) is float
    return ("ok", repr(score))


@given(text=FT_TEXT)
def test_fasttext_reference_parity(ref: Any, ft_model: ft.Model, text: str) -> None:
    """NeMo's code driving pyroid.fasttext (a drop-in for fastText's predict) gives the same."""
    assert pyroid_lang_id(ft_model, text) == lang_id_outcome(ref.FastTextLangId(ft_model), text)
    for label in ("__label__hq", "__label__lq", "xx"):
        filt = ref.FastTextQualityFilter(ft_model, label)
        assert pyroid_quality(ft_model, text, label) == nemo_outcome(filt, text)


@given(docs=st.lists(FT_TEXT, max_size=6))
def test_fasttext_batches(ref: Any, ft_model: ft.Model, docs: list[str]) -> None:
    # The two filters predict on different lines (quality drops "__label__", lang id strips),
    # so they can fail on different documents.
    want_id = [nemo_outcome(ref.FastTextLangId(ft_model), d) for d in docs]
    want_q = [nemo_outcome(ref.FastTextQualityFilter(ft_model), d) for d in docs]
    fail_id = [i for i, w in enumerate(want_id) if w[0] == "raise"]
    fail_q = [i for i, w in enumerate(want_q) if w[0] == "raise"]
    for kind in (docs, tuple(docs), _testing.column(docs, chunks=2)):
        if fail_id:
            with pytest.raises(IndexError) as info:
                nemo.fasttext_lang_id(ft_model, kind)
            assert info.value.__notes__ == [
                f"pyroid: raised for the document at index {fail_id[0]}"
            ]
        else:
            scores, langs = nemo.fasttext_lang_id(ft_model, kind, threads=2)
            assert scores.dtype == np.float64
            assert [repr(s) for s in nemo.lang_id_strings(scores, langs)] == [w[1] for w in want_id]
        if fail_q:
            with pytest.raises(IndexError) as info:
                nemo.fasttext_quality(ft_model, kind)
            assert info.value.__notes__ == [f"pyroid: raised for the document at index {fail_q[0]}"]
        else:
            q = nemo.fasttext_quality(ft_model, kind)
            assert q.dtype == np.float64
            assert [repr(float(x)) for x in q] == [w[1] for w in want_q]


def test_fasttext_errors(ft_model: ft.Model, pa: Any) -> None:
    with pytest.raises(IndexError, match="out of bounds"):
        nemo.fasttext_lang_id(ft_model, "zzz unknown")
    with pytest.raises(IndexError):
        nemo.fasttext_quality(ft_model, "")
    with pytest.raises(ValueError, match="index 1 is null"):
        nemo.fasttext_quality(ft_model, pa.array(["the", None]))
    with pytest.raises(TypeError):
        nemo.fasttext_lang_id("not a model", "the")  # type: ignore[call-overload]
    # NeMo replaces newlines before predicting, so fastText never sees one.
    assert nemo.fasttext_lang_id(ft_model, "the\ncat\n")[1] in {"hq", "lq", "o'k", "xx", "é"}


@pytest.mark.parametrize("which", ["fasttext", "lid.176.ftz"])
def test_fasttext_live(which: str, ft_path: Path, ft_model: ft.Model, ref: Any) -> None:
    """NeMo's own fastText filters (with fastText itself) where they are installed, and the
    lid.176.ftz model when $PYROID_DATA has it. Probabilities agree to 1e-6 (pyroid.fasttext's
    own guarantee against fastText); labels exactly."""
    texts = [
        "the cat sat",
        "  hq spam\nthe  ",
        "été 中文",
        "This is an English sentence.",
        "Ceci est une phrase en français.",
        "这是中文句子。",
        "__label__hq the",
    ]
    if which == "lid.176.ftz":
        path = DATA / "models" / "lid.176.ftz"
        if not path.is_file():
            pytest.skip(f"no {path}")
        model = ft.load_model(path)
    else:
        path, model = ft_path, ft_model
        texts = texts[:3] + texts[-1:]
    impls: list[tuple[Any, Any]] = [(ref, model)]
    live = live_nemo()
    oracle: Any = None
    with contextlib.suppress(ImportError):
        oracle = importlib.import_module("fasttext")
    if oracle is not None:
        ft_oracle = oracle.load_model(str(path))
        impls.append((ref, ft_oracle))
        if live is not None and hasattr(live, "FastTextLangId"):
            impls.append((live, ft_oracle))
    scores, langs = nemo.fasttext_lang_id(model, texts)
    quality = nemo.fasttext_quality(model, texts, "__label__en")
    for module, m in impls:
        lang_id = module.FastTextLangId.__new__(module.FastTextLangId)
        lang_id._fasttext_langid_model = m
        qual = module.FastTextQualityFilter.__new__(module.FastTextQualityFilter)
        qual._fasttext_quality_filter_model = m
        qual._label = "__label__en"
        for i, t in enumerate(texts):
            want_score, want_lang = ast.literal_eval(lang_id.score_document(t))
            assert langs[i] == want_lang
            assert math.isclose(scores[i], want_score, rel_tol=0, abs_tol=1e-6)
            assert math.isclose(quality[i], qual.score_document(t), rel_tol=0, abs_tol=1e-6)
            if m is model:  # the same predictions: exactly NeMo's numbers
                assert repr(float(scores[i])) == repr(want_score)
                assert repr(float(quality[i])) == repr(qual.score_document(t))


_lang = st.one_of(
    st.sampled_from(("en", "o'k", 'a"b', "a\\b", "é", "\t", "x\x7f")),
    st.text(st.one_of(CHARS, st.characters(categories=["Cs"]))),
)


@given(
    scores=st.lists(st.floats(allow_nan=True, allow_infinity=True), max_size=5),
    data=st.data(),
)
def test_lang_id_strings(scores: list[float], data: st.DataObject) -> None:
    langs = [data.draw(_lang) for _ in scores]
    want = [str([s, lang]) for s, lang in zip(scores, langs, strict=True)]
    assert nemo.lang_id_strings(scores, langs) == want
    assert nemo.lang_id_strings(np.array(scores, dtype=np.float64), tuple(langs)) == want


def test_lang_id_strings_errors() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        nemo.lang_id_strings([0.5], [])
    with pytest.raises(TypeError):
        nemo.lang_id_strings([0.5], [1])  # type: ignore[list-item]
    assert nemo.lang_id_strings(np.array([0.25], dtype=np.float32), ["x"]) == ["[0.25, 'x']"]


# ---- free-threading --------------------------------------------------------------------


@pytest.mark.skipif(not FREE_THREADED, reason="free-threaded interpreters only")
def test_gil_stays_disabled() -> None:
    nemo.word_count(["a b"] * 200, threads=2)
    nemo.repeating_top_ngrams(tie_corpus(2, 50))
    assert not sys._is_gil_enabled()  # type: ignore[attr-defined]
