# Derived from NLTK 3.10.4 (Apache-2.0), © 2001-2026 NLTK Project; modified.
# Confusable letters and quotes are the test data here.
# ruff: noqa: RUF001
"""pyroid.nltk: pyroid's conventions, Punkt's parameters and models, and the delegated parts.

- **Containers**: str in, a list out; a list, tuple or Arrow string column (``string``,
  ``large_string``, ``string_view``, sliced, several chunks, or ``_testing.column`` without
  pyarrow) in, one list per document out (``None`` for an Arrow null), for ``sent_tokenize``,
  ``word_tokenize``, the Punkt methods and both Treebank tokenizers.
- **Threads**: identical results for ``threads=None/1/2/4``; a 10k-document batch under a timeout.
- **Errors**: ``InputError`` for non-str items and wrong input types, ``ValueError`` /
  ``InputError`` for bad ``threads``, ``UnicodeEncodeError`` with a note for lone surrogates.
- **Punkt parameters**: in-place changes and reassignments of ``PunktParameters`` (pyroid's
  and NLTK's own), parameters from NLTK's ``PunktTrainer``, custom ``sent_end_chars`` (native)
  and other ``PunktLanguageVars`` overrides, custom ``token_cls`` (NLTK, with
  ``pyroid.FallbackWarning``), ``debug_decisions``.
- **Models**: ``LookupError`` for a missing model, with and without NLTK importable; loading
  from ``punkt_tab.zip`` and from an unpacked directory without NLTK; language-name validation
  (NLTK's messages); ``punkt_model_available``.
- **Treebank**: NLTK's pattern tables; changed tables run in NLTK (``FallbackWarning``).
- rouge-score's ``split_summaries`` still matches rouge-score.

The Hypothesis comparisons with NLTK itself are in test_nltk_parity.py.
"""

from __future__ import annotations

import copy
import functools
import importlib
import io
import pickle  # round trips of objects made in the tests (never data from outside)
import re
import sys
import threading
import types
import unicodedata
import warnings
import zipfile
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import pytest

import pyroid
import pyroid.nltk as pn
from pyroid._pyroid import _testing
from pyroid.nltk import _delegate

NLTK_VERSION = "3.10.4"


def _optional(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


nltk: Any = _optional("nltk")
HAVE_NLTK = nltk is not None and nltk.__version__ == NLTK_VERSION
needs_nltk = pytest.mark.skipif(not HAVE_NLTK, reason=f"needs nltk {NLTK_VERSION}")
REF: dict[str, Any] = {}
if HAVE_NLTK:  # NLTK's own objects, taken before test_nltk_patch.py can patch them
    for _path in (
        "nltk.tokenize:sent_tokenize",
        "nltk.tokenize:word_tokenize",
        "nltk.tokenize:punkt_model_available",
        "nltk.tokenize.punkt:PunktTokenizer",
        "nltk.tokenize.punkt:PunktSentenceTokenizer",
        "nltk.tokenize.punkt:PunktParameters",
        "nltk.tokenize.punkt:PunktLanguageVars",
        "nltk.tokenize.punkt:PunktTrainer",
        "nltk.tokenize.punkt:PunktToken",
        "nltk.tokenize.punkt:load_punkt_params",
        "nltk.tokenize.punkt:_punkt_language",
        "nltk.tokenize.treebank:TreebankWordTokenizer",
        "nltk.tokenize.treebank:TreebankWordDetokenizer",
        "nltk.tokenize.destructive:NLTKWordTokenizer",
        "nltk.tokenize.destructive:MacIntyreContractions",
    ):
        _mod, _name = _path.split(":")
        REF[_name] = getattr(importlib.import_module(_mod), _name)

DOCS = [
    "Mr. Smith went to Washington. He liked it!",
    "",
    "  \n ",
    "They'll pay $3.88 (roughly). \"Yes,\" he said. Didn't they?",
    "Ich muss usw. fur einen Kuchen einzukaufen. Ich muss.",
    "One.\nTwo.\r\nThree\x85four. «Cinq.» “Six.”",
    "no sentence end",
    "İSTANBUL'S ſtate... J. S. Bach. 3. Mai.",
]


def outcome(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, ...]:
    """``("ok", result)`` (generators listed) or ``("raise", type name, message)``."""
    try:
        r = fn(*args, **kwargs)
        if isinstance(r, types.GeneratorType):
            r = list(r)
        return ("ok", r)
    except Exception as e:
        return ("raise", type(e).__name__, str(e))


def native(tok: Any) -> Any:
    """The native Punkt a tokenizer runs with, or None (then NLTK runs)."""
    return tok._impl()


def listed(x: Any) -> Any:
    return list(x) if isinstance(x, types.GeneratorType) else x


@pytest.fixture
def fresh_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forget which FallbackWarnings were shown (they are shown once per process)."""
    monkeypatch.setattr(_delegate, "_WARNED", set())


# Every function with batch input: (needs a model, call on a str or a batch).
@functools.cache
def english() -> pn.PunktTokenizer:
    """One English PunktTokenizer for the container tests (loading a model takes a while)."""
    return pn.PunktTokenizer()


FUNCS: dict[str, tuple[bool, Callable[..., Any]]] = {
    "sent_tokenize": (True, lambda x, **kw: pn.sent_tokenize(x, **kw)),
    "sent_tokenize_german": (True, lambda x, **kw: pn.sent_tokenize(x, "german", **kw)),
    "word_tokenize": (True, lambda x, **kw: pn.word_tokenize(x, **kw)),
    "word_tokenize_preserve": (True, lambda x, **kw: pn.word_tokenize(x, preserve_line=True, **kw)),
    "punkt_tokenize": (True, lambda x, **kw: english().tokenize(x, **kw)),
    "punkt_noalign": (True, lambda x, **kw: english().tokenize(x, False, **kw)),
    "sentences_from_text": (True, lambda x, **kw: english().sentences_from_text(x, **kw)),
    "span_tokenize": (True, lambda x, **kw: listed(english().span_tokenize(x, **kw))),
    "span_noalign": (
        True,
        lambda x, **kw: listed(english().span_tokenize(x, realign_boundaries=False, **kw)),
    ),
    "empty_params": (False, lambda x, **kw: pn.PunktSentenceTokenizer().tokenize(x, **kw)),
    "treebank": (False, lambda x, **kw: pn.TreebankWordTokenizer().tokenize(x, **kw)),
    "treebank_ptb": (False, lambda x, **kw: pn.TreebankWordTokenizer().tokenize(x, True, **kw)),
    "nltkword": (False, lambda x, **kw: pn.NLTKWordTokenizer().tokenize(x, **kw)),
    "nltkword_ptb": (False, lambda x, **kw: pn.NLTKWordTokenizer().tokenize(x, True, **kw)),
    "wordpunct": (False, lambda x, **kw: pn.WordPunctTokenizer().tokenize(x, **kw)),
    "wordpunct_tokenize": (False, lambda x, **kw: pn.wordpunct_tokenize(x, **kw)),
    "whitespace": (False, lambda x, **kw: pn.WhitespaceTokenizer().tokenize(x, **kw)),
    "blankline_tokenize": (False, lambda x, **kw: pn.blankline_tokenize(x, **kw)),
    "regexp_money": (
        False,
        lambda x, **kw: pn.RegexpTokenizer(r"\w+|\$[\d\.]+|\S+").tokenize(x, **kw),
    ),
    "regexp_keep_empty": (
        False,
        lambda x, **kw: pn.RegexpTokenizer(r"\s+", gaps=True, discard_empty=False).tokenize(
            x, **kw
        ),
    ),
}


@pytest.fixture(params=list(FUNCS))
def func(request: pytest.FixtureRequest) -> Callable[..., Any]:
    """One of FUNCS; the punkt_tab models are set up for those that need them."""
    needs_model, f = FUNCS[request.param]
    if needs_model:
        request.getfixturevalue("punkt_tab")
    return f


def per_doc(f: Callable[..., Any], docs: list[str | None]) -> list[Any]:
    return [None if d is None else f(d) for d in docs]


# ---- containers ----------------------------------------------------------------------------


def test_str_gives_a_list(func: Callable[..., Any]) -> None:
    out = func(DOCS[0])
    assert type(out) is list
    assert all(type(x) in (str, tuple) for x in out)


def test_empty(func: Callable[..., Any]) -> None:
    # NLTK's realign_boundaries=False gives one empty sentence (span (0, 0)) for "".
    one = func("")
    assert one in ([], [""], [(0, 0)])
    assert func([]) == []
    assert func(()) == []
    assert func([""]) == [one]
    assert func(_testing.column([])) == []
    assert func(_testing.column([None, ""])) == [None, one]


@pytest.mark.parametrize("kind", ["list", "tuple"])
def test_list_and_tuple(func: Callable[..., Any], kind: str) -> None:
    x = list(DOCS) if kind == "list" else tuple(DOCS)
    out = func(x)
    assert type(out) is list
    assert all(type(r) is list for r in out)
    assert out == per_doc(func, list(DOCS))


@pytest.mark.parametrize("chunks", [1, 3])
@pytest.mark.parametrize("large", [False, True], ids=["string", "large_string"])
def test_column_without_pyarrow(func: Callable[..., Any], large: bool, chunks: int) -> None:
    values: list[str | None] = [*DOCS[:3], None, *DOCS[3:], None]
    col = _testing.column(values, large=large, chunks=chunks, name="text")
    out = func(col)
    assert type(out) is list
    assert out == per_doc(func, values)


@pytest.mark.parametrize("type_name", ["string", "large_string", "string_view"])
def test_pyarrow(pa: Any, func: Callable[..., Any], type_name: str) -> None:
    values: list[str | None] = [None, *DOCS, None]
    ty = getattr(pa, type_name)()
    arr = pa.array(values, ty)
    assert func(arr) == per_doc(func, values)
    assert func(arr[2:7]) == per_doc(func, values[2:7])  # offset > 0
    chunked = pa.chunked_array([pa.array(values[:4], ty), pa.array([], ty), arr[4:]])
    assert func(chunked) == per_doc(func, values)
    assert func(pa.chunked_array([], ty)) == []


def test_pyarrow_table(pa: Any, punkt_tab: Any) -> None:
    table = pa.table({"text": DOCS, "n": list(range(len(DOCS)))})
    assert pn.sent_tokenize(table.column("text")) == [pn.sent_tokenize(d) for d in DOCS]
    with pytest.raises(pyroid.InputError) as info:
        pn.sent_tokenize(table)
    assert "while processing 'text'" in info.value.__notes__
    with pytest.raises(pyroid.InputError) as info:
        pn.TreebankWordTokenizer().tokenize(table)
    assert "while processing 'text'" in info.value.__notes__


def test_batch_spans_are_lists_of_tuples(punkt_tab: Any) -> None:
    out = pn.PunktTokenizer().span_tokenize(DOCS)
    assert type(out) is list
    assert all(type(r) is list and all(type(s) is tuple for s in r) for r in out)


def test_tokenize_sents(punkt_tab: Any) -> None:
    tok = pn.PunktTokenizer()
    assert tok.tokenize_sents(DOCS) == [tok.tokenize(d) for d in DOCS]
    assert tok.tokenize_sents(tuple(DOCS)) == [tok.tokenize(d) for d in DOCS]
    assert tok.tokenize_sents(DOCS, threads=2) == [tok.tokenize(d) for d in DOCS]
    spans = tok.span_tokenize_sents(DOCS)
    assert isinstance(spans, types.GeneratorType)
    assert list(spans) == [list(tok.span_tokenize(d)) for d in DOCS]
    for wt in (pn.TreebankWordTokenizer(), pn.NLTKWordTokenizer()):
        assert wt.tokenize_sents(DOCS) == [wt.tokenize(d) for d in DOCS]
        assert wt.tokenize_sents(iter(DOCS)) == [wt.tokenize(d) for d in DOCS]  # type: ignore[arg-type]
        assert list(wt.span_tokenize_sents(DOCS)) == [list(wt.span_tokenize(d)) for d in DOCS]


# ---- generators ------------------------------------------------------------------------------


def test_generators(punkt_tab: Any) -> None:
    tok = pn.PunktTokenizer()
    text = DOCS[0]
    for g in (
        tok.span_tokenize(text),
        tok.span_tokenize(text, False),
        tok.span_tokenize_sents([text]),
        tok.sentences_from_text_legacy(text),
        tok.sentences_from_tokens(text.split()),
        tok.debug_decisions(text),
        pn.TreebankWordTokenizer().span_tokenize(text),
        pn.NLTKWordTokenizer().span_tokenize(text),
        pn.NLTKWordTokenizer().span_tokenize_sents([text]),
    ):
        assert isinstance(g, types.GeneratorType)
    assert type(tok.tokenize(text)) is list
    assert type(tok.sentences_from_text(text)) is list
    assert type(pn.align_tokens(["Mr."], text)) is list


@needs_nltk
def test_generator_types_match_nltk(punkt_tab: Any) -> None:
    ours, ref = pn.PunktTokenizer(), REF["PunktTokenizer"]()
    text = DOCS[3]
    for name in ("span_tokenize", "sentences_from_text_legacy", "debug_decisions"):
        assert type(getattr(ours, name)(text)) is type(getattr(ref, name)(text)), name
    for name in ("tokenize", "sentences_from_text", "text_contains_sentbreak"):
        assert type(getattr(ours, name)(text)) is type(getattr(ref, name)(text)), name


# ---- threads ---------------------------------------------------------------------------------

BIG = [DOCS[i % len(DOCS)] + f" Doc {i}. Ok." for i in range(300)]  # > 64 docs: the pool is used


def test_threads_equivalence(func: Callable[..., Any]) -> None:
    want = per_doc(func, list(BIG))
    col = _testing.column(BIG, chunks=4)
    for threads in (None, 1, 2, 4):
        assert func(BIG, threads=threads) == want, threads
        assert func(col, threads=threads) == want, threads


@pytest.mark.thread_unsafe(reason="10k documents")
@pytest.mark.timeout(300)
def test_ten_thousand_documents(punkt_tab: Any) -> None:
    docs = [f"{DOCS[i % len(DOCS)]} Dr. No {i}. The end." for i in range(10_000)]
    out = pn.sent_tokenize(docs)
    assert len(out) == 10_000
    assert out[1234] == pn.sent_tokenize(docs[1234])
    assert out[9999] == pn.sent_tokenize(docs[9999])
    words = pn.word_tokenize(docs, threads=4)
    assert words[5000] == pn.word_tokenize(docs[5000])
    spans = pn.PunktTokenizer().span_tokenize(docs, threads=2)
    assert spans[7] == list(pn.PunktTokenizer().span_tokenize(docs[7]))
    tb = pn.NLTKWordTokenizer().tokenize(_testing.column(docs, chunks=7))
    assert tb[42] == pn.NLTKWordTokenizer().tokenize(docs[42])


@pytest.mark.parametrize("bad", [0, -1])
def test_bad_threads(func: Callable[..., Any], bad: int) -> None:
    with pytest.raises(ValueError, match="threads must be None or a positive int"):
        func(BIG, threads=bad)


@pytest.mark.parametrize("wrong", ["2", 2.0])
def test_wrong_threads_type(func: Callable[..., Any], wrong: Any) -> None:
    with pytest.raises(pyroid.InputError):
        func(BIG, threads=wrong)


# ---- errors ----------------------------------------------------------------------------------


@pytest.mark.parametrize("item", [1, None, b"bytes", 2.5, ["nested"]])
def test_non_str_item(func: Callable[..., Any], item: Any) -> None:
    with pytest.raises(pyroid.InputError, match="at index 1"):
        func(["Fine. Text.", item])


@pytest.mark.parametrize("x", [5, None, b"bytes", {"a": 1}, {"a"}, 2.5])
def test_wrong_input_type(func: Callable[..., Any], x: Any) -> None:
    with pytest.raises(pyroid.InputError):
        func(x)
    assert issubclass(pyroid.InputError, TypeError)  # what NLTK raises for these


def test_arrow_non_string_column(pa: Any, func: Callable[..., Any]) -> None:
    with pytest.raises(pyroid.InputError, match="Arrow string column"):
        func(pa.array([1, 2]))
    with pytest.raises(pyroid.InputError):
        func(pa.array(["a", "b"]).dictionary_encode())


def test_lone_surrogates(func: Callable[..., Any]) -> None:
    with pytest.raises(UnicodeEncodeError) as info:
        func("A b.\ud800 C d.")
    assert any("lone surrogate" in n for n in getattr(info.value, "__notes__", []))
    with pytest.raises(UnicodeEncodeError) as info:
        func(["Fine.", "Not \udfff fine."])
    assert any("index 1" in n for n in getattr(info.value, "__notes__", []))


def test_lone_surrogates_elsewhere(punkt_tab: Any) -> None:
    tok = pn.PunktTokenizer()
    for call in (
        lambda: tok.text_contains_sentbreak("a.\ud800 B"),
        lambda: list(tok.debug_decisions("a. \ud800B")),
        lambda: list(tok.sentences_from_text_legacy("a. \ud800B")),
        lambda: list(tok.sentences_from_tokens(["a.", "\ud800"])),
        lambda: list(pn.TreebankWordTokenizer().span_tokenize("a \ud800")),
        lambda: pn.TreebankWordDetokenizer().detokenize(["a", "\ud800"]),
        lambda: pn.align_tokens(["a"], "a\ud800"),
        lambda: pn.align_tokens(["\ud800"], "a"),
    ):
        with pytest.raises(UnicodeEncodeError):
            call()


@needs_nltk
def test_detokenize_inputs_as_nltk() -> None:
    """Any iterable of str, as NLTK's ``" ".join(tokens)``; a non-str token is a TypeError."""
    ours, ref = pn.TreebankWordDetokenizer(), REF["TreebankWordDetokenizer"]()
    toks = ["I", "do", "n't", "know", "''", "."]
    makers: list[Callable[[list[str]], Any]] = [list, tuple, iter, lambda t: (x for x in t)]
    for make in makers:
        assert ours.detokenize(make(toks)) == ref.detokenize(make(toks)), make
        assert ours.detokenize(make([])) == ref.detokenize(make([]))
    assert ours.detokenize("abc") == ref.detokenize("abc") == "a b c"  # characters, as NLTK
    with pytest.raises(TypeError):
        ours.detokenize(["a", 1])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        ours.detokenize(5)  # type: ignore[arg-type]


@needs_nltk
def test_align_tokens_inputs_as_nltk() -> None:
    from_nltk = importlib.import_module("nltk.tokenize.util").align_tokens
    s = "The plane, bound for St Petersburg."
    toks = ["The", "plane", ",", "bound"]
    assert pn.align_tokens(toks, s) == from_nltk(toks, s)
    assert pn.align_tokens(tuple(toks), s) == from_nltk(tuple(toks), s)
    assert pn.align_tokens(iter(toks), s) == from_nltk(iter(toks), s)  # type: ignore[arg-type]
    assert outcome(pn.align_tokens, ["x"], s) == outcome(from_nltk, ["x"], s)


# ---- Punkt parameters --------------------------------------------------------------------------

TEXT = "I saw Dr. Who. He was there. Mr. X. ate a cake, i.e. the cake. U.S. troops left. Yes."


def _nltk_twin(params: Any) -> Any:
    """NLTK's PunktSentenceTokenizer with a copy of `params` (as plain sets and a defaultdict)."""
    p = REF["PunktParameters"]()
    p.abbrev_types = set(params.abbrev_types)
    p.collocations = set(params.collocations)
    p.sent_starters = set(params.sent_starters)
    p.ortho_context = defaultdict(int, params.ortho_context)
    return REF["PunktSentenceTokenizer"](p)


def _check(tok: pn.PunktSentenceTokenizer, text: str = TEXT) -> None:
    want = _nltk_twin(tok._params)
    for realign in (True, False):
        assert tok.tokenize(text, realign) == want.tokenize(text, realign)
        assert list(tok.span_tokenize(text, realign)) == list(want.span_tokenize(text, realign))
    assert list(tok.debug_decisions(text)) == list(want.debug_decisions(text))
    assert tok.text_contains_sentbreak(text) == want.text_contains_sentbreak(text)


# In-place changes: (container, change). Each changes what the tokenizer decides on TEXT.
MUTATIONS: dict[str, Callable[[Any], Any]] = {
    "abbrev.add": lambda p: p.abbrev_types.add("who"),
    "abbrev.update": lambda p: p.abbrev_types.update({"who", "there"}),
    "abbrev.|=": lambda p: p.abbrev_types.__ior__({"who"}),
    "abbrev.discard": lambda p: p.abbrev_types.discard("dr"),
    "abbrev.remove": lambda p: p.abbrev_types.remove("dr"),
    "abbrev.-=": lambda p: p.abbrev_types.__isub__({"dr", "mr"}),
    "abbrev.&=": lambda p: p.abbrev_types.__iand__({"i.e"}),
    "abbrev.^=": lambda p: p.abbrev_types.__ixor__({"dr", "who"}),
    "abbrev.difference_update": lambda p: p.abbrev_types.difference_update({"dr"}),
    "abbrev.intersection_update": lambda p: p.abbrev_types.intersection_update({"mr"}),
    "abbrev.symmetric_difference_update": lambda p: p.abbrev_types.symmetric_difference_update(
        {"dr", "who"}
    ),
    "abbrev.clear": lambda p: p.abbrev_types.clear(),
    "abbrev.pop-all": lambda p: [p.abbrev_types.pop() for _ in range(len(p.abbrev_types))],
    "abbrev.swap (same size)": lambda p: (p.abbrev_types.discard("dr"), p.abbrev_types.add("who")),
    "abbrev.clear_abbrevs": lambda p: p.clear_abbrevs(),
    "sent_starters.add": lambda p: p.sent_starters.add("ate"),
    "sent_starters.clear_sent_starters": lambda p: p.clear_sent_starters(),
    "collocations.add": lambda p: p.collocations.add(("who", "he")),
    "collocations.clear_collocations": lambda p: p.clear_collocations(),
    "ortho.setitem (new)": lambda p: p.ortho_context.__setitem__("he", 2),
    "ortho.setitem (existing)": lambda p: p.ortho_context.__setitem__("yes", 2),
    "ortho.add_ortho_context": lambda p: p.add_ortho_context("troops", 32),
    "ortho.|= (existing)": lambda p: p.ortho_context.__setitem__("yes", p.ortho_context["yes"] | 4),
    "ortho.delitem": lambda p: p.ortho_context.__delitem__("he"),
    "ortho.pop": lambda p: p.ortho_context.pop("he"),
    "ortho.popitem": lambda p: p.ortho_context.popitem(),
    "ortho.update": lambda p: p.ortho_context.update({"he": 0, "yes": 64}),
    "ortho.setdefault": lambda p: p.ortho_context.setdefault("troops", 32),
    "ortho.clear": lambda p: p.ortho_context.clear(),
    "ortho.clear_ortho_context": lambda p: p.clear_ortho_context(),
    "reassign abbrev": lambda p: setattr(p, "abbrev_types", {"who", "dr"}),
    "reassign ortho": lambda p: setattr(p, "ortho_context", defaultdict(int, {"he": 2})),
}


def _params(cls: Any) -> Any:
    p = cls()
    p.abbrev_types.update({"dr", "mr", "i.e", "u.s"})
    p.sent_starters.update({"he", "yes"})
    p.collocations.add(("x", "ate"))
    for k, v in {"he": 2 | 32, "yes": 2, "troops": 64, "cake": 32}.items():
        p.ortho_context[k] = v
    return p


@needs_nltk
@pytest.mark.parametrize("owner", ["pyroid", "nltk"])
@pytest.mark.parametrize("change", list(MUTATIONS))
def test_parameter_changes_are_seen(change: str, owner: str) -> None:
    original = _params(pn.PunktParameters if owner == "pyroid" else REF["PunktParameters"])
    tok = pn.PunktSentenceTokenizer(original)
    # pyroid's parameters are used as they are; NLTK's are adopted as a pyroid copy (see
    # test_nltk_parameters_are_adopted), which is what one edits through tok._params.
    params = cast(Any, tok)._params
    assert (params is original) == (owner == "pyroid")
    _check(tok)  # builds (and caches) the native tables
    MUTATIONS[change](params)
    _check(tok)
    # Changes on a reassigned container are seen too.
    params.abbrev_types = set(params.abbrev_types)
    _check(tok)
    params.abbrev_types.add("saw")
    _check(tok)
    params.abbrev_types.discard("saw")
    params.abbrev_types.add("there")  # same size, other contents
    _check(tok)


@needs_nltk
def test_nltk_parameters_are_adopted() -> None:
    """Documented difference: a tokenizer copies NLTK's PunktParameters (and a plain set given
    to pyroid's), so later edits through the original object are not seen; NLTK sees them."""
    nltk_params = _params(REF["PunktParameters"])
    ours, ref = pn.PunktSentenceTokenizer(nltk_params), REF["PunktSentenceTokenizer"](nltk_params)
    adopted = cast(Any, ours)._params
    assert isinstance(adopted, pn.PunktParameters)
    assert adopted is not nltk_params
    before = ours.tokenize(TEXT)
    assert before == ref.tokenize(TEXT)
    nltk_params.collocations.add(("who", "he"))
    assert ours.tokenize(TEXT) == before  # not seen
    assert ref.tokenize(TEXT) != before  # NLTK sees it
    p, plain = pn.PunktParameters(), {"who"}
    p.abbrev_types = plain
    assert p.abbrev_types == plain
    assert p.abbrev_types is not plain  # NLTK keeps the object itself


@needs_nltk
def test_parameter_object_reassigned() -> None:
    tok = pn.PunktSentenceTokenizer(_params(pn.PunktParameters))
    _check(tok)
    tok._params = _params(REF["PunktParameters"])
    _check(tok)
    tok._params.abbrev_types.add("who")
    _check(tok)
    tok._params = pn.PunktParameters()
    _check(tok)


@needs_nltk
def test_parameters_shared_by_tokenizers() -> None:
    params = _params(pn.PunktParameters)
    a, b = pn.PunktSentenceTokenizer(params), pn.PunktSentenceTokenizer(params)
    _check(a)
    _check(b)
    params.abbrev_types.add("who")
    _check(a)
    _check(b)


@needs_nltk
def test_parameter_values_rust_cannot_hold() -> None:
    """Non-str entries (allowed by NLTK's sets) give NLTK's results."""
    params = _params(pn.PunktParameters)
    params.abbrev_types.add(5)
    params.sent_starters.add(("a", "b"))
    params.collocations.add("ab")
    tok = pn.PunktSentenceTokenizer(params)
    _check(tok)


def test_punkt_parameters_api() -> None:
    p = pn.PunktParameters()
    assert p.abbrev_types == set()
    assert isinstance(p.abbrev_types, set)
    assert isinstance(p.ortho_context, defaultdict)
    assert p.ortho_context.default_factory is int
    assert p.ortho_context["unseen"] == 0
    p.add_ortho_context("w", 2)
    p.add_ortho_context("w", 4)
    assert p.ortho_context["w"] == 6
    assert sorted(cast(Any, p)._debug_ortho_context("w")) == ["BEG-UC", "MID-UC"]
    p.abbrev_types.add("x")
    clone = pickle.loads(pickle.dumps(p))
    assert clone.abbrev_types == {"x"}
    assert clone.ortho_context["w"] == 6
    assert copy.deepcopy(p).abbrev_types == {"x"}
    for clear, attr in (
        (p.clear_abbrevs, "abbrev_types"),
        (p.clear_collocations, "collocations"),
        (p.clear_sent_starters, "sent_starters"),
        (p.clear_ortho_context, "ortho_context"),
    ):
        old = getattr(p, attr)
        clear()
        assert getattr(p, attr) is not old
        assert len(getattr(p, attr)) == 0


@needs_nltk
def test_punkt_parameters_match_nltk_structure() -> None:
    ours, ref = pn.PunktParameters(), REF["PunktParameters"]()
    assert sorted(vars(ours)) == sorted(vars(ref))
    for name in vars(ref):
        assert type(getattr(ref, name)) in type(getattr(ours, name)).__mro__, name


@needs_nltk
@pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning")
def test_trainer_parameters() -> None:
    train = " ".join(DOCS) * 3 + " The Corp. ltd. is big. The Corp. ltd. is small. Inc. is a word."
    params = pn.PunktTrainer(train).get_params()
    assert isinstance(params, REF["PunktParameters"])
    ours, ref = pn.PunktSentenceTokenizer(params), REF["PunktSentenceTokenizer"](params)
    assert native(ours) is not None  # NLTK's parameters run natively
    for d in DOCS:
        assert ours.tokenize(d) == ref.tokenize(d)
        assert list(ours.debug_decisions(d)) == list(ref.debug_decisions(d))
    trained, ref_trained = pn.PunktSentenceTokenizer(train), REF["PunktSentenceTokenizer"](train)
    assert native(trained) is not None
    for d in DOCS:
        assert trained.tokenize(d) == ref_trained.tokenize(d)
    assert pn.PunktSentenceTokenizer(params).train(params) is params  # not a str: returned


@needs_nltk
@pytest.mark.thread_unsafe(reason="pytest.warns, one-time warning state")
def test_training_warns_once(fresh_warnings: None) -> None:
    _delegate._WARNED.clear()  # each --iterations run (pytest-run-parallel) warns anew
    with pytest.warns(pyroid.FallbackWarning, match="PunktTrainer") as record:
        pn.PunktTrainer("A b. C d.")
    assert [w.category for w in record] == [pyroid.FallbackWarning]
    assert record[0].filename == __file__
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pn.PunktTrainer("A b. C d.")
        pn.PunktSentenceTokenizer("A b. C d.")


@pytest.mark.thread_unsafe(reason="monkeypatches sys.modules")
def test_fallback_without_nltk(monkeypatch: pytest.MonkeyPatch, fresh_warnings: None) -> None:
    _delegate._WARNED.clear()  # each --iterations run (pytest-run-parallel) warns anew
    _block_nltk(monkeypatch)
    with pytest.raises(pyroid.PyroidError, match="not installed"):
        pn.PunktTrainer("A b. C d.")
    with pytest.raises(pyroid.PyroidError, match="not installed"):
        pn.PunktSentenceTokenizer("A b. C d.")
    tok = pn.TreebankWordTokenizer()
    tok.PUNCTUATION = []
    with pytest.raises(pyroid.PyroidError, match="not installed"):
        tok.tokenize("a, b")
    # What is ported needs no NLTK.
    assert pn.NLTKWordTokenizer().tokenize("Don't.") == ["Do", "n't", "."]
    text = "This is a test. And another one."  # (NLTK's result; "b." would be an initial)
    assert pn.PunktSentenceTokenizer().tokenize(text) == ["This is a test.", "And another one."]


def test_punkt_constants() -> None:
    assert pn.NLTK_VERSION == "3.10.4"
    assert pn.REGEX_VERSION == "2026.9.29"
    assert unicodedata.unidata_version == pn.UNICODE_VERSION
    assert tuple(";:,.!?") == pn.PunktSentenceTokenizer.PUNCTUATION
    assert isinstance(pn.PunktTokenizer, type)
    assert issubclass(pn.PunktTokenizer, pn.PunktSentenceTokenizer)


@needs_nltk
def test_punkt_constants_match_nltk() -> None:
    punkt = importlib.import_module("nltk.tokenize.punkt")
    for name in (
        "DEBUG_DECISION_FMT",
        "REASON_DEFAULT_DECISION",
        "REASON_KNOWN_COLLOCATION",
        "REASON_ABBR_WITH_ORTHOGRAPHIC_HEURISTIC",
        "REASON_ABBR_WITH_SENTENCE_STARTER",
        "REASON_INITIAL_WITH_ORTHOGRAPHIC_HEURISTIC",
        "REASON_NUMBER_WITH_ORTHOGRAPHIC_HEURISTIC",
        "REASON_INITIAL_WITH_SPECIAL_ORTHOGRAPHIC_HEURISTIC",
    ):
        assert getattr(pn, name) == getattr(punkt, name), name
    lv, ref_lv = pn.PunktLanguageVars(), REF["PunktLanguageVars"]()
    assert lv.sent_end_chars == ref_lv.sent_end_chars
    assert lv.internal_punctuation == ref_lv.internal_punctuation
    assert lv.word_tokenize("Hi, there... (ok)") == ref_lv.word_tokenize("Hi, there... (ok)")
    assert lv.period_context_re().pattern == ref_lv.period_context_re().pattern


@needs_nltk
def test_debug_decisions_match_nltk(punkt_tab: Any) -> None:
    text = " ".join(DOCS) + " J. S. Bach. 3. Mai. e.g. this. U.S. Army."
    for lang in ("english", "german"):
        ours = list(pn.PunktTokenizer(lang).debug_decisions(text))
        ref = list(REF["PunktTokenizer"](lang).debug_decisions(text))
        assert ours == ref
        for a, b in zip(ours, ref, strict=True):
            assert {k: type(v) for k, v in a.items()} == {k: type(v) for k, v in b.items()}
            assert pn.format_debug_decision(a) == importlib.import_module(
                "nltk.tokenize.punkt"
            ).format_debug_decision(b)
        assert {d["reason"] for d in ours} >= {pn.REASON_DEFAULT_DECISION}


@needs_nltk
def test_sentences_from_tokens_match_nltk(punkt_tab: Any) -> None:
    tok, ref = pn.PunktTokenizer(), REF["PunktTokenizer"]()
    tokens = ["Mr.", "Smith", "went", ".", "He", "liked", "it", "!", "J.", "S.", "Bach"]
    tokens += [".", "ok"]
    got = list(tok.sentences_from_tokens(tokens))
    assert got == list(ref.sentences_from_tokens(tokens))
    assert list(tok.sentences_from_tokens(iter(tokens))) == got
    assert list(tok.sentences_from_tokens([])) == []
    assert outcome(tok.sentences_from_tokens, ["a", ""]) == outcome(
        lambda: list(ref.sentences_from_tokens(["a", ""]))
    )


# ---- lang_vars and token_cls ---------------------------------------------------------------------

SAMPLE = "Subject: Some subject^ Attachments: x। Next one; yes! (Done.) Ok"


@needs_nltk
@pytest.mark.parametrize("base", ["pyroid", "nltk"])
def test_custom_sent_end_chars_run_natively(base: str) -> None:
    cls = pn.PunktLanguageVars if base == "pyroid" else REF["PunktLanguageVars"]
    ref_cls = REF["PunktLanguageVars"]
    for chars in ((".", "?", "!", "^"), (".", "।"), ("!",), (";", ".")):
        ours = pn.PunktSentenceTokenizer(lang_vars=type("V", (cls,), {"sent_end_chars": chars})())
        ref = REF["PunktSentenceTokenizer"](
            lang_vars=type("V", (ref_cls,), {"sent_end_chars": chars})()
        )
        assert native(ours) is not None, chars
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # native: no FallbackWarning
            assert ours.tokenize(SAMPLE) == ref.tokenize(SAMPLE), chars
            want = outcome(ref.debug_decisions, SAMPLE)
            assert outcome(ours.debug_decisions, SAMPLE) == want, chars
    internal = type("V", (cls,), {"internal_punctuation": ",:;'"})()
    assert native(pn.PunktSentenceTokenizer(lang_vars=internal)) is not None


@needs_nltk
@pytest.mark.thread_unsafe(reason="one-time FallbackWarning state")
def test_sent_end_chars_list_runs_in_nltk() -> None:
    # A list (not a tuple) makes NLTK's debug_decisions raise TypeError (str.endswith(list)):
    # pyroid hands such lang_vars to NLTK.
    ours = pn.PunktSentenceTokenizer(
        lang_vars=type("V", (pn.PunktLanguageVars,), {"sent_end_chars": [".", "^"]})()
    )
    ref = REF["PunktSentenceTokenizer"](
        lang_vars=type("V", (REF["PunktLanguageVars"],), {"sent_end_chars": [".", "^"]})()
    )
    assert native(ours) is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.FallbackWarning)
        assert ours.tokenize(SAMPLE) == ref.tokenize(SAMPLE)
        assert outcome(ours.debug_decisions, SAMPLE) == outcome(ref.debug_decisions, SAMPLE)


@needs_nltk
@pytest.mark.parametrize("base", ["pyroid", "nltk"])
def test_lang_vars_changes_after_use_are_seen(base: str) -> None:
    """Swapping a tokenizer's lang_vars, or changing a subclass's sent_end_chars before use.

    Changing ``sent_end_chars`` after a PunktLanguageVars object was used is a documented
    difference: NLTK keeps its patterns compiled at first use, pyroid uses the new characters.
    """
    cls: Any = pn.PunktLanguageVars if base == "pyroid" else REF["PunktLanguageVars"]
    ours_cls: Any = type("Ext", (cls,), {"sent_end_chars": (".", "?", "!")})
    ref_cls: Any = type("Ext", (REF["PunktLanguageVars"],), {"sent_end_chars": (".", "?", "!")})
    ours = pn.PunktSentenceTokenizer(lang_vars=ours_cls())
    ref = REF["PunktSentenceTokenizer"](lang_vars=ref_cls())
    text = "Subject: Some subject^ Attachments: Some attachments; ok. Fine"

    def same() -> None:
        assert ours.tokenize(text) == ref.tokenize(text)
        assert list(ours.span_tokenize(text, False)) == list(ref.span_tokenize(text, False))

    ours_cls.sent_end_chars = ref_cls.sent_end_chars = (".", "^")
    ours._lang_vars, ref._lang_vars = ours_cls(), ref_cls()  # fresh objects: NLTK recompiles
    same()
    ours._lang_vars = type("Semi", (cls,), {"sent_end_chars": (";",)})()
    ref._lang_vars = type("Semi", (REF["PunktLanguageVars"],), {"sent_end_chars": (";",)})()
    same()
    ours._lang_vars, ref._lang_vars = cls(), REF["PunktLanguageVars"]()
    same()


@needs_nltk
def test_lang_vars_instances_have_slots() -> None:
    for lv in (pn.PunktLanguageVars(), REF["PunktLanguageVars"]()):
        with pytest.raises(AttributeError):
            lv.sent_end_chars = ("!",)
    assert pickle.loads(pickle.dumps(pn.PunktLanguageVars())).sent_end_chars == (".", "?", "!")


# Overrides that only NLTK can run: (attribute, value).
NLTK_ONLY_OVERRIDES: list[tuple[str, Any]] = [
    ("_re_word_start", r"[^\(\"\`{\[:;&\#\*@\)}\]\-,!]"),
    ("_re_multi_char_punct", r"(?:\-{3,}|\.{2,})"),
    (
        "_period_context_fmt",
        r"""%(SentEndChars)s(?=(?P<after_tok>%(NonWord)s|\s+(?P<next_tok>\S+)))""",
    ),
    ("word_tokenize", lambda self, s: s.split()),
    ("sent_end_chars", ("..", "!")),  # not one char each
    ("sent_end_chars", ".!"),  # a str, not a tuple
]


@needs_nltk
@pytest.mark.thread_unsafe(reason="pytest.warns, one-time warning state")
@pytest.mark.parametrize(("attr", "value"), NLTK_ONLY_OVERRIDES)
@pytest.mark.parametrize("base", ["pyroid", "nltk"])
def test_other_lang_vars_overrides_fall_back(
    fresh_warnings: None, base: str, attr: str, value: Any
) -> None:
    _delegate._WARNED.clear()  # each --iterations run (pytest-run-parallel) warns anew
    cls = pn.PunktLanguageVars if base == "pyroid" else REF["PunktLanguageVars"]
    ours = pn.PunktSentenceTokenizer(lang_vars=type("V", (cls,), {attr: value})())
    ref_vars = type("V", (REF["PunktLanguageVars"],), {attr: value})()
    ref = REF["PunktSentenceTokenizer"](lang_vars=ref_vars)
    assert native(ours) is None
    with pytest.warns(pyroid.FallbackWarning, match="PunktLanguageVars") as record:
        got = outcome(ours.tokenize, SAMPLE)
    assert got == outcome(ref.tokenize, SAMPLE)
    assert [w.category for w in record] == [pyroid.FallbackWarning], [
        str(w.message) for w in record
    ]
    assert record[0].filename == __file__
    assert ours.tokenize([SAMPLE, "A b."]) == [ref.tokenize(SAMPLE), ref.tokenize("A b.")]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # once per process
        assert outcome(ours.span_tokenize, SAMPLE, False) == outcome(
            lambda: list(ref.span_tokenize(SAMPLE, False))
        )
        assert outcome(ours.debug_decisions, SAMPLE) == outcome(
            lambda: list(ref.debug_decisions(SAMPLE))
        )


@needs_nltk
@pytest.mark.thread_unsafe(reason="pytest.warns, one-time warning state")
def test_custom_token_cls_falls_back(fresh_warnings: None, punkt_tab: Any) -> None:
    _delegate._WARNED.clear()  # each --iterations run (pytest-run-parallel) warns anew
    token_base: Any = REF["PunktToken"]

    class Token(token_base):  # type: ignore[misc]
        @property
        def first_upper(self) -> bool:
            return bool(self.tok[0].isupper() or self.tok[0].isdigit())

    ours = pn.PunktSentenceTokenizer(pn.PunktTokenizer()._params, token_cls=Token)
    ref = REF["PunktSentenceTokenizer"](REF["PunktTokenizer"]()._params, token_cls=Token)
    text = "He left at 3 p.m. 4 people came. Mr. X. 5 more. J. 7 times."
    with pytest.warns(pyroid.FallbackWarning, match="token_cls") as record:
        assert ours.tokenize(text) == ref.tokenize(text)
    assert len(record) == 1, [str(w.message) for w in record]  # one warning per feature
    assert list(ours.span_tokenize(text)) == list(ref.span_tokenize(text))
    assert list(ours.debug_decisions(text)) == list(ref.debug_decisions(text))
    assert ours.tokenize([text, "A b."]) == [ref.tokenize(text), ref.tokenize("A b.")]
    plain = REF["PunktSentenceTokenizer"](REF["PunktTokenizer"]()._params)
    assert ref.tokenize(text) != plain.tokenize(text)  # the custom token class matters here


# ---- the regular-expression tokenizers --------------------------------------------------------

RX_TEXT = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks."


@needs_nltk
@pytest.mark.thread_unsafe(reason="pytest.warns, one-time warning state")
def test_regexp_fallback_warns_once(fresh_warnings: None) -> None:
    _delegate._WARNED.clear()  # each --iterations run (pytest-run-parallel) warns anew
    regexp = importlib.import_module("nltk.tokenize.regexp")
    ours, ref = pn.RegexpTokenizer(r"[A-Z]\w+"), regexp.RegexpTokenizer(r"[A-Z]\w+")
    assert not native_regexp(ours)
    with pytest.warns(pyroid.FallbackWarning, match="RegexpTokenizer") as record:
        got = ours.tokenize(RX_TEXT)
    assert got == ref.tokenize(RX_TEXT) == ["Good", "New", "York", "Please", "Thanks"]
    assert [w.category for w in record] == [pyroid.FallbackWarning]
    assert record[0].filename == __file__
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # once per process
        assert list(ours.span_tokenize(RX_TEXT)) == list(ref.span_tokenize(RX_TEXT))
        assert ours.tokenize([RX_TEXT, "", "Ab Cd"]) == [
            ref.tokenize(d) for d in (RX_TEXT, "", "Ab Cd")
        ]
        col = _testing.column([RX_TEXT, None])
        assert ours.tokenize(col) == [ref.tokenize(RX_TEXT), None]
        flags = pn.RegexpTokenizer(r"\w+", flags=re.I)
        assert flags.tokenize(RX_TEXT) == regexp.RegexpTokenizer(r"\w+", flags=re.I).tokenize(
            RX_TEXT
        )
        assert pn.regexp_tokenize(RX_TEXT, r"\d+") == regexp.regexp_tokenize(RX_TEXT, r"\d+")


def native_regexp(tok: Any) -> bool:
    """Whether a RegexpTokenizer runs in Rust (else NLTK runs)."""
    return bool(tok._native())


@needs_nltk
def test_regexp_pattern_objects_as_nltk() -> None:
    """A compiled pattern gives its ``.pattern`` (its flags are ignored, as in NLTK)."""
    regexp = importlib.import_module("nltk.tokenize.regexp")
    rx = importlib.import_module("regex")
    for pattern in (re.compile(r"\w+", re.I), rx.compile(r"\S+"), re.compile(r"\s+")):
        ours, ref = pn.RegexpTokenizer(pattern), regexp.RegexpTokenizer(pattern)
        assert native_regexp(ours)
        assert ours.tokenize(RX_TEXT) == ref.tokenize(RX_TEXT)
        assert repr(ours) == repr(ref)


@needs_nltk
@pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning")
@pytest.mark.parametrize("pattern", [5, None, b"\\w+", ["\\w+"]], ids=repr)
def test_regexp_bad_patterns_as_nltk(pattern: Any) -> None:
    regexp = importlib.import_module("nltk.tokenize.regexp")
    ours, ref = pn.RegexpTokenizer(pattern), regexp.RegexpTokenizer(pattern)
    assert outcome(ours.tokenize, RX_TEXT) == outcome(ref.tokenize, RX_TEXT)
    assert outcome(ours.span_tokenize, RX_TEXT) == outcome(ref.span_tokenize, RX_TEXT)


def test_regexp_docstring_examples() -> None:
    """nltk/tokenize/regexp.py's module and class docstrings."""
    tokens = [*("Good", "muffins", "cost", "$3.88", "in", "New", "York", ".", "Please", "buy")]
    tokens += ["me", "two", "of", "them", ".", "Thanks", "."]
    assert pn.RegexpTokenizer(r"\w+|\$[\d\.]+|\S+").tokenize(RX_TEXT) == tokens
    assert pn.regexp_tokenize(RX_TEXT, pattern=r"\w+|\$[\d\.]+|\S+") == tokens
    ws = [*("Good", "muffins", "cost", "$3.88", "in", "New", "York.", "Please", "buy", "me")]
    ws += ["two", "of", "them.", "Thanks."]
    assert pn.RegexpTokenizer(r"\s+", gaps=True).tokenize(RX_TEXT) == ws
    assert pn.WhitespaceTokenizer().tokenize(RX_TEXT) == ws
    wp = [*("Good", "muffins", "cost", "$", "3", ".", "88", "in", "New", "York", ".", "Please")]
    wp += ["buy", "me", "two", "of", "them", ".", "Thanks", "."]
    assert pn.WordPunctTokenizer().tokenize(RX_TEXT) == wp
    assert pn.wordpunct_tokenize(RX_TEXT) == wp
    blank = ["Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.", "Thanks."]
    assert pn.BlanklineTokenizer().tokenize(RX_TEXT) == blank
    assert pn.blankline_tokenize(RX_TEXT) == blank
    assert type(pn.WhitespaceTokenizer().span_tokenize(RX_TEXT)) is types.GeneratorType
    assert issubclass(pn.WordPunctTokenizer, pn.RegexpTokenizer)


@pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning")
@needs_nltk
def test_regexp_doctest_examples() -> None:
    """nltk/test/tokenize.doctest, "Regression Tests: Regexp Tokenizer" (patterns NLTK runs)."""
    s = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks."
    s2 = "Alas, it has not rained today. When, do you think, will it rain again?"
    s3 = "<p>Although this is <b>not</b> the case here, we must not relax our vigilance!</p>"
    rt = pn.regexp_tokenize
    assert rt(s2, r'[,\.\?!"]\s*', gaps=False) == [", ", ". ", ", ", ", ", "?"]
    assert rt(s2, r'[,\.\?!"]\s*', gaps=True) == [
        *("Alas", "it has not rained today", "When", "do you think", "will it rain again"),
    ]
    assert rt(s3, r"</?[bp]>", gaps=False) == ["<p>", "<b>", "</b>", "</p>"]
    assert rt(s3, r"</?(?:b|p)>", gaps=False) == ["<p>", "<b>", "</b>", "</p>"]
    assert rt(s3, r"</?(?:b|p)>", gaps=True) == [
        *("Although this is ", "not", " the case here, we must not relax our vigilance!"),
    ]
    assert rt(s3, r"</?(?P<named>b|p)>", gaps=False) == ["p", "b", "b", "p"]
    assert rt(s3, r"</?(?P<named>b|p)>", gaps=True) == [
        *("p", "Although this is ", "b", "not", "b"),
        *(" the case here, we must not relax our vigilance!", "p"),
    ]
    assert rt(s2, r"(?:h|r|l)a(?:s|(?:i|n0))", gaps=False) == ["las", "has", "rai", "rai"]
    assert rt(s2, r"(?:h|r|l)a(?:s|(?:i|n0))", gaps=True) == [
        *("A", ", it ", " not ", "ned today. When, do you think, will it ", "n again?"),
    ]
    assert rt("aabbbcccc", r"(.)\1") == ["a", "b", "c", "c"]
    assert rt(s, pattern=r"\.(?:\s+|$)", gaps=True) == [
        *("Good muffins cost $3.88\nin New York", "Please buy me\ntwo of them", "Thanks"),
    ]


# ---- Treebank pattern tables --------------------------------------------------------------------

TABLES: tuple[str, ...] = (
    "STARTING_QUOTES",
    "PUNCTUATION",
    "PARENS_BRACKETS",
    "CONVERT_PARENTHESES",
)
TABLES += ("DOUBLE_DASHES", "ENDING_QUOTES", "CONTRACTIONS2", "CONTRACTIONS3")


def _table(value: Any) -> Any:
    """A pattern table as (pattern source, flags, replacement) tuples."""
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], str):
        return (value[0].pattern, value[0].flags, value[1])
    if isinstance(value, list):
        return [_table(v) if isinstance(v, tuple) else (v.pattern, v.flags) for v in value]
    return value


@needs_nltk
@pytest.mark.parametrize(
    "name", ["TreebankWordTokenizer", "NLTKWordTokenizer", "TreebankWordDetokenizer"]
)
def test_pattern_tables_match_nltk(name: str) -> None:
    ours, ref = getattr(pn, name), REF[name]
    for table in TABLES:  # the detokenizer has the same eight, with its own patterns
        assert _table(getattr(ours, table)) == _table(getattr(ref, table)), f"{name}.{table}"
    mac, ref_mac = pn.MacIntyreContractions, REF["MacIntyreContractions"]
    for attr in ("CONTRACTIONS2", "CONTRACTIONS3", "CONTRACTIONS4"):
        assert getattr(mac, attr) == getattr(ref_mac, attr)


EDIT_TEXT = "He said: \"don't (go) -- it's 3,000... wanna-be!\" ok."


@needs_nltk
@pytest.mark.thread_unsafe(reason="pytest.warns, one-time warning state")
@pytest.mark.parametrize("name", ["TreebankWordTokenizer", "NLTKWordTokenizer"])
def test_changed_patterns_fall_back(fresh_warnings: None, name: str) -> None:
    _delegate._WARNED.clear()  # each --iterations run (pytest-run-parallel) warns anew
    ours_cls: Any = getattr(pn, name)
    ref_cls: Any = REF[name]
    regex = importlib.import_module("regex")
    extra = (regex.compile(r"!"), r" EXCLAIM ")

    # An instance attribute.
    ours, ref = ours_cls(), ref_cls()
    ours.PUNCTUATION = ref.PUNCTUATION = [*ref_cls.PUNCTUATION, extra]
    with pytest.warns(pyroid.FallbackWarning, match="changed patterns") as record:
        got = ours.tokenize(EDIT_TEXT)
    assert got == ref.tokenize(EDIT_TEXT)
    assert "EXCLAIM" in got
    assert record[0].filename == __file__
    assert list(ours.span_tokenize("a b")) == list(ref.span_tokenize("a b"))

    # A subclass.
    class Sub(ours_cls):  # type: ignore[misc]
        DOUBLE_DASHES = (regex.compile(r"-"), r" - ")

    class RefSub(ref_cls):  # type: ignore[misc]
        DOUBLE_DASHES = (regex.compile(r"-"), r" - ")

    with pytest.warns(pyroid.FallbackWarning, match="Sub with changed patterns"):  # its own reason
        assert Sub().tokenize(EDIT_TEXT) == RefSub().tokenize(EDIT_TEXT)
    assert Sub().tokenize(EDIT_TEXT, True) == RefSub().tokenize(EDIT_TEXT, True)
    assert Sub().tokenize([EDIT_TEXT]) == [RefSub().tokenize(EDIT_TEXT)]

    # The class's own list, changed in place (and put back).
    table = ours_cls.ENDING_QUOTES
    table.append(extra)
    try:
        assert ours_cls().tokenize(EDIT_TEXT) == type(
            "R", (ref_cls,), {"ENDING_QUOTES": [*ref_cls.ENDING_QUOTES, extra]}
        )().tokenize(EDIT_TEXT)
        if name == "NLTKWordTokenizer":  # word_tokenize uses it too
            assert pn.word_tokenize(EDIT_TEXT, preserve_line=True) == ours_cls().tokenize(EDIT_TEXT)
    finally:
        table.pop()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ours_cls().tokenize(EDIT_TEXT) == ref_cls().tokenize(EDIT_TEXT)  # native again


@needs_nltk
@pytest.mark.thread_unsafe(reason="pytest.warns, one-time warning state")
def test_changed_detokenizer_patterns_fall_back(fresh_warnings: None) -> None:
    _delegate._WARNED.clear()  # each --iterations run (pytest-run-parallel) warns anew
    regex = importlib.import_module("regex")
    ours, ref = pn.TreebankWordDetokenizer(), REF["TreebankWordDetokenizer"]()
    ours.STARTING_QUOTES = ref.STARTING_QUOTES = [(regex.compile(r"``"), "<<")]
    toks = ["``", "Hi", ",", "''", "he", "said", "."]
    with pytest.warns(pyroid.FallbackWarning, match="TreebankWordDetokenizer"):
        assert ours.detokenize(toks) == ref.detokenize(toks)
    assert pn.TreebankWordDetokenizer().detokenize(toks) == REF[
        "TreebankWordDetokenizer"
    ]().detokenize(toks)


@needs_nltk
@pytest.mark.thread_unsafe(reason="pytest.warns")
@pytest.mark.parametrize("name", ["TreebankWordTokenizer", "NLTKWordTokenizer"])
@pytest.mark.parametrize("return_str", [True, 1, 0, None, "x"])
def test_return_str_deprecation(name: str, return_str: Any) -> None:
    ours, ref = getattr(pn, name)(), REF[name]()
    with warnings.catch_warnings(record=True) as want:
        warnings.simplefilter("always")
        expected = ref.tokenize("Hi, there.", False, return_str)
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        assert ours.tokenize("Hi, there.", False, return_str) == expected
    assert [(w.category, str(w.message)) for w in got] == [
        (w.category, str(w.message)) for w in want
    ]
    assert all(w.filename == __file__ for w in got)


# ---- models -------------------------------------------------------------------------------


@needs_nltk
@pytest.mark.parametrize("lang", ["klingon", "-", "english_", "ｅｎｇｌｉｓｈ"])
def test_missing_model_with_nltk(punkt_tab: Any, lang: str) -> None:
    """The LookupError is NLTK's own (pyroid asks nltk.data.find)."""
    want = outcome(REF["PunktTokenizer"], lang)
    assert want[:2] == ("raise", "LookupError")
    assert outcome(pn.PunktTokenizer, lang) == want
    assert outcome(pn.sent_tokenize, "A b. C d.", lang) == outcome(
        REF["sent_tokenize"], "A b. C d.", lang
    )
    assert outcome(pn.word_tokenize, "A b.", lang) == outcome(REF["word_tokenize"], "A b.", lang)
    assert pn.punkt_model_available(lang) is False
    assert pn.word_tokenize("A b.", lang, preserve_line=True) == ["A", "b", "."]  # no model needed


BAD_LANGUAGES: list[Any] = [5, None, b"english", 2.5, "", "english/../german", "eng lish"]
BAD_LANGUAGES += ["english\n", "..", "a/b", "a\\b", "\x00", "english.", "~"]
BAD_LANGUAGES += ["e" + chr(0x301) + "nglish"]  # NFD: a combining mark is not alphanumeric


@needs_nltk
@pytest.mark.parametrize("lang", BAD_LANGUAGES, ids=repr)
def test_language_names_validated_as_nltk(lang: Any) -> None:
    want = outcome(REF["_punkt_language"], lang)
    assert want[0] == "raise"
    assert want[1] in ("TypeError", "ValueError")
    assert outcome(pn.PunktTokenizer, lang) == want
    assert outcome(pn.sent_tokenize, "A b.", lang) == outcome(REF["sent_tokenize"], "A b.", lang)
    assert outcome(pn.word_tokenize, "A b.", lang) == outcome(REF["word_tokenize"], "A b.", lang)
    assert outcome(pn.punkt_model_available, lang) == outcome(REF["punkt_model_available"], lang)
    tok = pn.PunktTokenizer.__new__(pn.PunktTokenizer)
    pn.PunktSentenceTokenizer.__init__(tok)
    assert outcome(tok.load_lang, lang) == want


@needs_nltk
def test_unhashable_language_as_nltk() -> None:
    for lang in (["english"], {"english": 1}):
        assert outcome(pn.sent_tokenize, "A b.", lang) == outcome(
            REF["sent_tokenize"], "A b.", lang
        )
        assert outcome(pn.punkt_model_available, lang) == outcome(
            REF["punkt_model_available"], lang
        )


def test_language_name_messages() -> None:
    with pytest.raises(TypeError, match=r"^A Punkt language name is a str, not 'int'$"):
        pn.PunktTokenizer(5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"^Invalid Punkt language name: 'english/\.\./german'$"):
        pn.PunktTokenizer("english/../german")
    with pytest.raises(ValueError, match="Invalid Punkt language name"):
        pn.sent_tokenize("A b.", "")


def test_str_subclass_language(punkt_tab: Any) -> None:
    class Lang(str):
        __slots__ = ()

        def __str__(self) -> str:
            return "german"

    tok = pn.PunktTokenizer(Lang("english"))
    assert cast(Any, tok)._lang == "english"
    assert type(cast(Any, tok)._lang) is str
    assert tok.tokenize(DOCS[0]) == pn.PunktTokenizer("english").tokenize(DOCS[0])


def test_punkt_model_available(punkt_tab: Any) -> None:
    assert pn.punkt_model_available() is True
    assert pn.punkt_model_available("german") is True
    assert pn.punkt_model_available("klingon") is False
    with pytest.raises(TypeError):
        pn.punkt_model_available(5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Invalid Punkt language name"):
        pn.punkt_model_available("a/b")


@needs_nltk
@pytest.mark.parametrize("lang", ["english", "german", "turkish", "malayalam"])
def test_loaded_parameters_match_nltk(punkt_tab: Any, lang: str) -> None:
    ours, ref = pn.PunktTokenizer(lang)._params, REF["PunktTokenizer"](lang)._params
    assert ours.abbrev_types == ref.abbrev_types
    assert ours.collocations == ref.collocations
    assert ours.sent_starters == ref.sent_starters
    assert dict(ours.ortho_context) == dict(ref.ortho_context)
    tok = pn.PunktTokenizer("english")
    tok.load_lang(lang)
    assert tok.tokenize(" ".join(DOCS)) == REF["PunktTokenizer"](lang).tokenize(" ".join(DOCS))


@needs_nltk
@pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning")  # an NLTK path pointer: NLTK reads it
def test_load_punkt_params(punkt_tab: Any, tmp_path: Path) -> None:
    lang_dir = tmp_path / "english"
    lang_dir.mkdir()
    for name, data in punkt_tab.files("english").items():
        (lang_dir / name).write_bytes(data)
    ours = pn.load_punkt_params(str(lang_dir))
    also = pn.load_punkt_params(lang_dir)
    ref = REF["load_punkt_params"](nltk.data.find("tokenizers/punkt_tab/english/"))
    from_pointer = pn.load_punkt_params(nltk.data.find("tokenizers/punkt_tab/english/"))
    for p in (ours, also, from_pointer):
        assert isinstance(p, pn.PunktParameters)
        assert p.abbrev_types == ref.abbrev_types
        assert p.collocations == ref.collocations
        assert p.sent_starters == ref.sent_starters
        assert dict(p.ortho_context) == dict(ref.ortho_context)


@pytest.mark.thread_unsafe(reason="pickles a tokenizer")
def test_pickle_and_copy(punkt_tab: Any) -> None:
    tok = pn.PunktTokenizer("german")
    for clone in (pickle.loads(pickle.dumps(tok)), copy.deepcopy(tok), copy.copy(tok)):
        assert clone.tokenize(DOCS[4]) == tok.tokenize(DOCS[4])
    for wt in (pn.TreebankWordTokenizer(), pn.NLTKWordTokenizer(), pn.TreebankWordDetokenizer()):
        clone = pickle.loads(pickle.dumps(wt))
        assert type(clone) is type(wt)


def _block_nltk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every ``import nltk...`` fail, as if NLTK were not installed."""
    for name in [m for m in sys.modules if m == "nltk" or m.startswith("nltk.")]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "nltk", None)


@pytest.fixture
def no_nltk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """No NLTK, and NLTK's default data path (NLTK_DATA, ~, sys.prefix) in an empty tmp dir."""
    _block_nltk(monkeypatch)
    home, prefix, data = tmp_path / "home", tmp_path / "prefix", tmp_path / "nltk_data"
    for d in (home, prefix, data):
        d.mkdir()
    monkeypatch.setenv("NLTK_DATA", str(data))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("APPDATA", str(home))  # Windows: %APPDATA%\nltk_data
    monkeypatch.delenv("APPENGINE_RUNTIME", raising=False)
    if sys.platform == "win32":
        for drive in ("C:", "D:", "E:"):
            if Path(drive, "\\nltk_data", "tokenizers").exists():
                pytest.skip(f"{drive}\\nltk_data has tokenizers installed system-wide")
    monkeypatch.setattr(sys, "prefix", str(prefix))
    for system in ("/usr/share", "/usr/local/share", "/usr/lib", "/usr/local/lib"):
        if Path(system, "nltk_data", "tokenizers").exists():
            pytest.skip(f"{system}/nltk_data has tokenizers installed system-wide")
    cast(Any, pn)._get_punkt_tokenizer.cache_clear()
    yield data
    cast(Any, pn)._get_punkt_tokenizer.cache_clear()


@pytest.mark.thread_unsafe(reason="monkeypatches sys.modules and the environment")
def test_missing_model_without_nltk(no_nltk: Path) -> None:
    with pytest.raises(LookupError) as info:
        pn.PunktTokenizer("english")
    msg = str(info.value)
    assert "Resource 'punkt_tab' not found." in msg
    assert "nltk.download('punkt_tab')" in msg
    assert "pip install nltk" in msg  # the downloader needs NLTK
    assert "Attempted to load 'tokenizers/punkt_tab/english/'" in msg
    assert repr(str(no_nltk)) in msg  # the search path is listed
    with pytest.raises(LookupError, match="punkt_tab"):
        pn.sent_tokenize("A b. C d.")
    assert pn.punkt_model_available() is False
    with pytest.raises(ValueError, match="Invalid Punkt language name"):
        pn.PunktTokenizer("../english")
    # Treebank needs no model, and no NLTK.
    assert pn.word_tokenize("Don't go.", preserve_line=True) == ["Do", "n't", "go", "."]


PUNKT_FILE_NAMES = (
    "collocations.tab",
    "sent_starters.txt",
    "abbrev_types.txt",
    "ortho_context.tab",
)


def _write_model(
    root: Path, files: dict[str, bytes], *, as_zip: bool, lang: str = "english"
) -> None:
    tok_dir = root / "tokenizers"
    tok_dir.mkdir(parents=True, exist_ok=True)
    if as_zip:
        with zipfile.ZipFile(tok_dir / "punkt_tab.zip", "w") as zf:
            zf.writestr("punkt_tab/", "")
            zf.writestr(f"punkt_tab/{lang}/", "")
            for name, data in files.items():
                zf.writestr(f"punkt_tab/{lang}/{name}", data)
    else:
        d = tok_dir / "punkt_tab" / lang
        d.mkdir(parents=True, exist_ok=True)  # pytest-run-parallel's --iterations reuse tmp_path
        for name in PUNKT_FILE_NAMES:
            (d / name).unlink(missing_ok=True)
        for name, data in files.items():
            (d / name).write_bytes(data)


def _lines(data: bytes) -> list[str]:
    """The lines of a model file as NLTK reads them (text mode, then one final "\\n" removed)."""
    return [x[:-1] if x.endswith("\n") else x for x in io.StringIO(data.decode("utf-8"), None)]


def _params_parsed_here(punkt_tab: Any) -> pn.PunktParameters:
    """The English parameters, parsed with NLTK's rules here (independently of pyroid's loader)."""
    files = {k: _lines(v) for k, v in punkt_tab.files("english").items()}
    params = pn.PunktParameters()
    params.abbrev_types = set(files["abbrev_types.txt"])
    params.sent_starters = set(files["sent_starters.txt"])
    params.collocations = {(a, b) for a, b in (x.split("\t") for x in files["collocations.tab"])}
    params.ortho_context = defaultdict(
        int, {a: int(b) for a, b in (x.split("\t") for x in files["ortho_context.tab"])}
    )
    return params


@pytest.mark.thread_unsafe(reason="monkeypatches sys.modules and the environment")
@pytest.mark.parametrize("as_zip", [False, True], ids=["directory", "zip"])
def test_model_without_nltk(punkt_tab: Any, no_nltk: Path, as_zip: bool) -> None:
    _write_model(no_nltk, punkt_tab.files("english"), as_zip=as_zip)
    tok = pn.PunktTokenizer()
    want = _params_parsed_here(punkt_tab)
    assert tok._params.abbrev_types == want.abbrev_types
    assert tok._params.collocations == want.collocations
    assert tok._params.sent_starters == want.sent_starters
    assert dict(tok._params.ortho_context) == dict(want.ortho_context)
    text = " ".join(DOCS)
    assert tok.tokenize(text) == pn.PunktSentenceTokenizer(want).tokenize(text)
    assert pn.sent_tokenize(text) == tok.tokenize(text)
    assert pn.punkt_model_available() is True
    assert pn.punkt_model_available("german") is False


@pytest.mark.thread_unsafe(reason="monkeypatches sys.modules and the environment")
@pytest.mark.parametrize("as_zip", [False, True], ids=["directory", "zip"])
def test_incomplete_model_without_nltk(punkt_tab: Any, no_nltk: Path, as_zip: bool) -> None:
    files = punkt_tab.files("english")
    del files["ortho_context.tab"]
    _write_model(no_nltk, files, as_zip=as_zip)
    with pytest.raises(LookupError, match=r"has no 'ortho_context\.tab'; reinstall it"):
        pn.PunktTokenizer()
    assert pn.punkt_model_available() is False


@needs_nltk
@pytest.mark.thread_unsafe(reason="changes nltk.data.path")
@pytest.mark.parametrize("as_zip", [False, True], ids=["directory", "zip"])
def test_incomplete_model_with_nltk(punkt_tab: Any, tmp_path: Path, as_zip: bool) -> None:
    files = punkt_tab.files("english")
    del files["sent_starters.txt"]
    _write_model(tmp_path, files, as_zip=as_zip, lang="incomplete")
    nltk.data.path.insert(0, str(tmp_path))
    try:
        want = outcome(REF["PunktTokenizer"], "incomplete")
        assert want[:2] == ("raise", "LookupError")
        assert outcome(pn.PunktTokenizer, "incomplete") == want
        assert pn.punkt_model_available("incomplete") is False
    finally:
        nltk.data.path.remove(str(tmp_path))
        cast(Any, pn)._get_punkt_tokenizer.cache_clear()


# ---- rouge-score's split_summaries -----------------------------------------------------------


@needs_nltk
def test_rouge_split_summaries_match_rouge_score(punkt_tab: Any) -> None:
    rouge_ref: Any = pytest.importorskip("rouge_score.rouge_scorer")
    from pyroid.metrics import rouge_scorer

    targets = [*DOCS, "Mr. Smith went. He did not. U.S. troops left. 3. Mai. «Fin.» Ok"]
    predictions = [*reversed(DOCS), "Smith went. He did."]
    for types_ in (["rougeLsum"], ["rouge1", "rougeLsum"]):
        ours = rouge_scorer.RougeScorer(types_, use_stemmer=True, split_summaries=True)
        theirs = rouge_ref.RougeScorer(types_, use_stemmer=True, split_summaries=True)
        for t, p in zip(targets, predictions, strict=True):
            assert repr(ours.score(t, p)) == repr(theirs.score(t, p)), (t, p)
        got = ours.score_batch(targets, predictions)
        assert [repr(d) for d in got] == [
            repr(theirs.score(t, p)) for t, p in zip(targets, predictions, strict=True)
        ]


# ---- concurrency ------------------------------------------------------------------------------


def test_shared_tokenizer_across_threads(punkt_tab: Any) -> None:
    tok = pn.PunktTokenizer()
    want = [tok.tokenize(d) for d in BIG[:40]]
    errors: list[BaseException] = []

    def work() -> None:
        try:
            for _ in range(5):
                assert [tok.tokenize(d) for d in BIG[:40]] == want
                assert tok.tokenize(BIG[:40], threads=2) == want
        except BaseException as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
