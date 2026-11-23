# Derived from NLTK 3.10.4 (Apache-2.0), © 2001-2026 NLTK Project; modified.
# Confusable letters, dashes, quotes and spaces are the test data here.
# ruff: noqa: RUF001, RUF002
"""Differential tests: pyroid.nltk must give NLTK 3.10.4's results exactly.

- **Hypothesis** (against the live NLTK, skipped without it): text built from the pieces the
  tokenizers treat specially (sentence ends, ellipses, straight, curly and angle quotes, ``--``
  and U+2012..U+2015, abbreviations and initials of several languages, numbers, clitics with
  ``İ``/``ı``/``ſ``, every kind of line break and whitespace including U+0085, U+2028 and
  U+001C..U+001F, combining marks, superscripts, CJK, emoji), through every public function and
  method, both ``realign_boundaries`` values and every ``punkt_tab`` language; the
  regular-expression tokenizers with every pattern pyroid runs natively (and some it
  hands to NLTK), with both ``gaps`` and ``discard_empty``; batches as
  lists, tuples and Arrow columns with several ``threads`` values.
- **Every code point** alone, and inside a few dense templates (every code point below U+3400
  and every 31st above; all of them with ``PYROID_NLTK_SWEEP=full``), through ``sent_tokenize``,
  ``word_tokenize``, both Treebank tokenizers and the detokenizer (in child processes, one per
  core, since NLTK takes minutes for 1.1M calls).
- **NLTK's own examples**: the inputs and expected outputs of NLTK's doctests
  (``tokenize.doctest``, the docstrings of ``punkt``, ``treebank``, ``destructive``) and unit
  tests (``test_tokenize.py``) for these tokenizers, checked without NLTK.

The Punkt tests need the ``punkt_tab`` models (the ``punkt_tab`` fixture); lone surrogates,
which pyroid rejects with ``UnicodeEncodeError``, are tested at the end.
"""

from __future__ import annotations

import functools
import importlib
import json
import os
import re
import subprocess
import sys
import textwrap
import types
from collections.abc import Callable
from typing import Any, cast

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

import pyroid.nltk as pn
from pyroid._pyroid import _testing

NLTK_VERSION = "3.10.4"
REGEX_VERSION = "2026.9.29"


def _optional(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


nltk: Any = _optional("nltk")
_regex: Any = _optional("regex")
if nltk is None:
    _NO_NLTK: str | None = "NLTK (the reference) is not installed"
elif (nltk.__version__, getattr(_regex, "__version__", None)) != (NLTK_VERSION, REGEX_VERSION):
    _NO_NLTK = (
        f"needs nltk {NLTK_VERSION} with regex {REGEX_VERSION}, found {nltk.__version__} "
        f"with {getattr(_regex, '__version__', None)}"
    )
else:
    _NO_NLTK = None
needs_nltk = pytest.mark.skipif(_NO_NLTK is not None, reason=_NO_NLTK or "")

#: NLTK's own objects, taken at import, before any test can patch them (test_nltk_patch.py).
REF: dict[str, Any] = {}
if _NO_NLTK is None:
    for _path in (
        "nltk.tokenize:sent_tokenize",
        "nltk.tokenize:word_tokenize",
        "nltk.tokenize.punkt:PunktTokenizer",
        "nltk.tokenize.punkt:PunktSentenceTokenizer",
        "nltk.tokenize.punkt:PunktParameters",
        "nltk.tokenize.punkt:PunktLanguageVars",
        "nltk.tokenize.punkt:format_debug_decision",
        "nltk.tokenize.treebank:TreebankWordTokenizer",
        "nltk.tokenize.treebank:TreebankWordDetokenizer",
        "nltk.tokenize.destructive:NLTKWordTokenizer",
        "nltk.tokenize.util:align_tokens",
    ):
        _mod, _name = _path.split(":")
        REF[_name] = getattr(importlib.import_module(_mod), _name)

LANGUAGES = (
    "czech",
    "danish",
    "dutch",
    "english",
    "estonian",
    "finnish",
    "french",
    "german",
    "greek",
    "italian",
    "malayalam",
    "norwegian",
    "polish",
    "portuguese",
    "russian",
    "slovene",
    "spanish",
    "swedish",
    "turkish",
)

examples = settings(suppress_health_check=[HealthCheck.too_slow])


# ---- comparing outcomes ----------------------------------------------------------------------


def outcome(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, ...]:
    """``("ok", result)`` (generators listed) or ``("raise", type name, message)``."""
    try:
        r = fn(*args, **kwargs)
        if isinstance(r, types.GeneratorType):
            r = list(r)
        return ("ok", r)
    except Exception as e:
        return ("raise", type(e).__name__, str(e))


def same(a: Any, b: Any) -> bool:
    """Equal, with the same types all the way down (True is not 1, a tuple is not a list)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return bool(a == b)


def assert_same(got: Any, want: Any, what: str) -> None:
    assert same(got, want), f"{what}\n pyroid: {got!r:.3000}\n   nltk: {want!r:.3000}"


def native(tok: Any) -> Any:
    """The native Punkt a tokenizer runs with, or None (then NLTK runs)."""
    return tok._impl()


@functools.cache
def ref_punkt(lang: str) -> Any:
    return REF["PunktTokenizer"](lang)


@functools.cache
def our_punkt(lang: str) -> pn.PunktTokenizer:
    return pn.PunktTokenizer(lang)


def ref_word_tokenizer(flavor: str) -> Any:
    return REF["NLTKWordTokenizer" if flavor == "nltk" else "TreebankWordTokenizer"]()


def our_word_tokenizer(flavor: str) -> pn.TreebankWordTokenizer | pn.NLTKWordTokenizer:
    return pn.NLTKWordTokenizer() if flavor == "nltk" else pn.TreebankWordTokenizer()


# ---- strategies --------------------------------------------------------------------------------

SENT_ENDS = [".", "?", "!", "...", "..", "....", ". . .", ". .", "?!", "!!", "!?", ".?", "…"]
SENT_ENDS += ["。", "؟", "।", "!!!", "?.", ".!"]
OPENERS = ['"', "'", "(", "[", "{", "<", "‘", "“", "„", "‚", "«", "‹", "``", "`", "```", "''"]
CLOSERS = ['"', "'", ")", "]", "}", ">", "’", "”", "»", "›", "''", "')", '")', ")'", "”)"]
DASHES = ["-", "--", "---", "----", "‐", "‑", "‒", "–", "—", "―"]
DASHES += ["−", "-x-", "a--b", "–1"]
ABBREVIATIONS = [
    *("Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St.", "Mt.", "Inc.", "Co.", "Ltd.", "Corp.", "Jr."),
    *("U.S.", "U.S.A.", "U.K.", "e.g.", "i.e.", "etc.", "vs.", "cf.", "al.", "Jan.", "Feb."),
    *("No.", "Nr.", "Fig.", "p.", "pp.", "ca.", "approx.", "a.m.", "p.m.", "A.B.", "S.", "J."),
    *("A.", "Z.", "M.", "x.", "z.B.", "usw.", "bzw.", "d.h.", "Str.", "Hr.", "Mme.", "av.", "Sr."),
    *("Sra.", "pág.", "т.е.", "т.д.", "г.", "ул.", "č.", "tj.", "tzv.", "vb.", "örn.", "Dz."),
    *("św.", "ks.", "nr.", "bl.a.", "t.ex.", "f.eks.", "jne.", "esim.", "κ.λπ.", "π.χ.", "sig."),
    *("ecc.", "Exmo.", "Dott.", "art.", "jf.", "mv.", "o.a.", "ds.", "kr.", "dvs.", "St", "Dr"),
]
NUMBERS = [
    *("1.", "2.", "12.", "1.5", "3.14", "1,000", "$3.88", "40.75%", "3:00", "12‒34", "3–1"),
    *("1st", "2nd", "IV.", "xii.", "²", "½", "١٢", "٣.", "०.", "1.2.3", "9.5", "525,600", "-1"),
    *(".5", "5.", "-1.", ",5", "1-2.", "1.-", "2024.", "10.000", "1.000,00", "³.", "Ⅻ.", "⑦."),
]
CLITICS = [
    *("don't", "Don't", "DON'T", "can't", "won't", "I'm", "you'll", "we've", "he'd", "she's"),
    *("it's", "'tis", "'Twas", "'TIS", "cannot", "Cannot", "CANNOT", "gonna", "gimme", "gotta"),
    *("lemme", "wanna", "wanna\n", "more'n", "d'ye", "o'clock", "O'Connor", "'re", "'ve", "'ll"),
    *("'s", "'S", "'M", "'D", "N'T", "n't", "'d", "İ'm", "İSN'T", "ıt's", "ſhe's", "DİDN'T"),
    *("dİdn't", "ſ's", "Iſ'nt", "can not", "whaddya", "whatcha", "l'homme", "dell'anno"),
    *("'hello'", "'Hard'", "'v'", "rock'n'roll", "'90s", "y'all", "ma'am", "''tis", "'''"),
]
WORDS = [
    *("the", "The", "THE", "a", "A", "i", "I", "dog", "Smith", "Washington", "and", "And"),
    *("however", "However", "But", "but", "In", "in", "He", "he", "It", "it", "Bach", "word"),
    *("Über", "über", "école", "École", "İzmir", "izmir", "ıssız", "ſtraße", "Straße", "STRASSE"),
    *("ǅuro", "ǆ", "Σοφία", "ΣΟΦΟΣ", "σοφός", "Москва", "москва", "Ёлка", "日本語", "中文"),
    *("한국어", "ภาษา", "مرحبا", "שלום", "हिन्दी", "മലയാളം", "😀", "👍🏽", "👨‍👩‍👧", "🇩🇪"),
    *("é", "Äbc", "x²", "ﬁne", "ẞ", "ǰ", "ΐ", "İ̇", "Ⅷ", "ⓐ", "𝐀𝐁", "ꞵ", "Ꞵ"),
    *("ǈ", "ῼ", "ꭰ", "Ꭰ", "ʼn", "ŉ", "Ω", "ω", "ǲ", "ß", "ﬀ", "K", "Å", "ϴ", "ϑ"),
]
SPACES = [" ", "  ", "   ", "\n", "\n\n", "\n \n", "\r\n", "\r\n\r\n", "\r", "\t", "\x0b", "\x0c"]
SPACES += ["\x85", " ", " ", "\x1c", "\x1d", "\x1e", "\x1f", "\xa0", "　", " "]
SPACES += ["​", "﻿", " \n ", "᠎", " ", " ", " "]
SYMBOLS = [",", ";", ":", "&", "#", "@", "$", "%", "*", "/", "\\", "^", "~", "|", "_", "+"]
SYMBOLS += ["=", "§", "¶", "•", "·", "…", "‰", "°", "©", "™", "-LRB-", "-RRB-", "-LSB-", "-RCB-"]
SYMBOLS += [",,", ",.", ".,", ":,", "1,", ",a", "a,", ":1", "a:", "!,", "?:", "*a", "a*"]

NOT_SURROGATE = st.characters(exclude_categories=["Cs"])
PIECE = st.one_of(
    st.sampled_from(WORDS),
    st.sampled_from(ABBREVIATIONS),
    st.sampled_from(NUMBERS),
    st.sampled_from(CLITICS),
    st.sampled_from(SENT_ENDS),
    st.sampled_from(OPENERS),
    st.sampled_from(CLOSERS),
    st.sampled_from(DASHES),
    st.sampled_from(SYMBOLS),
    st.sampled_from(SPACES),
    NOT_SURROGATE,
    st.text(NOT_SURROGATE, max_size=4),
)
SEP = st.sampled_from([" ", " ", " ", " ", "", "", *SPACES])
TEXT = st.lists(st.tuples(PIECE, SEP), max_size=30).map(lambda ps: "".join(a + b for a, b in ps))

# Sentence-like text: words, an end, maybe a closer, a separator; Punkt's decisions need these.
WORD_LIKE = st.one_of(
    st.sampled_from(WORDS), st.sampled_from(ABBREVIATIONS), st.sampled_from(NUMBERS), PIECE
)
SENTENCE = st.tuples(
    st.sampled_from(["", "", "", *OPENERS]),
    st.lists(WORD_LIKE, min_size=1, max_size=6).map(" ".join),
    st.sampled_from(SENT_ENDS),
    st.sampled_from(["", "", "", *CLOSERS]),
    SEP,
).map("".join)
SENTENCES = st.lists(SENTENCE, max_size=6).map("".join)
DOC = st.one_of(TEXT, SENTENCES, st.text(NOT_SURROGATE, max_size=40))

TOKEN = st.one_of(
    st.sampled_from(["``", "''", '"', "'", "`", "n't", "'s", "'ll", "...", "--", "-LRB-", "-RRB-"]),
    st.sampled_from(["-LSB-", "-RSB-", "-LCB-", "-RCB-", "(", ")", "[", "]", "{", "}", "<", ">"]),
    st.sampled_from(["$", "#", "%", ";", ":", ",", ".", "?", "!", "can", "not", "gon", "na"]),
    st.sampled_from(["'t", "is", "was", "d", "'ye", "more", "'n", "wan", "lem", "me", "got"]),
    st.sampled_from(WORDS),
    st.sampled_from(CLITICS),
    st.sampled_from(SPACES),
    PIECE,
)
TOKENS = st.lists(TOKEN, max_size=25)

LANG = st.sampled_from(LANGUAGES)

TRICKY_TEXTS = [
    "",
    " ",
    "\n",
    ".",
    "..",
    ". . .",
    "Mr. Smith went to Washington. He liked it.",
    "See Section 3).  Or Section 2).  ",
    '(How does it deal with this parenthesis?)  "It should be part of the\nprevious sentence."',
    "“First sentence.” Next one. Il a dit «bonjour.» Puis il est parti.",
    "This. . . That. This.. . That. This. ,. That",
    "1. This is R .\n2. This is A .\t3. That's all",
    "I can't go. 'Tis the season! Then\x85there was\x1cnone.",
    "İSTANBUL'S ſtate İ. J. S. Bach. ı. ſ. ǅ. Ǆ.",
    "He said ― wait — no – 12‒34.",
    "a\r\nb.\rC.\r\n\r\nD.",
]


# ---- Punkt ---------------------------------------------------------------------------------------


@needs_nltk
@examples
@given(text=DOC, lang=LANG)
@example(text="Mr. Smith went to Washington. He liked it.", lang="english")
@example(text="Ich muss usw. fur einen Kuchen einzukaufen. Ich muss.", lang="german")
def test_sent_tokenize(punkt_tab: Any, text: str, lang: str) -> None:
    assert_same(
        outcome(pn.sent_tokenize, text, lang),
        outcome(REF["sent_tokenize"], text, lang),
        f"sent_tokenize({text!r}, {lang!r})",
    )


@needs_nltk
@examples
@given(text=DOC, lang=LANG, realign=st.booleans())
def test_punkt_methods(punkt_tab: Any, text: str, lang: str, realign: bool) -> None:
    ours, ref = our_punkt(lang), ref_punkt(lang)
    calls: list[tuple[str, Callable[[Any], Any]]] = [
        ("tokenize", lambda t: t.tokenize(text, realign)),
        ("sentences_from_text", lambda t: t.sentences_from_text(text, realign)),
        ("span_tokenize", lambda t: t.span_tokenize(text, realign)),
        ("text_contains_sentbreak", lambda t: t.text_contains_sentbreak(text)),
        ("sentences_from_text_legacy", lambda t: list(t.sentences_from_text_legacy(text))),
        ("sentences_from_tokens", lambda t: list(t.sentences_from_tokens(text.split()))),
        ("debug_decisions", lambda t: list(t.debug_decisions(text))),
        ("tokenize_sents", lambda t: t.tokenize_sents([text, "", text])),
        ("span_tokenize_sents", lambda t: list(t.span_tokenize_sents([text, text]))),
    ]
    for name, call in calls:
        assert_same(outcome(call, ours), outcome(call, ref), f"{lang} {name}({text!r}, {realign})")


@needs_nltk
@examples
@given(tokens=TOKENS, lang=LANG)
def test_sentences_from_tokens(punkt_tab: Any, tokens: list[str], lang: str) -> None:
    got = outcome(our_punkt(lang).sentences_from_tokens, tokens)
    want = outcome(ref_punkt(lang).sentences_from_tokens, tokens)
    assert_same(got, want, f"sentences_from_tokens({tokens!r}) {lang}")
    if got[0] == "ok":  # the same token objects, grouped
        assert [t for s in got[1] for t in s] == tokens


@needs_nltk
@examples
@given(text=DOC, lang=LANG)
def test_debug_decisions_format(punkt_tab: Any, text: str, lang: str) -> None:
    # debug_decisions can raise (IndexError) part way, in NLTK too: compare the outcomes.
    ours, ref = our_punkt(lang), ref_punkt(lang)
    got = outcome(lambda: [pn.format_debug_decision(d) for d in ours.debug_decisions(text)])
    want = outcome(lambda: [REF["format_debug_decision"](d) for d in ref.debug_decisions(text)])
    assert_same(got, want, f"format_debug_decision of {lang} debug_decisions({text!r})")


END_CHARS = [".", "?", "!", "^", "।", "。", ";", "…", "]", "-", "\\", " ", "\n", "'", ")", "a"]


@needs_nltk
@pytest.mark.thread_unsafe(
    reason="NLTK compiles a new pattern per example; when the regex module's cache is full it "
    "evicts with the global random module, which Hypothesis reports in another thread"
)
@examples
@given(text=DOC, end_chars=st.lists(st.sampled_from(END_CHARS), min_size=1))
@example(text="Subject: Some subject^ Attachments: Some attachments", end_chars=[".", "^"])
def test_custom_sent_end_chars(text: str, end_chars: list[str]) -> None:
    """A PunktLanguageVars subclass changing only sent_end_chars runs natively, as NLTK."""
    chars = tuple(dict.fromkeys(end_chars))
    ours_cls = type("Vars", (pn.PunktLanguageVars,), {"sent_end_chars": chars})
    ref_cls = type("Vars", (REF["PunktLanguageVars"],), {"sent_end_chars": chars})
    ours = pn.PunktSentenceTokenizer(lang_vars=ours_cls())
    ref = REF["PunktSentenceTokenizer"](lang_vars=ref_cls())
    assert native(ours) is not None  # native
    for realign in (True, False):
        assert_same(
            outcome(ours.span_tokenize, text, realign),
            outcome(ref.span_tokenize, text, realign),
            f"{chars} span_tokenize({text!r}, {realign})",
        )
    assert_same(
        outcome(ours.debug_decisions, text),
        outcome(ref.debug_decisions, text),
        f"{chars} debug_decisions({text!r})",
    )


PARAM_WORDS = st.sampled_from(["dr", "mr", "smith", "the", "a", "x", "u.s", "e.g", "ab", "i"])
PARAM_WORDS = st.one_of(PARAM_WORDS, st.text(NOT_SURROGATE, min_size=1, max_size=3))


@needs_nltk
@examples
@given(
    text=DOC,
    abbrevs=st.sets(PARAM_WORDS, max_size=6),
    starters=st.sets(PARAM_WORDS, max_size=6),
    collocations=st.sets(st.tuples(PARAM_WORDS, PARAM_WORDS), max_size=6),
    ortho=st.dictionaries(PARAM_WORDS, st.integers(-300, 300), max_size=8),
)
def test_custom_parameters(
    text: str,
    abbrevs: set[str],
    starters: set[str],
    collocations: set[tuple[str, str]],
    ortho: dict[str, int],
) -> None:
    """Hand-made parameters (pyroid's and NLTK's PunktParameters) give NLTK's sentences."""
    for params_cls in (pn.PunktParameters, REF["PunktParameters"]):
        ours_params, ref_params = params_cls(), REF["PunktParameters"]()
        for p in (ours_params, ref_params):
            p.abbrev_types.update(abbrevs)
            p.sent_starters.update(starters)
            p.collocations.update(collocations)
            for k, v in ortho.items():
                p.ortho_context[k] = v
        ours = pn.PunktSentenceTokenizer(ours_params)
        ref = REF["PunktSentenceTokenizer"](ref_params)
        for realign in (True, False):
            assert_same(
                outcome(ours.span_tokenize, text, realign),
                outcome(ref.span_tokenize, text, realign),
                f"{params_cls.__module__} params span_tokenize({text!r}, {realign})",
            )
        assert_same(
            outcome(ours.debug_decisions, text),
            outcome(ref.debug_decisions, text),
            f"debug_decisions({text!r})",
        )


# ---- word_tokenize and the Treebank tokenizers ------------------------------------------------


@needs_nltk
@examples
@given(text=DOC, lang=LANG, preserve_line=st.booleans())
@example(text="They'll save and invest more. I cannot go!", lang="english", preserve_line=False)
def test_word_tokenize(punkt_tab: Any, text: str, lang: str, preserve_line: bool) -> None:
    assert_same(
        outcome(pn.word_tokenize, text, lang, preserve_line),
        outcome(REF["word_tokenize"], text, lang, preserve_line),
        f"word_tokenize({text!r}, {lang!r}, {preserve_line})",
    )


@needs_nltk
@examples
@given(text=DOC, flavor=st.sampled_from(["treebank", "nltk"]), convert=st.booleans())
@example(text="a... b \"c\" ''d'' ``e`` (f) [g] --h", flavor="treebank", convert=True)
def test_word_tokenizers(text: str, flavor: str, convert: bool) -> None:
    ours, ref = our_word_tokenizer(flavor), ref_word_tokenizer(flavor)
    calls: list[tuple[str, Callable[[Any], Any]]] = [
        ("tokenize", lambda t: t.tokenize(text, convert)),
        ("span_tokenize", lambda t: t.span_tokenize(text)),
        ("tokenize_sents", lambda t: t.tokenize_sents([text, "", text])),
        ("span_tokenize_sents", lambda t: list(t.span_tokenize_sents([text, text]))),
    ]
    for name, call in calls:
        assert_same(
            outcome(call, ours), outcome(call, ref), f"{flavor} {name}({text!r}, {convert})"
        )


@needs_nltk
@examples
@given(tokens=TOKENS, convert=st.booleans())
@example(tokens=["word", ",", "'", "''"], convert=False)
@example(tokens=["-LRB-", "hello", "-RRB-", "''", "."], convert=True)
def test_detokenize(tokens: list[str], convert: bool) -> None:
    ours, ref = pn.TreebankWordDetokenizer(), REF["TreebankWordDetokenizer"]()
    want = outcome(ref.detokenize, tokens, convert)
    assert_same(
        outcome(ours.detokenize, tokens, convert), want, f"detokenize({tokens!r}, {convert})"
    )
    assert_same(outcome(ours.tokenize, tuple(tokens), convert), want, f"tokenize({tokens!r})")


@needs_nltk
@examples
@given(text=DOC, flavor=st.sampled_from(["treebank", "nltk"]))
def test_detokenize_round_trip(text: str, flavor: str) -> None:
    ours, ref = pn.TreebankWordDetokenizer(), REF["TreebankWordDetokenizer"]()
    for convert in (False, True):
        tokens = ref_word_tokenizer(flavor).tokenize(text, convert)
        assert_same(
            outcome(ours.detokenize, tokens, convert),
            outcome(ref.detokenize, tokens, convert),
            f"detokenize({flavor} tokens of {text!r}, {convert})",
        )


@needs_nltk
@examples
@given(sentence=DOC, cuts=st.lists(st.tuples(st.integers(0, 60), st.integers(0, 8))), extra=TOKENS)
def test_align_tokens(sentence: str, cuts: list[tuple[int, int]], extra: list[str]) -> None:
    tokens = [sentence[i : i + n] for i, n in cuts]
    for toks in (tokens, sorted(tokens, key=len), tokens + extra[:1], extra):
        assert_same(
            outcome(pn.align_tokens, toks, sentence),
            outcome(REF["align_tokens"], toks, sentence),
            f"align_tokens({toks!r}, {sentence!r})",
        )


# ---- batches ----------------------------------------------------------------------------------

KINDS = st.sampled_from(["list", "tuple", "column", "column-large", "column-chunked"])
THREADS = st.sampled_from([None, 1, 2, 3])


def as_kind(docs: list[str], kind: str) -> Any:
    if kind == "list":
        return list(docs)
    if kind == "tuple":
        return tuple(docs)
    return _testing.column(
        docs, large=kind == "column-large", chunks=3 if kind == "column-chunked" else 1
    )


@needs_nltk
@examples
@given(
    docs=st.lists(DOC, max_size=8),
    lang=LANG,
    kind=KINDS,
    threads=THREADS,
    realign=st.booleans(),
)
def test_punkt_batches(
    punkt_tab: Any, docs: list[str], lang: str, kind: str, threads: int | None, realign: bool
) -> None:
    x = as_kind(docs, kind)
    ref = ref_punkt(lang)
    tok = our_punkt(lang)
    assert_same(
        pn.sent_tokenize(x, lang, threads=threads),
        [REF["sent_tokenize"](d, lang) for d in docs],
        f"sent_tokenize({kind} {docs!r})",
    )
    assert_same(
        pn.word_tokenize(x, lang, threads=threads),
        [REF["word_tokenize"](d, lang) for d in docs],
        f"word_tokenize({kind} {docs!r})",
    )
    assert_same(
        pn.word_tokenize(x, lang, True, threads=threads),
        [REF["word_tokenize"](d, lang, True) for d in docs],
        f"word_tokenize({kind} {docs!r}, preserve_line=True)",
    )
    assert_same(
        tok.tokenize(x, realign, threads=threads),
        [ref.tokenize(d, realign) for d in docs],
        f"tokenize({kind} {docs!r}, {realign})",
    )
    assert_same(
        tok.span_tokenize(x, realign, threads=threads),
        [list(ref.span_tokenize(d, realign)) for d in docs],
        f"span_tokenize({kind} {docs!r}, {realign})",
    )


@needs_nltk
@examples
@given(
    docs=st.lists(DOC, max_size=8),
    flavor=st.sampled_from(["treebank", "nltk"]),
    kind=KINDS,
    threads=THREADS,
    convert=st.booleans(),
)
def test_word_tokenizer_batches(
    docs: list[str], flavor: str, kind: str, threads: int | None, convert: bool
) -> None:
    got = our_word_tokenizer(flavor).tokenize(as_kind(docs, kind), convert, threads=threads)
    ref = ref_word_tokenizer(flavor)
    assert_same(got, [ref.tokenize(d, convert) for d in docs], f"{flavor} {kind} {docs!r}")


# ---- the regular-expression tokenizers ---------------------------------------------------------

#: The patterns pyroid's RegexpTokenizer runs natively (with flags within re.U | re.M | re.S).
REGEXP_PATTERNS = [
    r"\w+",
    r"\w+|[^\w\s]+",
    r"\w+|\$[\d\.]+|\S+",
    r"[\w']+",
    r"\S+",
    r"\s+",
    r"[a-zA-Z]+",
    r"(?:(?<!\s)\s*)?\n\s*\n\s*",
]
NATIVE_FLAGS = st.sampled_from([0, re.U, re.M, re.S, re.U | re.M, re.U | re.S, re.M | re.S])
NATIVE_FLAGS = st.one_of(NATIVE_FLAGS, st.just(re.U | re.M | re.S))
#: Patterns or flags that pyroid hands to NLTK.
FALLBACK_REGEXPS: list[tuple[str, int]] = [
    (r"[A-Z]\w*", re.U | re.M | re.S),
    (r"\w+", re.I),
    (r"\w+", re.A),
    (r"\S+", re.X | re.U),
    (r"\b", re.U),
    (r"(?:)", re.U),
    (r"a*", re.U),
    (r"(\w)(\w*)", re.U),
    (r"[,\.\?!\"]\s*", re.U | re.M | re.S),
    (r"\w+ ", re.U | re.M | re.S),  # a native pattern with something after it
]
REGEXP_TEXT = st.one_of(
    DOC,
    st.lists(st.sampled_from(["a", "B", "$3.88", "it's", "\n", "\n\n", " \n \n", "\t", " "])).map(
        "".join
    ),
)


def ref_regexp() -> Any:
    return importlib.import_module("nltk.tokenize.regexp")


@needs_nltk
@examples
@given(
    text=REGEXP_TEXT,
    pattern=st.sampled_from(REGEXP_PATTERNS),
    gaps=st.booleans(),
    discard_empty=st.booleans(),
    flags=NATIVE_FLAGS,
)
@example(
    text="\n\n a\n\n\nb \n", pattern=REGEXP_PATTERNS[-1], gaps=True, discard_empty=False, flags=0
)
def test_regexp_tokenizer(
    text: str, pattern: str, gaps: bool, discard_empty: bool, flags: int
) -> None:
    ours = pn.RegexpTokenizer(pattern, gaps, discard_empty, flags)
    ref = ref_regexp().RegexpTokenizer(pattern, gaps, discard_empty, flags)
    assert cast(Any, ours)._native(), (pattern, flags)
    what = f"RegexpTokenizer({pattern!r}, {gaps}, {discard_empty}, {flags})"
    calls: list[tuple[str, Callable[[Any], Any]]] = [
        ("tokenize", lambda t: t.tokenize(text)),
        ("span_tokenize", lambda t: t.span_tokenize(text)),
        ("tokenize_sents", lambda t: t.tokenize_sents([text, "", text])),
        ("span_tokenize_sents", lambda t: list(t.span_tokenize_sents([text, text]))),
        ("repr", repr),
    ]
    for name, call in calls:
        assert_same(outcome(call, ours), outcome(call, ref), f"{what}.{name}({text!r})")
    assert_same(
        outcome(pn.regexp_tokenize, text, pattern, gaps, discard_empty, flags),
        outcome(ref_regexp().regexp_tokenize, text, pattern, gaps, discard_empty, flags),
        f"regexp_tokenize({text!r}, {pattern!r})",
    )


@needs_nltk
@examples
@given(text=REGEXP_TEXT)
def test_regexp_subclasses_and_functions(text: str) -> None:
    ref = ref_regexp()
    for name in ("WhitespaceTokenizer", "BlanklineTokenizer", "WordPunctTokenizer"):
        ours, theirs = getattr(pn, name)(), getattr(ref, name)()
        assert_same(outcome(ours.tokenize, text), outcome(theirs.tokenize, text), name)
        assert_same(outcome(ours.span_tokenize, text), outcome(theirs.span_tokenize, text), name)
        assert repr(ours) == repr(theirs)
    for name in ("wordpunct_tokenize", "blankline_tokenize"):
        assert_same(outcome(getattr(pn, name), text), outcome(getattr(ref, name), text), name)


@needs_nltk
@pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning")
@examples
@given(
    text=REGEXP_TEXT,
    regexp=st.sampled_from(FALLBACK_REGEXPS),
    gaps=st.booleans(),
    discard_empty=st.booleans(),
)
def test_regexp_fallbacks(
    text: str, regexp: tuple[str, int], gaps: bool, discard_empty: bool
) -> None:
    pattern, flags = regexp
    ours = pn.RegexpTokenizer(pattern, gaps, discard_empty, flags)
    ref = ref_regexp().RegexpTokenizer(pattern, gaps, discard_empty, flags)
    assert not cast(Any, ours)._native(), (pattern, flags)
    for call in (
        lambda t: t.tokenize(text),
        lambda t: t.span_tokenize(text),
        lambda t: t.tokenize_sents([text, text]),
    ):
        assert_same(outcome(call, ours), outcome(call, ref), f"{pattern!r} {flags} {text!r}")
    got = outcome(ours.tokenize, [text, "", text])
    assert_same(got, outcome(lambda: [ref.tokenize(d) for d in (text, "", text)]), "batch")


@needs_nltk
@examples
@given(
    docs=st.lists(REGEXP_TEXT, max_size=8),
    pattern=st.sampled_from(REGEXP_PATTERNS),
    gaps=st.booleans(),
    kind=KINDS,
    threads=THREADS,
)
def test_regexp_batches(
    docs: list[str], pattern: str, gaps: bool, kind: str, threads: int | None
) -> None:
    ours = pn.RegexpTokenizer(pattern, gaps, False)
    ref = ref_regexp().RegexpTokenizer(pattern, gaps, False)
    want = [ref.tokenize(d) for d in docs]
    assert_same(ours.tokenize(as_kind(docs, kind), threads=threads), want, f"{pattern!r} {kind}")
    assert_same(
        pn.wordpunct_tokenize(as_kind(docs, kind), threads=threads),
        [ref_regexp().wordpunct_tokenize(d) for d in docs],
        "wordpunct_tokenize",
    )


# ---- every code point --------------------------------------------------------------------------

# Run in child processes (one per core): (function, mode, k, step, sample) -> the code points
# k, k + step, ..., alone ("alone") or inside dense templates ("templates"), through NLTK and
# pyroid (one call each, and pyroid's batch call over all of them); prints the mismatches as JSON.
# With sample = n > 1, the templates take every code point below U+3400 and every n-th above.
SWEEP_SCRIPT = r"""
import json, sys
from nltk.tokenize import punkt, regexp, treebank, destructive, word_tokenize
import pyroid.nltk as pn

func, mode = sys.argv[1], sys.argv[2]
k, step, sample = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
chars = [
    chr(c) for c in range(k, 0x110000, step)
    if not 0xD800 <= c <= 0xDFFF and (mode == "alone" or c < 0x3400 or c % sample == 0)
]

def sent_templates(c):
    return [
        f"{c}Mr.{c} Smith{c} went to Washington{c}. {c}He{c} said{c} \"hi.{c}\" Then{c}... it "
        f"{c}ended{c}!{c} {c}",
        f"A. B{c}. 1{c}.{c} x.{c}y. e.g{c}. {c}i.e. the U.S.{c} army. ({c}Yes.{c}) Ok{c}.\n{c}\n"
        f"{c}.{c}? {c}... {c}. . . {c}",
    ]

def word_templates(c):
    return [
        f"I{c}'m {c}don't{c} can{c}not {c}'tis {c}O'{c}Neil{c}'s \"{c}quote{c}\" ({c}) [{c}] "
        f"--{c}-- {c}...{c}. {c}gonna{c} {c}'{c}hard{c}' {c}",
        f"\"{c}\" ''{c}'' ``{c}`` ‘{c}’ “{c}” «{c}» {c}: {c}, 1{c},000 ${c}3.88{c}% {c}&{c} "
        f"{c}?{c}!{c}'{c} {c}.{c} {c}*{c} a{c}..{c}",
    ]

def detok_templates(c):
    return [
        ["``", c, "''", c, "n't", c, ",", "(", c, ")", "-LRB-", c, "-RRB-", c, "."],
        [c, "'s", c, "...", c, "--", c, "$", c, "%", c, "can", "not", c, "'", c, "''", c],
    ]

def regexp_templates(c):
    return [
        word_templates(c)[1],
        f"a{c}\n{c}\n\nb{c} \n \n{c}c ${c}3.{c}88 it'{c}s {c}\t{c}\n\n\n{c}",
    ]

PATTERNS = [
    r"\w+", r"\w+|[^\w\s]+", r"\w+|\$[\d\.]+|\S+", r"[\w']+", r"\S+", r"\s+", r"[a-zA-Z]+",
    r"(?:(?<!\s)\s*)?\n\s*\n\s*",
]
CONFIGS = [(p, False, True) for p in PATTERNS]
CONFIGS += [(p, True, d) for p in PATTERNS for d in (True, False)]
ref_rx = [regexp.RegexpTokenizer(*c) for c in CONFIGS]
our_rx = [pn.RegexpTokenizer(*c) for c in CONFIGS]

def rx_ours(x):
    if isinstance(x, list):  # the batch call of each tokenizer, regrouped per input
        return [list(r) for r in zip(*(t.tokenize(x) for t in our_rx))] if x else []
    return [t.tokenize(x) for t in our_rx]

pt = punkt.PunktTokenizer("english")
tb, nw = treebank.TreebankWordTokenizer(), destructive.NLTKWordTokenizer()
dt = treebank.TreebankWordDetokenizer()
our_tb, our_nw = pn.TreebankWordTokenizer(), pn.NLTKWordTokenizer()
our_dt = pn.TreebankWordDetokenizer()
FUNCS = {
    "sent_tokenize": (pt.tokenize, pn.sent_tokenize, sent_templates),
    "word_tokenize": (
        word_tokenize, pn.word_tokenize, lambda c: [sent_templates(c)[0], word_templates(c)[0]]
    ),
    "treebank": (tb.tokenize, our_tb.tokenize, word_templates),
    "nltkword": (nw.tokenize, our_nw.tokenize, word_templates),
    "treebank_ptb": (
        lambda s: tb.tokenize(s, True), lambda s: our_tb.tokenize(s, True), word_templates
    ),
    "detokenize": (dt.detokenize, our_dt.detokenize, detok_templates),
    "regexp": (lambda x: [t.tokenize(x) for t in ref_rx], rx_ours, regexp_templates),
    "regexp_spans": (
        lambda x: [list(t.span_tokenize(x)) for t in ref_rx],
        lambda x: [list(t.span_tokenize(x)) for t in our_rx],
        regexp_templates,
    ),
    "detokenize_ptb": (
        lambda t: dt.detokenize(t, True), lambda t: our_dt.detokenize(t, True), detok_templates
    ),
}
ref, ours, templates = FUNCS[func]
if mode == "alone":
    inputs = [[c] if func.startswith("detok") else c for c in chars]
else:
    inputs = [x for c in chars for x in templates(c)]

def outcome(f, x):
    try:
        return ["ok", f(x)]
    except Exception as e:
        return ["raise", type(e).__name__, str(e)]

def ref_outcome(x):
    # NLTK runs each regex under a 5 s wall-clock limit, which a stalled CI runner can hit on a
    # one-character input; retry once. pyroid has no such limit (a documented difference), so
    # an input NLTK still times out on is not compared.
    w = outcome(ref, x)
    if w[:2] == ["raise", "TimeoutError"]:
        w = outcome(ref, x)
    return w

want = [ref_outcome(x) for x in inputs]
bad = []
for x, w in zip(inputs, want):
    if w[:2] == ["raise", "TimeoutError"]:
        continue
    g = outcome(ours, x)
    if g != w:
        bad.append([x, g, w])
if not func.startswith("detok") and func != "regexp_spans":
    ok = [x for x, w in zip(inputs, want) if w[0] == "ok"]
    got = ours(ok)
    for x, g, w in zip(ok, got, (w for w in want if w[0] == "ok")):
        if ["ok", g] != w:
            bad.append(["batch", x, g, w])
print(json.dumps({"n": len(inputs), "nbad": len(bad), "bad": bad[:10]}))
"""

SWEEP_FUNCS = ["sent_tokenize", "word_tokenize", "treebank", "nltkword", "treebank_ptb"]
SWEEP_FUNCS += ["detokenize", "detokenize_ptb", "regexp", "regexp_spans"]
#: PYROID_NLTK_SWEEP=full puts every code point in the templates too (NLTK then takes about an
#: hour of CPU), PYROID_NLTK_SWEEP=<n> every n-th above U+3400; by default the templates get every
#: code point below U+3400 and every 31st above (the Rust replays cover every code point in the
#: templates; this keeps a CI leg to a few minutes). Each code point alone is always checked.
_SWEEP_ENV = os.environ.get("PYROID_NLTK_SWEEP", "")
SWEEP_SAMPLE = 1 if _SWEEP_ENV == "full" else int(_SWEEP_ENV) if _SWEEP_ENV.isdigit() else 31


def _sweep(func: str, mode: str) -> None:
    procs = max(1, min(os.cpu_count() or 2, 16))
    env = {k: v for k, v in os.environ.items() if k not in ("PYROID_NUM_THREADS", "PYTHON_GIL")}
    script = textwrap.dedent(SWEEP_SCRIPT)
    args = [func, mode, "", str(procs), str(SWEEP_SAMPLE)]
    children = [
        subprocess.Popen(
            [sys.executable, "-I", "-c", script, *args[:2], str(k), *args[3:]],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for k in range(procs)
    ]
    results = []
    for child in children:
        try:
            out, err = child.communicate(timeout=1500)
        except subprocess.TimeoutExpired:
            for c in children:
                c.kill()
            pytest.fail(f"code point sweep {func} {mode}: a child did not finish in 1500 s")
        assert child.returncode == 0, err
        results.append(json.loads(out))
    n = sum(r["n"] for r in results)
    nbad = sum(r["nbad"] for r in results)
    examples_ = [b for r in results for b in r["bad"]][:10]
    if mode == "alone":
        assert n == 0x110000 - 0x800  # every code point but the surrogates
    assert nbad == 0, f"{func} {mode}: {nbad} of {n} inputs differ, e.g.\n" + "\n".join(
        repr(b) for b in examples_
    )


@needs_nltk
@pytest.mark.thread_unsafe(reason="subprocesses, one per core")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", ["alone", "templates"])
@pytest.mark.parametrize("func", SWEEP_FUNCS)
def test_every_code_point(punkt_tab: Any, func: str, mode: str) -> None:
    if (func, mode) in _SWEPT:  # pytest-run-parallel's --iterations: one sweep per process
        return
    _sweep(func, mode)
    _SWEPT.add((func, mode))


_SWEPT: set[tuple[str, str]] = set()


# ---- lone surrogates ---------------------------------------------------------------------------

SURROGATE_TEXT = st.tuples(TEXT, st.characters(categories=["Cs"]), TEXT).map("".join)


@examples
@given(text=SURROGATE_TEXT, lang=st.sampled_from(["english", "german"]))
def test_lone_surrogates_raise(punkt_tab: Any, text: str, lang: str) -> None:
    tok = our_punkt(lang)
    calls: list[Callable[[], Any]] = [
        lambda: pn.sent_tokenize(text, lang),
        lambda: pn.word_tokenize(text, lang),
        lambda: pn.word_tokenize(text, lang, True),
        lambda: tok.tokenize(text, False),
        lambda: list(tok.span_tokenize(text)),
        lambda: tok.text_contains_sentbreak(text),
        lambda: list(tok.debug_decisions(text)),
        lambda: list(tok.sentences_from_text_legacy(text)),
        lambda: pn.TreebankWordTokenizer().tokenize(text),
        lambda: pn.NLTKWordTokenizer().tokenize(text, True),
        lambda: list(pn.NLTKWordTokenizer().span_tokenize(text)),
        lambda: pn.TreebankWordDetokenizer().detokenize(["a", text]),
        lambda: pn.align_tokens(["a"], text + "a"),
        lambda: pn.sent_tokenize(["ok", text], lang),
        lambda: pn.TreebankWordTokenizer().tokenize(["ok", text]),
    ]
    for call in calls:
        with pytest.raises(UnicodeEncodeError):
            call()


# ---- NLTK's own examples (doctests and unit tests), checked without NLTK ----------------------

# nltk/test/tokenize.doctest: "Regression Tests: NLTKWordTokenizer".
DOCTEST_WORD_TOKENIZE: list[tuple[str, list[str]]] = [
    (
        "On a $50,000 mortgage of 30 years at 8 percent, the monthly payment would be $366.88.",
        [
            *("On", "a", "$", "50,000", "mortgage", "of", "30", "years", "at", "8", "percent", ","),
            *("the", "monthly", "payment", "would", "be", "$", "366.88", "."),
        ],
    ),
    (
        '"We beat some pretty good teams to get here," Slocum said.',
        [
            *("``", "We", "beat", "some", "pretty", "good", "teams", "to", "get", "here", ","),
            *("''", "Slocum", "said", "."),
        ],
    ),
    (
        'Well, we couldn\'t have this predictable, cliche-ridden, "Touched by an Angel" (a show '
        "creator John Masius worked on) wanna-be if she didn't.",
        [
            *("Well", ",", "we", "could", "n't", "have", "this", "predictable", ","),
            *("cliche-ridden", ",", "``", "Touched", "by", "an", "Angel", "''", "(", "a"),
            *("show", "creator", "John", "Masius", "worked", "on", ")", "wanna-be", "if"),
            *("she", "did", "n't", "."),
        ],
    ),
    (
        "I cannot cannot work under these conditions!",
        ["I", "can", "not", "can", "not", "work", "under", "these", "conditions", "!"],
    ),
    (
        "The company spent $30,000,000 last year.",
        ["The", "company", "spent", "$", "30,000,000", "last", "year", "."],
    ),
    (
        "The company spent 40.75% of its income last year.",
        ["The", "company", "spent", "40.75", "%", "of", "its", "income", "last", "year", "."],
    ),
    ("He arrived at 3:00 pm.", ["He", "arrived", "at", "3:00", "pm", "."]),
    (
        "I bought these items: books, pencils, and pens.",
        ["I", "bought", "these", "items", ":", "books", ",", "pencils", ",", "and", "pens", "."],
    ),
    (
        "Though there were 150, 100 of them were old.",
        ["Though", "there", "were", "150", ",", "100", "of", "them", "were", "old", "."],
    ),
    (
        "There were 300,000, but that wasn't enough.",
        ["There", "were", "300,000", ",", "but", "that", "was", "n't", "enough", "."],
    ),
    ("It's more'n enough.", ["It", "'s", "more", "'n", "enough", "."]),
    ("\xabNow that I can do.\xbb", ["\xab", "Now", "that", "I", "can", "do", ".", "\xbb"]),
    (
        "The unicode 201C and 201D “LEFT(RIGHT) DOUBLE QUOTATION MARK” is also "
        "OPEN_PUNCT and CLOSE_PUNCT.",
        [
            *("The", "unicode", "201C", "and", "201D", "“", "LEFT", "(", "RIGHT", ")", "DOUBLE"),
            *("QUOTATION", "MARK", "”", "is", "also", "OPEN_PUNCT", "and", "CLOSE_PUNCT", "."),
        ],
    ),
    (
        "I called Dr. Jones. I called Dr. Jones.",
        ["I", "called", "Dr.", "Jones", ".", "I", "called", "Dr.", "Jones", "."],
    ),
    (
        "Ich muss unbedingt daran denken, Mehl, usw. fur einen Kuchen einzukaufen. Ich muss.",
        [
            *("Ich", "muss", "unbedingt", "daran", "denken", ",", "Mehl", ",", "usw", ".", "fur"),
            *("einen", "Kuchen", "einzukaufen", ".", "Ich", "muss", "."),
        ],
    ),
    # nltk/test/unit/test_tokenize.py
    (
        "This is a, *weird sentence with *asterisks in it.",
        [
            "This",
            "is",
            "a",
            ",",
            "*",
            "weird",
            "sentence",
            "with",
            "*",
            "asterisks",
            "in",
            "it",
            ".",
        ],
    ),
    (
        "The food—which was delicious—reminded me of home.",
        ["The", "food", "—", "which", "was", "delicious", "—", "reminded", "me", "of", "home", "."],
    ),
    ("The score was 3–1.", ["The", "score", "was", "3", "–", "1", "."]),
    (
        "He said ― wait for it ― a new thing.",
        ["He", "said", "―", "wait", "for", "it", "―", "a", "new", "thing", "."],
    ),
    ("The number is 12‒34.", ["The", "number", "is", "12", "‒", "34", "."]),
    ("state-of-the-art", ["state-of-the-art"]),
    (
        "Why did dotdot.. not get tokenized but dotdotdot... did? How about manydots.....",
        [
            *("Why", "did", "dotdot", "..", "not", "get", "tokenized", "but", "dotdotdot", "..."),
            *("did", "?", "How", "about", "manydots", "....."),
        ],
    ),
    (
        "The 'v', I've been fooled but I'll seek revenge.",
        [
            *("The", "'", "v", "'", ",", "I", "'ve", "been", "fooled", "but", "I", "'ll", "seek"),
            *("revenge", "."),
        ],
    ),
    ("'v' 're'", ["'", "v", "'", "'re", "'"]),
    ("'Hard' to tell", ["'", "Hard", "'", "to", "tell"]),
    ("He said 'hello' to me", ["He", "said", "'", "hello", "'", "to", "me"]),
    ("It's o'clock already", ["It", "'s", "o'clock", "already"]),
    ("O'Connor went home", ["O'Connor", "went", "home"]),
    # nltk/tokenize/__init__.py
    (
        "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks.",
        [
            *("Good", "muffins", "cost", "$", "3.88", "in", "New", "York", ".", "Please", "buy"),
            *("me", "two", "of", "them", ".", "Thanks", "."),
        ],
    ),
]


@pytest.mark.parametrize(("text", "want"), DOCTEST_WORD_TOKENIZE)
def test_nltk_examples_word_tokenize(punkt_tab: Any, text: str, want: list[str]) -> None:
    assert pn.word_tokenize(text) == want
    assert pn.word_tokenize([text, text]) == [want, want]


def test_nltk_examples_word_tokenize_german(punkt_tab: Any) -> None:
    s = "Ich muss unbedingt daran denken, Mehl, usw. fur einen Kuchen einzukaufen. Ich muss."
    assert pn.word_tokenize(s, "german") == [
        *("Ich", "muss", "unbedingt", "daran", "denken", ",", "Mehl", ",", "usw.", "fur"),
        *("einen", "Kuchen", "einzukaufen", ".", "Ich", "muss", "."),
    ]


MUFFINS = "Good muffins cost $3.88\nin New (York).  Please (buy) me\ntwo of them.\n(Thanks)."
MUFFIN_SPANS = [
    *((0, 4), (5, 12), (13, 17), (18, 19), (19, 23), (24, 26), (27, 30), (31, 32), (32, 36)),
    *((36, 37), (37, 38), (40, 46), (47, 48), (48, 51), (51, 52), (53, 55), (56, 59), (60, 62)),
    *((63, 68), (69, 70), (70, 76), (76, 77), (77, 78)),
]
MUFFIN_TOKENS = [
    *("Good", "muffins", "cost", "$", "3.88", "in", "New", "(", "York", ")", ".", "Please"),
    *("(", "buy", ")", "me", "two", "of", "them.", "(", "Thanks", ")", "."),
]
QUOTED = "I said, \"I'd like to buy some ''good muffins\" which cost $3.88\n each in New (York).\""
QUOTED_SPANS = [
    *((0, 1), (2, 6), (6, 7), (8, 9), (9, 10), (10, 12), (13, 17), (18, 20), (21, 24), (25, 29)),
    *((30, 32), (32, 36), (37, 44), (44, 45), (46, 51), (52, 56), (57, 58), (58, 62), (64, 68)),
    *((69, 71), (72, 75), (76, 77), (77, 81), (81, 82), (82, 83), (83, 84)),
]
QUOTED_TOKENS = [
    *("I", "said", ",", '"', "I", "'d", "like", "to", "buy", "some", "''", "good", "muffins"),
    *('"', "which", "cost", "$", "3.88", "each", "in", "New", "(", "York", ")", ".", '"'),
]
DUP = (
    'The DUP is similar to the "religious right" in the United States and takes a hardline '
    "stance on social issues"
)
DUP_SPANS = [
    *((0, 3), (4, 7), (8, 10), (11, 18), (19, 21), (22, 25), (26, 27), (27, 36), (37, 42)),
    *((42, 43), (44, 46), (47, 50), (51, 57), (58, 64), (65, 68), (69, 74), (75, 76), (77, 85)),
    *((86, 92), (93, 95), (96, 102), (103, 109)),
]
DUP2 = (
    "The DUP is similar to the \"religious right\" in the United States and takes a ``hardline'' "
    "stance on social issues"
)
DUP2_SPANS = [
    *((0, 3), (4, 7), (8, 10), (11, 18), (19, 21), (22, 25), (26, 27), (27, 36), (37, 42)),
    *((42, 43), (44, 46), (47, 50), (51, 57), (58, 64), (65, 68), (69, 74), (75, 76), (77, 79)),
    *((79, 87), (87, 89), (90, 96), (97, 99), (100, 106), (107, 113)),
]


def test_nltk_examples_spans() -> None:
    for tok in (pn.NLTKWordTokenizer(), pn.TreebankWordTokenizer()):
        assert list(tok.span_tokenize(MUFFINS)) == MUFFIN_SPANS
        assert [MUFFINS[s:e] for s, e in tok.span_tokenize(MUFFINS)] == MUFFIN_TOKENS
    nw = pn.NLTKWordTokenizer()
    assert list(nw.span_tokenize(QUOTED)) == QUOTED_SPANS
    assert [QUOTED[s:e] for s, e in nw.span_tokenize(QUOTED)] == QUOTED_TOKENS
    tb = pn.TreebankWordTokenizer()
    assert list(tb.span_tokenize(DUP)) == DUP_SPANS
    assert list(tb.span_tokenize(DUP2)) == DUP2_SPANS


TREEBANK_DOCSTRING = (
    "Good muffins cost $3.88 (roughly 3,36 euros)\nin New York.  Please buy me\ntwo of them.\n"
    "Thanks."
)
TREEBANK_DOCSTRING_TOKENS = [
    *("Good", "muffins", "cost", "$", "3.88", "(", "roughly", "3,36", "euros", ")", "in", "New"),
    *("York.", "Please", "buy", "me", "two", "of", "them.", "Thanks", "."),
]


def test_nltk_examples_treebank_docstrings() -> None:
    tb = pn.TreebankWordTokenizer()
    s = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks."
    assert tb.tokenize(s) == [
        *("Good", "muffins", "cost", "$", "3.88", "in", "New", "York.", "Please", "buy", "me"),
        *("two", "of", "them.", "Thanks", "."),
    ]
    assert tb.tokenize("They'll save and invest more.") == [
        *("They", "'ll", "save", "and", "invest", "more", "."),
    ]
    assert tb.tokenize("hi, my name can't hello,") == [
        *("hi", ",", "my", "name", "ca", "n't", "hello", ","),
    ]
    for tok in (tb, pn.NLTKWordTokenizer()):
        assert tok.tokenize(TREEBANK_DOCSTRING) == TREEBANK_DOCSTRING_TOKENS
        ptb = [
            "-LRB-" if t == "(" else "-RRB-" if t == ")" else t for t in TREEBANK_DOCSTRING_TOKENS
        ]
        assert tok.tokenize(TREEBANK_DOCSTRING, convert_parentheses=True) == ptb
    # align_tokens' docstring
    s = (
        "The plane, bound for St Petersburg, crashed in Egypt's Sinai desert just 23 minutes "
        "after take-off from Sharm el-Sheikh on Saturday."
    )
    tokens = tb.tokenize(s)
    expected = [
        *((0, 3), (4, 9), (9, 10), (11, 16), (17, 20), (21, 23), (24, 34), (34, 35), (36, 43)),
        *((44, 46), (47, 52), (52, 54), (55, 60), (61, 67), (68, 72), (73, 75), (76, 83)),
        *((84, 89), (90, 98), (99, 103), (104, 109), (110, 119), (120, 122), (123, 131)),
        (131, 132),
    ]
    assert pn.align_tokens(tokens, s) == expected
    assert tokens == [s[a:b] for a, b in expected]


DETOK_ROUND_TRIPS: list[tuple[str, str]] = [
    (
        "On a $50,000 mortgage of 30 years at 8 percent, the monthly payment would be $366.88.",
        "On a $50,000 mortgage of 30 years at 8 percent, the monthly payment would be $366.88.",
    ),
    (
        '"We beat some pretty good teams to get here," Slocum said.',
        '"We beat some pretty good teams to get here," Slocum said.',
    ),
    (
        'Well, we couldn\'t have this predictable, cliche-ridden, "Touched by an Angel" (a show '
        "creator John Masius worked on) wanna-be if she didn't.",
        'Well, we couldn\'t have this predictable, cliche-ridden, "Touched by an Angel" (a show '
        "creator John Masius worked on) wanna-be if she didn't.",
    ),
    (
        "I cannot cannot work under these conditions!",
        "I cannot cannot work under these conditions!",
    ),
    ("The company spent $30,000,000 last year.", "The company spent $30,000,000 last year."),
    (
        "The company spent 40.75% of its income last year.",
        "The company spent 40.75% of its income last year.",
    ),
    ("He arrived at 3:00 pm.", "He arrived at 3:00 pm."),
    (
        "I bought these items: books, pencils, and pens.",
        "I bought these items: books, pencils, and pens.",
    ),
    (
        "Though there were 150, 100 of them were old.",
        "Though there were 150, 100 of them were old.",
    ),
    ("There were 300,000, but that wasn't enough.", "There were 300,000, but that wasn't enough."),
    ('How "are" you?', 'How "are" you?'),
    ("Hello (world)", "Hello (world)"),
    (
        '<A sentence> with (many) [kinds] of {parentheses}. "Sometimes it\'s inside (quotes)". '
        '("Sometimes the otherway around").',
        '<A sentence> with (many) [kinds] of {parentheses}. "Sometimes it\'s inside (quotes)". '
        '("Sometimes the otherway around").',
    ),
    ("Sentence ending with (parentheses)", "Sentence ending with (parentheses)"),
    ("(Sentence) starting with parentheses.", "(Sentence) starting with parentheses."),
    ("I've", "I've"),
    ("Don't", "Don't"),
    ("I'd", "I'd"),
]


@pytest.mark.parametrize(("text", "want"), DETOK_ROUND_TRIPS)
def test_nltk_examples_detokenize_word_tokenize(punkt_tab: Any, text: str, want: str) -> None:
    assert pn.TreebankWordDetokenizer().detokenize(pn.word_tokenize(text)) == want


# nltk/test/unit/test_tokenize.py: TestTreebankWordDetokenizer (the exact assertions).
DETOKENIZE: list[tuple[list[str], bool, str]] = [
    (["Hello", ",", "world", "."], False, "Hello, world."),
    (["I", "'m", "sure", "."], False, "I'm sure."),
    (["You", "'ll", "see", "."], False, "You'll see."),
    (["I", "do", "n't", "know", "."], False, "I don't know."),
    (["``", "Hello", "''"], False, '"Hello"'),
    (["He", "said", "``", "hi", "''", "."], False, 'He said "hi".'),
    (["``", "Yes", ",", "''", "he", "said", "."], False, '"Yes," he said.'),
    (["The", "dog", "'s", "bone", "."], False, "The dog's bone."),
    (["-LRB-", "hello", "-RRB-"], True, "(hello)"),
    (["foo", "--", "bar"], False, "foo--bar"),
    (["word", ",", "'", "''"], False, "word,'\""),
    (["a", ";", "b"], False, "a; b"),
    (["a", ":", "b"], False, "a: b"),
    (["we", "'ve", "seen", "."], False, "we've seen."),
    (["who", "'d", "know", "."], False, "who'd know."),
    (["word", ".", "''"], False, 'word."'),
    (["word", ",", "''"], False, 'word,"'),
    (["word", "!", "''"], False, 'word!"'),
    (["word", "?", "''"], False, 'word?"'),
    (["``", "A", "B", "C", "''"], False, '"A B C"'),
    (["``", "Hi", ",", "''", "said", "Jo", "."], False, '"Hi," said Jo.'),
    (
        ["Lorem", "ipsum", "dolor", "sit", "amet", ".", "consectetur", "adipiscing", "elit", "."],
        False,
        "Lorem ipsum dolor sit amet. consectetur adipiscing elit.",
    ),
    (
        ["I", "called", "Dr.", "Jones", ".", "I", "called", "Dr.", "Jones", "."],
        False,
        "I called Dr. Jones. I called Dr. Jones.",
    ),
    (["wait", "...", "what"], False, "wait...what"),
    # nltk/tokenize/treebank.py's TreebankWordDetokenizer docstring
    (
        ["hello", ",", "i", "ca", "n't", "feel", "my", "feet", "!", "Help", "!", "!"],
        False,
        "hello, i can't feel my feet! Help!!",
    ),
    (
        [
            *("hello", ",", "i", "can't", "feel", ";", "my", "feet", "!", "Help", "!", "!", "He"),
            *("said", ":", "Help", ",", "help", "?", "!"),
        ],
        False,
        "hello, i can't feel; my feet! Help!! He said: Help, help?!",
    ),
]


@pytest.mark.parametrize(("tokens", "convert", "want"), DETOKENIZE)
def test_nltk_examples_detokenize(tokens: list[str], convert: bool, want: str) -> None:
    d = pn.TreebankWordDetokenizer()
    assert d.detokenize(tokens, convert_parentheses=convert) == want
    assert d.tokenize(tokens, convert) == want


def test_nltk_examples_detokenize_loose() -> None:
    d = pn.TreebankWordDetokenizer()
    result = d.detokenize(["`", "Hello", "'"])
    assert "`" in result or "'" in result
    assert '"yes"' in d.detokenize(["He", "said", "``", "yes", "''", "."])
    issue_3260 = ["``", "Shippers", "are", "saying", "`", "the", "party", "'s", "over", ","]
    issue_3260 += ["'", "''", "said", "Mr.", "LaLonde", "."]
    assert ",'" in d.detokenize(issue_3260)
    assert " ," not in d.detokenize(["a", ",", "b"])
    assert " ." not in d.detokenize(["a", "."])
    assert '"' in d.detokenize(["``", "Hello", ",", "''", "he", "said", "."])
    tb = pn.TreebankWordTokenizer()
    assert d.detokenize(tb.tokenize("Hello, world.")) == "Hello, world."
    s = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks."
    assert d.detokenize(tb.tokenize(s)) == (
        "Good muffins cost $3.88 in New York. Please buy me two of them. Thanks."
    )
    ptb = [
        *("Good", "muffins", "cost", "$", "3.88", "in", "New", "-LRB-", "York", "-RRB-", "."),
        *("Please", "-LRB-", "buy", "-RRB-", "me", "two", "of", "them.", "-LRB-", "Thanks"),
        *("-RRB-", "."),
    ]
    assert tb.tokenize(MUFFINS, convert_parentheses=True) == ptb
    assert d.detokenize(ptb, convert_parentheses=True) == (
        "Good muffins cost $3.88 in New (York). Please (buy) me two of them. (Thanks)."
    )


# nltk/test/unit/test_tokenize.py: test_sent_tokenize, and the issue 2333 regression test.
SENT_TOKENIZE: list[tuple[str, list[str]]] = [
    ("this is a test. . new sentence.", ["this is a test.", ".", "new sentence."]),
    ("This. . . That", ["This.", ".", ".", "That"]),
    ("This..... That", ["This..... That"]),
    ("This... That", ["This... That"]),
    ("This.. . That", ["This.. .", "That"]),
    ("This. .. That", ["This.", ".. That"]),
    ("This. ,. That", ["This.", ",.", "That"]),
    ("This!!! That", ["This!!!", "That"]),
    ("This! That", ["This!", "That"]),
    (
        "1. This is R .\n2. This is A .\n3. That's all",
        ["1.", "This is R .", "2.", "This is A .", "3.", "That's all"],
    ),
    (
        "1. This is R .\t2. This is A .\t3. That's all",
        ["1.", "This is R .", "2.", "This is A .", "3.", "That's all"],
    ),
    ("Hello.\tThere", ["Hello.", "There"]),
    ("“First sentence.” Next one.", ["“First sentence.”", "Next one."]),
    ("He said “this is great.” Then left.", ["He said “this is great.”", "Then left."]),
    (
        "She asked, “Are you coming?” He nodded.",
        ["She asked, “Are you coming?”", "He nodded."],
    ),
    ("Il a dit «bonjour.» Puis il est parti.", ["Il a dit «bonjour.»", "Puis il est parti."]),
    ('He said "hi." Then left.', ['He said "hi."', "Then left."]),
    ("This is one sentence. This is another.", ["This is one sentence.", "This is another."]),
    # nltk/tokenize/__init__.py
    (
        "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks.",
        ["Good muffins cost $3.88\nin New York.", "Please buy me\ntwo of them.", "Thanks."],
    ),
]


@pytest.mark.parametrize(("text", "want"), SENT_TOKENIZE)
def test_nltk_examples_sent_tokenize(punkt_tab: Any, text: str, want: list[str]) -> None:
    assert pn.sent_tokenize(text) == want
    assert pn.sent_tokenize((text,)) == [want]


def test_nltk_examples_word_tokenize_per_sentence(punkt_tab: Any) -> None:
    s = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks."
    assert [pn.word_tokenize(t) for t in pn.sent_tokenize(s)] == [
        ["Good", "muffins", "cost", "$", "3.88", "in", "New", "York", "."],
        ["Please", "buy", "me", "two", "of", "them", "."],
        ["Thanks", "."],
    ]


PUNKT_DOCSTRING = (
    "Punkt knows that the periods in Mr. Smith and Johann S. Bach\ndo not mark sentence "
    "boundaries.  And sometimes sentences\ncan start with non-capitalized words.  i is a good "
    "variable\nname."
)
PARENS_DOCSTRING = (
    '(How does it deal with this parenthesis?)  "It should be part of the\nprevious sentence." '
    "\"(And the same with this one.)\" ('And this one!')\n\"('(And (this)) '?)\" [(and this. )]"
)


def test_nltk_examples_punkt_docstring(punkt_tab: Any) -> None:
    sent_detector = pn.PunktTokenizer()
    assert sent_detector.tokenize(PUNKT_DOCSTRING) == [
        "Punkt knows that the periods in Mr. Smith and Johann S. Bach\ndo not mark sentence "
        "boundaries.",
        "And sometimes sentences\ncan start with non-capitalized words.",
        "i is a good variable\nname.",
    ]
    assert sent_detector.tokenize(PARENS_DOCSTRING) == [
        "(How does it deal with this parenthesis?)",
        '"It should be part of the\nprevious sentence."',
        '"(And the same with this one.)"',
        "('And this one!')",
        "\"('(And (this)) '?)\"",
        "[(and this. )]",
    ]
    assert sent_detector.tokenize(PARENS_DOCSTRING, realign_boundaries=False) == [
        "(How does it deal with this parenthesis?",
        ')  "It should be part of the\nprevious sentence.',
        '" "(And the same with this one.',
        ")\" ('And this one!",
        "')\n\"('(And (this)) '?",
        ')" [(and this.',
        ")]",
    ]


def test_nltk_examples_punkt_regressions() -> None:
    pst = pn.PunktSentenceTokenizer()
    assert pst.tokenize("See Section 3).  Or Section 2).  ") == [
        "See Section 3).",
        "Or Section 2).",
    ]
    assert pst.tokenize("See Section 3.)  Or Section 2.)  ") == [
        "See Section 3.)",
        "Or Section 2.)",
    ]
    assert pst.tokenize("See Section 3.)  Or Section 2.)  ", realign_boundaries=False) == [
        "See Section 3.",
        ")  Or Section 2.",
        ")",
    ]
    assert pst._params is not pn.PunktSentenceTokenizer()._params
    pst = pn.PunktSentenceTokenizer(lang_vars=None)
    assert type(pst._lang_vars) is pn.PunktLanguageVars
    assert pst.tokenize(". This input starts with a dot. This used to cause issues.") == [
        ".",
        "This input starts with a dot.",
        "This used to cause issues.",
    ]


BENGALI = (
    "উপরাষ্ট্রপতি শ্রী এম ভেঙ্কাইয়া নাইডু সোমবার আই আই টি দিল্লির হীরক জয়ন্তী উদযাপনের উদ্বোধন করেছেন। "
    "অনলাইনের মাধ্যমে এই অনুষ্ঠানে কেন্দ্রীয় মানব সম্পদ উন্নয়নমন্ত্রী শ্রী রমেশ পোখরিয়াল ‘নিশাঙ্ক’  "
    "উপস্থিত ছিলেন। এই উপলক্ষ্যে উপরাষ্ট্রপতি হীরকজয়ন্তীর লোগো এবং ২০৩০-এর জন্য প্রতিষ্ঠানের লক্ষ্য ও "
    "পরিকল্পনার নথি প্রকাশ করেছেন।"
)


def test_nltk_examples_custom_lang_vars() -> None:
    class BengaliLanguageVars(pn.PunktLanguageVars):
        sent_end_chars = (".", "?", "!", "।")

    obj = pn.PunktSentenceTokenizer(lang_vars=BengaliLanguageVars())
    first, rest = BENGALI.split("। ", 1)
    second, third = rest.split("। ", 1)
    assert obj.tokenize(BENGALI) == [first + "।", second + "।", third]
    assert native(obj) is not None  # still native
    assert pn.PunktSentenceTokenizer().tokenize(BENGALI) == [BENGALI]


@pytest.mark.parametrize(
    ("text", "n_sents", "n_splits", "end"),
    [
        ("Subject: Some subject. Attachments: Some attachments", 2, 1, None),
        ("Subject: Some subject! Attachments: Some attachments", 2, 1, None),
        ("This is just a normal sentence, just like any other.", 1, 0, None),
        ("Subject: Some subject^ Attachments: Some attachments", 2, 1, "^"),
    ],
)
def test_nltk_examples_debug_decisions(
    text: str, n_sents: int, n_splits: int, end: str | None
) -> None:
    tokenizer = pn.PunktSentenceTokenizer()
    if end is not None:
        chars = (".", "?", "!", end)

        class ExtLangVars(pn.PunktLanguageVars):
            sent_end_chars = chars

        tokenizer._lang_vars = ExtLangVars()
    assert len(tokenizer.tokenize(text)) == n_sents
    assert len(list(tokenizer.debug_decisions(text))) == n_splits


ALIGN: list[tuple[list[str], str, list[tuple[int, int]] | str]] = [
    ([""], "", [(0, 0)]),
    ([""], " ", [(0, 0)]),
    ([], "", []),
    ([], " ", []),
    (["a"], "a", [(0, 1)]),
    (["abc", "def"], "abcdef", [(0, 3), (3, 6)]),
    (["abc", "def"], "abc def", [(0, 3), (4, 7)]),
    (["ab", "cd"], "ab cd ef", [(0, 2), (3, 5)]),
    (["ab", "cd", "ef"], "ab cd ef", [(0, 2), (3, 5), (6, 8)]),
    (["ab", "cd", "efg"], "ab cd ef", 'substring "efg" not found in "ab cd ef"'),
    (["ab", "cd", "ef", "gh"], "ab cd ef", 'substring "gh" not found in "ab cd ef"'),
]


@pytest.mark.parametrize(("tokens", "sentence", "want"), ALIGN)
def test_nltk_examples_align_tokens(
    tokens: list[str], sentence: str, want: list[tuple[int, int]] | str
) -> None:
    if isinstance(want, str):
        with pytest.raises(ValueError, match=f"^{want}$"):
            pn.align_tokens(tokens, sentence)
    else:
        assert pn.align_tokens(tokens, sentence) == want


@needs_nltk
@pytest.mark.parametrize("text", TRICKY_TEXTS)
def test_tricky_texts_everywhere(punkt_tab: Any, text: str) -> None:
    """The hand-picked texts through every function, in every language."""
    for lang in LANGUAGES:
        assert_same(
            outcome(pn.sent_tokenize, text, lang), outcome(REF["sent_tokenize"], text, lang), lang
        )
        assert_same(
            outcome(pn.word_tokenize, text, lang), outcome(REF["word_tokenize"], text, lang), lang
        )
        ours, ref = our_punkt(lang), ref_punkt(lang)
        for realign in (True, False):
            assert_same(
                outcome(ours.span_tokenize, text, realign),
                outcome(ref.span_tokenize, text, realign),
                lang,
            )
        assert_same(
            outcome(ours.debug_decisions, text),
            outcome(ref.debug_decisions, text),
            lang,
        )
