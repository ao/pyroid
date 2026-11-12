# Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace; modified.
"""pyroid.filters against datatrove 0.10.1 itself.

- Live differential tests (Hypothesis, need datatrove): documents built to reach every
  check of every filter, for each filter's default config and several non-default ones,
  for ``apply()`` chains and for every container kind. The reference for one document is
  what the parity harness pins::

      doc = Document(text=t, id="0")
      keep, reason = get_filter_result(Filter(**cfg).filter(doc))  # text = doc.text

  or the exception's type name when datatrove raises; pyroid must then raise the same type.
- Pinned decisions (need pyarrow only): ``parity/filters-ref.parquet`` over the FineWeb dev
  slice and the adversarial set, all five harness cases. ``PYROID_PARITY_DATA`` points at
  the data directory (default ``/data``); the tests skip when the files are missing.

``PYROID_FILTERS_EXAMPLES=n`` sets the Hypothesis examples per test (default: the profile's).
"""

# Look-alike characters and other scripts' punctuation on purpose.
# ruff: noqa: RUF001

from __future__ import annotations

import importlib.metadata
import inspect
import os
import string
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from hypothesis import HealthCheck, event, given, note, settings
from hypothesis import strategies as st

import pyroid.filters as pf
from pyroid._pyroid import _testing

pytestmark = pytest.mark.thread_unsafe(reason="datatrove and spaCy are not thread-safe")

_HEALTH = [HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much]
_EXAMPLES = os.environ.get("PYROID_FILTERS_EXAMPLES")
examples = (
    settings(suppress_health_check=_HEALTH, max_examples=int(_EXAMPLES))
    if _EXAMPLES
    else settings(suppress_health_check=_HEALTH)
)

#: ("ok", keep, reason, text) or ("raise", exception type name).
Outcome = tuple[Any, ...]


# ---- the reference -------------------------------------------------------------


class Datatrove:
    """datatrove's filters, one instance per config (as the harness builds them)."""

    CLASSES: ClassVar[dict[type, str]] = {
        pf.GopherQuality: "GopherQualityFilter",
        pf.GopherRepetition: "GopherRepetitionFilter",
        pf.C4Quality: "C4QualityFilter",
        pf.FineWebQuality: "FineWebQualityFilter",
    }

    def __init__(self) -> None:
        self.filters: Any = pytest.importorskip("datatrove.pipeline.filters")
        self.base: Any = pytest.importorskip("datatrove.pipeline.filters.base_filter")
        self.data: Any = pytest.importorskip("datatrove.data")
        self.text: Any = pytest.importorskip("datatrove.utils.text")
        version = importlib.metadata.version("datatrove")
        if version != pf.DATATROVE_VERSION:
            pytest.skip(f"needs datatrove {pf.DATATROVE_VERSION}, found {version}")
        self._cache: dict[str, Any] = {}

    def filter(self, cfg: Any) -> Any:
        key = repr(cfg)
        if key not in self._cache:
            cls = getattr(self.filters, self.CLASSES[type(cfg)])
            self._cache[key] = cls(**cfg._asdict())
        return self._cache[key]

    def run(self, steps: list[Any], text: str) -> Outcome:
        """Run a document through ``steps`` as datatrove's executor does."""
        doc = self.data.Document(text=text, id="0")
        try:
            # numpy warns ("Mean of empty slice") where Gopher averages no words.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for step in steps:
                    keep, reason = self.base.get_filter_result(self.filter(step).filter(doc))
                    if not keep:
                        return ("ok", False, reason, doc.text)
        except Exception as e:  # the exception type is part of parity
            return ("raise", type(e).__name__)
        return ("ok", True, None, doc.text)


@pytest.fixture(scope="module")
def dt() -> Datatrove:
    return Datatrove()


def candidate(steps: list[Any], text: str) -> Outcome:
    try:
        keep, reason, out = pf.apply(text, steps)
    except Exception as e:
        return ("raise", type(e).__name__)
    return ("ok", keep, reason, out)


# ---- configs -------------------------------------------------------------------

_OFF4: dict[str, Any] = {"dup_line_frac": None, "dup_para_frac": None, "dup_line_char_frac": None}
CONFIGS: list[Any] = [
    pytest.param(pf.GopherQuality(), id="gq-default"),
    pytest.param(pf.GopherQuality(min_doc_words=None), id="gq-no-min-words"),
    pytest.param(
        pf.GopherQuality(min_doc_words=None, max_symbol_word_ratio=None), id="gq-no-symbols"
    ),
    pytest.param(
        pf.GopherQuality(
            min_doc_words=None,
            min_avg_word_length=None,
            max_avg_word_length=None,
            max_symbol_word_ratio=None,
            max_bullet_lines_ratio=None,
            max_ellipsis_lines_ratio=None,
        ),
        id="gq-alpha-and-stop-words",
    ),
    pytest.param(
        pf.GopherQuality(
            min_doc_words=5,
            max_doc_words=120,
            min_avg_word_length=4,
            max_avg_word_length=6,
            max_symbol_word_ratio=0.02,
            max_bullet_lines_ratio=0.3,
            max_ellipsis_lines_ratio=0.1,
            max_non_alpha_words_ratio=0.95,
            min_stop_words=1,
        ),
        id="gq-tight",
    ),
    pytest.param(
        pf.GopherQuality(
            min_doc_words=8, stop_words=("house", "garden", "The", "é", "..."), min_stop_words=1
        ),
        id="gq-own-stop-words",
    ),
    pytest.param(
        pf.GopherQuality(
            min_doc_words=None,
            min_avg_word_length=float("nan"),
            max_avg_word_length=float("inf"),
            max_symbol_word_ratio=0.5,
            min_stop_words=0.5,
        ),
        id="gq-nan-inf",
    ),
    pytest.param(pf.GopherRepetition(), id="gr-default"),
    pytest.param(
        pf.GopherRepetition(
            dup_line_frac=0.05,
            dup_para_frac=0.05,
            dup_line_char_frac=0.05,
            dup_para_char_frac=0.05,
            top_n_grams=((1, 0.3), (2, 0.1), (3, 0.08)),
            dup_n_grams=((1, 0.2), (2, 0.1), (3, 0.05), (5, 0.03)),
        ),
        id="gr-small-fractions",
    ),
    pytest.param(
        pf.GopherRepetition(
            **_OFF4,
            dup_para_char_frac=None,
            top_n_grams=((2, 0.05), (3, 0.05)),
            dup_n_grams=((4, 0.02), (11, 0.01)),
        ),
        id="gr-ngrams-only",
    ),
    pytest.param(
        pf.GopherRepetition(
            **_OFF4, dup_para_char_frac=0, top_n_grams=((3, 0.0),), dup_n_grams=((2, 0.0),)
        ),
        id="gr-zero-fractions",
    ),
    pytest.param(pf.C4Quality(), id="c4-default"),
    pytest.param(pf.C4Quality(filter_no_terminal_punct=False), id="c4-fineweb"),
    pytest.param(pf.C4Quality(split_paragraph=False), id="c4-sentences"),
    pytest.param(
        pf.C4Quality(split_paragraph=False, filter_no_terminal_punct=False, min_num_sentences=2),
        id="c4-sentences-fineweb",
    ),
    pytest.param(pf.C4Quality(min_num_sentences=-1), id="c4-no-min-sentences"),
    pytest.param(pf.C4Quality(max_word_length=10), id="c4-max-word-length-10"),
    pytest.param(
        pf.C4Quality(
            remove_citations=False, min_words_per_line=-1, max_word_length=-1, min_num_sentences=1
        ),
        id="c4-loose",
    ),
    pytest.param(
        pf.C4Quality(
            filter_lorem_ipsum=False,
            filter_javascript=False,
            filter_curly_bracket=False,
            filter_policy=False,
            min_words_per_line=0,
            min_num_sentences=2,
        ),
        id="c4-flags-off",
    ),
    pytest.param(pf.FineWebQuality(), id="fw-default"),
    pytest.param(pf.FineWebQuality(line_punct_exclude_zero=True), id="fw-exclude-zero"),
    pytest.param(pf.FineWebQuality(stop_chars=(".", "!", "?")), id="fw-ascii-stops"),
    pytest.param(pf.FineWebQuality(stop_chars=("…", "...", "。", '"', "")), id="fw-odd-stops"),
    pytest.param(pf.FineWebQuality(stop_chars="?!"), id="fw-str-stops"),
    pytest.param(pf.FineWebQuality(stop_chars=(), line_punct_exclude_zero=True), id="fw-no-stops"),
    pytest.param(
        pf.FineWebQuality(
            line_punct_thr=0.0,
            short_line_thr=1.0,
            short_line_length=5,
            char_duplicates_ratio=0.5,
            new_line_ratio=2.0,
        ),
        id="fw-loose",
    ),
    pytest.param(
        pf.FineWebQuality(
            line_punct_thr=0.9,
            short_line_thr=0.1,
            short_line_length=60,
            char_duplicates_ratio=0.0,
            new_line_ratio=0.05,
        ),
        id="fw-strict",
    ),
]

CHAINS: list[Any] = [
    pytest.param(
        [
            pf.GopherRepetition(),
            pf.GopherQuality(),
            pf.C4Quality(filter_no_terminal_punct=False),
            pf.FineWebQuality(),
        ],
        id="fineweb-pipeline",
    ),
    pytest.param(
        [pf.C4Quality(min_num_sentences=1), pf.FineWebQuality(), pf.GopherQuality(min_doc_words=5)],
        id="c4-first",
    ),
    pytest.param(
        [
            pf.C4Quality(split_paragraph=False, min_num_sentences=-1),
            pf.GopherRepetition(dup_line_frac=0.1),
            pf.C4Quality(min_num_sentences=2),
            pf.GopherQuality(min_doc_words=None),
        ],
        id="two-c4s",
    ),
]


# ---- documents -----------------------------------------------------------------

STOP = list(pf.STOP_WORDS)
COMMON = [
    "house", "garden", "river", "committee", "budget", "data", "model", "people", "time",
    "year", "work", "world", "information", "a", "I", "an", "is", "was", "it", "we", "on",
    "for", "you", "this", "are", "or", "by", "from", "at", "as", "not", "but", "all",
]  # fmt: skip
TRICKY = [
    "it's", "don't", "can't", "I'm", "U.S.", "e.g.", "i.e.", "Dr.", "Mr.", "etc.", "3.5",
    "1,000", "2024", "10%", "$5", "#", "#tag", "##", "@user", "http://example.com/a?b=1",
    "www.example.org", "user@example.com", "co-operate", "well-known", "C++", "naïve",
    "café", "Straße", "İstanbul", "ﬁne", "\N{KELVIN SIGN}elvin", "日本語", "中文",
    "Москва", "Ελλάδα",
    "عربي", "हिन्दी", "😀", "👍🏽", "á", "​", "\x01", "\x7f", "\x9f", "１", "％",
    ":)", "--", "—", "–", "(", ")", '"quoted"', "'single'", "«guillemets»", "„low“",
    "‘curly’", "...", "…", "•", "-", "{", "}", "[1]", "[edit]", "Lorem", "ipsum",
    "JavaScript", "cookies", "w" * 1001,
]  # fmt: skip
WORDS = st.one_of(
    st.sampled_from(STOP),
    st.sampled_from(COMMON),
    st.sampled_from(COMMON),
    st.sampled_from(TRICKY),
    st.text(st.characters(categories=["Lu", "Ll", "Lo"]), min_size=1, max_size=10),
    st.text(st.characters(categories=["L", "M", "N", "P", "S"]), min_size=1, max_size=5),
    st.text(st.characters(categories=["P", "S"]), min_size=1, max_size=3),
    st.text(string.ascii_lowercase, min_size=11, max_size=24),
)
#: Whitespace inside a line: some of it is a line boundary for str.splitlines() (and so
#: for Gopher quality and C4) but not for split("\n") (FineWeb) or "\n+" (repetition).
INLINE_WS = [" "] * 24 + [
    "  ", "\t", "\xa0", "　", " ", "​", "\x0b", "\x0c", "\x1c", "\x1d",
    "\x1e", "\x85", " ", " ", "\r", " \t ",
]  # fmt: skip
LINE_SEPS = ["\n"] * 12 + ["\n\n"] * 4 + [
    "\n\n\n", "\r\n", "\r", "\n \n", "\n\t\n", "\x0b", "\x0c", "\x1c", "\x85", " ",
    " ", " \n", "\n ", "\r\n\r\n",
]  # fmt: skip
BULLET_PREFIXES = [
    "•", "-", "- ", "• ", "* ", "·", "‣ ", "► ", "–", "—", "1. ", "  - ", "\t• ", "\xa0-", "--",
]  # fmt: skip
#: Line endings: C4's END_PUNCTUATION, ellipses, citations, and some of datatrove's
#: TERMINAL_PUNCTUATION from other scripts (FineWeb), plus non-terminal ones.
ENDINGS = ["."] * 6 + [""] * 4 + [
    "!", "?", '"', "'", '."', ".'", "?!", "...", "…", " ...", "....", "[1].", ".[2]",
    ".[edit]", "[citation needed].", ". [citation needed]", "。", "！", "？",
    "؟", "।", "‼", "⁉", "｡", "．", "۔", "꓿",
    "\U00011143", "\U00011f43", "\U00016a6e", "\U0001da88", "᠃", ";", ":", ",", ")",
    ". ", ".\t", ".\r", ". \xa0", "•", "……",
]  # fmt: skip
#: Things inserted anywhere in a line.
INSERTS = [
    "lorem ipsum", "Lorem Ipsum", "LOREM IPSUM", "lorem  ipsum", "lorem İpsum",
    "loremipsum", "javascript", "JavaScript", "JAVASCRIPT", "java script", "terms of use",
    "Terms Of Use", "privacy policy", "Privacy Policy", "cookie policy", "cooKie policy",
    "uses cookies", "use of cookies", "use cookies", "USE COOKIES", "TERMS OF ŪSE", "{",
    "}", "{{x}}", "...", "…", " ... ", "#", "# ", "[1]", "[]", "[edit]",
    "[citation needed]", "[12a]", " - ", " • ", "\t", "  ", '"', "'",
]  # fmt: skip
SENTENCES = [
    "The committee met on Tuesday to review the annual budget and the new hiring plan.",
    "Several members asked whether the proposed changes would affect existing projects.",
    "After a long discussion, the chair agreed to publish a detailed summary next week.",
    "Local residents have been invited to share their views with the council by email.",
    "It was the best of times, it was the worst of times, and that is all we know.",
    "The results... were not what anyone had expected from the trial.",
    'She asked, "Have you seen the new library on the corner of Main Street?"',
    "In 2023, about 1,500 people visited the museum every day during the summer.",
    "Click here to read more about our services and the team behind them",
    "Home | About | Contact | Privacy Policy",
    "This website uses cookies to give you the best experience.",
    "Copyright © 2024 Example Inc. All rights reserved.",
    "You must enable JavaScript to use this site.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    'The function returns {"ok": true} when the request succeeds.',
    "Read more...",
    "Posted by admin on March 3, 2021",
    "日本語の文章もここにあります。",
    "¿Dónde está la biblioteca? ¡Aquí está!",
    "Water boils at 100 degrees Celsius at sea level.[1]",
    "Mr. Smith went to Washington, D.C. on Jan. 5th. He stayed for a week!",
    "Use the -v flag (or --verbose) to see what the tool does.",
]
BLANKS = [
    "", " ", "  ", "\n", "\n\n", " \n ", "\t", "\r\n", "\x0b", "\x0c", "\x1c", "\x85",
    " ", "　", "\xa0", "​",
]  # fmt: skip


def _rare(n: int) -> st.SearchStrategy[bool]:
    """True about once in ``n`` draws."""
    return st.integers(0, n - 1).map(lambda x: x == 0)


#: Plain words for prose: long enough on average for Gopher, with its stop words.
PROSE_WORDS = sorted(
    {w.strip(".,!?\"'|()-").lower() for s in SENTENCES[:16] for w in s.split()} - {""}
    | set(COMMON)
    | set(STOP)
)
#: Per-document styles: how the lines of one document are decorated.
STYLES = ["plain"] * 6 + ["bullets", "ellipses", "hashtags", "listy"]


def _decorate(draw: st.DrawFn, line: str, style: str) -> str:
    if style == "bullets" or draw(_rare(12)):
        line = draw(st.sampled_from(BULLET_PREFIXES)) + line
    if style == "hashtags" or draw(_rare(15)):
        line += " " + " ".join(draw(st.lists(st.sampled_from(["#tag", "#", "#news"]), max_size=4)))
    if style == "ellipses" and draw(st.booleans()):
        line += draw(st.sampled_from(["...", "…", " ...", "… ", ".....", "...\t"]))
    if style == "ellipses" and draw(st.booleans()):
        pos = draw(st.integers(0, len(line)))
        line = line[:pos] + draw(st.sampled_from(["...", " … ", "…"])) + line[pos:]
    return line


@st.composite
def lines(draw: st.DrawFn, style: str = "plain") -> str:
    """A noisy line: any words, inserted phrases and markers, any ending."""
    if draw(_rare(4)):
        line = draw(st.sampled_from(SENTENCES))
    else:
        ws = draw(st.lists(WORDS, max_size=16))
        seps = draw(st.lists(st.sampled_from(INLINE_WS), min_size=len(ws), max_size=len(ws)))
        line = "".join(x for w, s in zip(ws, seps, strict=True) for x in (s, w))[1:]
    for extra in draw(st.lists(st.sampled_from(INSERTS), max_size=2)):
        pos = draw(st.integers(0, len(line)))
        line = line[:pos] + extra + line[pos:]
    if not line.endswith(tuple(".!?")) or draw(_rare(3)):
        line += draw(st.sampled_from(ENDINGS))
    line = _decorate(draw, line, style)
    if draw(_rare(6)):
        line = draw(st.sampled_from(INLINE_WS)) + line + draw(st.sampled_from(INLINE_WS))
    return line


@st.composite
def prose_lines(draw: st.DrawFn, style: str = "plain") -> str:
    """A line of ordinary prose (what the filters keep), sometimes with one oddity."""
    if draw(_rare(6)):
        line = draw(st.sampled_from(SENTENCES))
    else:
        ws = draw(st.lists(st.sampled_from(PROSE_WORDS), min_size=3, max_size=18))
        line = " ".join(ws).capitalize() + draw(st.sampled_from(["."] * 8 + ENDINGS[-30:]))
    if draw(_rare(8)):
        pos = draw(st.integers(0, len(line)))
        line = line[:pos] + draw(st.sampled_from(INSERTS + TRICKY[:-1])) + line[pos:]
    return _decorate(draw, line, style)


def _join(draw: st.DrawFn, pool: list[str], seps: st.SearchStrategy[str]) -> str:
    """Pool entries in order with a few repeated, or (sometimes) many repeats."""
    if draw(_rare(4)):
        picks = draw(st.lists(st.integers(0, len(pool) - 1), min_size=1, max_size=30))
    else:
        picks = list(range(len(pool)))
        for _ in range(draw(st.sampled_from([0, 0, 0, 1, 2, 3]))):
            picks.insert(draw(st.integers(0, len(picks))), draw(st.integers(0, len(pool) - 1)))
    between = draw(st.lists(seps, min_size=len(picks), max_size=len(picks)))
    doc = "".join(pool[i] + s for i, s in zip(picks, between, strict=True))
    if draw(st.booleans()):
        doc = doc[: -len(between[-1])]
    return doc


@st.composite
def documents(draw: st.DrawFn) -> str:
    """Documents of 0 to ~300 words that reach every check of the four filters."""
    shape = draw(st.integers(0, 19))
    if shape == 0:  # empty or whitespace only
        return "".join(draw(st.lists(st.sampled_from(BLANKS), max_size=4)))
    if shape == 1:  # anything
        return draw(st.text(st.characters(codec="utf-8"), max_size=60))
    if shape == 2:  # one phrase over and over, among other words: top and duplicated n-grams
        phrase = " ".join(draw(st.lists(WORDS, min_size=1, max_size=6)))
        parts = [phrase] * draw(st.integers(1, 25)) + draw(st.lists(WORDS, max_size=40))
        order = draw(st.permutations(range(len(parts))))
        sep = draw(st.sampled_from([" ", " ", "\n", ". ", "\n\n"]))
        return sep.join(parts[i] for i in order)
    style = draw(st.sampled_from(STYLES))
    # "listy": blank lines between lines (FineWeb's newline-per-word ratio).
    seps = st.sampled_from(["\n" * 3, "\n" * 5, "\n \n \n"] if style == "listy" else LINE_SEPS)
    if shape <= 9:  # prose, one line per sentence (or a few sentences per line)
        # At least 4 lines: shorter texts fail Gopher's top n-gram fractions with no
        # repetition at all (one 4-gram of 25 characters is 16% of 156).
        n = draw(st.integers(4, 24))
        pool = draw(st.lists(prose_lines(style), min_size=n, max_size=n))
        if draw(_rare(4)):
            pool = [" ".join(pool[i : i + 3]) for i in range(0, len(pool), 3)]
        return _join(draw, pool, seps)
    if shape <= 12:  # paragraphs of prose, some repeated (Gopher's paragraph checks)
        paras = st.lists(prose_lines(style), min_size=1, max_size=5).map("\n".join)
        pool = draw(st.lists(paras, min_size=1, max_size=8))
        return _join(draw, pool, st.sampled_from(["\n\n", "\n\n", "\n\n\n", "\n \n", "\r\n\r\n"]))
    return _join(draw, draw(st.lists(lines(style), min_size=1, max_size=20)), seps)


# ---- live differential tests ---------------------------------------------------


def _check(dt: Datatrove, steps: list[Any], doc: str) -> Outcome:
    want = dt.run(steps, doc)
    got = candidate(steps, doc)
    note(f"datatrove: {want!r}")
    note(f"pyroid:    {got!r}")
    # --hypothesis-show-statistics lists how often each outcome was reached.
    if want[0] == "raise":
        event(f"raises {want[1]}")
    else:
        event(f"drop: {want[2]}" if want[2] else "kept, text changed" if want[3] != doc else "kept")
    assert got == want
    return got


@pytest.mark.parametrize("cfg", CONFIGS)
@examples
@given(doc=documents())
def test_matches_datatrove(dt: Datatrove, cfg: Any, doc: str) -> None:
    got = _check(dt, [cfg], doc)
    # The datatrove instance as config, and the single-purpose functions, agree.
    assert candidate([dt.filter(cfg)], doc) == got
    if got[0] == "ok":
        name = {
            pf.GopherQuality: "gopher_quality",
            pf.GopherRepetition: "gopher_repetition",
            pf.C4Quality: "c4_quality",
            pf.FineWebQuality: "fineweb_quality",
        }[type(cfg)]
        keep: Callable[..., Any] = getattr(pf, name)
        reasons: Callable[..., Any] = getattr(pf, name + "_reasons")
        assert keep(doc, cfg) is got[1]
        assert reasons(doc, cfg) == got[2]
        if isinstance(cfg, pf.C4Quality):
            out = pf.c4_lines(doc, cfg)
            assert out == got[3]
            if out == doc:
                assert out is doc


@pytest.mark.parametrize("steps", CHAINS)
@examples
@given(doc=documents())
def test_chains_match_datatrove(dt: Datatrove, steps: list[Any], doc: str) -> None:
    _check(dt, steps, doc)


def _arrow(docs: list[str | None], large: bool) -> Any:
    return _testing.column(docs, large=large, chunks=2 if len(docs) > 1 else 1)


@examples
@given(
    docs=st.lists(documents(), max_size=6),
    kind=st.sampled_from(["list", "tuple", "string", "large_string"]),
    nulls=st.lists(st.integers(0, 6), max_size=2),
    steps=st.sampled_from([p.values[0] for p in CHAINS] + [[p.values[0]] for p in CONFIGS]),
)
def test_batches_match_datatrove(
    dt: Datatrove, docs: list[str], kind: str, nulls: list[int], steps: list[Any]
) -> None:
    want = [dt.run(steps, d) for d in docs]
    raised = {w[1] for w in want if w[0] == "raise"}
    values: list[str | None] = list(docs)
    if kind in ("string", "large_string"):
        for i in nulls:
            values.insert(min(i, len(values)), None)
        texts: Any = _arrow(values, kind == "large_string")
    else:
        texts = values if kind == "list" else tuple(values)
    if raised:
        with pytest.raises(Exception) as info:  # noqa: PT011  the type is checked below
            pf.apply(texts, steps)
        assert type(info.value).__name__ in raised
        return
    res = pf.apply(texts, steps)
    if kind in ("string", "large_string"):
        it: Iterator[Outcome] = iter(want)
        full = [None if v is None else next(it) for v in values]
        assert res.keep.to_pylist() == [w and w[1] for w in full]
        assert res.reason.to_pylist() == [w and w[2] for w in full]
        assert res.text.to_pylist() == [w and w[3] for w in full]
    else:
        assert type(res.keep) is type(texts)
        assert [("ok", *r) for r in zip(*res, strict=True)] == want
        assert all(o is d for o, d in zip(res.text, docs, strict=True) if o == d)


SURROGATES = st.characters(categories=["Cs"])


@examples
@given(
    doc=documents(),
    surrogates=st.lists(st.tuples(st.integers(0, 10**6), SURROGATES), min_size=1, max_size=3),
    others=st.lists(documents(), max_size=3),
    cfg=st.sampled_from([p.values[0] for p in CONFIGS]),
)
def test_lone_surrogates_raise(
    doc: str, surrogates: list[tuple[int, str]], others: list[str], cfg: Any
) -> None:
    for pos, s in surrogates:
        pos %= len(doc) + 1
        doc = doc[:pos] + s + doc[pos:]
    # A high surrogate followed by a low one is a pair in UTF-16, but in a Python str it is
    # still two lone surrogates.
    with pytest.raises(UnicodeEncodeError):
        pf.apply(doc, [cfg])
    batch = [*others, doc]
    with pytest.raises(UnicodeEncodeError) as info:
        pf.apply(batch, [cfg])
    assert f"index {len(others)}" in "\n".join(getattr(info.value, "__notes__", []))
    with pytest.raises(UnicodeEncodeError):
        pf.apply(tuple(reversed(batch)), [cfg])


def test_reasons_are_the_documented_ones(dt: Datatrove) -> None:
    """The examples above reach every reason; this guards the strings themselves."""
    src = {
        name: inspect.getsource(getattr(dt.filters, name)) for name in Datatrove.CLASSES.values()
    }
    for reason in (
        "gopher_short_doc",
        "gopher_long_doc",
        "gopher_below_avg_threshold",
        "gopher_above_avg_threshold",
        "gopher_too_many_hashes",
        "gopher_too_many_ellipsis",
        "gopher_too_many_bullets",
        "gopher_too_many_end_ellipsis",
        "gopher_below_alpha_threshold",
        "gopher_enough_stop_words",
    ):
        assert f'"{reason}"' in src["GopherQualityFilter"]
    for reason in ("empty", "dup_para_frac", "dup_para_char_frac", "dup_line_frac"):
        assert f'"{reason}"' in src["GopherRepetitionFilter"]
    assert 'f"top_{n}_gram"' in src["GopherRepetitionFilter"]
    assert 'f"duplicated_{n}_n_grams"' in src["GopherRepetitionFilter"]
    for reason in ("lorem_ipsum", "curly_bracket", "too_few_sentences"):
        assert f'"{reason}"' in src["C4QualityFilter"]
    for reason in ("empty", "line_punct_ratio", "short_line_ratio", "char_dup_ratio", "list_ratio"):
        assert f'"{reason}"' in src["FineWebQualityFilter"]


def test_config_defaults_are_datatrove_defaults(dt: Datatrove) -> None:
    for config, name in Datatrove.CLASSES.items():
        cls: Any = config
        params = inspect.signature(getattr(dt.filters, name)).parameters
        assert set(params) - {"exclusion_writer"} == set(cls._fields), name
        for field, default in cls._field_defaults.items():
            assert params[field].default == default, (name, field)
    gq_module: Any = pytest.importorskip("datatrove.pipeline.filters.gopher_quality_filter")
    assert tuple(gq_module.STOP_WORDS) == pf.STOP_WORDS
    assert set(pf.english_languages()) >= {"eng", "en"}


@pytest.mark.parametrize("lang", ["en", "eng", "sco", "eng_Latn"])
def test_english_codes_tokenize_like_datatrove(dt: Datatrove, lang: str) -> None:
    assert lang in pf.english_languages()
    docs = [*SENTENCES, "\n".join(SENTENCES), " ".join(TRICKY[:-1])]
    for cfg in (pf.GopherQuality(language=lang, min_doc_words=3), pf.C4Quality(language=lang)):
        for d in docs:
            assert candidate([cfg], d) == dt.run([cfg], d)


# ---- pinned decisions ------------------------------------------------------------

DATA = Path(os.environ.get("PYROID_PARITY_DATA", "/data"))
REF = DATA / "parity" / "filters-ref.parquet"
CORPORA = {
    "dev": DATA / "fineweb" / "fineweb-dev-10k.parquet",
    "adversarial": DATA / "parity" / "adversarial.parquet",
}
#: The harness's FILTER_CASES (parity/pyroid_parity/specs.py).
PINNED_CASES = {
    "gopher_repetition": pf.GopherRepetition(),
    "gopher_quality": pf.GopherQuality(),
    "c4_quality": pf.C4Quality(),
    "c4_quality@fineweb": pf.C4Quality(filter_no_terminal_punct=False),
    "fineweb_quality": pf.FineWebQuality(),
}


def _parquet() -> Any:
    pytest.importorskip("pyarrow")
    pq: Any = pytest.importorskip("pyarrow.parquet")
    missing = [p for p in (REF, *CORPORA.values()) if not p.is_file()]
    if missing:
        pytest.skip(f"pinned data not found: {', '.join(map(str, missing))}")
    return pq


@pytest.fixture(scope="module")
def pinned() -> dict[str, dict[str, Outcome]]:
    t = _parquet().read_table(REF).to_pydict()
    out: dict[str, dict[str, Outcome]] = {case: {} for case in PINNED_CASES}
    rows = zip(t["case"], t["id"], t["outcome"], t["keep"], t["reason"], t["text_out"], strict=True)
    for case, doc_id, outcome, keep, reason, text in rows:
        out[case][doc_id] = (outcome, keep, reason, text) if outcome == "ok" else (outcome, reason)
    return out


@pytest.fixture(scope="module")
def corpora() -> dict[str, Any]:
    pq = _parquet()
    return {name: pq.read_table(path, columns=["id", "text"]) for name, path in CORPORA.items()}


def test_pinned_reference_is_complete(
    pinned: dict[str, dict[str, Outcome]], corpora: dict[str, Any]
) -> None:
    ids = {i for t in corpora.values() for i in t.column("id").to_pylist()}
    assert len(ids) == sum(t.num_rows for t in corpora.values())
    for case in PINNED_CASES:
        assert set(pinned[case]) == ids, case


@pytest.mark.parametrize("corpus", list(CORPORA))
@pytest.mark.parametrize("case", list(PINNED_CASES))
def test_pinned_decisions(
    case: str, corpus: str, pinned: dict[str, dict[str, Outcome]], corpora: dict[str, Any]
) -> None:
    table = corpora[corpus]
    ids: list[str] = table.column("id").to_pylist()
    texts: list[str] = table.column("text").to_pylist()
    want = [pinned[case][i] for i in ids]
    cfg = PINNED_CASES[case]
    # Documents datatrove raised on are checked one by one (a batch raises as a whole).
    ok = [i for i, w in enumerate(want) if w[0] == "ok"]
    for i, w in enumerate(want):
        if w[0] == "raise":
            assert candidate([cfg], texts[i]) == w, ids[i]
    ok_texts = [texts[i] for i in ok]
    res = pf.apply(ok_texts, [cfg])
    got = [
        ("ok", k, r, None if out == text else out)
        for k, r, out, text in zip(*res, ok_texts, strict=True)
    ]
    bad = [(ids[i], g, want[i]) for i, g in zip(ok, got, strict=True) if g != want[i]]
    assert not bad, f"{len(bad)}/{len(ok)} decisions differ for {case} on {corpus}: {bad[:3]}"
    # Unchanged text is the caller's object.
    assert all(o is t for o, t in zip(res.text, ok_texts, strict=True) if o == t)
    if len(ok) == len(texts):
        # The same decisions straight from the Arrow column (as many chunks as parquet gave).
        col = pf.apply(table.column("text"), [cfg])
        assert col.keep.to_pylist() == res.keep
        assert col.reason.to_pylist() == res.reason
        assert col.text.to_pylist() == res.text
