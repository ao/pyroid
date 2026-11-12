# Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace; modified.
"""pyroid.filters beyond English: language codes, the opt-in Python tokenizer, parity.

- Language table and errors (no datatrove needed): every datatrove code, which are native,
  ``ValueError`` for unknown codes, ``NotImplementedError`` without the opt-in, ``PyroidError``
  when datatrove or a segmenter is missing (in a child interpreter that hides it).
- The Python-tokenizer path with a duck-typed tokenizer (no datatrove needed): container
  kinds, Arrow nulls and types, identity of unchanged texts, ``threads=``, the warning.
- Live datatrove (``pytest.importorskip``): Hypothesis documents in each script, every filter
  and ``apply`` chains, for native languages (pyroid's own tokenizers) and, through
  ``python_tokenizer=True``, datatrove's tokenizers with the segmenters installed (SudachiPy,
  pyvi, PyThaiNLP). Vietnamese is always native, Japanese and Thai when pyroid finds a
  SudachiDict or PyThaiNLP; test_filters_japanese.py and test_filters_vithai.py cover them.
- The FineWeb-2 slices (``$PYROID_PARITY_DATA/fineweb2/<lang>.parquet``, default ``/data``)
  with FineWeb-2's own per-language filter settings, against live datatrove.

``PYROID_FILTERS_EXAMPLES=n`` sets the Hypothesis examples per test (default: the profile's).
"""

# Other scripts' letters and punctuation on purpose.
# ruff: noqa: RUF001

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import threading
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

import pyroid
import pyroid.dedup as dd
import pyroid.filters as pf
from pyroid._pyroid import _testing

pytestmark = [
    # datatrove's and spaCy's tokenizers are not thread-safe; the fallback warning is global.
    pytest.mark.thread_unsafe(reason="datatrove, spaCy and the once-per-process warning"),
    # Which path ran, not what the result is (MODULES.md); parity suites ignore it.
    pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning"),
    # Segmenters warn on import (jieba: invalid escapes; others: deprecations).
    pytest.mark.filterwarnings("ignore::SyntaxWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::FutureWarning"),
]

_HEALTH = [HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much]
_EXAMPLES = os.environ.get("PYROID_FILTERS_EXAMPLES")
examples = (
    settings(suppress_health_check=_HEALTH, max_examples=int(_EXAMPLES))
    if _EXAMPLES
    else settings(suppress_health_check=_HEALTH)
)

#: ("ok", keep, reason, text) or ("raise", exception type name).
Outcome = tuple[Any, ...]

ALL_FUNCS: list[Callable[..., Any]] = [
    pf.gopher_quality,
    pf.gopher_quality_reasons,
    pf.gopher_repetition,
    pf.gopher_repetition_reasons,
    pf.c4_quality,
    pf.c4_quality_reasons,
    pf.c4_lines,
    pf.fineweb_quality,
    pf.fineweb_quality_reasons,
]
CONFIG_OF: dict[Callable[..., Any], type] = {
    pf.gopher_quality: pf.GopherQuality,
    pf.gopher_quality_reasons: pf.GopherQuality,
    pf.gopher_repetition: pf.GopherRepetition,
    pf.gopher_repetition_reasons: pf.GopherRepetition,
    pf.c4_quality: pf.C4Quality,
    pf.c4_quality_reasons: pf.C4Quality,
    pf.c4_lines: pf.C4Quality,
    pf.fineweb_quality: pf.FineWebQuality,
    pf.fineweb_quality_reasons: pf.FineWebQuality,
}
WHAT: dict[Callable[..., Any], int] = {
    pf.gopher_quality: 0,
    pf.gopher_quality_reasons: 1,
    pf.gopher_repetition: 0,
    pf.gopher_repetition_reasons: 1,
    pf.c4_quality: 0,
    pf.c4_quality_reasons: 1,
    pf.c4_lines: 2,
    pf.fineweb_quality: 0,
    pf.fineweb_quality_reasons: 1,
}


# ---- the language table ----------------------------------------------------------


def sudachi_dictionary() -> bool:
    """Whether pyroid finds a SudachiDict as it looks for one (Japanese is then native)."""
    if "PYROID_SUDACHI_DICT" in os.environ:
        return True
    spec = importlib.util.find_spec("sudachidict_core")
    return spec is not None and spec.origin is not None


def pythainlp_corpus() -> bool:
    """Whether pyroid finds PyThaiNLP's corpus as it looks for it (Thai is then native)."""
    if "PYROID_PYTHAINLP_CORPUS" in os.environ:
        return True
    spec = importlib.util.find_spec("pythainlp")
    return spec is not None and spec.origin is not None


def test_languages_table() -> None:
    langs = pf.languages()
    assert isinstance(langs, dict)
    assert list(langs) == sorted(langs)
    assert all(isinstance(v, pf.Language) for v in langs.values())
    # The counts the stubs document: 4,025 native codes, and the four Japanese ones with a
    # SudachiDict, the twenty Thai ones with PyThaiNLP.
    ja = sudachi_dictionary()
    th = pythainlp_corpus()
    assert len(langs) == 4343
    native = {c for c, v in langs.items() if v.native}
    assert len(native) == 4025 + 4 * ja + 20 * th
    by_class: dict[str, int] = {}
    for v in langs.values():
        if v.native:
            by_class[v.tokenizer] = by_class.get(v.tokenizer, 0) + 1
    assert by_class == {
        "SpaCyTokenizer": 3768 + 4 * ja,
        "IndicNLPTokenizer": 129,
        "WhitespaceTokenizer": 128,
        **({"ThaiTokenizer": 20} if th else {}),
    }
    spacy_native = {v.arg for v in langs.values() if v.native and v.tokenizer == "SpaCyTokenizer"}
    # The 57 rule-based languages, zh (jieba), vi (pyvi) and, with a dictionary, ja (SudachiPy).
    assert len(spacy_native) == 59 + ja
    assert sum(v.arg == "xx" for v in langs.values()) == 1995
    assert langs["deu_Latn"] == pf.Language("SpaCyTokenizer", "de", True)
    assert langs["uzn_Latn"] == pf.Language("SpaCyTokenizer", "xx", True)
    assert langs["hin_Deva"] == pf.Language("IndicNLPTokenizer", "hi", True)
    assert langs["kat_Geor"] == pf.Language("WhitespaceTokenizer", "", True)
    assert langs["cmn_Hani"] == pf.Language("SpaCyTokenizer", "zh", True)
    assert langs["jpn_Jpan"] == pf.Language("SpaCyTokenizer", "ja", ja)
    assert langs["vie_Latn"] == pf.Language("SpaCyTokenizer", "vi", True)
    assert langs["tha_Thai"] == pf.Language("ThaiTokenizer", "", th)
    assert langs["kor_Hang"] == pf.Language("KiwiTokenizer", "", False)
    assert langs["kaz_Cyrl"] == pf.Language("StanzaTokenizer", "kk", False)
    eng = {c for c, v in langs.items() if (v.tokenizer, v.arg) == ("SpaCyTokenizer", "en")}
    assert set(pf.english_languages()) == eng
    # A copy: callers can't change pyroid's table.
    langs.clear()
    assert pf.languages()
    assert dd.languages is pf.languages
    assert dd.Language is pf.Language


def test_languages_match_datatrove() -> None:
    wt: Any = pytest.importorskip("datatrove.utils.word_tokenizers")
    factories = wt.load_tokenizer_assignments()
    langs = pf.languages()
    assert set(langs) == set(factories)
    for code, f in factories.items():
        cls, arg = f.args  # partial(tok_factory_wrapper, class_name, tok_code)
        assert (langs[code].tokenizer, langs[code].arg) == (cls, arg), code


# ---- errors ----------------------------------------------------------------------

DATATROVE_UNKNOWN = (
    "Language '{}' doesn't have a tokenizer assigned. Pass in a WordTokenizer directly or "
    "update tokenizer_assignment.csv"
)


def _call(f: Callable[..., Any], texts: Any, **kw: Any) -> Any:
    return f(texts, **kw)


@pytest.mark.parametrize("lang", ["xxx_Zzzz", "", "ENG", "eng_Latin", "de-DE"])
@pytest.mark.parametrize("python_tokenizer", [False, True])
def test_unknown_language_raises_datatrove_value_error(lang: str, python_tokenizer: bool) -> None:
    msg = DATATROVE_UNKNOWN.format(lang)
    for f in ALL_FUNCS:
        with pytest.raises(ValueError, match="doesn't have a tokenizer assigned") as e:
            f(["x"], language=lang, python_tokenizer=python_tokenizer)
        assert str(e.value) == msg
    with pytest.raises(ValueError, match="doesn't have a tokenizer assigned"):
        pf.apply([], [pf.C4Quality(language=lang)], python_tokenizer=python_tokenizer)
    with pytest.raises(ValueError, match="doesn't have a tokenizer assigned"):
        dd.minhash_signatures(["x"], language=lang, python_tokenizer=python_tokenizer)
    with pytest.raises(ValueError, match="doesn't have a tokenizer assigned"):
        dd.MinHashDedup(language=lang, python_tokenizer=python_tokenizer)


def test_unknown_language_message_is_datatrove_s() -> None:
    wt: Any = pytest.importorskip("datatrove.utils.word_tokenizers")
    with pytest.raises(ValueError, match="tokenizer") as e:
        wt.load_word_tokenizer("xxx_Zzzz")
    with pytest.raises(ValueError, match="tokenizer") as mine:
        pf.gopher_quality("x", language="xxx_Zzzz")
    assert str(mine.value) == str(e.value)


@pytest.mark.parametrize(
    ("lang", "names"),
    [
        # jpn_Jpan without a dictionary: test_filters_japanese.py; tha_Thai without PyThaiNLP:
        # test_filters_vithai.py.
        ("kaz_Cyrl", ['StanzaTokenizer("kk")', "stanza"]),
        ("kor_Hang", ["KiwiTokenizer()", "kiwipiepy"]),
        ("khm_Khmr", ["KhmerTokenizer()", "khmer-nltk"]),
        ("mya_Mymr", ["BurmeseTokenizer()", "pyidaungsu"]),
    ],
)
def test_non_native_language_needs_opt_in(lang: str, names: list[str]) -> None:
    for f in ALL_FUNCS:
        with pytest.raises(NotImplementedError) as e:
            f(["x"], language=lang)
        for name in [*names, "python_tokenizer=True", lang]:
            assert name in str(e.value), (f, name)
    with pytest.raises(NotImplementedError, match="python_tokenizer=True"):
        pf.apply([], [pf.GopherQuality(), pf.FineWebQuality(language=lang)])
    # The error comes before any document is read.
    with pytest.raises(NotImplementedError, match="python_tokenizer=True"):
        pf.gopher_quality(42, language=lang)  # type: ignore[call-overload]
    with pytest.raises(NotImplementedError, match="MinHash still run in Rust"):
        dd.minhash_dedup(["x"], language=lang)


@pytest.mark.parametrize("bad", [42, b"deu", None, object()])
def test_language_type_errors(bad: Any) -> None:
    with pytest.raises(TypeError, match="language must be"):
        pf.gopher_quality("x", language=bad)
    with pytest.raises(TypeError, match="language must be"):
        pf.apply("x", [pf.FineWebQuality(language=bad)], python_tokenizer=True)
    with pytest.raises(TypeError, match="language must be"):
        dd.minhash_signatures("x", language=bad)


def test_c4_tokenizer_object_needs_span_tokenize() -> None:
    with pytest.raises(TypeError, match="span_tokenize"):
        pf.c4_quality("x", language=WordsOnly())  # type: ignore[call-overload]
    # The other filters only need word_tokenize.
    assert pf.gopher_quality_reasons("x", language=WordsOnly()) == "gopher_short_doc"


HIDE = """
import sys, warnings
for name in {hide!r}:
    sys.modules[name] = None  # import fails
import pyroid, pyroid.filters as pf, pyroid.dedup as dd
warnings.simplefilter("ignore", SyntaxWarning)
for call in (
    lambda: pf.gopher_quality(["x"], language={lang!r}, python_tokenizer=True),
    lambda: pf.apply(["x"], [pf.C4Quality(language={lang!r})], python_tokenizer=True),
    lambda: dd.minhash_signatures(["x"], language={lang!r}, python_tokenizer=True),
    lambda: dd.MinHashDedup(language={lang!r}, python_tokenizer=True),
):
    try:
        call()
    except pyroid.PyroidError as e:
        assert type(e) is pyroid.PyroidError, type(e)
        print("PyroidError:", e)
    else:
        raise SystemExit("no PyroidError")
"""


@pytest.mark.parametrize(
    ("lang", "hide", "needs"),
    [
        ("deu_Latn", ["datatrove"], "datatrove"),
        ("cmn_Hani", ["datatrove"], "datatrove"),
        ("cmn_Hani", ["jieba"], "jieba"),
        ("jpn_Jpan", ["sudachipy"], "sudachipy"),
        ("vie_Latn", ["pyvi"], "pyvi"),
        ("tha_Thai", ["pythainlp"], "pythainlp"),
    ],
)
def test_missing_packages_raise_pyroid_error(
    run_python: Callable[..., Any], lang: str, hide: list[str], needs: str
) -> None:
    if hide != ["datatrove"]:
        pytest.importorskip("datatrove.utils.word_tokenizers")
    r = run_python(HIDE.format(hide=hide, lang=lang))
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("PyroidError:")]
    assert len(lines) == 4, r.stdout
    for ln in lines:
        assert needs in ln, ln
        assert "pip install" in ln, ln


WARN_ONCE = """
import warnings, pyroid, pyroid.filters as pf, pyroid.dedup as dd

class Split:
    def word_tokenize(self, text):
        return text.split()

tok = Split()
docs = ["one two three.", "four five six seven."]
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    for _ in range(3):
        pf.gopher_quality(docs, language=tok)
        pf.gopher_quality_reasons(docs, language=tok)
        pf.apply(docs, [pf.GopherQuality(language=tok), pf.FineWebQuality(language=tok)])
        dd.minhash_signatures(docs, language=tok, n_grams=2)
        dd.MinHashDedup(language=tok, n_grams=2).fit(docs)
for x in w:
    assert type(x.message) is pyroid.FallbackWarning, (x.category, x.message)
    assert x.filename == "<string>", x.filename  # attributed to the caller
    print(x.message)
"""


def test_fallback_warning_once_per_function_and_reason(run_python: Callable[..., Any]) -> None:
    r = run_python(WARN_ONCE)
    assert r.returncode == 0, r.stderr
    got = r.stdout.splitlines()
    assert len(got) == 5, got
    for name in (
        "pyroid.filters.gopher_quality:",
        "pyroid.filters.gopher_quality_reasons:",
        "pyroid.filters.apply:",
        "pyroid.dedup.minhash_signatures:",
        "pyroid.dedup.MinHashDedup:",
    ):
        assert sum(m.startswith(name) for m in got) == 1, (name, got)
    for m in got:
        for part in ("Split", "in Python", "calling thread"):
            assert part in m, m


def test_fallback_warning_names_language_and_tokenizer(run_python: Callable[..., Any]) -> None:
    pytest.importorskip("datatrove.utils.word_tokenizers")
    pytest.importorskip("spacy")
    r = run_python(
        """
        import warnings, pyroid, pyroid.filters as pf
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pf.apply(["Ein Satz. Noch einer."],
                     [pf.GopherQuality(language="deu_Latn"), pf.C4Quality(language="deu_Latn")],
                     python_tokenizer=True)
        for x in w:
            if type(x.message) is pyroid.FallbackWarning:
                print(x.message)
        """
    )
    assert r.returncode == 0, r.stderr
    got = r.stdout.splitlines()
    assert len(got) == 2, got  # the word steps and the C4 step are two reasons
    for m in got:
        for part in ("'deu_Latn'", 'SpaCyTokenizer("de")', "natively"):
            assert part in m, m
    assert any("C4QualityFilter runs as datatrove's own filter" in m for m in got)


# ---- the Python path with a duck-typed tokenizer (no datatrove needed) ------------


class Split:
    """A duck-typed word tokenizer: ``str.split`` words, sentences at ". "."""

    def word_tokenize(self, text: str) -> list[str]:
        return text.split()

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        spans, start = [], 0
        for part in text.split(". "):
            end = start + len(part)
            if part.strip():
                spans.append((start, min(end + 1, len(text))))
            start = end + 2
        return spans or [(0, len(text))]


class WordsOnly:
    """Only ``word_tokenize`` (enough for every filter but C4)."""

    def word_tokenize(self, text: str) -> list[str]:
        return text.split()


class Counting(Split):
    """``Split``, counting the texts it is asked to tokenize."""

    def __init__(self) -> None:
        self.seen: list[str] = []
        self.lock = threading.Lock()

    def word_tokenize(self, text: str) -> list[str]:
        with self.lock:
            self.seen.append(text)
        return super().word_tokenize(text)


class NoWords(Split):
    def word_tokenize(self, text: str) -> list[str]:
        return []


SPLIT = Split()
DOCS = [
    "The quick brown fox jumps over the lazy dog. And then it runs away.\nNew line here.",
    "",
    "short",
    "repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat",
    "\n".join(["the line is here and it is long enough to count."] * 6),
    " ".join(["the", "of", "and", "to", "be"] * 20) + ".",
    "Ünïcödé wörds hère, with the and of. " * 5,
]


def python_configs(lang: Any) -> list[Any]:
    return [
        pf.GopherQuality(language=lang, min_doc_words=3),
        pf.GopherRepetition(language=lang),
        pf.FineWebQuality(language=lang),
        pf.GopherQuality(language=lang),
    ]


def test_python_path_kinds_and_identity() -> None:
    for cfg in python_configs(SPLIT):
        res = pf.apply(DOCS, [cfg])
        assert isinstance(res.keep, list)
        assert isinstance(res.reason, list)
        assert all(t is d for t, d in zip(res.text, DOCS, strict=True))  # never rewritten
        tup = pf.apply(tuple(DOCS), [cfg])
        assert tup.keep == tuple(res.keep)
        assert tup.reason == tuple(res.reason)
        assert isinstance(tup.text, tuple)
        for i, d in enumerate(DOCS):
            one = pf.apply(d, [cfg])
            assert (one.keep, one.reason) == (res.keep[i], res.reason[i])
            assert one.text is d
        funcs: dict[type, Callable[..., Any]] = {
            pf.GopherQuality: pf.gopher_quality,
            pf.GopherRepetition: pf.gopher_repetition,
            pf.FineWebQuality: pf.fineweb_quality,
        }
        f = funcs[type(cfg)]
        assert f(DOCS, cfg) == res.keep
        assert f(DOCS[0], cfg) is res.keep[0]
        assert type(f(DOCS[0], cfg)) is bool


@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("chunks", [1, 3])
def test_python_path_arrow_without_pyarrow(large: bool, chunks: int) -> None:
    values: list[str | None] = [*DOCS[:3], None, *DOCS[3:], None]
    col = _testing.column(values, large=large, chunks=chunks, name="text")
    want = pf.apply([v for v in values if v is not None], [pf.GopherRepetition(language=SPLIT)])
    res = pf.apply(col, [pf.GopherRepetition(language=SPLIT)])
    it = iter(zip(want.keep, want.reason, want.text, strict=True))
    exp = [(None, None, None) if v is None else next(it) for v in values]
    assert res.keep.to_pylist() == [e[0] for e in exp]
    assert res.reason.to_pylist() == [e[1] for e in exp]
    assert res.text.to_pylist() == [e[2] for e in exp]
    assert res.keep.type == "bool"
    assert res.reason.type == "string"
    assert res.text.type == ("large_string" if large else "string")
    assert res.keep.name == "text"
    assert res.keep.null_count == 2
    mask = pf.fineweb_quality(col, language=SPLIT)
    assert mask.to_pylist() == pf.apply(col, [pf.FineWebQuality(language=SPLIT)]).keep.to_pylist()


def test_python_path_pyarrow(pa: Any) -> None:
    arr = pa.chunked_array([DOCS[:4], [None, *DOCS[4:]]], type=pa.string_view())
    res = pf.apply(arr, [pf.GopherQuality(language=SPLIT, min_doc_words=3)])
    want = pf.apply(DOCS, [pf.GopherQuality(language=SPLIT, min_doc_words=3)])
    assert res.keep.to_pylist() == [*want.keep[:4], None, *want.keep[4:]]
    assert pa.chunked_array(res.reason).to_pylist() == [*want.reason[:4], None, *want.reason[4:]]
    sliced = pa.array(DOCS, type=pa.large_string()).slice(2)
    got = pf.gopher_repetition_reasons(sliced, language=SPLIT)
    assert got.to_pylist() == pf.gopher_repetition_reasons(DOCS[2:], language=SPLIT)


def test_python_path_input_errors() -> None:
    with pytest.raises(pyroid.InputError):
        pf.gopher_quality(["ok", 3], language=SPLIT)  # type: ignore[list-item]
    with pytest.raises(pyroid.InputError):
        pf.gopher_quality(b"bytes", language=SPLIT)  # type: ignore[call-overload]
    with pytest.raises(UnicodeEncodeError):
        pf.gopher_quality(["fine", "bad \ud800"], language=SPLIT)

    class Bad:
        def word_tokenize(self, text: str) -> Any:
            return [1, 2]

    with pytest.raises(pyroid.InputError, match="expected str"):
        pf.gopher_quality("words here", language=Bad())

    class Raises:
        def word_tokenize(self, text: str) -> list[str]:
            raise RuntimeError("tokenizer failed")

    with pytest.raises(RuntimeError, match="tokenizer failed"):
        pf.gopher_quality("words here", language=Raises())


def test_python_path_zero_division_names_document() -> None:
    cfg = pf.FineWebQuality(
        language=NoWords(), line_punct_thr=0, short_line_thr=999, char_duplicates_ratio=1
    )
    with pytest.raises(ZeroDivisionError) as e:
        pf.apply(["ok.", "a line.\nanother line."], [cfg])
    assert any("document 0" in n for n in getattr(e.value, "__notes__", [])), e.value


def test_python_path_only_tokenizes_documents_that_need_words() -> None:
    tok = Counting()
    docs = ["", "a.\n" * 40, "Some normal words here, the and of.\nAnother line, with words."]
    # GopherRepetition drops the empty one and the duplicated lines before it needs words;
    # FineWebQuality then needs them for the last one only.
    pf.apply(docs, [pf.GopherRepetition(language=tok), pf.FineWebQuality(language=tok)])
    assert tok.seen == [docs[2]]  # once: the two steps share the words
    tok.seen.clear()
    pf.apply(docs, [pf.GopherQuality(language=tok)])
    assert tok.seen == docs  # Gopher quality always tokenizes


@pytest.mark.parametrize("threads", [None, 1, 2, 4])
def test_python_path_threads(threads: int | None) -> None:
    docs = [d + f" {i}" for i in range(50) for d in DOCS]
    chain = python_configs(SPLIT)
    assert pf.apply(docs, chain, threads=threads) == pf.apply(docs, chain, threads=1)
    with pytest.raises(ValueError, match="threads"):
        pf.apply(docs, chain, threads=0)


def test_python_path_from_many_threads() -> None:
    # The Python path holds no lock while it calls the tokenizer: concurrent callers (on
    # free-threaded builds, truly in parallel) get the same results.
    docs = [d + f" {i}" for i in range(20) for d in DOCS]
    chain = python_configs(SPLIT)
    want = pf.apply(docs, chain)
    out: list[Any] = []
    errors: list[BaseException] = []

    def work() -> None:
        try:
            out.extend(pf.apply(docs, chain) for _ in range(5))
        except BaseException as e:  # re-raised below
            errors.append(e)

    ts = [threading.Thread(target=work) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=120)
    assert not errors, errors
    assert len(out) == 20
    assert all(r == want for r in out)


def test_python_tokenizer_on_native_language_is_identical() -> None:
    pytest.importorskip("datatrove.utils.word_tokenizers")
    pytest.importorskip("spacy")
    docs = [
        "Der schnelle braune Fuchs springt über den faulen Hund. Und dann läuft er weg.",
        "Straße, Größe, Äpfel – und die Katze schläft in der Sonne!\nNoch eine Zeile.",
        "z.B. 1.000,50 € am 3. Okt. 2024 (vgl. S. 12f.) ... https://beispiel.de/a?b=1",
    ]
    for cfg in [
        pf.GopherQuality(language="deu_Latn", min_doc_words=3),
        pf.GopherRepetition(language="deu_Latn"),
        pf.C4Quality(language="deu_Latn", min_num_sentences=1),
        pf.FineWebQuality(language="deu_Latn"),
    ]:
        assert pf.apply(docs, [cfg], python_tokenizer=True) == pf.apply(docs, [cfg])


# ---- live datatrove ------------------------------------------------------------------


class Datatrove:
    """datatrove's filters, one instance per config (as the parity harness builds them)."""

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
        self.wt: Any = pytest.importorskip("datatrove.utils.word_tokenizers")
        pytest.importorskip("spacy")
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

    def run(self, steps: Sequence[Any], text: str) -> Outcome:
        """Run a document through ``steps`` as datatrove's executor does."""
        doc = self.data.Document(text=text, id="0")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # numpy's "Mean of empty slice"
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


def candidate(steps: Sequence[Any], text: str, python_tokenizer: bool) -> Outcome:
    try:
        keep, reason, out = pf.apply(text, steps, python_tokenizer=python_tokenizer)
    except Exception as e:
        return ("raise", type(e).__name__)
    return ("ok", keep, reason, out)


def check(dt: Datatrove, steps: Sequence[Any], docs: list[str], python_tokenizer: bool) -> None:
    """pyroid's results for ``docs`` (batched, and per document) are datatrove's."""
    want = [dt.run(steps, d) for d in docs]
    for w in want:
        event(f"{type(steps[-1]).__name__}: {w[2] if w[0] == 'ok' else w[1]}")
    if all(w[0] == "ok" for w in want):
        res = pf.apply(docs, steps, python_tokenizer=python_tokenizer)
        got = [("ok", *r) for r in zip(*res, strict=True)]
        assert got == want, (steps, docs)
    else:
        for d, w in zip(docs, want, strict=True):
            assert candidate(steps, d, python_tokenizer) == w, (steps, d)


# ---- script-specific documents ----------------------------------------------------

#: Words per script / language family: diacritics, elisions, hyphens, abbreviations,
#: clitics, numbers in local formats, punctuation runs.
POOLS: dict[str, list[str]] = {
    "de": [
        "der",
        "die",
        "und",
        "Straße",
        "Größe",
        "Äpfel",
        "über",
        "z.B.",
        "u.a.",
        "bzw.",
        "Baden-Württemberg",
        "E-Mail",
        "1.000,50",
        "€",
        "3.",
        "Okt.",
        "geht's",
        "„Zitat“",
        "Haus-",
        "-tür",
        "S.",
        "vgl.",
        "GmbH",
        "Dr.",
        "ß",
        "Müller-Lüdenscheidt",
        "für",
    ],
    "fr": [
        "l'homme",
        "d'un",
        "aujourd'hui",
        "qu'il",
        "c'est",
        "l’école",
        "j’ai",
        "porte-monnaie",
        "peut-être",
        "dit-il",
        "«",
        "»",
        "M.",
        "Mme",
        "etc.",
        "1 000,5",
        "à",
        "où",
        "ça",
        "naïve",
        "Noël",
        "œuvre",
        "de",
        "la",
        "le",
        "et",
        "est-ce",
        "va-t-on",
    ],
    "es": [
        "¿Qué?",
        "¡Hola!",
        "niño",
        "año",
        "el",
        "la",
        "de",
        "que",
        "Sr.",
        "Sra.",
        "pág.",
        "1.234,56",
        "dámelo",
        "corazón",
        "pingüino",
        "¿",
        "¡",
        "EE.UU.",
        "del",
        "al",
    ],
    "it": [
        "dell'anno",
        "perché",
        "c'è",
        "l'Italia",
        "un'altra",
        "città",
        "più",
        "gli",
        "sig.",
        "po'",
        "nell'ambito",
        "è",
        "così",
        "quest'anno",
        "d'Italia",
    ],
    "pt": [
        "não",
        "coração",
        "dá-lhe",
        "fazê-lo",
        "São",
        "Paulo",
        "é",
        "à",
        "às",
        "pôr",
        "Sr.ª",
        "1.º",
        "n.º",
        "disse-me",
        "vê-se",
        "ação",
        "pão",
    ],
    "ca": [
        "col·lecció",
        "l·l",
        "per què",
        "d'això",
        "l'any",
        "dels",
        "pel",
        "anar-hi",
        "Barcelona",
        "també",
        "és",
        "què",
        "s'ha",
        "n'hi",
        "fer-ho",
        "il·lusió",
    ],
    "nl": [
        "'s-Hertogenbosch",
        "z'n",
        "m'n",
        "België",
        "één",
        "zo'n",
        "auto's",
        "café's",
        "het",
        "een",
        "van",
        "de",
        "bijv.",
        "d.w.z.",
        "enz.",
        "ruïne",
    ],
    "cyr": [
        "Москва",
        "в",
        "т.ч.",
        "т.е.",
        "и",
        "года",
        "г.",
        "ул.",
        "Україна",
        "ї",
        "ґ",
        "Київ",
        "България",
        "щ",
        "ъ",
        "по-русски",
        "что-то",
        "1 000,5",
        "№",
        "«ёлка»",
        "т.д.",
        "им.",
        "из-за",
        "п'ять",
        "м'ясо",
    ],
    "el": [
        "Ελλάδα",
        "ό,τι",
        "κ.λπ.",
        "τ.μ.",
        "αυτό;",
        "και",
        "του",
        "της",
        "Αθήνα",
        "ΐ",
        "ΰ",
        "στο",
        "π.Χ.",
        "μ.Χ.",
        "δεν",
        "είναι",
        "ς",
    ],
    "ar": [
        "العربية",
        "في",
        "من",
        "على",
        "إلى",
        "،",
        "؟",
        "٣٤٥",
        "كتاب",
        "والكتاب",
        "بالعربية",
        "ـ",
        "ﷲ",
        "١٢٫٥",
        "؛",
        "«قال»",
        "مدرسة",
        "الأمم",
        "المتحدة",
    ],
    "fa": [
        "فارسی",
        "می‌خواهم",
        "است",
        "که",
        "این",
        "۱۲۳",
        "کتاب‌ها",
        "ی",
        "ک",
        "،",
        "؟",
        "نمی‌دانم",
        "را",
        "و",
    ],
    "he": ["שלום", "ה־", "בית", "״", "׳", "ירושלים", "של", "ו", "ב", "צה״ל", "וכו׳"],
    "hy": ["Հայաստան", "։", "՞", "Երևան", "և", "է", "՝", "՛", "ու", "հայերեն", "«բառ»"],
    "ka": ["საქართველო", "და", "თბილისი", "არის", "ქართული", "ენა", "წელს", "„ციტატა“"],
    "uz": [
        "O'zbekiston",
        "oʻzbek",
        "va",
        "bilan",
        "Toshkent",
        "g'alaba",
        "shahar",
        "yil",
        "Ўзбекистон",
        "ва",
        "1-yanvar",
    ],
    "et": ["ኢትዮጵያ", "።", "፣", "ትግርኛ", "አማርኛ", "ቋንቋ", "ነው", "፤", "፡", "፩", "ኤርትራ", "እና"],
    "tr": [
        "İstanbul",
        "ıı",
        "evlerimizden",
        "Ankara'da",
        "1.000",
        "%50",
        "3,14",
        "ve",
        "bir",
        "Türkiye'nin",
        "değil",
        "çok",
        "şey",
        "ğ",
        "Dr.",
        "vb.",
        "Iğdır",
    ],
    "hu": [
        "Magyarország",
        "évben",
        "2021-ben",
        "15-én",
        "ő",
        "ű",
        "és",
        "a",
        "az",
        "hogy",
        "Budapesten",
        "stb.",
        "pl.",
        "kb.",
        "100-at",
        "1.-jén",
        "tíz",
    ],
    "hi": [
        "हिन्दी",
        "में",
        "है",
        "।",
        "१२३",
        "भारत",
        "के",
        "की",
        "और",
        "से",
        "क़",
        "डॉ.",
        "रु.",
        "१,००,०००",
        "॥",
        "नमस्ते",
    ],
    "zh": [
        "中文",
        "我们",
        "今天",
        "公园",
        "散步",
        "。",
        "，",
        "、",
        "的",
        "是",
        "在",
        "中华人民共和国",
        "北京",
        "１２３",
        "２０２４年",
        "New York",
        "iPhone",
        "“引号”",
        "《书名》",
        "！",
        "？",
    ],
    "ja": [
        "日本語",
        "東京",
        "は",
        "の",
        "首都",
        "です",
        "。",
        "、",
        "私",
        "学生",
        "カタカナ",
        "ひらがな",
        "２０２４年",
        "ｶﾀｶﾅ",
        "「引用」",
        "！",
        "？",
        "〜",
        "ー",
    ],
    "vi": [
        "Hà Nội",
        "Việt Nam",
        "thủ đô",
        "của",
        "là",
        "và",
        "những",
        "người",
        "Thành phố",
        "Hồ Chí Minh",
        "năm",
        "đã",
        "được",
        "tiếng Việt",
        "TP.",
        "1.000",
    ],
    "th": ["ภาษาไทย", "ฉัน", "รัก", "มาก", "กรุงเทพมหานคร", "และ", "ที่", "๑๒๓", "ๆ", "ฯ"],
}
SHARED = [
    "2024", "12,5", "1.000.000", "3.14", "50%", "#tag", "@user", "https://example.org/a?b=1",
    "www.example.com", "user@example.com", "...", "…", "!!!", "?!", "—", "–", "-", "(", ")",
    "[1]", "[edit]", '"', "'", "•", "😀", "👍🏽", "C++", "e-mail", "{", "}",
    "lorem ipsum", "javascript", "uses cookies", "terms of use",
]  # fmt: skip
#: Whitespace inside a line, line separators, sentence endings per script.
INLINE_WS = [" "] * 20 + ["  ", "\t", "\xa0", "　", " ", "​", "\x0c", "\r"]
LINE_SEPS = ["\n"] * 10 + ["\n\n"] * 3 + ["\r\n", "\n \n", "\x85", " ", " \n"]
ENDINGS: dict[str, list[str]] = {
    "ar": ["؟", ".", "۔", "!"],
    "fa": ["؟", ".", "!"],
    "hy": ["։", ".", "՞"],
    "et": ["።", ".", "፧"],
    "hi": ["।", ".", "॥"],
    "zh": ["。", "！", "？", "."],
    "ja": ["。", "！", "？", "."],
    "th": [" ", ".", "ฯ"],
}

NATIVE: dict[str, str] = {
    "deu_Latn": "de",
    "fra_Latn": "fr",
    "spa_Latn": "es",
    "ita_Latn": "it",
    "por_Latn": "pt",
    "cat_Latn": "ca",
    "nld_Latn": "nl",
    "rus_Cyrl": "cyr",
    "ukr_Cyrl": "cyr",
    "bul_Cyrl": "cyr",
    "ell_Grek": "el",
    "arb_Arab": "ar",
    "fas_Arab": "fa",
    "heb_Hebr": "he",
    "hye_Armn": "hy",
    "kat_Geor": "ka",
    "uzn_Latn": "uz",
    "amh_Ethi": "et",
    "tir_Ethi": "et",
    "tur_Latn": "tr",
    "hun_Latn": "hu",
    "hin_Deva": "hi",
    "cmn_Hani": "zh",
}
#: The languages of the ``python_tokenizer=True`` tests: those whose datatrove tokenizer needs a
#: segmenter (the modules), each with its word pool. Vietnamese is native too, Thai and Japanese
#: when pyroid finds PyThaiNLP or a SudachiDict (which datatrove needs too).
PYTHON_PATH: dict[str, tuple[str, tuple[str, ...]]] = {
    "vie_Latn": ("vi", ("pyvi",)),
    # crfcut, PyThaiNLP's sentence splitter, needs python-crfsuite.
    "tha_Thai": ("th", ("pythainlp", "pycrfsuite")),
    "jpn_Jpan": ("ja", ("sudachipy", "sudachidict_core")),
}
_OTHER = [w for pool in ("de", "cyr", "ar", "zh", "el", "hi") for w in POOLS[pool][:6]]


def test_test_languages_are_what_they_say() -> None:
    langs = pf.languages()
    assert all(langs[c].native for c in NATIVE)
    assert langs["vie_Latn"].native
    assert langs["tha_Thai"].native is pythainlp_corpus()
    assert langs["jpn_Jpan"].native is sudachi_dictionary()


def _words(pool: str) -> st.SearchStrategy[str]:
    return st.one_of(
        st.sampled_from(POOLS[pool]),
        st.sampled_from(POOLS[pool]),
        st.sampled_from(POOLS[pool]),
        st.sampled_from(SHARED),
        st.sampled_from(_OTHER),  # mixed scripts
        st.text(st.sampled_from("".join(POOLS[pool])), min_size=1, max_size=8),
        st.text(st.characters(categories=["L", "M", "N", "P", "S"]), min_size=1, max_size=4),
    )


@st.composite
def lines(draw: st.DrawFn, pool: str) -> str:
    ws = draw(st.lists(_words(pool), max_size=18))
    seps = draw(st.lists(st.sampled_from(INLINE_WS), min_size=len(ws), max_size=len(ws)))
    line = "".join(x for w, s in zip(ws, seps, strict=True) for x in (s, w))[1:]
    ends = ENDINGS.get(pool, [".", "!", "?", "...", ":"])
    line += draw(st.sampled_from([*ends, *ends, ".", "", '."', ".[1]", "…"]))
    if draw(st.integers(0, 7)) == 0:
        line = draw(st.sampled_from(INLINE_WS)) + line
    return line


@st.composite
def documents(draw: st.DrawFn, pool: str) -> str:
    """0 to ~20 lines of script-specific words, sometimes repeated (repetition checks)."""
    shape = draw(st.integers(0, 9))
    if shape == 0:
        return draw(st.sampled_from(["", " ", "\n", "\n\n", "　", "\xa0"]))
    if shape == 1:  # one phrase over and over
        phrase = " ".join(draw(st.lists(_words(pool), min_size=1, max_size=5)))
        n = draw(st.integers(1, 30))
        return draw(st.sampled_from([" ", "\n", ". ", "\n\n"])).join([phrase] * n)
    pool_lines = draw(st.lists(lines(pool), min_size=1, max_size=20))
    picks = list(range(len(pool_lines)))
    for _ in range(draw(st.sampled_from([0, 0, 1, 3, 8]))):
        picks.insert(draw(st.integers(0, len(picks))), draw(st.integers(0, len(pool_lines) - 1)))
    seps = draw(st.lists(st.sampled_from(LINE_SEPS), min_size=len(picks), max_size=len(picks)))
    return "".join(pool_lines[i] + s for i, s in zip(picks, seps, strict=True)).rstrip("\n")


def configs(kind: str, lang: Any, stops: list[str]) -> list[Any]:
    """A spread of configs per filter, reaching every check."""
    if kind == "gq":
        return [
            pf.GopherQuality(language=lang),
            pf.GopherQuality(language=lang, min_doc_words=3, stop_words=stops, min_stop_words=1),
            pf.GopherQuality(
                language=lang,
                min_doc_words=None,
                min_avg_word_length=1,
                max_avg_word_length=15,
                max_non_alpha_words_ratio=0.5,
                stop_words=stops,
                min_stop_words=2,
            ),
        ]
    if kind == "gr":
        return [
            pf.GopherRepetition(language=lang),
            pf.GopherRepetition(
                language=lang,
                dup_line_frac=None,
                dup_para_frac=None,
                dup_line_char_frac=None,
                dup_para_char_frac=None,
                top_n_grams=((1, 0.2), (2, 0.1), (3, 0.1)),
                dup_n_grams=((2, 0.1), (4, 0.05)),
            ),
        ]
    if kind == "fw":
        return [
            pf.FineWebQuality(language=lang),
            pf.FineWebQuality(
                language=lang,
                short_line_thr=999,
                char_duplicates_ratio=0.1,
                line_punct_thr=0.1,
                new_line_ratio=0.3,
            ),
            pf.FineWebQuality(language=lang, line_punct_thr=0, new_line_ratio=0.05),
        ]
    return [
        pf.C4Quality(language=lang),
        pf.C4Quality(language=lang, split_paragraph=False),
        pf.C4Quality(language=lang, min_num_sentences=1, filter_no_terminal_punct=False),
        pf.C4Quality(
            language=lang, split_paragraph=False, min_num_sentences=2, min_words_per_line=1
        ),
    ]


def _need(lang: str) -> None:
    if pf.languages()[lang].tokenizer == "IndicNLPTokenizer":
        pytest.importorskip("indicnlp")  # datatrove's reference needs it
    if pf.languages()[lang].arg == "zh":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # jieba: invalid escapes
            pytest.importorskip("jieba")  # datatrove's reference needs it
    if pf.languages()[lang].arg == "ja":
        pytest.importorskip("sudachipy")  # datatrove's reference needs SudachiPy and its dictionary
        pytest.importorskip("sudachidict_core")
    if pf.languages()[lang].arg == "vi":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # pyvi: invalid escapes
            pytest.importorskip("pyvi")  # datatrove's reference needs pyvi
    if pf.languages()[lang].tokenizer == "ThaiTokenizer":
        pytest.importorskip("pythainlp")  # and PyThaiNLP, whose sentences need python-crfsuite
        pytest.importorskip("pycrfsuite")


@pytest.mark.parametrize("kind", ["gq", "gr", "fw", "c4"])
@examples
@given(data=st.data())
def test_native_languages_match_datatrove(dt: Datatrove, kind: str, data: st.DataObject) -> None:
    lang = data.draw(st.sampled_from(sorted(NATIVE)), label="language")
    _need(lang)
    pool = NATIVE[lang]
    cfg = data.draw(st.sampled_from(configs(kind, lang, POOLS[pool][:4])), label="config")
    docs = data.draw(st.lists(documents(pool), min_size=1, max_size=4), label="docs")
    check(dt, [cfg], docs, python_tokenizer=False)


CHAINS: list[Callable[[Any, list[str]], list[Any]]] = [
    # FineWeb-2's order and settings shape.
    lambda lang, stops: [
        pf.GopherRepetition(
            language=lang, dup_para_frac=0, dup_line_char_frac=0, dup_para_char_frac=0
        ),
        pf.FineWebQuality(language=lang, short_line_thr=999, char_duplicates_ratio=0.1),
        pf.GopherQuality(language=lang, stop_words=stops, min_stop_words=2, min_doc_words=5),
    ],
    # C4 rewrites the text the later steps tokenize.
    lambda lang, stops: [
        pf.C4Quality(language=lang, min_num_sentences=1, filter_no_terminal_punct=False),
        pf.GopherQuality(language=lang, min_doc_words=3, stop_words=stops, min_stop_words=1),
        pf.FineWebQuality(language=lang),
    ],
    lambda lang, stops: [
        pf.GopherRepetition(language=lang, dup_line_frac=0.5),
        pf.C4Quality(language=lang, split_paragraph=False, min_num_sentences=1),
        pf.C4Quality(language=lang, min_num_sentences=1),
        pf.GopherQuality(language=lang, min_doc_words=None),
    ],
]


@examples
@given(data=st.data())
def test_native_chains_match_datatrove(dt: Datatrove, data: st.DataObject) -> None:
    lang = data.draw(st.sampled_from(sorted(NATIVE)), label="language")
    _need(lang)
    pool = NATIVE[lang]
    chain = data.draw(st.sampled_from(CHAINS))(lang, POOLS[pool][:4])
    docs = data.draw(st.lists(documents(pool), min_size=1, max_size=4), label="docs")
    check(dt, chain, docs, python_tokenizer=False)


def installed(modules: Sequence[str]) -> bool:
    """Whether ``modules`` import (quietly: segmenters warn on import)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            for module in modules:
                __import__(module)
        except ImportError:
            return False
    return True


@pytest.fixture(scope="module")
def python_langs() -> list[str]:
    """The ``PYTHON_PATH`` languages whose segmenter (and dictionary) is installed."""
    out = [lang for lang, (_, modules) in PYTHON_PATH.items() if installed(modules)]
    if not out:
        pytest.skip("no segmenter installed (sudachipy + sudachidict-core, pyvi, pythainlp)")
    return out


@pytest.mark.parametrize("kind", ["gq", "gr", "fw", "c4"])
@examples
@given(data=st.data())
def test_python_tokenizer_matches_datatrove(
    dt: Datatrove, python_langs: list[str], kind: str, data: st.DataObject
) -> None:
    lang = data.draw(st.sampled_from(python_langs), label="language")
    pool = PYTHON_PATH[lang][0]
    cfg = data.draw(st.sampled_from(configs(kind, lang, POOLS[pool][:4])), label="config")
    docs = data.draw(st.lists(documents(pool), min_size=1, max_size=3), label="docs")
    check(dt, [cfg], docs, python_tokenizer=True)


@examples
@given(data=st.data())
def test_python_tokenizer_chains_match_datatrove(
    dt: Datatrove, python_langs: list[str], data: st.DataObject
) -> None:
    lang = data.draw(st.sampled_from(python_langs), label="language")
    pool = PYTHON_PATH[lang][0]
    chain = data.draw(st.sampled_from(CHAINS))(lang, POOLS[pool][:4])
    docs = data.draw(st.lists(documents(pool), min_size=1, max_size=3), label="docs")
    check(dt, chain, docs, python_tokenizer=True)


def test_mixed_native_and_python_steps(dt: Datatrove, python_langs: list[str]) -> None:
    """Native steps and Python-tokenizer steps in one chain, in datatrove's order."""
    lang = python_langs[0]
    tok = dt.wt.load_word_tokenizer(lang)  # a datatrove WordTokenizer instance
    pool = PYTHON_PATH[lang][0]
    docs = [
        "\n".join(" ".join(POOLS[pool][i : i + 6]) + "." for i in range(0, 12, 2)),
        "Ein Satz mit Wörtern. Und noch einer!\n" + " ".join(POOLS[pool]) + ".",
        "",
    ]
    chain = [
        pf.C4Quality(language="deu_Latn", min_num_sentences=1, filter_no_terminal_punct=False),
        pf.GopherQuality(language=tok, min_doc_words=2),
        pf.FineWebQuality(language="fra_Latn"),
        pf.GopherRepetition(language=tok),
    ]
    want = [dt.run(chain, d) for d in docs]
    assert [candidate(chain, d, python_tokenizer=False) for d in docs] == want
    res = pf.apply(docs, chain)
    assert [("ok", *r) for r in zip(*res, strict=True)] == want
    # Without the instance, a non-native code needs the opt-in; a native one (vie_Latn, and
    # jpn_Jpan or tha_Thai with their data) gives the instance's results.
    by_code = [chain[0], pf.GopherQuality(language=lang, min_doc_words=2)]
    if pf.languages()[lang].native:
        assert pf.apply(docs, by_code) == pf.apply(docs, [chain[0], chain[1]])
    else:
        with pytest.raises(NotImplementedError):
            pf.apply(docs, by_code)


def test_tokenizer_objects_match_datatrove(dt: Datatrove) -> None:
    """``language=`` a datatrove WordTokenizer, or a duck-typed object, as datatrove allows."""

    class SplitWT(dt.wt.WordTokenizer):  # type: ignore[misc, name-defined]
        def word_tokenize(self, text: str) -> list[str]:
            return SPLIT.word_tokenize(text)

        def sent_tokenize(self, text: str) -> list[str]:
            return [text[a:b] for a, b in SPLIT.span_tokenize(text)]

        def span_tokenize(self, text: str) -> list[tuple[int, int]]:
            return SPLIT.span_tokenize(text)

    docs = [*DOCS, "One. Two three four. Five six seven eight.\nAnd a second line here."]
    for duck, wt in ((SPLIT, SplitWT()), (dt.wt.load_word_tokenizer("deu_Latn"),) * 2):
        for kind in ("gq", "gr", "fw", "c4"):
            for cfg in configs(kind, duck, ["the", "and"]):
                ref = cfg._replace(language=wt)
                want = [dt.run([ref], d) for d in docs]
                got = [candidate([cfg], d, python_tokenizer=False) for d in docs]
                assert got == want, cfg


# ---- the FineWeb-2 slices ------------------------------------------------------------

DATA = Path(os.environ.get("PYROID_PARITY_DATA", "/data"))

#: FineWeb-2's per-language filter thresholds, from
#: https://github.com/huggingface/fineweb-2/tree/main/configs (<lang>.yml, Apache-2.0),
#: as fineweb-2-pipeline.py passes them to datatrove's filters.
FINEWEB2: dict[str, dict[str, Any]] = {
    "arb_Arab": {
        "dup_line_frac": 0.304,
        "top_n_grams": ((2, 0.197), (3, 0.172), (4, 0.146)),
        "dup_n_grams": ((5, 0.165), (6, 0.153), (7, 0.142), (8, 0.131), (9, 0.12), (10, 0.109)),
        "line_punct_thr": 0.143,
        "new_line_ratio": 0.189,
        "max_avg_word_length": 9,
        "min_avg_word_length": 2,
        "max_non_alpha_words_ratio": 0.787,
        "stopwords": [
            "،", "في", "من", "على", "إلى", "عام", "أن", "مع", "أو", "هو", "عن", "التي",
            "كان", "بين", "ما", "كانت", "هي", "المتحدة", "بعد",
        ],
    },
    "cmn_Hani": {
        "dup_line_frac": 0.287,
        "top_n_grams": ((2, 0.256), (3, 0.201), (4, 0.171)),
        "dup_n_grams": ((5, 0.198), (6, 0.182), (7, 0.167), (8, 0.154), (9, 0.14), (10, 0.127)),
        "line_punct_thr": 0.106,
        "new_line_ratio": 0.178,
        "max_avg_word_length": 5,
        "min_avg_word_length": 1,
        "max_non_alpha_words_ratio": 0.735,
        "stopwords": [
            "的", "年", "在", "月", "是", "和", "日", "了", "於", "·", "為", "有", "被", "人",
            "中", "为", "他", "與", "後", "也", "而", "由",
        ],
    },
    "deu_Latn": {
        "dup_line_frac": 0.263,
        "top_n_grams": ((2, 0.169), (3, 0.153), (4, 0.137)),
        "dup_n_grams": ((5, 0.141), (6, 0.134), (7, 0.126), (8, 0.117), (9, 0.108), (10, 0.098)),
        "line_punct_thr": 0.0,
        "new_line_ratio": 0.152,
        "max_avg_word_length": 15,
        "min_avg_word_length": 0,
        "max_non_alpha_words_ratio": 0.774,
        "stopwords": [
            "der", "und", "die", "in", "von", "im", "den", "des", "mit", "das", "er", "dem",
            "als", "wurde", "für",
        ],
    },
    "fra_Latn": {
        "dup_line_frac": 0.264,
        "top_n_grams": ((2, 0.161), (3, 0.149), (4, 0.134)),
        "dup_n_grams": ((5, 0.15), (6, 0.141), (7, 0.131), (8, 0.121), (9, 0.111), (10, 0.1)),
        "line_punct_thr": 0.1,
        "new_line_ratio": 0.185,
        "max_avg_word_length": 11,
        "min_avg_word_length": 2,
        "max_non_alpha_words_ratio": 0.812,
        "stopwords": [
            "de", "la", "le", "et", "à", "en", "l'", "des", "du", "les", "est", "d'", "un",
            "une", "il", "dans", "par", "au",
        ],
    },
    "hin_Deva": {
        "dup_line_frac": 0.206,
        "top_n_grams": ((2, 0.208), (3, 0.198), (4, 0.161)),
        "dup_n_grams": ((5, 0.135), (6, 0.125), (7, 0.116), (8, 0.108), (9, 0.099), (10, 0.09)),
        "line_punct_thr": 0.091,
        "new_line_ratio": 0.316,
        "max_avg_word_length": 21,
        "min_avg_word_length": 2,
        "max_non_alpha_words_ratio": 0.837,
        "stopwords": ["के", "में", "है", "की", "और", "से", "का", "को", "एक", "हैं", "पर"],
    },
    "jpn_Jpan": {
        "dup_line_frac": 0.328,
        "top_n_grams": ((2, 0.239), (3, 0.196), (4, 0.172)),
        "dup_n_grams": ((5, 0.243), (6, 0.225), (7, 0.207), (8, 0.19), (9, 0.175), (10, 0.159)),
        "line_punct_thr": 0.096,
        "new_line_ratio": 0.13,
        "max_avg_word_length": 6,
        "min_avg_word_length": 1,
        "max_non_alpha_words_ratio": 0.759,
        "stopwords": [
            "の", "に", "を", "は", "た", "て", "が", "と", "で", "年", "し", "・", "月", "れ",
            "さ",
        ],
    },
    "rus_Cyrl": {
        "dup_line_frac": 0.322,
        "top_n_grams": ((2, 0.184), (3, 0.164), (4, 0.146)),
        "dup_n_grams": ((5, 0.168), (6, 0.156), (7, 0.145), (8, 0.133), (9, 0.121), (10, 0.109)),
        "line_punct_thr": 0.231,
        "new_line_ratio": 0.139,
        "max_avg_word_length": 11,
        "min_avg_word_length": 3,
        "max_non_alpha_words_ratio": 0.713,
        "stopwords": [
            "в", "и", "на", "с", "года", "по", "году", "из", "был", "к", "не", "от", "что",
            "за", "для", "его", "как", "а", "он", "также", "до", "после",
        ],
    },
    "tha_Thai": {
        "dup_line_frac": 0.349,
        "top_n_grams": ((2, 0.221), (3, 0.197), (4, 0.162)),
        "dup_n_grams": ((5, 0.185), (6, 0.168), (7, 0.152), (8, 0.137), (9, 0.124), (10, 0.111)),
        "line_punct_thr": 0.0,
        "new_line_ratio": 0.153,
        "max_avg_word_length": 10,
        "min_avg_word_length": 2,
        "max_non_alpha_words_ratio": 0.9,
        "stopwords": ["ใน", "ที่", "และ", "ของ", "เป็น", "มี", "การ", "ได้"],
    },
    "vie_Latn": {
        "dup_line_frac": 0.293,
        "top_n_grams": ((2, 0.223), (3, 0.197), (4, 0.171)),
        "dup_n_grams": ((5, 0.178), (6, 0.163), (7, 0.15), (8, 0.136), (9, 0.124), (10, 0.111)),
        "line_punct_thr": 0.333,
        "new_line_ratio": 0.103,
        "max_avg_word_length": 11,
        "min_avg_word_length": 3,
        "max_non_alpha_words_ratio": 0.803,
        "stopwords": [
            "là", "và", "của", "một", "được", "trong", "năm", "các", "có", "loài", "này", "đã",
            "người",
        ],
    },
    "spa_Latn": {
        "dup_line_frac": 0.252,
        "top_n_grams": ((2, 0.156), (3, 0.14), (4, 0.125)),
        "dup_n_grams": ((5, 0.145), (6, 0.136), (7, 0.127), (8, 0.117), (9, 0.108), (10, 0.099)),
        "line_punct_thr": 0.182,
        "new_line_ratio": 0.138,
        "max_avg_word_length": 14,
        "min_avg_word_length": 3,
        "max_non_alpha_words_ratio": 0.825,
        "stopwords": [
            "de", "la", "en", "el", "y", "a", "que", "del", "los", "se", "por", "un", "con",
            "una", "las", "su",
        ],
    },
}  # fmt: skip


def fineweb2_steps(lang: str) -> list[Any]:
    """fineweb-2-pipeline.py's three filters for ``lang``, in its order."""
    c = FINEWEB2[lang]
    return [
        pf.GopherRepetition(
            language=lang,
            dup_para_frac=0,
            dup_line_char_frac=0,
            dup_para_char_frac=0,
            dup_line_frac=c["dup_line_frac"],
            top_n_grams=c["top_n_grams"],
            dup_n_grams=c["dup_n_grams"],
        ),
        pf.FineWebQuality(
            language=lang,
            short_line_thr=999,
            char_duplicates_ratio=0.1,
            line_punct_thr=c["line_punct_thr"],
            new_line_ratio=c["new_line_ratio"],
        ),
        pf.GopherQuality(
            language=lang,
            max_avg_word_length=c["max_avg_word_length"],
            min_avg_word_length=c["min_avg_word_length"],
            stop_words=c["stopwords"],
            max_non_alpha_words_ratio=c["max_non_alpha_words_ratio"],
            min_stop_words=2,
        ),
    ]


def fineweb2_texts(lang: str) -> list[str]:
    pq: Any = pytest.importorskip("pyarrow.parquet")
    path = DATA / "fineweb2" / f"{lang}.parquet"
    if not path.is_file():
        pytest.skip(f"FineWeb-2 slice not found: {path}")
    texts: list[str] = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    return texts


@pytest.mark.parametrize("lang", sorted(FINEWEB2))
def test_fineweb2_slices_match_datatrove(dt: Datatrove, lang: str) -> None:
    texts = fineweb2_texts(lang)
    python = not pf.languages()[lang].native
    if python:
        module = {"cmn_Hani": "jieba", "jpn_Jpan": "sudachipy", "tha_Thai": "pythainlp"}[lang]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pytest.importorskip(module)
    _need(lang)
    steps = fineweb2_steps(lang)
    # The pipeline, then each filter on every document (so later filters see them all).
    for chain in [steps, *([s] for s in steps)]:
        want = [dt.run(chain, t) for t in texts]
        assert all(w[0] == "ok" for w in want), lang
        res = pf.apply(texts, chain, python_tokenizer=python)
        got = [("ok", *r) for r in zip(*res, strict=True)]
        bad = [(i, g, w) for i, (g, w) in enumerate(zip(got, want, strict=True)) if g != w]
        assert not bad, f"{len(bad)}/{len(texts)} differ for {lang} {chain}: {bad[:2]}"
    # FineWeb-2 kept these documents; most still pass its filters.
    kept = pf.apply(texts, steps, python_tokenizer=python).keep
    assert sum(kept) > len(texts) // 2
