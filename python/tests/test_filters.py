# Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace; modified.
"""pyroid.filters: datatrove's decisions on hand-written documents, container kinds,
configs, errors and threads.

The expected values are worked out from datatrove 0.10.1's filter code
(``pipeline/filters/{gopher_quality,gopher_repetition,c4,fineweb_quality}_filter.py``),
so this file needs no datatrove and runs on every interpreter, free-threaded ones
included; test_filters_parity.py compares with datatrove itself. Every reason string
each filter can give is reached at least once.
"""

# Look-alike characters (KELVIN SIGN, dotted capital I, fullwidth punctuation) on purpose.
# ruff: noqa: RUF001

from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Any

import pytest

import pyroid
import pyroid.filters as pf
from pyroid._pyroid import _testing

# ---- documents ---------------------------------------------------------------

#: Six well-formed lines: every filter keeps this document with its defaults, and C4
#: keeps every line, so its text comes back unchanged.
GOOD_LINES = (
    "The committee met on Tuesday to review the annual budget and the new hiring plan.",
    "Several members asked whether the proposed changes would affect existing projects.",
    "After a long discussion, the chair agreed to publish a detailed summary next week.",
    "Local residents have been invited to share their views with the council by email.",
    "A final decision on the budget is expected before the end of the month.",
    "Meanwhile, work on the bridge repairs will continue as planned through the winter.",
)
GOOD = "\n".join(GOOD_LINES)

#: 20 lines that all start with a bullet (``•`` or ``-``, some indented).
BULLETS = "\n".join(
    f"{'  ' if i % 3 == 0 else ''}{'•' if i % 2 else '-'} bullet point number {i} about the garden"
    for i in range(20)
)
_GARDEN = "the garden was full of flowers and the birds sang all day long"
#: 4 of 10 lines end with an ellipsis, but there are few ellipses per word.
END_ELLIPSIS = "\n".join(
    [f"{_GARDEN} {i}." for i in range(6)]
    + [f"{_GARDEN} ...", f"{_GARDEN}...", f"{_GARDEN} …", f"{_GARDEN}…  "]
)
LONG_PARA = "This paragraph is long and it is repeated once in the document."
#: 30 distinct words, said twice: every 5-gram of the second copy is a repeat.
RIVER = (
    "The river runs quietly past old mills, green fields and narrow lanes before it "
    "reaches the busy harbour town where fishing boats wait for the evening tide"
)
#: One 3-gram twice (top 3-gram fraction 32/114) among unique words (top 2-gram 20/114).
TOP3 = (
    "alpha beta gamma one two three four five six seven alpha beta gamma "
    "eight nine ten eleven twelve thirteen fourteen"
)
#: One 4-gram twice (44/190), its 3-grams 32/190, its 2-grams 20/190.
TOP4 = (
    "alpha beta gamma delta one two three four five six seven eight nine ten "
    "alpha beta gamma delta eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen twenty thirty forty"
)
#: Lines ending in terminal punctuation from other scripts (datatrove's TERMINAL_PUNCTUATION).
UNICODE_STOPS = "\n".join(
    [
        "This line ends with an ideographic full stop。",
        "This line ends with a fullwidth exclamation mark！",
        "This line ends with an Arabic question mark؟",
        "This line ends with a Devanagari danda sign।",
    ]
)


def _case(doc: str, reason: str | None, /, id: str, **kwargs: Any) -> Any:
    return pytest.param(doc, kwargs, reason, id=id)


# (doc, kwargs, reason): the reason is datatrove's, None when it keeps the document.
GOPHER_QUALITY = [
    _case(GOOD, None, id="kept"),
    _case("Far too short to be a document.", "gopher_short_doc", id="short_doc"),
    # Words made only of punctuation do not count towards min_doc_words.
    _case(" ".join(["house"] * 49 + ["..."] * 20), "gopher_short_doc", id="symbols_dont_count"),
    _case(GOOD, "gopher_long_doc", id="long_doc", max_doc_words=20),
    _case(" ".join(["an"] * 60), "gopher_below_avg_threshold", id="below_avg"),
    _case(" ".join(["extraordinarily"] * 60), "gopher_above_avg_threshold", id="above_avg"),
    _case(" ".join(["house"] * 50 + ["#"] * 10), "gopher_too_many_hashes", id="hashes"),
    _case(
        " ".join(["house"] * 50 + ["..."] * 4 + ["…"] * 4),
        "gopher_too_many_ellipsis",
        id="ellipsis",
    ),
    _case(BULLETS, "gopher_too_many_bullets", id="bullets"),
    _case(END_ELLIPSIS, "gopher_too_many_end_ellipsis", id="end_ellipsis"),
    _case(" ".join(["house"] * 50 + ["123"] * 20), "gopher_below_alpha_threshold", id="alpha"),
    _case(" ".join(["house"] * 60), "gopher_enough_stop_words", id="stop_words"),
    _case(
        " ".join(["house", "garden"] * 30),
        None,
        id="own_stop_words",
        stop_words=["house", "garden"],
    ),
    _case(
        " ".join(["house", "garden"] * 30),
        "gopher_enough_stop_words",
        id="one_stop_word_is_not_enough",
        stop_words=["house", "x"],
    ),
    _case(" ".join(["house"] * 60), None, id="stop_words_off", min_stop_words=None),
    _case(" ".join(["house"] * 60), None, id="zero_disables", min_stop_words=0),
    # Bullet lines off: "-", "•" and the numbers are 2 of 8 words without a letter.
    _case(BULLETS, "gopher_below_alpha_threshold", id="bullets_off", max_bullet_lines_ratio=None),
    # No word has a letter: the mean word length is NaN, which passes both checks.
    _case("...", "gopher_too_many_ellipsis", id="nan_mean", min_doc_words=None),
    _case(
        "!!! ???",
        None,
        id="nan_mean_kept",
        min_doc_words=None,
        max_non_alpha_words_ratio=None,
        min_stop_words=None,
    ),
]

GOPHER_REPETITION = [
    _case(GOOD, None, id="kept"),
    _case("", "empty", id="empty"),
    _case(
        "First paragraph here.\n\nFirst paragraph here.\n\nA different paragraph.",
        "dup_para_frac",
        id="dup_para_frac",
    ),
    _case(
        f"{LONG_PARA}\n\n{LONG_PARA}\n\nShort one.\n\nShort two.",
        "dup_para_char_frac",
        id="dup_para_char_frac",
    ),
    _case("same line here\nsame line here\nanother line", "dup_line_frac", id="dup_line_frac"),
    # "\n \n" is not a paragraph break (two newlines in a row are), but splits lines.
    _case("Same.\n \nSame.", "dup_line_frac", id="para_break_is_two_newlines"),
    _case(
        f"{LONG_PARA}\n{LONG_PARA}\nShort one.\nShort two.",
        "dup_line_char_frac",
        id="dup_line_char_frac",
    ),
    _case("word word word word", "top_2_gram", id="top_2_gram"),
    _case(TOP3, "top_3_gram", id="top_3_gram"),
    _case(TOP4, "top_4_gram", id="top_4_gram"),
    _case(f"{RIVER} {RIVER}", "duplicated_5_n_grams", id="duplicated_5"),
    _case(f"{RIVER} {RIVER}", "duplicated_7_n_grams", id="duplicated_7", dup_n_grams=((7, 0.13),)),
    _case("aaa aaa aaa b", "top_1_gram", id="top_1_gram", top_n_grams=((1, 0.5),)),
    _case(
        "same\nsame\nsame",
        None,
        id="all_off",
        dup_line_frac=None,
        dup_para_frac=None,
        dup_line_char_frac=0,
        dup_para_char_frac=0,
        top_n_grams=(),
        dup_n_grams=[],
    ),
]

_JS = "Please enable JavaScript to view this page."
C4_CASES: list[Any] = [
    # (doc, kwargs, reason, text after the filter)
    pytest.param(GOOD, {}, None, GOOD, id="kept_unchanged"),
    pytest.param(
        GOOD + "\nLorem ipsum dolor sit amet, consectetur.", {}, "lorem_ipsum", None, id="lorem"
    ),
    pytest.param("LOREM IPSUM DOLOR SIT AMET.\n" + GOOD, {}, "lorem_ipsum", None, id="lorem_upper"),
    # "İ".lower() is "i̇" (two code points), so this is not "lorem ipsum".
    pytest.param(GOOD + "\nLOREM İPSUM DOLOR SIT AMET.", {}, None, None, id="dotted_capital_i"),
    pytest.param(
        GOOD + "\nThe template uses {braces} in this line.", {}, "curly_bracket", None, id="curly"
    ),
    # The line without terminal punctuation is removed before the bracket check.
    pytest.param(GOOD + "\nfunction() { return 1 }", {}, None, GOOD, id="curly_in_removed_line"),
    pytest.param("Only one short sentence is here.", {}, "too_few_sentences", None, id="few"),
    pytest.param(GOOD + "\n" + _JS, {}, None, GOOD, id="javascript_line"),
    pytest.param(
        GOOD + "\nBy continuing you accept our Terms of Use.", {}, None, GOOD, id="policy_line"
    ),
    # "K" (KELVIN SIGN).lower() is "k": "uses cookies".
    pytest.param(GOOD + "\nThis site uses cooKies to improve things.", {}, None, GOOD, id="kelvin"),
    pytest.param(
        GOOD + "\nWater boils at one hundred degrees[1] at sea level[edit].[citation needed]",
        {},
        None,
        GOOD + "\nWater boils at one hundred degrees at sea level.",
        id="citations",
    ),
    pytest.param(
        GOOD + "\nWater boils at one hundred degrees.[2]",
        {"remove_citations": False},
        None,
        GOOD,
        id="citations_kept_fail_punct",
    ),
    pytest.param(
        "Menu Home About\n"
        + GOOD
        + "\nTo be continued...\nYes.\n   Indented line that ends well.   \n",
        {},
        None,
        GOOD + "\nIndented line that ends well.",
        id="line_removal",
    ),
    pytest.param(
        "\n".join([*GOOD_LINES[:2], 'She said "this ends in a quote."']),
        {"min_num_sentences": 3},
        None,
        None,
        id="quote_is_terminal",
    ),
    pytest.param(GOOD.replace("\n", "\r\n"), {}, None, GOOD, id="crlf"),
    pytest.param(
        GOOD,
        {"max_word_length": 10},
        None,
        "\n".join(GOOD_LINES[:2] + GOOD_LINES[3:]),
        id="max_word_length",
    ),
    pytest.param("Short.", {"min_num_sentences": -1}, None, "", id="min_sentences_off"),
    pytest.param(
        "Menu Home About\n" + GOOD,
        {"filter_no_terminal_punct": False},
        None,
        None,
        id="fineweb_config",
    ),
    pytest.param(GOOD, {"split_paragraph": False}, None, " ".join(GOOD_LINES), id="sentences"),
    pytest.param(
        GOOD + "\nLorem ipsum dolor sit amet, consectetur.\n" + _JS,
        {"filter_lorem_ipsum": False, "filter_javascript": False},
        None,
        None,
        id="flags_off",
    ),
]

FINEWEB = [
    _case(GOOD, None, id="kept"),
    _case("", "empty", id="empty"),
    _case("\n \n\t\n", "empty", id="blank_lines"),
    _case(
        "no punctuation at the end of this line\nnor at the end of this line here",
        "line_punct_ratio",
        id="line_punct_ratio",
    ),
    _case(
        "no punctuation at the end of this line\nnor at the end of this line here",
        None,
        id="line_punct_exclude_zero",
        line_punct_exclude_zero=True,
    ),
    # Lines are not stripped: a trailing space or "\r" hides the full stop.
    _case(GOOD.replace("\n", " \n"), "line_punct_ratio", id="trailing_space", line_punct_thr=0.2),
    _case(GOOD.replace("\n", "\r\n"), "line_punct_ratio", id="crlf", line_punct_thr=0.2),
    _case("Short.\nAlso short.\nTiny one.", "short_line_ratio", id="short_line_ratio"),
    _case(GOOD + "\n" + GOOD_LINES[0], "char_dup_ratio", id="char_dup_ratio"),
    _case("This is a single sentence that is long enough." + "\n" * 20, "list_ratio", id="list"),
    _case(GOOD, "line_punct_ratio", id="own_stop_chars", stop_chars=("!", "?")),
    _case(GOOD, None, id="stop_chars_str", stop_chars="."),
    _case(UNICODE_STOPS, None, id="unicode_stops"),
    _case(UNICODE_STOPS, "line_punct_ratio", id="unicode_stops_off", stop_chars=(".",)),
]


@pytest.mark.parametrize(("doc", "kwargs", "reason"), GOPHER_QUALITY)
def test_gopher_quality(doc: str, kwargs: dict[str, Any], reason: str | None) -> None:
    assert pf.gopher_quality_reasons(doc, **kwargs) == reason
    assert pf.gopher_quality(doc, **kwargs) is (reason is None)


@pytest.mark.parametrize(("doc", "kwargs", "reason"), GOPHER_REPETITION)
def test_gopher_repetition(doc: str, kwargs: dict[str, Any], reason: str | None) -> None:
    assert pf.gopher_repetition_reasons(doc, **kwargs) == reason
    assert pf.gopher_repetition(doc, **kwargs) is (reason is None)


@pytest.mark.parametrize(("doc", "kwargs", "reason", "text"), C4_CASES)
def test_c4_quality(doc: str, kwargs: dict[str, Any], reason: str | None, text: str | None) -> None:
    assert pf.c4_quality_reasons(doc, **kwargs) == reason
    assert pf.c4_quality(doc, **kwargs) is (reason is None)
    out = pf.c4_lines(doc, **kwargs)
    if text is None or text == doc:
        assert out is doc  # dropped, or every line kept as it was
    else:
        assert out == text


@pytest.mark.parametrize(("doc", "kwargs", "reason"), FINEWEB)
def test_fineweb_quality(doc: str, kwargs: dict[str, Any], reason: str | None) -> None:
    assert pf.fineweb_quality_reasons(doc, **kwargs) == reason
    assert pf.fineweb_quality(doc, **kwargs) is (reason is None)


def test_every_reason_is_covered() -> None:
    def reasons(cases: list[Any]) -> set[str]:
        return {p.values[2] for p in cases if p.values[2] is not None}

    assert reasons(GOPHER_QUALITY) == {
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
    }
    assert reasons(GOPHER_REPETITION) >= {
        "empty",
        "dup_para_frac",
        "dup_para_char_frac",
        "dup_line_frac",
        "dup_line_char_frac",
        "top_2_gram",
        "top_3_gram",
        "top_4_gram",
        "duplicated_5_n_grams",
    }
    assert reasons(C4_CASES) == {"lorem_ipsum", "curly_bracket", "too_few_sentences"}
    assert reasons(FINEWEB) == {
        "empty",
        "line_punct_ratio",
        "short_line_ratio",
        "char_dup_ratio",
        "list_ratio",
    }


# ---- container kinds -----------------------------------------------------------

DOCS = [
    GOOD,
    "",
    "Far too short.",
    GOOD + "\n" + _JS,
    BULLETS,
    f"{RIVER} {RIVER}",
    "\n \n\t\n",
    GOOD + "\nThe template uses {braces} in this line.",
    "Short.\nAlso short.\nTiny one.",
    "Menu Home About\n" + GOOD + "\nYes.",
]

#: (predicate, reasons) per filter, with their default configs.
FILTERS: list[tuple[str, Callable[..., Any], Callable[..., Any], Any]] = [
    ("gopher_quality", pf.gopher_quality, pf.gopher_quality_reasons, pf.GopherQuality()),
    (
        "gopher_repetition",
        pf.gopher_repetition,
        pf.gopher_repetition_reasons,
        pf.GopherRepetition(),
    ),
    ("c4_quality", pf.c4_quality, pf.c4_quality_reasons, pf.C4Quality()),
    ("fineweb_quality", pf.fineweb_quality, pf.fineweb_quality_reasons, pf.FineWebQuality()),
]
FILTER_IDS = [f[0] for f in FILTERS]


@pytest.mark.parametrize(("name", "keep", "reasons", "cfg"), FILTERS, ids=FILTER_IDS)
def test_str_list_tuple_kinds(
    name: str, keep: Callable[..., Any], reasons: Callable[..., Any], cfg: Any
) -> None:
    want_reasons = [reasons(d) for d in DOCS]
    want_keep = [r is None for r in want_reasons]
    assert all(type(keep(d)) is bool for d in DOCS)
    assert all(r is None or type(r) is str for r in want_reasons)
    assert [keep(d) for d in DOCS] == want_keep
    out = keep(DOCS)
    assert type(out) is list
    assert out == want_keep
    assert all(type(m) is bool for m in out)
    assert reasons(DOCS) == want_reasons
    out_t = keep(tuple(DOCS))
    assert type(out_t) is tuple
    assert out_t == tuple(want_keep)
    assert reasons(tuple(DOCS)) == tuple(want_reasons)
    assert keep([]) == []
    assert keep(()) == ()
    assert reasons([]) == []
    assert keep(DOCS, cfg) == want_keep
    res = pf.apply(DOCS, [cfg])
    assert isinstance(res, pf.FilterResult)
    assert res.keep == want_keep
    assert res.reason == want_reasons
    assert any(want_keep)
    assert not all(want_keep), name


def test_c4_lines_kinds_and_identity() -> None:
    want = [pf.c4_lines(d) for d in DOCS]
    out = pf.c4_lines(DOCS)
    assert type(out) is list
    assert out == want
    out_t = pf.c4_lines(tuple(DOCS))
    assert type(out_t) is tuple
    assert out_t == tuple(want)
    reasons = pf.c4_quality_reasons(DOCS)
    for i, (d, o, r) in enumerate(zip(DOCS, out, reasons, strict=True)):
        if r is not None or o == d:
            assert o is d, i  # dropped or unchanged: the caller's object
            assert out_t[i] is d, i
        else:
            assert type(o) is str
    assert out[0] is DOCS[0]
    assert out[3] == GOOD
    assert out[9] == GOOD
    assert pf.c4_lines([]) == []
    assert pf.c4_lines(()) == ()


def test_same_object_repeated_in_a_batch() -> None:
    changed = GOOD + "\n" + _JS
    docs = [GOOD, changed] * 100
    out = pf.c4_lines(docs)
    assert all(o is GOOD for o in out[::2])
    assert out[1::2] == [GOOD] * 100
    assert all(o is not changed for o in out[1::2])


def test_str_subclass_documents() -> None:
    class MyStr(str):
        __slots__ = ()

    keep, changed = MyStr(GOOD), MyStr(GOOD + "\n" + _JS)
    assert pf.c4_lines(keep) is keep
    out = pf.c4_lines(changed)
    assert type(out) is str
    assert out == GOOD
    assert pf.c4_lines([keep, changed]) == [keep, GOOD]
    assert pf.gopher_quality([keep, changed]) == [True, True]


def test_input_is_not_modified() -> None:
    docs = list(DOCS)
    pf.apply(docs, [pf.C4Quality(), pf.FineWebQuality()])
    pf.c4_lines(docs)
    assert docs == DOCS


ARROW_VALUES: list[str | None] = [*DOCS[:4], None, *DOCS[4:], None]


def _expected(values: list[str | None], fn: Callable[..., Any]) -> list[Any]:
    return [None if v is None else fn(v) for v in values]


@pytest.mark.parametrize("large", [False, True], ids=["string", "large_string"])
@pytest.mark.parametrize("chunks", [1, 3])
def test_arrow_without_pyarrow(large: bool, chunks: int) -> None:
    col = _testing.column(ARROW_VALUES, large=large, chunks=chunks, name="text")
    for _, keep, reasons, _cfg in FILTERS:
        mask = keep(col)
        assert isinstance(mask, pyroid.Column)
        assert mask.type == "bool"
        assert mask.name == "text"
        assert mask.null_count == 2
        assert mask.num_chunks == chunks
        assert mask.to_pylist() == _expected(ARROW_VALUES, keep)
        why = reasons(col)
        assert isinstance(why, pyroid.Column)
        assert why.type == "string"
        assert why.name == "text"
        assert why.to_pylist() == _expected(ARROW_VALUES, reasons)
        # Kept documents and nulls have no reason.
        assert why.null_count == sum(m is not False for m in mask.to_pylist())
    lines = pf.c4_lines(col)
    assert isinstance(lines, pyroid.Column)
    assert lines.type == ("large_string" if large else "string")
    assert lines.name == "text"
    assert lines.null_count == 2
    assert lines.to_pylist() == _expected(ARROW_VALUES, pf.c4_lines)
    steps = [pf.GopherRepetition(), pf.C4Quality(), pf.FineWebQuality()]
    res = pf.apply(col, steps)
    assert isinstance(res, pf.FilterResult)
    assert all(isinstance(x, pyroid.Column) for x in res)
    assert [x.type for x in res] == ["bool", "string", lines.type]
    want = [None if v is None else _chain(v, steps) for v in ARROW_VALUES]
    assert res.keep.to_pylist() == [w and w[0] for w in want]
    assert res.reason.to_pylist() == [w and w[1] for w in want]
    assert res.text.to_pylist() == [w and w[2] for w in want]


def test_arrow_with_pyarrow(pa: Any) -> None:
    values = ARROW_VALUES
    arrays = [
        pa.array(values),
        pa.array(values, pa.large_string()),
        pa.array(values, pa.string_view()),
        pa.chunked_array([values[:4], values[4:7], values[7:]]),
        pa.chunked_array([values[:4], values[4:]], pa.large_string()),
        pa.array(["skip", "this", *values])[2:],  # offset > 0
        pa.chunked_array([pa.array(["skip", *values[:5]])[1:], values[5:]]),
    ]
    for arr in arrays:
        for _, keep, reasons, _cfg in FILTERS:
            assert pa.chunked_array(keep(arr)).to_pylist() == _expected(values, keep)
            assert pa.chunked_array(reasons(arr)).to_pylist() == _expected(values, reasons)
        out = pf.c4_lines(arr)
        assert pa.chunked_array(out).to_pylist() == _expected(values, pf.c4_lines)
        res = pf.apply(arr, [pf.C4Quality(), pf.FineWebQuality()])
        chain = [
            None if v is None else pf.apply(v, [pf.C4Quality(), pf.FineWebQuality()])
            for v in values
        ]
        assert pa.chunked_array(res.keep).to_pylist() == [c and c.keep for c in chain]
        assert pa.chunked_array(res.reason).to_pylist() == [c and c.reason for c in chain]
        assert pa.chunked_array(res.text).to_pylist() == [c and c.text for c in chain]
    # The result columns are ordinary Arrow columns.
    table = pa.table({"text": pa.array(values)})
    keep_col = pf.gopher_repetition(table["text"])
    assert pa.chunked_array(keep_col).type == pa.bool_()
    assert pa.chunked_array(pf.c4_lines(pa.array(values, pa.large_string()))).type == (
        pa.large_string()
    )
    assert pa.chunked_array(pf.c4_quality_reasons(pa.array(values))).type == pa.string()


def test_arrow_invalid_utf8(pa: Any) -> None:
    bad = pa.array([b"ok", b"\xff\xfe"], pa.binary()).view(pa.string())
    with pytest.raises(UnicodeDecodeError) as info:
        pf.gopher_quality(bad)
    assert "row 1 of chunk 0" in "\n".join(getattr(info.value, "__notes__", []))


# ---- configs -------------------------------------------------------------------


def test_config_defaults_are_datatrove_defaults() -> None:
    assert pf.DATATROVE_VERSION == "0.10.1"
    assert pf.STOP_WORDS == ("the", "be", "to", "of", "and", "that", "have", "with")
    assert pf.GopherQuality() == (50, 100000, 3, 10, 0.1, 0.9, 0.3, 0.8, 2, None, "eng")
    rep = pf.GopherRepetition()
    assert rep[:4] == (0.3, 0.3, 0.2, 0.2)
    assert tuple(rep.top_n_grams) == ((2, 0.2), (3, 0.18), (4, 0.16))
    assert tuple(rep.dup_n_grams) == (
        (5, 0.15),
        (6, 0.14),
        (7, 0.13),
        (8, 0.12),
        (9, 0.11),
        (10, 0.10),
    )
    assert rep.language == "eng"
    assert pf.C4Quality() == (True, True, True, 5, 3, 1000, True, True, True, True, "eng")
    assert pf.FineWebQuality() == (0.12, False, None, 0.67, 30, 0.01, 0.3, "eng")


def test_config_object_kwargs_and_overrides() -> None:
    docs = [*DOCS, " ".join(["house"] * 20), GOOD]
    by_kwargs = pf.gopher_quality_reasons(docs, min_doc_words=5, max_doc_words=60)
    cfg = pf.GopherQuality(min_doc_words=5, max_doc_words=60)
    assert pf.gopher_quality_reasons(docs, cfg) == by_kwargs
    assert by_kwargs != pf.gopher_quality_reasons(docs)
    # Keyword arguments override the config's fields.
    assert pf.gopher_quality_reasons(docs, cfg, min_doc_words=50, max_doc_words=100000) == (
        pf.gopher_quality_reasons(docs)
    )
    assert pf.gopher_quality_reasons(docs, cfg, max_doc_words=None) == pf.gopher_quality_reasons(
        docs, min_doc_words=5
    )
    assert pf.apply(docs, [cfg]).reason == by_kwargs
    c4 = pf.C4Quality(min_num_sentences=1)
    assert pf.c4_lines(docs, c4) == pf.c4_lines(docs, min_num_sentences=1)
    assert pf.c4_lines(docs, c4, min_num_sentences=5) == pf.c4_lines(docs)
    fw = pf.FineWebQuality(line_punct_exclude_zero=True, short_line_thr=1.0)
    assert pf.fineweb_quality_reasons(docs, fw) == pf.fineweb_quality_reasons(
        docs, line_punct_exclude_zero=True, short_line_thr=1.0
    )
    rep = pf.GopherRepetition(top_n_grams=[(2, 0.5)], dup_n_grams=[])
    assert pf.gopher_repetition_reasons(docs, rep) == pf.gopher_repetition_reasons(
        docs, top_n_grams=((2, 0.5),), dup_n_grams=()
    )


def test_stop_words_are_a_set_of_strings() -> None:
    doc = " ".join(["house", "garden"] * 30)
    assert pf.gopher_quality(doc, stop_words=("house", "garden"))
    assert pf.gopher_quality(doc, stop_words={"house", "garden", 3})  # type: ignore[arg-type]
    assert not pf.gopher_quality(doc, stop_words=["house", "house"])
    assert pf.gopher_quality(doc, stop_words=["house"], min_stop_words=1)
    assert not pf.gopher_quality(doc, stop_words=[])
    assert pf.gopher_quality(doc, stop_words=iter(["house", "garden"]))


def test_english_languages() -> None:
    langs = pf.english_languages()
    assert isinstance(langs, list)
    assert {"eng", "en", "eng_Latn"} <= set(langs)
    assert "fra" not in langs
    for lang in ("en", "eng", "sco"):
        assert pf.gopher_quality_reasons(DOCS, language=lang) == pf.gopher_quality_reasons(DOCS)
        assert pf.c4_lines(DOCS, language=lang) == pf.c4_lines(DOCS)


def test_datatrove_filter_instances_as_configs() -> None:
    dt: Any = pytest.importorskip("datatrove.pipeline.filters")
    pairs: list[tuple[Any, Any]] = [
        (
            dt.GopherQualityFilter(min_doc_words=5, stop_words=["house"]),
            pf.GopherQuality(min_doc_words=5, stop_words=["house"]),
        ),
        (
            dt.GopherRepetitionFilter(top_n_grams=((2, 0.5),)),
            pf.GopherRepetition(top_n_grams=((2, 0.5),)),
        ),
        (dt.C4QualityFilter(split_paragraph=False), pf.C4Quality(split_paragraph=False)),
        (
            dt.FineWebQualityFilter(short_line_thr=0.9, stop_chars=(".",)),
            pf.FineWebQuality(short_line_thr=0.9, stop_chars=(".",)),
        ),
    ]
    for dt_filter, cfg in pairs:
        assert pf.apply(DOCS, [dt_filter]) == pf.apply(DOCS, [cfg])
    assert pf.c4_lines(DOCS, pairs[2][0]) == pf.c4_lines(DOCS, split_paragraph=False)
    # Keyword arguments override a datatrove instance's settings too.
    assert pf.c4_lines(DOCS, pairs[2][0], split_paragraph=True) == pf.c4_lines(DOCS)
    assert pf.gopher_quality(DOCS, pairs[0][0]) == pf.gopher_quality(DOCS, pairs[0][1])
    with pytest.raises(TypeError, match="GopherQuality"):
        pf.gopher_quality(DOCS, pairs[2][0])


# ---- apply ---------------------------------------------------------------------

FINEWEB_CHAIN = [
    pf.GopherRepetition(),
    pf.GopherQuality(),
    pf.C4Quality(filter_no_terminal_punct=False),
    pf.FineWebQuality(),
]


def _chain(doc: str, steps: list[Any]) -> tuple[bool, str | None, str]:
    """datatrove's executor: the first drop wins; C4's text is what later steps see."""
    text = doc
    for step in steps:
        if isinstance(step, pf.C4Quality):
            reason = pf.c4_quality_reasons(text, step)
            if reason is None:
                text = pf.c4_lines(text, step)
        elif isinstance(step, pf.GopherQuality):
            reason = pf.gopher_quality_reasons(text, step)
        elif isinstance(step, pf.GopherRepetition):
            reason = pf.gopher_repetition_reasons(text, step)
        else:
            reason = pf.fineweb_quality_reasons(text, step)
        if reason is not None:
            return False, reason, text
    return True, None, text


def test_apply_chains_like_a_pipeline() -> None:
    steps = [pf.C4Quality(), pf.FineWebQuality()]
    res = pf.apply(DOCS, steps)
    assert list(zip(res.keep, res.reason, res.text, strict=True)) == [
        _chain(d, steps) for d in DOCS
    ]
    for chain in (
        FINEWEB_CHAIN,
        FINEWEB_CHAIN[::-1],
        [pf.C4Quality(), pf.C4Quality(split_paragraph=False)],
    ):
        res = pf.apply(DOCS, chain)
        assert list(zip(res.keep, res.reason, res.text, strict=True)) == [
            _chain(d, chain) for d in DOCS
        ]


def test_apply_sees_c4s_rewritten_text() -> None:
    # FineWeb on the original text sees 7 lines, one of them short ("Ok."); C4 removes
    # that line (fewer than 3 words), so FineWeb after C4 sees GOOD.
    doc = GOOD + "\nOk."
    assert pf.fineweb_quality_reasons(doc, short_line_thr=0.1) == "short_line_ratio"
    res = pf.apply(doc, [pf.C4Quality(), pf.FineWebQuality(short_line_thr=0.1)])
    assert tuple(res) == (True, None, GOOD)
    # ... and a document C4 rewrites can still be dropped later, with C4's text.
    res = pf.apply(GOOD + "\nOk.", [pf.C4Quality(), pf.FineWebQuality(stop_chars="!")])
    assert tuple(res) == (False, "line_punct_ratio", GOOD)
    # The first drop wins.
    res = pf.apply("", [pf.GopherRepetition(), pf.FineWebQuality()])
    assert tuple(res) == (False, "empty", "")
    res = pf.apply(BULLETS, [pf.GopherQuality(), pf.C4Quality()])
    assert tuple(res) == (False, "gopher_too_many_bullets", BULLETS)
    assert res.text is BULLETS


def test_is_alpha_documents_are_ordinary_text() -> None:
    # Documented difference: spaCy (so datatrove) can fail on the token IS_ALPHA and then
    # retokenizes without it, depending on earlier documents. pyroid's words never do.
    docs = [GOOD + " IS_ALPHA", "IS_ALPHA " * 60, GOOD, "IS_ALPHA" + GOOD]
    res = pf.apply(docs, FINEWEB_CHAIN)
    back = pf.apply(docs[::-1], FINEWEB_CHAIN)
    assert [x[::-1] for x in back] == list(res)
    assert [pf.apply(d, FINEWEB_CHAIN) for d in docs] == list(
        map(pf.FilterResult._make, zip(*res, strict=True))
    )
    # 60 words "IS_ALPHA" (8 characters, with letters, no stop words).
    assert pf.gopher_quality_reasons("IS_ALPHA " * 60) == "gopher_enough_stop_words"
    assert (
        pf.gopher_quality_reasons("IS_ALPHA " * 60, stop_words=["IS_ALPHA"], min_stop_words=1)
        is None
    )


def test_apply_kinds() -> None:
    res = pf.apply(GOOD, FINEWEB_CHAIN)
    assert tuple(res) == (True, None, GOOD)
    assert res.text is GOOD
    lst = pf.apply(DOCS, FINEWEB_CHAIN)
    tup = pf.apply(tuple(DOCS), FINEWEB_CHAIN)
    assert all(type(x) is list for x in lst)
    assert all(type(x) is tuple for x in tup)
    assert tuple(map(tuple, lst)) == tup
    assert [pf.apply(d, FINEWEB_CHAIN) for d in DOCS] == list(
        map(pf.FilterResult._make, zip(*lst, strict=True))
    )
    # No steps: everything is kept, unchanged.
    none = pf.apply(DOCS, [])
    assert none.keep == [True] * len(DOCS)
    assert none.reason == [None] * len(DOCS)
    assert all(o is d for o, d in zip(none.text, DOCS, strict=True))
    # Steps that don't rewrite text hand back the caller's objects.
    res = pf.apply(DOCS, (pf.GopherQuality(), pf.FineWebQuality()))
    assert all(o is d for o, d in zip(res.text, DOCS, strict=True))
    res = pf.apply(DOCS, iter([pf.C4Quality()]))
    assert res.text == pf.c4_lines(DOCS)


# ---- errors --------------------------------------------------------------------


def _notes(exc: BaseException) -> str:
    return "\n".join(getattr(exc, "__notes__", []))


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


BAD_INPUTS: list[Any] = [b"bytes", 3, None, ["ok", 3], ("ok", b"x"), {"a"}, [None]]


@pytest.mark.parametrize("bad", BAD_INPUTS)
@pytest.mark.parametrize("fn", ALL_FUNCS, ids=lambda f: f.__name__)
def test_wrong_input_types_raise_input_error(fn: Callable[..., Any], bad: Any) -> None:
    with pytest.raises(pyroid.InputError):
        fn(bad)
    with pytest.raises(pyroid.InputError):
        pf.apply(bad, [pf.GopherQuality()])


def test_wrong_arrow_type_raises_input_error(pa: Any) -> None:
    with pytest.raises(pyroid.InputError, match="int64"):
        pf.gopher_quality(pa.array([1, 2]))
    mixed: list[Any] = ["ok", 3]
    with pytest.raises(pyroid.InputError, match="index 1"):
        pf.c4_lines(mixed)


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (pf.gopher_quality, {"min_doc_words": "50"}),
        (pf.gopher_quality, {"max_symbol_word_ratio": [0.1]}),
        (pf.gopher_quality, {"language": 3}),
        (pf.gopher_quality, {"bogus": 1}),
        (pf.gopher_repetition, {"dup_line_frac": "0.3"}),
        (pf.gopher_repetition, {"top_n_grams": ((2.0, 0.2),)}),
        (pf.gopher_repetition, {"dup_n_grams": ((True, 0.2),)}),
        (pf.gopher_repetition, {"dup_n_grams": ((5, "x"),)}),
        (pf.gopher_repetition, {"top_n_grams": 3}),
        (pf.c4_quality, {"min_num_sentences": "5"}),
        (pf.c4_lines, {"max_word_length": None}),
        (pf.fineweb_quality, {"line_punct_thr": None}),
        (pf.fineweb_quality, {"short_line_length": "30"}),
        (pf.fineweb_quality, {"stop_chars": 3}),
    ],
)
def test_bad_parameter_types_raise_type_error(
    fn: Callable[..., Any], kwargs: dict[str, Any]
) -> None:
    # Checked before any document is looked at, so even an empty batch raises.
    for texts in ("x", [], ["x"]):
        with pytest.raises(TypeError) as info:
            fn(texts, **kwargs)
        assert not isinstance(info.value, pyroid.InputError)


def test_wrong_config_type() -> None:
    with pytest.raises(TypeError, match="GopherQuality"):
        pf.gopher_quality("x", pf.C4Quality())  # type: ignore[call-overload]
    with pytest.raises(TypeError, match="C4Quality"):
        pf.c4_lines("x", {"split_paragraph": False})  # type: ignore[call-overload]
    with pytest.raises(TypeError, match="datatrove filter"):
        pf.apply("x", [pf.GopherQuality(), object()])


@pytest.mark.parametrize("key", ["top_n_grams", "dup_n_grams"])
@pytest.mark.parametrize("n", [0, -1])
def test_ngram_size_below_one_raises_value_error(key: str, n: int) -> None:
    # datatrove loops forever on a dup_n_grams size below 1; pyroid refuses it.
    one: dict[str, Any] = {key: ((n, 0.2),)}
    two: dict[str, Any] = {key: ((2, 0.2), (n, 0.2))}
    with pytest.raises(ValueError, match="at least 1"):
        pf.gopher_repetition("some words here", **one)
    with pytest.raises(ValueError, match="at least 1"):
        pf.apply([], [pf.GopherRepetition(**two)])


@pytest.mark.parametrize("lang", ["kaz_Cyrl", "kor_Hang", "khm_Khmr"])
def test_language_without_native_tokenizer_raises_not_implemented(lang: str) -> None:
    # Codes datatrove tokenizes with a segmenter pyroid does not port (Stanza,
    # Kiwi, khmer-nltk); python_tokenizer=True is the opt-in. (Japanese and Thai
    # depend on installed data: test_filters_japanese.py, test_filters_vithai.py.)
    with pytest.raises(NotImplementedError, match="tokenizer"):
        pf.gopher_quality("x", language=lang)
    with pytest.raises(NotImplementedError, match="tokenizer"):
        pf.apply([], [pf.C4Quality(language=lang)])


@pytest.mark.parametrize("lang", ["", "ENG", "xyz_Zzzz"])
def test_language_unknown_to_datatrove_raises_value_error(lang: str) -> None:
    with pytest.raises(ValueError, match="tokenizer"):
        pf.gopher_quality("x", language=lang)
    with pytest.raises(ValueError, match="tokenizer"):
        pf.apply([], [pf.C4Quality(language=lang)])


@pytest.mark.parametrize(
    ("fn", "kwargs", "doc"),
    [
        # min_doc_words off: text.count("#") / n_words with no words.
        (pf.gopher_quality, {"min_doc_words": None}, ""),
        (pf.gopher_quality, {"min_doc_words": None}, " \n\t"),
        # ... then the bullet ratio's / len(text.splitlines()).
        (pf.gopher_quality, {"min_doc_words": None, "max_symbol_word_ratio": None}, ""),
        # ... then the alpha ratio's / n_words.
        (
            pf.gopher_quality,
            {
                "min_doc_words": None,
                "max_symbol_word_ratio": None,
                "max_bullet_lines_ratio": None,
                "max_ellipsis_lines_ratio": None,
            },
            "   ",
        ),
    ],
)
def test_zero_division_like_datatrove(
    fn: Callable[..., Any], kwargs: dict[str, Any], doc: str
) -> None:
    with pytest.raises(ZeroDivisionError) as info:
        fn(doc, **kwargs)
    assert "the str" in _notes(info.value)
    with pytest.raises(ZeroDivisionError) as info:
        fn([GOOD, "ok", doc, GOOD], **kwargs)
    assert "index 2" in _notes(info.value)
    with pytest.raises(ZeroDivisionError) as info:
        fn((doc,), **kwargs)
    assert "index 0" in _notes(info.value)
    col = _testing.column([GOOD, None, doc], chunks=2)
    with pytest.raises(ZeroDivisionError) as info:
        fn(col, **kwargs)
    assert "row 0 of chunk 1" in _notes(info.value)
    with pytest.raises(ZeroDivisionError):
        pf.apply([doc], [pf.GopherQuality(**kwargs)])
    # A filter that drops the document first means the division never happens.
    assert pf.apply(doc, [pf.FineWebQuality(), pf.GopherQuality(**kwargs)]) == (
        False,
        "empty",
        doc,
    )
    # Documents with words are fine.
    assert fn([GOOD, "Plenty of words."], **kwargs) == [True, False]


SURROGATE_DOCS = ["a\udc00b", "\ud800", GOOD + "\ud83d", "\udfff" + GOOD]


@pytest.mark.parametrize("fn", ALL_FUNCS, ids=lambda f: f.__name__)
def test_lone_surrogates_raise_unicode_encode_error(fn: Callable[..., Any]) -> None:
    for doc in SURROGATE_DOCS:
        with pytest.raises(UnicodeEncodeError):
            fn(doc)
        with pytest.raises(UnicodeEncodeError) as info:
            fn([GOOD, doc])
        assert "index 1" in _notes(info.value)
        with pytest.raises(UnicodeEncodeError):
            fn((doc, GOOD))
        with pytest.raises(UnicodeEncodeError):
            pf.apply([doc], [pf.FineWebQuality()])
    # A surrogate pair is an ordinary (astral) character.
    assert len(fn([GOOD + " \U0001f600", "\U0001f600"])) == 2


# ---- threads -------------------------------------------------------------------

THREAD_DOCS = [f"{d}\n{i}" if i % 3 else d for i, d in enumerate(DOCS * 200)]


@pytest.fixture(scope="module")
def single_threaded() -> dict[str, Any]:
    """Every result for THREAD_DOCS, computed on the calling thread."""
    out: dict[str, Any] = {"apply": pf.apply(THREAD_DOCS, FINEWEB_CHAIN, threads=1)}
    for name, keep, reasons, _cfg in FILTERS:
        out[name] = (reasons(THREAD_DOCS, threads=1), keep(tuple(THREAD_DOCS), threads=1))
    out["c4_lines"] = pf.c4_lines(THREAD_DOCS, threads=1)
    return out


@pytest.mark.parametrize("threads", [None, 1, 2, 4])
def test_threads_give_identical_results(
    threads: int | None, single_threaded: dict[str, Any]
) -> None:
    assert len(THREAD_DOCS) == 2000
    assert pf.apply(THREAD_DOCS, FINEWEB_CHAIN, threads=threads) == single_threaded["apply"]
    for name, keep, reasons, _cfg in FILTERS:
        want_reasons, want_keep = single_threaded[name]
        assert reasons(THREAD_DOCS, threads=threads) == want_reasons
        assert keep(tuple(THREAD_DOCS), threads=threads) == want_keep
        assert want_keep == tuple(r is None for r in want_reasons)
    assert pf.c4_lines(THREAD_DOCS, threads=threads) == single_threaded["c4_lines"]
    col = _testing.column(THREAD_DOCS, chunks=3)
    assert pf.c4_lines(col, threads=threads).to_pylist() == single_threaded["c4_lines"]


@pytest.mark.parametrize("bad", [0, -1])
def test_bad_thread_counts(bad: int) -> None:
    with pytest.raises(ValueError, match="threads"):
        pf.gopher_quality(DOCS, threads=bad)
    with pytest.raises(ValueError, match="threads"):
        pf.apply(DOCS, [pf.C4Quality()], threads=bad)


@pytest.mark.thread_unsafe(reason="10k documents through four filters: too slow per thread")
@pytest.mark.timeout(240)
def test_ten_thousand_documents() -> None:
    docs = [f"{d}\n{i}." for i in range(1000) for d in DOCS]
    assert len(docs) == 10_000
    res = pf.apply(docs, FINEWEB_CHAIN)
    assert len(res.keep) == len(res.reason) == len(res.text) == 10_000
    first = pf.apply(docs[:10], FINEWEB_CHAIN)
    assert [x[:10] for x in res] == list(first)
    assert res.keep.count(True) == 1000 * first.keep.count(True)
    col = pf.apply(_testing.column(docs, chunks=4), FINEWEB_CHAIN)
    assert col.keep.to_pylist() == res.keep
    assert col.reason.to_pylist() == res.reason
    assert col.text.to_pylist() == res.text


# ---- module surface ------------------------------------------------------------


def test_module_surface() -> None:
    # The quality filters' names; the further datatrove steps check theirs in test_*_more.py.
    assert set(pf.__all__).issuperset(
        [
            "DATATROVE_VERSION",
            "STOP_WORDS",
            "C4Quality",
            "FilterResult",
            "FineWebQuality",
            "GopherQuality",
            "GopherRepetition",
            "Language",
            "apply",
            "c4_lines",
            "c4_quality",
            "c4_quality_reasons",
            "english_languages",
            "fineweb_quality",
            "fineweb_quality_reasons",
            "gopher_quality",
            "gopher_quality_reasons",
            "gopher_repetition",
            "gopher_repetition_reasons",
            "languages",
        ]
    )
    for f in [*ALL_FUNCS, pf.apply, pf.english_languages]:
        assert f.__module__ == "pyroid.filters"
        assert f.__doc__


@pytest.mark.thread_unsafe(reason="runs a subprocess")
def test_import_pyroid_does_not_import_filters(
    run_python: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    proc = run_python(
        """
        import sys, pyroid
        assert "pyroid.filters" not in sys.modules, sorted(sys.modules)
        import pyroid.filters as f
        assert "datatrove" not in sys.modules
        assert "numpy" not in sys.modules
        print(f.gopher_quality("x"), f.c4_lines("Yes."))
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False Yes."
