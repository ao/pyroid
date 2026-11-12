"""pyroid.integrations.nemo_curator vs NeMo Curator 1.4.0's own stages, in the same process.

Every check runs a stock NeMo stage and the pyroid stage on the same DocumentBatch and compares the
whole output DataFrame (values, dtypes, index, column order), or the exception. Skips without NeMo
Curator, which needs Python < 3.14.
"""

from __future__ import annotations

import os
import random
import warnings
from collections.abc import Callable
from typing import Any

import pytest

pytest.importorskip("nemo_curator.stages.text.filters", reason="NeMo Curator is not installed")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")

import nemo_curator.stages.text.filters as nc_filters  # noqa: E402
import nemo_curator.stages.text.filters.heuristic as nc_h  # noqa: E402
import nemo_curator.stages.text.filters.heuristic.repetition as nc_r  # noqa: E402
import nemo_curator.stages.text.modifiers as nc_mod  # noqa: E402
from nemo_curator.stages.text.modifiers.fasttext import FastTextLabelModifier as NcLabel  # noqa: E402
from nemo_curator.tasks import DocumentBatch  # noqa: E402

import pyroid  # noqa: E402
import pyroid.integrations.nemo_curator as ours  # noqa: E402

pytestmark = pytest.mark.thread_unsafe  # numpy's global random state, pandas

EDGE = [
    "",
    " ",
    "\n",
    "\n\n",
    "\n\n\n\n",
    "word",
    "### ...",
    "#a ## b … [...] -»",
    "• one\n• two\n‣ three\nfour",
    "Read more...\nREAD MORE\nsomething [...]",
    "Terms of Use\n\nreal text here\n\nprivacy policy",
    "Lorem Ipsum dolor",
    "see https://x.y/%20z and http://porn.example/a (b) {c} [d] ⟨e⟩",
    "The cat and the dog have to be with the that of",
    "a a a a b b b b c c",
    "the the the the the the",
    "x" * 1200,
    "1234567890 numbers 42",
    "\t\r\x08 spaces   ",
    "coo\u212aie policy\n\nstuff",
    "prİvacy policy",
    "　\u2028 odd\xa0spaces\x1c",
    "**bold** *it* _u_ [a](b)",
    '"quoted text"',
    "line\nline\nline\nother",
    "para\n\npara\n\npara",
    "\U0001f600 emoji \U0001f600 emoji",
]


def _corpus(n: int = 600, seed: int = 0) -> list[str]:
    words = [
        "the", "a", "of", "and", "to", "be", "that", "have", "with", "data", "cat", "#tag", "...",
        "http://u.v/w", "www.x.y", "(1)", "[2]", "•", "é", "日本", "123", "Read", "more", "\u212a",
    ]  # fmt: skip
    rng = random.Random(seed)
    docs = list(EDGE)
    while len(docs) < n:
        lines = [
            " ".join(rng.choice(words) for _ in range(rng.randint(0, 40)))
            for _ in range(rng.randint(1, 8))
        ]
        docs.append(rng.choice(["\n", "\n\n", " "]).join(lines))
    try:
        import pyarrow.parquet as pq

        path = os.environ.get("PYROID_FINEWEB_DEV", "/data/fineweb/fineweb-dev-10k.parquet")
        docs += pq.read_table(path, columns=["text"]).column("text").to_pylist()[:400]
    except (OSError, ImportError):
        pass
    return docs


DOCS = _corpus()
NONEMPTY = [d for d in DOCS if d.split()]


def _frames(docs: list[str]) -> dict[str, Callable[[], Any]]:
    """The text column as each reader gives it: Arrow table, string[pyarrow], str, object."""
    table = pa.table({"id": list(range(len(docs))), "text": pa.array(docs, pa.large_string())})
    return {
        "arrow-table": lambda: table,
        "string[pyarrow]": lambda: table.to_pandas(types_mapper=pd.ArrowDtype),
        "object": lambda: pd.DataFrame(
            {"id": list(range(len(docs))), "text": pd.Series(docs, dtype=object)}
        ),
    }


def _batch(data: Any) -> Any:
    return DocumentBatch(dataset_name="t", data=data)


def _run(stage: Any, data: Any) -> Any:
    if hasattr(stage, "setup"):
        stage.setup()
    try:
        out = stage.process(_batch(data))
    except Exception as e:
        return (type(e), str(e))
    return out.to_pandas()


def _same(a: Any, b: Any) -> None:
    if isinstance(a, tuple) or isinstance(b, tuple):
        assert a == b
        return
    assert list(a.columns) == list(b.columns)
    assert a.dtypes.to_dict() == b.dtypes.to_dict()
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    for col in a.columns:
        if a[col].dtype.kind == "f":
            assert [repr(x) for x in a[col].tolist()] == [repr(x) for x in b[col].tolist()]


HEURISTIC: list[tuple[str, dict[str, Any]]] = [
    ("NonAlphaNumericFilter", {}),
    ("SymbolsToWordsFilter", {}),
    ("NumbersFilter", {}),
    ("UrlsFilter", {}),
    ("WhiteSpaceFilter", {}),
    ("ParenthesesFilter", {}),
    ("BoilerPlateStringFilter", {}),
    ("WordCountFilter", {"min_words": 5, "max_words": 300}),
    ("PunctuationFilter", {}),
    ("WordsWithoutAlphabetsFilter", {}),
    ("CommonEnglishWordsFilter", {}),
    ("CommonEnglishWordsFilter", {"min_num_common_words": 3, "stop_at_false": False}),
    ("MeanWordLengthFilter", {}),
    ("LongWordFilter", {"max_word_length": 20}),
    ("EllipsisFilter", {}),
    ("BulletsFilter", {"max_bullet_lines_ratio": 0.5}),
    ("PornographicUrlsFilter", {}),
    ("SubstringFilter", {"substring": "the", "position": "any"}),
    ("SubstringFilter", {"substring": "", "position": "prefix"}),
]
REPETITION: list[tuple[str, dict[str, Any]]] = [
    ("RepeatedLinesFilter", {}),
    ("RepeatedParagraphsFilter", {}),
    ("RepeatedLinesByCharFilter", {}),
    ("RepeatedParagraphsByCharFilter", {}),
    ("RepeatingTopNGramsFilter", {"n": 2}),
    ("RepeatingTopNGramsFilter", {"n": 3, "max_repeating_ngram_ratio": 0.18}),
    ("RepeatingTopNGramsFilter", {"n": 4, "max_repeating_ngram_ratio": 0.16}),
    ("RepeatingDuplicateNGramsFilter", {"n": 5, "max_repeating_duplicate_ngram_ratio": 0.15}),
    ("RepeatingDuplicateNGramsFilter", {"n": 10}),
]


def _stock_cls(name: str) -> Any:
    return getattr(nc_h, name, None) or getattr(nc_r, name)


@pytest.mark.parametrize("frame", ["arrow-table", "string[pyarrow]", "object"])
@pytest.mark.parametrize(("name", "kwargs"), HEURISTIC + REPETITION, ids=lambda v: str(v))
def test_score_filter_matches_nemo(frame: str, name: str, kwargs: dict[str, Any]) -> None:
    make = _frames(NONEMPTY)[frame]
    stock = _run(nc_filters.ScoreFilter(_stock_cls(name)(**kwargs), score_field="s"), make())
    # pyroid's stage with NeMo's filter object, and with pyroid's filter object.
    _same(_run(ours.ScoreFilter(_stock_cls(name)(**kwargs), score_field="s"), make()), stock)
    _same(_run(ours.ScoreFilter(getattr(ours, name)(**kwargs), score_field="s"), make()), stock)
    # pyroid's filter object inside NeMo's stage (one document at a time).
    _same(
        _run(nc_filters.ScoreFilter(getattr(ours, name)(**kwargs), score_field="s"), make()), stock
    )
    # No score field, inverted.
    stock_inv = _run(nc_filters.ScoreFilter(_stock_cls(name)(**kwargs), invert=True), make())
    _same(_run(ours.ScoreFilter(getattr(ours, name)(**kwargs), invert=True), make()), stock_inv)


@pytest.mark.parametrize(("name", "kwargs"), HEURISTIC + REPETITION, ids=lambda v: str(v))
def test_errors_match_nemo(name: str, kwargs: dict[str, Any]) -> None:
    # Empty and blank documents make several of NeMo's filters raise; the stage must raise the same.
    make = _frames(EDGE)["arrow-table"]
    stock = _run(nc_filters.ScoreFilter(_stock_cls(name)(**kwargs), score_field="s"), make())
    _same(_run(ours.ScoreFilter(getattr(ours, name)(**kwargs), score_field="s"), make()), stock)
    for doc in EDGE:
        try:
            want: Any = _stock_cls(name)(**kwargs).score_document(doc)
        except Exception as e:
            want = (type(e), str(e))
        try:
            got: Any = getattr(ours, name)(**kwargs).score_document(doc)
        except Exception as e:
            got = (type(e), str(e))
        assert repr(got) == repr(want), doc


def test_filter_chain_and_score_stage() -> None:
    make = _frames(NONEMPTY)["string[pyarrow]"]
    filters_stock = [
        nc_h.WordCountFilter(min_words=3),
        nc_r.RepeatingTopNGramsFilter(n=2),
        nc_h.UrlsFilter(),
    ]
    filters_ours = [
        ours.WordCountFilter(min_words=3),
        ours.RepeatingTopNGramsFilter(n=2),
        ours.UrlsFilter(),
    ]
    fields = ["a", "b", "c"]
    _same(
        _run(
            ours.ScoreFilter(filters_ours, score_field=fields, invert=[False, True, False]), make()
        ),
        _run(
            nc_filters.ScoreFilter(filters_stock, score_field=fields, invert=[False, True, False]),
            make(),
        ),
    )
    _same(
        _run(ours.Score(filters_ours, score_field=fields), make()),
        _run(nc_filters.Score(filters_stock, score_field=fields), make()),
    )
    scored = _run(nc_filters.Score(filters_stock, score_field=fields), make())
    _same(
        _run(
            ours.Filter(filters_ours, filter_field=fields, invert=[True, False, False]),
            scored.copy(),
        ),
        _run(
            nc_filters.Filter(filters_stock, filter_field=fields, invert=[True, False, False]),
            scored.copy(),
        ),
    )


MODIFIERS: list[tuple[str, dict[str, Any]]] = [
    ("BoilerPlateStringModifier", {}),
    ("BoilerPlateStringModifier", {"remove_if_at_top_or_bottom": False}),
    ("MarkdownRemover", {}),
    ("NewlineNormalizer", {}),
    ("QuotationRemover", {}),
    ("UrlRemover", {}),
    ("LineRemover", {"patterns": ["line", "", "• one"]}),
    ("Slicer", {"left": "the", "right": 40, "include_left": False}),
    ("Slicer", {"left": -30, "right": "a", "strip": False}),
]


def _stock_modifier(name: str, kwargs: dict[str, Any]) -> Any:
    if name == "FastTextLabelModifier":
        return NcLabel(**kwargs)
    if name == "UnicodeReformatter":
        from nemo_curator.stages.text.modifiers.unicode import UnicodeReformatter

        return UnicodeReformatter(**kwargs)
    return getattr(nc_mod, name)(**kwargs)


@pytest.mark.parametrize("frame", ["arrow-table", "string[pyarrow]", "object"])
@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        *MODIFIERS,
        ("FastTextLabelModifier", {"label": "__label__hq"}),
        ("UnicodeReformatter", {}),
        (
            "UnicodeReformatter",
            {"normalization": "NFC", "uncurl_quotes": True, "fix_line_breaks": True},
        ),
    ],
    ids=lambda v: str(v),
)
def test_modify_matches_nemo(frame: str, name: str, kwargs: dict[str, Any]) -> None:
    if name == "UnicodeReformatter":
        pytest.importorskip("ftfy")
    docs = [*DOCS, "Ã©tÃ© &amp; café", "\x1b[31mred\x1b[0m"]
    make = _frames(docs)[frame]
    stock = _run(nc_mod.Modify(_stock_modifier(name, kwargs)), make())
    _same(_run(ours.Modify(_stock_modifier(name, kwargs)), make()), stock)
    _same(_run(ours.Modify(getattr(ours, name)(**kwargs)), make()), stock)
    _same(_run(nc_mod.Modify(getattr(ours, name)(**kwargs)), make()), stock)
    out = _run(ours.Modify(getattr(ours, name)(**kwargs), output_fields="new"), make())
    _same(out, _run(nc_mod.Modify(_stock_modifier(name, kwargs), output_fields="new"), make()))


def _lid_model() -> str:
    path = os.environ.get("PYROID_LID_MODEL", "/data/models/lid.176.ftz")
    if not os.path.exists(path):
        pytest.skip("no fastText language-ID model")
    return path


def test_fasttext_lang_id() -> None:
    path = _lid_model()
    make = _frames(NONEMPTY)["arrow-table"]
    for lang in (None, "en", "EN", "fr"):
        f = ours.FastTextLangId(model_path=path, min_langid_score=0.3, lang=lang)
        f.load_model()
        want = _batch(make()).to_pandas()
        want["s"] = want["text"].apply(f.score_document)
        want = want[want["s"].apply(f.keep_document)]
        got = _run(ours.ScoreFilter(ours.FastTextLangId(path, 0.3, lang), score_field="s"), make())
        _same(got, want)


def test_fasttext_lang_id_matches_nemo() -> None:
    path = _lid_model()
    pytest.importorskip("fasttext", reason="fasttext is not installed")
    from nemo_curator.stages.text.filters.fasttext import FastTextLangId as NcLangId

    make = _frames(NONEMPTY)["arrow-table"]
    for lang in (None, "en"):
        stock = _run(nc_filters.ScoreFilter(NcLangId(path, 0.3, lang), score_field="s"), make())
        _same(
            _run(ours.ScoreFilter(ours.FastTextLangId(path, 0.3, lang), score_field="s"), make()),
            stock,
        )


def test_fasttext_quality() -> None:
    path = _lid_model()  # any fastText model works: the "high-quality" label is __label__en here
    make = _frames(NONEMPTY)["string[pyarrow]"]
    f = ours.FastTextQualityFilter(model_path=path, label="__label__en", alpha=3, seed=7)
    f.load_model()
    want = make()
    want["s"] = want["text"].apply(f.score_document)
    np.random.seed(7)
    want = want[want["s"].apply(f.keep_document)]
    got = _run(
        ours.ScoreFilter(ours.FastTextQualityFilter(path, "__label__en", 3, 7), score_field="s"),
        make(),
    )
    _same(got, want)


def test_fallbacks_warn_once_and_match() -> None:
    make = _frames(NONEMPTY)["arrow-table"]
    ours._WARNED.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = _run(
            ours.ScoreFilter(ours.UrlsFilter(url_regex=r"https?://\S+"), score_field="s"), make()
        )
        _run(ours.ScoreFilter(ours.UrlsFilter(url_regex=r"https?://\S+"), score_field="s"), make())
    want = _run(
        nc_filters.ScoreFilter(nc_h.UrlsFilter(url_regex=r"https?://\S+"), score_field="s"), make()
    )
    _same(got, want)
    fallbacks = [w for w in caught if w.category is pyroid.FallbackWarning]
    assert len(fallbacks) == 1
    assert "url_regex" in str(fallbacks[0].message)


def test_subclasses_run_their_own_code() -> None:
    class Mine(ours.WordCountFilter):
        def score_document(self, text: str) -> int:
            return 7

    make = _frames(NONEMPTY)["object"]
    out = _run(ours.ScoreFilter(Mine(min_words=1), score_field="s"), make())
    assert set(out["s"]) == {7}


def test_top_ngram_ties_follow_this_process() -> None:
    # Ties are broken by set order, which depends on the hash seed: same process, same answer.
    docs = [
        " ".join(random.Random(i).sample(["a", "b", "c", "d", "e", "f", "g"], 7))
        for i in range(300)
    ]
    f_stock = nc_r.RepeatingTopNGramsFilter(n=2)
    f_ours = ours.RepeatingTopNGramsFilter(n=2)
    assert [repr(f_ours.score_document(d)) for d in docs] == [
        repr(f_stock.score_document(d)) for d in docs
    ]


def test_stage_registry_keeps_nemo_classes() -> None:
    from nemo_curator.stages.base import get_stage_class

    assert get_stage_class("ScoreFilter") is nc_filters.ScoreFilter
    assert get_stage_class("Modify") is nc_mod.Modify


def test_threads_argument() -> None:
    make = _frames(NONEMPTY)["arrow-table"]
    want = _run(nc_filters.ScoreFilter(nc_h.WordCountFilter(min_words=5), score_field="s"), make())
    for threads in (None, 1, 2):
        _same(
            _run(
                ours.ScoreFilter(
                    nc_h.WordCountFilter(min_words=5), score_field="s", threads=threads
                ),
                make(),
            ),
            want,
        )
    with pytest.raises(ValueError, match="threads"):
        ours.ScoreFilter(nc_h.WordCountFilter(), threads=0)


@pytest.mark.parametrize(("name", "kwargs"), HEURISTIC + REPETITION, ids=lambda v: str(v))
def test_batches_do_not_call_nemo_code(
    monkeypatch: pytest.MonkeyPatch, name: str, kwargs: dict[str, Any]
) -> None:
    # On documents NeMo scores without raising, the stages never call NeMo's per-row methods.
    cls = _stock_cls(name)

    def boom(*_: Any) -> Any:
        raise AssertionError(f"{name}: NeMo's code ran")

    make = _frames(NONEMPTY)["arrow-table"]
    want = _run(nc_filters.ScoreFilter(cls(**kwargs), score_field="s"), make())
    monkeypatch.setattr(cls, "score_document", boom)
    monkeypatch.setattr(cls, "keep_document", boom)
    _same(_run(ours.ScoreFilter(cls(**kwargs), score_field="s"), make()), want)
    # In NeMo's stage, pyroid's filter scores each row itself; keep_document is NeMo's.
    monkeypatch.undo()
    monkeypatch.setattr(cls, "score_document", boom)
    _same(
        _run(nc_filters.ScoreFilter(getattr(ours, name)(**kwargs), score_field="s"), make()), want
    )


@pytest.mark.parametrize(("name", "kwargs"), MODIFIERS, ids=lambda v: str(v))
def test_modify_does_not_call_nemo_code(
    monkeypatch: pytest.MonkeyPatch, name: str, kwargs: dict[str, Any]
) -> None:
    cls = getattr(nc_mod, name)
    make = _frames(DOCS)["string[pyarrow]"]
    want = _run(nc_mod.Modify(cls(**kwargs)), make())

    def boom(*_: Any) -> Any:
        raise AssertionError(f"{name}: NeMo's code ran")

    monkeypatch.setattr(cls, "modify_document", boom)
    _same(_run(ours.Modify(cls(**kwargs)), make()), want)
    _same(_run(nc_mod.Modify(getattr(ours, name)(**kwargs)), make()), want)


def test_other_batch_paths_do_not_call_per_row_code(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: Any) -> Any:
        raise AssertionError("per-row code ran")

    path = _lid_model()
    make = _frames(NONEMPTY)["arrow-table"]
    want_lid = _run(ours.ScoreFilter(ours.FastTextLangId(path, 0.3, "en"), score_field="s"), make())
    np.random.seed(3)
    want_q = _run(
        ours.ScoreFilter(ours.FastTextQualityFilter(path, "__label__en", 3, 3), score_field="q"),
        make(),
    )
    want_u = _run(ours.Modify(ours.UnicodeReformatter()), make())
    stock_filters = [nc_h.WordCountFilter(min_words=3), nc_r.RepeatingTopNGramsFilter(n=3)]
    want_score = _run(nc_filters.Score(stock_filters, score_field=["a", "b"]), make())
    want_filter = _run(nc_filters.Filter(stock_filters, filter_field=["a", "b"]), want_score.copy())
    for cls in (
        ours.FastTextLangId,
        ours.FastTextQualityFilter,
        nc_h.WordCountFilter,
        nc_r.RepeatingTopNGramsFilter,
    ):
        monkeypatch.setattr(cls, "score_document", boom)
        monkeypatch.setattr(cls, "keep_document", boom)
    monkeypatch.setattr(ours.UnicodeReformatter, "modify_document", boom)
    _same(
        _run(ours.ScoreFilter(ours.FastTextLangId(path, 0.3, "en"), score_field="s"), make()),
        want_lid,
    )
    _same(
        _run(
            ours.ScoreFilter(
                ours.FastTextQualityFilter(path, "__label__en", 3, 3), score_field="q"
            ),
            make(),
        ),
        want_q,
    )
    _same(_run(ours.Modify(ours.UnicodeReformatter()), make()), want_u)
    _same(_run(ours.Score(stock_filters, score_field=["a", "b"]), make()), want_score)
    _same(_run(ours.Filter(stock_filters, filter_field=["a", "b"]), want_score.copy()), want_filter)
