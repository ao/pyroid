# mypy: disallow-subclassing-any=False
"""pyroid.integrations.datatrove: drop-in steps give the same documents as datatrove's own.

Live comparisons need datatrove (and spaCy for its filters, ftfy for its formatter, fasttext-predict
and the pinned lid.176 model for its LanguageFilter); each test skips when what it needs is missing.
"""

from __future__ import annotations

import copy
import functools
import inspect
import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("datatrove")
pytest.importorskip("regex")  # datatrove.pipeline.filters imports regex and tokenizers,
pytest.importorskip("tokenizers")  # which datatrove does not declare
pytest.importorskip("numpy")

from datatrove.data import Document  # type: ignore[import-not-found,import-untyped,unused-ignore]

import pyroid
import pyroid.filters
from pyroid.integrations import datatrove as pdt

DATA = Path(os.environ.get("PYROID_DATA", "/data"))
DEV = DATA / "fineweb" / "fineweb-dev-10k.parquet"
LID_BIN = DATA / "models" / "lid.176.bin"
LID_FTZ = DATA / "models" / "lid.176.ftz"

EDGE = [
    "",
    " ",
    "\n\n\n",
    "word",
    "The cat sat on the mat. " * 3,
    "lorem ipsum dolor sit amet, the and of to be that have with.\n" * 6,
    "function() { return 1; } the and of to be that have with this line.\n" * 6,
    "Please enable javascript to view this page and the rest of it.\nThe story ends here.\n" * 4,
    "â€œmojibakeâ€\x9d is fixed &amp; so is this ﬁ ligature and "
    "\uff26\uff35\uff2c\uff2c\uff37\uff29\uff24\uff34\uff28.",
    "• a bullet line\n• another bullet line\n• and a third one here\n" * 10,
    "Short.\nLines.\nOnly.\nHere.\n" * 20,
    "repeat this line exactly\n" * 30,
    "Hello world! This is a perfectly normal English paragraph, with the usual words that you "
    "would expect to see in a web document, and it has to be long enough to pass the minimum word "
    "count of fifty words that the Gopher quality filter applies, so here are some more words to "
    "get there; the end is coming soon, and with it the last sentence of this test paragraph. "
    "Thank you.",
]


@functools.cache
def _texts_cached(n: int) -> tuple[str, ...]:
    out = list(EDGE)
    if DEV.exists():
        pq = pytest.importorskip("pyarrow.parquet")
        out += pq.read_table(DEV, columns=["text"]).slice(0, n).column("text").to_pylist()
    return tuple(out)


def _texts(n: int = 300) -> list[str]:
    return list(_texts_cached(n))


_STOCK: dict[str, Any] = {}  # stock datatrove results, computed once per case


def _docs(texts: list[str]) -> list[Document]:
    return [Document(text=t, id=str(i)) for i, t in enumerate(texts)]


def _run_filter(step: Any, docs: list[Document]) -> list[tuple[Any, str, dict[str, Any]]]:
    """Every doc's (filter result, text after, metadata after), through the step's own run()."""
    from datatrove.pipeline.filters.base_filter import get_filter_result  # type: ignore[import-not-found,import-untyped,unused-ignore]

    docs = copy.deepcopy(docs)
    out = []
    for batch_start in range(0, len(docs), step.batch_size):
        batch = docs[batch_start : batch_start + step.batch_size]
        results = step.filter_batch(batch)
        for d, r in zip(batch, results, strict=True):
            keep, reason = get_filter_result(r)
            out.append(((bool(keep), reason), d.text, dict(d.metadata)))
    return out


def _datatrove_words_cache_clear() -> None:
    from datatrove.utils.text import split_into_parts  # type: ignore[import-not-found,import-untyped,unused-ignore]

    split_into_parts.cache_clear()


# ------------------------------------------------------------------------------------- signatures


@pytest.mark.parametrize(
    "name",
    [
        "FTFYFormatter",
        "LanguageFilter",
        "GopherRepetitionFilter",
        "GopherQualityFilter",
        "C4QualityFilter",
        "FineWebQualityFilter",
        "MinhashDedupSignature",
    ],
)
def test_subclass_and_signature(name: str) -> None:
    import datatrove.pipeline.dedup as dd  # type: ignore[import-not-found,import-untyped,unused-ignore]
    import datatrove.pipeline.filters as df  # type: ignore[import-not-found,import-untyped,unused-ignore]
    import datatrove.pipeline.formatters as dfmt  # type: ignore[import-not-found,import-untyped,unused-ignore]

    ours = getattr(pdt, name)
    theirs = next(getattr(m, name) for m in (df, dd, dfmt) if hasattr(m, name))
    assert issubclass(ours, theirs)
    assert ours.__name__ == theirs.__name__
    sig = inspect.signature(ours)
    base = list(inspect.signature(theirs).parameters.values())
    params = list(sig.parameters.values())
    assert params[: len(base)] == base
    extra = [p.name for p in params[len(base) :]]
    assert set(extra) <= {"model_path", "batch_size", "threads"}
    assert {"batch_size", "threads"} <= set(extra)


def test_bad_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        pdt.GopherQualityFilter(batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        pdt.FTFYFormatter(batch_size=True)


def test_steps_pickle_before_use() -> None:
    steps = [
        pdt.FTFYFormatter(),
        pdt.GopherRepetitionFilter(),
        pdt.GopherQualityFilter(),
        pdt.C4QualityFilter(),
        pdt.FineWebQualityFilter(),
        pdt.LanguageFilter(languages=["en"], model_path="/nonexistent/lid.176.bin"),
    ]
    # Our own freshly built objects (datatrove pickles steps to its workers): nothing untrusted.
    for s in steps:
        assert type(pickle.loads(pickle.dumps(s))) is type(s)


# ------------------------------------------------------------------------------------------ ftfy


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # datatrove's gentle defaults
        {  # ftfy's default config (the benchmark chain)
            "fix_latin_ligatures": True,
            "fix_character_width": True,
            "uncurl_quotes": True,
            "fix_line_breaks": True,
            "normalization": "NFC",
        },
        {"unescape_html": False, "normalization": "NFKC", "remove_control_chars": False},
    ],
)
@pytest.mark.parametrize("batch_size", [1, 7, 1000])
def test_ftfy_formatter_matches_datatrove(kwargs: dict[str, Any], batch_size: int) -> None:
    pytest.importorskip("ftfy")
    from datatrove.pipeline.formatters import FTFYFormatter  # type: ignore[import-not-found,import-untyped,unused-ignore]

    texts = _texts()
    stock = FTFYFormatter(**kwargs)
    want = [stock.format(t) for t in texts]
    ours = pdt.FTFYFormatter(**kwargs, batch_size=batch_size)
    got = [d.text for d in ours.run(iter(_docs(texts)))]
    assert got == want
    assert [ours.format(t) for t in texts[:20]] == want[:20]
    assert ours.config == tuple(stock.config)


def test_ftfy_formatter_positional_args() -> None:
    a = pdt.FTFYFormatter(False, True, True)
    assert a.config.unescape_html is False
    assert a.config.normalization is None


# ------------------------------------------------------------------------------------- filters


FILTER_CASES = [
    ("GopherRepetitionFilter", {}),
    ("GopherQualityFilter", {}),
    ("GopherQualityFilter", {"min_doc_words": 10, "stop_words": ["the", "a"], "min_stop_words": 1}),
    ("C4QualityFilter", {}),
    ("C4QualityFilter", {"filter_no_terminal_punct": False}),
    ("C4QualityFilter", {"split_paragraph": False, "remove_citations": False}),
    ("FineWebQualityFilter", {}),
    ("FineWebQualityFilter", {"short_line_thr": 0.5, "stop_chars": (".", "!")}),
]


@pytest.mark.parametrize(("name", "kwargs"), FILTER_CASES)
@pytest.mark.parametrize(("batch_size", "threads"), [(1, 1), (1000, None), (13, 4)])
def test_filters_match_datatrove(
    name: str, kwargs: dict[str, Any], batch_size: int, threads: Any
) -> None:
    pytest.importorskip("spacy")
    import datatrove.pipeline.filters as df  # type: ignore[import-not-found,import-untyped,unused-ignore]
    from datatrove.pipeline.filters.base_filter import get_filter_result  # type: ignore[import-not-found,import-untyped,unused-ignore]

    texts = _texts()
    key = f"{name}{sorted(kwargs.items())}"
    if key not in _STOCK:
        stock = getattr(df, name)(**kwargs)
        want = []
        for d in _docs(texts):
            _datatrove_words_cache_clear()
            keep, reason = get_filter_result(stock.filter(d))
            want.append(((bool(keep), reason), d.text, dict(d.metadata)))
        _STOCK[key] = want
    want = _STOCK[key]
    ours = getattr(pdt, name)(**kwargs, batch_size=batch_size, threads=threads)
    assert _run_filter(ours, _docs(texts)) == want
    assert any(not w[0][0] for w in want) or len(texts) < 50


def test_filter_single_doc_api() -> None:
    f = pdt.GopherQualityFilter()
    d = Document(text="too short", id="x")
    assert f.filter(d) == (False, "gopher_short_doc")


def test_c4_rewrites_kept_text_only() -> None:
    f = pdt.C4QualityFilter(filter_no_terminal_punct=False, min_num_sentences=1)
    d = Document(text="Keep this sentence here.\nJavaScript must be enabled for this.\n", id="0")
    assert f.filter_batch([d]) == [True]
    assert d.text == "Keep this sentence here."


# ------------------------------------------------------------------------------------ languages

# Native languages, one per word tokenizer kind pyroid ports: spaCy rule-based (Latin, Cyrillic,
# Arabic scripts), IndicNLP, and WhitespaceTokenizer (spaCy xx words, regex sentences).
NATIVE_LANGUAGES = {
    "fra_Latn": None,
    "deu_Latn": None,
    "rus_Cyrl": None,
    "arb_Arab": None,
    "hin_Deva": "indicnlp",
    "kat_Geor": None,
}
FW2 = DATA / "fineweb2"
CHINESE = "我们今天去公园散步\uff0c天气很好。"  # cmn_Hani: datatrove tokenizes it with jieba
GEORGIAN = [
    "საქართველო მდებარეობს კავკასიაში. დედაქალაქი არის თბილისი, ძველი და ლამაზი ქალაქი.\n",
    "ეს არის მოკლე ტექსტი.\n",
    "ჩვენ ვსწავლობთ ქართულ ენას და ვკითხულობთ წიგნებს ყოველ დღე, რადგან ეს საინტერესოა.\n",
    "• პუნქტი ერთი\n• პუნქტი ორი\n",
]


@functools.cache
def _language_texts(lang: str) -> tuple[str, ...]:
    """The language's FineWeb-2 slice (first 80 docs) when /data has it, plus generated texts."""
    out = ["", "x", "1, 2 , 000.\n\n", "word " * 60]
    if (FW2 / f"{lang}.parquet").exists():
        pq = pytest.importorskip("pyarrow.parquet")
        docs = pq.read_table(FW2 / f"{lang}.parquet", columns=["text"]).column("text")
        sample = docs.slice(0, 80).to_pylist()
        out += sample
        # Rewritten copies: short lines, repeated lines and a truncated document get dropped.
        out += [t[: len(t) // 7] for t in sample[:10]]
        out += ["\n".join(t.split(". ")) for t in sample[10:20]]
        out += [t.split("\n")[0] + "\n" + t.split("\n")[0] * 3 for t in sample[20:30]]
    else:
        out += [
            "".join(GEORGIAN[i % len(GEORGIAN)] for i in range(k, k + n))
            for k in range(4)
            for n in (1, 3, 12, 40)
        ]
    return tuple(out)


def _stop_words(texts: tuple[str, ...]) -> list[str]:
    """The 12 most frequent whitespace tokens: stand-ins for the language's stop words."""
    from collections import Counter

    return [w for w, _ in Counter(w for t in texts for w in t.split()).most_common(12)]


def _stock_results(step: Any, docs: list[Document]) -> list[Any]:
    """datatrove's per-doc (result, text, metadata), or the exception it raises for that doc."""
    from datatrove.pipeline.filters.base_filter import get_filter_result  # type: ignore[import-not-found,import-untyped,unused-ignore]

    want: list[Any] = []
    for d in copy.deepcopy(docs):
        _datatrove_words_cache_clear()
        try:
            keep, reason = get_filter_result(step.filter(d))
        except ValueError as e:  # IndicNLP's sentence splitter vs C4's text.index (see filters.pyi)
            want.append((type(e), str(e)))
            continue
        want.append(((bool(keep), reason), d.text, dict(d.metadata)))
    return want


LANGUAGE_FILTER_CASES = [
    ("GopherRepetitionFilter", {}),
    ("GopherQualityFilter", {"stop_words": "sample", "min_doc_words": 20}),
    ("C4QualityFilter", {"filter_no_terminal_punct": False}),
    ("C4QualityFilter", {"split_paragraph": False, "min_num_sentences": 2}),
    ("FineWebQualityFilter", {}),
]


@pytest.mark.thread_unsafe
@pytest.mark.parametrize("lang", sorted(NATIVE_LANGUAGES))
@pytest.mark.parametrize(("name", "kwargs"), LANGUAGE_FILTER_CASES)
def test_filters_native_languages_match_datatrove(
    lang: str, name: str, kwargs: dict[str, Any]
) -> None:
    """Native non-English codes run in Rust (no FallbackWarning) with datatrove's results."""
    pytest.importorskip("spacy")
    if backend := NATIVE_LANGUAGES[lang]:
        pytest.importorskip(backend)
    import datatrove.pipeline.filters as df  # type: ignore[import-not-found,import-untyped,unused-ignore]

    assert pyroid.filters.languages()[lang].native
    texts = _language_texts(lang)
    kwargs = dict(kwargs, language=lang)
    if kwargs.get("stop_words") == "sample":
        kwargs["stop_words"] = _stop_words(texts)
    docs = _docs(list(texts))
    key = f"{lang}/{name}{sorted(kwargs.items())}"
    if key not in _STOCK:
        _STOCK[key] = _stock_results(getattr(df, name)(**kwargs), docs)
    want = _STOCK[key]
    raising = [i for i, w in enumerate(want) if isinstance(w[0], type)]
    ok = [i for i in range(len(docs)) if i not in raising]
    with warnings.catch_warnings():
        warnings.simplefilter("error", pyroid.FallbackWarning)
        for batch_size, threads in ((1000, None), (7, 2)):
            ours = getattr(pdt, name)(**kwargs, batch_size=batch_size, threads=threads)
            assert _run_filter(ours, [docs[i] for i in ok]) == [want[i] for i in ok]
        for i in raising:  # the same exception, for the document datatrove raises on
            with pytest.raises(want[i][0]) as exc:  # pyroid adds a note naming the document
                ours.filter_batch(copy.deepcopy([docs[i]]))
            assert str(exc.value) == want[i][1]
    assert len({w[0] for w in want if not isinstance(w[0], type)}) > 1  # kept and dropped docs


@pytest.mark.thread_unsafe
def test_filter_non_native_language_falls_back() -> None:
    """A code pyroid does not tokenize (cmn_Hani: jieba) runs datatrove's own code, warned once."""
    pytest.importorskip("spacy")
    pytest.importorskip("jieba")
    import datatrove.pipeline.filters as df  # type: ignore[import-not-found,import-untyped,unused-ignore]

    assert not pyroid.filters.languages()["cmn_Hani"].native
    text = CHINESE * 30
    stock = df.GopherQualityFilter(language="cmn_Hani", min_doc_words=5)
    want = stock.filter(Document(text=text, id="0"))
    pdt._WARNED.clear()
    ours = pdt.GopherQualityFilter(language="cmn_Hani", min_doc_words=5)
    with pytest.warns(pyroid.FallbackWarning, match="language='cmn_Hani'"):
        got = ours.filter_batch([Document(text=text, id="0")])
    assert got == [want]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ours.filter_batch([Document(text=text, id="1")])  # once per process


@pytest.mark.thread_unsafe
def test_filter_unknown_language_falls_back_to_datatrove_error() -> None:
    """A code datatrove has no tokenizer for: datatrove's own ValueError, after the warning."""
    import datatrove.pipeline.filters as df  # type: ignore[import-not-found,import-untyped,unused-ignore]

    with pytest.raises(ValueError, match="doesn't have a tokenizer assigned") as stock:
        df.GopherQualityFilter(language="und_Zzzz").filter(Document(text="a b c", id="0"))
    pdt._WARNED.clear()
    ours = pdt.GopherQualityFilter(language="und_Zzzz")
    with (
        pytest.warns(pyroid.FallbackWarning, match="language='und_Zzzz'"),
        pytest.raises(ValueError, match="doesn't have a tokenizer assigned") as got,
    ):
        ours.filter_batch([Document(text="a b c", id="0")])
    assert str(got.value) == str(stock.value)


# ------------------------------------------------------------------------------------------- LID


def _stock_language_filter(model: Path, **kw: Any) -> Any:
    """datatrove's LanguageFilter + FastTextLID code on fasttext-predict, numpy probabilities."""
    import numpy as np

    fasttext = pytest.importorskip("fasttext")
    from datatrove.pipeline.filters import LanguageFilter  # type: ignore[import-not-found,import-untyped,unused-ignore]
    from datatrove.utils.lid import FT176LID  # type: ignore[import-not-found,import-untyped,unused-ignore]

    class Stock(LanguageFilter):
        def __new__(
            cls, *a: Any, **k: Any
        ) -> Any:  # skip the fasttext-numpy2-wheel/fasteners check
            return object.__new__(cls)

    class NumpyProbs:
        def __init__(self, m: Any) -> None:
            self.m = m

        def predict(
            self, text: str, k: int = 1, threshold: float = 0.0, on_unicode_error: str = "strict"
        ) -> Any:
            labels, probs = self.m.predict(text, k, threshold, on_unicode_error)
            return labels, np.asarray(probs)

    f = Stock(**kw)
    lid = FT176LID(f.languages)
    if "ft-model" not in _STOCK:
        _STOCK["ft-model"] = NumpyProbs(fasttext.load_model(str(model)))
    lid._model = _STOCK["ft-model"]
    f.model = lid
    return f


def _model() -> Path:
    for m in (LID_BIN, LID_FTZ):
        if m.exists():
            return m
    pytest.skip("no pinned lid.176 model under $PYROID_DATA/models")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"languages": ["en"], "language_threshold": 0.65},
        {"languages": "en", "language_threshold": 0.3},  # below 0.5: all labels are needed
        {"languages": ["en", "fr", "de"], "language_threshold": 0.5},
        {"languages": None, "language_threshold": 0.65},
        {"languages": ["en"], "label_only": True},
        {"languages": ["en"], "keep_top_pairs_threshold": 0.01},
        {"languages": None, "keep_top_pairs_threshold": 0.001},
    ],
)
@pytest.mark.parametrize("batch_size", [1, 1000])
def test_language_filter_matches_datatrove(kwargs: dict[str, Any], batch_size: int) -> None:
    model = _model()
    texts = [
        *_texts(),
        "Bonjour tout le monde, comment allez-vous ?",
        "Hallo Welt, wie geht es dir?",
    ]
    key = f"lid{sorted(kwargs.items())}"
    if key not in _STOCK:
        _STOCK[key] = _run_filter(_stock_language_filter(model, **kwargs), _docs(texts))
    want = _STOCK[key]
    ours = pdt.LanguageFilter(**kwargs, model_path=model, batch_size=batch_size)
    got = _run_filter(ours, _docs(texts))
    assert got == want
    assert ours.model.model.queue == type(ours.model.model.queue)()


# --------------------------------------------------------------------------------------- MinHash


def _sig_files(folder: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(folder)): p.read_bytes() for p in sorted(folder.rglob("*.minhash.sig"))
    }


def _minhash_texts() -> list[str]:
    base = _texts(200)
    # near-duplicates: one changed word, appended noise, a truncated copy
    dups = [t.replace(" the ", " a ", 1) for t in base[13:60]] + [t + " extra" for t in base[60:90]]
    return base + dups + ["", "two words", "one two three four five"]


@pytest.mark.parametrize(
    "cfg",
    [
        {"hash_config": ("sha1", 64), "num_buckets": 14, "hashes_per_bucket": 8, "n_grams": 5},
        {
            "hash_config": ("xxhash", 32),
            "num_buckets": 4,
            "hashes_per_bucket": 3,
            "n_grams": 3,
            "seed": 7,
        },
    ],
)
@pytest.mark.parametrize("batch_size", [1, 1000])
def test_minhash_signature_files_identical(
    cfg: dict[str, Any], batch_size: int, tmp_path: Path
) -> None:
    pytest.importorskip("spacy")
    from datatrove.pipeline.dedup import MinhashConfig, MinhashDedupSignature  # type: ignore[import-not-found,import-untyped,unused-ignore]
    from datatrove.utils.hashing import HashConfig  # type: ignore[import-not-found,import-untyped,unused-ignore]

    cfg = dict(cfg)  # parametrize shares the dict across cases
    hc = cfg.pop("hash_config")
    if hc[0] == "xxhash":
        pytest.importorskip("xxhash")
    config = MinhashConfig(hash_config=HashConfig(hash_fc=hc[0], precision=hc[1]), **cfg)
    texts = _minhash_texts()
    for rank in (0, 3):
        MinhashDedupSignature(output_folder=str(tmp_path / "stock"), config=config).run(
            iter(_docs(texts)), rank=rank, world_size=4
        )
        pdt.MinhashDedupSignature(
            output_folder=str(tmp_path / "ours"), config=config, batch_size=batch_size
        ).run(iter(_docs(texts)), rank=rank, world_size=4)
    stock, ours = _sig_files(tmp_path / "stock"), _sig_files(tmp_path / "ours")
    assert stock.keys() == ours.keys()
    assert len(stock) == 2 * config.num_buckets
    assert stock == ours


def test_minhash_later_stages_on_pyroid_signatures(tmp_path: Path) -> None:
    """datatrove's stages 2-4 on pyroid's signature files remove the same docs as on its own."""
    pytest.importorskip("spacy")
    from datatrove.pipeline.dedup import (  # type: ignore[import-not-found,import-untyped,unused-ignore]
        MinhashConfig,
        MinhashDedupCluster,
        MinhashDedupFilter,
        MinhashDedupSignature,
    )
    from datatrove.pipeline.dedup.minhash import MinhashDedupBuckets  # type: ignore[import-not-found,import-untyped,unused-ignore]

    config = MinhashConfig()
    texts = _minhash_texts()

    def removed(sig_step: type, root: Path) -> list[str]:
        sig_step(output_folder=str(root / "sigs"), config=config).run(iter(_docs(texts)), 0, 1)
        for b in range(config.num_buckets):
            MinhashDedupBuckets(
                input_folder=str(root / "sigs"), output_folder=str(root / "buckets"), config=config
            ).run(None, b, config.num_buckets)
        MinhashDedupCluster(
            input_folder=str(root / "buckets"), output_folder=str(root / "ids"), config=config
        ).run(None, 0, 1)
        kept = MinhashDedupFilter(input_folder=str(root / "ids")).run(iter(_docs(texts)), 0, 1)
        kept_ids = {d.id for d in kept}
        return [d.id for d in _docs(texts) if d.id not in kept_ids]

    want = removed(MinhashDedupSignature, tmp_path / "stock")
    got = removed(pdt.MinhashDedupSignature, tmp_path / "ours")
    assert got == want
    assert len(want) >= 20  # the near-duplicates above are found


def test_minhash_skip_existing_sigs(tmp_path: Path) -> None:
    texts = _minhash_texts()[:50]
    pdt.MinhashDedupSignature(output_folder=str(tmp_path)).run(iter(_docs(texts)), 0, 1)
    first = _sig_files(tmp_path)
    # with skip_existing_sigs the data is not even read
    pdt.MinhashDedupSignature(output_folder=str(tmp_path), skip_existing_sigs=True).run(
        iter(()), 0, 1
    )
    assert _sig_files(tmp_path) == first
    assert pdt.MinhashDedupSignature(output_folder=str(tmp_path)).skip_existing_sigs is False


@pytest.mark.thread_unsafe
@pytest.mark.parametrize("lang", [*sorted(NATIVE_LANGUAGES), "cmn_Hani"])
def test_minhash_signature_files_identical_languages(lang: str, tmp_path: Path) -> None:
    """FineWeb-2's MinhashConfig with the language's tokenizer: the same files; native codes warn
    nothing, cmn_Hani (jieba) runs datatrove's step with a FallbackWarning."""
    pytest.importorskip("spacy")
    pytest.importorskip("xxhash")
    native = pyroid.filters.languages()[lang].native
    backend = NATIVE_LANGUAGES.get(lang) or (None if native else "jieba")
    if backend:
        pytest.importorskip(backend)
    from datatrove.pipeline.dedup import MinhashConfig, MinhashDedupSignature  # type: ignore[import-not-found,import-untyped,unused-ignore]
    from datatrove.utils.hashing import HashConfig  # type: ignore[import-not-found,import-untyped,unused-ignore]

    config = MinhashConfig(hash_config=HashConfig(hash_fc="xxhash", precision=64))
    texts = list(_language_texts(lang)) if native else [CHINESE * k for k in range(1, 40)]
    texts += [t[:-3] + "!" for t in texts[4:30]]  # near-duplicates
    MinhashDedupSignature(output_folder=str(tmp_path / "stock"), config=config, language=lang).run(
        iter(_docs(texts)), rank=1, world_size=2
    )
    pdt._WARNED.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pdt.MinhashDedupSignature(
            output_folder=str(tmp_path / "ours"), config=config, language=lang, batch_size=17
        ).run(iter(_docs(texts)), rank=1, world_size=2)
    fallbacks = [w for w in caught if w.category is pyroid.FallbackWarning]
    assert len(fallbacks) == (0 if native else 1)
    stock, ours = _sig_files(tmp_path / "stock"), _sig_files(tmp_path / "ours")
    assert len(stock) == config.num_buckets
    assert stock == ours


@pytest.mark.thread_unsafe
def test_minhash_native_language_without_datatrove_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """datatrove's constructor loads the word tokenizer (spaCy): pyroid does not need it for a
    native code, so a missing one is only an error for the other codes."""
    pytest.importorskip("spacy")
    from datatrove.pipeline.dedup import minhash as dt_minhash  # type: ignore[import-not-found,import-untyped,unused-ignore]

    texts = list(_language_texts("fra_Latn"))
    dt_minhash.MinhashDedupSignature(
        output_folder=str(tmp_path / "stock"), language="fra_Latn"
    ).run(iter(_docs(texts)), 0, 1)

    def missing(language: Any) -> Any:
        raise ImportError("spaCy is not installed")

    monkeypatch.setattr(dt_minhash, "load_word_tokenizer", missing)
    with pytest.raises(ImportError, match="spaCy"):
        pdt.MinhashDedupSignature(output_folder=str(tmp_path / "x"), language="cmn_Hani")
    ours = pdt.MinhashDedupSignature(output_folder=str(tmp_path / "ours"), language="fra_Latn")
    assert ours.word_tokenizer is None
    with warnings.catch_warnings():
        warnings.simplefilter("error", pyroid.FallbackWarning)
        ours.run(iter(_docs(texts)), 0, 1)
    assert _sig_files(tmp_path / "ours") == _sig_files(tmp_path / "stock")


# ------------------------------------------------------------------------- LocalPipelineExecutor


def test_executor_end_to_end(tmp_path: Path) -> None:
    """A small pipeline in datatrove's LocalPipelineExecutor: same JSONL output, stock vs pyroid."""
    pytest.importorskip("spacy")
    pytest.importorskip("ftfy")
    from datatrove.executor import LocalPipelineExecutor  # type: ignore[import-not-found,import-untyped,unused-ignore]
    from datatrove.pipeline import filters as df  # type: ignore[import-not-found,import-untyped,unused-ignore]
    from datatrove.pipeline.formatters import FTFYFormatter  # type: ignore[import-not-found,import-untyped,unused-ignore]
    from datatrove.pipeline.readers import JsonlReader  # type: ignore[import-not-found,import-untyped,unused-ignore]
    from datatrove.pipeline.writers import JsonlWriter  # type: ignore[import-not-found,import-untyped,unused-ignore]

    src = tmp_path / "in"
    src.mkdir()
    import json

    texts = _texts(400)
    for shard in range(2):
        with open(src / f"{shard}.jsonl", "w", encoding="utf-8") as f:
            for i, t in enumerate(texts[shard::2]):
                if t.strip():  # datatrove's readers skip empty texts anyway
                    f.write(json.dumps({"text": t, "id": f"{shard}-{i}"}) + "\n")

    def run(mod: Any, name: str) -> dict[str, list[str]]:
        steps = [
            JsonlReader(str(src)),
            mod.FTFYFormatter() if mod is pdt else FTFYFormatter(),
            mod.GopherRepetitionFilter(
                exclusion_writer=JsonlWriter(str(tmp_path / name / "x_rep"))
            ),
            mod.GopherQualityFilter(exclusion_writer=JsonlWriter(str(tmp_path / name / "x_q"))),
            mod.C4QualityFilter(filter_no_terminal_punct=False),
            mod.FineWebQualityFilter(),
            JsonlWriter(str(tmp_path / name / "out")),
        ]
        LocalPipelineExecutor(
            pipeline=steps, tasks=2, workers=1, logging_dir=str(tmp_path / name / "logs")
        ).run()
        out = {}
        for d in ("out", "x_rep", "x_q"):
            files = sorted((tmp_path / name / d).glob("*.jsonl*"))
            out[d] = [line for p in files for line in _read_jsonl(p)]
        return out

    stock = run(df, "stock")
    ours = run(pdt, "ours")
    assert ours == stock
    assert stock["out"]
    assert stock["x_q"]


def _read_jsonl(p: Path) -> list[str]:
    import gzip

    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]
