# Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc.; modified.
"""pyroid.dedup beyond English: ``language=`` and the opt-in Python tokenizer.

- The Python-tokenizer path with a duck-typed tokenizer (no datatrove needed): container
  kinds, Arrow nulls, ``MinHashDedup.add``/``fit``, ``threads=``, consistency with the
  pre-tokenized hooks.
- Live datatrove (``pytest.importorskip``): ``MinhashDedupSignature(language=...)``
  signatures and the removals of its stages 1-3, for native languages (Hypothesis documents
  in each script) and for the installed segmenters (``python_tokenizer=True``), and the
  FineWeb-2 slices with FineWeb-2's MinHash configuration.

Errors and the fallback warning are covered with the filters' (test_filters_multilingual.py),
which share the machinery.
"""

# Other scripts' letters and punctuation on purpose.

from __future__ import annotations

import os
import struct
import tempfile
import warnings
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import pyroid.dedup as dd
import pyroid.filters as pf
from pyroid._pyroid import _testing
from pyroid._pyroid import dedup as native_dedup

# The documents and languages of the filters' suite (same directory, imported by path so
# the suite runs without python/tests on sys.path).
_HERE = Path(__file__).parent

pytestmark = [
    pytest.mark.thread_unsafe(reason="datatrove, spaCy and the once-per-process warning"),
    pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning"),
    pytest.mark.filterwarnings("ignore::SyntaxWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::FutureWarning"),
]

slow = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


class Split:
    """A duck-typed word tokenizer: ``str.split``."""

    def word_tokenize(self, text: str) -> list[str]:
        return text.split()


SPLIT = Split()
DOCS = [
    "The quiet river runs past the old stone mill, slowly and without a sound.",
    "the quiet river runs past the old stone mill slowly and without a sound",
    "Something else entirely: a café in İstanbul serves 12,000 cups for £100.",
    "short",
    "",
    "Der Fluss fließt still an der alten Mühle vorbei, langsam und ohne Laut.",
    "der fluss fliesst still an der alten muehle vorbei langsam und ohne laut",
]
SMALL: dict[str, Any] = {"n_grams": 2, "num_buckets": 4, "hashes_per_bucket": 2}


def split_tokens(texts: list[str], cfg: dd.MinhashConfig) -> list[list[str]]:
    """What the Python path feeds the native hooks: ``word_tokenize(simplify_text(t))``."""
    return [SPLIT.word_tokenize(dd.simplify_text(t, cfg.norm_config)) for t in texts]


# ---- the Python path with a duck-typed tokenizer -----------------------------------


def test_python_path_signatures_equal_the_token_hooks() -> None:
    cfg = dd.MinhashConfig(**SMALL)
    want = native_dedup._signatures_from_tokens(split_tokens(DOCS, cfg), dd._native_config(cfg))
    assert dd.minhash_signatures(DOCS, cfg, language=SPLIT) == want
    assert dd.minhash_signatures(tuple(DOCS), cfg, language=SPLIT) == want
    for d, w in zip(DOCS, want, strict=True):
        assert dd.minhash_signatures(d, cfg, language=SPLIT) == w
    assert want[4] is None  # the empty document has no shingles
    # Shorthands and kwargs still apply.
    assert dd.minhash_signatures(DOCS, language=SPLIT, **SMALL) == want


def test_python_path_dedup() -> None:
    cfg = dd.MinhashConfig(**SMALL)
    ref = dd.MinHashDedup(cfg)
    ref._state._add_tokens(split_tokens(DOCS, cfg))
    keep = ref.keep_mask()
    assert keep != [True] * len(DOCS)  # the near-duplicates really are found
    assert dd.minhash_dedup(DOCS, cfg, language=SPLIT) == keep
    assert dd.minhash_dedup(tuple(DOCS), cfg, language=SPLIT) == tuple(keep)
    assert dd.minhash_dedup(DOCS[0], cfg, language=SPLIT) is True
    d = dd.MinHashDedup(cfg, language=SPLIT)
    assert d.language is SPLIT
    assert d.add(DOCS[:3]) == range(3)
    assert d.add(DOCS[3:]) == range(3, len(DOCS))
    assert d.keep_mask() == keep
    assert d.duplicates() == ref.duplicates()
    assert d.clusters() == ref.clusters()
    assert [d.signature(i) for i in range(len(DOCS))] == [
        ref.signature(i) for i in range(len(DOCS))
    ]
    # fit replaces the index.
    assert d.fit(DOCS) == keep
    assert len(d) == len(DOCS)
    assert d.fit(DOCS[:2]) == [True, False] == keep[:2]
    assert len(d) == 2
    assert "language=" in repr(d)
    assert "python_tokenizer" not in repr(d)


@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("chunks", [1, 3])
def test_python_path_arrow(large: bool, chunks: int) -> None:
    cfg = dd.MinhashConfig(**SMALL)
    values: list[str | None] = [*DOCS[:2], None, *DOCS[2:], None]
    col = _testing.column(values, large=large, chunks=chunks, name="doc")
    plain = [v for v in values if v is not None]
    sigs = iter(dd.minhash_signatures(plain, cfg, language=SPLIT))
    want = [None if v is None else next(sigs) for v in values]
    assert dd.minhash_signatures(col, cfg, language=SPLIT) == want
    keep = iter(dd.minhash_dedup(plain, cfg, language=SPLIT))
    mask = dd.minhash_dedup(col, cfg, language=SPLIT)
    assert mask.to_pylist() == [None if v is None else next(keep) for v in values]
    assert mask.type == "bool"
    assert mask.name == "doc"
    assert mask.num_chunks == chunks
    d = dd.MinHashDedup(cfg, language=SPLIT)
    assert d.add(col) == range(len(values))
    assert d.signature(2) is None  # nulls get an index and no signature
    assert d.fit(col).to_pylist() == mask.to_pylist()


def test_python_path_pyarrow(pa: Any) -> None:
    cfg = dd.MinhashConfig(**SMALL)
    arr = pa.chunked_array([DOCS[:3], [None, *DOCS[3:]]], type=pa.string_view())
    mask = dd.minhash_dedup(arr, cfg, language=SPLIT)
    assert pa.chunked_array(mask).to_pylist() == [
        *dd.minhash_dedup(DOCS[:3], cfg, language=SPLIT)[:3],
        None,
        *dd.minhash_dedup(DOCS, cfg, language=SPLIT)[3:],
    ]


@pytest.mark.parametrize("threads", [None, 1, 2, 4])
def test_python_path_threads(threads: int | None) -> None:
    docs = [d + f" {i % 7}" for i in range(40) for d in DOCS]
    want = dd.minhash_signatures(docs, language=SPLIT, threads=1, **SMALL)
    assert dd.minhash_signatures(docs, language=SPLIT, threads=threads, **SMALL) == want
    keep = dd.minhash_dedup(docs, language=SPLIT, threads=1, **SMALL)
    assert dd.minhash_dedup(docs, language=SPLIT, threads=threads, **SMALL) == keep
    d = dd.MinHashDedup(language=SPLIT, threads=threads, **SMALL)
    d.add(docs, threads=threads)
    assert d.keep_mask() == keep


def test_python_path_input_errors() -> None:
    import pyroid

    with pytest.raises(pyroid.InputError):
        dd.minhash_signatures(["ok", 3], language=SPLIT)  # type: ignore[list-item]
    with pytest.raises(UnicodeEncodeError):
        dd.minhash_dedup(["fine", "bad \ud800"], language=SPLIT)

    class Bad:
        def word_tokenize(self, text: str) -> Any:
            return [1, 2]

    with pytest.raises(pyroid.InputError):
        dd.minhash_signatures(["two words"], language=Bad())


def test_native_language_codes_select_the_tokenizer() -> None:
    # A native code needs no datatrove; "eng" (the default) is datatrove's default.
    docs = ["Der Fluss fließt still: z.B. an der Mühle (vgl. S. 12f.) vorbei.", *DOCS]
    assert dd.minhash_signatures(docs, language="eng") == dd.minhash_signatures(docs)
    assert dd.minhash_signatures(docs, language="en") == dd.minhash_signatures(docs)
    # English splits "cannot" and "gonna" (spaCy exceptions); German does not.
    odd = "i cannot say it gonna rain"
    assert dd.minhash_signatures(odd, language="deu_Latn", n_grams=1) != dd.minhash_signatures(
        odd, n_grams=1
    )
    assert dd.MinHashDedup(language="deu_Latn", n_grams=1).fit(docs) == dd.minhash_dedup(
        docs, language="deu_Latn", n_grams=1
    )
    assert repr(dd.MinHashDedup()).startswith("MinHashDedup(MinhashConfig(")
    assert "language='deu_Latn'" in repr(dd.MinHashDedup(language="deu_Latn"))


# ---- live datatrove ------------------------------------------------------------------

HASHES = [("sha1", 64), ("sha1", 32), ("xxhash", 64), ("xxhash", 32)]


class Datatrove:
    """datatrove's MinHash stages, with ``language`` (as ``MinhashDedupSignature`` takes it)."""

    def __init__(self, tmp: Path) -> None:
        pytest.importorskip("spacy")
        self.minhash: Any = pytest.importorskip("datatrove.pipeline.dedup.minhash")
        self.hashing: Any = pytest.importorskip("datatrove.utils.hashing")
        self.Document: Any = pytest.importorskip("datatrove.data").Document
        self._steps: dict[Any, Any] = {}
        self._tmp = tmp

    def config(self, hash_fc: str, precision: int, **kw: Any) -> Any:
        hc = self.hashing.HashConfig(hash_fc=hash_fc, precision=precision)
        return self.minhash.MinhashConfig(hash_config=hc, **kw)

    def step(self, cfg: Any, language: str) -> Any:
        key = (repr(cfg), language)
        if key not in self._steps:
            self._steps[key] = self.minhash.MinhashDedupSignature(
                output_folder=str(self._tmp / f"s{len(self._steps)}"), config=cfg, language=language
            )
        return self._steps[key]

    def signatures(self, cfg: Any, language: str, texts: list[str]) -> list[list[int] | None]:
        step = self.step(cfg, language)
        out: list[list[int] | None] = []
        for t in texts:
            sh = step.get_shingles(t)
            out.append(
                None if sh.size == 0 else [int(x) for b in step.get_signature(sh) for x in b]
            )
        return out

    def removed(self, cfg: Any, language: str, texts: list[str]) -> dict[str, Any]:
        """Stages 1-3 in process: removed ids, cluster ids and sizes."""
        with tempfile.TemporaryDirectory(prefix="pyroid-dt-") as w:
            sig = self.minhash.MinhashDedupSignature(
                output_folder=f"{w}/s", config=cfg, language=language
            )
            docs = [self.Document(text=t, id=str(i)) for i, t in enumerate(texts)]
            sig.run(iter(docs), 0, 1)
            for b in range(cfg.num_buckets):
                self.minhash.MinhashDedupBuckets(
                    input_folder=f"{w}/s", output_folder=f"{w}/b", config=cfg
                ).run(None, b, cfg.num_buckets)
            self.minhash.MinhashDedupCluster(
                input_folder=f"{w}/b",
                output_folder=f"{w}/r",
                config=cfg,
                save_cluster_id=True,
                save_cluster_size=True,
            ).run(None, 0, 1)

            def read(name: str, fmt: str) -> list[Any]:
                p = Path(w) / "r" / name
                return list(struct.iter_unpack(fmt, p.read_bytes())) if p.exists() else []

            ids, sizes = [-1] * len(texts), [1] * len(texts)
            for d, c in read("000000.clusters", "<2I"):
                ids[d] = c
            for d, s in read("000000.sizes", "<2I"):
                sizes[d] = s
            return {
                "removed": [d for (d,) in read("000000.remove", "<I")],
                "cluster_ids": ids,
                "cluster_sizes": sizes,
            }


@pytest.fixture(scope="module")
def dt(tmp_path_factory: pytest.TempPathFactory) -> Datatrove:
    return Datatrove(tmp_path_factory.mktemp("datatrove"))


def _filters_suite() -> Any:
    """test_filters_multilingual: its script pools and languages (one source of truth)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_pyroid_filters_multilingual", _HERE / "test_filters_multilingual.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SUITE = _filters_suite()
POOLS: dict[str, list[str]] = _SUITE.POOLS
NATIVE: dict[str, str] = _SUITE.NATIVE
#: The languages of the python_tokenizer=True tests (Japanese among them, native or not).
PYTHON_PATH: dict[str, tuple[str, tuple[str, ...]]] = _SUITE.PYTHON_PATH


@st.composite
def families(draw: st.DrawFn, pool: str) -> list[str]:
    """Near-duplicate documents in one script: case, whitespace, punctuation and word edits."""
    words = POOLS[pool]
    docs: list[str] = []
    for _ in range(draw(st.integers(1, 3))):
        ws = draw(st.lists(st.sampled_from(words), max_size=25))
        docs.append(" ".join(ws))
        for _ in range(draw(st.integers(0, 3))):
            w = list(ws)
            op = draw(st.sampled_from(["upper", "space", "punct", "edit", "swap"]))
            if w:
                i = draw(st.integers(0, len(w) - 1))
                if op == "upper":
                    w[i] = w[i].upper()
                elif op == "punct":
                    w[i] += draw(st.sampled_from([",", "!", "...", "?!", "،", "。", "»", "।"]))
                elif op == "edit":
                    w[i] = draw(st.sampled_from(words))
                elif op == "swap":
                    j = draw(st.integers(0, len(w) - 1))
                    w[i], w[j] = w[j], w[i]
            sep = draw(st.sampled_from([" ", "  ", "\n", "\xa0", "　"])) if op == "space" else " "
            docs.append(sep.join(w))
    return draw(st.permutations(docs))


def _need(lang: str) -> None:
    if pf.languages()[lang].tokenizer == "IndicNLPTokenizer":
        pytest.importorskip("indicnlp")
    if pf.languages()[lang].arg == "zh":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # jieba: invalid escapes
            pytest.importorskip("jieba")  # datatrove's reference needs it
    if pf.languages()[lang].arg == "ja":
        pytest.importorskip("sudachipy")  # datatrove's reference needs SudachiPy and its dictionary
        pytest.importorskip("sudachidict_core")


def check(
    dt: Datatrove,
    lang: str,
    texts: list[str],
    hc: tuple[str, int],
    n: int,
    python: bool | None = None,
) -> None:
    """pyroid's signatures and removals for ``texts`` are datatrove's.

    ``python``: tokenize with ``python_tokenizer=True`` (default: when ``lang`` is not native).
    """
    if python is None:
        python = not pf.languages()[lang].native
    kw: dict[str, Any] = {"n_grams": n, "num_buckets": 6, "hashes_per_bucket": 2}
    ref = dt.config(*hc, **kw)
    cfg = dd.MinhashConfig(hash_config=dd.HashConfig(hc[1], hc[0]), **kw)
    want = dt.signatures(ref, lang, texts)
    assert dd.minhash_signatures(texts, cfg, language=lang, python_tokenizer=python) == want
    stages = dt.removed(ref, lang, texts)
    d = dd.MinHashDedup(cfg, language=lang, python_tokenizer=python)
    d.add(texts)
    assert [d.signature(i) for i in range(len(texts))] == want
    assert d.duplicates() == stages["removed"]
    assert d.cluster_ids() == stages["cluster_ids"]
    assert d.cluster_sizes() == stages["cluster_sizes"]
    keep = dd.minhash_dedup(texts, cfg, language=lang, python_tokenizer=python)
    assert keep == [i not in stages["removed"] for i in range(len(texts))]


@slow
@given(data=st.data())
def test_native_languages_match_datatrove(dt: Datatrove, data: st.DataObject) -> None:
    lang = data.draw(st.sampled_from(sorted(NATIVE)), label="language")
    _need(lang)
    texts = data.draw(families(NATIVE[lang]), label="texts")
    hc = data.draw(st.sampled_from(HASHES), label="hash")
    n = data.draw(st.sampled_from([1, 2, 5]), label="n_grams")
    check(dt, lang, texts, hc, n)


@pytest.fixture(scope="module")
def python_langs() -> list[str]:
    """The ``PYTHON_PATH`` languages whose segmenter (and dictionary) is installed."""
    out = [lang for lang, (_, modules) in PYTHON_PATH.items() if _SUITE.installed(modules)]
    if not out:
        pytest.skip("no segmenter installed (sudachipy + sudachidict-core, pyvi, pythainlp)")
    return out


@slow
@given(data=st.data())
def test_python_tokenizer_matches_datatrove(
    dt: Datatrove, python_langs: list[str], data: st.DataObject
) -> None:
    lang = data.draw(st.sampled_from(python_langs), label="language")
    texts = data.draw(families(PYTHON_PATH[lang][0]), label="texts")
    hc = data.draw(st.sampled_from(HASHES), label="hash")
    n = data.draw(st.sampled_from([1, 2, 5]), label="n_grams")
    check(dt, lang, texts, hc, n, python=True)


def test_python_tokenizer_on_native_language_is_identical(dt: Datatrove) -> None:
    texts = ["Der Fluss fließt still: z.B. an der Mühle (vgl. S. 12f.) vorbei.", *DOCS]
    for lang in ("deu_Latn", "rus_Cyrl", "eng"):
        want = dd.minhash_signatures(texts, language=lang, n_grams=2)
        assert dd.minhash_signatures(texts, language=lang, n_grams=2, python_tokenizer=True) == want
        wt: Any = pytest.importorskip("datatrove.utils.word_tokenizers")
        tok = wt.load_word_tokenizer(lang)  # a datatrove WordTokenizer instance
        assert dd.minhash_signatures(texts, language=tok, n_grams=2) == want


DATA = Path(os.environ.get("PYROID_PARITY_DATA", "/data"))
SLICES = [
    "arb_Arab", "cmn_Hani", "deu_Latn", "fra_Latn", "hin_Deva", "jpn_Jpan", "rus_Cyrl", "spa_Latn",
]  # fmt: skip


@pytest.mark.parametrize("lang", SLICES)
def test_fineweb2_slices_match_datatrove(dt: Datatrove, lang: str) -> None:
    pq: Any = pytest.importorskip("pyarrow.parquet")
    path = DATA / "fineweb2" / f"{lang}.parquet"
    if not path.is_file():
        pytest.skip(f"FineWeb-2 slice not found: {path}")
    python = not pf.languages()[lang].native
    if python:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pytest.importorskip({"cmn_Hani": "jieba", "jpn_Jpan": "sudachipy"}[lang])
    _need(lang)
    table = pq.read_table(path, columns=["text"])
    texts: list[str] = table.column("text").to_pylist()
    # fineweb-2-pipeline.py's MinhashConfig (xxhash/64, 14 buckets of 8, 5-grams).
    ref = dt.config("xxhash", 64, num_buckets=14, hashes_per_bucket=8, n_grams=5)
    cfg = dd.MinhashConfig(num_buckets=14, hashes_per_bucket=8, n_grams=5)
    want = dt.signatures(ref, lang, texts)
    got = dd.minhash_signatures(texts, cfg, language=lang, python_tokenizer=python)
    bad = [i for i, (g, w) in enumerate(zip(got, want, strict=True)) if g != w]
    assert not bad, f"{len(bad)}/{len(texts)} signatures differ for {lang}"
    # The same from the Arrow column, and the removals of datatrove's stages.
    col = table.column("text")
    assert dd.minhash_signatures(col, cfg, language=lang, python_tokenizer=python) == want
    stages = dt.removed(ref, lang, texts)
    keep = dd.minhash_dedup(col, cfg, language=lang, python_tokenizer=python).to_pylist()
    assert keep == [i not in stages["removed"] for i in range(len(texts))]
