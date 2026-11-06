# mypy: disallow-subclassing-any=False
"""pyroid.dedup.SentenceDedup and the sentence-dedup drop-ins against live datatrove 0.10.1.

The comparisons with datatrove need datatrove and spaCy (its English word tokenizer) and skip
cleanly without them; the API tests (container kinds, nulls, threads, errors) do not.
"""

# The inputs are full of look-alike characters on purpose.
# ruff: noqa: RUF001

from __future__ import annotations

import inspect
import os
import pickle
import random
import struct
import warnings
from pathlib import Path
from typing import Any, Self

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import pyroid
import pyroid.dedup as dedup

try:
    import spacy  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: F401
    from datatrove.data import Document  # type: ignore[import-not-found,import-untyped,unused-ignore]
    from datatrove.pipeline.dedup import sentence_dedup as dt_sd  # type: ignore[import-not-found,import-untyped,unused-ignore]
    from datatrove.utils.hashing import HashConfig as DtHashConfig  # type: ignore[import-not-found,import-untyped,unused-ignore]
    from datatrove.utils.text import TextNormConfig as DtNormConfig  # type: ignore[import-not-found,import-untyped,unused-ignore]

    HAVE_DT = True
except ImportError:  # pragma: no cover - the oracle env has them
    HAVE_DT = False

needs_dt = pytest.mark.skipif(not HAVE_DT, reason="needs datatrove 0.10.1 and spaCy")
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

DATA = Path(os.environ.get("PYROID_DATA", "/data"))
DEV = DATA / "fineweb" / "fineweb-dev-10k.parquet"

slow = settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
HASHES = [("sha1", 64), ("sha1", 32), ("xxhash", 64), ("xxhash", 32)]
SENTENCES = [
    "The cat sat on the mat.", "It was raining again!", "Was it?", "Prices rose 3.5% in May.",
    "Mr. Smith went to Washington.", "U.S. stocks fell sharply.", "Café au lait, s'il vous plaît.",
    "ΣΟΦΙΑ είναι εδώ.", "日本語の文です。", "😀 Emoji here!", "A line without a full stop",
    "e.g. this one...", '"Quoted," he said.', "(Parenthetical.)", "Numbers 1,000 and 12.000!",
    "Tab\tseparated words.", "naïve résumé coöperate", "-- dashes — and – more.", "Ok", "?!",
    "", " ", "...", "A.", "B. C.", "İstanbul’da ﬁx ﬂy.", "Monday January 1st.", "x² ½ ³",
]  # fmt: skip
SEPS = [" ", "  ", "\n", "\n\n", " \n", "\t", "", "\r\n", " ", " ", "\x0c", " \n\n "]
EXAMPLE: dict[str, Any] = {
    "n_sentences": 3,
    "split_sentences": True,
    "only_dedup_in_index": True,
    "min_doc_words": 50,
}

DOCS = [
    "One. Two. Three. Four.",
    "One. Two. Three. Five.",
    "Unique text here, nothing repeated.",
    "",
    "   ",
    "Four. One. Two. Three.",
]
# n=3, no word/sentence minimums: doc 1 loses "One. Two. Three." (doc 0 has it first) and doc 5
# loses its last three sentences; the empty doc is dropped (datatrove drops empty texts).
NO_MIN: dict[str, Any] = {"min_doc_words": 0, "min_num_sentences": 0}
WANT_TEXT = [DOCS[0], "Five.", DOCS[2], "", "   ", "Four."]
WANT_KEEP = [True, True, True, False, True, True]


def dt_config(hc: tuple[str, int] = ("xxhash", 64), norm: Any = None, **kw: Any) -> Any:
    norm_cfg = DtNormConfig(**vars(norm)) if norm is not None else DtNormConfig()
    return dt_sd.SentDedupConfig(
        hash_config=DtHashConfig(precision=hc[1], hash_fc=hc[0]), norm_config=norm_cfg, **kw
    )


def py_config(hc: tuple[str, int] = ("xxhash", 64), norm: Any = None, **kw: Any) -> Any:
    return dedup.SentDedupConfig(
        hash_config=dedup.HashConfig(hc[1], hc[0]),
        norm_config=norm if norm is not None else dedup.TextNormConfig(),
        **kw,
    )


class MemWriter:
    """An exclusion writer that records (rank, id, text)."""

    def __init__(self) -> None:
        self.docs: list[tuple[int, str, Any]] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def write(self, doc: Any, rank: int = 0) -> None:
        self.docs.append((rank, doc.id, doc.text))


def tree(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def run_stages(
    mod: Any,
    texts: list[str],
    cfg: Any,
    work: Path,
    *,
    fw: int = 1,
    ranks: int = 1,
    index_from: list[str] | None = None,
    language: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """The three stages of ``mod`` (datatrove's module or pyroid's drop-ins) via run()."""
    lang = {} if language is None else {"language": language}
    bounds = [len(texts) * r // ranks for r in range(ranks + 1)]

    def docs(r: int) -> Any:
        return iter(Document(text=texts[i], id=str(i)) for i in range(bounds[r], bounds[r + 1]))

    index = None
    if index_from is not None:
        mod.SentenceDedupSignature(
            output_folder=str(work / "isig"), config=cfg, **lang, **extra
        ).run(iter(Document(text=t, id=f"i{i}") for i, t in enumerate(index_from)), 0, 1)
        mod.SentenceDedupBuildIndex(
            data_folder=str(work / "isig"),
            output_folder=str(work / "index"),
            index_name="idx",
            config=cfg,
            **extra,
        ).run(None, 0, 1)
        index = str(work / "index")
    for r in range(ranks):
        mod.SentenceDedupSignature(
            output_folder=str(work / "sig"), finder_workers=fw, config=cfg, **lang, **extra
        ).run(docs(r), r, ranks)
    for r in range(fw):
        mod.SentenceFindDedups(
            data_folder=str(work / "sig"),
            output_folder=str(work / "dup"),
            index_folder=index,
            config=cfg,
            **extra,
        ).run(None, r, fw)
    writer = MemWriter()
    kept: list[tuple[str, str]] = []
    for r in range(ranks):
        step = mod.SentenceDedupFilter(
            data_folder=str(work / "dup"), config=cfg, exclusion_writer=writer, **lang, **extra
        )
        kept.extend((d.id, d.text) for d in step.run(docs(r), r, ranks))
    return {
        "sig": tree(work / "sig"),
        "dup": tree(work / "dup"),
        "index": tree(work / "index"),
        "kept": kept,
        "excl": writer.docs,
    }


def dup_pairs(files: dict[str, bytes]) -> list[tuple[int, int]]:
    return sorted(p for b in files.values() for p in struct.iter_unpack("<IH", b))


@st.composite
def corpora(draw: st.DrawFn) -> list[str]:
    """Small corpora whose documents share sentences, with varied whitespace."""
    extra = st.text(st.characters(exclude_categories=["Cs"]), max_size=25)
    pool = draw(st.lists(st.sampled_from(SENTENCES) | extra, min_size=1, max_size=8))
    docs = []
    for _ in range(draw(st.integers(1, 6))):
        k = draw(st.integers(0, 10))
        doc = draw(st.sampled_from(["", *SEPS]))
        for _ in range(k):
            doc += draw(st.sampled_from(pool)) + draw(st.sampled_from(SEPS))
        docs.append(doc)
    return docs


norm_configs = st.builds(
    dedup.TextNormConfig,
    lowercase=st.booleans(),
    norm_whitespace=st.booleans(),
    remove_punctuation=st.booleans(),
    norm_unicode_diacritics=st.booleans(),
    norm_numbers=st.booleans(),
    norm_weekdays=st.booleans(),
    norm_monthnames=st.booleans(),
)


# ------------------------------------------------------------------------------- live datatrove


@needs_dt
@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    texts=corpora(),
    n=st.integers(1, 4),
    split=st.booleans(),
    hc=st.sampled_from(HASHES),
    norm=norm_configs,
)
def test_window_hashes_match_get_hashes(
    texts: list[str], n: int, split: bool, hc: tuple[str, int], norm: Any
) -> None:
    kw: dict[str, Any] = {"n_sentences": n, "split_sentences": split}
    step = dt_sd.SentenceDedupSignature(
        output_folder="/nonexistent", config=dt_config(hc, norm, **kw)
    )
    want = [[(h, s) for h, _, s in step.get_hashes(Document(text=t, id="x"), 0)] for t in texts]
    native = dedup._native_sent(py_config(hc, norm, **kw), "eng")
    got = [[tuple(x) for x in doc] for doc in dedup._native._sentence_hashes(texts, native)]
    assert got == want


@needs_dt
@slow
@given(
    texts=corpora(),
    n=st.integers(1, 3),
    split=st.booleans(),
    hc=st.sampled_from(HASHES),
    mins=st.sampled_from([(0, 0, 0), (5, 2, 0), (50, 3, 0), (0, 0, 3), (3, 1, 2)]),
    fw=st.sampled_from([1, 2]),
)
def test_stages_match_live_datatrove(
    texts: list[str],
    n: int,
    split: bool,
    hc: tuple[str, int],
    mins: tuple[int, int, int],
    fw: int,
    tmp_path_factory: Any,
) -> None:
    import pyroid.integrations.datatrove as pdt

    kw: dict[str, Any] = {
        "n_sentences": n,
        "split_sentences": split,
        "min_doc_words": mins[0],
        "min_num_sentences": mins[1],
        "min_words_to_remove_span": mins[2],
    }
    work = tmp_path_factory.mktemp("sd")
    want = run_stages(dt_sd, texts, dt_config(hc, **kw), work / "dt", fw=fw)
    got = run_stages(pdt, texts, dt_config(hc, **kw), work / "pr", fw=fw, batch_size=2)
    assert got == want
    # The in-memory API makes the same decisions.
    d = dedup.SentenceDedup(py_config(hc, **kw))
    text, keep = d.fit(texts)
    kept = dict(want["kept"])
    assert keep == [str(i) in kept for i in range(len(texts))]
    assert [t for t, k in zip(text, keep, strict=True) if k] == [t for _, t in want["kept"]]
    assert d.duplicates() == dup_pairs(want["dup"])


@needs_dt
def test_whitespace_only_last_sentence_shifts_like_datatrove(tmp_path: Path) -> None:
    """Stage 1 drops spaCy's whitespace-only last sentence, stage 3 keeps it as a span."""
    import pyroid.integrations.datatrove as pdt

    texts = ["A cat. A dog. A cow.  ", "Hi there. A cat. A dog. A cow.\n\n", "A cat. A dog. A cow."]
    cfg = dt_config(**NO_MIN)
    want = run_stages(dt_sd, texts, cfg, tmp_path / "dt")
    assert run_stages(pdt, texts, cfg, tmp_path / "pr") == want
    assert dict(want["kept"]) == {"0": texts[0], "1": "Hi there.\n\n"}
    assert dedup.sentence_dedup(texts, **NO_MIN).text == [texts[0], "Hi there.\n\n", ""]


@needs_dt
@pytest.mark.parametrize(
    "case",
    [
        {"fw": 2, "ranks": 2},
        {"fw": 3, "ranks": 1, "hc": ("sha1", 32)},
        {"fw": 1, "ranks": 3, "split_sentences": False},
        {"index": True},
        {"index": True, "only_dedup_in_index": False},
        {"min_words_to_remove_span": 6},
    ],
    ids=["fw2-ranks2", "fw3-sha1-32", "lines-ranks3", "index-only", "index-all", "mwrs6"],
)
def test_drop_ins_write_the_same_files(case: dict[str, Any], tmp_path: Path) -> None:
    import pyroid.integrations.datatrove as pdt

    case = dict(case)
    fw, ranks, hc = case.pop("fw", 1), case.pop("ranks", 1), case.pop("hc", ("xxhash", 64))
    index = case.pop("index", False)
    texts = _corpus(600)
    cfg = dt_config(hc, **{**EXAMPLE, **case})
    kw: dict[str, Any] = {"fw": fw, "ranks": ranks}
    if index:
        kw["index_from"], texts = texts[:300], texts[300:]
    want = run_stages(dt_sd, texts, cfg, tmp_path / "dt", **kw)
    got = run_stages(pdt, texts, cfg, tmp_path / "pr", batch_size=64, threads=2, **kw)
    assert got["sig"] == want["sig"]
    assert got["dup"] == want["dup"]
    assert got["index"] == want["index"]
    assert got["kept"] == want["kept"]
    assert got["excl"] == want["excl"]
    assert want["dup"], "the corpus has duplicate windows"
    assert any(t != texts[int(i)] for i, t in want["kept"]), "some text was changed"


def _corpus(n: int, seed: int = 7) -> list[str]:
    """Edge cases, dev docs (if available) and repeated multi-sentence spans."""
    rng = random.Random(seed)
    base: list[str] = []
    if DEV.exists():
        pq = pytest.importorskip("pyarrow.parquet")
        base = pq.read_table(DEV, columns=["text"]).slice(0, n).column("text").to_pylist()
    while len(base) < n:
        base.append(" ".join(rng.choice(SENTENCES) for _ in range(rng.randint(0, 40))))
    spans = [
        "This exact paragraph appears many times. It is boilerplate! Nobody reads it.",
        "Subscribe to our newsletter.\nFollow us on social media.\nAll rights reserved.",
        "Click here to learn more. Terms and conditions apply. Offer ends soon.",
    ]
    out = []
    for t in base:
        r = rng.random()
        if r < 0.4:
            at = rng.choice([0, len(t), *[i for i, c in enumerate(t) if c == "\n"][:20]])
            t = t[:at] + rng.choice(["\n", " ", "\n\n"]) + rng.choice(spans) + "\n" + t[at:]
        elif r < 0.45:
            t = rng.choice(spans) + rng.choice(["", "  ", "\n"])
        elif r < 0.5:
            t = rng.choice(["", " ", "\n", t + "  "])
        out.append(t)
    return out


@needs_dt
def test_executor_example_pipeline(tmp_path: Path) -> None:
    """datatrove's examples/sentence_deduplication.py in LocalPipelineExecutor, stock vs pyroid."""
    import json

    from datatrove.executor import LocalPipelineExecutor  # type: ignore[import-not-found,import-untyped,unused-ignore]

    import pyroid.integrations.datatrove as pdt

    texts = _corpus(300, seed=11)
    cfg = dt_sd.SentDedupConfig(**EXAMPLE)

    def reader(data: Any, rank: int = 0, world_size: int = 1) -> Any:
        for i in range(rank, len(texts), world_size):
            yield Document(text=texts[i], id=str(i))

    def run(mod: Any, name: str) -> dict[str, Any]:
        out = tmp_path / name

        def collect(data: Any, rank: int = 0, world_size: int = 1) -> Any:
            with open(out / f"kept-{rank}.jsonl", "w", encoding="utf-8") as f:
                for d in data:
                    f.write(json.dumps([d.id, d.text]) + "\n")
                    yield d

        out.mkdir()
        p1 = [
            reader,
            mod.SentenceDedupSignature(
                output_folder=str(out / "sigs"), config=cfg, finder_workers=2
            ),
        ]
        p2 = [
            mod.SentenceFindDedups(
                data_folder=str(out / "sigs"), output_folder=str(out / "dups"), config=cfg
            )
        ]
        p3 = [reader, mod.SentenceDedupFilter(data_folder=str(out / "dups"), config=cfg), collect]
        for p, tasks, logs in ((p1, 2, "l1"), (p2, 2, "l2"), (p3, 2, "l3")):
            LocalPipelineExecutor(
                pipeline=p, tasks=tasks, workers=1, logging_dir=str(out / logs)
            ).run()
        return {
            "sigs": tree(out / "sigs"),
            "dups": tree(out / "dups"),
            "kept": [(out / f"kept-{r}.jsonl").read_text("utf-8") for r in range(2)],
        }

    assert run(pdt, "pr") == run(dt_sd, "dt")


@needs_dt
def test_window_index_overflow_matches_numpy(tmp_path: Path) -> None:
    text = "\n".join(f"line {i}" for i in range(65_540))
    cfg = dt_config(split_sentences=False, n_sentences=1)
    with pytest.raises(OverflowError) as want:
        dt_sd.SentenceDedupSignature(output_folder=str(tmp_path / "a"), config=cfg).run(
            iter([Document(text="x", id="0"), Document(text=text, id="1")]), 0, 1
        )
    import pyroid.integrations.datatrove as pdt

    with pytest.raises(OverflowError) as got:
        pdt.SentenceDedupSignature(output_folder=str(tmp_path / "b"), config=cfg).run(
            iter([Document(text="x", id="0"), Document(text=text, id="1")]), 0, 1
        )
    assert str(got.value) == str(want.value) == "Python integer 65536 out of bounds for uint16"
    assert not (tmp_path / "a").exists()
    assert not (tmp_path / "b").exists()
    d = dedup.SentenceDedup(split_sentences=False, n_sentences=1)
    with pytest.raises(OverflowError, match="65536 out of bounds for uint16"):
        d.add(["ok", text])
    assert len(d) == 0  # nothing from the failed batch
    # 65,536 windows (indices 0..65535) still fit.
    assert d.add(["\n".join(f"line {i}" for i in range(65_536))]) == range(1)


@needs_dt
def test_duplicate_listed_twice_is_datatrove_error(tmp_path: Path) -> None:
    import pyroid.integrations.datatrove as pdt

    (tmp_path / "dup" / "0000").mkdir(parents=True)
    (tmp_path / "dup" / "0000" / "00000.c4_dup").write_bytes(struct.pack("<IHIH", 0, 1, 0, 1))
    cfg = dt_config(split_sentences=False, n_sentences=1)
    docs = [Document(text="a\nb\nc", id="0")]
    for mod in (dt_sd, pdt):
        step = mod.SentenceDedupFilter(data_folder=str(tmp_path / "dup"), config=cfg)
        with pytest.raises(ValueError, match="Error with duplicate line index"):
            list(step.run(iter([Document(text=d.text, id=d.id) for d in docs]), 0, 1))


@needs_dt
@settings(max_examples=60, deadline=None)
@given(
    texts=corpora(),
    pos=st.integers(0, 50),
    n=st.integers(1, 3),
    split=st.booleans(),
)
def test_surrogates(texts: list[str], pos: int, n: int, split: bool) -> None:
    """pyroid raises UnicodeEncodeError; so does datatrove, unless it hashes nothing (lines)."""
    text = "".join(texts)
    text = text[: pos % (len(text) + 1)] + "\ud800" + text[pos % (len(text) + 1) :]
    kw: dict[str, Any] = {"n_sentences": n, "split_sentences": split}
    step = dt_sd.SentenceDedupSignature(output_folder="/nonexistent", config=dt_config(**kw))
    with pytest.raises(UnicodeEncodeError):
        dedup.SentenceDedup(py_config(**kw)).add([text])
    # spaCy itself fails on surrogates; lines only fail once a window is hashed.
    if not split and len(text.splitlines()) < n:
        assert step.get_hashes(Document(text=text, id="0"), 0) == []
    else:
        with pytest.raises(UnicodeEncodeError):
            step.get_hashes(Document(text=text, id="0"), 0)


@needs_dt
def test_drop_in_signatures_and_pickle() -> None:
    import pyroid.integrations.datatrove as pdt

    for name in (
        "SentenceDedupSignature",
        "SentenceFindDedups",
        "SentenceDedupFilter",
        "SentenceDedupBuildIndex",
    ):
        ours, theirs = getattr(pdt, name), getattr(dt_sd, name)
        assert issubclass(ours, theirs)
        assert ours.__name__ == theirs.__name__
        assert name in pdt.__all__
        base = list(inspect.signature(theirs).parameters.values())
        params = list(inspect.signature(ours).parameters.values())
        assert params[: len(base)] == base
        assert [p.name for p in params[len(base) :]] == ["batch_size", "threads"]
    cfg = dt_sd.SentDedupConfig()
    steps = [
        pdt.SentenceDedupSignature(output_folder="/nonexistent", config=cfg),
        pdt.SentenceFindDedups(data_folder="/nonexistent", output_folder="/nonexistent"),
        pdt.SentenceDedupFilter(data_folder="/nonexistent", config=cfg),
        pdt.SentenceDedupBuildIndex(data_folder="/x", output_folder="/y", index_name="i"),
    ]
    for s in steps:  # our own freshly built objects: nothing untrusted
        assert type(pickle.loads(pickle.dumps(s))) is type(s)
    # datatrove reads config.hash_config from the argument: config=None fails the same way.
    with pytest.raises(AttributeError, match="hash_config"):
        pdt.SentenceDedupSignature(output_folder="/nonexistent")
    with pytest.raises(ValueError, match="batch_size"):
        pdt.SentenceDedupFilter(data_folder="/nonexistent", batch_size=0)


@needs_dt
@pytest.mark.thread_unsafe
def test_unsupported_language_falls_back(tmp_path: Path) -> None:
    import pyroid.integrations.datatrove as pdt

    texts = ["Le chat dort. Il pleut. Il fait froid. Fin."] * 3 + ["Autre chose ici."]
    cfg = dt_config(**NO_MIN)
    want = run_stages(dt_sd, texts, cfg, tmp_path / "dt", language="fra")
    pdt._WARNED.clear()
    with pytest.warns(pyroid.FallbackWarning, match="language='fra'"):
        got = run_stages(pdt, texts, cfg, tmp_path / "pr", language="fra")
    assert got == want
    # Lines need no tokenizer: no fallback, same files.
    lines = dt_config(split_sentences=False, **NO_MIN)
    want = run_stages(
        dt_sd, [t.replace(". ", ".\n") for t in texts], lines, tmp_path / "dl", language="fra"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", pyroid.FallbackWarning)
        got = run_stages(
            pdt, [t.replace(". ", ".\n") for t in texts], lines, tmp_path / "pl", language="fra"
        )
    assert got == want


@needs_dt
def test_datatrove_config_objects() -> None:
    cfg = dt_config(("sha1", 32), n_sentences=2, min_doc_words=-1)
    d = dedup.SentenceDedup(cfg)
    assert d.config == dedup.SentDedupConfig(
        n_sentences=2, min_doc_words=-1, hash_config=dedup.HashConfig(32, "sha1")
    )


# ------------------------------------------------------------------------------------ the API


def test_small_example() -> None:
    text, keep = dedup.sentence_dedup(DOCS, **NO_MIN)
    assert text == WANT_TEXT
    assert keep == WANT_KEEP
    for i in (0, 2, 3, 4):
        assert text[i] is DOCS[i]  # unchanged documents are the original objects
    d = dedup.SentenceDedup(**NO_MIN)
    assert d.add(DOCS[:2]) == range(2)
    assert d.add(DOCS[2:]) == range(2, 6)
    assert len(d) == 6
    assert d.duplicates() == [(1, 0), (5, 1)]
    assert d.filter(DOCS[:2]) == (WANT_TEXT[:2], WANT_KEEP[:2])
    assert d.filter(DOCS[2:], start=2) == (WANT_TEXT[2:], WANT_KEEP[2:])
    # The default minimums (50 words, 3 sentences) drop the changed documents.
    assert dedup.sentence_dedup(DOCS).keep == [True, False, True, False, True, False]
    d.reset()
    assert len(d) == 0
    assert d.duplicates() == []


def test_container_kinds(pa: Any) -> None:
    assert dedup.sentence_dedup(tuple(DOCS), **NO_MIN) == (tuple(WANT_TEXT), tuple(WANT_KEEP))
    assert dedup.sentence_dedup("One. Two. Three. One. Two. Three.", **NO_MIN) == (
        "One. Two. Three.",
        True,
    )
    s = "x"
    r = dedup.sentence_dedup(s)
    assert r.text is s
    assert r.keep is True
    with_nulls = [DOCS[0], None, *DOCS[1:]]
    for arr in (
        pa.array(with_nulls),
        pa.array(with_nulls, type=pa.large_string()),
        pa.array(with_nulls, type=pa.string_view()),
        pa.chunked_array([with_nulls[:3], with_nulls[3:]]),
        pa.array(["pad", *with_nulls]).slice(1),
    ):
        text, keep = dedup.sentence_dedup(arr, **NO_MIN)
        assert isinstance(text, pyroid.Column)
        assert isinstance(keep, pyroid.Column)
        want_text = [WANT_TEXT[0], None, *WANT_TEXT[1:]]
        want_keep = [WANT_KEEP[0], None, *WANT_KEEP[1:]]
        assert text.to_pylist() == want_text
        assert keep.to_pylist() == want_keep
        assert len(text) == len(keep) == len(with_nulls)
        want_type = pa.large_string() if arr.type == pa.large_string() else pa.string()
        assert pa.chunked_array(text).type == want_type
    # Nulls take an index: the in-memory store numbers them like documents.
    d = dedup.SentenceDedup(**NO_MIN)
    assert d.add(pa.array(with_nulls)) == range(7)
    assert d.duplicates() == [(2, 0), (6, 1)]


def test_without_pyarrow() -> None:
    from pyroid._pyroid import _testing

    col = _testing.column([*DOCS, None], large=False, chunks=2)
    text, keep = dedup.sentence_dedup(col, **NO_MIN)
    assert text.to_pylist() == [*WANT_TEXT, None]
    assert keep.to_pylist() == [*WANT_KEEP, None]


def test_lines_mode() -> None:
    docs = ["a\nb\nc\nd", "x\na\nb\nc", "a\nb\nc"]
    r = dedup.sentence_dedup(docs, split_sentences=False, **NO_MIN)
    assert r.text == [docs[0], "x", ""]
    assert r.keep == [True, True, False]
    # min_words_to_remove_span: only the first line of the removed run comes back.
    r = dedup.sentence_dedup(docs, split_sentences=False, min_words_to_remove_span=2, **NO_MIN)
    assert r.text == [docs[0], "x\na", "a"]


def test_language_without_tokenizer() -> None:
    with pytest.raises(NotImplementedError, match="language 'fra'"):
        dedup.SentenceDedup(language="fra")
    d = dedup.SentenceDedup(language="fra", split_sentences=False, **NO_MIN)
    assert d.fit(["a\nb\nc", "a\nb\nc"]) == (["a\nb\nc", ""], [True, False])


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"n_sentences": 0}, ValueError),
        ({"n_sentences": 1.5}, TypeError),
        ({"min_doc_words": "50"}, TypeError),
        ({"precision": 16}, ValueError),
        ({"hash_fc": "md5"}, ValueError),
        ({"bogus": 1}, TypeError),
    ],
)
def test_bad_config(kwargs: dict[str, Any], exc: type[Exception]) -> None:
    with pytest.raises(exc):
        dedup.SentenceDedup(**kwargs)


def test_bad_inputs() -> None:
    d = dedup.SentenceDedup()
    with pytest.raises(pyroid.InputError):
        d.add([1])  # type: ignore[list-item]
    with pytest.raises(UnicodeEncodeError):
        d.add(["One. Two. Three. \ud800"])
    with pytest.raises(ValueError, match="start"):
        d.filter(["x"], start=-1)
    with pytest.raises(TypeError):
        dedup.SentenceDedup(language=1)  # type: ignore[arg-type]


def test_config_forms() -> None:
    a = dedup.SentenceDedup({"n_sentences": 2, "hash_fc": "sha1"})
    b = dedup.SentenceDedup(dedup.SentDedupConfig(n_sentences=2), hash_fc="sha1")
    assert a.config == b.config
    assert a.config.hash_config == dedup.HashConfig(64, "sha1")
    assert "SentenceDedup(" in repr(a)


def _synthetic(n: int, seed: int = 3) -> list[str]:
    rng = random.Random(seed)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota"]

    def sentence() -> str:
        return " ".join(rng.choice(words) for _ in range(rng.randint(3, 9))).capitalize() + "."

    hot = [" ".join(sentence() for _ in range(3)) for _ in range(40)]
    out = []
    for _ in range(n):
        parts = [sentence() for _ in range(rng.randint(0, 25))]
        for _ in range(rng.randint(0, 2)):
            parts.insert(rng.randint(0, len(parts)), rng.choice(hot))
        out.append(rng.choice([" ", "\n", "  "]).join(parts))
    return out


@pytest.mark.parametrize("split", [True, False])
def test_threads_equivalent(split: bool) -> None:
    texts = _synthetic(800)
    want = dedup.sentence_dedup(texts, split_sentences=split, threads=1)
    for t in (None, 2, 4):
        assert dedup.sentence_dedup(texts, split_sentences=split, threads=t) == want
    assert not all(want.keep)


@pytest.mark.thread_unsafe  # heavy memory
def test_10k_documents_batched() -> None:
    texts = _synthetic(10_000, seed=5)
    want = dedup.SentenceDedup().fit(texts)
    d = dedup.SentenceDedup(threads=2)
    ranges = [d.add(texts[i : i + 1234]) for i in range(0, len(texts), 1234)]
    got_text: list[str] = []
    got_keep: list[bool] = []
    for r in ranges:
        text, keep = d.filter(texts[r.start : r.stop], start=r.start)
        got_text += text
        got_keep += keep
    assert (got_text, got_keep) == (want.text, want.keep)
    assert len(want.text) == 10_000
    assert sum(t is not s for t, s in zip(want.text, texts, strict=True)) > 1000
