"""Generate the adversarial corpus and the pinned reference outputs in ``/data/parity`` (run in Docker, oracle env).

    python -m pyroid_parity.refs all --pin          # adversarial, ftfy, fasttext, filters, dedup
    python -m pyroid_parity.refs filters --pin --workers 6

Outputs (each pinned in bench/manifest.json with the oracle versions that produced it):
    parity/adversarial.parquet   id, text, kind, source_id     mojibake / non-English / near-dup / edge docs
    parity/ftfy-ref.parquet      id, fixed                       ftfy.fix_text(text) with default config
    parity/fasttext-ref.parquet  model, id, labels, probs        top-5, on text.replace("\\n", " ") like datatrove
    parity/filters-ref.parquet   case, id, outcome, keep, reason, text_out
    parity/minhash-ref.parquet   id, signature, removed          datatrove MinhashDedup* on dev + adversarial
"""

from __future__ import annotations

import argparse
import html
import importlib.metadata as md
import os
import random
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from pyroid_parity import data as D
from pyroid_parity import specs

SEED = 20261105
OUT = {
    "adversarial": "parity/adversarial.parquet",
    "ftfy": "parity/ftfy-ref.parquet",
    "fasttext": "parity/fasttext-ref.parquet",
    "filters": "parity/filters-ref.parquet",
    "dedup": "parity/minhash-ref.parquet",
}


def log(msg: str) -> None:
    print(f"[refs {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def versions(*dists: str) -> dict[str, str]:
    out = {}
    for d in dists:
        try:
            out[d] = md.version(d)
        except md.PackageNotFoundError:
            pass
    return out


def write(table: Any, rel: str, pin: bool, **extra: Any) -> None:
    import pyarrow.parquet as pq

    out = D.DATA / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd", compression_level=3, row_group_size=10_000,
                   write_statistics=False)
    tmp.replace(out)
    log(f"wrote {rel}: {table.num_rows} rows, {out.stat().st_size / 1e6:.1f} MB")
    if pin:
        D.pin(rel, rows=table.num_rows, **extra)
        D._verified.cache_clear()


def pmap(fn: Callable[[list[Any]], list[Any]], items: list[Any], workers: int, chunk: int = 200) -> list[Any]:
    """Order-preserving chunked process map (fork: workers inherit loaded models/filters)."""
    import multiprocessing as mp

    chunks = [items[i : i + chunk] for i in range(0, len(items), chunk)]
    if workers <= 1:
        return [x for c in chunks for x in fn(c)]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        return [x for part in ex.map(fn, chunks) for x in part]


# ----------------------------------------------------------------------------------------- adversarial corpus

INJECT = ["café", "naïve", "résumé", "—", "“quoted”", "‘single’", "😀", "Ünïcödé", "Straße", "€100", "…", "ﬁne",
          "日本語", "Привет", "½", "©", " ", "señor", "jalapeño", "Zürich"]
GARBLE = ["sloppy-windows-1252", "sloppy-windows-1252", "latin-1", "sloppy-windows-1251", "macroman",
          "sloppy-windows-1250", "cp437", "iso-8859-2"]
EDGE = [
    "", " ", "\n", "\n\n\n", "\r\n\r\n", "a", "word", "...", "…", "#", "# # # #", "- - - -", "•\n•\n•",
    "!!! ???", "IS_ALPHA", "lorem ipsum dolor sit amet. " * 10, "{ curly } bracket. " * 10,
    "The the the the the the the the the the.", "x" * 5000, ("y" * 1001 + " ") * 3,
    "Line one.\nLine two.\nLine three.\nLine four.\nLine five.\nLine six.",
    "  　", "\t\t\t", "\x00\x01\x02", "😀" * 200, "ＡＢＣ　ｆｕｌｌｗｉｄｔｈ", "ﬁﬂﬀ ligatures",
    "&amp; &lt; &gt; &eacute; &#x1F600;", "\x1b[31mred\x1b[0m text", "Ã©Ã¨Ã  mojibake",
    "Javascript is required. Please enable javascript.\n" * 6, "terms of use. privacy policy. cookie policy.\n" * 6,
]


def _garble(s: str, enc: str, layers: int) -> str:
    for _ in range(layers):
        s = s.encode("utf-8").decode(enc, errors="replace")
    return s


def _inject(rng: random.Random, text: str, n: int) -> str:
    words = text.split(" ")
    for _ in range(n):
        words.insert(rng.randrange(len(words) + 1), rng.choice(INJECT))
    return " ".join(words)


def _edit(rng: random.Random, words: list[str], rate: float) -> list[str]:
    out = []
    vocab = words or ["x"]
    for w in words:
        r = rng.random()
        if r < rate / 3:
            continue  # deletion
        if r < 2 * rate / 3:
            out.append(rng.choice(vocab))  # substitution
            continue
        out.append(w)
        if r < rate:
            out.append(rng.choice(vocab))  # insertion
    return out


def build_adversarial(pin: bool) -> None:
    import ftfy.bad_codecs  # noqa: F401  registers sloppy-windows-*
    import pyarrow as pa

    rng = random.Random(SEED)
    dev = D.corpus("dev")
    rows: list[tuple[str, str, str, str | None]] = []  # id, text, kind, source_id

    # 1. non-English: FineWeb-2 test-split docs as-is
    fw2: list[tuple[str, str, str]] = []
    for rel in sorted(k for k in D.load_manifest()["files"] if k.startswith("fineweb2/")):
        lang = Path(rel).stem
        cols = D.read_parquet_columns(D.path(rel), ["id", "text"])
        for i, (did, text) in enumerate(zip(cols["id"], cols["text"], strict=True)):
            fw2.append((lang, did, text))
            rows.append((f"adv-fw2-{lang}-{i:04d}", text, f"nonenglish:{lang}", did))

    # 2. mojibake: UTF-8 bytes misread as legacy charsets, whole-doc or one span, 1-2 layers; HTML-escaped
    for n in range(600):
        if n % 2:
            lang, src_id, text = fw2[(n * 7) % len(fw2)]
        else:
            i = (n * 13 + 7) % len(dev)
            src_id, text = dev.ids[i], _inject(rng, dev.texts[i], 8)
        text = text[:1500]
        style = n % 6
        enc = GARBLE[n % len(GARBLE)]
        if style == 0:
            out, kind = _garble(text, enc, 1), f"mojibake:{enc}:1"
        elif style == 1:
            out, kind = _garble(text, "sloppy-windows-1252", 2), "mojibake:sloppy-windows-1252:2"
        elif style == 2:
            a = rng.randrange(max(1, len(text) // 2))
            out, kind = text[:a] + _garble(text[a : a + 300], enc, 1) + text[a + 300 :], f"mojibake-span:{enc}:1"
        elif style == 3:
            out, kind = html.escape(text).encode("ascii", "xmlcharrefreplace").decode(), "html-entities"
        elif style == 4:
            out, kind = text.replace(" ", " ", 20).replace("\n", "\r\n") + "\x1b[0m", "whitespace-controls"
        else:
            out, kind = _garble(text, enc, 1).replace("\n", "<br>\n"), f"mojibake-html:{enc}"
        rows.append((f"adv-moji-{n:04d}", out, kind, src_id))

    # 3. near-duplicates of dev docs (word-level edits / truncation / exact copies) for MinHash gates
    rates = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4]
    for n in range(600):
        i = (n * 31 + 3) % len(dev)
        words = dev.texts[i].split(" ")
        op = n % 9
        if op < len(rates):
            out, kind = " ".join(_edit(rng, words, rates[op])), f"neardup:edit:{rates[op]}"
        elif op == 7:
            out, kind = " ".join(words[: max(1, int(len(words) * 0.85))]), "neardup:truncate:0.85"
        else:
            out, kind = dev.texts[i] + "\n\nSubscribe to our newsletter for more updates.", "neardup:append"
        rows.append((f"adv-dup-{n:04d}", out, kind, dev.ids[i]))

    # 4. hand-written edge cases (empty, whitespace-only, punctuation-only, very long words, ...)
    rows += [(f"adv-edge-{n:03d}", t, "edge", None) for n, t in enumerate(EDGE)]

    ids, texts, kinds, srcs = map(list, zip(*rows, strict=True))
    assert len(set(ids)) == len(ids)
    table = pa.table({"id": ids, "text": texts, "kind": kinds, "source_id": srcs})
    write(table, OUT["adversarial"], pin, content_sha256=D.content_sha256(ids, texts), seed=SEED,
          generator="pyroid_parity.refs:build_adversarial", oracle=versions("ftfy"))


def all_docs() -> D.Corpus:
    dev, adv = D.corpus("dev"), D.corpus("adversarial")
    return D.Corpus("dev+adversarial", dev.ids + adv.ids, dev.texts + adv.texts, dev.meta + adv.meta)


# ---------------------------------------------------------------------------------------------------- ftfy


def _ftfy_chunk(texts: list[str]) -> list[str]:
    import ftfy

    return [ftfy.fix_text(t) for t in texts]


def ref_ftfy(pin: bool, workers: int) -> None:
    import pyarrow as pa

    docs = all_docs()
    fixed = pmap(_ftfy_chunk, docs.texts, workers)
    write(pa.table({"id": docs.ids, "fixed": fixed}), OUT["ftfy"], pin, oracle=versions("ftfy"),
          call="ftfy.fix_text(text)", corpora=["dev", "adversarial"])


# ------------------------------------------------------------------------------------------------ fastText

_FT: dict[str, Any] = {}


def _ft_chunk(job: list[tuple[str, str]]) -> list[tuple[list[str], list[float]]]:
    import fasttext

    out = []
    for model, text in job:
        if model not in _FT:
            _FT[model] = fasttext.load_model(str(D.path(specs.LID_MODELS[model])))
        labels, probs = _FT[model].predict(text.replace("\n", " "), k=specs.LID_TOPK, threshold=0.0)
        out.append((list(labels), [float(p) for p in probs]))
    return out


def ref_fasttext(pin: bool, workers: int) -> None:
    import pyarrow as pa

    docs = all_docs()
    models, ids, jobs = [], [], []
    for m in specs.LID_MODELS:
        for i, t in docs:
            models.append(m)
            ids.append(i)
            jobs.append((m, t))
    res = pmap(_ft_chunk, jobs, workers, chunk=500)
    table = pa.table({"model": models, "id": ids, "labels": [r[0] for r in res],
                      "probs": pa.array([r[1] for r in res], pa.list_(pa.float64()))})
    write(table, OUT["fasttext"], pin, oracle=versions("fasttext-predict"), k=specs.LID_TOPK,
          call='model.predict(text.replace("\\n", " "), k=5, threshold=0.0)',
          models={m: D.load_manifest()["files"][rel]["sha256"] for m, rel in specs.LID_MODELS.items()})


# ------------------------------------------------------------------------------------------------- filters

_FILTERS: dict[str, Any] = {}


def datatrove_filter(case: str) -> Any:
    if case not in _FILTERS:
        from datatrove.pipeline import filters as F

        _, cls, cfg = specs.FILTER_CASES[case]
        _FILTERS[case] = getattr(F, cls)(**cfg)
    return _FILTERS[case]


def datatrove_decide(case: str, doc_id: str, text: str) -> tuple[Any, ...]:
    """Reference decision for one doc: ("ok", keep, reason, text_out|None) or ("raise", ExceptionName)."""
    from datatrove.data import Document
    from datatrove.pipeline.filters.base_filter import get_filter_result

    doc = Document(text=text, id=doc_id)
    try:
        keep, reason = get_filter_result(datatrove_filter(case).filter(doc))
    except Exception as e:  # noqa: BLE001  exception type is part of parity
        return ("raise", type(e).__name__)
    return ("ok", bool(keep), reason, doc.text if doc.text != text else None)


def _filters_chunk(job: list[tuple[str, str, str]]) -> list[tuple[Any, ...]]:
    return [datatrove_decide(case, i, t) for case, i, t in job]


def ref_filters(pin: bool, workers: int) -> None:
    import pyarrow as pa

    docs = all_docs()
    cases, ids, jobs = [], [], []
    for case in specs.FILTER_CASES:
        for i, t in docs:
            cases.append(case)
            ids.append(i)
            jobs.append((case, i, t))
    res = pmap(_filters_chunk, jobs, workers, chunk=100)
    table = pa.table({
        "case": cases, "id": ids,
        "outcome": [r[0] for r in res],
        "keep": [r[1] if r[0] == "ok" else None for r in res],
        "reason": [r[2] if r[0] == "ok" else r[1] for r in res],
        "text_out": [r[3] if r[0] == "ok" else None for r in res],
    })
    write(table, OUT["filters"], pin, oracle=versions("datatrove", "spacy"),
          cases={k: {"class": v[1], "kwargs": v[2]} for k, v in specs.FILTER_CASES.items()})


# --------------------------------------------------------------------------------------------------- MinHash


def minhash_config() -> Any:
    from datatrove.pipeline.dedup.minhash import MinhashConfig
    from datatrove.utils.hashing import HashConfig

    c = specs.MINHASH_CONFIG
    return MinhashConfig(n_grams=c["n_grams"], num_buckets=c["num_buckets"], hashes_per_bucket=c["hashes_per_bucket"],
                         seed=c["seed"], hash_config=HashConfig(hash_fc=c["hash_fc"], precision=c["precision"]))


_SIG: list[Any] = []


def _sig_chunk(texts: list[str]) -> list[list[int] | None]:
    if not _SIG:
        from datatrove.pipeline.dedup import MinhashDedupSignature

        _SIG.append(MinhashDedupSignature(output_folder=tempfile.mkdtemp(prefix="pyroid-sig-"), config=minhash_config()))
    step = _SIG[0]
    out: list[list[int] | None] = []
    for t in texts:
        sh = step.get_shingles(t)
        out.append(None if sh.size == 0 else [int(x) for b in step.get_signature(sh) for x in b])
    return out


def datatrove_minhash_removed(texts: list[str], ids: list[str], workdir: Path) -> list[bool]:
    """Run datatrove's four MinHash stages exactly as examples/fineweb.py chains them; return the removed mask."""
    import struct

    from datatrove.data import Document
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.dedup import MinhashDedupCluster, MinhashDedupSignature
    from datatrove.pipeline.dedup.minhash import MinhashDedupBuckets

    cfg = minhash_config()
    docs = [Document(text=t, id=i) for i, t in zip(ids, texts, strict=True)]
    w = str(workdir)
    stages = [
        ([docs, MinhashDedupSignature(output_folder=f"{w}/sigs", config=cfg)], 1),
        ([MinhashDedupBuckets(input_folder=f"{w}/sigs", output_folder=f"{w}/buckets", config=cfg)], cfg.num_buckets),
        ([MinhashDedupCluster(input_folder=f"{w}/buckets", output_folder=f"{w}/remove_ids", config=cfg)], 1),
    ]
    for n, (pipeline, tasks) in enumerate(stages):
        LocalPipelineExecutor(pipeline=pipeline, tasks=tasks, workers=1, logging_dir=f"{w}/logs/{n}").run()
    removed = [False] * len(docs)
    rf = Path(w) / "remove_ids" / "000000.remove"
    if rf.exists():
        raw = rf.read_bytes()
        for (idx,) in struct.iter_unpack("<I", raw):
            removed[idx] = True
    return removed


def ref_dedup(pin: bool, workers: int) -> None:
    import pyarrow as pa

    docs = all_docs()
    sigs = pmap(_sig_chunk, docs.texts, workers, chunk=250)
    with tempfile.TemporaryDirectory(prefix="pyroid-minhash-") as tmp:
        removed = datatrove_minhash_removed(docs.texts, docs.ids, Path(tmp))
    table = pa.table({"id": docs.ids, "signature": pa.array(sigs, pa.list_(pa.uint64())), "removed": removed})
    write(table, OUT["dedup"], pin, oracle=versions("datatrove", "spacy", "numpy"), config=specs.MINHASH_CONFIG,
          removed=sum(removed), corpora=["dev", "adversarial"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["adversarial", "ftfy", "fasttext", "filters", "dedup", "all"])
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    a = ap.parse_args()
    steps = list(OUT) if a.what == "all" else [a.what]
    for s in steps:
        t = time.perf_counter()
        log(f"{s} ...")
        if s == "adversarial":
            build_adversarial(a.pin)
            D.corpus.cache_clear()
        else:
            globals()[f"ref_{s}"](a.pin, a.workers)
        log(f"{s} done in {time.perf_counter() - t:.0f} s")


if __name__ == "__main__":
    main()
