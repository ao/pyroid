#!/usr/bin/env python3
"""PGO training workload: every pyroid module, called the way the benchmarks call it, on docs they never time.

    python bench/pgo_train.py --corpus-dir pgo/corpus                  # inputs from bench/pgo_corpus.py
    python bench/pgo_train.py --corpus-dir pgo/corpus --threads 1,4 --docs 2000

Run it once per interpreter the release wheels serve, with an instrumented build of pyroid installed
(rustc ``-Cprofile-generate``); ``llvm-profdata merge`` then combines the profiles. bench/pgo_build.sh and
.github/workflows/release.yml do both.

Each module runs on list[str] and Arrow input and single str calls, at every ``--threads`` value:
ftfy (FineWeb text, plus mojibake made from it), translit, fastText LID (.bin and .ftz), the Gopher, C4 and
FineWeb filters and the formatters, MinHash and exact and sentence dedup, pyroid.text, pyroid.jieba,
pyroid.json_repair (broken LLM output from its parity corpus generator, with another seed), pyroid.metrics
(ROUGE, BLEU, chrF), pyroid.datasketch, and the FineWeb-2 filter and MinHash configs of every language
pyroid tokenizes natively here (Japanese only with SudachiDict installed).

Needs only pyroid and numpy. pyarrow is optional: without it, as on Windows ARM64, the Arrow calls are
skipped. Nothing is checked or timed: an instrumented build is never shipped.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parity"))  # pyroid_parity: stdlib-only specs
from pyroid_parity import json_repair_corpus as JC
from pyroid_parity import specs

# json_repair_corpus.SEED (645) makes the gated corpus; training uses another.
TRAIN_SEED = 1

try:
    import pyarrow as pa
except ImportError:
    pa = None


def read_jsonl(p: Path, n: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if n is not None and len(rows) >= n:
                break
    return rows


def kinds(texts: list[str]) -> list[Any]:
    """The input kinds the harness times: list[str], and an Arrow string array when pyarrow is there."""
    return [texts] + ([pa.array(texts, pa.string())] if pa is not None else [])


def mojibake(texts: list[str]) -> list[str]:
    """UTF-8 read as Latin-1 or Windows-1252, the mix ftfy is for, from the docs that have non-ASCII text."""
    out = []
    for i, t in enumerate(t for t in texts if not t.isascii()):
        b = t.encode("utf-8")
        out.append(b.decode("latin-1") if i % 2 else b.decode("cp1252", "replace"))
    return out


def broken_json(n: int) -> list[str]:
    """Broken LLM output from the json_repair parity corpus's generator, with a seed the suite never uses."""
    return [c.text for c in JC.cases(n, seed=TRAIN_SEED)]


def shingles(text: str, n: int = 5) -> list[bytes]:
    w = text.split()
    return [" ".join(w[i : i + n]).encode("utf-8") for i in range(max(len(w) - n + 1, 1))]


def native(lang: str) -> bool:
    import pyroid.filters as F

    try:
        F.apply(["Probe."], [F.GopherQuality(language=lang)])
    except NotImplementedError:
        return False
    return True


class Trainer:
    def __init__(self, corpus: Path, docs: int, per_doc: int, threads: list[int]) -> None:
        self.rows = read_jsonl(corpus / "fineweb.jsonl", docs)
        self.texts = [r["text"] for r in self.rows]
        self.per_doc = per_doc
        self.threads = threads
        self.corpus = corpus
        self.langs = {
            p.stem.removeprefix("fineweb2-"): [r["text"] for r in read_jsonl(p)]
            for p in sorted(corpus.glob("fineweb2-*.jsonl"))
        }
        self.timings: dict[str, float] = {}

    def section(self, name: str, fn: Callable[[], None]) -> None:
        t0 = time.perf_counter()
        fn()
        self.timings[name] = time.perf_counter() - t0
        print(f"  {name}: {self.timings[name]:.1f} s", flush=True)

    def ftfy(self) -> None:
        import pyroid.ftfy as FX

        bad = mojibake(self.texts)
        for t in self.threads:
            for docs in kinds(self.texts) + kinds(bad):
                FX.fix_text_batch(docs, threads=t)
        for s in self.texts[: self.per_doc] + bad[: self.per_doc]:
            FX.fix_text(s)
        for s in bad[:200]:
            FX.fix_encoding(s)
            FX.fix_and_explain(s)

    def translit(self) -> None:
        import pyroid.translit as T

        other = [t for texts in self.langs.values() for t in texts]
        for t in self.threads:
            for docs in kinds(self.texts) + kinds(other):
                T.anyascii(docs, threads=t)
                T.unidecode(docs, threads=t)
        for s in self.texts[: self.per_doc] + other[: self.per_doc]:
            T.anyascii(s)
            T.unidecode(s)

    def fasttext(self) -> None:
        import pyroid.fasttext as FT

        flat = [t.replace("\n", " ") for t in self.texts]
        other = [t.replace("\n", " ") for texts in self.langs.values() for t in texts]
        for name in specs.LID_MODELS:
            m = FT.load_model(str(self.corpus / name))
            for t in self.threads:
                for k in (1, 5):
                    m.predict(flat, k=k, threads=t)
                    m.predict(other, k=k, threads=t)
            for s in flat[: self.per_doc // 2] + other[: self.per_doc // 2]:
                m.predict(s)

    def filters(self) -> None:
        import pyroid.filters as F

        chain = [
            F.GopherRepetition(),
            F.GopherQuality(),
            F.C4Quality(filter_no_terminal_punct=False),
            F.FineWebQuality(),
        ]
        some = self.texts[: max(len(self.texts) // 5, 1)]
        for t in self.threads:
            for docs in kinds(self.texts):
                F.apply(docs, chain, threads=t)
            for docs in kinds(some):
                for fn, _, cfg in specs.FILTER_CASES.values():
                    getattr(F, fn)(docs, threads=t, **cfg)
                    getattr(F, f"{fn}_reasons")(docs, threads=t, **cfg)
                F.c4_lines(docs, threads=t)
                F.pii_replace(docs, threads=t)
                F.symbol_lines(docs, threads=t)
        for s in self.texts[: self.per_doc]:
            F.gopher_repetition_reasons(s)
            F.gopher_quality_reasons(s)
            F.c4_quality_reasons(s)
            F.fineweb_quality_reasons(s)

    def dedup(self) -> None:
        import pyroid.dedup as D

        some = self.texts[: max(len(self.texts) // 5, 1)]
        for t in self.threads:
            for docs in kinds(self.texts):
                D.minhash_signatures(docs, threads=t, **specs.MINHASH_CONFIG)
                D.exact_dedup(docs, threads=t)
                D.simplify_text(docs, threads=t)
            D.MinHashDedup(threads=t, **specs.MINHASH_CONFIG).fit(self.texts)
            D.sentence_dedup(some, threads=t)
        for s in self.texts[: self.per_doc]:
            D.minhash_signatures(s, **specs.MINHASH_CONFIG)

    def text(self) -> None:
        import pyroid.text as TX

        for t in self.threads:
            for docs in kinds(self.texts):
                TX.normalize_whitespace(docs, threads=t)
                TX.min_chars(docs, 500, threads=t)
        for s in self.texts[: self.per_doc]:
            TX.normalize_whitespace(s)
            TX.min_chars(s, 500)

    def multilingual(self) -> None:
        import pyroid.dedup as D
        import pyroid.filters as F

        build = {
            "gopher_repetition": F.GopherRepetition,
            "fineweb_quality": F.FineWebQuality,
            "gopher_quality": F.GopherQuality,
        }
        for lang, texts in self.langs.items():
            if not native(lang):
                print(f"    {lang}: not native here, skipped", flush=True)
                continue
            cases = specs.fineweb2_filter_cases(lang)
            steps = [build[fn](**cfg) for fn, _, cfg in cases.values() if fn in build]
            mh = {**specs.MINHASH_CONFIG_FINEWEB2, "language": lang}
            for t in self.threads:
                F.apply(texts, steps, threads=t)
                for fn, _, cfg in cases.values():
                    # datatrove's own errors (C4 on some scripts) are raised as datatrove raises them.
                    with contextlib.suppress(ValueError):
                        getattr(F, f"{fn}_reasons")(texts, threads=t, **cfg)
                D.minhash_signatures(texts, threads=t, **mh)
                D.minhash_signatures(texts, threads=t, **{**specs.MINHASH_CONFIG, "language": lang})

    def jieba(self) -> None:
        import pyroid.jieba as J

        zh = self.langs.get("cmn_Hani", [])
        for t in self.threads:
            for docs in kinds(zh):
                J.lcut(docs, threads=t)
                J.lcut_for_search(docs, threads=t)
                J.lcut(docs, HMM=False, threads=t)
        for s in zh[: self.per_doc // 4]:
            J.lcut(s)
            J.lcut(s, cut_all=True)
            J.lcut_for_search(s)
            list(J.tokenize(s))
            list(J.tokenize(s, mode="search"))

    def json_repair(self) -> None:
        import pyroid.json_repair as JR

        docs = broken_json(max(len(self.rows) // 2, 1))
        for t in self.threads:
            JR.repair_json_batch(docs, threads=t)
            JR.repair_json_batch(docs, return_objects=True, threads=t)
        for s in docs[: self.per_doc]:
            JR.repair_json(s)
            JR.loads(s)
            JR.repair_json(s, ensure_ascii=False, skip_json_loads=True)

    def metrics(self) -> None:
        from pyroid.metrics import rouge_scorer, sacrebleu

        n = max(len(self.texts) // 5, 1)
        hyp = [t[:300] for t in self.texts[:n]]
        ref = [t[150:450] or t for t in self.texts[:n]]
        zh = self.langs.get("cmn_Hani", [])
        for stem in (False, True):
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=stem)
            for t in self.threads:
                scorer.score_batch(ref, hyp, threads=t)
            for a, b in zip(ref[:200], hyp[:200], strict=True):
                scorer.score(a, b)
        for t in self.threads:
            sacrebleu.sentence_bleu_batch(hyp, [ref], threads=t)
            sacrebleu.sentence_chrf_batch(hyp, [ref], threads=t)
            sacrebleu.BLEU().corpus_score(hyp, [ref], threads=t)
            sacrebleu.BLEU(tokenize="intl").corpus_score(hyp, [ref], threads=t)
            sacrebleu.CHRF().corpus_score(hyp, [ref], threads=t)
            sacrebleu.CHRF(word_order=2).corpus_score(hyp, [ref], threads=t)
            if zh:
                sacrebleu.BLEU(tokenize="zh").corpus_score(
                    [s[:200] for s in zh], [[s[100:300] for s in zh]], threads=t
                )
        for a, b in zip(hyp[:200], ref[:200], strict=True):
            sacrebleu.sentence_bleu(a, [b])
            sacrebleu.sentence_chrf(a, [b])

    def datasketch(self) -> None:
        import pyroid.datasketch as DS

        n = max(len(self.texts) // 5, 1)
        sh = [shingles(t) for t in self.texts[:n]]
        for t in self.threads:
            for scheme in (None, "affine32"):
                kw = {"scheme": scheme} if scheme else {}
                ms = DS.MinHash.bulk(sh, num_perm=128, threads=t, **kw)
                sig = DS.minhash_batch(self.texts[:n], num_perm=128, ngram=5, threads=t, **kw)
            lsh = DS.MinHashLSH(threshold=0.9, num_perm=128)
            lsh.insert_batch(range(n), sig, threads=t)
            lsh.query_batch(sig, threads=t)
        lsh = DS.MinHashLSH(threshold=0.5, num_perm=128)
        for i, m in enumerate(ms[:500]):
            lean = DS.LeanMinHash(m)
            lsh.insert(i, lean)
            lsh.query(lean)
            m.jaccard(ms[0])

    def run(self) -> dict[str, float]:
        warnings.simplefilter("ignore")  # FallbackWarning and friends: the paths still run
        sections = ["ftfy", "translit", "fasttext", "filters", "dedup", "text", "multilingual", "jieba"]
        sections += ["json_repair", "metrics", "datasketch"]
        for name in sections:
            self.section(name, getattr(self, name))
        return self.timings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", type=Path, required=True, help="the folder bench/pgo_corpus.py wrote")
    ap.add_argument("--docs", type=int, default=10_000, help="FineWeb docs for the batch calls")
    ap.add_argument("--per-doc", type=int, default=1_000, help="docs for the single-str calls")
    ap.add_argument("--threads", default="1,8", help="comma-separated thread counts")
    a = ap.parse_args()
    import pyroid

    threads = [int(x) for x in a.threads.split(",")]
    print(
        f"pgo_train: pyroid {pyroid.__version__}, Python {sys.version.split()[0]}"
        f"{' free-threaded' if not getattr(sys, '_is_gil_enabled', lambda: True)() else ''},"
        f" threads {threads}, pyarrow {'yes' if pa is not None else 'no (Arrow calls skipped)'}",
        flush=True,
    )
    t0 = time.perf_counter()
    Trainer(a.corpus_dir, a.docs, a.per_doc, threads).run()
    print(f"pgo_train: done in {time.perf_counter() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
