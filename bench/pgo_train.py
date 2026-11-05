#!/usr/bin/env python3
"""PGO training workload: the harness's candidate calls, ungated, on docs the benchmarks never time.

    python pgo_train.py --corpus headline --offset 10000 --docs 10000

Runs every pyroid module the way the harness rows do (list[str] and Arrow batches, single-str calls) at
threads=1 and threads=8, so an instrumented build (rustc -Cprofile-generate) records the hot paths. No
equivalence gates: an instrumented build's outputs are not timed, and the oracles (datatrove, spaCy,
fasttext-predict) do not install on every interpreter. The default slice, headline[10000:20000], is disjoint
from the dev slice (headline[:10000]) that the A/B and the published numbers use.
"""

from __future__ import annotations

import argparse
import time

import pyarrow as pa
from pyroid_parity import specs

import gates as G


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="headline", choices=sorted(G.SEGMENTS))
    ap.add_argument("--offset", type=int, default=10000)
    ap.add_argument("--docs", type=int, default=10000)
    ap.add_argument("--per-doc", type=int, default=1000, help="docs for the single-str calls")
    a = ap.parse_args()
    import pyroid.dedup as D
    import pyroid.fasttext as FT
    import pyroid.filters as F
    import pyroid.ftfy as FX
    import pyroid.translit as T

    _, texts = G.load_slice(a.corpus, a.offset, a.docs)
    arrow = pa.array(texts, pa.string())
    lid = [FT.load_model(G.lid_model_path(m)) for m in specs.LID_MODELS]
    flat = [t.replace("\n", " ") for t in texts]
    chain = [F.GopherRepetition(), F.GopherQuality(), F.C4Quality(filter_no_terminal_punct=False), F.FineWebQuality()]
    t0 = time.perf_counter()
    for threads in (1, 8):
        for docs in (texts, arrow):
            FX.fix_text_batch(docs, threads=threads)
            T.anyascii(docs, threads=threads)
            T.unidecode(docs, threads=threads)
            D.minhash_signatures(docs, threads=threads, **specs.MINHASH_CONFIG)
            D.exact_dedup(docs, threads=threads)
            F.apply(docs, chain, threads=threads)
            for fn, _, cfg in specs.FILTER_CASES.values():
                getattr(F, fn)(docs, threads=threads, **cfg)
        for m in lid:
            for k in (1, 5):
                m.predict(flat, k=k, threads=threads)
        D.MinHashDedup(threads=threads, **specs.MINHASH_CONFIG).fit(texts)
    for t in texts[: a.per_doc]:
        FX.fix_text(t)
        T.anyascii(t)
        D.minhash_signatures(t, **specs.MINHASH_CONFIG)
        F.gopher_repetition_reasons(t)
        lid[0].predict(t.replace("\n", " "))
    print(f"pgo_train: {a.docs} docs from {a.corpus}[{a.offset}:], {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
