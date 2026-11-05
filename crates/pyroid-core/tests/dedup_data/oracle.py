"""Record what datatrove 0.10.1's MinHash pieces do, for pyroid's dedup differential tests.

Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc.; modified.

Run inside the pyroid-dev container with the parity oracle env (datatrove 0.10.1, spaCy, numpy,
regex), never on the host:

    python oracle.py tokens OUT.parquet CORPUS.parquet...
        simplify_text output and spaCy word tokens per doc
    python oracle.py params OUT.json
        numpy RandomState permutation parameters
    python oracle.py stages OUT.json HASH_FC PRECISION CORPUS.parquet...
        stages 1-3 over the corpora: stage-2 pairs, removed ids, cluster ids and sizes
    python oracle.py simplify OUT.jsonl N SEED
        simplify_text on N random strings, over all configs

``tokens`` is how the stages are tested before the Rust spaCy tokenizer lands: the tokens are
exactly ``MinhashDedupSignature.word_tokenizer.word_tokenize(simplify_text(text))``, so feeding
them to pyroid reproduces datatrove's shingles.
"""

# The inputs are full of look-alike characters on purpose.
# ruff: noqa: RUF001

from __future__ import annotations

import itertools
import json
import random
import sys
import unicodedata
from dataclasses import fields
from pathlib import Path


def tokens(out: Path, corpora: list[str]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datatrove.pipeline.dedup.minhash import MinhashDedupSignature
    from datatrove.utils.text import simplify_text

    step = MinhashDedupSignature(output_folder="/tmp/pyroid-oracle-sig")
    tok = step.word_tokenizer
    ids: list[str] = []
    simplified: list[str] = []
    toks: list[list[str]] = []
    split_same = 0
    for c in corpora:
        t = pq.read_table(c, columns=["id", "text"]).to_pydict()
        for i, text in zip(t["id"], t["text"], strict=True):
            s = simplify_text(text, step.config.norm_config)
            w = tok.word_tokenize(s)
            ids.append(i)
            simplified.append(s)
            toks.append(w)
            split_same += w == s.split()
    tok_col = pa.array(toks, pa.list_(pa.string()))
    pq.write_table(pa.table({"id": ids, "simplified": simplified, "tokens": tok_col}), out)
    sys.stderr.write(f"{len(ids)} docs; spaCy tokens == str.split() for {split_same}\n")


def stages(out: Path, corpora: list[str], hash_fc: str, precision: int) -> None:
    """datatrove's stages 1-3 over the concatenated corpora (one task): the stage-2 pairs of every
    bucket in file order, the removed ids, and the cluster ids and sizes of stage 3."""
    import struct
    import tempfile

    import pyarrow.parquet as pq
    from datatrove.data import Document
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.dedup import MinhashDedupCluster, MinhashDedupSignature
    from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets
    from datatrove.utils.hashing import HashConfig

    texts: list[str] = []
    for c in corpora:
        texts += pq.read_table(c, columns=["text"]).column("text").to_pylist()
    cfg = MinhashConfig(hash_config=HashConfig(hash_fc=hash_fc, precision=precision))
    docs = [Document(text=t, id=str(i)) for i, t in enumerate(texts)]
    with tempfile.TemporaryDirectory(prefix="pyroid-oracle-") as w:
        steps = [
            ([docs, MinhashDedupSignature(output_folder=f"{w}/sigs", config=cfg)], 1),
            (
                [MinhashDedupBuckets(input_folder=f"{w}/sigs", output_folder=f"{w}/buckets", config=cfg)],  # noqa: E501
                cfg.num_buckets,
            ),
            (
                [
                    MinhashDedupCluster(
                        input_folder=f"{w}/buckets",
                        output_folder=f"{w}/remove_ids",
                        config=cfg,
                        save_cluster_id=True,
                        save_cluster_size=True,
                    )
                ],
                1,
            ),
        ]
        for n, (pipeline, tasks) in enumerate(steps):
            LocalPipelineExecutor(
                pipeline=pipeline, tasks=tasks, workers=1, logging_dir=f"{w}/logs/{n}"
            ).run()
        pairs = []
        for b in range(cfg.num_buckets):
            raw = (Path(w) / "buckets" / f"{b:05d}_00.dups").read_bytes()
            pairs.append([[d1, d2] for (_, d1, _, d2) in struct.iter_unpack("<4I", raw)])

        def read(name: str, fmt: str) -> list:
            p = Path(w) / "remove_ids" / name
            return list(struct.iter_unpack(fmt, p.read_bytes())) if p.exists() else []

        result = {
            "config": {"hash_fc": hash_fc, "precision": precision},
            "n": len(texts),
            "pairs": pairs,
            "removed": [x for (x,) in read("000000.remove", "<I")],
            "cluster_ids": read("000000.clusters", "<2I"),
            "cluster_sizes": read("000000.sizes", "<2I"),
        }
    out.write_text(json.dumps(result))
    n_pairs = sum(map(len, pairs))
    sys.stderr.write(f"{len(texts)} docs, {n_pairs} pairs, {len(result['removed'])} removed\n")


def params(out: Path) -> None:
    import numpy as np

    p = np.uint64((1 << 61) - 1)
    cases = []
    for seed, n in [(0, 1), (1, 112), (1, 9), (42, 256), (2**32 - 1, 20), (12345, 3)]:
        gen = np.random.RandomState(seed)
        a = gen.randint(1, p, dtype=np.uint64, size=(1, n))
        b = gen.randint(0, p, dtype=np.uint64, size=(1, n))
        cases.append({"seed": seed, "n": n, "a": a[0].tolist(), "b": b[0].tolist()})
    out.write_text(json.dumps(cases))


# Characters simplify_text treats specially, plus a sample of everything else.
INTERESTING = (
    "aAzZ09 \t\n\r\x0b\x0c\x1c\x1f\x85\xa0 　.,،٫⎖⎗⎘!?;:'\"’‘“”«»-–—_/\\%$€£¥١٢३৪๕"
    "１２０٠۰߀०০੦૦୦௦౦೦൦෦๐໐༠၀႐០᠐᥆᧐᪀᪐᭐᮰᱀᱐꘠꣐꤀꧐꧰꩐꯰０𐒠𑁦𑃰𑄶𑇐𑋰𑑐𑓐𑙐𑛀𑜰𑣠𝟎𝟘𝟢𝟬𝟶𞥐"
    "ΣσςΑΒΓ İıIiÀÉÎÕÜàéîõüçÇñÑåÅøØæÆßẞǅǈǋǲ़ְً̧̀́̂̈ͅ"
    ";·`´ΩKÅ̀́̓̈́ẛཱཱཱིྀུ"
    "\U0001d15e\U0001d165\U0001d16d〮〯יִײַ가힣각"
    "mondaytuesdayjanuarymayjune"
)


def rand_text(rng: random.Random) -> str:
    n = rng.choice([0, 1, 2, 5, 10, 30, 80])
    out = []
    for _ in range(n):
        r = rng.random()
        if r < 0.55:
            out.append(rng.choice(INTERESTING))
        elif r < 0.7:
            out.append(rng.choice(["monday", "May", "DECEMBER", "Sunday", "1.5", "12,000", "3،4", "ΣΑΣ "]))
        elif r < 0.9:
            cp = rng.randrange(0x20, 0x3000)
            out.append(chr(cp))
        else:
            cp = rng.randrange(0, 0x110000)
            if not 0xD800 <= cp <= 0xDFFF:
                out.append(chr(cp))
    return "".join(out)


def simplify(out: Path, n: int, seed: int) -> None:
    from datatrove.utils.text import TextNormConfig, simplify_text

    names = [f.name for f in fields(TextNormConfig)]
    configs = [dict(zip(names, bits, strict=True)) for bits in itertools.product([True, False], repeat=len(names))]
    rng = random.Random(seed)
    with out.open("w") as f:
        for i in range(n):
            text = rand_text(rng)
            cfgs = configs if i % 16 == 0 else [rng.choice(configs), {}]
            for cfg in cfgs:
                got = simplify_text(text, TextNormConfig(**cfg))
                f.write(json.dumps({"text": text, "config": cfg, "out": got}) + "\n")
    print(f"unicode {unicodedata.unidata_version}", file=sys.stderr)


def main() -> None:
    cmd, *rest = sys.argv[1:]
    if cmd == "tokens":
        tokens(Path(rest[0]), rest[1:])
    elif cmd == "params":
        params(Path(rest[0]))
    elif cmd == "stages":
        out, hash_fc, precision, *corpora = rest
        stages(Path(out), corpora, hash_fc, int(precision))
    elif cmd == "simplify":
        simplify(Path(rest[0]), int(rest[1]), int(rest[2]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
