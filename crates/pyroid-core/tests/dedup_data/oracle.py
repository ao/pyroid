"""Record what datatrove 0.10.1's MinHash pieces do, for pyroid's dedup differential tests.

Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc.; modified.

Run with the parity oracle env (datatrove 0.10.1, spaCy, numpy, regex), on Linux or in a Linux
container:

    python oracle.py tokens OUT.parquet CORPUS.parquet...
        simplify_text output and spaCy word tokens per doc
    python oracle.py params OUT.json
        numpy RandomState permutation parameters
    python oracle.py stages OUT.json HASH_FC PRECISION CORPUS.parquet...
        stages 1-3 over the corpora: stage-2 pairs, removed ids, cluster ids and sizes
    python oracle.py simplify OUT.jsonl N SEED
        simplify_text on N random strings, over all configs
    python oracle.py digests OUT.json [--vendored]
        CRC-32 per block of 4096 code points of simplify_text on single-code-point inputs

``simplify`` and ``digests`` also run under the other CPythons (3.11-3.15, only ``regex``
installed) with a vendored copy of ``simplify_text``; with datatrove installed the copy is
checked against it on every call.

``tokens`` is how the stages are tested before the Rust spaCy tokenizer lands: the tokens are
exactly ``MinhashDedupSignature.word_tokenizer.word_tokenize(simplify_text(text))``, so feeding
them to pyroid reproduces datatrove's shingles.
"""

# The inputs are full of look-alike characters on purpose.
# ruff: noqa: RUF001

from __future__ import annotations

import json
import random
import re
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
                [
                    MinhashDedupBuckets(
                        input_folder=f"{w}/sigs", output_folder=f"{w}/buckets", config=cfg
                    )
                ],
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
    out.write_text(json.dumps(cases) + "\n")


# Characters simplify_text treats specially, plus a sample of everything else.
INTERESTING = (
    "aAzZ09 \t\n\r\x0b\x0c\x1c\x1f\x85\xa0 　.,،٫⎖⎗⎘!?;:'\"’‘“”«»-–—_/\\%$€£¥١٢३৪๕"
    "１２０٠۰߀०০੦૦୦௦౦೦൦෦๐໐༠၀႐០᠐᥆᧐᪀᪐᭐᮰᱀᱐꘠꣐꤀꧐꧰꩐꯰０𐒠𑁦𑃰𑄶𑇐𑋰𑑐𑓐𑙐𑛀𑜰𑣠𝟎𝟘𝟢𝟬𝟶𞥐"
    "ΣσςΑΒΓ İıIiÀÉÎÕÜàéîõüçÇñÑåÅøØæÆßẞǅǈǋǲ़ְً̧̀́̂̈ͅ"
    ";·`´ΩKÅ̀́̓̈́ẛཱཱཱིྀུ"
    "\U0001d15e\U0001d165\U0001d16d〮〯יִײַ가힣각"
    "mondaytuesdayjanuarymayjune"
)


WORDS = [
    "monday", "May", "DECEMBER", "Sunday", "wednesdays", "marchmay", "FRİDAY", "1.5", "12,000",
    "3،4", "1.2.3", "5.", ".5", "1..2", "ΣΑΣ ", "ΟΔΟΣ.", "İSTANBUL", "ΑΣ'Α", "ǅUNGLA",
]  # fmt: skip


def _chars(cps: str) -> str:
    return "".join(chr(int(c, 16)) for c in cps.split())


# Final_Sigma contexts: cased (capital sigma, ASCII and Greek letters, a titlecase, Roman
# numeral one, circled a, math bold A, Deseret), case-ignorable and cased (modifier letters,
# ypogegrammeni), case-ignorable (' . : ^ ` soft hyphen, middle dot, acute, right quote, ZWSP,
# ZWJ, VS1, tags), neither (space 1 -).
SIGMA_POOL = _chars(
    "3a3 3a3 3a3 3a3 41 61 391 3b1 1c5 2160 24d0 1d400 10400 1d43 2b0 345"
    " 27 2e 3a 5e 60 ad b7 301 2019 200b 200d fe00 e0001 e0100 20 31 2d"
)
# Combining marks of many classes; the non-Mn ones whose class shows (they are reordered and
# survive: 1D165 1D166 1D16D 1D16E 302E 302F 1734 1715 16FF0 16FF1 1134D); CGJ (Mn, class 0);
# U+1171E (Mn until 15.1, Mc from 16.0); U+0CF3 (new in 15.0); starters.
MARKS = _chars(
    "334 5b0 f71 f72 f74 1dce 302a 316 302e 302f 301 345 34f 315 5bf e48 321 327 31b e38 f39"
    " 1d165 1d166 1d16d 1d16e 1d167 1734 1715 16ff0 16ff1 1134d 1171e cf3 61 3b5 1100"
)
# NFD maps these to (or through) punctuation, whitespace, ASCII or a mark sequence.
NFD_SPECIAL = _chars(
    "37e 387 1fef 1ffd 1fee 1fed 1fc1 2000 2001 212a 212b 2126 1ffa 340 341 343 344 374 1fbe"
    " 2329 232a f73 f75 f81 fb1d fb1f f900 2f800 958 2adc 1e9b 1c4 130"
)
# Code points whose properties differ between Unicode 14.0 and 17.0 (generate_tables.py dumps).
VERSIONED = (
    "295 897 ece 1acf-1aeb 1c89-1c8a a7cb-a7dc a7f1 105c9 105e4 10d4e-10d85 10ec5 10efa-10eff"
    " 11241 11383-11391 113bb-113e2 1171e 11b60-11b66 11dd9 11f00-11f5a 13439-13455 1611e-1612f"
    " 16d40-16d6c 16ea0-16ed3 16ff2-16ff3 1df25-1df2a 1e030-1e08f 1e4eb-1e4ef 1e5ee-1e5ef"
    " 1e6e3-1e6ff"
)
VERSIONED_CPS = [
    cp
    for r in VERSIONED.split()
    for a, _, b in [r.partition("-")]
    for cp in range(int(a, 16), int(b or a, 16) + 1)
]
SEPARATORS = ".,،٫⎖⎗⎘"
_DIGITS: list[str] = []


def _digits() -> list[str]:
    """Every `\\p{Decimal_Number}` character, as the (one, pinned) regex release sees it."""
    if not _DIGITS:
        import regex

        digit_re = regex.compile(r"\p{Decimal_Number}")
        _DIGITS.extend(chr(cp) for cp in range(0x110000) if digit_re.match(chr(cp)))
    return _DIGITS


def _rand_number(rng: random.Random) -> str:
    def run() -> str:
        k = rng.randrange(1, 4)
        return "".join(
            rng.choice("0123456789" if rng.random() < 0.3 else _digits()) for _ in range(k)
        )

    s = run()
    for _ in range(rng.choice([0, 0, 1, 2])):
        s += rng.choice([*SEPARATORS, "..", " ", "a"]) + run()
    return s + (rng.choice(SEPARATORS) if rng.random() < 0.2 else "")


def rand_text(rng: random.Random) -> str:
    n = rng.choice([0, 1, 2, 5, 10, 30, 80])
    out = []
    for _ in range(n):
        r = rng.random()
        if r < 0.38:
            out.append(rng.choice(INTERESTING))
        elif r < 0.46:
            out.append(rng.choice(WORDS))
        elif r < 0.54:
            out.append("".join(rng.choice(SIGMA_POOL) for _ in range(rng.randrange(1, 6))))
        elif r < 0.62:
            out.append("".join(rng.choice(MARKS) for _ in range(rng.randrange(1, 5))))
        elif r < 0.68:
            out.append(_rand_number(rng))
        elif r < 0.72:
            out.append(rng.choice(NFD_SPECIAL))
        elif r < 0.76:
            out.append(chr(rng.choice(VERSIONED_CPS)))
        elif r < 0.78:
            out.append(chr(rng.choice([*range(0x20), *range(0x7F, 0xA0)])))
        elif r < 0.9:
            out.append(chr(rng.randrange(0x20, 0x3000)))
        else:
            cp = rng.randrange(0, 0x110000)
            if not 0xD800 <= cp <= 0xDFFF:
                out.append(chr(cp))
    return "".join(out)


def _vendored_simplify() -> tuple:
    """datatrove 0.10.1's `TextNormConfig` and `simplify_text` (datatrove/utils/text.py), copied
    with only `re`, `regex` and `unicodedata`, for interpreters without datatrove."""
    import re
    from dataclasses import dataclass

    import regex

    punctuation = (
        "!/—”:％１〈&(、━\\【#%「」，】；+^]~“《„';’{|∶´[=-`*．（–？！：$～«〉,><》)?）。…@_.\"}►»"
        + "".join(
            map(chr, (x for a, b in ((0, 9), (11, 13), (13, 32), (127, 160)) for x in range(a, b)))
        )
    )
    punctuation_trans = str.maketrans(punctuation, " " * len(punctuation))
    # datatrove's NUMBERS_PATTERN, with the digit class spelt out (the same set in `regex`).
    numbers = regex.compile(
        r"\p{Decimal_Number}+([.,،٫⎖⎗⎘]{1}\p{Decimal_Number}+)?", regex.VERBOSE | regex.UNICODE
    )
    whitespace = re.compile(r"\s+")
    weekdays = re.compile(r"monday|tuesday|wednesday|thursday|friday|saturday|sunday")
    months = re.compile(
        r"january|february|march|april|may|june|july|august|september|october|november|december"
    )

    @dataclass
    class TextNormConfig:
        lowercase: bool = True
        norm_whitespace: bool = True
        remove_punctuation: bool = True
        norm_unicode_diacritics: bool = True
        norm_numbers: bool = True
        norm_weekdays: bool = False
        norm_monthnames: bool = False

    def simplify_text(text: str, config: TextNormConfig) -> str:
        if config.lowercase:
            text = text.lower()
        if config.norm_numbers:
            text = numbers.sub("0", text)
        if config.norm_weekdays:
            text = weekdays.sub("WEEKDAY", text)
        if config.norm_monthnames:
            text = months.sub("MONTH", text)
        if config.remove_punctuation:
            text = text.translate(punctuation_trans)
        if config.norm_whitespace:
            text = whitespace.sub(" ", text.strip())
        if config.norm_unicode_diacritics:
            text = "".join(
                c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
            )
        return text.strip()

    return TextNormConfig, simplify_text, punctuation


def _simplifiers(vendored: bool) -> tuple:
    """(config for bit mask, simplify_text, uses datatrove). Config bit i is field i of
    `TextNormConfig`: lowercase, norm_whitespace, remove_punctuation, norm_unicode_diacritics,
    norm_numbers, norm_weekdays, norm_monthnames. With datatrove installed, the vendored copy
    is checked against it on every call unless `vendored` asks for the copy alone."""
    v_cfg, v_simplify, v_punct = _vendored_simplify()
    names = [f.name for f in fields(v_cfg)]
    try:
        from datatrove.utils import text as dt
    except ImportError:
        dt = None
    if dt is None or vendored:

        def config(bits: int) -> object:
            return v_cfg(**{n: bool(bits >> i & 1) for i, n in enumerate(names)})

        return config, v_simplify, False
    assert [f.name for f in fields(dt.TextNormConfig)] == names
    assert v_punct == dt.PUNCTUATION
    assert dt.TextNormConfig() == dt.TextNormConfig(**vars(v_cfg()))

    def checked(text: str, bits: int) -> str:
        kw = {n: bool(bits >> i & 1) for i, n in enumerate(names)}
        got = dt.simplify_text(text, dt.TextNormConfig(**kw))
        assert got == v_simplify(text, v_cfg(**kw)), (text, bits)
        return got

    return (lambda bits: bits), checked, True


def _header(uses_datatrove: bool, **extra: object) -> dict:
    return {
        "unicode": unicodedata.unidata_version,
        "python": sys.version.split()[0],
        "datatrove": uses_datatrove,
        **extra,
    }


DEFAULT_BITS = 0b0011111
# The typos linter reads one random two-letter word as a misspelling: leave such texts out.
TYPOS_FLAGGED = re.compile(r"(?<![A-Za-z])[Ii][Yy](?![A-Za-z])")


def simplify(out: Path, n: int, seed: int) -> None:
    """A header line, then one `{"text", "out": {config bit mask: output}}` line per text: every
    32nd text under all 128 configs, the others under the default, everything on, one flag on its
    own (in turn) and a random config. Texts the typos linter would flag (TYPOS_FLAGGED) are
    skipped, after drawing their configs, so the other lines do not change."""
    config, simplify_text, uses_dt = _simplifiers(vendored=False)
    rng = random.Random(seed)
    cases = 0
    with out.open("w") as f:
        f.write(json.dumps(_header(uses_dt, seed=seed, n=n)) + "\n")
        for i in range(n):
            text = rand_text(rng)
            if i % 32 == 0:
                cfgs = list(range(128))
            else:
                cfgs = sorted({DEFAULT_BITS, 127, 1 << (i % 7), rng.randrange(128)})
            outs = {str(bits): simplify_text(text, config(bits)) for bits in cfgs}
            if any(TYPOS_FLAGGED.search(s) for s in (text, *outs.values())):
                continue
            f.write(json.dumps({"text": text, "out": outs}, ensure_ascii=False) + "\n")
            cases += len(outs)
    sys.stderr.write(f"{out}: {n} texts, {cases} cases, Unicode {unicodedata.unidata_version}\n")


# (config bits, template): `{}` is replaced by each code point in turn.
DIGEST_CASES = [
    (DEFAULT_BITS, "{}"),
    (DEFAULT_BITS, "A{}Σ"),
    (0b0000001, "{}"),
    (0b0000001, "A{}Σ"),
    (0b0000001, "{}Σ"),
    (0b0000001, "AΣ{}"),
    (0b0001000, "{}"),
    (0b0001000, "\U0001d16d{}\U0001d165"),
    (0b0010000, "{}"),
    (0b0010000, "1{}2"),
    (0b0000110, " {} x"),
    (0b0000010, "x{}x"),
    (0b1111111, "{}"),
]
DIGEST_BLOCK = 4096


def digests(out: Path, vendored: bool) -> None:
    """For each DIGEST_CASES entry and each block of 4096 code points (surrogates skipped), the
    CRC-32 of the outputs, each as a little-endian u32 byte length followed by its UTF-8."""
    import zlib

    config, simplify_text, uses_dt = _simplifiers(vendored)
    cases = []
    for bits, template in DIGEST_CASES:
        cfg = config(bits)
        crcs = []
        for start in range(0, 0x110000, DIGEST_BLOCK):
            crc = 0
            for cp in range(start, start + DIGEST_BLOCK):
                if 0xD800 <= cp <= 0xDFFF:
                    continue
                b = simplify_text(template.replace("{}", chr(cp)), cfg).encode()
                crc = zlib.crc32(b, zlib.crc32(len(b).to_bytes(4, "little"), crc))
            crcs.append(f"{crc:08x}")
        cases.append({"config": bits, "template": template, "crc32": "".join(crcs)})
    head = _header(uses_dt, block=DIGEST_BLOCK)
    out.write_text(json.dumps({**head, "cases": cases}, ensure_ascii=False, indent=0) + "\n")
    sys.stderr.write(f"{out}: {len(cases)} cases, Unicode {unicodedata.unidata_version}\n")


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
    elif cmd == "digests":
        digests(Path(rest[0]), vendored=rest[1:] == ["--vendored"])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
