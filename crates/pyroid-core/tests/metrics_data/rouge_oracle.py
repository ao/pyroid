# Derived from rouge-score 0.1.2 (Apache-2.0), © The rouge_score Authors, and NLTK 3.10.4
# (Apache-2.0), © NLTK Project; modified.
"""Record what rouge-score 0.1.2 and nltk 3.10.4 really return, for metrics_rouge.rs.

Run in the dev container (rouge-score==0.1.2, nltk==3.10.4, pyarrow installed):

    python crates/pyroid-core/tests/metrics_data/rouge_oracle.py \\
        /data/fineweb/fineweb-dev-10k.parquet

writes ``rouge_golden.jsonl`` next to this file. One JSON object per line:

- ``{"stem": word, "out": stem}``: ``PorterStemmer().stem(word)``;
- ``{"text": t, "stemmer": b, "tokens": [...]}``: ``DefaultTokenizer(b).tokenize(t)``;
- ``{"target": t, "prediction": p, "stemmer": b, "types": [...], "scores": [...]}``:
  ``RougeScorer(types, use_stemmer=b).score(t, p)``, one entry per type, either
  ``"int0"`` (``Score(0, 0, 0)`` with ints) or the three floats' IEEE bits as hex.
"""

from __future__ import annotations

import json
import random
import struct
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from nltk.stem.porter import PorterStemmer
from rouge_score import rouge_scorer, tokenize, tokenizers

TYPES = [f"rouge{n}" for n in range(1, 10)] + ["rougeL", "rougeLsum"]
SUFFIXES = [
    "sses", "ies", "ss", "s", "eed", "ed", "ing", "at", "bl", "iz", "y", "ational", "tional",
    "enci", "anci", "izer", "bli", "alli", "entli", "eli", "ousli", "ization", "ation", "ator",
    "alism", "iveness", "fulness", "ousness", "aliti", "iviti", "biliti", "fulli", "logi",
    "icate", "ative", "alize", "iciti", "ical", "ful", "ness", "al", "ance", "ence", "er", "ic",
    "able", "ible", "ant", "ement", "ment", "ent", "ion", "sion", "tion", "ou", "ism", "ate",
    "iti", "ous", "ive", "ize", "e", "ll", "ied",
]  # fmt: skip
EDGE_TEXTS = [
    "",
    " \n\t ",
    "The quick brown fox jumps over the lazy dog",
    "\u0130STANBUL \u0130x a\u0130b",
    "\u212aelvin 5\u212a \u212a",
    "Straße café ÀB \uff21\uff22 x\u0301y 日本語",
    "line one\nline two\n\nline   three\r\nfour\rfive",
    "y yy yyy yyyy yyyyy happy happily skies dying 1990s",
]


def bits(x: float) -> str:
    return f"{struct.unpack('<Q', struct.pack('<d', x))[0]:016x}"


def encode(score: Any) -> Any:
    if all(type(v) is int and v == 0 for v in score):
        return "int0"
    assert all(type(v) is float for v in score), score
    return [bits(v) for v in score]


def main(parquet: str) -> None:
    rng = random.Random(20261106)
    docs = [d for d in pq.read_table(parquet, columns=["text"]).column(0).to_pylist() if d]
    rows: list[dict[str, Any]] = []

    # Porter: a spread of real tokens longer than 3, plus suffix-heavy shapes.
    distinct = sorted({t for d in docs[:3000] for t in tokenize.tokenize(d, None) if len(t) > 3})
    words = distinct[:: max(1, len(distinct) // 1500)]
    letters = "aeiouybcdlnrstzgw1"
    for _ in range(600):
        stem = "".join(rng.choice(letters) for _ in range(rng.randrange(0, 6)))
        words.append(stem + "".join(rng.choice(SUFFIXES) for _ in range(rng.randrange(0, 3))))
    porter = PorterStemmer()
    rows += [{"stem": w, "out": porter.stem(w)} for w in sorted(set(words))]

    # Tokenizer.
    for text in EDGE_TEXTS + [d[:300] for d in docs[3000:3010]]:
        for stemmer in (False, True):
            toks = tokenizers.DefaultTokenizer(stemmer).tokenize(text)
            rows.append({"text": text, "stemmer": stemmer, "tokens": toks})

    # Scores: real document starts (with their newlines) and edge cases.
    pairs = [(a, b) for a in EDGE_TEXTS[:4] for b in EDGE_TEXTS[2:]]
    for i in range(45):
        a, b = docs[4000 + 2 * i], docs[4001 + 2 * i]
        pairs.append((a[: rng.randrange(50, 450)], b[: rng.randrange(50, 450)]))
    for target, prediction in pairs:
        for stemmer in (False, True):
            result = rouge_scorer.RougeScorer(TYPES, use_stemmer=stemmer).score(target, prediction)
            rows.append(
                {
                    "target": target,
                    "prediction": prediction,
                    "stemmer": stemmer,
                    "types": TYPES,
                    "scores": [encode(result[t]) for t in TYPES],
                }
            )

    out = Path(__file__).with_name("rouge_golden.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    sys.stdout.write(f"wrote {len(rows)} rows to {out}\n")


if __name__ == "__main__":
    main(sys.argv[1])
