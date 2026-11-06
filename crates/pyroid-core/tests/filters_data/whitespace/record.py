"""Record datatrove 0.10.1 `WhitespaceTokenizer` sentences, for `filters_whitespace.rs`.

Derived from datatrove 0.10.1 (Apache-2.0), (c) HuggingFace; modified.

Run inside the pyroid-dev container (never on the host), in a venv with
datatrove==0.10.1, regex==2026.9.29 and pyarrow (for `corpus`):

    python record.py golden OUT.jsonl       # the committed recording
    python record.py corpus OUT.jsonl       # every /data/fineweb2 doc and line
    python record.py fuzz OUT.jsonl N SEED
    python record.py rerecord IN.jsonl OUT.jsonl

The sentence regex is the `regex` module's, so the CPython version does not
matter (only `str.strip()` does, and it is the same in every version); "u" is
recorded anyway. One line per text:

    {"t": text, "u": unidata_version, "k": kind,
     "e": [b for a, b in WhitespaceTokenizer().span_tokenize(text)],
     "s": len(split_into_sentences(text, LANG)), "z": sent_tokenize(text)}

"z" is left out when it would be long. Raw U+0085, U+2028 and U+2029 are
written as JSON escapes.
"""

# ruff: noqa: RUF001  (look-alike characters on purpose)

from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from multiprocessing import Pool
from pathlib import Path

UNIDATA = unicodedata.unidata_version
LANG = "kat_Geor"  # any language datatrove gives WhitespaceTokenizer()
GOLDEN_SEED = 20261107
GOLDEN_FUZZ = 1500
MAX_Z = 3000
SLICES = [
    "arb_Arab",
    "cmn_Hani",
    "deu_Latn",
    "fra_Latn",
    "hin_Deva",
    "jpn_Jpan",
    "rus_Cyrl",
    "spa_Latn",
]
QUOTES = "\"'”“’«»"
SPACES = [
    " ",
    " ",
    " ",
    "  ",
    "\t",
    "\n",
    "\n\n",
    "\r\n",
    "\r",
    "\x0b",
    "\x0c",
    "\x1c",
    "\x85",
    "\xa0",
    "\u2000",
    "\u3000",
    "\u2028",
    "\u2029",
    "\u200b",
]


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def record(item: tuple[str, dict]) -> str:
    from datatrove.utils.text import split_into_sentences
    from datatrove.utils.word_tokenizers import WhitespaceTokenizer, load_word_tokenizer

    text, extra = item
    tok = load_word_tokenizer(LANG)
    assert type(tok) is WhitespaceTokenizer
    row = {"t": text, "u": UNIDATA, **extra}
    for key, fn in (
        ("e", lambda: [b for _, b in tok.span_tokenize(text)]),
        ("s", lambda: len(split_into_sentences(text, LANG))),
        ("z", lambda: tok.sent_tokenize(text)),
    ):
        try:
            row[key] = fn()
        except Exception as e:
            row[key + "!"] = f"{type(e).__name__}: {e}"
    if "z" in row and sum(map(len, row["z"])) > MAX_Z:
        del row["z"]
    line = json.dumps(row, ensure_ascii=False)
    return (
        line.replace("\u0085", "\\u0085").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    )


def write(path: str, items: list) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Pool() as pool:
        lines = pool.map(record, items, chunksize=32)
    Path(path).write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    log(f"wrote {path}: {len(lines)} rows (Python {sys.version.split()[0]}, Unicode {UNIDATA})")


def corpus(doc_ids, line_ids) -> list:
    import pyarrow.parquet as pq

    out = []
    for name in SLICES:
        docs = (
            pq.read_table(f"/data/fineweb2/{name}.parquet", columns=["text"])
            .column("text")
            .to_pylist()
        )
        out += [(docs[i], {"k": "doc", "id": f"{name}/{i}"}) for i in doc_ids(len(docs))]
        for i in line_ids(len(docs)):
            out += [
                (ln, {"k": "line", "id": f"{name}/{i}"})
                for ln in docs[i].splitlines()
                if ln.strip()
            ]
    return out


class Fuzz:
    def __init__(self, rng: random.Random) -> None:
        import regex
        from datatrove.utils.text import TERMINAL_PUNCTUATION

        self.rng = rng
        self.terminal = sorted(TERMINAL_PUNCTUATION)
        # Uppercase letters of every script, by the `regex` module's own data;
        # some are not Lu in this CPython's unicodedata (newer Unicode).
        lu = regex.compile(r"\p{Lu}")
        self.upper = [
            chr(c)
            for c in range(0x80, 0x110000)
            if not 0xD800 <= c < 0xE000 and lu.fullmatch(chr(c))
        ]
        self.newer = [c for c in self.upper if unicodedata.category(c) != "Lu"]
        self.lower = [
            chr(c)
            for c in range(0x80, 0x3000)
            if unicodedata.category(chr(c)) in ("Ll", "Lo", "Mn")
        ]

    def piece(self) -> str:
        r = self.rng
        kind = r.choices(
            [
                "word",
                "Word",
                "U.",
                "U.U.",
                "term",
                "term+",
                "dot",
                "quote",
                "space",
                "newline",
                "digit",
                "other",
            ],
            [25, 10, 10, 5, 8, 4, 8, 6, 5, 6, 4, 9],
        )[0]
        if kind == "word":
            return "".join(r.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(r.randint(1, 7)))
        if kind == "Word":
            pool = self.newer if r.random() < 0.2 else self.upper
            return r.choice(pool) + "".join(r.choice(self.lower) for _ in range(r.randint(0, 5)))
        if kind == "U.":
            return (
                r.choice(
                    self.newer if r.random() < 0.2 else self.upper if r.random() < 0.7 else "ABCXYZ"
                )
                + "."
            )
        if kind == "U.U.":
            return "".join(r.choice("ABCDEUSНЖΣ") + "." for _ in range(r.randint(2, 4)))
        if kind == "term":
            return r.choice(self.terminal)
        if kind == "term+":
            return "".join(
                r.choice([*self.terminal, "\n", ".", "!", "?"]) for _ in range(r.randint(2, 4))
            )
        if kind == "dot":
            return "." * r.choice([1, 1, 1, 2, 3])
        if kind == "quote":
            return r.choice(QUOTES)
        if kind == "space":
            return r.choice(SPACES)
        if kind == "newline":
            return r.choice(["\n", "\n\n", "\r\n", "\n \n"])
        if kind == "digit":
            return str(r.randint(0, 9999))
        return r.choice(self.lower)

    def text(self) -> str:
        r = self.rng
        out = []
        for i in range(r.randint(1, r.choice([3, 10, 40]))):
            if i:
                x = r.random()
                out.append("" if x < 0.3 else " " if x < 0.8 else r.choice(SPACES))
            out.append(self.piece())
        return "".join(out)

    def long(self) -> str:
        """A long text without terminals, maybe with one at the end."""
        r = self.rng
        body = " ".join(
            r.choice(["abc", "A", "Ab", "XY", "h\u0303", "Д", "\u00e9t\u00e9"])
            for _ in range(r.randint(300, 3000))
        )
        return body + r.choice(["", "", ".", "A.", '!"', "\n"])

    def items(self, n: int, long: float = 0.0) -> list:
        """`n` texts, a fraction `long` of them long."""
        return [
            (self.long() if self.rng.random() < long else self.text(), {"k": "fuzz"})
            for _ in range(n)
        ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["golden", "corpus", "fuzz", "rerecord"])
    p.add_argument("args", nargs="+")
    a = p.parse_args()
    if a.cmd == "golden":
        (out,) = a.args
        items = corpus(lambda n: [], lambda n: range(1))
        items += Fuzz(random.Random(GOLDEN_SEED)).items(GOLDEN_FUZZ)
        write(out, items)
    elif a.cmd == "corpus":
        (out,) = a.args
        write(out, corpus(range, range))
    elif a.cmd == "fuzz":
        out, n, seed = a.args
        write(out, Fuzz(random.Random(int(seed))).items(int(n), long=0.004))
    else:
        src, out = a.args
        items = []
        for line in Path(src).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            items.append((row["t"], {k: row[k] for k in ("k", "id") if k in row}))
        write(out, items)


if __name__ == "__main__":
    main()
