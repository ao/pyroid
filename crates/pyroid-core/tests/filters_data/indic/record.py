"""Record what datatrove 0.10.1's `IndicNLPTokenizer` returns, for `filters_indic.rs`.

Derived from datatrove 0.10.1 (Apache-2.0), (c) HuggingFace, indic-nlp-library 0.92
(MIT), (c) Anoop Kunchukuttan, and urduhack 1.0.3 (MIT), (c) Urdu Hack; modified.

Run inside the pyroid-dev container (never on the host), in a venv with
datatrove==0.10.1, indic-nlp-library (0.92) and pyarrow (for `corpus`):

    python record.py golden OUT.jsonl --eos EOS     # the committed recording
    python record.py corpus OUT.jsonl --eos EOS     # every /data/fineweb2 doc and line
    python record.py fuzz OUT.jsonl N SEED --eos EOS
    python record.py rerecord IN.jsonl OUT.jsonl --eos EOS   # IN's texts, this interpreter

`--eos` is urduhack 1.0.3's `urduhack/tokenization/eos.py` (with `tokenizer.py`
next to it). indic-nlp's `sentence_split(text, "ur")` runs
`from urduhack.tokenization import sentence_tokenizer`; importing the real
package needs TensorFlow, so the two pure-Python files that define
`sentence_tokenizer` are loaded under their real module names, with the
TensorFlow-only `keras_tokenizer` and `config` imports stubbed out. Without
`--eos`, `ur` sentence rows record the ModuleNotFoundError datatrove raises.

One line per (text, language):

    {"t": text, "c": indic-nlp code, "l": datatrove language, "u": unidata_version,
     "k": kind, "w": split_into_words(text, l), "s": len(split_into_sentences(text, l)),
     "e": [b for a, b in span_tokenize(text)], "z": sent_tokenize(text)}

A call that raises is recorded as "w!" / "s!" / "e!" / "z!": "ExceptionType: message".
"z" is left out when it would be long. Raw U+0085, U+2028 and U+2029 are written
as JSON escapes so that every line splitter agrees.
"""

# ruff: noqa: RUF001  (look-alike characters on purpose)

from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import json
import random
import string
import sys
import types
import unicodedata
from multiprocessing import Pool
from pathlib import Path

UNIDATA = unicodedata.unidata_version
GOLDEN_SEED = 20261106
GOLDEN_FUZZ = 1200
GOLDEN_DOCS = 6
GOLDEN_LINE_DOCS = 8
MAX_Z = 3000  # characters of "z" kept
# (slice, indic-nlp code) pairs `corpus` records: Hindi text under danda and non-danda
# Devanagari languages, Arabic text under Urdu.
CORPUS = [
    ("hin_Deva", "hi"),
    ("hin_Deva", "mr"),
    ("hin_Deva", "ne"),
    ("hin_Deva", "sa"),
    ("arb_Arab", "ur"),
]

# The scripts of indic-nlp's languages: indic-nlp code -> (block start, block end).
BLOCKS = {
    "hi": (0x900, 0x97F),
    "mr": (0x900, 0x97F),
    "ne": (0x900, 0x97F),
    "sa": (0x900, 0x97F),
    "kK": (0x900, 0x97F),
    "bn": (0x980, 0x9FF),
    "as": (0x980, 0x9FF),
    "pa": (0xA00, 0xA7F),
    "gu": (0xA80, 0xAFF),
    "or": (0xB00, 0xB7F),
    "ta": (0xB80, 0xBFF),
    "te": (0xC00, 0xC7F),
    "kn": (0xC80, 0xCFF),
    "ml": (0xD00, 0xD7F),
    "si": (0xD80, 0xDFF),
    "ur": (0x600, 0x6FF),
}
DELIMS = ".?!।॥\uaaf0\uaaf1\uabeb\uabec\uabed\uabee\uabef\u1c7e\u1c7f"
URDU_PUNCT = "\u0609\u060a\u060c\u061e\u066a\u066b\u066c\u066d\u06d4\u061f"
SPACES = [
    " ",
    " ",
    " ",
    " ",
    "  ",
    "\t",
    "\n",
    "\n\n",
    "\r\n",
    "\x0b",
    "\x0c",
    "\x1c",
    "\x85",
    "\xa0",
    "\u2000",
    "\u3000",
    "\u2028",
    "\u200b",
    " \t ",
]
# Characters whose str.isnumeric() differs between Unicode 14.0 and 17.0.
VERSION_NUMERIC = [
    (0x4E24, 0x4E24),
    (0x4EAC, 0x4EAC),
    (0x4FE9, 0x4FE9),
    (0x5006, 0x5006),
    (0x62D0, 0x62D0),
    (0x6D1E, 0x6D1E),
    (0x7695, 0x7695),
    (0x79ED, 0x79ED),
    (0x920E, 0x920E),
    (0x94A9, 0x94A9),
    (0x10D40, 0x10D49),
    (0x116D0, 0x116E3),
    (0x11BF0, 0x11BF9),
    (0x11DE0, 0x11DE9),
    (0x11F50, 0x11F59),
    (0x12038, 0x12039),
    (0x12079, 0x12079),
    (0x12226, 0x12226),
    (0x1222B, 0x1222B),
    (0x1230B, 0x1230B),
    (0x1230D, 0x1230D),
    (0x12399, 0x12399),
    (0x16130, 0x16139),
    (0x16D70, 0x16D79),
    (0x16FF4, 0x16FF6),
    (0x1CCF0, 0x1CCF9),
    (0x1D2C0, 0x1D2D3),
    (0x1E4F0, 0x1E4F9),
    (0x1E5F1, 0x1E5FA),
]
OTHER_NUMERIC = "½²³¹¼Ⅻⅰ一二三十百千万亿零〇०९০৯੦૦୦௦௰౦౸೦൦෦٠٩۰۹"


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


# --------------------------------------------------------------------------
# urduhack without TensorFlow
# --------------------------------------------------------------------------


def install_urduhack(eos: str) -> None:
    """Make `from urduhack.tokenization import sentence_tokenizer` work with the
    real eos.py and tokenizer.py of urduhack, minus their TensorFlow imports."""
    tok_dir = Path(eos).resolve().parent
    pkg = types.ModuleType("urduhack")
    pkg.__path__ = [str(tok_dir.parent)]
    sub = types.ModuleType("urduhack.tokenization")
    sub.__path__ = [str(tok_dir)]
    keras = types.ModuleType("urduhack.tokenization.keras_tokenizer")
    for name in ("_is_model_exist", "_preprocess_sentence", "_retrieve_words", "_load_model"):
        setattr(keras, name, None)
    config = types.ModuleType("urduhack.config")
    config.MODEL_PATH = config.VOCAB_PATH = None
    sys.modules.update(
        {
            "urduhack": pkg,
            "urduhack.tokenization": sub,
            "urduhack.tokenization.keras_tokenizer": keras,
            "urduhack.config": config,
        }
    )
    for name in ("eos", "tokenizer"):
        spec = importlib.util.spec_from_file_location(
            f"urduhack.tokenization.{name}", tok_dir / f"{name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    sub.sentence_tokenizer = sys.modules["urduhack.tokenization.tokenizer"].sentence_tokenizer


# --------------------------------------------------------------------------
# recording
# --------------------------------------------------------------------------


def languages() -> dict[str, str]:
    """indic-nlp code -> one datatrove language using it (3-letter code with script)."""
    from datatrove.utils.word_tokenizers import load_tokenizer_assignments

    out: dict[str, str] = {}
    for code, factory in sorted(load_tokenizer_assignments().items()):
        cls, arg = factory.args
        if cls == "IndicNLPTokenizer" and "_" in code and len(code.split("_")[0]) == 3:
            out.setdefault(arg, code)
    return out


def _call(row: dict, key: str, fn) -> None:
    try:
        row[key] = fn()
    except Exception as e:
        row[key + "!"] = f"{type(e).__name__}: {e}"


def record(item: tuple[str, str, str, dict]) -> str:
    from datatrove.utils.text import split_into_sentences, split_into_words
    from datatrove.utils.word_tokenizers import load_word_tokenizer

    text, code, lang, extra = item
    row = {"t": text, "c": code, "l": lang, "u": UNIDATA, **extra}
    tok = load_word_tokenizer(lang)
    assert type(tok).__name__ == "IndicNLPTokenizer", (lang, tok)
    assert tok.language == code, (lang, tok)
    _call(row, "w", lambda: split_into_words(text, lang))
    _call(row, "s", lambda: len(split_into_sentences(text, lang)))
    _call(row, "e", lambda: [b for _, b in tok.span_tokenize(text)])
    _call(row, "z", lambda: tok.sent_tokenize(text))
    if "z" in row and sum(map(len, row["z"])) > MAX_Z:
        del row["z"]
    line = json.dumps(row, ensure_ascii=False)
    return (
        line.replace("\u0085", "\\u0085").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    )


def write(path: str, items: list, eos: str | None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Pool(
        initializer=install_urduhack if eos else None, initargs=(eos,) if eos else ()
    ) as pool:
        lines = pool.map(record, items, chunksize=64)
    Path(path).write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    log(f"wrote {path}: {len(lines)} rows (Python {sys.version.split()[0]}, Unicode {UNIDATA})")


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def corpus_docs(name: str) -> list[str]:
    import pyarrow.parquet as pq

    return (
        pq.read_table(f"/data/fineweb2/{name}.parquet", columns=["text"]).column("text").to_pylist()
    )


class Fuzz:
    """Texts built from pieces that hit every branch of the port."""

    def __init__(self, rng: random.Random, langs: dict[str, str]) -> None:
        from indicnlp.tokenize import sentence_tokenize

        self.rng = rng
        self.langs = langs
        self.codes = sorted(langs)
        src = inspect.getsource(sentence_tokenize.is_acronym_abbvr)
        node = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Set))
        self.ack = sorted(ast.literal_eval(node))
        eos = sys.modules.get("urduhack.tokenization.eos")
        self.urdu_words = (
            sorted(set(eos._URDU_NEWLINE_WORDS) | set(eos._URDU_CONJUNCTIONS))
            if eos
            else ["ہے", "اور"]
        )
        self.sinhala = {
            v: k
            for k, v in __import__(
                "indicnlp.transliterate.sinhala_transliterator", fromlist=["x"]
            ).SinhalaDevanagariTransliterator.sinhala_devnag_map.items()
        }

    def char(self, lo: int, hi: int) -> str:
        return chr(self.rng.randint(lo, hi))

    def word(self, code: str) -> str:
        lo, hi = BLOCKS[self.rng.choice(self.codes) if self.rng.random() < 0.1 else code]
        return "".join(self.char(lo, hi) for _ in range(self.rng.randint(1, 6)))

    def acronym(self, code: str) -> str:
        a = self.rng.choice(self.ack)
        if code == "si":
            a = "".join(self.sinhala.get(c, c) for c in a)
        elif code != "ur" and self.rng.random() < 0.8:
            base = BLOCKS[code][0]
            a = "".join(chr(ord(c) - 0x900 + base) for c in a)
        return a

    def number(self) -> str:
        r = self.rng
        sep = lambda: r.choice(["", "", " ", "\t", "  "])  # noqa: E731
        parts = [str(r.randint(0, 10 ** r.randint(1, 4)))]
        for _ in range(r.randint(0, 3)):
            parts += [sep() + r.choice(",.:/,-;") + sep(), str(r.randint(0, 999))]
        s = "".join(parts)
        if r.random() < 0.2:
            s = r.choice(["x", "क", "ab"]) + s
        if r.random() < 0.2:
            s += r.choice(["x", "क", "%"])
        return s

    def numeric(self) -> str:
        if self.rng.random() < 0.5:
            lo, hi = self.rng.choice(VERSION_NUMERIC)
            return self.char(lo, hi)
        return self.rng.choice(OTHER_NUMERIC)

    def piece(self, code: str) -> str:
        r = self.rng
        kind = r.choices(
            [
                "word",
                "word.",
                "ack.",
                "delim",
                "number",
                "numeric.",
                "punct",
                "urdu",
                "latin",
                "space",
                "digits",
            ],
            [30, 8, 8, 10, 8, 5, 6, 6, 5, 3, 3],
        )[0]
        if kind == "word":
            return self.word(code)
        if kind == "word.":
            return self.word(code) + r.choice([".", ".", "?", "।"])
        if kind == "ack.":
            return self.acronym(code) + "."
        if kind == "delim":
            return r.choice(DELIMS) * r.choice([1, 1, 2])
        if kind == "number":
            return self.number()
        if kind == "numeric.":
            return self.numeric() + r.choice(DELIMS)
        if kind == "punct":
            return r.choice(string.punctuation + URDU_PUNCT)
        if kind == "urdu":
            return r.choice(self.urdu_words) if r.random() < 0.7 else r.choice("۔؟،")
        if kind == "latin":
            return r.choice(["Dr", "U.S", "abc", "A", "Mr.", "e.g.", "x", "\U0010ffff"])
        if kind == "space":
            return r.choice(SPACES)
        return str(r.randint(0, 99999))

    def text(self, code: str) -> str:
        r = self.rng
        out = []
        for i in range(r.randint(1, r.choice([4, 12, 40]))):
            if i:
                x = r.random()
                out.append("" if x < 0.2 else " " if x < 0.75 else r.choice(SPACES))
            out.append(self.piece(code))
        s = "".join(out)
        if r.random() < 0.15:
            s = r.choice(SPACES) + s
        if r.random() < 0.15:
            s += r.choice(SPACES)
        return s

    def items(self, n: int) -> list:
        out = []
        for _ in range(n):
            code = self.rng.choice(self.codes)
            out.append((self.text(code), code, self.langs[code], {"k": "fuzz"}))
        return out


def corpus_items(name: str, code: str, langs: dict[str, str], doc_ids, line_ids) -> list:
    """Docs `doc_ids` and the non-blank lines of docs `line_ids` of a FineWeb-2 slice."""
    docs = corpus_docs(name)
    lang = langs[code]
    out = [(docs[i], code, lang, {"k": "doc", "id": f"{name}/{i}"}) for i in doc_ids]
    for i in line_ids:
        out += [
            (ln, code, lang, {"k": "line", "id": f"{name}/{i}"})
            for ln in docs[i].splitlines()
            if ln.strip()
        ]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["golden", "corpus", "fuzz", "rerecord"])
    p.add_argument("args", nargs="+")
    p.add_argument("--eos")
    a = p.parse_args()
    if a.eos:
        install_urduhack(a.eos)
    langs = languages()
    if a.cmd == "golden":
        (out,) = a.args
        items = corpus_items(
            "hin_Deva",
            "hi",
            langs,
            range(GOLDEN_DOCS),
            range(GOLDEN_DOCS, GOLDEN_DOCS + GOLDEN_LINE_DOCS),
        )
        items += corpus_items("arb_Arab", "ur", langs, [], range(2))
        items += Fuzz(random.Random(GOLDEN_SEED), langs).items(GOLDEN_FUZZ)
        write(out, items, a.eos)
    elif a.cmd == "corpus":
        (out,) = a.args
        items = []
        for name, code in CORPUS:
            n = len(corpus_docs(name))
            items += corpus_items(name, code, langs, range(n), range(n))
        write(out, items, a.eos)
    elif a.cmd == "fuzz":
        out, n, seed = a.args
        write(out, Fuzz(random.Random(int(seed)), langs).items(int(n)), a.eos)
    else:
        src, out = a.args
        items = []
        for line in Path(src).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            extra = {k: row[k] for k in ("k", "id") if k in row}
            items.append((row["t"], row["c"], row["l"], extra))
        write(out, items, a.eos)


if __name__ == "__main__":
    main()
