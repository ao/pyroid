"""pyroid.fasttext against fastText itself.

The models are synthetic: :func:`build` writes ``.bin`` files byte for byte in
fastText's format (dense, product-quantized, ``qnorm``, ``qout``, pruned; every
head; format versions 11 and 12), with weights from a seeded RNG.

- **Live-oracle parity**: every model predicts the same labels and
  probabilities (|dp| <= 1e-6; 1e-4 on macOS, see ``PROB_TOL``), and raises
  the same errors, as fastText's Python package (fasttext-predict 0.9.2.4,
  ``import fasttext``), on a fixed corpus and on Hypothesis-generated lines.
  Skipped where the oracle is not installed (it has no free-threaded wheels).
- **Reference model**: what fasttext-predict does not expose (vectors,
  matrices, ids, subwords, ``get_line``) is checked against :class:`Ref`, a
  pure-Python copy of fastText's dictionary and float32 arithmetic over the
  weights the writer produced.
- Return types, container kinds (list, tuple, Arrow), threads, load and
  argument errors, and the ImportError without numpy.
"""

import functools
import importlib
import math
import os
import random
import re
import struct
import subprocess
import sys
import sysconfig
import threading
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

try:
    import numpy as np
    import numpy.typing as npt
except ImportError:  # pragma: no cover - numpy is the one dependency of pyroid[fasttext]
    pytest.skip("pyroid.fasttext needs numpy", allow_module_level=True)

import pyroid
import pyroid.fasttext as ft
import pyroid.text
from pyroid._pyroid import _testing

FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
RunPython = Callable[..., subprocess.CompletedProcess[str]]
F32 = npt.NDArray[np.float32]
#: The Hypothesis profile conftest.py loaded (some tests scale their example counts).
PROFILE = settings()

# ---- the model writer --------------------------------------------------------------

MAGIC = 793_712_314
LOSSES = {"hs": 1, "ns": 2, "softmax": 3, "ova": 4}
MODELS = {"cbow": 1, "sg": 2, "sup": 3}
KSUB = 256  # centroids per sub-quantizer (8-bit codes)

WORDS: tuple[str | bytes, ...] = (
    "</s>",
    "the",
    "cat",
    "sat",
    "on",
    "mat",
    "café",
    "日本",
    "naïve",
    "\U0001f642ok",
)
LABELS: tuple[str | bytes, ...] = tuple(f"__label__{c}" for c in "abcde")
UNICODE_WORDS: tuple[str | bytes, ...] = (
    "</s>",
    "the",
    "café",  # 2-byte UTF-8
    "日本",  # 3-byte
    "\U0001f642ok",  # 4-byte
    "e\u0301",  # combining acute accent
    "straße",
    "a\xa0b",  # NBSP is not a fastText separator...
    "\u3000x",  # ...nor is the ideographic space...
    "\x1cfs",  # ...nor U+001C (str.split() splits on all three)
    "مرحبا",
)


@dataclass(frozen=True)
class Spec:
    """One synthetic model: its header arguments, vocabulary and weight storage."""

    words: tuple[str | bytes, ...] = WORDS
    labels: tuple[str | bytes, ...] = LABELS
    label_counts: tuple[int, ...] | None = None
    dim: int = 8
    loss: str = "softmax"
    model: str = "sup"
    minn: int = 2
    maxn: int = 4
    word_ngrams: int = 1
    bucket: int = 97
    version: int = 12
    quant: bool = False
    qnorm: bool = False
    qout: bool = False
    dsub: int = 2
    #: The pruning index of a quantized model: (n-gram bucket, row) pairs.
    prune: tuple[tuple[int, int], ...] | None = None
    #: Give every input (output) weight or centroid this value instead of a random one.
    fill_in: float | None = None
    fill_out: float | None = None
    scale: float = 1.0
    seed: int = 0


@dataclass
class Synth:
    """A written model and its weights (quantized rows as fastText decodes them)."""

    spec: Spec
    data: bytes
    words: list[bytes]
    labels: list[bytes]
    word_counts: list[int]
    label_counts: list[int]
    input_rows: F32
    output_rows: F32


def _bytes(s: str | bytes) -> bytes:
    return s.encode() if isinstance(s, str) else s


def _f32(values: F32) -> bytes:
    return values.astype("<f4").tobytes()


def _weights(rng: np.random.Generator, n: int, scale: float, fill: float | None) -> F32:
    if fill is not None:
        return np.full(n, fill, dtype=np.float32)
    return rng.uniform(-scale, scale, n).astype(np.float32)


def _dense(
    rng: np.random.Generator, rows: int, dim: int, scale: float, fill: float | None
) -> tuple[bytes, F32]:
    """``DenseMatrix::save``."""
    w = _weights(rng, rows * dim, scale, fill).reshape(rows, dim)
    return struct.pack("<qq", rows, dim) + _f32(w), w


def _quantized(
    rng: np.random.Generator,
    rows: int,
    dim: int,
    dsub: int,
    qnorm: bool,
    scale: float,
    fill: float | None,
) -> tuple[bytes, F32]:
    """``QuantMatrix::save``, and each row as ``addRowToVector`` decodes it."""
    nsubq = -(-dim // dsub)
    lastdsub = dim % dsub or dsub
    cents = _weights(rng, dim * KSUB, scale, fill)
    codes = rng.integers(0, KSUB, rows * nsubq, dtype=np.uint8)
    out = struct.pack("<?qqi", qnorm, rows, dim, codes.size) + codes.tobytes()
    out += struct.pack("<4i", dim, nsubq, dsub, lastdsub) + _f32(cents)
    norms = np.ones(rows, dtype=np.float32)
    if qnorm:
        norm_codes = rng.integers(0, KSUB, rows, dtype=np.uint8)
        norm_cents = rng.uniform(0.1, 2.0, KSUB).astype(np.float32)
        out += norm_codes.tobytes() + struct.pack("<4i", 1, 1, 1, 1) + _f32(norm_cents)
        norms = norm_cents[norm_codes]
    decoded = np.zeros((rows, dim), dtype=np.float32)
    for i in range(rows):
        for m in range(nsubq):
            code = int(codes[i * nsubq + m])
            if m == nsubq - 1:  # ProductQuantizer::get_centroids
                start, width = m * KSUB * dsub + code * lastdsub, lastdsub
            else:
                start, width = (m * KSUB + code) * dsub, dsub
            # x += alpha * c with x == 0: one rounding, fused or not.
            decoded[i, m * dsub : m * dsub + width] = norms[i] * cents[start : start + width]
    return out, decoded


def build(spec: Spec) -> Synth:
    """Write ``spec`` as a fastText model file (``FastText::saveModel``)."""
    rng = np.random.default_rng(spec.seed)
    words = [_bytes(w) for w in spec.words]
    labels = [_bytes(x) for x in spec.labels]
    word_counts = [1000 - 3 * i for i in range(len(words))]
    label_counts = list(spec.label_counts or [500 - 7 * i for i in range(len(labels))])
    prune = None if spec.prune is None else dict(spec.prune)

    out = struct.pack("<ii", MAGIC, spec.version)
    # Args::save: dim ws epoch minCount neg wordNgrams loss model bucket minn maxn
    # lrUpdateRate t.
    out += struct.pack(
        "<12id",
        spec.dim,
        5,
        5,
        1,
        5,
        spec.word_ngrams,
        LOSSES[spec.loss],
        MODELS[spec.model],
        spec.bucket,
        spec.minn,
        spec.maxn,
        100,
        1e-4,
    )
    # Dictionary::save: sizes, then the entries (words first), then the pruning index.
    out += struct.pack(
        "<iiiqq",
        len(words) + len(labels),
        len(words),
        len(labels),
        12_345,
        -1 if prune is None else len(prune),
    )
    for w, c in zip(words, word_counts, strict=True):
        out += w + b"\0" + struct.pack("<qb", c, 0)
    for w, c in zip(labels, label_counts, strict=True):
        out += w + b"\0" + struct.pack("<qb", c, 1)
    for pair in (prune or {}).items():
        out += struct.pack("<ii", *pair)

    ngram_rows = spec.bucket if prune is None else max(prune.values(), default=-1) + 1
    rows_in = len(words) + ngram_rows
    rows_out = len(labels) if spec.model == "sup" else len(words)
    out += struct.pack("<?", spec.quant)
    if spec.quant:
        chunk, inp = _quantized(
            rng, rows_in, spec.dim, spec.dsub, spec.qnorm, spec.scale, spec.fill_in
        )
    else:
        chunk, inp = _dense(rng, rows_in, spec.dim, spec.scale, spec.fill_in)
    out += chunk + struct.pack("<?", spec.qout)
    if spec.quant and spec.qout:  # fastText quantizes the output with dsub 2
        chunk, outp = _quantized(rng, rows_out, spec.dim, 2, spec.qnorm, spec.scale, spec.fill_out)
    else:
        chunk, outp = _dense(rng, rows_out, spec.dim, spec.scale, spec.fill_out)
    out += chunk
    return Synth(spec, out, words, labels, word_counts, label_counts, inp, outp)


# ---- a pure-Python fastText dictionary ------------------------------------------------

EOS = b"</s>"
LABEL_PREFIX = b"__label__"
MASK64 = (1 << 64) - 1
#: readWord's separators (the text never contains "\n": predict rejects it).
_TOKEN_SPLIT = re.compile(rb"[ \r\t\v\f\x00]+")
#: ``iss >> word`` in the C locale (unsupervised sentence vectors): no NUL.
_ISSPACE_SPLIT = re.compile(r"[ \t\n\v\f\r]+")


def fnv1a(data: bytes) -> int:
    """``Dictionary::hash``: FNV-1a, XORing each byte sign-extended (char is signed)."""
    h = 2_166_136_261
    for b in data:
        h = ((h ^ (b | 0xFFFF_FF00 if b >= 0x80 else b)) * 16_777_619) & 0xFFFF_FFFF
    return h


def _widen(h: int) -> int:
    """A uint32 hash stored as int32, then converted to uint64 (``addWordNgrams``)."""
    return h | 0xFFFF_FFFF_0000_0000 if h & 0x8000_0000 else h


def _mean(rows: F32, ids: Sequence[int], dim: int) -> F32:
    """``addInputVector`` row by row in float32, then ``mul(1.0 / n)``."""
    acc = np.zeros(dim, dtype=np.float32)
    for i in ids:
        acc += rows[i]
    if ids:
        acc *= np.float32(1.0 / len(ids))
    return acc


class Ref:
    """fastText's dictionary and float32 arithmetic over one :class:`Synth`'s weights."""

    def __init__(self, synth: Synth) -> None:
        spec = synth.spec
        self.synth = synth
        self.entries = synth.words + synth.labels
        self.ids = {e: i for i, e in enumerate(self.entries)}
        self.nwords = len(synth.words)
        self.dim = spec.dim
        self.minn = spec.minn
        # Version-11 supervised models do not use character n-grams.
        self.maxn = 0 if spec.version == 11 and spec.model == "sup" else spec.maxn
        self.bucket = spec.bucket
        self.word_ngrams = spec.word_ngrams
        self.prune = None if spec.prune is None else dict(spec.prune)
        self.supervised = spec.model == "sup"
        self._ngrams: dict[bytes, list[tuple[bytes, int]]] = {}

    def _push(self, out: list[int], bucket_id: int) -> None:
        """``pushHash``: through the pruning index, if any."""
        if self.prune is not None:
            if bucket_id not in self.prune:  # (an empty index drops every n-gram)
                return
            bucket_id = self.prune[bucket_id]
        out.append(self.nwords + bucket_id)

    def char_ngrams(self, word: bytes) -> list[tuple[bytes, int]]:
        """``computeSubwords(BOW + word + EOW)``: each n-gram and its bucket."""
        cached = self._ngrams.get(word)
        if cached is not None:
            return cached
        w = b"<" + word + b">"
        out = []
        for i in range(len(w)):
            if w[i] & 0xC0 == 0x80:
                continue
            j, n = i, 1
            while j < len(w) and n <= self.maxn:
                j += 1
                while j < len(w) and w[j] & 0xC0 == 0x80:
                    j += 1
                if n >= self.minn and not (n == 1 and (i == 0 or j == len(w))):
                    out.append((w[i:j], fnv1a(w[i:j]) % self.bucket))
                n += 1
        self._ngrams[word] = out
        return out

    def _ngram_rows(self, word: bytes) -> list[int]:
        out: list[int] = []
        if word != EOS:
            for _, b in self.char_ngrams(word):
                self._push(out, b)
        return out

    def word_rows(self, word: bytes) -> list[int]:
        """``getSubwords(word)``: the rows a word vector averages."""
        i = self.ids.get(word, -1)
        return [i, *self._ngram_rows(word)] if i >= 0 else self._ngram_rows(word)

    def subwords(self, word: str) -> tuple[list[str], list[int]]:
        """The Python ``get_subwords``: n-grams dropped by pruning keep their string."""
        w = word.encode()
        strings, ids = [], []
        i = self.ids.get(w, -1)
        if i >= 0:
            strings.append(w)
            ids.append(i)
        if w != EOS:
            for s, b in self.char_ngrams(w):
                strings.append(s)
                self._push(ids, b)
        return [s.decode() for s in strings], ids

    @staticmethod
    def tokens(text: str) -> list[bytes]:
        """``readWord`` over ``text + "\\n"``, up to the first ``</s>``."""
        out = []
        for tok in _TOKEN_SPLIT.split(text.encode()):
            if tok:
                out.append(tok)
                if tok == EOS:
                    return out
        out.append(EOS)
        return out

    def _is_label(self, tok: bytes) -> bool:
        i = self.ids.get(tok, -1)
        return i >= self.nwords if i >= 0 else tok.startswith(LABEL_PREFIX)

    def line_rows(self, text: str) -> list[int]:
        """``Dictionary::getLine``: the input rows ``predict`` averages."""
        rows: list[int] = []
        hashes = []
        for tok in self.tokens(text):
            if self._is_label(tok):
                continue
            i = self.ids.get(tok, -1)
            if i < 0:
                rows += self._ngram_rows(tok)
            elif self.maxn <= 0:
                rows.append(i)
            else:
                rows += self.word_rows(tok)
            hashes.append(_widen(fnv1a(tok)))
        for i in range(len(hashes)):
            h = hashes[i]
            for j in range(i + 1, min(len(hashes), i + self.word_ngrams)):
                h = (h * 116_049_371 + hashes[j]) & MASK64
                self._push(rows, h % self.bucket)
        return rows

    def get_line(self, text: str) -> tuple[list[str], list[str]]:
        words, labels = [], []
        for tok in self.tokens(text):
            if not self._is_label(tok):
                words.append(tok.decode())
            elif tok in self.ids:
                labels.append(tok.decode())
        return words, labels

    def word_vector(self, word: str) -> F32 | None:
        """``getWordVector``; None if a row is outside the matrix (pyroid: ValueError)."""
        rows = self.word_rows(word.encode())
        if any(r >= len(self.synth.input_rows) for r in rows):
            return None
        return _mean(self.synth.input_rows, rows, self.dim)

    def sentence_vector(self, text: str) -> F32:
        """``getSentenceVector``."""
        if self.supervised:
            return _mean(self.synth.input_rows, self.line_rows(text), self.dim)
        svec = np.zeros(self.dim, dtype=np.float32)
        count = 0
        for word in _ISSPACE_SPLIT.split(text):
            vec = self.word_vector(word) if word else None
            if vec is None:
                continue
            sq = np.float32(0.0)
            for x in vec:
                sq = np.float32(sq + x * x)
            norm = np.sqrt(sq)
            if norm > 0:
                svec += vec * np.float32(1.0 / float(norm))
                count += 1
        if count:
            svec *= np.float32(1.0 / count)
        return svec

    def word_id(self, word: str) -> int:
        return self.ids.get(word.encode(), -1)

    def label_id(self, label: str) -> int:
        i = self.word_id(label)
        return i if i == -1 else i - self.nwords


# ---- the model zoo ---------------------------------------------------------------------


def _specs() -> dict[str, Spec]:
    storages: dict[str, dict[str, Any]] = {
        "dense": {},
        "quant": {"quant": True},
        "qnorm": {"quant": True, "qnorm": True},
        "qout": {"quant": True, "qout": True},
        "qnorm-qout": {"quant": True, "qnorm": True, "qout": True},
        "pruned": {"quant": True, "prune": tuple((3 * i, i) for i in range(20))},
    }
    specs: dict[str, Spec] = {}
    seed = 0
    for loss in LOSSES:
        for storage, kw in storages.items():
            seed += 1
            specs[f"{loss}-{storage}"] = Spec(loss=loss, word_ngrams=1 + seed % 3, seed=seed, **kw)
    many = tuple(f"__label__{i}" for i in range(60))
    specs |= {
        "softmax-v11": Spec(version=11, word_ngrams=2, seed=101),
        "hs-v11-quant": Spec(version=11, loss="hs", quant=True, qout=True, seed=102),
        # All-zero weights: every label ties, so the order is libstdc++'s heap order.
        "softmax-zero-in": Spec(fill_in=0.0, seed=103),
        "hs-zero-in": Spec(loss="hs", fill_in=0.0, seed=104),
        "ova-zero-out": Spec(loss="ova", fill_out=0.0, seed=105),
        "hs-zero-out": Spec(loss="hs", fill_out=0.0, seed=106),
        "ns-zero": Spec(loss="ns", fill_in=0.0, fill_out=0.0, seed=107),
        "softmax-zero-quant": Spec(quant=True, qout=True, fill_in=0.0, fill_out=0.0, seed=108),
        "hs-equal-counts": Spec(loss="hs", label_counts=(50,) * 5, seed=109),
        # No character n-grams and no buckets at all.
        "softmax-no-subwords": Spec(maxn=0, bucket=0, seed=110),
        "hs-no-subwords": Spec(loss="hs", maxn=0, bucket=0, seed=111),
        "softmax-pruned-empty": Spec(quant=True, prune=(), seed=112),
        "softmax-many-labels": Spec(labels=many, dim=10, dsub=3, quant=True, qout=True, seed=113),
        "hs-many-labels": Spec(loss="hs", labels=many, dim=10, word_ngrams=3, seed=114),
        "ova-many-labels": Spec(loss="ova", labels=many, seed=115),
        "softmax-unicode": Spec(words=UNICODE_WORDS, minn=1, maxn=6, word_ngrams=2, seed=116),
        "hs-unicode-quant": Spec(words=UNICODE_WORDS, loss="hs", quant=True, minn=3, seed=117),
        "softmax-no-eos": Spec(words=WORDS[1:], seed=118),
        "softmax-one-label": Spec(labels=LABELS[:1], seed=119),
        "hs-one-label": Spec(loss="hs", labels=LABELS[:1], seed=120),
        "ova-one-label": Spec(loss="ova", labels=LABELS[:1], seed=121),
        "softmax-scaled": Spec(scale=20.0, seed=122),
        "ova-scaled": Spec(loss="ova", scale=5.0, word_ngrams=2, seed=123),
        "hs-scaled": Spec(loss="hs", scale=5.0, seed=124),
        "softmax-dsub3": Spec(quant=True, qnorm=True, dsub=3, seed=125),
        "ova-dim1": Spec(loss="ova", dim=1, seed=126),
        "softmax-minn3-big-bucket": Spec(minn=3, maxn=3, bucket=1000, word_ngrams=3, seed=127),
        # Unsupervised models: predict raises, the vectors still work.
        "cbow-ns": Spec(model="cbow", loss="ns", labels=(), seed=128),
        "sg-hs": Spec(model="sg", loss="hs", labels=(), seed=129),
        "cbow-no-eos": Spec(model="cbow", loss="ns", words=WORDS[1:], labels=(), seed=130),
    }
    return specs


SPECS = _specs()
UNSUPERVISED = [n for n, s in SPECS.items() if s.model != "sup"]
#: Supervised dense models: fastText computes their vectors with float32 adds only.
EXACT = [n for n, s in SPECS.items() if not s.quant and s.model == "sup"]


@dataclass
class Entry:
    synth: Synth
    path: Path
    model: ft.Model
    ref: Ref = field(init=False)

    def __post_init__(self) -> None:
        self.ref = Ref(self.synth)


class Zoo:
    """The synthetic models, written and loaded on first use (thread-safe)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._entries: dict[str, Entry] = {}
        self._lock = threading.Lock()

    def __getitem__(self, name: str) -> Entry:
        with self._lock:
            if name not in self._entries:
                synth = build(SPECS[name])
                path = self.root / f"{name}.bin"
                path.write_bytes(synth.data)
                self._entries[name] = Entry(synth, path, ft.load_model(path))
            return self._entries[name]


@pytest.fixture(scope="session")
def zoo(tmp_path_factory: pytest.TempPathFactory) -> Zoo:
    return Zoo(tmp_path_factory.mktemp("fasttext-models"))


def _write(directory: Path, data: bytes) -> Path:
    """A file with a unique name (a test may run on several threads at once)."""
    path = directory / f"{uuid.uuid4().hex}.bin"
    path.write_bytes(data)
    return path


# ---- inputs ----------------------------------------------------------------------------

SEPARATORS = (" ", "\t", "\r", "\v", "\f", "\0")
#: Whitespace to Python, but not fastText separators: they stay inside tokens.
LOOKALIKES = ("\x1c", "\x85", "\xa0", "\u2028", "\u3000", "\u200b", "\ufeff")
TOKENS = sorted(
    {w.decode() for s in SPECS.values() for w in map(_bytes, s.words + s.labels)}
    | {"</s>", "__label__", "__label__zz", "the__label__a", "<the>", "he", "th", "zebra"}
)
KS = (1, 2, 3, -1, 1000)
THRESHOLDS = (0.0, 0.05, 0.5, math.nan)
COMBOS = [(k, t) for k in KS for t in THRESHOLDS] + [(0, 0.0), (-2, 0.0), (5, 2.0), (-1, -1.0)]
NEWLINE_MSG = "predict processes one line at a time (remove '\\n')"
GET_LINE_MSG = "get_line processes one line at a time (remove '\\n')"

_LINE_CHARS = st.characters(
    exclude_categories=["Cs"], exclude_characters="\n" + "".join(SEPARATORS)
)
token_st = st.one_of(
    st.sampled_from(TOKENS),
    st.sampled_from(LOOKALIKES),
    st.text(_LINE_CHARS, min_size=1, max_size=8),
)
#: Lines of vocabulary, label and OOV tokens, ``</s>``, and every separator.
line_st = st.lists(
    st.one_of(token_st, st.text(st.sampled_from(SEPARATORS), min_size=1, max_size=3)),
    max_size=14,
).map("".join)
line_lists = st.lists(line_st, max_size=100)  # crosses the 64-doc pool threshold
optional_lines = st.lists(st.none() | line_st, max_size=80)


def _corpus(rng: random.Random, n: int) -> list[str]:
    """``n`` random lines of TOKENS, OOV words and separators."""
    oov = "abcxyzé日\U0001f642ß\u0301"
    lines = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(0, 12)):
            if rng.random() < 0.2:
                parts.append("".join(rng.choice(oov) for _ in range(rng.randint(1, 8))))
            else:
                parts.append(rng.choice(TOKENS))
            parts.append("".join(rng.choice(SEPARATORS) for _ in range(rng.randint(1, 2))))
        if rng.random() < 0.1:
            parts.insert(rng.randrange(len(parts) + 1), rng.choice(LOOKALIKES))
        lines.append("".join(parts[: len(parts) - rng.randint(0, 1)]))
    return lines


CORPUS = [
    "",
    " ",
    "\t\0\v\f\r ",
    "</s>",
    "the </s> cat",
    "</s> the cat",
    "__label__a",
    "__label__a the cat",
    "the\vcat\fsat\rmat\0on\tthe",
    "the" * 50,
    " ".join(["the cat"] * 300),
    "café 日本 naïve \U0001f642ok",
    "\xa0the\u3000cat\x85 \x1cfs",
    "a" * 1000,
    *_corpus(random.Random(20_250_101), 120),
]


# ---- oracle helpers --------------------------------------------------------------------


def _load_oracle() -> ModuleType | None:
    try:
        mod = importlib.import_module("fasttext")
    except ImportError:
        return None
    return mod if hasattr(mod, "load_model") else None


ORACLE = _load_oracle()


@pytest.fixture(scope="session")
def oracle() -> ModuleType:
    """fastText's own Python package (fasttext-predict 0.9.2.4), or skip."""
    if ORACLE is None:
        pytest.skip("needs the fastText oracle: pip install fasttext-predict==0.9.2.4")
    return ORACLE


@functools.lru_cache(maxsize=4)
def _oracle_model(path: str) -> Any:
    assert ORACLE is not None
    return ORACLE.load_model(path)


Outcome = tuple[str, Any]


def _outcome(fn: Callable[..., Any], *args: Any) -> Outcome:
    """``("ok", result)``, or the exception's type name and message."""
    try:
        return "ok", fn(*args)
    except Exception as e:
        return type(e).__name__, str(e)


# pyroid replays fastText's Linux builds (GCC, libstdc++), which the Windows (MSVC) oracle matched
# on every case here. fasttext-predict's macOS wheels are built by clang against libc++: exact ties
# are ordered, and cut at k, differently (docs/guides/fasttext.md, "Tie order"), and dot products
# round differently in the last bits, which large logits amplify. Measured on macos-15 (arm64):
# softmax-scaled (scale 20) gives 0.5745614 there, 0.5745577 in pyroid and in fasttext-predict on
# Linux. There labels may differ in ties and near-ties, and probabilities by up to PROB_TOL.
ORACLE_BUILD_DIFFERS = sys.platform == "darwin"
PROB_TOL = 1e-4 if ORACLE_BUILD_DIFFERS else 1e-6


def _only_ties_differ(every: Outcome, got: Outcome) -> bool:
    """Whether each of ``got``'s labels has the probability fastText gives it in ``every``
    (its ``predict(line, -1, -1.0)``): then ``got`` differs from fastText's top-k in ties only."""
    if every[0] != "ok" or got[0] != "ok":
        return False
    prob = dict(zip(*every[1], strict=True))
    labels, probs = got[1]
    return len(set(labels)) == len(labels) and all(
        label in prob and abs(prob[label] - float(p)) <= PROB_TOL
        for label, p in zip(labels, probs, strict=True)
    )


def _prediction_mismatch(
    want: Outcome, got: Outcome, every: Callable[[], Outcome] | None = None
) -> str | None:
    """Why pyroid's ``predict(str)`` outcome differs from fastText's, or None.

    ``every`` gives fastText's prediction of every label for the same line; with it, labels that
    differ only in tie order are accepted where ``ORACLE_BUILD_DIFFERS``.
    """
    if want[0] != "ok" or got[0] != "ok":
        return None if want == got else f"{want} != {got}"
    (want_labels, want_probs), (labels, probs) = want[1], got[1]
    if type(labels) is not tuple or not isinstance(probs, np.ndarray):
        return f"types {type(labels)}, {type(probs)}"
    if probs.dtype != np.float64 or probs.shape != (len(labels),):
        return f"probs {probs.dtype} {probs.shape}"
    if labels != tuple(want_labels):
        ties_only = (
            ORACLE_BUILD_DIFFERS
            and every is not None
            and len(labels) == len(want_labels)
            and _only_ties_differ(every(), got)
        )
        if not ties_only:
            return f"labels {labels} != {want_labels}"
    worst = max((abs(float(a) - b) for a, b in zip(probs, want_probs, strict=True)), default=0)
    return None if worst <= PROB_TOL else f"|dp| = {worst} ({list(probs)} vs {list(want_probs)})"


def _notes(exc: BaseException | None) -> str:
    return "\n".join(getattr(exc, "__notes__", []))


Batch = list[str] | tuple[str, ...]


def _check_batch(model: ft.Model, lines: list[str], k: int, threshold: float) -> None:
    """A list or tuple gives each line's ``predict(str)`` result, or its first error."""
    singles = [_outcome(model.predict, s, k, threshold) for s in lines]
    errors = [(i, o) for i, o in enumerate(singles) if o[0] != "ok"]
    batches: list[Batch] = [lines, tuple(lines)]
    for batch in batches:
        if errors:
            first, (kind, message) = errors[0]
            with pytest.raises(Exception, match=re.escape(message)) as info:
                model.predict(batch, k, threshold)
            assert type(info.value).__name__ == kind
            assert f"index {first}" in _notes(info.value)
            continue
        labels, probs = model.predict(batch, k, threshold)
        assert type(labels) is list
        assert type(probs) is list
        assert len(labels) == len(probs) == len(lines)
        for (_, (want_labels, want_probs)), got_labels, got_probs in zip(
            singles, labels, probs, strict=True
        ):
            assert got_labels == list(want_labels)
            assert got_probs.dtype == np.float32
            np.testing.assert_array_equal(got_probs.astype(np.float64), want_probs)


# ---- 1. live-oracle parity ---------------------------------------------------------------


@pytest.mark.parametrize("name", SPECS)
def test_oracle_parity_on_the_corpus(oracle: ModuleType, zoo: Zoo, name: str) -> None:
    entry = zoo[name]
    ref = _oracle_model(str(entry.path))
    mismatches = []
    for k, threshold in COMBOS:
        for line in CORPUS:
            want = _outcome(ref.predict, line, k, threshold)
            got = _outcome(entry.model.predict, line, k, threshold)
            every = functools.partial(_outcome, ref.predict, line, -1, -1.0)
            why = _prediction_mismatch(want, got, every)
            if why is not None:
                mismatches.append((line[:60], k, threshold, why))
    assert not mismatches, f"{len(mismatches)} mismatches, e.g. {mismatches[:3]}"
    for k, threshold in [(1, 0.0), (-1, 0.05), (0, 0.0)]:
        _check_batch(entry.model, CORPUS, k, threshold)


@pytest.mark.parametrize("name", SPECS)
@settings(max_examples=max(10, PROFILE.max_examples // 4))
@given(
    lines=st.lists(line_st, min_size=1, max_size=6),
    k=st.sampled_from(KS),
    threshold=st.sampled_from(THRESHOLDS),
)
def test_oracle_parity_hypothesis(
    oracle: ModuleType, zoo: Zoo, name: str, lines: list[str], k: int, threshold: float
) -> None:
    entry = zoo[name]
    ref = _oracle_model(str(entry.path))
    for line in lines:
        want = _outcome(ref.predict, line, k, threshold)
        got = _outcome(entry.model.predict, line, k, threshold)
        every = functools.partial(_outcome, ref.predict, line, -1, -1.0)
        assert _prediction_mismatch(want, got, every) is None, (line, want, got)
    _check_batch(entry.model, lines, k, threshold)


@pytest.mark.parametrize("line", ["a\nb", "\n", "the cat\n", "\ud800 x", "x \udfff"], ids=repr)
@pytest.mark.parametrize("k", [1, 0, -1])
@pytest.mark.parametrize("name", ["softmax-dense", "cbow-ns"])
def test_oracle_parity_bad_lines(
    oracle: ModuleType, zoo: Zoo, name: str, line: str, k: int
) -> None:
    # "\n": fastText's ValueError, before any other check. A lone surrogate
    # can't be converted to std::string: pybind11 raises TypeError.
    entry = zoo[name]
    want = _outcome(_oracle_model(str(entry.path)).predict, line, k)
    got = _outcome(entry.model.predict, line, k)
    assert got[0] == want[0]
    if want[0] != "TypeError":  # (pybind11's message lists the overloads)
        assert got == want


@pytest.mark.parametrize("x", ["a\ud800\nb", ["ok", "a\nb", "\ud800"]], ids=repr)
def test_oracle_parity_newline_before_surrogate(oracle: ModuleType, zoo: Zoo, x: Any) -> None:
    entry = zoo["softmax-dense"]
    assert _outcome(_oracle_model(str(entry.path)).predict, x) == ("ValueError", NEWLINE_MSG)
    assert _outcome(entry.model.predict, x) == ("ValueError", NEWLINE_MSG)


def test_oracle_parity_k_types(oracle: ModuleType, zoo: Zoo) -> None:
    entry = zoo["softmax-dense"]
    ref = _oracle_model(str(entry.path))
    for k in (True, False, 10**6):
        want = _outcome(ref.predict, "the cat sat", k)
        got = _outcome(entry.model.predict, "the cat sat", k)
        assert _prediction_mismatch(want, got) is None, (k, want, got)
    for k_bad in (1.0, "1", None):
        assert _outcome(ref.predict, "the", k_bad)[0] == "TypeError"
        assert _outcome(entry.model.predict, "the", k_bad)[0] == "TypeError"


_NOT_UTF8 = Spec(
    words=(b"</s>", b"ok", b"bad\xff"),
    labels=(b"__label__ok", b"__label__\xff\xfe", b"__label__ma\xc3"),
    fill_out=0.0,  # ties: every label is predicted for k=-1
    seed=7,
)


@pytest.mark.parametrize(
    "handler", ["strict", "replace", "ignore", "backslashreplace", "surrogateescape"]
)
def test_oracle_parity_labels_that_are_not_utf8(
    oracle: ModuleType, tmp_path: Path, handler: str
) -> None:
    path = _write(tmp_path, build(_NOT_UTF8).data)
    model, ref = ft.load_model(path), oracle.load_model(str(path))
    for line in ("ok", "", "bad\xff"):
        for k in (1, 2, -1):
            want = _outcome(ref.predict, line, k, 0.0, handler)
            got = _outcome(model.predict, line, k, 0.0, handler)
            assert _prediction_mismatch(want, got) is None, (line, k, want, got)
        batch = _outcome(model.predict, [line, line], -1, 0.0, handler)
        if handler == "strict":
            assert batch[0] == "UnicodeDecodeError"
        else:
            assert batch[1][0] == [list(model.predict(line, -1, 0.0, handler)[0])] * 2


def test_oracle_parity_load_errors(oracle: ModuleType, tmp_path: Path) -> None:
    good = build(Spec()).data
    for data in (
        b"\0" * 64,
        struct.pack("<i", MAGIC + 1) + good[4:],
        good[:4] + struct.pack("<i", 13) + good[8:],
        b"not a model at all, just some text\n" * 10,
    ):
        path = _write(tmp_path, data)
        want = _outcome(oracle.load_model, str(path))
        assert want == ("ValueError", f"{path} has wrong file format!")
        with pytest.raises(ft.ModelError) as info:
            ft.load_model(path)
        assert str(info.value).startswith(want[1])
    # A pruned dictionary with a dense input matrix.
    path = _write(tmp_path, build(Spec(prune=((0, 0),))).data)
    want = _outcome(oracle.load_model, str(path))
    assert want[0] == "ValueError"
    assert want[1].startswith("Invalid model file.")
    with pytest.raises(ft.ModelError, match=r"(?s)Invalid model file\..*issue #332"):
        ft.load_model(path)


# ---- 2. return types and dtypes ----------------------------------------------------------


def test_predict_str_types(zoo: Zoo) -> None:
    entry = zoo["softmax-dense"]
    m = entry.model
    out = m.predict(text="the cat", k=2, threshold=0.0, on_unicode_error="strict")
    assert type(out) is tuple
    labels, probs = out
    assert type(labels) is tuple
    assert all(type(x) is str for x in labels)
    assert type(probs) is np.ndarray
    assert probs.dtype == np.float64
    assert probs.shape == (2,)
    assert probs[0] >= probs[1]
    # No predictions: an empty tuple and an empty float64 array.
    labels, probs = m.predict("the cat", threshold=2.0)
    assert labels == ()
    assert probs.dtype == np.float64
    assert probs.shape == (0,)
    # k=-1: every label above the threshold, best first.
    labels, probs = m.predict("the cat", k=-1)
    assert sorted(labels) == sorted(x.decode() for x in entry.synth.labels)
    assert list(probs) == sorted(probs, reverse=True)


def test_predict_batch_types(zoo: Zoo) -> None:
    m = zoo["hs-dense"].model
    out = m.predict(["the cat", "sat", ""], k=2)
    assert type(out) is tuple
    labels, probs = out
    assert type(labels) is list
    assert type(probs) is list
    assert all(type(x) is list for x in labels)
    assert all(type(x) is str for doc in labels for x in doc)
    assert all(type(p) is np.ndarray for p in probs)
    assert all(p.dtype == np.float32 for p in probs)
    assert [p.shape for p in probs] == [(2,), (2,), (2,)]
    labels, probs = m.predict(("the cat", "sat"), threshold=2.0)
    assert labels == [[], []]
    assert all(p.dtype == np.float32 for p in probs)
    assert all(p.shape == (0,) for p in probs)
    assert m.predict([]) == ([], [])
    assert m.predict(()) == ([], [])


def test_lines_without_tokens_have_no_predictions(zoo: Zoo) -> None:
    # Without "</s>" in the vocabulary an empty line has no input rows: no
    # predictions, and no k or supervision checks (fastText's order).
    m = zoo["softmax-no-eos"].model
    for line in ("", " \t", "__label__a", "</s>"):
        labels, probs = m.predict(line, k=0)
        assert labels == ()
        assert probs.dtype == np.float64
        assert probs.shape == (0,)
    labels_b, probs_b = m.predict(["", "__label__b"], k=-2)
    assert labels_b == [[], []]
    assert all(p.dtype == np.float32 for p in probs_b)
    assert zoo["cbow-no-eos"].model.predict("", k=1)[0] == ()
    # With "</s>" (the usual case) an empty line still predicts.
    assert len(zoo["softmax-dense"].model.predict("", k=1)[0]) == 1


def test_vector_types(zoo: Zoo) -> None:
    entry = zoo["softmax-dense"]
    m, dim = entry.model, entry.synth.spec.dim
    for v in (m.get_word_vector("the"), m["zebra"], m.get_sentence_vector("the cat")):
        assert type(v) is np.ndarray
        assert v.dtype == np.float32
        assert v.shape == (dim,)
    assert m.get_input_vector(0).dtype == np.float32
    batch = m.get_sentence_vector(["the", "cat"])
    assert batch.dtype == np.float32
    assert batch.shape == (2, dim)
    assert m.get_sentence_vector([]).shape == (0, dim)
    strings, ids = m.get_subwords("the")
    assert type(strings) is list
    assert ids.dtype == np.int64
    for value in (m.get_word_id("the"), m.get_label_id("x"), m.get_subword_id("<th")):
        assert type(value) is int
    assert type(m.get_dimension()) is int
    assert type(m.is_quantized()) is bool
    assert type(m.get_line("a b")) is tuple


# ---- 3. container kinds ----------------------------------------------------------------


def _check_kind(model: ft.Model, values: Sequence[str | None], x: Any, k: int = 2) -> None:
    """``x`` (an Arrow column of ``values``) gives each str's result; nulls stay None."""
    labels, probs = model.predict(x, k)
    assert type(labels) is list
    assert len(labels) == len(probs) == len(values)
    vectors = model.get_sentence_vector(x)
    assert vectors.dtype == np.float32
    assert vectors.shape == (len(values), model.get_dimension())
    words, known = model.get_line(x)
    for i, v in enumerate(values):
        if v is None:
            assert labels[i] is None
            assert probs[i] is None
            assert np.isnan(vectors[i]).all()
            assert words[i] is None
            assert known[i] is None
            continue
        want_labels, want_probs = model.predict(v, k)
        assert labels[i] == list(want_labels)
        assert probs[i].dtype == np.float32
        np.testing.assert_array_equal(probs[i].astype(np.float64), want_probs)
        np.testing.assert_array_equal(vectors[i], model.get_sentence_vector(v))
        assert (words[i], known[i]) == model.get_line(v)


@given(line_lists, st.sampled_from(KS))
def test_list_and_tuple_match_str(zoo: Zoo, lines: list[str], k: int) -> None:
    m = zoo["ova-qnorm"].model
    _check_batch(m, lines, k, 0.0)
    batches: list[Batch] = [lines, tuple(lines)]
    for batch in batches:
        vectors = m.get_sentence_vector(batch)
        assert vectors.shape == (len(lines), m.get_dimension())
        for row, line in zip(vectors, lines, strict=True):
            np.testing.assert_array_equal(row, m.get_sentence_vector(line))
        words, labels = m.get_line(batch)
        assert words == [m.get_line(s)[0] for s in lines]
        assert labels == [m.get_line(s)[1] for s in lines]


_ARROW_TYPES = ["string", "large_string", "string_view"]


@given(optional_lines, st.sampled_from(_ARROW_TYPES), st.integers(0, 5))
def test_pyarrow_arrays(
    pa: ModuleType, zoo: Zoo, values: list[str | None], type_name: str, offset: int
) -> None:
    arr: Any = pa.array(values, type=getattr(pa, type_name)()).slice(offset)
    _check_kind(zoo["softmax-qout"].model, values[offset:], arr)


@given(st.lists(optional_lines, max_size=4), st.sampled_from(_ARROW_TYPES))
def test_pyarrow_chunked_arrays(
    pa: ModuleType, zoo: Zoo, chunks: list[list[str | None]], type_name: str
) -> None:
    arr: Any = pa.chunked_array(chunks, type=getattr(pa, type_name)())
    _check_kind(zoo["hs-pruned"].model, [v for c in chunks for v in c], arr)


@given(optional_lines, st.booleans(), st.integers(1, 4))
def test_column_without_pyarrow(
    zoo: Zoo, values: list[str | None], large: bool, chunks: int
) -> None:
    col = _testing.column(values, large=large, chunks=chunks)
    _check_kind(zoo["softmax-unicode"].model, values, col)


def test_polars_series(pl: ModuleType, zoo: Zoo) -> None:
    values = ["the cat", None, "café sat", ""]
    _check_kind(zoo["softmax-dense"].model, values, pl.Series("text", values))


def test_pyroid_column_input(zoo: Zoo) -> None:
    col = pyroid.text.normalize_whitespace(_testing.column(["  the   cat ", None, "sat"]))
    _check_kind(zoo["softmax-dense"].model, ["the cat", None, "sat"], col)


# ---- 4. threads ------------------------------------------------------------------------

THREAD_DOCS = _corpus(random.Random(7), 3_000)


@dataclass
class Baseline:
    """One model's single-threaded results on THREAD_DOCS."""

    model: ft.Model
    labels: list[list[str]]
    probs: list[F32]
    vectors: F32


@pytest.fixture(scope="session")
def baseline(zoo: Zoo) -> Baseline:
    m = zoo["hs-qnorm-qout"].model
    labels, probs = m.predict(THREAD_DOCS, 3, threads=1)
    return Baseline(m, labels, probs, m.get_sentence_vector(THREAD_DOCS, threads=1))


def _same(a: Sequence[F32], b: Sequence[F32]) -> bool:
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("threads", [None, 1, 2, 8])
@pytest.mark.parametrize("kind", ["list", "tuple", "column", "column-3-chunks"])
def test_threads_equivalence(baseline: Baseline, kind: str, threads: int | None) -> None:
    x: Any = {
        "list": THREAD_DOCS,
        "tuple": tuple(THREAD_DOCS),
        "column": _testing.column(THREAD_DOCS),
        "column-3-chunks": _testing.column(THREAD_DOCS, chunks=3),
    }[kind]
    labels, probs = baseline.model.predict(x, 3, threads=threads)
    assert labels == baseline.labels
    assert _same(probs, baseline.probs)
    vectors = baseline.model.get_sentence_vector(x, threads=threads)
    np.testing.assert_array_equal(vectors, baseline.vectors)


def test_batches_match_str_calls(baseline: Baseline) -> None:
    m = baseline.model
    for i in range(0, len(THREAD_DOCS), 97):
        labels, probs = m.predict(THREAD_DOCS[i], 3)
        assert list(labels) == baseline.labels[i]
        np.testing.assert_array_equal(probs, baseline.probs[i].astype(np.float64))
        np.testing.assert_array_equal(m.get_sentence_vector(THREAD_DOCS[i]), baseline.vectors[i])


@pytest.mark.timeout(60)
def test_large_batches(zoo: Zoo) -> None:
    m = zoo["softmax-many-labels"].model
    docs = THREAD_DOCS * 4  # 12,000 documents
    labels, probs = m.predict(docs, k=5)
    assert len(labels) == len(probs) == len(docs)
    n = len(THREAD_DOCS)
    assert labels[:n] == labels[n : 2 * n]
    col_labels, col_probs = m.predict(_testing.column(docs, chunks=4), k=5)
    assert col_labels == labels
    assert _same(col_probs, probs)
    assert m.get_sentence_vector(docs).shape == (len(docs), m.get_dimension())


@pytest.mark.thread_unsafe(reason="starts its own threads")
@pytest.mark.timeout(120)
def test_one_model_shared_by_many_python_threads(baseline: Baseline) -> None:
    m = baseline.model
    n_threads = 8
    barrier = threading.Barrier(n_threads)

    def work(i: int) -> bool:
        barrier.wait()
        ok = True
        for it in range(4):
            threads = [None, 1, 2][(i + it) % 3]
            labels, probs = m.predict(THREAD_DOCS, 3, threads=threads)
            ok = ok and labels == baseline.labels and _same(probs, baseline.probs)
            vectors = m.get_sentence_vector(THREAD_DOCS, threads=threads)
            ok = ok and np.array_equal(vectors, baseline.vectors)
            for j in range(i, len(THREAD_DOCS), 211):
                ok = ok and list(m.predict(THREAD_DOCS[j], 3)[0]) == baseline.labels[j]
        return ok

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        assert all(ex.map(work, range(n_threads)))


@pytest.mark.skipif(not FREE_THREADED, reason="needs a free-threaded CPython build")
@pytest.mark.thread_unsafe(reason="runs a subprocess")
def test_gil_stays_disabled_after_use(run_python: RunPython, zoo: Zoo) -> None:
    path = zoo["softmax-qout"].path
    proc = run_python(
        f"""
        import sys, threading
        import pyroid.fasttext as ft
        from pyroid._pyroid import _testing
        m = ft.load_model({str(path)!r})
        docs = ["the cat sat", "na\\u00efve on the mat", ""] * 3000
        labels, probs = m.predict(docs, k=2)
        assert len(labels) == len(docs)
        m.predict(_testing.column(docs, chunks=2), k=-1)
        m.get_sentence_vector(docs)
        m.get_words(include_freq=True), m.get_labels(), m.get_line(docs[:10])
        m.get_word_vector("zebra"), m.get_subwords("cat"), m.get_input_vector(0)
        ts = [threading.Thread(target=m.predict, args=(docs,)) for _ in range(4)]
        for t in ts: t.start()
        for t in ts: t.join()
        print(sys._is_gil_enabled())
        """,
        args=["-W", "error::RuntimeWarning"],
        env={"PYTHON_GIL": None},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", proc.stderr


# ---- 5. errors -----------------------------------------------------------------------------


def test_model_error_type() -> None:
    assert issubclass(ft.ModelError, pyroid.PyroidError)
    assert issubclass(ft.ModelError, ValueError)
    assert not issubclass(ft.ModelError, pyroid.InputError)
    assert ft.ModelError.__module__ == "pyroid.fasttext"


@pytest.mark.skipif(
    not sysconfig.get_platform().startswith("linux"), reason="RLIMIT_AS and /proc are Linux's"
)
@pytest.mark.thread_unsafe(reason="runs a subprocess")
def test_failed_allocation_is_a_memory_error(run_python: RunPython, tmp_path: Path) -> None:
    # As fastText's std::bad_alloc: fast-langdetect's model="auto" falls back to the small model
    # on MemoryError. A valid model with a 64 MiB input matrix, loaded under a tighter RLIMIT_AS.
    path = tmp_path / "big.bin"
    path.write_bytes(build(Spec(dim=16, bucket=1 << 20, fill_in=0.0, fill_out=0.0)).data)
    assert ft.load_model(path).get_dimension() == 16
    proc = run_python(
        f"""
        import resource
        import pyroid.fasttext as ft
        with open("/proc/self/status") as f:
            vm = next(int(x.split()[1]) for x in f if x.startswith("VmSize:")) * 1024
        hard = resource.getrlimit(resource.RLIMIT_AS)[1]
        resource.setrlimit(resource.RLIMIT_AS, (vm + (24 << 20), hard))
        try:
            ft.load_model({str(path)!r})
        except MemoryError as e:
            print("MemoryError:", e)
        else:
            print("loaded")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("MemoryError:"), proc.stdout
    assert f"{path}: input matrix: cannot allocate 67" in proc.stdout, proc.stdout


def test_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.bin"
    for path in (missing, str(missing)):
        with pytest.raises(FileNotFoundError) as info:
            ft.load_model(path)
        assert os.fspath(info.value.filename) == str(missing)


@pytest.mark.parametrize("empty", [False, True], ids=["full", "empty"])
def test_directory_is_an_os_error(tmp_path: Path, empty: bool) -> None:
    if not empty:
        (tmp_path / ("x" * 40)).write_bytes(b"x")  # st_size >= 4 everywhere
    with pytest.raises(OSError) as info:  # noqa: PT011 (IsADirectoryError, or PermissionError on Windows)
        ft.load_model(tmp_path)
    assert isinstance(info.value, IsADirectoryError | PermissionError)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x00",
        b"PK\x03\x04 not a model",
        struct.pack("<ii", MAGIC + 1, 12),
        struct.pack("<ii", MAGIC, 13),
        struct.pack("<ii", MAGIC, 2**31 - 1),
    ],
    ids=repr,
)
def test_not_a_model(tmp_path: Path, data: bytes) -> None:
    path = _write(tmp_path, data + bytes(200))
    with pytest.raises(ft.ModelError, match="has wrong file format!") as info:
        ft.load_model(path)
    assert str(path) in str(info.value)


def test_empty_file(tmp_path: Path) -> None:
    with pytest.raises(ft.ModelError):
        ft.load_model(_write(tmp_path, b""))


def test_version_13_header(tmp_path: Path) -> None:
    path = _write(tmp_path, build(Spec(version=13)).data)
    with pytest.raises(ft.ModelError, match=r"has wrong file format!.*version 13"):
        ft.load_model(path)


@pytest.mark.parametrize("quant", [False, True], ids=["dense", "quant"])
def test_truncated_at_every_offset(tmp_path: Path, quant: bool) -> None:
    spec = Spec(
        words=("</s>", "ab", "é"),
        labels=LABELS[:2],
        dim=3,
        bucket=5,
        quant=quant,
        qnorm=quant,
        qout=quant,
    )
    data = build(spec).data
    ft.load_model(_write(tmp_path, data))  # the whole file loads
    path = _write(tmp_path, b"")
    offsets = sorted({*range(min(400, len(data))), *range(400, len(data), 7), len(data) - 1})
    for n in offsets:
        path.write_bytes(data[:n])
        with pytest.raises(ft.ModelError) as info:
            ft.load_model(path)
        msg = str(info.value)
        assert any(s in msg for s in ("truncated", "wrong file format", "not a valid")), (n, msg)


@settings(max_examples=max(50, PROFILE.max_examples * 2))
@given(
    name=st.sampled_from(["softmax-dense", "hs-qnorm-qout", "ova-pruned", "cbow-ns"]),
    edits=st.lists(st.tuples(st.floats(0, 1, exclude_max=True), st.integers(0, 255)), max_size=4),
    cut=st.integers(0, 8),
)
def test_corrupted_files_load_or_raise_model_error(
    tmp_path_factory: pytest.TempPathFactory,
    zoo: Zoo,
    name: str,
    edits: list[tuple[float, int]],
    cut: int,
) -> None:
    # Most edits land in the header and the dictionary, where the sizes are.
    data = bytearray(zoo[name].synth.data)
    for where, value in edits:
        data[int(where**3 * len(data))] = value
    path = _write(tmp_path_factory.getbasetemp(), bytes(data[: len(data) - cut]))
    try:
        model = ft.load_model(path)
    except ft.ModelError:
        return
    finally:
        path.unlink()
    # A file that loads is usable: predictions and vectors never crash.
    for line in ("the cat", "", "zebra 日本"):
        try:
            model.predict(line, -1, 0.0, "replace")  # labels may be edited too
        except ValueError as e:  # an edited model type
            assert str(e) == "Model needs to be supervised for prediction!"  # noqa: PT017
        except RuntimeError as e:  # edited weights
            assert str(e) == "Encountered NaN."  # noqa: PT017
        model.get_sentence_vector(line)
        model.get_line(line)


@pytest.mark.parametrize("bad", [None, 1, 1.5, b"the cat", bytearray(b"x"), {"a": 1}, iter(["a"])])
def test_bad_text_types(zoo: Zoo, bad: Any) -> None:
    m = zoo["softmax-dense"].model
    for call in (m.predict, m.get_sentence_vector, m.get_line):
        with pytest.raises(pyroid.InputError):
            call(bad)
    for batch in (["ok", bad], ("ok", "fine", bad)):
        with pytest.raises(pyroid.InputError, match="index"):
            m.predict(batch)


def test_bad_argument_types(zoo: Zoo) -> None:
    m: Any = zoo["softmax-dense"].model
    bad: Any
    for bad in (1.5, "2", None):
        with pytest.raises(TypeError):
            m.predict("the", k=bad)
    for bad in ("0.5", None):
        with pytest.raises(TypeError):
            m.predict("the", threshold=bad)
    with pytest.raises(TypeError):
        m.predict("the", on_unicode_error=1)
    for word_fn in (m.get_word_vector, m.get_subwords, m.get_word_id, m.get_label_id):
        for bad in (1, b"the", None):
            with pytest.raises(TypeError):
                word_fn(bad)
    for bad in ("1", 1.0, None):
        with pytest.raises(TypeError):
            m.get_input_vector(bad)
    load: Any = ft.load_model
    with pytest.raises(TypeError):
        load(1)
    model_type: Any = ft.Model
    with pytest.raises(TypeError):
        model_type()


@pytest.mark.parametrize("threads", [0, -1, -(2**70)])
def test_threads_must_be_positive(zoo: Zoo, threads: int) -> None:
    m = zoo["softmax-dense"].model
    for x in ("the", ["the"], ["the"] * 100):
        with pytest.raises(ValueError, match="threads must be None or a positive int"):
            m.predict(x, threads=threads)
        with pytest.raises(ValueError, match="threads must be None or a positive int"):
            m.get_sentence_vector(x, threads=threads)


@pytest.mark.parametrize("threads", ["2", 2.0, True])
def test_threads_must_be_an_int(zoo: Zoo, threads: Any) -> None:
    with pytest.raises(pyroid.InputError, match="threads"):
        zoo["softmax-dense"].model.predict(["the"], threads=threads)


@pytest.mark.parametrize("index", [0, 3, 70, 199])
def test_newline_in_a_batch_names_the_index(zoo: Zoo, index: int) -> None:
    m = zoo["softmax-dense"].model
    docs = ["the cat"] * 200
    docs[index] = "the\ncat"
    for x in (docs, tuple(docs), _testing.column(docs, chunks=2)):
        with pytest.raises(ValueError, match=re.escape(NEWLINE_MSG)) as info:
            m.predict(x)
        assert f"index {index}" in _notes(info.value)
        with pytest.raises(ValueError, match=re.escape(NEWLINE_MSG)) as info:
            m.get_sentence_vector(x)
        assert f"index {index}" in _notes(info.value)
        with pytest.raises(ValueError, match=re.escape(GET_LINE_MSG)):
            m.get_line(x)


def test_newline_errors_come_first(zoo: Zoo) -> None:
    # fastText checks every line for "\n" before predicting any.
    m = zoo["softmax-dense"].model
    with pytest.raises(ValueError, match=re.escape(NEWLINE_MSG)) as info:
        m.predict(["the", "cat", "a\nb"], k=0)
    assert "index 2" in _notes(info.value)
    with pytest.raises(ValueError, match="k needs to be 1 or higher!") as info:
        m.predict(["the", "cat", "sat"], k=0)
    assert "index 0" in _notes(info.value)


def test_str_errors_have_no_index_note(zoo: Zoo) -> None:
    m = zoo["softmax-dense"].model
    for line, k, msg in (("a\nb", 1, NEWLINE_MSG), ("the", 0, "k needs"), ("the", -2, "k needs")):
        with pytest.raises(ValueError, match=re.escape(msg)) as info:
            m.predict(line, k)
        assert _notes(info.value) == ""
    with pytest.raises(ValueError, match=re.escape(NEWLINE_MSG)):
        m.get_sentence_vector("a\nb")
    with pytest.raises(ValueError, match=re.escape(GET_LINE_MSG)):
        m.get_line("a\nb")


@pytest.mark.parametrize("name", UNSUPERVISED)
def test_unsupervised_predict(zoo: Zoo, name: str) -> None:
    m = zoo[name].model
    msg = "Model needs to be supervised for prediction!"
    with pytest.raises(ValueError, match=msg):
        m.predict("the cat")
    with pytest.raises(ValueError, match=msg) as info:
        m.predict(["the", "cat"])
    assert "index 0" in _notes(info.value)


@pytest.mark.parametrize("loss", list(LOSSES))
def test_nan_weights(tmp_path: Path, loss: str) -> None:
    m = ft.load_model(_write(tmp_path, build(Spec(loss=loss, fill_out=math.nan)).data))
    with pytest.raises(RuntimeError, match=r"^Encountered NaN\.$"):
        m.predict("the cat")
    with pytest.raises(RuntimeError, match=r"^Encountered NaN\.") as info:
        m.predict(["the", "cat"] * 50)
    assert "index 0" in _notes(info.value)
    assert np.isnan(m.get_output_matrix()).all()
    assert not np.isnan(m.get_sentence_vector("the cat")).any()


@pytest.mark.parametrize("cp", [0xD800, 0xDBFF, 0xDC00, 0xDFFF])
def test_lone_surrogates_raise_type_error(zoo: Zoo, cp: int) -> None:
    m = zoo["softmax-dense"].model
    bad = f"the {chr(cp)} cat"
    calls: list[Callable[[str], object]] = [
        m.predict,
        m.get_sentence_vector,
        m.get_line,
        m.get_word_vector,
        m.get_subwords,
        m.get_word_id,
        m.get_label_id,
        m.get_subword_id,
        m.__getitem__,
    ]
    for call in calls:
        with pytest.raises(TypeError, match="surrogate"):
            call(bad)
    assert bad not in m
    with pytest.raises(TypeError, match="surrogate") as info:
        m.predict(["ok", "fine", bad])
    assert isinstance(info.value.__cause__, UnicodeEncodeError)
    assert "index 2" in _notes(info.value.__cause__)


def test_quantized_matrices_are_not_available(zoo: Zoo) -> None:
    for name in ("softmax-quant", "hs-qout", "ova-pruned"):
        m = zoo[name].model
        for call in (m.get_input_matrix, m.get_output_matrix):
            with pytest.raises(ValueError, match=r"^Can't get quantized Matrix$"):
                call()


def test_input_vector_out_of_range(zoo: Zoo) -> None:
    for name in ("softmax-dense", "softmax-pruned"):
        entry = zoo[name]
        rows = len(entry.synth.input_rows)
        for ind in (-1, rows, rows + 1, 2**40, -(2**63)):
            with pytest.raises(IndexError, match="out of range"):
                entry.model.get_input_vector(ind)


def test_subword_id_without_buckets(zoo: Zoo) -> None:
    with pytest.raises(ValueError, match="no subword buckets"):
        zoo["softmax-no-subwords"].model.get_subword_id("<th")


@pytest.mark.parametrize("name", ["softmax-pruned-empty", "softmax-no-subwords"])
def test_word_vector_of_a_label_past_the_matrix(zoo: Zoo, name: str) -> None:
    # A label's word vector uses row nwords + label id: past the end of these
    # input matrices (fastText would read out of bounds).
    entry = zoo[name]
    label = entry.synth.labels[-1].decode()
    assert entry.ref.word_vector(label) is None
    with pytest.raises(ValueError, match="outside the input matrix"):
        entry.model.get_word_vector(label)


# ---- 6. the rest of the API (against the reference model) -----------------------------------


@pytest.mark.parametrize("name", SPECS)
def test_header_and_vocabulary(zoo: Zoo, name: str) -> None:
    entry = zoo[name]
    m, synth = entry.model, entry.synth
    assert m.get_dimension() == synth.spec.dim
    assert m.is_quantized() is synth.spec.quant
    words = [w.decode() for w in synth.words]
    labels = [x.decode() for x in synth.labels]
    assert m.get_words() == words
    got_words, word_counts = m.get_words(include_freq=True)
    assert got_words == words
    assert word_counts.dtype == np.int64
    assert word_counts.tolist() == synth.word_counts
    got_labels, label_counts = m.get_labels(include_freq=True)
    assert label_counts.dtype == np.int64
    if synth.spec.model == "sup":
        assert m.get_labels() == labels
        assert got_labels == labels
        assert label_counts.tolist() == synth.label_counts
    else:  # unsupervised models list their words
        assert m.get_labels() == words
        assert got_labels == words
        assert label_counts.tolist() == synth.word_counts
    assert m.words == m.get_words()
    assert m.labels == m.get_labels()
    assert m.words is m.words
    assert m.labels is m.labels
    r = repr(m)
    # The repr quotes and escapes the path, so a Windows path shows doubled backslashes.
    quoted_path = '"' + str(entry.path).replace("\\", "\\\\") + '"'
    for part in (
        "pyroid.fasttext.Model",
        quoted_path,
        f"dim {synth.spec.dim}",
        f"{len(words)} words",
        f"{len(labels)} labels",
        "quantized" if synth.spec.quant else "dense",
    ):
        assert part in r, (part, r)


def test_words_that_are_not_utf8(tmp_path: Path) -> None:
    m = ft.load_model(_write(tmp_path, build(_NOT_UTF8).data))
    for call in (m.get_words, m.get_labels):
        with pytest.raises(UnicodeDecodeError):
            call()
    with pytest.raises(UnicodeDecodeError):
        _ = m.words
    with pytest.raises(UnicodeDecodeError):
        _ = m.labels
    assert m.get_words(on_unicode_error="replace") == ["</s>", "ok", "bad\ufffd"]
    assert m.get_words(False, "surrogateescape")[2] == "bad\udcff"
    assert m.get_labels(on_unicode_error="ignore") == ["__label__ok", "__label__", "__label__ma"]
    labels, counts = m.get_labels(True, "backslashreplace")
    assert labels == ["__label__ok", "__label__\\xff\\xfe", "__label__ma\\xc3"]
    assert counts.tolist() == [500, 493, 486]


@pytest.mark.parametrize("name", SPECS)
def test_matrices_and_input_vectors(zoo: Zoo, name: str) -> None:
    entry = zoo[name]
    m, synth = entry.model, entry.synth
    for i in range(len(synth.input_rows)):
        np.testing.assert_array_equal(m.get_input_vector(i), synth.input_rows[i])
    if synth.spec.quant:
        return
    pairs = [(m.get_input_matrix(), synth.input_rows), (m.get_output_matrix(), synth.output_rows)]
    for got, want in pairs:
        assert got.dtype == np.float32
        assert got.shape == want.shape
        assert got.flags.c_contiguous
        np.testing.assert_array_equal(got, want)
        got[:] = 42.0  # a copy: the model is untouched
    np.testing.assert_array_equal(m.get_input_matrix(), synth.input_rows)
    np.testing.assert_array_equal(m.get_output_matrix(), synth.output_rows)


word_st = st.one_of(
    st.sampled_from(TOKENS),
    st.text(st.characters(exclude_categories=["Cs"]), max_size=10),
    st.sampled_from(LOOKALIKES),
)


def _check_vector(entry: Entry, got: F32, want: F32) -> None:
    assert got.dtype == np.float32
    assert got.shape == want.shape
    if entry.synth.spec.quant or entry.synth.spec.model != "sup":
        # Quantized rows accumulate with fused multiply-adds on some platforms,
        # and fastText's norm loop may be contracted too: not bit-exact in numpy.
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    else:
        np.testing.assert_array_equal(got, want)


REF_SPECS = [
    "softmax-dense",
    "hs-quant",
    "ova-qnorm-qout",
    "softmax-pruned",
    "softmax-pruned-empty",
    "softmax-v11",
    "softmax-no-subwords",
    "softmax-unicode",
    "hs-unicode-quant",
    "softmax-no-eos",
    "softmax-minn3-big-bucket",
    "ova-dim1",
    "cbow-ns",
    "cbow-no-eos",
]


@pytest.mark.parametrize("name", REF_SPECS)
@settings(max_examples=max(10, PROFILE.max_examples // 2))
@given(word=word_st, line=line_st)
def test_words_and_lines_match_the_reference(zoo: Zoo, name: str, word: str, line: str) -> None:
    entry = zoo[name]
    m, ref = entry.model, entry.ref
    # Words: vectors, subwords and ids.
    want = ref.word_vector(word)
    if want is None:
        with pytest.raises(ValueError, match="outside the input matrix"):
            m.get_word_vector(word)
    else:
        _check_vector(entry, m.get_word_vector(word), want)
        _check_vector(entry, m[word], want)
    strings, ids = m.get_subwords(word)
    want_strings, want_ids = ref.subwords(word)
    assert strings == want_strings
    assert ids.tolist() == want_ids
    assert m.get_word_id(word) == ref.word_id(word)
    assert m.get_label_id(word) == ref.label_id(word)
    assert (word in m) == (0 <= ref.word_id(word) < ref.nwords)
    if entry.synth.spec.bucket > 0:
        assert m.get_subword_id(word) == ref.nwords + fnv1a(word.encode()) % ref.bucket
    # Lines: tokens, known labels and the sentence vector.
    words, labels = m.get_line(line)
    assert type(words) is list
    assert type(labels) is list
    assert (words, labels) == ref.get_line(line)
    _check_vector(entry, m.get_sentence_vector(line), ref.sentence_vector(line))


@pytest.mark.parametrize("name", EXACT)
def test_sentence_vectors_are_the_hidden_layer(zoo: Zoo, name: str) -> None:
    # Supervised dense models: bit-exact on the whole corpus.
    entry = zoo[name]
    got = entry.model.get_sentence_vector(CORPUS)
    for row, line in zip(got, CORPUS, strict=True):
        np.testing.assert_array_equal(row, entry.ref.sentence_vector(line))


def test_known_values(zoo: Zoo) -> None:
    entry = zoo["softmax-dense"]
    m, synth, nwords = entry.model, entry.synth, len(WORDS)
    assert m.get_word_id("the") == 1
    assert m.get_word_id("__label__b") == nwords + 1
    assert m.get_word_id("zebra") == -1
    assert m.get_label_id("__label__c") == 2
    assert m.get_label_id("__label__zz") == -1
    assert m.get_label_id("cat") == 2 - nwords  # fastText's arithmetic for a word
    assert m.get_label_id("\U0001f642ok") == -1  # the last word: -1 like an unknown one
    assert m.get_line("hello __label__a world __label__zz") == (
        ["hello", "world", "</s>"],
        ["__label__a"],
    )
    assert m.get_line("a </s> b") == (["a", "</s>"], [])
    assert m.get_line("") == (["</s>"], [])
    strings, ids = m.get_subwords("</s>")
    assert (strings, ids.tolist()) == (["</s>"], [0])
    strings, ids = m.get_subwords("ab")
    assert strings == ["<a", "<ab", "<ab>", "ab", "ab>", "b>"]
    assert ids.tolist() == [nwords + fnv1a(s.encode()) % 97 for s in strings]
    np.testing.assert_array_equal(m.get_word_vector("</s>"), synth.input_rows[0])
    empty = _mean(synth.input_rows, [nwords + fnv1a(b"<>") % 97], synth.spec.dim)
    np.testing.assert_array_equal(m.get_word_vector(""), empty)
    assert "the" in m
    assert "zebra" not in m
    assert "__label__a" not in m
    assert 1 not in m


def test_pruned_subwords_list_more_strings_than_ids(zoo: Zoo) -> None:
    entry = zoo["softmax-pruned"]
    strings, ids = entry.model.get_subwords("zebra")
    assert len(strings) > len(ids)
    assert (strings, ids.tolist()) == entry.ref.subwords("zebra")
    strings, ids = zoo["softmax-pruned-empty"].model.get_subwords("the")
    assert len(strings) > 1
    assert ids.tolist() == [1]  # every n-gram is pruned away


# ---- 7. numpy is required ----------------------------------------------------------------


@pytest.mark.thread_unsafe(reason="runs a subprocess")
def test_import_error_without_numpy(run_python: RunPython, zoo: Zoo) -> None:
    path = zoo["softmax-dense"].path
    proc = run_python(
        f"""
        import sys
        import warnings
        sys.modules["numpy"] = None  # import numpy raises ModuleNotFoundError
        import pyroid, pyroid.text
        assert pyroid.text.normalize_whitespace(" a  b ") == "a b"
        try:
            import pyroid.fasttext
        except ImportError as e:
            # A ModuleNotFoundError naming numpy, chained to numpy's own.
            assert type(e) is ModuleNotFoundError and e.name == "numpy", repr(e)
            cause = e.__cause__
            assert type(cause) is ModuleNotFoundError and cause.name == "numpy", repr(cause)
            print("module:", e)
        else:
            raise AssertionError("pyroid.fasttext imported without numpy")
        # So pytest.importorskip skips, without its warning for other ImportErrors.
        import pytest
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                pytest.importorskip("pyroid.fasttext")
            except pytest.skip.Exception as e:
                print("skip:", e)
            else:
                raise AssertionError("importorskip did not skip")
        assert not caught, [str(w.message) for w in caught]
        # The native module fails cleanly too (no rust-numpy panic).
        from pyroid._pyroid import fasttext as native
        m = native.load_model({str(path)!r})
        assert m.get_words()[:2] == ["</s>", "the"]
        assert m.get_line("the cat") == (["the", "cat", "</s>"], [])
        for call in (
            lambda: m.predict("the cat"),
            lambda: m.predict(["the cat"]),
            lambda: m.get_sentence_vector("the"),
            lambda: m.get_word_vector("the"),
            lambda: m.get_input_matrix(),
            lambda: m.get_words(include_freq=True),
        ):
            try:
                call()
            except ImportError as e:
                print("native:", e)
            else:
                raise AssertionError("no ImportError")
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert len(lines) == 8, proc.stdout
    assert lines[0] == (
        "module: pyroid.fasttext needs numpy: install it with pip install 'pyroid[fasttext]' "
        "(or pip install numpy)"
    )
    assert lines[1].startswith("skip: "), proc.stdout
    assert all("pyroid[fasttext]" in line for line in lines), proc.stdout


# ---------------------------------------------------------------- tokenize


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("", []),
        ("\n", ["</s>"]),
        (" a\tb\x0bc\x0cd\re\x00f ", ["a", "b", "c", "d", "e", "f"]),
        ("a\nb\n\n", ["a", "</s>", "b", "</s>", "</s>"]),
        ("a </s> b", ["a", "</s>", "b"]),  # a literal </s> does not stop it
        ("x\xa0y\u0085z \u3000w", ["x\xa0y\u0085z", "\u3000w"]),  # only ASCII separators
    ],
)
def test_tokenize(text: str, want: list[str]) -> None:
    assert ft.tokenize(text) == want


def test_tokenize_matches_upstream_where_available() -> None:
    upstream = pytest.importorskip("fasttext")
    if not hasattr(upstream, "tokenize"):
        pytest.skip("this fasttext build has no tokenize")
    for text in ["", "a b\n c\n", "\x00\x0b x é日 </s> y\r\n"]:
        assert ft.tokenize(text) == upstream.tokenize(text)


def test_tokenize_rejects_surrogates_like_fasttext() -> None:
    with pytest.raises(TypeError):
        ft.tokenize("a\ud800")
