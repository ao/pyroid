"""pyroid.integrations.patch under real third-party call sites, unpatched vs patched.

Each library runs unmodified in a fresh interpreter, once with the real package and once after
``patch()``; the outputs must be identical, and the patched run must have used pyroid. They
skip without the library (they run in Docker venvs that have them; see the integrations report):

- open_clip: its tokenizer cleans every text with ``ftfy.fix_text`` (``basic_clean``).
- fast-langdetect: ``fasttext.load_model`` + ``predict`` on its bundled lid.176.ftz; also
  patched with no fastText installed (as on CPython 3.14+ and free-threaded builds).
- datatrove: ``FT176LID`` (``from fasttext.FastText import _FastText``; ``score.item()``).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.thread_unsafe(reason="subprocesses and process-wide import state")

DATA = Path(os.environ.get("PYROID_DATA", "/data"))
FINEWEB = DATA / "fineweb/fineweb-dev-10k.parquet"
LID_BIN = DATA / "models/lid.176.bin"


def installed(*names: str) -> bool:
    return all(importlib.util.find_spec(name) is not None for name in names)


needs_fineweb = pytest.mark.skipif(
    not (FINEWEB.is_file() and installed("pyarrow")), reason=f"needs {FINEWEB} and pyarrow"
)

# The first N FineWeb docs, plus mojibake made from the non-ASCII ones (UTF-8 read as Latin-1
# and as Windows-1252), so that ftfy has real work to do.
PRELUDE = """\
import hashlib, importlib, importlib.abc, json, os, sys, time, warnings
HIDE = set(filter(None, os.environ.get("TEST_HIDE", "").split(",")))
class _Hide(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in HIDE:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None
if HIDE:
    sys.meta_path.insert(0, _Hide())
def patch_quietly(*names):
    import pyroid.integrations as pi
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pi.patch(*names)
def fineweb(n):
    import pyarrow.parquet as pq
    docs = pq.read_table(os.environ["FINEWEB"], columns=["text"]).column("text").to_pylist()[:n]
    odd = [d for d in docs if not d.isascii()]
    return (
        docs
        + [d.encode("utf-8").decode("latin-1") for d in odd]
        + [d.encode("utf-8").decode("cp1252", "ignore") for d in odd]
    )
def digest(obj):
    return hashlib.sha256(json.dumps(obj).encode()).hexdigest()[:16]
"""


def child(
    code: str, *args: str, env: dict[str, str] | None = None, hide: str = ""
) -> dict[str, Any]:
    """Run `code` in a fresh interpreter; return the JSON it printed last."""
    child_env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYROID_PATCH")}
    child_env.update({"FINEWEB": str(FINEWEB), "TEST_HIDE": hide, **(env or {})})
    proc = subprocess.run(
        [sys.executable, "-c", PRELUDE + textwrap.dedent(code), *args],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    result: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
    return result


def first_difference(a: list[Any], b: list[Any]) -> int | None:
    return next((i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y), None)


# ---------------------------------------------------------------------------
# open_clip: ftfy.fix_text in its tokenizer
# ---------------------------------------------------------------------------

OPEN_CLIP = """
mode, n = sys.argv[1], int(sys.argv[2])
texts = fineweb(n)
if mode != "real":
    patch_quietly("ftfy")
import open_clip
import open_clip.tokenizer as tk
tokenizer = open_clip.get_tokenizer("ViT-B-32")  # bundled BPE vocabulary: no network
start = time.perf_counter()
ids = tokenizer(texts).tolist()
seconds = time.perf_counter() - start
clean = [tk.basic_clean(t) for t in texts]  # whole texts (ids stop at 77 tokens)
import pyroid.ftfy
print(json.dumps({
    "n": len(texts),
    "ids": [digest(row) for row in ids],
    "clean": [digest(c) for c in clean],
    "fixed": sum(tk.ftfy.fix_text(t) != t for t in texts),
    "seconds": seconds,
    "shim": getattr(tk.ftfy, "__pyroid__", False),
    "pyroid_fix_text": tk.ftfy.fix_text is pyroid.ftfy.fix_text,
    "real_ftfy_loaded": getattr(tk.ftfy, "__wrapped__", tk.ftfy) is not None,
}))
"""


@needs_fineweb
@pytest.mark.skipif(not installed("open_clip", "torch", "ftfy"), reason="needs open_clip_torch")
def test_open_clip_tokens_are_identical_patched() -> None:
    n = os.environ.get("PYROID_PATCH_TEST_DOCS", "300")
    real = child(OPEN_CLIP, "real", n)
    patched = child(OPEN_CLIP, "patched", n)
    # And with the real ftfy unimportable: then only pyroid can have done the fixing.
    alone = child(OPEN_CLIP, "patched", n, hide="ftfy")
    assert real["shim"] is False
    assert patched["shim"] is alone["shim"] is True
    assert patched["pyroid_fix_text"] is alone["pyroid_fix_text"] is True
    assert alone["real_ftfy_loaded"] is False
    assert real["fixed"] > 0  # ftfy changed some texts: the comparison exercises it
    for run in (patched, alone):
        assert run["n"] == real["n"]
        assert run["fixed"] == real["fixed"]
        assert first_difference(run["clean"], real["clean"]) is None
        assert first_difference(run["ids"], real["ids"]) is None
    sys.stdout.write(
        f"\nopen_clip ViT-B-32: {real['n']} texts ({real['fixed']} changed by ftfy), "
        f"identical ids and basic_clean; tokenize {real['seconds']:.2f}s real ftfy, "
        f"{patched['seconds']:.2f}s patched\n"
    )


# ---------------------------------------------------------------------------
# fast-langdetect: fasttext.load_model + predict
# ---------------------------------------------------------------------------

SHORT = [
    "Hello, how are you today?",
    "Bonjour, comment allez-vous aujourd'hui ?",
    "Guten Morgen, wie geht es dir?",
    "Buenos días, ¿cómo estás?",
    "今日はいい天気ですね。",
    "Сегодня хорошая погода.",
    "مرحبا كيف حالك",
    "",
]

LANGDETECT = """
mode, n = sys.argv[1], int(sys.argv[2])
texts = json.loads(sys.argv[3]) + fineweb(n)
if mode != "real":
    patch_quietly("fasttext")
import fast_langdetect
from fast_langdetect import detect
results = [detect(t, model="lite", k=3) for t in texts]
models = list(fast_langdetect.infer._default_detector._models.values())
import fasttext
print(json.dumps({
    "results": results,
    "model_types": sorted({f"{type(m).__module__}.{type(m).__qualname__}" for m in models}),
    "shim": getattr(fasttext, "__pyroid__", False),
    "real_fasttext": getattr(fasttext, "__wrapped__", fasttext) is not None,
}))
"""


def check_langdetect(run: dict[str, Any]) -> None:
    assert run["shim"] is True
    assert run["model_types"] == ["pyroid.fasttext.Model"]  # pyroid did the predicting
    top = [r[0]["lang"] if r else None for r in run["results"][: len(SHORT)]]
    assert top[:7] == ["en", "fr", "de", "es", "ja", "ru", "ar"], top


@needs_fineweb
@pytest.mark.skipif(not installed("fast_langdetect"), reason="needs fast-langdetect")
def test_fast_langdetect_is_identical_patched() -> None:
    n = os.environ.get("PYROID_PATCH_TEST_DOCS", "300")
    patched = child(LANGDETECT, "patched", n, json.dumps(SHORT))
    check_langdetect(patched)
    alone = child(LANGDETECT, "patched", n, json.dumps(SHORT), hide="fasttext,fasttext_pybind")
    check_langdetect(alone)
    assert alone["real_fasttext"] is False
    assert alone["results"] == patched["results"]
    if not installed("fasttext"):  # e.g. free-threaded CPython: no fastText wheel at all
        return
    real = child(LANGDETECT, "real", n, json.dumps(SHORT))
    assert real["shim"] is False
    assert real["model_types"] == ["fasttext.FastText._FastText"]
    assert len(real["results"]) == len(patched["results"])
    for want, got in zip(real["results"], patched["results"], strict=True):
        assert [r["lang"] for r in got] == [r["lang"] for r in want]
        assert [r["score"] for r in got] == pytest.approx([r["score"] for r in want], abs=1e-6)
    exact = sum(got == want for want, got in zip(real["results"], patched["results"], strict=True))
    sys.stdout.write(
        f"\nfast-langdetect lite: {len(real['results'])} texts, identical languages; "
        f"{exact} bit-identical results\n"
    )


# ---------------------------------------------------------------------------
# datatrove: FT176LID (from fasttext.FastText import _FastText; score.item())
# ---------------------------------------------------------------------------

DATATROVE = """
mode, n, model_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
texts = fineweb(n)
if mode == "reference":  # fastText's C++ (fasttext-predict), called the way datatrove does
    import fasttext
    m = fasttext.load_model(model_path)
    pairs = []
    for t in texts:
        langs, scores = m.predict(t.replace("\\n", " "), k=3)
        pairs.append({lang.split("__")[2]: float(s) for lang, s in zip(langs, scores)})
    print(json.dumps({"pairs": pairs}))
    sys.exit()
patch_quietly("fasttext")
import datatrove.io, datatrove.utils.lid as lid
from datatrove.data import Document
# Not part of what is tested: datatrove insists on the fasttext-numpy2-wheel *distribution*
# (importlib.metadata), and would download lid.176.bin.
lid.check_required_dependencies = lambda *args, **kwargs: None
datatrove.io.download_file = lambda remote, local, progress=True: os.symlink(model_path, local)
model = lid.FT176LID(k=3)
pairs = [model.predict(Document(text=t, id=str(i)))[1] for i, t in enumerate(texts)]
print(json.dumps({"pairs": pairs, "model": type(model.model).__module__}))
"""


@needs_fineweb
@pytest.mark.skipif(not LID_BIN.is_file(), reason=f"needs {LID_BIN}")
@pytest.mark.skipif(
    not installed("datatrove", "fasteners", "aiohttp"), reason="needs datatrove, fasteners, aiohttp"
)
def test_datatrove_ft176lid_patched(tmp_path: Path) -> None:
    n = os.environ.get("PYROID_PATCH_TEST_DOCS", "300")
    env = {"HF_ASSETS_CACHE": str(tmp_path)}
    patched = child(DATATROVE, "patched", n, str(LID_BIN), env=env)
    assert patched["model"] == "pyroid.fasttext"
    assert all(len(p) == 3 for p in patched["pairs"])
    if not installed("fasttext"):
        return
    ref = child(DATATROVE, "reference", n, str(LID_BIN))
    assert len(ref["pairs"]) == len(patched["pairs"])
    for want, got in zip(ref["pairs"], patched["pairs"], strict=True):
        assert list(got) == list(want)
        assert list(got.values()) == pytest.approx(list(want.values()), abs=1e-6)
    digest = hashlib.sha256(json.dumps(patched["pairs"]).encode()).hexdigest()[:12]
    sys.stdout.write(f"\ndatatrove FT176LID: {len(ref['pairs'])} docs identical ({digest})\n")
