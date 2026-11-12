# Derived from pyvi 0.1.1 (MIT), © 2016 Viet-Trung Tran.
# ruff: noqa: T201  (a build script that reports on stdout)
"""Write `data/`: the pyvi 0.1.1 data that spaCy's Vietnamese tokenizer uses.

spaCy's `VietnameseTokenizer` (with `use_pyvi = true`, its default) labels syllables
with `pyvi.ViTokenizer.ViTokenizer.model`, an sklearn-crfsuite `CRF` unpickled from
`pyvi/models/pyvi3.pkl`, and builds its n-gram features from `pyvi/models/words.txt`.
This script copies both out of an installed pyvi 0.1.1:

* `data/pyvi3.crfsuite`: the CRFsuite model file inside the pickle (the bytes
  python-crfsuite opens), unchanged;
* `data/words.txt`: `pyvi/models/words.txt`, unchanged.

It checks the sha256 of the inputs and of what it writes, so a different pyvi fails.

Usage (in the pyroid-dev container, never on the host; an interpreter with pyvi==0.1.1):

    python -I generate.py
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKL_SHA256 = "6cdd73aee04b6a4ad7320ca85d27d931a7ef870cfdd41beb148e86bdb8aca134"
MODEL_SHA256 = "4716a0e60bbb16135917cce8440351f9790309879dd8f069c773070da9a3e0c1"
WORDS_SHA256 = "517743d27e71d43343e4b8e9e89015cdc68d84ebe1d48106449e501d918ece4a"


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    import importlib.metadata
    import importlib.util

    assert importlib.metadata.version("pyvi") == "0.1.1", importlib.metadata.version("pyvi")
    spec = importlib.util.find_spec("pyvi")
    assert spec is not None
    assert spec.origin is not None
    models = Path(spec.origin).parent / "models"
    pkl = (models / "pyvi3.pkl").read_bytes()
    assert sha256(pkl) == PKL_SHA256, sha256(pkl)
    # pyvi's own pinned pickle, its sha256 checked above.
    crf = pickle.loads(pkl)
    model = Path(crf.modelfile.name).read_bytes()
    assert sha256(model) == MODEL_SHA256, sha256(model)
    assert model[:4] == b"lCRF"
    words = (models / "words.txt").read_bytes()
    assert sha256(words) == WORDS_SHA256, sha256(words)
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    (out / "pyvi3.crfsuite").write_bytes(model)
    (out / "words.txt").write_bytes(words)
    print(f"wrote {out}/pyvi3.crfsuite ({len(model)} bytes) and words.txt ({len(words)} bytes)")


if __name__ == "__main__":
    main()
