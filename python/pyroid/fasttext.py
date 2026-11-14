"""fastText inference in Rust (``pyroid.fasttext``).

A drop-in for the inference API of the ``fasttext`` package: the same
``load_model`` / ``Model.predict`` calls, the same ``.bin`` and quantized
``.ftz`` files, and the same labels and probabilities, from a pure-Rust port of
fastText's C++. ``predict`` and ``get_sentence_vector`` also take a list,
tuple or Arrow column and run it on pyroid's thread pool without the GIL.

Needs numpy (``pip install 'pyroid[fasttext]'``). Types and docs are in
``fasttext.pyi``. fastText is Copyright (c) 2016-present, Facebook, Inc.
(MIT License).
"""

try:
    import numpy as _numpy  # noqa: F401  (imported to fail early, with a clear message)
except ImportError as _exc:  # pragma: no cover - test_fasttext.py runs this in a subprocess
    # ModuleNotFoundError naming numpy, as `import numpy` itself raises: pytest.importorskip skips.
    raise ModuleNotFoundError(
        "pyroid.fasttext needs numpy: install it with pip install 'pyroid[fasttext]' "
        "(or pip install numpy)",
        name="numpy",
    ) from _exc

from pyroid import _versions
from pyroid._pyroid import fasttext as _native

# fasttext-predict 0.9.2.4 is the reference; fasttext 0.9.3 (NeMo Curator's) gave the same
# results. Other builds of the fasttext module can differ in the last bits of a probability
# (docs/guides/fasttext.md: fasttext-numpy2-wheel by up to 1.79e-7).
_versions.check(
    __name__,
    "fasttext",
    "fasttext-predict==0.9.2.4",
    same=("fasttext==0.9.3",),
    distributions=("fasttext-wheel", "fasttext-numpy2-wheel"),
)

Model = _native.Model
ModelError = _native.ModelError
load_model = _native.load_model
tokenize = _native.tokenize

__all__ = ["Model", "ModelError", "load_model", "tokenize"]
