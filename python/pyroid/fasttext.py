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
except ImportError as _exc:  # pragma: no cover - exercised in a numpy-free env
    raise ImportError(
        "pyroid.fasttext needs numpy: install it with pip install 'pyroid[fasttext]' "
        "(or pip install numpy)"
    ) from _exc

from pyroid._pyroid import fasttext as _native

Model = _native.Model
ModelError = _native.ModelError
load_model = _native.load_model
tokenize = _native.tokenize

__all__ = ["Model", "ModelError", "load_model", "tokenize"]
