"""fastText inference in Rust (``pyroid.fasttext``).

A drop-in for the inference API of the ``fasttext`` package: the same
``load_model`` / ``Model.predict`` calls, the same ``.bin`` and quantized
``.ftz`` files, and the same labels and probabilities, from a pure-Rust port of
fastText's C++. ``predict`` and ``get_sentence_vector`` also take a list,
tuple or Arrow column and run it on pyroid's thread pool without the GIL.

Needs numpy (``pip install 'pyroid[fasttext]'``), which the first call that
returns an array imports. Types and docs are in
``fasttext.pyi``. fastText is Copyright (c) 2016-present, Facebook, Inc.
(MIT License).
"""

import importlib.util as _importlib_util
import os

# numpy is imported by the first call that returns an array, not here (it takes about 90 ms),
# but a missing numpy fails the import, with a clear message.
if _importlib_util.find_spec("numpy") is None:  # pragma: no cover - test_fasttext.py runs this
    try:
        import numpy  # noqa: F401  (fails: its error is chained below)
    except ImportError as _exc:
        # ModuleNotFoundError naming numpy, as `import numpy` raises: pytest.importorskip skips.
        raise ModuleNotFoundError(
            "pyroid.fasttext needs numpy: install it with pip install 'pyroid[fasttext]' "
            "(or pip install numpy)",
            name="numpy",
        ) from _exc

from pyroid import _versions
from pyroid._pyroid import fasttext as _native

# fasttext-predict 0.9.2.4 is the reference; fasttext 0.9.3 (NeMo Curator's) gave the same
# results. Upstream fastText 0.9.2's wheels (fasttext-numpy2-wheel, which datatrove[processing]
# installs, and fasttext-wheel) round the output layer differently on aarch64; pyroid computes
# as they do when one of them is the installed fasttext (docs/guides/fasttext.md).
_UPSTREAM_BUILDS = ("fasttext-numpy2-wheel==0.9.2", "fasttext-wheel==0.9.2")
_versions.check(
    __name__,
    "fasttext",
    "fasttext-predict==0.9.2.4",
    same=("fasttext==0.9.3", *_UPSTREAM_BUILDS),
)
_UPSTREAM = _versions.distribution(
    "fasttext", ("fasttext-predict", *(b.partition("==")[0] for b in _UPSTREAM_BUILDS))
) in {b.partition("==")[0] for b in _UPSTREAM_BUILDS}

Model = _native.Model
ModelError = _native.ModelError
tokenize = _native.tokenize


def load_model(path: str | os.PathLike[str]) -> Model:
    """Load a fastText model: a ``.bin`` file, or a quantized ``.ftz``.

    The whole file is read and validated in one pass (the GIL is released
    meanwhile). The model is immutable: share it across threads.

    Its probabilities are those of the fastText build that is installed:
    fasttext-numpy2-wheel's or fasttext-wheel's (upstream fastText 0.9.2, what
    ``datatrove[processing]`` installs) when one of them is, else
    fasttext-predict 0.9.2.4's. The two differ only on aarch64, in the last
    bits.

    Args:
        path: The model file.

    Returns:
        The loaded :class:`Model`.

    Raises:
        ModelError: The file is not a fastText model (``"... has wrong file
            format!"``), is truncated, or is internally inconsistent (sizes or
            ids that fastText itself would read out of bounds).
        OSError: The file cannot be opened or read (``FileNotFoundError``,
            ``PermissionError``, ...; fastText raises ``ValueError`` here).
        MemoryError: A matrix of a valid file does not fit in memory (as
            fastText's ``std::bad_alloc``).
    """
    return _native.load_model(path, _upstream=_UPSTREAM)


def _load_upstream(path: str | os.PathLike[str]) -> Model:
    """:func:`load_model` computing as upstream fastText 0.9.2's wheels do, whatever is installed.

    For the steps that reproduce datatrove, whose ``processing`` extra installs
    fasttext-numpy2-wheel.
    """
    return _native.load_model(path, _upstream=True)

def _load_predict(path: str | os.PathLike[str]) -> Model:
    """:func:`load_model` computing as fasttext-predict 0.9.2.4 does, whatever is installed.

    For lid.176.bin that gives the same bits on aarch64 as every build gives on
    x86-64: ``pyroid clean``'s output is the same on every platform.
    """
    return _native.load_model(path, _upstream=False)

__all__ = ["Model", "ModelError", "load_model", "tokenize"]
