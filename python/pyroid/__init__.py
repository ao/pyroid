"""Rust versions of popular Python libraries, with the same API and the same output.

Each module is a drop-in for the library it is named after, or the one in parentheses:
``pyroid.yaml`` (PyYAML), ``pyroid.jmespath``, ``pyroid.tabulate``, ``pyroid.dateutil.parser``
(python-dateutil), ``pyroid.json_repair``, ``pyroid.difflib`` (the standard library's),
``pyroid.packaging``, ``pyroid.sqlparse``, ``pyroid.ftfy``, ``pyroid.translit`` (anyascii),
``pyroid.fasttext``, ``pyroid.langdetect``, ``pyroid.nltk``, ``pyroid.jieba``,
``pyroid.trafilatura``, ``pyroid.warc`` (warcio),
``pyroid.filters`` and ``pyroid.dedup`` (datatrove), ``pyroid.datasketch``, ``pyroid.nemo``
(NeMo Curator) and ``pyroid.metrics`` (rouge-score and sacrebleu). Change the import
(``import pyroid.yaml as yaml``), or call :func:`pyroid.patch` to make code you don't own use
them.

The batch functions (``pyroid.ftfy.fix_text_batch``, ``pyroid.filters``, ...) take a ``str``,
a ``list`` or ``tuple`` of ``str``, or an Arrow string column (pyarrow, polars, pandas, Hugging
Face ``datasets``), and return the same kind. Arrow input gives a :class:`pyroid.Column`.

pyroid 1.0 was a rewrite: the 0.7.x API is on the ``legacy`` branch and in the 0.7.x releases
on PyPI.
"""

from types import ModuleType


def _load_extension() -> ModuleType:
    # The native extension is imported unconditionally. If it is missing or
    # broken, importing pyroid fails: no pure-Python fallbacks, no dummies.
    try:
        from pyroid import _pyroid
    except ImportError as exc:
        raise ImportError(
            "pyroid's native extension (pyroid._pyroid) failed to load. Install a "
            "pyroid wheel built for this Python, or build one from a source "
            "checkout with `maturin develop`."
        ) from exc
    return _pyroid


_pyroid = _load_extension()

from pyroid._pyroid import (  # noqa: E402
    Column,
    FallbackWarning,
    InputError,
    PyroidError,
    VersionMismatchWarning,
)

__version__: str = _pyroid.__version__


def patch(*names: str) -> None:
    """Make ``import <name>`` give pyroid's drop-in, for each of `names`.

    ``pyroid.patch("ftfy", "fasttext")`` is :func:`pyroid.integrations.patch`: explicit,
    reversible ``sys.modules`` shims (see there). Nothing is imported until it is called.
    """
    from pyroid.integrations._patch import patch as _patch

    _patch(*names)


def unpatch(*names: str) -> None:
    """Undo :func:`patch` for ``names`` (all if none): :func:`pyroid.integrations.unpatch`."""
    from pyroid.integrations._patch import unpatch as _unpatch

    _unpatch(*names)


__all__ = [
    "Column",
    "FallbackWarning",
    "InputError",
    "PyroidError",
    "VersionMismatchWarning",
    "__version__",
    "patch",
    "unpatch",
]
