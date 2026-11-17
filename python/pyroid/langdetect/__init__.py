# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""A drop-in for langdetect 1.0.9 (``from pyroid.langdetect import detect``), in Rust.

``detect``, ``detect_langs``, :class:`DetectorFactory`, ``Detector``, custom profiles and
``LangDetectException`` behave as langdetect's, and give the same languages and the same
probabilities, bit for bit, for the same ``DetectorFactory.seed``. Without a seed each
detection draws from a generator seeded by the operating system, as langdetect's does.
:func:`detect_batch` and :func:`detect_langs_batch` detect many texts on pyroid's thread
pool, each one as a separate ``detect`` call would.
"""

from __future__ import annotations

from typing import Any

from pyroid._pyroid import langdetect as _native
from pyroid.langdetect import _compat
from pyroid.langdetect._compat import LANGDETECT_VERSION, ErrorCode, LangDetectException, Language
from pyroid.langdetect.detector import Detector, _params, _seed_arg, _Unsupported
from pyroid.langdetect.detector_factory import (
    PROFILES_DIRECTORY,
    DetectorFactory,
    detect,
    detect_langs,
    init_factory,
)
from pyroid.langdetect.utils.lang_profile import LangProfile

__all__ = [
    "LANGDETECT_VERSION",
    "PROFILES_DIRECTORY",
    "Detector",
    "DetectorFactory",
    "ErrorCode",
    "LangDetectException",
    "LangProfile",
    "Language",
    "detect",
    "detect_batch",
    "detect_langs",
    "detect_langs_batch",
    "init_factory",
]

__version__ = LANGDETECT_VERSION

_compat.check_version()


def _errors_none(errors: Any) -> bool:
    if errors == "raise" and type(errors) is str:
        return False
    if errors == "none" and type(errors) is str:
        return True
    raise ValueError(f"errors must be 'raise' or 'none', not {errors!r}")


def _batch(texts: Any, probs: bool, errors: Any, threads: Any) -> Any:
    from pyroid.langdetect import detector_factory as factory_module

    errors_none = _errors_none(errors)
    init_factory()
    factory = factory_module._factory
    assert factory is not None
    try:
        seed = _seed_arg(factory.seed)
        params = _params(Detector.ALPHA_DEFAULT, 7, Detector)
    except _Unsupported:
        seed = params = None
    if params is None or type(Detector.PROB_THRESHOLD) not in (int, float):
        # Seeds or class constants of unusual types: one call per document.
        one = detect_langs if probs else detect
        if isinstance(texts, (list, tuple)):
            items = texts
        else:  # an Arrow array: its values as str (None for nulls)
            to_pylist = getattr(texts, "to_pylist", None)
            items = to_pylist() if to_pylist is not None else list(texts)
        out = []
        for text in items:
            if text is None and not isinstance(texts, (list, tuple)):
                out.append(None)
                continue
            try:
                out.append(one(text))
            except LangDetectException:
                if not errors_none:
                    raise
                out.append(None)
        return tuple(out) if isinstance(texts, tuple) else out
    result, index, code = _native.detect_many(
        factory._profiles,
        factory.langlist,
        texts,
        seed,
        params,
        Detector.PROB_THRESHOLD,
        probs,
        Detector.UNKNOWN_LANG,
        Language,
        errors_none,
        threads,
    )
    if index < 0:
        return result
    note = f"pyroid.langdetect: raised for the text at index {index}"
    try:
        if code == 1:
            raise LangDetectException(ErrorCode.CantDetectError, "No features in text.")
        if code == 4:
            Detector.URL_RE.sub(" ", texts[index])  # langdetect's error for a non-str
        if code == 2:
            raise IndexError("list index out of range")
        if code == 3:
            zero = 0.0
            _ = 1.0 / zero
        raise RuntimeError(f"pyroid.langdetect: unexpected batch result {code}")
    except Exception as exc:
        exc.add_note(note)
        raise


def detect_batch(texts: Any, *, errors: str = "raise", threads: int | None = None) -> Any:
    """``detect(text)`` for every text: a list (tuple for a tuple; a Column for Arrow)."""
    return _batch(texts, False, errors, threads)


def detect_langs_batch(texts: Any, *, errors: str = "raise", threads: int | None = None) -> Any:
    """``detect_langs(text)`` for every text: one list of ``Language`` per text."""
    return _batch(texts, True, errors, threads)
