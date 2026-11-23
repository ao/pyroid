# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""A drop-in for langdetect 1.0.9 (``from pyroid.langdetect import detect``), in Rust.

``detect``, ``detect_langs``, :class:`DetectorFactory`, ``Detector``, custom profiles and
``LangDetectException`` behave as langdetect's, and give the same languages and the same
probabilities, bit for bit, for the same ``DetectorFactory.seed``. Without a seed each
detection draws from a generator seeded by the operating system, as langdetect's does.
:func:`detect_batch` and :func:`detect_langs_batch` detect many texts on pyroid's thread
pool, each one as a separate ``detect`` call would.
"""

from typing import Any as _Any

from pyroid import _versions
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

_versions.check(
    __name__,
    "langdetect",
    f"langdetect=={LANGDETECT_VERSION}",
    same=tuple(f"langdetect=={v}" for v in _compat.SAME_RELEASES),
)


def _errors_none(errors: _Any) -> bool:
    if errors == "raise" and type(errors) is str:
        return False
    if errors == "none" and type(errors) is str:
        return True
    raise ValueError(f"errors must be 'raise' or 'none', not {errors!r}")


def _batch(texts: _Any, probs: bool, errors: _Any, threads: _Any) -> _Any:
    from pyroid.langdetect import detector_factory as factory_module

    errors_none = _errors_none(errors)
    factory_module.init_factory()
    factory: _Any = factory_module._factory
    try:
        if not factory_module._plain(factory) or type(Detector.PROB_THRESHOLD) not in (int, float):
            raise _Unsupported
        seed = _seed_arg(factory.seed)
        params = _params(Detector.ALPHA_DEFAULT, 7, Detector)
    except _Unsupported:
        # A customised factory or class, or settings of unusual types: one call per document.
        one = factory_module.detect_langs if probs else factory_module.detect
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


def detect_batch(texts: _Any, *, errors: str = "raise", threads: int | None = None) -> _Any:
    """The language of every text, as ``[detect(t) for t in texts]`` gives them.

    Each text is detected as a separate :func:`detect` call would (with the default factory,
    seeded with ``DetectorFactory.seed``), so the results equal the loop's; the texts are
    processed on pyroid's shared thread pool, and Ctrl-C stops the call within a fraction of
    a second.

    Args:
        texts: A list or tuple of str, or an Arrow string array (pyarrow, polars, ...). A
            single str is not accepted.
        errors: ``"raise"``: the first text langdetect cannot detect (no letters, only a URL)
            raises :class:`LangDetectException`, with a note naming its index; ``"none"``: its
            result is ``None``.
        threads: At most this many tasks at once on pyroid's pool (``None``: all of it; ``1``:
            the calling thread).

    Returns:
        A list (a tuple for a tuple) of language codes, or a :class:`pyroid.Column` of strings
        for Arrow input, where an Arrow null gives a null.

    Raises:
        LangDetectException: With ``errors="raise"``, for a text with no features.
        TypeError: `texts` is not a list, tuple or Arrow string array, or an item is not a str
            (langdetect's own message, with a note).
        ValueError: `errors` is not ``"raise"`` or ``"none"``.
    """
    return _batch(texts, False, errors, threads)


def detect_langs_batch(texts: _Any, *, errors: str = "raise", threads: int | None = None) -> _Any:
    """The languages and probabilities of every text, as ``[detect_langs(t) for t in texts]``.

    Args:
        texts: A list or tuple of str, or an Arrow string array.
        errors: As for :func:`detect_batch`.
        threads: As for :func:`detect_batch`.

    Returns:
        One list of :class:`Language` per text, in a list (a tuple for a tuple); ``None`` for an
        Arrow null, and with ``errors="none"`` for a text with no features.

    Raises:
        LangDetectException: With ``errors="raise"``, for a text with no features.
        TypeError: `texts` is not a list, tuple or Arrow string array, or an item is not a str.
        ValueError: `errors` is not ``"raise"`` or ``"none"``.
    """
    return _batch(texts, True, errors, threads)
