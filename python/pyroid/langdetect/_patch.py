# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""``pyroid.patch("langdetect")``.

With langdetect installed, langdetect's own ``Detector.append`` and ``Detector._detect_block``
are replaced by pyroid's, in place, and ``detect``/``detect_langs`` take one native call when
the default factory and classes are langdetect's own, unmodified. Each replaced method works
on langdetect's own instance state and gives langdetect's results, so objects made before the
patch, subclasses, and calls running while ``unpatch()`` restores the originals stay correct
whatever mix of original and patched methods they meet. The n-gram table for a langdetect
factory is built from its ``word_lang_prob_map`` the first time it detects (about 0.1 s for
the default profiles), and rebuilt when its language list or map grows.

Without langdetect, ``langdetect`` and its submodules are shims holding pyroid's classes.
"""

from __future__ import annotations

import importlib
import threading
import types
from collections.abc import Callable
from typing import Any

from pyroid._pyroid import langdetect as _native

# The detector's private helpers (not in its stub).
_pd: Any = importlib.import_module("pyroid.langdetect.detector")

#: The langdetect modules the shims stand in for, and the pyroid module each takes names from.
SHIMMED = {
    "langdetect": "pyroid.langdetect",
    "langdetect.detector_factory": "pyroid.langdetect.detector_factory",
    "langdetect.detector": "pyroid.langdetect.detector",
    "langdetect.language": "pyroid.langdetect.language",
    "langdetect.lang_detect_exception": "pyroid.langdetect.lang_detect_exception",
    "langdetect.utils": "pyroid.langdetect.utils",
    "langdetect.utils.lang_profile": "pyroid.langdetect.utils.lang_profile",
    "langdetect.utils.ngram": "pyroid.langdetect.utils.ngram",
}
# langdetect 1.0.9's top-level names.
_TOP = ("DetectorFactory", "PROFILES_DIRECTORY", "detect", "detect_langs", "LangDetectException")
_SUB = {
    "langdetect.detector_factory": (
        "DetectorFactory",
        "PROFILES_DIRECTORY",
        "detect",
        "detect_langs",
        "init_factory",
        "Detector",
        "ErrorCode",
        "LangDetectException",
        "LangProfile",
    ),
    "langdetect.detector": ("Detector", "ErrorCode", "LangDetectException", "Language", "NGram"),
    "langdetect.language": ("Language",),
    "langdetect.lang_detect_exception": ("ErrorCode", "LangDetectException"),
    "langdetect.utils": (),
    "langdetect.utils.lang_profile": ("LangProfile", "NGram"),
    "langdetect.utils.ngram": ("NGram",),
}


def shims(make: Callable[..., types.ModuleType], missing: str) -> dict[str, types.ModuleType]:
    """The shims for langdetect and its submodules (langdetect not installed)."""
    import pyroid.langdetect as top
    from pyroid.langdetect.utils.ngram import NGram

    out: dict[str, types.ModuleType] = {}
    for name in reversed(list(SHIMMED)):
        impl = importlib.import_module(SHIMMED[name])
        names = _TOP if name == "langdetect" else _SUB[name]
        provided: dict[str, Any] = {}
        for n in names:
            provided[n] = NGram if n == "NGram" else getattr(impl, n, None) or getattr(top, n)
        for sub, module in out.items():
            parent, _, child = sub.rpartition(".")
            if parent == name:
                provided[child] = module
        package = name in ("langdetect", "langdetect.utils")
        out[name] = make(name, None, missing, impl, provided, package=package)
    return out


# ---------------------------------------------------------------------------
# langdetect installed: its classes run pyroid's code
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
# id(word_lang_prob_map) -> (map, langlist, len(langlist), len(map), native store or None)
_STORES: dict[int, tuple[Any, Any, int, int, Any]] = {}


def _store(word_map: Any, langlist: Any) -> Any:
    """The native table for a langdetect factory's map (None: use langdetect's own code)."""
    entry = _STORES.get(id(word_map))
    if (
        entry is not None
        and entry[0] is word_map
        and entry[1] is langlist
        and entry[2] == len(langlist)
        and entry[3] == len(word_map)
    ):
        return entry[4]
    if type(word_map) is not dict:
        return None
    store = _native._Profiles._from_map(word_map)
    with _LOCK:
        if len(_STORES) >= 4:
            _STORES.clear()
        _STORES[id(word_map)] = (word_map, langlist, len(langlist), len(word_map), store)
    return store


class _Patched:
    """The originals replaced by one install(), and the functions that replaced them."""

    def __init__(self) -> None:
        import langdetect
        import langdetect.detector as ld_detector
        import langdetect.detector_factory as ld_factory

        self.langdetect = langdetect
        self.factory_module = ld_factory
        self.detector = ld_detector.Detector
        self.factory = ld_factory.DetectorFactory
        d, f = self.detector, self.factory
        self.original_detect_block = d.__dict__["_detect_block"]
        self.original_detect = ld_factory.__dict__["detect"]
        self.original_detect_langs = ld_factory.__dict__["detect_langs"]
        # What the one-call path stands for: create(), Detector(), append(), detect().
        self.plain = (
            f.__dict__.get("create"),
            f.__dict__.get("_create_detector"),
            d.__dict__.get("__init__"),
            d.__dict__.get("detect"),
            d.__dict__.get("get_probabilities"),
            d.__dict__.get("_sort_probability"),
        )
        self.saved: list[tuple[Any, str, Any]] = []

    def is_plain(self, factory: Any) -> bool:
        f, d = self.factory, self.detector
        return (
            type(factory) is f
            and self.factory_module.__dict__.get("Detector") is d
            and d.__dict__.get("append") is _append
            and d.__dict__.get("_detect_block") is _detect_block
            and (
                f.__dict__.get("create"),
                f.__dict__.get("_create_detector"),
                d.__dict__.get("__init__"),
                d.__dict__.get("detect"),
                d.__dict__.get("get_probabilities"),
                d.__dict__.get("_sort_probability"),
            )
            == self.plain
        )


_ACTIVE: _Patched | None = None


def _append(self: Any, text: Any) -> None:
    _pd._append(self, text)


def _detect_block(self: Any) -> None:
    active = _ACTIVE
    store = _store(self.word_lang_prob_map, self.langlist)
    if store is not None:
        try:
            _pd._native_block(self, store)
            return
        except _pd._Unsupported:
            pass
    original = active.original_detect_block if active is not None else _ORIGINALS["_detect_block"]
    original(self)


_ORIGINALS: dict[str, Any] = {}


def _one_call(text: Any, probs: bool) -> Any:
    active = _ACTIVE
    if active is None:
        raise _pd._Unsupported
    module = active.factory_module
    module.init_factory()
    factory = module._factory
    if not isinstance(text, str) or not active.is_plain(factory):
        raise _pd._Unsupported
    store = _store(factory.word_lang_prob_map, factory.langlist)
    if store is None:
        raise _pd._Unsupported
    d = active.detector
    seed = _pd._seed_arg(factory.seed)
    params = _pd._params(d.ALPHA_DEFAULT, 7, d)
    if type(d.PROB_THRESHOLD) not in (int, float):
        raise _pd._Unsupported
    code, value = _native.detect_one(
        store, factory.langlist, text, seed, params, d.PROB_THRESHOLD, probs, d.UNKNOWN_LANG
    )
    if code == 0:
        return value
    if code == 1:
        from pyroid.langdetect._compat import ErrorCode, LangDetectException

        raise LangDetectException(ErrorCode.CantDetectError, "No features in text.")
    raise _pd._Unsupported


def detect(text: Any) -> Any:
    """Langdetect's ``detect``, in one native call when nothing is customised."""
    try:
        return _one_call(text, False)
    except _pd._Unsupported:
        pass
    active = _ACTIVE
    original = active.original_detect if active is not None else _ORIGINALS["detect"]
    return original(text)


def detect_langs(text: Any) -> Any:
    """Langdetect's ``detect_langs``, in one native call when nothing is customised."""
    from pyroid.langdetect._compat import Language

    try:
        return [Language(lang, p) for lang, p in _one_call(text, True)]
    except _pd._Unsupported:
        pass
    active = _ACTIVE
    original = active.original_detect_langs if active is not None else _ORIGINALS["detect_langs"]
    return original(text)


def install() -> Callable[[], None]:
    """Patch the installed langdetect in place; returns the undo function."""
    global _ACTIVE
    active = _Patched()
    _ORIGINALS.update(
        _detect_block=active.original_detect_block,
        detect=active.original_detect,
        detect_langs=active.original_detect_langs,
    )

    def swap(owner: Any, name: str, new: Any) -> None:
        active.saved.append((owner, name, owner.__dict__.get(name, _MISSING)))
        setattr(owner, name, new)

    _ACTIVE = active
    swap(active.detector, "append", _append)
    swap(active.detector, "_detect_block", _detect_block)
    for module in (active.factory_module, active.langdetect):
        if module.__dict__.get("detect") is active.original_detect:
            swap(module, "detect", detect)
        if module.__dict__.get("detect_langs") is active.original_detect_langs:
            swap(module, "detect_langs", detect_langs)

    def undo() -> None:
        global _ACTIVE
        for owner, name, original in reversed(active.saved):
            if original is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, original)
        if _ACTIVE is active:
            _ACTIVE = None
        with _LOCK:
            _STORES.clear()

    return undo


_MISSING = object()
