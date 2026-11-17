# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""``pyroid.patch("langdetect")``.

With langdetect installed, four of langdetect's own functions run pyroid's code: their
``__code__`` is swapped, so they stay the same objects (a name imported before the patch is
patched too, and every one of them pickles as langdetect's). ``Detector.append`` and
``Detector._detect_block`` run in Rust on langdetect's own instance state, and ``detect`` /
``detect_langs`` take one native call, whenever langdetect's classes are as langdetect defines
them; anything customised (a pattern, a subclass's method, ``NGram``) runs langdetect's own
code. Each replacement gives langdetect's results on langdetect's object state, so objects
made before the patch, subclasses, and calls running while ``unpatch()`` restores the
originals stay correct whatever mix of original and patched code they meet. The n-gram table
for a langdetect factory is built from its ``word_lang_prob_map`` the first time it detects
(about 0.1 s for the default profiles), and rebuilt when its language list or map grows.

Without langdetect, ``langdetect`` and its submodules are shims holding pyroid's classes;
``langdetect.detector_factory`` reads and sets ``_factory``, ``PROFILES_DIRECTORY`` and the
other module globals langdetect's code uses on :mod:`pyroid.langdetect.detector_factory`.
"""

from __future__ import annotations

import importlib
import random
import sys
import threading
import types
from collections.abc import Callable
from typing import Any

from pyroid._pyroid import langdetect as _native

# The detector's private helpers (not in its stub).
_pd: Any = importlib.import_module("pyroid.langdetect.detector")
_png: Any = importlib.import_module("pyroid.langdetect.utils.ngram")

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
# The module globals langdetect's code reads when it runs: on a shim, reading or setting one
# reads or sets pyroid's module's (``detector_factory._factory = custom`` is the default
# factory of ``langdetect.detect`` then, as in langdetect).
_LIVE = {
    "langdetect.detector_factory": (
        "_factory",
        "PROFILES_DIRECTORY",
        "DetectorFactory",
        "Detector",
        "ErrorCode",
        "LangDetectException",
        "LangProfile",
    ),
    "langdetect.detector": ("ErrorCode", "LangDetectException", "Language", "NGram"),
    "langdetect.utils.lang_profile": ("NGram",),
}


def _live_shim(shim: types.ModuleType, impl: types.ModuleType, names: tuple[str, ...]) -> None:
    """Make `shim` read and set `names` on `impl`."""
    ns = shim.__dict__
    forward = ns["__getattr__"]
    for n in names:
        ns.pop(n, None)

    def getattr_(attr: str) -> Any:
        if attr in names:
            return getattr(impl, attr)
        return forward(attr)

    getattr_.__name__ = getattr_.__qualname__ = "__getattr__"
    ns["__getattr__"] = getattr_
    base = type(shim)

    class _LiveShim(base):  # type: ignore[valid-type, misc]
        __slots__ = ()

        def __setattr__(self, attr: str, value: Any) -> None:
            if attr in names:
                setattr(impl, attr, value)
            else:
                super().__setattr__(attr, value)

        def __delattr__(self, attr: str) -> None:
            if attr in names:
                delattr(impl, attr)
            else:
                super().__delattr__(attr)

        def __dir__(self) -> list[str]:
            return sorted({*super().__dir__(), *(n for n in names if hasattr(impl, n))})

    _LiveShim.__name__ = _LiveShim.__qualname__ = base.__name__
    _LiveShim.__module__ = base.__module__
    shim.__class__ = _LiveShim


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
        shim = make(name, None, missing, impl, provided, package=package)
        if name in _LIVE:
            _live_shim(shim, impl, _LIVE[name])
        out[name] = shim
    return out


# ---------------------------------------------------------------------------
# langdetect installed: its functions run pyroid's code
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
# id(word_lang_prob_map) -> (map, langlist, len(langlist), len(map), native store or None)
_STORES: dict[int, tuple[Any, Any, int, int, Any]] = {}
_Random = random.Random
# A global of langdetect's patched modules while the patch is on: this module, for the
# swapped code to find (it imports it if the global is gone, after an unpatch()).
_HOOK = "__pyroid_langdetect__"
# Never equal to a function: what a check compares with when langdetect's own was not found.
_NOT_FOUND = object()


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


# The code langdetect's four functions run under the patch (their globals are langdetect's).


def _t_append(self: Any, text: Any) -> Any:
    hook = globals().get("__pyroid_langdetect__") or __import__("pyroid").langdetect._patch
    return hook._append(self, text)


def _t_detect_block(self: Any) -> Any:
    hook = globals().get("__pyroid_langdetect__") or __import__("pyroid").langdetect._patch
    return hook._detect_block(self)


def _t_detect(text: Any) -> Any:
    hook = globals().get("__pyroid_langdetect__") or __import__("pyroid").langdetect._patch
    return hook.detect(text)


def _t_detect_langs(text: Any) -> Any:
    hook = globals().get("__pyroid_langdetect__") or __import__("pyroid").langdetect._patch
    return hook.detect_langs(text)


# (module, class or None, name, pyroid's code): the functions the patch swaps the code of.
_SWAPS = (
    ("langdetect.detector", "Detector", "append", _t_append),
    ("langdetect.detector", "Detector", "_detect_block", _t_detect_block),
    ("langdetect.detector_factory", None, "detect", _t_detect),
    ("langdetect.detector_factory", None, "detect_langs", _t_detect_langs),
)
_CODES = {
    name: template.__code__.replace(co_name=name, co_qualname=f"{owner}.{name}" if owner else name)
    for _, owner, name, template in _SWAPS
}
# Each swapped function -> its original code (shared by every install, so a function running
# pyroid's code is never taken for langdetect's).
_ORIGINAL_CODE: dict[Any, types.CodeType] = {}
# name -> a function running langdetect's original code (the fallback of pyroid's).
_ORIGINALS: dict[str, Any] = {}


def _ours(code: Any) -> bool:
    return any(code is c for c in _CODES.values())


def _langdetects(fn: Any, module: str, qualname: str) -> bool:
    """`fn` is langdetect's own function `module.qualname` (not a replacement)."""
    return (
        type(fn) is types.FunctionType
        and fn.__module__ == module
        and fn.__qualname__ == qualname
        and not fn.__closure__
    )


def _twin(fn: Any, code: types.CodeType) -> Any:
    """A function like `fn` running `code`."""
    twin = types.FunctionType(code, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)
    twin.__kwdefaults__ = fn.__kwdefaults__
    twin.__qualname__ = fn.__qualname__
    twin.__module__ = fn.__module__
    return twin


def _language() -> Any:
    from pyroid.langdetect._compat import Language

    return Language


class _Patched:
    """langdetect's classes as one install() found them, and what the checks compare with."""

    def __init__(self) -> None:
        # By name: langdetect ships no types.
        self.langdetect = importlib.import_module("langdetect")
        self.detector_module = importlib.import_module("langdetect.detector")
        self.factory_module = importlib.import_module("langdetect.detector_factory")
        self.ngram_module = importlib.import_module("langdetect.utils.ngram")
        # Typed Any: the checks compare their entries with `is`, which mypy would narrow by.
        self.detector_globals: Any = self.detector_module.__dict__
        self.factory_globals: Any = self.factory_module.__dict__
        self.ngram_globals: Any = self.ngram_module.__dict__
        unicode_module = importlib.import_module("langdetect.utils.unicode_block")
        self.detector = self.detector_module.Detector
        self.factory = self.factory_module.DetectorFactory
        self.ngram = self.ngram_module.NGram
        self.saved: list[tuple[Any, str, Any]] = []
        block = unicode_module.__dict__.get("unicode_block", _NOT_FOUND)
        self.unicode_block = (
            block if _langdetects(block, unicode_module.__name__, "unicode_block") else _NOT_FOUND
        )

        def own(owner: Any, name: str, module: str, qualname: str) -> Any:
            fn = owner.__dict__.get(name, _NOT_FOUND)
            if isinstance(fn, classmethod):
                fn = fn.__func__
            return fn if _langdetects(fn, module, qualname) else _NOT_FOUND

        d, f, g = self.detector, self.factory, self.ngram
        ld, ldf, ldn = (
            "langdetect.detector",
            "langdetect.detector_factory",
            "langdetect.utils.ngram",
        )
        names = (
            "__init__",
            "append",
            "cleaning_text",
            "detect",
            "get_probabilities",
            "_detect_block",
            "_extract_ngrams",
            "_init_probability",
            "_update_lang_prob",
            "_normalize_prob",
            "_sort_probability",
        )
        self.methods = {n: own(d, n, ld, f"Detector.{n}") for n in names}
        self.create = own(f, "create", ldf, "DetectorFactory.create")
        self.create_detector = own(f, "_create_detector", ldf, "DetectorFactory._create_detector")
        self.ngram_init = own(g, "__init__", ldn, "NGram.__init__")
        self.ngram_add_char = own(g, "add_char", ldn, "NGram.add_char")
        self.normalize = own(g, "normalize", ldn, "NGram.normalize")
        self.normalize_vi = own(g, "normalize_vi", ldn, "NGram.normalize_vi")
        cjk = g.__dict__.get("CJK_MAP")
        self.cjk_map: Any = cjk if type(cjk) is dict and cjk == _png._CJK_MAP else _NOT_FOUND
        self.cjk_size = len(_png._CJK_MAP)

    def ngram_stock(self) -> bool:
        """Whether langdetect's ``NGram`` (and the ``unicode_block`` it calls) are langdetect's.

        ``CJK_MAP`` is compared by identity and size: changing its entries in place is not
        seen.
        """
        g: Any = self.ngram
        gd: Any = g.__dict__
        n = g.N_GRAM
        cjk: Any = g.CJK_MAP
        normalize = gd.get("normalize")
        normalize_vi = gd.get("normalize_vi")
        return (
            self.detector_globals.get("NGram") is g
            and type(n) is int
            and n == 3
            and g.LATIN1_EXCLUDED == _png._LATIN1
            and cjk is self.cjk_map
            and len(cjk) == self.cjk_size
            and gd.get("__init__") is self.ngram_init
            and gd.get("add_char") is self.ngram_add_char
            and type(normalize) is classmethod
            and normalize.__func__ is self.normalize
            and type(normalize_vi) is classmethod
            and normalize_vi.__func__ is self.normalize_vi
            and _png._vi_stock(g)
            and self.unicode_block is not _NOT_FOUND
            and self.detector_globals.get("unicode_block") is self.unicode_block
            and self.ngram_globals.get("unicode_block") is self.unicode_block
        )

    def append_stock(self, detector: Any) -> bool:
        """``append`` can run in Rust: the patterns and the Vietnamese tables are langdetect's."""
        g = self.ngram
        normalize_vi = g.__dict__.get("normalize_vi")
        return (
            _pd._pattern_stock(detector.URL_RE, _pd._URL_RE)
            and _pd._pattern_stock(detector.MAIL_RE, _pd._MAIL_RE)
            and self.detector_globals.get("NGram") is g
            and type(normalize_vi) is classmethod
            and normalize_vi.__func__ is self.normalize_vi
            and _png._vi_stock(g)
        )

    def block_mode(self, detector: Any) -> int:
        """As ``pyroid.langdetect.detector._block_mode``, for langdetect's ``Detector``."""
        cls = type(detector)
        d = detector.__dict__
        m = self.methods
        if (
            cls._extract_ngrams is not m["_extract_ngrams"]
            or cls._init_probability is not m["_init_probability"]
            or cls._update_lang_prob is not m["_update_lang_prob"]
            or cls._normalize_prob is not m["_normalize_prob"]
            or type(d.get("random")) is not _Random
            or any(n in d for n in _pd._BLOCK_METHODS)
            or not self.ngram_stock()
        ):
            return 2
        return 0 if cls.cleaning_text is m["cleaning_text"] and "cleaning_text" not in d else 1

    def is_plain(self, factory: Any) -> bool:
        """`factory.create()` + ``append`` + ``detect()`` give what one native call gives."""
        f: Any = self.factory
        d: Any = self.detector
        m = self.methods
        fd = factory.__dict__
        return (
            type(factory) is f
            and "create" not in fd
            and "_create_detector" not in fd
            and f.create is self.create
            and f._create_detector is self.create_detector
            and self.factory_globals.get("Detector") is d
            and d.__init__ is m["__init__"]
            and d.append is m["append"]
            and d.append.__code__ is _CODES["append"]
            and d._detect_block is m["_detect_block"]
            and d._detect_block.__code__ is _CODES["_detect_block"]
            and d.detect is m["detect"]
            and d.get_probabilities is m["get_probabilities"]
            and d._sort_probability is m["_sort_probability"]
            and d.cleaning_text is m["cleaning_text"]
            and d._extract_ngrams is m["_extract_ngrams"]
            and d._init_probability is m["_init_probability"]
            and d._update_lang_prob is m["_update_lang_prob"]
            and d._normalize_prob is m["_normalize_prob"]
            and _pd._pattern_stock(d.URL_RE, _pd._URL_RE)
            and _pd._pattern_stock(d.MAIL_RE, _pd._MAIL_RE)
            and self.detector_globals.get("Language") is _language()
            and self.detector_globals.get("random") is random
            and random.Random is _Random
            and self.ngram_stock()
        )


def _guarded(method: Any, default: Any) -> Any:
    """Wrap the check `method` to give `default` (langdetect's own code) if it raises.

    An attribute is missing then, and langdetect's code raises it where langdetect does.
    """

    def guarded(*args: Any) -> Any:
        try:
            return method(*args)
        except Exception:
            return default

    guarded.__name__ = guarded.__qualname__ = method.__name__
    guarded.__doc__ = method.__doc__
    return guarded


for _name, _default in (
    ("ngram_stock", False),
    ("append_stock", False),
    ("block_mode", 2),
    ("is_plain", False),
):
    setattr(_Patched, _name, _guarded(getattr(_Patched, _name), _default))
del _name, _default

_ACTIVE: _Patched | None = None


def _append(self: Any, text: Any) -> None:
    """Run langdetect's ``Detector.append`` under the patch."""
    active = _ACTIVE
    if active is not None and active.append_stock(self) and _pd._append_native(self, text):
        return
    _ORIGINALS["append"](self, text)


def _detect_block(self: Any) -> None:
    """Run langdetect's ``Detector._detect_block`` under the patch."""
    active = _ACTIVE
    mode = active.block_mode(self) if active is not None else 2
    if mode != 2:
        store = _store(self.word_lang_prob_map, self.langlist)
        if store is not None:
            if mode == 0:
                try:
                    _pd._native_block(self, store)
                    return
                except _pd._Unsupported:
                    pass
            else:
                # langdetect's steps, the detector's own cleaning_text first, the rest in Rust.
                self.cleaning_text()
                _pd._block_cleaned(self, store)
                return
    _ORIGINALS["_detect_block"](self)


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
    """``langdetect.detect`` under the patch: one native call when nothing is customised."""
    try:
        return _one_call(text, False)
    except _pd._Unsupported:
        pass
    return _ORIGINALS["detect"](text)


def detect_langs(text: Any) -> Any:
    """``langdetect.detect_langs`` under the patch: one native call when nothing is customised."""
    try:
        language = _language()
        return [language(lang, p) for lang, p in _one_call(text, True)]
    except _pd._Unsupported:
        pass
    return _ORIGINALS["detect_langs"](text)


def install() -> Callable[[], None]:
    """Patch the installed langdetect in place; returns the undo function.

    A KeyboardInterrupt part-way undoes the swaps made so far, and the undo can run twice.
    """
    global _ACTIVE
    active = _Patched()
    found: list[tuple[Any, types.CodeType, types.CodeType]] = []
    for module_name, owner_name, name, _ in _SWAPS:
        module = sys.modules[module_name]
        owner = getattr(module, owner_name) if owner_name else module
        fn: Any = owner.__dict__.get(name)
        qualname = f"{owner_name}.{name}" if owner_name else name
        if not _langdetects(fn, module_name, qualname):
            continue  # replaced by someone else: left alone
        original = _ORIGINAL_CODE.get(fn, fn.__code__)
        if _ours(original):
            continue
        _ORIGINAL_CODE[fn] = original
        _ORIGINALS[name] = _twin(fn, original)
        found.append((fn, original, _CODES[name]))
    modules = (active.detector_module, active.factory_module)

    def undo() -> None:
        global _ACTIVE
        for owner, name, value in reversed(active.saved):
            if value is _NOT_FOUND:
                if name in owner.__dict__:
                    delattr(owner, name)
            else:
                setattr(owner, name, value)
        if _ACTIVE is active:
            _ACTIVE = None
        with _LOCK:
            _STORES.clear()

    try:
        _ACTIVE = active
        for module in modules:
            if _HOOK not in module.__dict__:
                active.saved.append((module, _HOOK, _NOT_FOUND))
                setattr(module, _HOOK, sys.modules[__name__])
        for fn, original, code in found:
            if fn.__code__ is not code:
                active.saved.append((fn, "__code__", original))
                fn.__code__ = code
    except BaseException:
        undo()
        raise
    return undo
