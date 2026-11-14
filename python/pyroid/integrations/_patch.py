"""Make existing ``import ftfy`` / ``import fasttext`` code use pyroid, without edits.

Nothing happens until you ask: installing pyroid changes no imports (there are no ``.pth``
files), and :func:`patch` is the only switch::

    import pyroid.integrations

    pyroid.integrations.patch("ftfy", "fasttext")
    import ftfy  # pyroid's drop-in: ftfy.fix_text is pyroid.ftfy.fix_text
    from fasttext.FastText import _FastText

    model = _FastText("lid.176.bin")  # a pyroid.fasttext.Model
    pyroid.integrations.unpatch()  # sys.modules is back to how it was

The names are ``"ftfy"`` (:mod:`pyroid.ftfy`, which reproduces ftfy 6.3.1), ``"fasttext"``
(:mod:`pyroid.fasttext`: ``load_model``, ``tokenize``, ``FastText._FastText(path)`` and
``FastText.load_model`` give pyroid models), ``"anyascii"`` (:func:`pyroid.translit.anyascii`)
and ``"datasketch"`` (:mod:`pyroid.datasketch`, which reproduces datasketch 2.0.0: ``datasketch``
and its ``minhash``, ``lean_minhash``, ``hashfunc``, ``lsh``, ``lshforest`` and ``lshensemble``
submodules give pyroid's classes), ``"trafilatura"`` (:mod:`pyroid.trafilatura`, which
reproduces trafilatura 2.3.1 or 1.11.0, following the installed release: ``extract``,
``bare_extraction`` and (2.3.1) ``extract_with_metadata`` of ``trafilatura`` and
``trafilatura.core``, ``trafilatura.deduplication.LRUCache`` / ``LRU_TEST`` and
``trafilatura.meta.reset_caches``), ``"nltk"`` (:mod:`pyroid.nltk`, which reproduces NLTK
3.10.4's tokenizers) and ``"warcio"`` (:mod:`pyroid.warc`, which reproduces warcio 1.8.2's
``ArchiveIterator``, ``WARCIterator`` and ``ARCIterator``). There is no ``"unidecode"``:
:func:`pyroid.translit.unidecode` transliterates with anyascii's tables, so its output differs
from Unidecode's.

:func:`patch` puts a shim in ``sys.modules[name]``: a module named like the real one whose
functions are pyroid's (the same objects, so a call costs what calling pyroid costs). If the
real package is installed, :func:`patch` imports it right away, because the shim needs its
``__path__`` and ``__spec__`` up front (loading it later under the shim's name would be
fragile). Then:

- every attribute the shim does not provide comes from the real package (module
  ``__getattr__``): ``ftfy.guess_bytes``, ``ftfy.fix_file``, ``fasttext.train_supervised``...;
- real submodules keep working, as attributes and as ``import ftfy.fixes``: the shim shares the
  real ``__path__``, and its ``__spec__`` is a copy of the real one, so
  ``importlib.util.find_spec`` and ``importlib.resources`` see the real package;
- ``__pyroid__`` is ``True``, ``__wrapped__`` is the real module, and ``ftfy.__version__`` /
  ``anyascii.__version__`` are the versions pyroid reproduces (fasttext has none, as fastText);
- modules that imported the real package before :func:`patch` keep (and can still use) it.

Without the real package the shim still works for what pyroid provides; other attributes raise
``AttributeError`` saying so, and ``importlib.util.find_spec(name)`` still finds the shim (useful
where no wheel exists: fastText on CPython 3.14+ and free-threaded builds). ftfy's
``apply_plan``, ``explain_unicode``, ``fix_file`` and ``guess_bytes``, which pyroid only
delegates, are the real ftfy's when it is installed. Calls that pyroid hands to the real package
(option values the Rust port does not support; ``_FastText()`` without a model file) go to the
real one, never back to the shim, with the usual one-time :class:`pyroid.FallbackWarning`, or
raise :class:`pyroid.PyroidError` if it is not installed. :func:`patch` warns once per process
and name with :class:`PatchWarning`.

Child processes: :func:`run` is ``pyroid run [-p NAMES] (-m MODULE | SCRIPT) [ARGS...]``. It
patches ``NAMES`` (default ``$PYROID_PATCH``, else ``ftfy,fasttext``) and runs the target as
``python -m MODULE`` / ``python SCRIPT`` would, in this process. It also sets ``PYROID_PATCH`` and
puts :data:`HOOK_DIR` first on ``PYTHONPATH``, so the Python processes it starts
(``multiprocessing`` spawn and forkserver workers, ``subprocess``) are patched at start-up by the
``sitecustomize`` in that directory, which then runs the environment's own ``sitecustomize``, if
any. The same works by hand, with no ``pyroid run``::

    export PYROID_PATCH=ftfy,fasttext
    export PYTHONPATH="$(python -c 'import pyroid.integrations._patch as p; print(p.HOOK_DIR)')"
    python script.py

The hook warns once per process tree, not in every worker. It does nothing in an interpreter that
cannot import pyroid, nor in one started with ``-I`` or ``-E`` (``PYTHONPATH`` is ignored there).

``"nltk"`` works differently when NLTK is installed: no shim, but ``sent_tokenize``,
``word_tokenize``, ``punkt_model_available``, ``PunktSentenceTokenizer``, ``PunktTokenizer``,
``TreebankWordTokenizer``, ``NLTKWordTokenizer`` and ``TreebankWordDetokenizer`` are replaced by
pyroid's on NLTK's own modules (``nltk``, ``nltk.tokenize`` and the ``punkt``, ``treebank`` and
``destructive`` submodules), so code calling ``nltk.word_tokenize(...)`` gets pyroid's, and
:func:`unpatch` puts NLTK's back. Everything else (``PunktTrainer``, the taggers, corpora, other
tokenizers, rouge-score's Porter stemmer) stays NLTK's. Without NLTK, ``nltk`` and those
submodules are shims providing just these names.

``"warcio"`` works the same way. With warcio installed, ``ArchiveIterator`` (on ``warcio`` and
``warcio.archiveiterator``), ``WARCIterator`` and ``ARCIterator`` (on ``warcio.archiveiterator``)
become subclasses of pyroid's iterators whose records are warcio's own classes: the headers are
warcio's ``StatusAndHeaders``, the record a subclass of warcio's ``ArcWarcRecord`` (with pyroid's
``content_stream()``), the digest checker warcio's ``DigestChecker``, the errors warcio's
``ArchiveLoadFailed`` and ``StatusAndHeadersParserException``, and the warnings go to warcio's
``warcio.recordloader`` logger. datatrove's ``WarcReader`` imports ``ArchiveIterator`` from
``warcio.archiveiterator`` when it opens each file, so a patched process reads WARC files with
pyroid in stock datatrove. Without warcio, ``warcio`` and ``warcio.archiveiterator`` are shims
holding :mod:`pyroid.warc`'s classes.

``"nemo_curator"`` works like ``"nltk"``, and needs NeMo Curator installed: the stages
``ScoreFilter``, ``Score``, ``Filter`` and ``Modify``, the heuristic and fastText filters,
``UnicodeReformatter`` and the string modifiers of :mod:`pyroid.integrations.nemo_curator` are
set on NeMo's own modules (``nemo_curator.stages.text.filters`` and the modules that define
them), so pipelines built afterwards, including from NeMo's YAML configs, use them. Ray workers
need no patch: the stages they receive are pyroid's classes.

``"yaml"`` works like ``"nltk"`` when PyYAML is installed: ``safe_load``, ``safe_load_all``,
``load``, ``load_all``, ``full_load``, ``full_load_all``, ``unsafe_load``, ``unsafe_load_all``,
``safe_dump``, ``safe_dump_all``, ``dump`` and ``dump_all`` on the ``yaml`` module become
:mod:`pyroid.yaml`'s, which reproduce PyYAML 6.0.3's pure-Python ``SafeLoader`` and
``SafeDumper`` and hand everything they do not port (customised loaders and dumpers,
``!!python/`` tags) to PyYAML's own classes. Exceptions stay PyYAML's classes. Without PyYAML,
``yaml`` and its ``error``, ``reader``, ``scanner``, ``parser``, ``composer``, ``constructor``,
``representer``, ``emitter``, ``serializer``, ``resolver``, ``loader`` and ``dumper`` submodules
are shims holding :mod:`pyroid.yaml`'s functions and classes.

Limitations: a shim is not the real module object, so code that already holds the real module (or
compares module objects) is not redirected; ``importlib.reload`` of a shim is not supported.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import os
import pkgutil
import runpy
import sys
import threading
import types
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib.machinery import ModuleSpec
from typing import Any

from pyroid import FallbackWarning, PyroidError

__all__ = [
    "DEFAULT_RUN_NAMES",
    "ENV_VAR",
    "HOOK_DIR",
    "SUPPORTED",
    "PatchWarning",
    "patch",
    "patched",
    "patching",
    "run",
    "unpatch",
]

#: The module names :func:`patch` accepts.
SUPPORTED: tuple[str, ...] = (
    "ftfy",
    "fasttext",
    "anyascii",
    "datasketch",
    "trafilatura",
    "nltk",
    "warcio",
    "nemo_curator",
    "yaml",
)

#: What ``pyroid run`` patches when neither ``-p`` nor ``$PYROID_PATCH`` says.
DEFAULT_RUN_NAMES: tuple[str, ...] = ("ftfy", "fasttext")

#: The environment variable read by ``pyroid run`` and the start-up hook (comma-separated names).
ENV_VAR = "PYROID_PATCH"

#: The directory holding the start-up hook (a ``sitecustomize.py``): put it on ``PYTHONPATH``.
HOOK_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_autopatch")

# Names whose PatchWarning was already shown in this process tree (inherited by children).
_WARNED_ENV = "_PYROID_PATCH_WARNED"


class PatchWarning(UserWarning):
    """:func:`patch` replaced a module with pyroid's drop-in.

    Says what was patched, which version pyroid reproduces and how to undo it; shown once per
    process for each name, plus once more if the installed package is another version. Silence
    it with ``warnings.simplefilter("ignore", pyroid.integrations.PatchWarning)``.
    """


# ---------------------------------------------------------------------------
# Shims
# ---------------------------------------------------------------------------


class _Shim(types.ModuleType):
    """A pyroid stand-in for a third-party module, installed in ``sys.modules`` by patch()."""

    def __dir__(self) -> list[str]:
        names = set(self.__dict__)
        real = self.__dict__.get("__wrapped__")
        if real is not None:
            names.update(dir(real))
        return sorted(names)

    def __repr__(self) -> str:
        real = self.__dict__.get("__wrapped__")
        what = f"wrapping {real!r}" if real is not None else "the real module is not installed"
        return f"<module {self.__name__!r} (pyroid shim; {what})>"


def _spec(name: str, real: Any, *, package: bool, origin: str | None) -> ModuleSpec:
    """The shim's ``__spec__``: a copy of the real module's, else one with no loader."""
    real_spec = getattr(real, "__spec__", None)
    if isinstance(real_spec, ModuleSpec):
        spec = ModuleSpec(
            name,
            real_spec.loader,
            origin=real_spec.origin,
            loader_state=real_spec.loader_state,
            is_package=real_spec.submodule_search_locations is not None,
        )
        spec.submodule_search_locations = real_spec.submodule_search_locations
        spec.has_location = real_spec.has_location
        if real_spec.has_location:
            spec.cached = real_spec.cached
        return spec
    return ModuleSpec(name, None, origin=origin, is_package=package)


def _forwarder(name: str, real: Any, missing: str) -> Callable[[str], Any]:
    """The shim's module ``__getattr__`` (PEP 562: only called when the shim lacks `attr`)."""
    if real is not None:

        def forward(attr: str) -> Any:
            return getattr(real, attr)

    else:

        def forward(attr: str) -> Any:
            raise AttributeError(
                f"module {name!r} has no attribute {attr!r}: pyroid's {name} shim "
                f"(pyroid.integrations.patch) does not provide it, and {missing}",
                name=attr,
            )

    forward.__name__ = forward.__qualname__ = "__getattr__"
    return forward


def _shim(
    name: str,
    real: Any,
    missing: str,
    impl: types.ModuleType,
    provided: dict[str, Any],
    *,
    package: bool,
    version: str | None = None,
) -> _Shim:
    """A shim for `name` providing `provided` and forwarding the rest to `real`."""
    shim = _Shim(
        name,
        f"pyroid's drop-in for {name}, installed by pyroid.integrations.patch().\n\n"
        f"{', '.join(sorted(provided))} come from {impl.__name__}; everything else from the "
        f"real {name} (``__wrapped__``) when it is installed.",
    )
    spec = _spec(name, real, package=package, origin=getattr(impl, "__file__", None))
    ns = shim.__dict__
    ns["__package__"] = name if package else name.rpartition(".")[0]
    ns["__spec__"] = spec
    ns["__loader__"] = spec.loader
    ns["__pyroid__"] = True
    ns["__wrapped__"] = real
    if package:
        locations = spec.submodule_search_locations
        ns["__path__"] = locations if locations is not None else []
    if version is not None:
        ns["__version__"] = version
    ns.update(provided)
    # `from name import *` gets what it gets from the real package (plus pyroid's extras).
    exported = {n for n in provided if not n.startswith("_")}
    if real is not None:
        star = getattr(real, "__all__", None)
        exported.update(star if star is not None else (n for n in vars(real) if n[:1] != "_"))
    ns["__all__"] = sorted(exported)
    ns["__getattr__"] = _forwarder(name, real, missing)
    return shim


# ---------------------------------------------------------------------------
# The supported modules
# ---------------------------------------------------------------------------

_Built = tuple[dict[str, types.ModuleType], list[Callable[[], None]]]


@dataclass(frozen=True)
class _Target:
    impl: str  # the pyroid module providing the drop-in
    reproduces: str  # what it reproduces, for the warning
    version_attr: str | None  # impl attribute: the real __version__ pyroid reproduces
    pip: str  # how to install the real package
    rest: str  # examples of what the real package still provides, for the warning
    build: Callable[[types.ModuleType, Any, str], _Built]
    # impl attribute: the real versions pyroid reproduces as modes (no version warning for them)
    modes_attr: str | None = None
    # the installed version, from the real module (default: its __version__)
    installed: Callable[[Any], str | None] | None = None


def _caller_stacklevel() -> int:
    """The ``stacklevel`` that points a warning here at the first frame outside this module.

    ``pyroid.patch`` (in ``pyroid/__init__.py``) is skipped too, so ``pyroid.patch(...)`` warns
    at the caller's line like ``pyroid.integrations.patch(...)``.
    """
    frame = sys._getframe(1)
    level = 1
    while frame.f_back is not None and frame.f_globals.get("__name__") in (__name__, "pyroid"):
        frame = frame.f_back
        level += 1
    return level


def _build_ftfy(impl: types.ModuleType, real: Any, missing: str) -> _Built:
    # ftfy's helpers that pyroid only delegates (DELEGATED) are the real ones when ftfy is
    # installed (forwarded); without it, pyroid's, which raise PyroidError when called.
    skip = {"FTFY_VERSION", *(impl.DELEGATED if real is not None else ())}
    provided = {n: getattr(impl, n) for n in impl.__all__ if n not in skip}
    shim = _shim("ftfy", real, missing, impl, provided, package=True, version=impl.FTFY_VERSION)
    # pyroid.ftfy's own fallback (`_ftfy`) sees through the shim to `__wrapped__`: nothing to undo.
    return {"ftfy": shim}, []


_FALLBACK_WARNED: set[str] = set()


class _ModelClass(type):
    """Metaclass of the shim's ``_FastText``: ``isinstance()`` is true for pyroid models too."""

    _accepts: tuple[type, ...] = ()

    def __instancecheck__(cls, obj: object) -> bool:
        return isinstance(obj, cls._accepts) or super().__instancecheck__(obj)


def _fasttext_class(impl: types.ModuleType, real_cls: Any, missing: str) -> type:
    """The shim's ``fasttext.FastText._FastText``: ``_FastText(path)`` loads a pyroid model."""
    load = impl.load_model

    def fallback(model_path: Any, args: Any) -> Any:
        what = "args=..." if args is not None else "model_path=None"
        if real_cls is None:
            raise PyroidError(
                f"fasttext.FastText._FastText({what}): pyroid's fasttext shim only loads model "
                f"files (pyroid.fasttext.load_model), and {missing} to fall back to"
            )
        if what not in _FALLBACK_WARNED:
            _FALLBACK_WARNED.add(what)
            warnings.warn(
                f"fasttext.FastText._FastText({what}) is not supported by pyroid.fasttext; "
                "delegating to the installed fasttext (slow path). This warning is shown once.",
                FallbackWarning,
                stacklevel=_caller_stacklevel(),
            )
        return real_cls(model_path, args)

    class _FastText(metaclass=_ModelClass):
        """fastText's model class: ``_FastText(path)`` returns a :class:`pyroid.fasttext.Model`.

        Anything else (no path, or ``args``: training) goes to the real fastText.
        """

        _accepts = (impl.Model,) if real_cls is None else (impl.Model, real_cls)

        def __new__(cls, model_path: Any = None, args: Any = None) -> Any:
            if model_path is not None and args is None:
                return load(model_path)
            return fallback(model_path, args)

    return _FastText


def _build_fasttext(impl: types.ModuleType, real: Any, missing: str) -> _Built:
    real_sub = getattr(real, "FastText", None) if real is not None else None
    if not isinstance(real_sub, types.ModuleType):
        real_sub = sys.modules.get("fasttext.FastText") if real is not None else None
    sub_missing = missing if real is None else "the installed fasttext has no FastText module"
    sub = _shim(
        "fasttext.FastText",
        real_sub,
        sub_missing,
        impl,
        {
            "_FastText": _fasttext_class(impl, getattr(real_sub, "_FastText", None), missing),
            "load_model": impl.load_model,
            "tokenize": impl.tokenize,
        },
        package=False,
    )
    provided = {"FastText": sub, "load_model": impl.load_model, "tokenize": impl.tokenize}
    top = _shim("fasttext", real, missing, impl, provided, package=True)
    return {"fasttext": top, "fasttext.FastText": sub}, []


def _build_anyascii(impl: types.ModuleType, real: Any, missing: str) -> _Built:
    provided = {"anyascii": impl.anyascii}
    version = impl.ANYASCII_VERSION
    shim = _shim("anyascii", real, missing, impl, provided, package=True, version=version)
    return {"anyascii": shim}, []


# datasketch's submodules that pyroid ports, and what each shim provides; the others
# (storage, weighted_minhash, hyperloglog, b_bit_minhash, ...) stay datasketch's own.
_DATASKETCH_SUBMODULES: dict[str, tuple[str, ...]] = {
    "hashfunc": ("sha1_hash32", "sha1_hash64"),
    "minhash": ("MinHash", "minhash_batch"),
    "lean_minhash": ("LeanMinHash",),
    "lsh": ("MinHashLSH", "MinHashLSHInsertionSession", "MinHashLSHDeletionSession"),
    "lshforest": ("MinHashLSHForest",),
    "lshensemble": ("MinHashLSHEnsemble",),
}


def _build_datasketch(impl: types.ModuleType, real: Any, missing: str) -> _Built:
    shims: dict[str, types.ModuleType] = {}
    for sub, names in _DATASKETCH_SUBMODULES.items():
        sub_impl = importlib.import_module(f"{impl.__name__}.{sub}")
        full = f"datasketch.{sub}"
        real_sub = sys.modules.get(full) if real is not None else None
        sub_missing = missing if real is None else f"the installed datasketch has no {sub} module"
        provided = {n: getattr(sub_impl, n) for n in names}
        shims[full] = _shim(full, real_sub, sub_missing, sub_impl, provided, package=False)
    provided = {n: getattr(impl, n) for n in impl.__all__ if n != "DATASKETCH_VERSION"}
    provided.update({sub: shims[f"datasketch.{sub}"] for sub in _DATASKETCH_SUBMODULES})
    top = _shim(
        "datasketch", real, missing, impl, provided, package=True, version=impl.DATASKETCH_VERSION
    )
    # pyroid.datasketch's own fallback (`_common.fallback`) sees through the shim to `__wrapped__`.
    return {"datasketch": top, **shims}, []


def _build_trafilatura(impl: types.ModuleType, real: Any, missing: str) -> _Built:
    # The mode is the installed trafilatura's version when pyroid reproduces it, else the
    # default (_announce warns). The functions have that version's exact signature (no compat=).
    found = getattr(real, "__version__", None) if real is not None else None
    mode = found if found in impl.TRAFILATURA_VERSIONS else impl.TRAFILATURA_VERSION
    # Their config= default is the real trafilatura.settings.DEFAULT_CONFIG (of that version).
    settings = getattr(real, "settings", None) if found == mode else None
    config = getattr(settings, "DEFAULT_CONFIG", None)
    api = impl._api(mode, config)

    def real_sub(name: str) -> Any:
        if real is None:
            return None
        module = sys.modules.get(f"trafilatura.{name}")
        if module is None:
            try:
                module = importlib.import_module(f"trafilatura.{name}")
            except ImportError:
                return None
        return module

    def sub(name: str, provided: dict[str, Any]) -> _Shim:
        module = real_sub(name)
        sub_missing = missing if real is None else f"the installed trafilatura has no {name} module"
        return _shim(f"trafilatura.{name}", module, sub_missing, impl, provided, package=False)

    real_reset = getattr(real_sub("meta"), "reset_caches", None)

    def reset_caches() -> None:
        """``trafilatura.meta.reset_caches()``, emptying pyroid's dedup cache too."""
        impl.reset_caches()
        if real_reset is not None:
            real_reset()

    shims = {
        "trafilatura.core": sub("core", dict(api)),
        "trafilatura.deduplication": sub(
            "deduplication", {"LRUCache": impl.LRUCache, "LRU_TEST": impl.LRU_TEST}
        ),
        "trafilatura.meta": sub("meta", {"reset_caches": reset_caches}),
    }
    provided = {**api, **{key.rpartition(".")[2]: shim for key, shim in shims.items()}}
    top = _shim("trafilatura", real, missing, impl, provided, package=True, version=mode)
    # pyroid.trafilatura's own fallback (`_real_module`) sees through the shim to `__wrapped__`.
    return {"trafilatura": top, **shims}, []


# What patch("nltk") swaps: pyroid.nltk's names, set on NLTK's own modules (by module).
_NLTK_SWAPS: dict[str, tuple[str, ...]] = {
    "nltk": (
        "sent_tokenize",
        "word_tokenize",
        "punkt_model_available",
        "PunktSentenceTokenizer",
        "PunktTokenizer",
        "TreebankWordTokenizer",
        "TreebankWordDetokenizer",
        "NLTKWordTokenizer",
        "wordpunct_tokenize",
        "WordPunctTokenizer",
        "RegexpTokenizer",
        "WhitespaceTokenizer",
        "BlanklineTokenizer",
        "regexp_tokenize",
        "blankline_tokenize",
    ),
    "nltk.tokenize": (
        "sent_tokenize",
        "word_tokenize",
        "punkt_model_available",
        "PunktSentenceTokenizer",
        "PunktTokenizer",
        "TreebankWordTokenizer",
        "TreebankWordDetokenizer",
        "NLTKWordTokenizer",
        "wordpunct_tokenize",
        "WordPunctTokenizer",
        "RegexpTokenizer",
        "WhitespaceTokenizer",
        "BlanklineTokenizer",
        "regexp_tokenize",
        "blankline_tokenize",
    ),
    "nltk.tokenize.punkt": ("PunktSentenceTokenizer", "PunktTokenizer"),
    "nltk.tokenize.regexp": (
        "RegexpTokenizer",
        "WhitespaceTokenizer",
        "BlanklineTokenizer",
        "WordPunctTokenizer",
        "regexp_tokenize",
        "blankline_tokenize",
        "wordpunct_tokenize",
    ),
    "nltk.tokenize.treebank": ("TreebankWordTokenizer", "TreebankWordDetokenizer"),
    "nltk.tokenize.destructive": ("NLTKWordTokenizer",),
}


def _build_nltk(impl: types.ModuleType, real: Any, missing: str) -> _Built:
    from pyroid.nltk import _delegate

    if real is None:  # no NLTK: shims providing pyroid.nltk's names
        shims: dict[str, types.ModuleType] = {}
        for mod, names in reversed(_NLTK_SWAPS.items()):
            provided: dict[str, Any] = {n: getattr(impl, n) for n in names}
            provided.update(
                {k.rpartition(".")[2]: v for k, v in shims.items() if k.rpartition(".")[0] == mod}
            )
            shims[mod] = _shim(mod, None, missing, impl, provided, package=mod.count(".") < 2)
        return shims, []
    # NLTK installed: set pyroid's objects on NLTK's modules, keeping the originals for the
    # fallbacks (pyroid.nltk._delegate.ORIGINALS) and for unpatch().
    saved: list[tuple[types.ModuleType, str, Any]] = []
    for mod, names in _NLTK_SWAPS.items():
        module = importlib.import_module(mod)
        for n in names:
            original = module.__dict__.get(n, _MISSING)
            saved.append((module, n, original))
            if original is not _MISSING:
                _delegate.ORIGINALS.setdefault(f"{mod}:{n}", original)
            setattr(module, n, getattr(impl, n))
    tokenizer_i = getattr(importlib.import_module("nltk.tokenize.api"), "TokenizerI", None)
    if isinstance(tokenizer_i, type) and hasattr(tokenizer_i, "register"):
        for cls in (
            "PunktSentenceTokenizer",
            "TreebankWordTokenizer",
            "TreebankWordDetokenizer",
            "NLTKWordTokenizer",
            "RegexpTokenizer",
        ):
            tokenizer_i.register(getattr(impl, cls))  # isinstance(tok, TokenizerI) stays true

    def undo() -> None:
        for module, n, original in reversed(saved):
            if original is _MISSING:
                module.__dict__.pop(n, None)
            else:
                setattr(module, n, original)
        _delegate.ORIGINALS.clear()
        # NLTK's lru-cached PunktTokenizer per language may hold tokenizers made while patched.
        cache_clear = getattr(
            getattr(sys.modules.get("nltk.tokenize"), "_get_punkt_tokenizer", None),
            "cache_clear",
            None,
        )
        if cache_clear is not None:
            cache_clear()

    return {}, [undo]


# What patch("warcio") swaps on warcio's own modules (warcio 1.8.2's names; warcio itself only
# re-exports ArchiveIterator).
_WARCIO_SWAPS: dict[str, tuple[str, ...]] = {
    "warcio": ("ArchiveIterator",),
    "warcio.archiveiterator": ("ArchiveIterator", "WARCIterator", "ARCIterator"),
}
# pyroid's iterator classes over warcio's record, header and exception classes: built once per
# process, so iterators from an earlier patch() stay instances of the patched names.
_WARCIO_CLASSES: dict[str, type] = {}


def _warcio_classes(impl: types.ModuleType) -> dict[str, type]:
    """``ArchiveIterator``, ``WARCIterator``, ``ARCIterator`` making warcio's own records."""
    if _WARCIO_CLASSES:
        return _WARCIO_CLASSES
    from warcio import (  # type: ignore[import-not-found,import-untyped,unused-ignore]
        digestverifyingreader,
        exceptions,
        recordloader,
        statusandheaders,
    )

    # type() takes __module__ (this module) and __qualname__ (the name) from here.
    record = type(
        "ArcWarcRecord",
        (recordloader.ArcWarcRecord,),
        {
            "__doc__": "warcio's ``ArcWarcRecord`` with pyroid's ``content_stream()``.",
            "content_stream": impl.ArcWarcRecord.content_stream,
        },
    )
    base = type(
        "ArchiveIterator",
        (impl.ArchiveIterator,),
        {
            "__doc__": "warcio's ``ArchiveIterator`` (same arguments) on "
            ":class:`pyroid.warc.ArchiveIterator`, making warcio's own records.",
            "record_cls": record,
            "headers_cls": statusandheaders.StatusAndHeaders,
            "digest_checker_cls": digestverifyingreader.DigestChecker,
            "archive_load_failed": exceptions.ArchiveLoadFailed,
            "parser_exception": statusandheaders.StatusAndHeadersParserException,
            "logger": recordloader.logger,
        },
    )
    _WARCIO_CLASSES["ArcWarcRecord"] = record
    _WARCIO_CLASSES["ArchiveIterator"] = base
    for name, fmt in (("WARCIterator", "warc"), ("ARCIterator", "arc")):
        doc = f"warcio's ``{name}`` on pyroid's iterator."
        _WARCIO_CLASSES[name] = type(name, (base,), {"__doc__": doc, "_known_format": fmt})
    return _WARCIO_CLASSES


def _build_warcio(impl: types.ModuleType, real: Any, missing: str) -> _Built:
    if real is None:  # no warcio: shims providing pyroid.warc's classes
        sub = _shim(
            "warcio.archiveiterator",
            None,
            missing,
            impl,
            {
                "ArchiveIterator": impl.ArchiveIterator,
                "WARCIterator": impl.WARCIterator,
                "ARCIterator": impl.ARCIterator,
                "ArchiveLoadFailed": impl.ArchiveLoadFailed,
            },
            package=False,
        )
        top = _shim(
            "warcio",
            None,
            missing,
            impl,
            {
                "ArchiveIterator": impl.ArchiveIterator,
                "StatusAndHeaders": impl.StatusAndHeaders,
                "archiveiterator": sub,
            },
            package=True,
        )
        return {"warcio": top, "warcio.archiveiterator": sub}, []
    classes = _warcio_classes(impl)
    saved: list[tuple[types.ModuleType, str, Any]] = []
    for mod, names in _WARCIO_SWAPS.items():
        module = importlib.import_module(mod)
        for n in names:
            saved.append((module, n, module.__dict__.get(n, _MISSING)))
            setattr(module, n, classes[n])

    def undo() -> None:
        for module, n, original in reversed(saved):
            if original is _MISSING:
                module.__dict__.pop(n, None)
            else:
                setattr(module, n, original)

    return {}, [undo]


# What patch("nemo_curator") swaps: pyroid.integrations.nemo_curator's classes, set on NeMo
# Curator's own modules (where each class is defined and where it is re-exported).
_NEMO_STRING_FILTERS = (
    "BoilerPlateStringFilter",
    "BulletsFilter",
    "CommonEnglishWordsFilter",
    "EllipsisFilter",
    "LongWordFilter",
    "MeanWordLengthFilter",
    "NonAlphaNumericFilter",
    "NumbersFilter",
    "ParenthesesFilter",
    "PornographicUrlsFilter",
    "PunctuationFilter",
    "SubstringFilter",
    "SymbolsToWordsFilter",
    "UrlsFilter",
    "WhiteSpaceFilter",
    "WordCountFilter",
    "WordsWithoutAlphabetsFilter",
)
_NEMO_REPETITION_FILTERS = (
    "RepeatedLinesByCharFilter",
    "RepeatedLinesFilter",
    "RepeatedParagraphsByCharFilter",
    "RepeatedParagraphsFilter",
    "RepeatingDuplicateNGramsFilter",
    "RepeatingTopNGramsFilter",
)
_NEMO_STRING_MODIFIERS = {
    "c4": "BoilerPlateStringModifier",
    "line_remover": "LineRemover",
    "markdown_remover": "MarkdownRemover",
    "newline_normalizer": "NewlineNormalizer",
    "quotation_remover": "QuotationRemover",
    "slicer": "Slicer",
    "url_remover": "UrlRemover",
}
_T = "nemo_curator.stages.text"
_NEMO_SWAPS: dict[str, tuple[str, ...]] = {
    f"{_T}.filters": ("ScoreFilter", "Score", "Filter"),
    f"{_T}.filters.score_filter": ("ScoreFilter", "Score", "Filter"),
    f"{_T}.filters.heuristic": _NEMO_STRING_FILTERS,
    f"{_T}.filters.heuristic.string": _NEMO_STRING_FILTERS,
    f"{_T}.filters.heuristic.repetition": _NEMO_REPETITION_FILTERS,
    f"{_T}.filters.heuristic.repetition.repetition": _NEMO_REPETITION_FILTERS,
    f"{_T}.filters.fasttext": ("FastTextLangId", "FastTextQualityFilter"),
    f"{_T}.filters.fasttext.fasttext_filters": ("FastTextLangId", "FastTextQualityFilter"),
    f"{_T}.modifiers": ("Modify", *_NEMO_STRING_MODIFIERS.values()),
    f"{_T}.modifiers.modifier": ("Modify",),
    f"{_T}.modifiers.string": tuple(_NEMO_STRING_MODIFIERS.values()),
    **{f"{_T}.modifiers.string.{m}": (c,) for m, c in _NEMO_STRING_MODIFIERS.items()},
    f"{_T}.modifiers.unicode": ("UnicodeReformatter",),
    f"{_T}.modifiers.unicode.unicode_reformatter": ("UnicodeReformatter",),
    f"{_T}.modifiers.fasttext": ("FastTextLabelModifier",),
    f"{_T}.modifiers.fasttext.fasttext_label": ("FastTextLabelModifier",),
}
del _T


def _build_nemo_curator(impl: types.ModuleType, real: Any, missing: str) -> _Built:
    if real is None:
        raise ImportError(f"patch('nemo_curator') needs NeMo Curator: {missing}")
    saved: list[tuple[types.ModuleType, str, Any]] = []
    added: list[tuple[str, Any]] = []
    for mod, names in _NEMO_SWAPS.items():
        try:
            module = importlib.import_module(mod)
        except ImportError:
            # NeMo's fastText and ftfy modules import fasttext / ftfy at the top; pyroid's classes
            # need neither, so a module named like NeMo's provides them (for YAML `_target_`s).
            module = types.ModuleType(mod, f"pyroid's drop-ins for {mod} (pyroid.patch)")
            added.append((mod, sys.modules.get(mod, _MISSING)))
            sys.modules[mod] = module
        for n in names:
            saved.append((module, n, module.__dict__.get(n, _MISSING)))
            setattr(module, n, getattr(impl, n))

    def undo() -> None:
        for module, n, original in reversed(saved):
            if original is _MISSING:
                module.__dict__.pop(n, None)
            else:
                setattr(module, n, original)
        for mod, previous in reversed(added):
            if previous is _MISSING:
                sys.modules.pop(mod, None)
            else:
                sys.modules[mod] = previous

    return {}, [undo]


def _distribution_version(name: str) -> Callable[[Any], str | None]:
    """The installed version of distribution `name` (for packages without ``__version__``)."""

    def installed(real: Any) -> str | None:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(name)
        except PackageNotFoundError:
            return None

    return installed


# What patch("yaml") sets on PyYAML's `yaml` module.
_YAML_SWAPS = (
    "safe_load",
    "safe_load_all",
    "load",
    "load_all",
    "full_load",
    "full_load_all",
    "unsafe_load",
    "unsafe_load_all",
    "safe_dump",
    "safe_dump_all",
    "dump",
    "dump_all",
)
# The shims without PyYAML: submodule -> the pyroid.yaml names it holds.
_YAML_SHIMS: dict[str, tuple[str, ...]] = {
    "yaml.error": ("Mark", "YAMLError", "MarkedYAMLError"),
    "yaml.reader": ("ReaderError",),
    "yaml.scanner": ("ScannerError",),
    "yaml.parser": ("ParserError",),
    "yaml.composer": ("ComposerError",),
    "yaml.constructor": ("ConstructorError",),
    "yaml.representer": ("RepresenterError",),
    "yaml.emitter": ("EmitterError",),
    "yaml.serializer": ("SerializerError",),
    "yaml.resolver": ("ResolverError",),
    "yaml.loader": ("BaseLoader", "FullLoader", "SafeLoader", "Loader", "UnsafeLoader"),
    "yaml.dumper": ("SafeDumper", "Dumper"),
}


def _build_yaml(impl: types.ModuleType, real: Any, missing: str) -> _Built:
    if real is None:  # no PyYAML: shims holding pyroid.yaml's names
        shims: dict[str, types.ModuleType] = {}
        for mod, names in _YAML_SHIMS.items():
            shims[mod] = _shim(
                mod, None, missing, impl, {n: getattr(impl, n) for n in names}, package=False
            )
        provided: dict[str, Any] = {n: getattr(impl, n) for n in impl.__all__}
        provided.update({k.rpartition(".")[2]: v for k, v in shims.items()})
        provided["__with_libyaml__"] = False
        shims["yaml"] = _shim(
            "yaml", None, missing, impl, provided, package=True, version=impl.PYYAML_VERSION
        )
        return shims, []
    saved: list[tuple[Any, str, Any]] = []
    for n in _YAML_SWAPS:
        saved.append((real, n, real.__dict__.get(n, _MISSING)))
        setattr(real, n, getattr(impl, n))

    def undo() -> None:
        for module, n, original in reversed(saved):
            if original is _MISSING:
                module.__dict__.pop(n, None)
            else:
                setattr(module, n, original)

    return {}, [undo]


_TARGETS: dict[str, _Target] = {
    "ftfy": _Target(
        "pyroid.ftfy",
        "which reproduces ftfy 6.3.1",
        "FTFY_VERSION",
        "pip install ftfy==6.3.1",
        " (apply_plan, explain_unicode, fix_file, guess_bytes, ftfy.fixes and the other"
        " submodules)",
        _build_ftfy,
    ),
    "fasttext": _Target(
        "pyroid.fasttext",
        "fastText 0.9.2's inference, as fasttext-predict 0.9.2.4: load_model, tokenize and "
        "FastText._FastText(path) give pyroid models",
        None,
        "pip install fasttext-predict",
        " (training, ...)",
        _build_fasttext,
    ),
    "anyascii": _Target(
        "pyroid.translit",
        "anyascii(), which reproduces anyascii 0.3.3",
        "ANYASCII_VERSION",
        "pip install anyascii==0.3.3",
        "",
        _build_anyascii,
    ),
    "datasketch": _Target(
        "pyroid.datasketch",
        "which reproduces datasketch 2.0.0's MinHash, LeanMinHash, MinHashLSH, MinHashLSHForest "
        "and MinHashLSHEnsemble",
        "DATASKETCH_VERSION",
        "pip install datasketch==2.0.0",
        " (WeightedMinHash, HyperLogLog, bBitMinHash, HNSW, MinHashLSHBloom, Redis and Cassandra"
        " storage)",
        _build_datasketch,
    ),
    "trafilatura": _Target(
        "pyroid.trafilatura",
        "which reproduces the extract and bare_extraction of trafilatura 2.3.1 or 1.11.0, "
        "following the installed release; options it does not port go to the installed one",
        "TRAFILATURA_VERSION",
        "pip install trafilatura==2.3.1",
        " (metadata extraction, downloads, feeds, sitemaps, the spider, the command line and the "
        "other submodules)",
        _build_trafilatura,
        modes_attr="TRAFILATURA_VERSIONS",
    ),
    "nltk": _Target(
        "pyroid.nltk",
        "NLTK 3.10.4's sent_tokenize, word_tokenize, PunktSentenceTokenizer, PunktTokenizer, "
        "TreebankWordTokenizer, NLTKWordTokenizer, TreebankWordDetokenizer and the regexp "
        "tokenizers",
        "NLTK_VERSION",
        "pip install nltk==3.10.4",
        " (training, tagging, corpora, the other tokenizers)",
        _build_nltk,
    ),
    "warcio": _Target(
        "pyroid.warc",
        "which reproduces warcio 1.8.2's ArchiveIterator, WARCIterator and ARCIterator",
        "WARCIO_VERSION",
        "pip install warcio==1.8.2",
        " (WARCWriter, capture_http, the indexer, the checker, the command line)",
        _build_warcio,
        installed=_distribution_version("warcio"),
    ),
    "nemo_curator": _Target(
        "pyroid.integrations.nemo_curator",
        "whose ScoreFilter, Score, Filter and Modify stages, heuristic filters, fastText filters, "
        "UnicodeReformatter and string modifiers reproduce NeMo Curator 1.4.0's",
        "NEMO_CURATOR_VERSION",
        "pip install nemo-curator==1.4.0",
        " (readers, writers, deduplication, classifiers, the other stages)",
        _build_nemo_curator,
    ),
    "yaml": _Target(
        "pyroid.yaml",
        "which reproduces PyYAML 6.0.3's safe_load, safe_load_all, safe_dump and safe_dump_all, "
        "and load/dump with the stock loaders and dumpers",
        "PYYAML_VERSION",
        "pip install pyyaml==6.0.3",
        " (customised loaders and dumpers, !!python/ tags, the event and token APIs, YAMLObject)",
        _build_yaml,
    ),
}

# Names patched by setting pyroid's objects on the installed package's own modules (no shims).
_SWAPPED = ("nltk", "warcio", "nemo_curator", "yaml")


# ---------------------------------------------------------------------------
# patch / unpatch
# ---------------------------------------------------------------------------


@dataclass
class _Active:
    shims: dict[str, types.ModuleType]
    saved: dict[str, Any] | None  # sys.modules entries for name and name.* (None: not touched)
    undo: list[Callable[[], None]] = field(default_factory=list)


_LOCK = threading.RLock()
_ACTIVE: dict[str, _Active] = {}
_WARNED: set[str] = set()
_MISSING = object()


def _owned(key: str, name: str) -> bool:
    return key == name or key.startswith(name + ".")


def _real(name: str) -> tuple[Any, str]:
    """The real module `name`, importing it if needed; or ``None`` and why it is missing."""
    current = sys.modules.get(name, _MISSING)
    if current is None:
        return None, f"importing {name} is blocked (sys.modules[{name!r}] is None)"
    if current is not _MISSING:
        return current, ""
    try:
        return importlib.import_module(name), ""
    except ModuleNotFoundError as exc:
        if exc.name == name:
            return None, f"{name} is not installed"
        return None, f"{name} failed to import ({type(exc).__name__}: {exc})"
    except Exception as exc:  # a broken install: the shim works without it
        return None, f"{name} failed to import ({type(exc).__name__}: {exc})"


def _check(names: Sequence[str]) -> list[str]:
    """`names` without duplicates; ValueError for one that cannot be patched."""
    out: list[str] = []
    for name in names:
        if name not in _TARGETS:
            hint = ""
            if name == "unidecode":
                hint = (
                    " (pyroid.translit.unidecode transliterates with anyascii's tables, so its "
                    "output differs from Unidecode's: it is not a drop-in)"
                )
            supported = ", ".join(map(repr, SUPPORTED))
            raise ValueError(f"cannot patch {name!r}: the supported names are {supported}{hint}")
        if name not in out:
            out.append(name)
    return out


def _announce(name: str, impl: types.ModuleType, real: Any, missing: str) -> None:
    """The once-per-process PatchWarning(s) for `name`."""
    target = _TARGETS[name]
    if name not in _WARNED:
        _WARNED.add(name)
        what = f"`import {name}` now gives pyroid's drop-in"
        if real is None:
            rest = f"Everything else{target.rest} is unavailable: {missing} ({target.pip})."
        elif name in _SWAPPED:
            what = f"the installed {name} now uses pyroid's drop-ins"
            example = {
                "nltk": "from nltk import word_tokenize",
                "warcio": "from warcio.archiveiterator import ArchiveIterator",
                "nemo_curator": "from nemo_curator.stages.text.filters import ScoreFilter",
                "yaml": "from yaml import safe_load",
            }[name]
            rest = (
                f"Everything else{target.rest} is still {name}'s own. Names imported from "
                f"{name} before this call ({example}) keep {name}'s objects."
            )
        else:
            rest = (
                f"Everything else{target.rest} still comes from the installed {name}. Modules "
                f"that imported {name} before this call keep the real one."
            )
        warnings.warn(
            f"pyroid.integrations.patch: {what} "
            f"({target.impl}, {target.reproduces}). {rest} Undo with "
            f"pyroid.integrations.unpatch({name!r}). This warning is shown once.",
            PatchWarning,
            stacklevel=_caller_stacklevel(),
        )
    expected = getattr(impl, target.version_attr) if target.version_attr else None
    if real is None:
        found = None
    elif target.installed is not None:
        found = target.installed(real)
    else:
        found = getattr(real, "__version__", None)
    modes = getattr(impl, target.modes_attr) if target.modes_attr else ()
    if real is not None and expected is not None and found != expected and found not in modes:
        key = f"{name}:version"
        if key not in _WARNED:
            _WARNED.add(key)
            warnings.warn(
                f"pyroid.integrations.patch: the installed {name} is {found or 'unversioned'}, "
                f"but {target.impl} reproduces {name} {expected}: the patched functions give "
                f"{name} {expected}'s results, the rest is {name} {found or '(unversioned)'}.",
                PatchWarning,
                stacklevel=_caller_stacklevel(),
            )


def _install(name: str, *, warn: bool) -> None:
    """Patch one (checked) name; idempotent."""
    with _LOCK:
        active = _ACTIVE.get(name)
        if active is not None:  # already patched: put the shims back if someone displaced them
            for key, module in active.shims.items():
                if sys.modules.get(key) is not module:
                    sys.modules[key] = module
            return
        target = _TARGETS[name]
        impl = importlib.import_module(target.impl)
        saved = {k: v for k, v in sys.modules.copy().items() if _owned(k, name)}
        real, missing = _real(name)
        try:
            shims, undo = target.build(impl, real, missing)
        except BaseException:
            _restore(name, saved, [])
            raise
        sys.modules.update(shims)  # the other runtime changes (undo) are already in place
        # Attributes swapped on the real modules (no shims): sys.modules stays as it is.
        _ACTIVE[name] = _Active(shims, saved if shims else None, undo)
        if warn:
            _announce(name, impl, real, missing)


def _restore(name: str, saved: dict[str, Any] | None, undo: list[Callable[[], None]]) -> None:
    if saved is None:
        for step in reversed(undo):
            step()
        return
    for key in sys.modules.copy():
        if _owned(key, name) and key not in saved:
            sys.modules.pop(key, None)
    sys.modules.update(saved)
    for step in reversed(undo):  # after sys.modules: the fallback never sees the shim
        step()


def patch(*names: str) -> None:
    """Make ``import <name>`` give pyroid's drop-in, for each of `names`.

    ``patch("ftfy", "fasttext")`` puts shims in ``sys.modules``, so a later ``import ftfy``,
    ``from ftfy import fix_text``, ``import fasttext.FastText``... gets pyroid's functions, with
    everything pyroid does not provide forwarded to the real package (imported now, if it is
    installed). Modules that imported the real package earlier keep it. Patching a patched name
    does nothing; :func:`unpatch` undoes it.

    Args:
        *names: ``"ftfy"``, ``"fasttext"``, ``"anyascii"``, ``"datasketch"``,
            ``"trafilatura"``, ``"nltk"``, ``"warcio"`` and/or ``"nemo_curator"``.

    Raises:
        TypeError: No names were given.
        ValueError: A name is not supported (nothing is patched then).
        ImportError: The pyroid module behind a name cannot be imported (``pyroid.fasttext``
            and ``pyroid.datasketch`` need numpy).

    Warns:
        PatchWarning: Once per process for each name: what was patched and how to undo it;
            and if the installed package is not the version pyroid reproduces.
    """
    if not names:
        raise TypeError(f"patch() needs at least one name: {', '.join(map(repr, SUPPORTED))}")
    checked = _check(names)
    for name in checked:  # fail before patching anything
        importlib.import_module(_TARGETS[name].impl)
    for name in checked:
        _install(name, warn=True)


def unpatch(*names: str) -> None:
    """Undo :func:`patch` for `names` (all patched names if none are given).

    ``sys.modules`` gets back exactly the entries it had for each name and its submodules
    before :func:`patch` (the real module if it was imported, else no entry; submodules
    imported meanwhile are dropped, so the next import is fresh), and pyroid's fallback to the
    real package is restored. Modules that imported a shim keep it, and it keeps working.
    Names that are not patched are ignored.

    Raises:
        ValueError: A name is not supported.
    """
    with _LOCK:
        todo = _check(names) if names else list(_ACTIVE)
        for name in reversed(todo):
            active = _ACTIVE.pop(name, None)
            if active is not None:
                _restore(name, active.saved, active.undo)


def patched() -> tuple[str, ...]:
    """The names currently patched, in :data:`SUPPORTED` order."""
    return tuple(name for name in SUPPORTED if name in _ACTIVE)


class _Patching(contextlib.AbstractContextManager[None]):
    def __init__(self, names: list[str]) -> None:
        self._names = names
        self._mine: list[str] = []

    def __enter__(self) -> None:
        with _LOCK:
            self._mine = [n for n in self._names if n not in _ACTIVE]
            patch(*self._names)

    def __exit__(self, *exc_info: object) -> None:
        if self._mine:
            unpatch(*self._mine)


def patching(*names: str) -> contextlib.AbstractContextManager[None]:
    """:func:`patch` `names` for the duration of a ``with`` block.

    On exit, only the names this block patched are unpatched (one patched before stays).

    Raises:
        TypeError: No names were given.
        ValueError: A name is not supported.
    """
    if not names:
        raise TypeError(f"patching() needs at least one name: {', '.join(map(repr, SUPPORTED))}")
    return _Patching(_check(names))


# ---------------------------------------------------------------------------
# Environment activation: pyroid run, and the sitecustomize hook
# ---------------------------------------------------------------------------


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _apply(names: Sequence[str], *, lenient: bool) -> list[str]:
    """Patch checked `names`, warning once per process tree; return the names patched.

    ``lenient``: skip (with a PatchWarning) a name whose pyroid module cannot be imported.
    """
    warned = set(_split(os.environ.get(_WARNED_ENV, "")))
    done: list[str] = []
    for name in names:
        try:
            importlib.import_module(_TARGETS[name].impl)
        except ImportError as exc:
            if not lenient:
                raise
            warnings.warn(
                f"pyroid: not patching {name}: {exc}", PatchWarning, stacklevel=_caller_stacklevel()
            )
            continue
        _install(name, warn=name not in warned)
        done.append(name)
    os.environ[_WARNED_ENV] = ",".join(n for n in SUPPORTED if n in warned or n in done)
    return done


def _autopatch() -> None:
    """Patch the names in ``$PYROID_PATCH`` (the start-up hook calls this)."""
    names = _split(os.environ.get(ENV_VAR, ""))
    unknown = [n for n in names if n not in _TARGETS]
    if unknown:
        warnings.warn(
            f"pyroid: ignoring {', '.join(map(repr, unknown))} in ${ENV_VAR}: the supported "
            f"names are {', '.join(map(repr, SUPPORTED))}",
            PatchWarning,
            stacklevel=_caller_stacklevel(),
        )
    _apply(_check([n for n in names if n in _TARGETS]), lenient=True)


def _export(names: Sequence[str]) -> None:
    """Make the Python processes started from here patch `names` too."""
    os.environ[ENV_VAR] = ",".join(names)
    parts = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if HOOK_DIR not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([HOOK_DIR, *parts])


_RUN_USAGE = "usage: pyroid run [-p NAMES] (-m MODULE | SCRIPT) [ARGS...]\n"
_RUN_HELP = f"""{_RUN_USAGE}
Run a Python script or module, as `python SCRIPT` / `python -m MODULE` would, with
`import ftfy` / `fasttext` / `anyascii` / `datasketch` / `trafilatura` / `nltk` /
`warcio` / `nemo_curator` giving pyroid's drop-ins. The Python processes it starts
(multiprocessing workers, subprocesses) are patched too.

options:
  -p, --patch NAMES  comma-separated modules to patch: {", ".join(SUPPORTED)}
                     (default: ${ENV_VAR}, else {",".join(DEFAULT_RUN_NAMES)})
  -m MODULE          run a library module as a script
  -h, --help         show this help and exit
"""


class _UsageError(Exception):
    pass


_Parsed = tuple[str | None, str | None, str | None, list[str]]


def _parse_run(argv: Sequence[str]) -> _Parsed | None:
    """``(names, module, script, args)``, or None for ``--help``; options stop at the target."""
    names: str | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-h", "--help"):
            return None
        if arg in ("-p", "--patch"):
            if i + 1 == len(argv):
                raise _UsageError(f"argument {arg}: expected a comma-separated list of names")
            names = argv[i + 1]
            i += 2
        elif arg.startswith("--patch="):
            names = arg.partition("=")[2]
            i += 1
        elif arg.startswith("-p"):
            names = arg[2:]
            i += 1
        elif arg.startswith("-m"):
            module = arg[2:] or (argv[i + 1] if i + 1 < len(argv) else "")
            if not module:
                raise _UsageError("argument -m: expected a module name")
            return names, module, None, list(argv[i + (1 if arg[2:] else 2) :])
        elif arg == "--":
            if i + 1 == len(argv):
                raise _UsageError("no script given after --")
            return names, None, argv[i + 1], list(argv[i + 2 :])
        elif arg.startswith("-"):
            raise _UsageError(f"unrecognized option {arg!r} (reading stdin is not supported)")
        else:
            return names, None, arg, list(argv[i + 1 :])
    raise _UsageError("give a SCRIPT or -m MODULE to run")


def _show_run_warning(message: Warning | str, category: type[Warning], *_: Any, **__: Any) -> None:
    sys.stderr.write(f"pyroid run: {category.__name__}: {message}\n")


def run(argv: Sequence[str]) -> int:
    """``pyroid run [-p NAMES] (-m MODULE | SCRIPT) [ARGS...]``: run a target with patches.

    Patches ``NAMES`` (comma-separated; default ``$PYROID_PATCH``, else ``ftfy,fasttext``,
    skipping ``fasttext`` with a :class:`PatchWarning` if numpy is missing), exports
    ``PYROID_PATCH`` and puts :data:`HOOK_DIR` first on ``PYTHONPATH`` (so child Python
    processes are patched too), then runs the target in this process with :mod:`runpy`, as
    ``python -m MODULE ARGS`` / ``python SCRIPT ARGS`` would: ``sys.argv``, ``sys.path[0]``,
    ``__name__ == "__main__"``. It is meant to be the last thing a process does: the patches,
    the environment and ``sys.argv`` stay as they are afterwards.

    Args:
        argv: The arguments after ``pyroid run``.

    Returns:
        0 when the target finishes; 2 for a usage error (message on stderr) or a missing
        script; 1 when the module cannot be found (as ``python``).

    Raises:
        SystemExit: The target exited (``sys.exit``); its code is the exit code.
        BaseException: Whatever else the target raises propagates, traceback and all.
    """
    try:
        parsed = _parse_run(argv)
    except _UsageError as exc:
        sys.stderr.write(f"{_RUN_USAGE}pyroid run: error: {exc}\n")
        return 2
    if parsed is None:
        sys.stdout.write(_RUN_HELP)
        return 0
    names_arg, module, script, args = parsed
    lenient = False
    if names_arg is not None:
        value, source = names_arg, "-p"
    elif os.environ.get(ENV_VAR, "").strip():
        value, source = os.environ[ENV_VAR], f"${ENV_VAR}"
    else:
        value, source, lenient = ",".join(DEFAULT_RUN_NAMES), "the default", True
    try:
        names = _check(_split(value))
    except ValueError as exc:
        sys.stderr.write(f"{_RUN_USAGE}pyroid run: error: {source}: {exc}\n")
        return 2
    if not names:
        sys.stderr.write(f"{_RUN_USAGE}pyroid run: error: {source}: no module names given\n")
        return 2
    if script is not None and not os.path.exists(script):
        path = os.path.abspath(script)
        sys.stderr.write(f"pyroid run: can't open file {path!r}: No such file or directory\n")
        return 2
    try:
        # The PatchWarnings have no line of the user's to point at (pyroid run is the caller):
        # show them as plain lines. The warning filters (-W, PYTHONWARNINGS) still apply.
        with warnings.catch_warnings():
            warnings.showwarning = _show_run_warning
            done = _apply(names, lenient=lenient)
    except ImportError as exc:
        sys.stderr.write(f"pyroid run: error: cannot patch: {exc}\n")
        return 2
    _export(done)
    if module is not None:
        try:
            found = importlib.util.find_spec(module)
        except (ImportError, ValueError) as exc:
            sys.stderr.write(
                f"pyroid run: error while finding module specification for {module!r} "
                f"({type(exc).__name__}: {exc})\n"
            )
            return 1
        if found is None:
            sys.stderr.write(f"pyroid run: No module named {module}\n")
            return 1
        sys.argv[:] = [module, *args]  # runpy sets argv[0] to the module's file
        if not sys.flags.safe_path:
            _set_path0(os.getcwd())
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return 0
    assert script is not None
    sys.argv[:] = [script, *args]
    # A directory or zip file is a sys.path entry that runpy puts first itself.
    if not sys.flags.safe_path and pkgutil.get_importer(script) is None:
        _set_path0(os.path.dirname(os.path.realpath(script)))
    runpy.run_path(script, run_name="__main__")
    return 0


def _set_path0(entry: str) -> None:
    """Replace the launcher's ``sys.path[0]`` with the target's, as ``python`` sets it."""
    if sys.path:
        sys.path[0] = entry
    else:
        sys.path.insert(0, entry)
