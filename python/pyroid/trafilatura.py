# Derived from trafilatura 2.3.1 and 1.11.0 (Apache-2.0), © Adrien Barbaresi; modified.
"""A drop-in, Rust-backed replacement for trafilatura's text extraction (2.3.1 and 1.11.0).

``extract`` and ``bare_extraction`` take trafilatura's arguments and return the same text, byte
for byte, for HTML given as a str (or UTF-8 bytes): the whole pipeline (libxml2's HTML parser as
lxml 6.1.3 drives it, Trafilatura's cleaning, extractors and fallbacks, jusText 3.0.2) is ported
to Rust, for trafilatura 2.3.1 (the default) and 1.11.0 (``compat="1.11.0"``). Options the
port does not cover are delegated to the installed trafilatura of that release with a one-time
:class:`pyroid.FallbackWarning`, or raise :class:`pyroid.PyroidError` without it. See
``trafilatura.pyi``.
"""

from __future__ import annotations

import configparser
import contextvars
import importlib
import importlib.util
import inspect
import re
import sys
import threading
import unicodedata
import warnings
import zlib
from collections.abc import Callable, Mapping
from html import unescape as _unescape
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from pyroid import _versions
from pyroid._pyroid import FallbackWarning, PyroidError
from pyroid._pyroid import trafilatura as _native

# "Document" is built on first use (module __getattr__).
__all__ = [  # noqa: F822
    "DEFAULT_CONFIG",
    "DEFAULT_CONFIGS",
    "DELEGATED",
    "LRU_SIZE",
    "LRU_TEST",
    "REPRODUCED_VERSIONS",
    "TRAFILATURA_VERSION",
    "TRAFILATURA_VERSIONS",
    "Document",
    "LRUCache",
    "bare_extraction",
    "extract",
    "extract_batch",
    "extract_with_metadata",
    "reset_caches",
]

#: The trafilatura release reproduced by default (``compat=``'s default).
TRAFILATURA_VERSION = "2.3.1"

#: The trafilatura releases pyroid reproduces, one ``compat=`` mode each.
TRAFILATURA_VERSIONS: tuple[str, ...] = tuple(_native.supported_versions())

#: For each mode, the exact versions of the libraries whose combined behaviour it reproduces.
REPRODUCED_VERSIONS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "2.3.1": MappingProxyType(
            {"trafilatura": "2.3.1", "lxml": "6.1.3", "libxml2": "2.14.6", "justext": "3.0.2"}
        ),
        # as datatrove[processing]==0.10.1 resolves it; the last four do not change the text
        "1.11.0": MappingProxyType(
            {
                "trafilatura": "1.11.0",
                "lxml": "6.1.3",
                "libxml2": "2.14.6",
                "justext": "3.0.2",
                "lxml-html-clean": "0.4.6",
                "charset-normalizer": "3.5.2",
                "courlan": "1.4.0",
                "htmldate": "1.11.0",
            }
        ),
    }
)

# An installed release that compat= reproduces gets a warning naming that compat= (pyroid.patch,
# datatrove's Trafilatura step and the FineWeb recipe pick the release themselves: no warning).
_versions.check(
    __name__,
    "trafilatura",
    f"trafilatura=={TRAFILATURA_VERSION}",
    modes={
        f"trafilatura=={v}": f'compat="{v}"'
        for v in TRAFILATURA_VERSIONS
        if v != TRAFILATURA_VERSION
    },
)

#: Functions that always run the real trafilatura (it must be installed; 2.3.1 only).
DELEGATED = ("extract_with_metadata",)

#: trafilatura.settings.LRU_SIZE: the size of the module-level dedup cache.
LRU_SIZE: int = _native.LRU_SIZE

#: trafilatura.deduplication.LRUCache, in Rust (safe to share between threads).
LRUCache = _native.LRUCache

#: The cache ``deduplicate=True`` reads and updates (trafilatura.deduplication.LRU_TEST).
LRU_TEST = LRUCache(maxsize=LRU_SIZE)


def reset_caches():  # no annotation: inspect.signature() is trafilatura's "()"
    """Empty the module-level dedup cache :data:`LRU_TEST` (``trafilatura.meta.reset_caches``)."""
    LRU_TEST.clear()


# trafilatura 2.3.1's settings.cfg, verbatim.
_SETTINGS_CFG = """\
# Defines settings for trafilatura (https://trafilatura.readthedocs.io/en/latest/settings.html)

[DEFAULT]

# Download
DOWNLOAD_TIMEOUT = 30
MAX_FILE_SIZE = 20000000
MIN_FILE_SIZE = 10

# sleep between requests
SLEEP_TIME = 5.0

# one line per user-agent
USER_AGENTS =
#     "agent1"
#     "agent2"

# cookie for HTTP requests
COOKIE =

# maximum number of redirects that we will follow
MAX_REDIRECTS = 2

# block requests to non-public IP addresses (loopback, private, link-local)
# set to "off" for intranet use cases
SSRF_PROTECTION = on

# retry without certificate verification after an SSL error
# set to "off" to refuse unverified connections
INSECURE_SSL_FALLBACK = on


# Extraction
MIN_EXTRACTED_SIZE = 250
MIN_OUTPUT_SIZE = 1
MIN_OUTPUT_COMM_SIZE = 1


# discard documents with too many elements
MAX_TREE_SIZE =


# CLI file processing only, set to 0 to disable
EXTRACTION_TIMEOUT = 30


# Deduplication
MIN_DUPLCHECK_SIZE = 100
MAX_REPETITIONS = 2


# Extraction option for Htmldate
EXTENSIVE_DATE_SEARCH = on


# URLs in feeds and sitemaps
EXTERNAL_URLS = off
"""


def _default_config(text: str = _SETTINGS_CFG) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read_string(text)
    return config


#: trafilatura.settings.DEFAULT_CONFIG: settings.cfg's values, the default of ``config=``.
DEFAULT_CONFIG = _default_config()

# trafilatura 1.11.0's settings.cfg, verbatim.
_SETTINGS_CFG_1_11 = """\
# Defines settings for trafilatura (https://trafilatura.readthedocs.io/en/latest/settings.html)

[DEFAULT]

# Download
DOWNLOAD_TIMEOUT = 30
MAX_FILE_SIZE = 20000000
MIN_FILE_SIZE = 10
# sleep between requests
SLEEP_TIME = 5
# user-agents here: agent1,agent2,...
USER_AGENTS =
# cookie for HTTP requests
COOKIE =
# Maximum number of redirects that we will follow
MAX_REDIRECTS = 2

# Extraction
MIN_EXTRACTED_SIZE = 250
MIN_EXTRACTED_COMM_SIZE = 1
MIN_OUTPUT_SIZE = 1
MIN_OUTPUT_COMM_SIZE = 1

# CLI file processing only, set to 0 to disable
EXTRACTION_TIMEOUT = 30

# Deduplication
MIN_DUPLCHECK_SIZE = 100
MAX_REPETITIONS = 2

# Extraction option for Htmldate
EXTENSIVE_DATE_SEARCH = on

# URLs in feeds and sitemaps
EXTERNAL_URLS = off
"""

#: Each mode's trafilatura.settings.DEFAULT_CONFIG (its release's settings.cfg). With
#: ``compat="1.11.0"``, the default ``config`` (:data:`DEFAULT_CONFIG`) stands for 1.11.0's.
DEFAULT_CONFIGS: Mapping[str, configparser.ConfigParser] = MappingProxyType(
    {"2.3.1": DEFAULT_CONFIG, "1.11.0": _default_config(_SETTINGS_CFG_1_11)}
)


def _use_config(filename: str) -> configparser.ConfigParser:
    """trafilatura.settings.use_config(filename): the defaults, overridden by the file."""
    if not Path(filename).is_file():
        raise FileNotFoundError("The given config file does not exist")
    config = _default_config()
    config.read(filename)
    return config


def _unicode_version() -> str:
    """The interpreter's Unicode version, or the newest one the port knows."""
    version = unicodedata.unidata_version
    supported = _native.supported_unicode_versions()
    if version in supported:
        return version
    newest: str = supported[-1]
    warnings.warn(
        f"pyroid.trafilatura has no tables for Unicode {version} (this Python's unicodedata); "
        f"using Unicode {newest}. Output can differ from trafilatura for characters added "
        f"after {newest}.",
        RuntimeWarning,
        stacklevel=3,
    )
    return newest


_UNICODE = _unicode_version()

# CPython's default sys.get_int_max_str_digits(), which the port's html.unescape assumes.
_INT_MAX_STR_DIGITS = 4300


def _mode(compat: object, modes: tuple[str, ...] = TRAFILATURA_VERSIONS) -> str:
    if not isinstance(compat, str) or compat not in modes:
        choices = ", ".join(map(repr, modes))
        raise ValueError(f"compat must be one of {choices}, not {compat!r}")
    return compat


# ---------------------------------------------------------------------------
# Delegation
# ---------------------------------------------------------------------------

_WARNED: set[str] = set()
_UNSET: Any = object()  # an argument not given
_DELEGATE_LOCK = threading.Lock()


def _caller_stacklevel() -> int:
    """The ``stacklevel`` that points ``warnings.warn`` at the first frame outside this module."""
    frame = sys._getframe(1)
    level = 1
    while frame.f_back is not None and frame.f_globals.get("__name__") == __name__:
        frame = frame.f_back
        level += 1
    return level


def _loaded(name: str) -> Any:
    """``sys.modules[name]`` if imported (the real module under a pyroid patch shim), else None."""
    module = sys.modules.get(name)
    if getattr(module, "__pyroid__", False):
        module = getattr(module, "__wrapped__", None)
    return module


def _real_module() -> Any:
    """The real trafilatura package, or None if it is not installed."""
    try:
        import trafilatura as real
    except ImportError:
        return None
    # Under a pyroid patch shim, the real module it wraps, never the shim.
    if getattr(real, "__pyroid__", False):
        real = real.__wrapped__
    return real


def _trafilatura(name: str, reason: str, mode: str) -> Any:
    """The real trafilatura for a delegated call (warning once), or raise PyroidError."""
    real = _real_module()
    pip = f"pip install trafilatura=={mode}"
    if real is None:
        raise PyroidError(
            f"pyroid.trafilatura.{name}: {reason} is not supported by pyroid's Rust port, and "
            f"trafilatura is not installed to fall back to ({pip})"
        )
    found = getattr(real, "__version__", None)
    if found != mode:
        raise PyroidError(
            f"pyroid.trafilatura.{name}: {reason} is not supported by pyroid's Rust port, and "
            f"the installed trafilatura is {found or 'unversioned'}, not {mode} (the version "
            f"this call reproduces), so it cannot stand in ({pip}, or pass compat= the "
            "installed version if pyroid supports it)"
        )
    where = name if mode == TRAFILATURA_VERSION else f"{name} (compat={mode!r})"
    key = f"{where}:{reason}"
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(
            f"pyroid.trafilatura.{where}: {reason} is not supported by the Rust port; "
            f"delegating to trafilatura {mode} (slow path). This warning is shown once.",
            FallbackWarning,
            stacklevel=_caller_stacklevel(),
        )
    return real


def _is_real_lru(x: object) -> bool:
    module = _loaded("trafilatura.deduplication")
    return module is not None and isinstance(x, module.LRUCache)


# The cache that stands for the module-level one in this context: the datatrove step gives each
# task its own (datatrove runs each task's trafilatura in a fresh child process).
_TASK_CACHE: contextvars.ContextVar[Any] = contextvars.ContextVar("_TASK_CACHE", default=None)


def _pyroid_cache(deduplicate: object, mode: str = TRAFILATURA_VERSION) -> Any:
    """The pyroid cache a call reads and updates, or None.

    2.3.1: the argument if it is an :class:`LRUCache`, else the module cache. 1.11.0 has no
    per-call cache: any true ``deduplicate`` means the module cache.
    """
    if not deduplicate:
        return None
    if mode == TRAFILATURA_VERSION:
        if isinstance(deduplicate, LRUCache):
            return deduplicate
        if _is_real_lru(deduplicate):
            return None  # delegated with it
    task = _TASK_CACHE.get()
    return LRU_TEST if task is None else task


def _lru_items(lru: Any) -> list[tuple[str, int]]:
    """A trafilatura 1.11.0 LRUCache's entries, oldest first (its circular linked list)."""
    out = []
    root = lru.root
    link = root[1]
    while link is not root:
        out.append((link[2], link[3]))
        link = link[1]
    return out


def _call_real(fn: Callable[..., Any], cache: Any, args: dict[str, Any], mode: str) -> Any:
    """``fn(**args)``; a pyroid dedup cache is lent to it.

    2.3.1 takes a cache as ``deduplicate``. 1.11.0 only has its module cache, so that one holds
    the pyroid cache's entries during the call, and gets its own back afterwards.
    """
    if cache is None:
        return fn(**args)
    real = _real_module()
    if mode != TRAFILATURA_VERSION:
        with _DELEGATE_LOCK:
            module_cache = real.deduplication.LRU_TEST
            own = _lru_items(module_cache)
            module_cache.clear()
            for key, value in cache._items():
                module_cache.put(key, value)
            try:
                return fn(**args)
            finally:
                cache._set_items(_lru_items(module_cache))
                module_cache.clear()
                for key, value in own:
                    module_cache.put(key, value)
    with _DELEGATE_LOCK:
        stand_in = real.deduplication.LRUCache(maxsize=cache.maxsize)
        for key, value in cache._items():
            stand_in.put(key, value)
        args = {**args, "deduplicate": stand_in}
        try:
            return fn(**args)
        finally:
            cache._set_items(list(stand_in.cache.items()))


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


def _input_reason(filecontent: object) -> str | None:
    """Why a non-str input must be delegated (lxml trees, responses), or None."""
    kind = type(filecontent)
    if kind is str or kind is bytes:
        return None
    lxml_html = _loaded("lxml.html")
    if lxml_html is not None and isinstance(filecontent, lxml_html.HtmlElement):
        return "an lxml tree input"
    if hasattr(filecontent, "data"):
        return "a response object input"
    return None


def _option_reason(args: dict[str, Any], python_format: bool) -> str | None:
    """Why a call cannot run natively, judged from its arguments alone; or None."""
    fmt = args["output_format"]
    if fmt != ("python" if python_format else "txt"):
        return f"output_format={fmt!r}"
    for name in ("include_formatting", "include_links", "include_images"):
        if args[name]:
            return f"{name}=True"
    for name in ("with_metadata", "only_with_metadata"):
        if args[name]:
            return f"{name}=True"
    for name in ("url_blacklist", "author_blacklist"):
        if args[name]:
            return f"{name} (metadata extraction)"
    if args["target_language"]:
        return "target_language"
    if args["prune_xpath"] is not None:
        return "prune_xpath"
    options = args["options"]
    settings = _loaded("trafilatura.settings")
    if options and settings is not None and isinstance(options, settings.Extractor):
        return "options=Extractor(...)"
    if _is_real_lru(args["deduplicate"]):
        return "deduplicate=trafilatura.deduplication.LRUCache(...)"
    if sys.get_int_max_str_digits() != _INT_MAX_STR_DIGITS:
        return "a sys.set_int_max_str_digits() limit other than 4300"
    return _input_reason(args["filecontent"])


def _check_deprecation(fast: Any, no_fallback: Any, as_dict: Any, max_tree_size: Any) -> Any:
    """trafilatura.core._check_deprecation: the effective ``fast``."""
    if no_fallback:
        warnings.warn(
            '"no_fallback" will be removed, use "fast" instead',
            DeprecationWarning,
            stacklevel=_caller_stacklevel(),
        )
    if as_dict:
        warnings.warn(
            '"as_dict" will be removed, use the .as_dict() method instead',
            DeprecationWarning,
            stacklevel=_caller_stacklevel(),
        )
    if max_tree_size:
        raise ValueError('"max_tree_size" will be removed, use settings.cfg instead')
    return fast or no_fallback


# Size settings above this behave like infinity against any real text length.
_HUGE = 2**40


def _clamp(n: int) -> int:
    return max(-_HUGE, min(n, _HUGE))


class _Settings:
    """The settings.cfg values of one call, read as trafilatura.settings.Extractor reads them."""

    __slots__ = ("max_file_size", "values")

    def __init__(self, config: Any, url: Any, date_params: Any) -> None:
        # Extractor.__init__, in its order: the source (a str check by .encode), then the config.
        source = url
        if source:
            source.encode("utf-8", "replace").decode("utf-8")
        config = config or DEFAULT_CONFIG
        get = config.getint
        sizes = (
            get("DEFAULT", "MIN_EXTRACTED_SIZE"),
            get("DEFAULT", "MIN_OUTPUT_SIZE"),
            get("DEFAULT", "MIN_OUTPUT_COMM_SIZE"),
            get("DEFAULT", "MIN_DUPLCHECK_SIZE"),
            get("DEFAULT", "MAX_REPETITIONS"),
        )
        self.max_file_size: int = get("DEFAULT", "MAX_FILE_SIZE")
        get("DEFAULT", "MIN_FILE_SIZE")
        value = config.get("DEFAULT", "MAX_TREE_SIZE", fallback="").strip()
        max_tree = int(value) if value.isdigit() else None
        if not date_params:
            config.getboolean("DEFAULT", "EXTENSIVE_DATE_SEARCH")
        self.values = (*sizes, max_tree)

    def reason(self) -> str | None:
        """Why these settings must be delegated (a negative size setting), or None."""
        if any(v < 0 for v in self.values[:4]):
            return "a negative size setting in config"
        return None

    def options(self, mode: str, args: dict[str, Any], fast: Any) -> Any:
        return _native_options(mode, args, fast, self.values[:5], self.values[5])


def _native_options(
    mode: str, args: dict[str, Any], fast: Any, sizes: tuple[int, ...], max_tree: int | None
) -> Any:
    """The native options for a call (cached per distinct value set)."""
    if args["favor_recall"]:
        focus = "recall"
    elif args["favor_precision"]:
        focus = "precision"
    else:
        focus = "balanced"
    ext, out, out_comm, dupl, reps = sizes
    key = (
        mode,
        focus,
        bool(args["include_comments"]),
        bool(fast),
        bool(args["include_tables"]),
        _clamp(ext),
        _clamp(out),
        _clamp(out_comm),
        _clamp(dupl),
        _clamp(reps),
        None if max_tree is None else _clamp(max_tree),
    )
    opts = _OPTIONS.get(key)
    if opts is None:
        opts = _native._Options(
            version=mode,
            unicode=_UNICODE,
            focus=focus,
            comments=key[2],
            fast=key[3],
            tables=key[4],
            min_extracted_size=key[5],
            min_output_size=key[6],
            min_output_comm_size=key[7],
            min_duplcheck_size=key[8],
            max_repetitions=key[9],
            max_tree_size=key[10],
        )
        if len(_OPTIONS) > 64:
            _OPTIONS.clear()
        _OPTIONS[key] = opts
    return opts


_OPTIONS: dict[tuple[Any, ...], Any] = {}

_BROTLI: list[Any] = []


def _brotli() -> Any:
    """The brotli module if trafilatura would use it (``utils.HAS_BROTLI``), else None."""
    if not _BROTLI:
        try:
            import brotli  # type: ignore[import-not-found,unused-ignore]

            try:
                brotli.Decompressor().process(b"", output_buffer_limit=1)
            except Exception:
                brotli = None
        except ImportError:
            brotli = None
        _BROTLI.append(brotli)
    return _BROTLI[0]


def _decompresses(raw: bytes, max_size: int) -> bool:
    """Whether trafilatura's handle_compressed_file decompresses `raw` (valid UTF-8 bytes).

    Its gzip and zstd magic numbers are not valid UTF-8, so only its speculative brotli and
    zlib attempts can apply.
    """
    brotli = _brotli()
    if brotli is not None:
        try:
            dec = brotli.Decompressor()
            out = dec.process(raw, output_buffer_limit=max_size + 1)
            if len(out) <= max_size and dec.is_finished():
                return True
        except brotli.error:
            pass
    try:
        inflate = zlib.decompressobj(zlib.MAX_WBITS)
        out = inflate.decompress(raw, max_size + 1)
    except zlib.error:
        return False
    return len(out) <= max_size and inflate.eof


def _bytes_reason(filecontent: object, settings: _Settings) -> str | None:
    if not isinstance(filecontent, bytes):
        return None
    if not _native.is_utf8(filecontent):
        return "bytes that are not UTF-8"
    if _decompresses(filecontent, settings.max_file_size):
        return "compressed bytes"
    return None


# ---------------------------------------------------------------------------
# trafilatura 1.11.0 (compat="1.11.0")
# ---------------------------------------------------------------------------

V1_11 = "1.11.0"


def _use_config_111(filename: Any, config: Any) -> Any:
    """Trafilatura 1.11.0's settings.use_config(filename, config): ``config`` wins."""
    if config is not None:
        # pyroid's default config is 2.3.1's: here it stands for 1.11.0's own
        return DEFAULT_CONFIGS[V1_11] if config is DEFAULT_CONFIG else config
    if filename is None:
        return DEFAULT_CONFIGS[V1_11]
    if not Path(filename).is_file():
        raise FileNotFoundError("The given config file does not exist")
    config = configparser.ConfigParser()
    config.read(filename)
    return config


class _Settings111:
    """The settings of one call, read as trafilatura 1.11.0's Extractor reads them."""

    __slots__ = ("values",)

    def __init__(self, config: Any, date_params: Any) -> None:
        if config is DEFAULT_CONFIG:
            config = DEFAULT_CONFIGS[V1_11]
        get = config.getint  # Extractor._add_config, CONFIG_MAPPING's order
        ext = get("DEFAULT", "MIN_EXTRACTED_SIZE")
        out = get("DEFAULT", "MIN_OUTPUT_SIZE")
        out_comm = get("DEFAULT", "MIN_OUTPUT_COMM_SIZE")
        get("DEFAULT", "MIN_EXTRACTED_COMM_SIZE")  # only logged
        dupl = get("DEFAULT", "MIN_DUPLCHECK_SIZE")
        reps = get("DEFAULT", "MAX_REPETITIONS")
        get("DEFAULT", "MAX_FILE_SIZE")
        get("DEFAULT", "MIN_FILE_SIZE")
        if not date_params:
            config.getboolean("DEFAULT", "EXTENSIVE_DATE_SEARCH")
        self.values = (ext, out, out_comm, dupl, reps)

    def reason(self, max_tree_size: Any) -> str | None:
        if any(v < 0 for v in self.values[:4]):
            return "a negative size setting in config"
        if max_tree_size and (not isinstance(max_tree_size, int) or max_tree_size < 0):
            return f"max_tree_size={max_tree_size!r}"
        return None

    def options(self, args: dict[str, Any]) -> Any:
        max_tree = args["max_tree_size"]
        return _native_options(
            V1_11,
            args,
            args["no_fallback"],
            self.values,
            int(max_tree) if max_tree else None,
        )


def _option_reason_111(args: dict[str, Any], python_format: bool) -> str | None:
    """Why a 1.11.0 call cannot run natively, judged from its arguments alone; or None."""
    fmt = args["output_format"]
    if fmt != ("python" if python_format else "txt"):
        return f"output_format={fmt!r}"
    for name in ("include_formatting", "include_links", "include_images"):
        if args[name]:
            return f"{name}=True"
    for name in ("with_metadata", "only_with_metadata"):
        if args[name]:
            return f"{name}=True"
    if args["url_blacklist"]:  # 1.11.0 extracts metadata for it (not for author_blacklist)
        return "url_blacklist (metadata extraction)"
    if args["target_language"]:
        return "target_language"
    if args["prune_xpath"] is not None:
        return "prune_xpath"
    options = args["options"]
    settings = _loaded("trafilatura.settings")
    if options and settings is not None and isinstance(options, settings.Extractor):
        return "options=Extractor(...)"
    if sys.get_int_max_str_digits() != _INT_MAX_STR_DIGITS:
        return "a sys.set_int_max_str_digits() limit other than 4300"
    return _input_reason(args["filecontent"])


_BROTLI_111: list[Any] = []


def _decompresses_111(raw: bytes) -> bool:
    """Whether trafilatura 1.11.0's handle_compressed_file decompresses `raw` (valid UTF-8).

    Its gzip and zstd magic numbers are not valid UTF-8; brotli (any installed one) and zlib are
    tried on everything else.
    """
    if not _BROTLI_111:
        try:
            import brotli  # type: ignore[import-not-found,unused-ignore]
        except ImportError:
            brotli = None
        _BROTLI_111.append(brotli)
    brotli = _BROTLI_111[0]
    if brotli is not None:
        try:
            brotli.decompress(raw)
            return True
        except brotli.error:
            pass
    try:
        zlib.decompress(raw)
    except zlib.error:
        return False
    return True


def _bytes_reason_111(filecontent: object) -> str | None:
    if not isinstance(filecontent, bytes):
        return None
    if not _native.is_utf8(filecontent):
        return "bytes that are not UTF-8"
    if _decompresses_111(filecontent):
        return "compressed bytes"
    return None


def _real_111_args(args: dict[str, Any]) -> dict[str, Any]:
    """The arguments of a delegated 1.11.0 call: 2.3.1's ``fast`` does not exist there."""
    out = {k: v for k, v in args.items() if k != "fast"}
    if out.get("config") is DEFAULT_CONFIG:
        out["config"] = DEFAULT_CONFIGS[V1_11]
    return out


def _extract_111(args: dict[str, Any]) -> str | None:
    """Trafilatura 1.11.0's extract (``fast`` is not one of its arguments: it is ignored)."""
    reason = _option_reason_111(args, python_format=False)
    deduplicate = args["deduplicate"]
    if reason is None:
        # Extractor(config=use_config(settingsfile, config), ...): its exceptions propagate
        config = _use_config_111(args["settingsfile"], args["config"])
        settings = _Settings111(config, args["date_extraction_params"])
        reason = settings.reason(args["max_tree_size"]) or _bytes_reason_111(args["filecontent"])
    if reason is not None:
        real = _trafilatura("extract", reason, V1_11)
        cache = _pyroid_cache(deduplicate, V1_11)
        return _call_real(real.extract, cache, _real_111_args(args), V1_11)  # type: ignore[no-any-return]
    filecontent = args["filecontent"]
    if not isinstance(filecontent, (str, bytes)):
        return None  # load_html's TypeError, caught
    cache = _pyroid_cache(deduplicate, V1_11)
    result: str | None = _native.extract(filecontent, settings.options(args), cache)
    return result


def _discard_111(options: Any) -> None:
    """1.11.0's bare_extraction ``except`` clause: ``LOGGER.warning(..., options.source)``.

    Before the Extractor exists, ``options`` is the argument (None by default), so this raises
    AttributeError, as trafilatura 1.11.0 does.
    """
    options.source  # noqa: B018


def _bare_111(args: dict[str, Any]) -> Any:
    """Trafilatura 1.11.0's bare_extraction."""
    if args["fast"]:
        raise TypeError("bare_extraction() got an unexpected keyword argument 'fast'")
    reason = _option_reason_111(args, python_format=True)
    filecontent, options = args["filecontent"], args["options"]
    deduplicate = args["deduplicate"]
    settings = None
    if reason is None:
        if not isinstance(filecontent, (str, bytes)):
            return _discard_111(options)  # load_html's TypeError
        reason = _bytes_reason_111(filecontent)
    if reason is None:
        # load_html, then Extractor(config=config, ...), inside `try: ... except (TypeError,
        # ValueError)`
        try:
            settings = _Settings111(args["config"], args["date_extraction_params"])
        except (TypeError, ValueError):
            return _discard_111(options)
        except Exception:
            if not _native.loads(filecontent, V1_11):
                return _discard_111(options)
            raise
        reason = settings.reason(args["max_tree_size"])
    if reason is not None:
        real = _trafilatura("bare_extraction", reason, V1_11)
        cache = _pyroid_cache(deduplicate, V1_11)
        return _call_real(real.bare_extraction, cache, _real_111_args(args), V1_11)
    assert settings is not None
    cache = _pyroid_cache(deduplicate, V1_11)
    try:
        parts = _native.bare_extraction(filecontent, settings.options(args), cache)
    except AttributeError as e:
        if options is None or getattr(e, "name", None) != "source":
            raise
        return _discard_111(options)  # the page does not load
    if parts is None:
        return None
    doc = _document(parts, V1_11)
    return doc.as_dict() if args["as_dict"] else doc


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

_EXTRACT_DEFAULTS: dict[str, Any] = {
    "url": None,
    "record_id": None,
    "fast": False,
    "no_fallback": False,
    "favor_precision": False,
    "favor_recall": False,
    "include_comments": True,
    "output_format": "txt",
    "tei_validation": False,
    "target_language": None,
    "include_tables": True,
    "include_images": False,
    "include_formatting": None,
    "include_links": False,
    "deduplicate": False,
    "date_extraction_params": None,
    "with_metadata": False,
    "only_with_metadata": False,
    "max_tree_size": None,
    "url_blacklist": None,
    "author_blacklist": None,
    "settingsfile": None,
    "prune_xpath": None,
    "config": DEFAULT_CONFIG,
    "options": None,
}


def _delegate_extract(args: dict[str, Any], reason: str, mode: str, name: str = "extract") -> Any:
    real = _trafilatura(name, reason, mode)
    return _call_real(real.extract, _pyroid_cache(args["deduplicate"]), args, mode)


def _prepare_extract(args: dict[str, Any], mode: str) -> tuple[Any, _Settings, str | None]:
    """Trafilatura's argument handling up to the extraction: (fast, settings, delegate reason).

    Emits trafilatura's DeprecationWarning and raises its exceptions, in its order.
    """
    fast = _check_deprecation(args["fast"], args["no_fallback"], False, args["max_tree_size"])
    settingsfile = args["settingsfile"]
    config = _use_config(settingsfile) if settingsfile else args["config"]
    settings = _Settings(config, args["url"], args["date_extraction_params"])
    return fast, settings, settings.reason()


def _after_warnings(args: dict[str, Any], fast: Any) -> dict[str, Any]:
    """The arguments for a call delegated after the deprecation checks ran (no second warning)."""
    return {**args, "fast": fast, "no_fallback": False}


def extract(  # noqa: D417  # **kwargs is compat="1.11.0" only, and the stub is 2.3.1's
    filecontent: Any,
    url: str | None = None,
    record_id: str | None = None,
    fast: bool = False,
    no_fallback: bool = False,
    favor_precision: bool = False,
    favor_recall: bool = False,
    include_comments: bool = True,
    output_format: str = "txt",
    tei_validation: bool = False,
    target_language: str | None = None,
    include_tables: bool = True,
    include_images: bool = False,
    include_formatting: bool | None = None,
    include_links: bool = False,
    deduplicate: Any = False,
    date_extraction_params: dict[str, Any] | None = None,
    with_metadata: bool = False,
    only_with_metadata: bool = False,
    max_tree_size: int | None = None,
    url_blacklist: set[str] | None = None,
    author_blacklist: set[str] | None = None,
    settingsfile: str | None = None,
    prune_xpath: str | list[str] | None = None,
    config: configparser.ConfigParser = DEFAULT_CONFIG,
    options: Any = None,
    *,
    compat: str = TRAFILATURA_VERSION,
    **kwargs: Any,
) -> str | None:
    """Extract the main text of an HTML page: ``trafilatura.extract``.

    >>> page = "<html><body><article><p>" + "Some words. " * 30 + "</p></article></body></html>"
    >>> extract(page)[:22]
    'Some words. Some words'

    The arguments, defaults and results are trafilatura 2.3.1's; ``compat`` is pyroid's only
    addition. The text is NFC-normalized, as trafilatura's.

    Args:
        filecontent: The page: a str, or bytes (UTF-8, else delegated). lxml trees and response
            objects are delegated; other types give ``None``, as in trafilatura.
        url: The page's URL (only metadata and links use it).
        record_id: An ID for the metadata (not part of the txt output).
        fast: Skip the readability and jusText comparison.
        no_fallback: Deprecated alias of ``fast`` (trafilatura's ``DeprecationWarning``).
        favor_precision: Prefer less text but correct extraction.
        favor_recall: Prefer more text when unsure (it wins over ``favor_precision``).
        include_comments: Extract comments along with the main text.
        output_format: ``"txt"`` runs natively; the others are delegated.
        tei_validation: Only used for ``"xmltei"`` (delegated).
        target_language: Delegated (language detection).
        include_tables: Take ``<table>`` content into account.
        include_images: Delegated.
        include_formatting: ``True`` is delegated.
        include_links: Delegated.
        deduplicate: ``True`` deduplicates through the module cache :data:`LRU_TEST`; an
            :class:`LRUCache` scopes it to that cache (a trafilatura ``LRUCache`` is delegated).
        date_extraction_params: htmldate parameters (only used with metadata).
        with_metadata: Delegated.
        only_with_metadata: Delegated.
        max_tree_size: Deprecated: a truthy value raises trafilatura's ``ValueError``.
        url_blacklist: A non-empty set is delegated (it needs metadata).
        author_blacklist: A non-empty set is delegated (it needs metadata).
        settingsfile: A settings.cfg file overriding the defaults.
        prune_xpath: Delegated.
        config: A ``ConfigParser`` with trafilatura's settings (default :data:`DEFAULT_CONFIG`).
        options: A ``trafilatura.settings.Extractor`` is delegated; other values are ignored, as
            in trafilatura.
        compat: The trafilatura release to reproduce, one of :data:`TRAFILATURA_VERSIONS`
            (``"1.11.0"``: see the module docstring for how its arguments differ).

    Returns:
        The text, or ``None`` where trafilatura returns ``None`` (nothing extracted, too short,
        a duplicate, not HTML).

    Raises:
        ValueError: ``max_tree_size``; a page with a ``&#`` reference of more than 4300 digits
            (CPython's ``int()`` limit, as in trafilatura); a ``compat`` pyroid does not
            reproduce.
        AttributeError: An unknown ``output_format`` (from trafilatura, delegated).
        pyroid.PyroidError: The call needs trafilatura (see the module docstring), and it is
            not installed, or is another version than ``compat``.
    """
    args = {k: v for k, v in locals().items() if k not in ("compat", "kwargs")}
    mode = _mode(compat)
    if kwargs:
        if mode != V1_11:
            raise TypeError(f"extract() got an unexpected keyword argument {next(iter(kwargs))!r}")
        if any(k in kwargs for k in _DEPRECATED_111):
            raise NameError(
                'Deprecated argument: use output_format instead, e.g. output_format="xml"'
            )
    if mode == V1_11:
        return _extract_111(args)
    reason = _option_reason(args, python_format=False)
    if reason is not None:
        return _delegate_extract(args, reason, mode)  # type: ignore[no-any-return]
    fast, settings, reason = _prepare_extract(args, mode)
    reason = reason or _bytes_reason(filecontent, settings)
    if reason is not None:
        return _delegate_extract(_after_warnings(args, fast), reason, mode)  # type: ignore[no-any-return]
    if not isinstance(filecontent, (str, bytes)):
        return None  # trafilatura: TypeError("incompatible input type"), caught
    opts = settings.options(mode, args, fast)
    result: str | None = _native.extract(filecontent, opts, _pyroid_cache(deduplicate))
    return result


# ---------------------------------------------------------------------------
# bare_extraction and Document
# ---------------------------------------------------------------------------

_DOCUMENT_SLOTS = (
    "title",
    "author",
    "url",
    "hostname",
    "description",
    "sitename",
    "date",
    "categories",
    "tags",
    "fingerprint",
    "id",
    "license",
    "body",
    "comments",
    "commentsbody",
    "raw_text",
    "text",
    "language",
    "image",
    "pagetype",
    "filedate",
)
_PLAIN_SLOTS = tuple(s for s in _DOCUMENT_SLOTS if s not in ("body", "commentsbody", "text"))


class _XML(str):
    """A tree's ``etree.tostring(..., encoding="unicode")``, not yet parsed into lxml."""

    __slots__ = ()


def _lxml_etree() -> Any:
    try:
        from lxml import etree
    except ImportError:
        return None
    return etree


def _from_xml(xml: str) -> Any:
    """The lxml element for `xml` (as ``etree.tostring`` wrote it), or the str without lxml.

    ``tostring`` writes the element's tail after it, escaped (so with no ``<`` or ``>``): it is
    what follows the last ``>``, and becomes the parsed element's tail.
    """
    xml = str(xml)
    etree = _lxml_etree()
    if etree is None or "&#xFFFD;" in xml:
        # A control character in the tree (1.11.0 keeps them; tostring writes each as
        # &#xFFFD;): lxml cannot build such a tree from Python, so the XML stays a str.
        return xml
    end = xml.rfind(">") + 1
    tree, tail = xml[:end], xml[end:]
    try:
        parser = etree.XMLParser(huge_tree=True, resolve_entities=False, strip_cdata=False)
        element = etree.fromstring(tree, parser)
    except etree.XMLSyntaxError:
        # An attribute with an undeclared prefix (an HTML tree can hold one): recover, and keep
        # the result only if it serializes back to the same XML.
        parser = etree.XMLParser(huge_tree=True, resolve_entities=False, recover=True)
        element = etree.fromstring(tree, parser)
        if element is None or etree.tostring(element, encoding="unicode") != tree:
            return xml
    if tail:
        element.tail = _unescape(tail)  # &amp; &lt; &gt; &#13; only
    if "></" in tree:
        _restore_empty_texts(element, tree)
    return element


def _restore_empty_texts(root: Any, xml: str) -> None:
    """Give back the empty text nodes that parsing drops.

    libxml2 writes an element whose only child is an empty text node (``text == ""``) as
    ``<p></p>``, and one with no children at all as ``<p/>``; the XML parser reads both as
    ``text is None``. Start tags come in the order of ``root.iter()`` (comments included), and
    ``<`` and ``>`` only appear as markup outside comments (``tostring`` escapes them).
    """
    flags = []
    i = 0
    while True:
        j = xml.find("<", i)
        if j < 0:
            break
        if xml.startswith("<!--", j):
            flags.append(False)
            i = xml.find("-->", j) + 3
            continue
        if xml.startswith("</", j):
            i = j + 2
            continue
        k = xml.find(">", j)
        flags.append(xml[k - 1] != "/" and xml.startswith("</", k + 1))
        i = k + 1
    for element, empty in zip(root.iter(), flags, strict=False):
        if empty and element.text is None and len(element) == 0:
            element.text = ""


def _empty_body() -> Any:
    etree = _lxml_etree()
    return etree.Element("body") if etree is not None else "<body/>"


class _DocumentFields:
    """trafilatura.settings.Document's fields, for when trafilatura 2.3.1 is not installed."""

    __slots__ = _DOCUMENT_SLOTS

    def __init__(
        self,
        *,
        title: str | None = None,
        author: str | None = None,
        url: str | None = None,
        hostname: str | None = None,
        description: str | None = None,
        sitename: str | None = None,
        date: str | None = None,
        categories: list[str] | None = None,
        tags: list[str] | None = None,
        fingerprint: str | None = None,
        idval: str | None = None,
        license_val: str | None = None,
        body: Any = None,
        comments: str | None = None,
        commentsbody: Any = None,
        raw_text: str | None = None,
        text: str | None = None,
        language: str | None = None,
        image: str | None = None,
        pagetype: str | None = None,
        filedate: str | None = None,
    ) -> None:
        self.title = title
        self.author = author
        self.url = url
        self.hostname = hostname
        self.description = description
        self.sitename = sitename
        self.date = date
        self.categories = categories
        self.tags = tags
        self.fingerprint = fingerprint
        self.id = idval
        self.license = license_val
        self.body = body if body is not None else _empty_body()
        self.comments = comments
        self.commentsbody = commentsbody if commentsbody is not None else _empty_body()
        self.raw_text = raw_text
        self.text = text
        self.language = language
        self.image = image
        self.pagetype = pagetype
        self.filedate = filedate

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Any:
        """Set a series of attributes using a dictionary."""
        doc = cls()
        for key, value in data.items():
            setattr(doc, key, value)
        return doc

    def as_dict(self) -> dict[str, Any]:
        """Convert the document to a dictionary."""
        return {attr: getattr(self, attr, None) for attr in _DOCUMENT_SLOTS}


class _DocumentFields111:
    """trafilatura 1.11.0's settings.Document, for when trafilatura 1.11.0 is not installed."""

    __slots__ = _DOCUMENT_SLOTS

    def __init__(self) -> None:
        for slot in _DOCUMENT_SLOTS:
            setattr(self, slot, None)

    def __getattr__(self, name: str) -> None:
        raise AttributeError("% attribute not present in Document", name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _DOCUMENT_SLOTS:
            object.__setattr__(self, name, value)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Any:
        """Set a series of attributes using a dictionary."""
        doc = cls()
        for key, value in data.items():
            setattr(doc, key, value)
        return doc

    def set_attributes(self, **kwargs: Any) -> None:
        """Helper function to (re-)set a series of attributes."""
        for key, value in kwargs.items():
            if value:
                setattr(self, key, value)

    def as_dict(self) -> dict[str, Any]:
        """Convert the document to a dictionary."""
        return {attr: getattr(self, attr, None) for attr in _DOCUMENT_SLOTS}


def _raw_tree(doc: Any, name: str) -> Any:
    """A tree slot's value without parsing it (the base class's slot under the property)."""
    return type(doc)._pyroid_tree_slots[name].__get__(doc, type(doc))


def _getstate(self: Any) -> dict[str, Any]:
    """Pickle state: the fields, with lxml trees as their XML (lxml elements do not pickle)."""
    state = {}
    for slot in _DOCUMENT_SLOTS:
        if slot in ("body", "commentsbody"):
            value = _raw_tree(self, slot)
            if type(value).__module__.startswith("lxml."):
                value = _XML(_lxml_etree().tostring(value, encoding="unicode"))
        else:
            value = getattr(self, slot, None)
        state[slot] = value
    return state


def _setstate(self: Any, state: dict[str, Any]) -> None:
    for slot, value in state.items():
        setattr(self, slot, value)


_DOCUMENT: dict[str, type] = {}
_DOCUMENT_LOCK = threading.Lock()


def _tree_property(slot: Any, name: str) -> property:
    def get(self: Any) -> Any:
        value = slot.__get__(self, type(self))
        if type(value) is _XML:
            value = _from_xml(value)
            slot.__set__(self, value)
        return value

    def put(self: Any, value: Any) -> None:
        slot.__set__(self, value)

    return property(get, put, doc=f"``{name}``: an lxml element (its XML as a str without lxml).")


_DOCUMENT_NAMES = {TRAFILATURA_VERSION: "Document", "1.11.0": "_Document_1_11_0"}


def _document_class(mode: str = TRAFILATURA_VERSION) -> type:
    """The class of bare_extraction's results for a mode, built on first use.

    A subclass of ``trafilatura.settings.Document`` when that trafilatura release is installed,
    else of a class with the same fields; ``body`` and ``commentsbody`` are parsed into lxml
    elements when first read.
    """
    cls = _DOCUMENT.get(mode)
    if cls is not None:
        return cls
    with _DOCUMENT_LOCK:
        cls = _DOCUMENT.get(mode)
        if cls is not None:
            return cls
        base: type = _DocumentFields if mode == TRAFILATURA_VERSION else _DocumentFields111
        try:
            found = importlib.util.find_spec("trafilatura") is not None
        except (ImportError, ValueError):
            found = False
        if found:
            real = _real_module()
            if real is not None and getattr(real, "__version__", None) == mode:
                settings = importlib.import_module(real.__name__ + ".settings")
                base = getattr(settings, "Document", base)
        ns = {
            "__module__": __name__,
            "__qualname__": _DOCUMENT_NAMES[mode],
            "__doc__": (
                "What :func:`bare_extraction` returns: trafilatura.settings.Document's fields.\n\n"
                "``body`` and ``commentsbody`` become lxml elements when first read."
            ),
            "body": _tree_property(base.__dict__["body"], "body"),
            "commentsbody": _tree_property(base.__dict__["commentsbody"], "commentsbody"),
            "_pyroid_tree_slots": {
                "body": base.__dict__["body"],
                "commentsbody": base.__dict__["commentsbody"],
            },
            "__getstate__": _getstate,
            "__setstate__": _setstate,
        }
        cls = type("Document", (base,), ns)
        _DOCUMENT[mode] = cls
        return cls


def __getattr__(name: str) -> Any:
    for mode, qualname in _DOCUMENT_NAMES.items():
        if name == qualname:
            return _document_class(mode)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _document(parts: tuple[str, str | None, str, str | None], mode: str) -> Any:
    cls = _document_class(mode)
    text, comments, body, commentsbody = parts
    doc = cls.__new__(cls)
    for slot in _PLAIN_SLOTS:
        setattr(doc, slot, None)
    doc.text = text
    doc.raw_text = text
    doc.comments = comments
    doc.body = _XML(body)
    doc.commentsbody = None if commentsbody is None else _XML(commentsbody)
    return doc


_BARE_DEFAULTS: dict[str, Any] = {
    "url": None,
    "fast": False,
    "no_fallback": False,
    "favor_precision": False,
    "favor_recall": False,
    "include_comments": True,
    "output_format": "python",
    "target_language": None,
    "include_tables": True,
    "include_images": False,
    "include_formatting": None,
    "include_links": False,
    "deduplicate": False,
    "date_extraction_params": None,
    "with_metadata": False,
    "only_with_metadata": False,
    "max_tree_size": None,
    "url_blacklist": None,
    "author_blacklist": None,
    "as_dict": False,
    "prune_xpath": None,
    "config": DEFAULT_CONFIG,
    "options": None,
}


def bare_extraction(
    filecontent: Any,
    url: str | None = None,
    fast: bool = False,
    no_fallback: bool = False,
    favor_precision: bool = False,
    favor_recall: bool = False,
    include_comments: bool = True,
    output_format: str = "python",
    target_language: str | None = None,
    include_tables: bool = True,
    include_images: bool = False,
    include_formatting: bool | None = None,
    include_links: bool = False,
    deduplicate: Any = False,
    date_extraction_params: dict[str, Any] | None = None,
    with_metadata: bool = False,
    only_with_metadata: bool = False,
    max_tree_size: int | None = None,
    url_blacklist: set[str] | None = None,
    author_blacklist: set[str] | None = None,
    as_dict: Any = _UNSET,
    prune_xpath: str | list[str] | None = None,
    config: configparser.ConfigParser = DEFAULT_CONFIG,
    options: Any = None,
    *,
    compat: str = TRAFILATURA_VERSION,
) -> Any:
    """Extract a page's text and trees: ``trafilatura.bare_extraction``.

    The arguments are :func:`extract`'s, with ``output_format="python"`` (native; the others
    are delegated) and the deprecated ``as_dict`` (``DeprecationWarning``, then
    ``Document.as_dict()``).

    Returns:
        A :class:`Document` (``text`` is not NFC-normalized, as in trafilatura), its
        ``as_dict()`` with ``as_dict=True``, or ``None``.

    Raises:
        ValueError: As for :func:`extract`.
        pyroid.PyroidError: As for :func:`extract`.
    """
    args = {k: v for k, v in locals().items() if k != "compat"}
    mode = _mode(compat)
    if as_dict is _UNSET:  # each release's default: 2.3.1 False, 1.11.0 True
        as_dict = args["as_dict"] = mode == V1_11
    if mode == V1_11:
        return _bare_111(args)
    reason = _option_reason(args, python_format=True)
    if reason is not None:
        real = _trafilatura("bare_extraction", reason, mode)
        return _call_real(real.bare_extraction, _pyroid_cache(deduplicate), args, mode)
    fast = _check_deprecation(fast, no_fallback, as_dict, max_tree_size)
    settings = _Settings(config, url, date_extraction_params)
    reason = settings.reason() or _bytes_reason(filecontent, settings)
    if reason is not None:
        real = _trafilatura("bare_extraction", reason, mode)
        delegated = {**args, "fast": fast, "no_fallback": False, "as_dict": False}
        doc = _call_real(real.bare_extraction, _pyroid_cache(deduplicate), delegated, mode)
        return doc.as_dict() if as_dict and doc is not None else doc
    if not isinstance(filecontent, (str, bytes)):
        return None  # trafilatura: TypeError("incompatible input type"), caught
    opts = settings.options(mode, args, fast)
    parts = _native.bare_extraction(filecontent, opts, _pyroid_cache(deduplicate))
    if parts is None:
        return None
    doc = _document(parts, mode)
    return doc.as_dict() if as_dict else doc


def extract_with_metadata(
    filecontent: Any,
    url: str | None = None,
    record_id: str | None = None,
    fast: bool = False,
    favor_precision: bool = False,
    favor_recall: bool = False,
    include_comments: bool = True,
    output_format: str = "txt",
    tei_validation: bool = False,
    target_language: str | None = None,
    include_tables: bool = True,
    include_images: bool = False,
    include_formatting: bool | None = None,
    include_links: bool = False,
    deduplicate: Any = False,
    date_extraction_params: dict[str, Any] | None = None,
    url_blacklist: set[str] | None = None,
    author_blacklist: set[str] | None = None,
    settingsfile: str | None = None,
    prune_xpath: str | list[str] | None = None,
    config: configparser.ConfigParser = DEFAULT_CONFIG,
    options: Any = None,
    *,
    compat: str = TRAFILATURA_VERSION,
) -> Any:
    """``trafilatura.extract_with_metadata``: always delegated (metadata extraction).

    Returns trafilatura's ``Document``. Needs trafilatura 2.3.1; a pyroid :class:`LRUCache`
    passed as ``deduplicate`` is lent to it and updated.

    Raises:
        pyroid.PyroidError: trafilatura is not installed, or is another version.
        ValueError: ``compat="1.11.0"`` (trafilatura 1.11.0 has no ``extract_with_metadata``).
    """
    args = {k: v for k, v in locals().items() if k != "compat"}
    mode = _mode(compat, (TRAFILATURA_VERSION,))
    real = _trafilatura("extract_with_metadata", "metadata extraction", mode)
    return _call_real(real.extract_with_metadata, _pyroid_cache(deduplicate), args, mode)


# ---------------------------------------------------------------------------
# extract_batch
# ---------------------------------------------------------------------------


# trafilatura 1.11.0's extract() rejects these (NameError) and ignores any other unknown keyword.
_DEPRECATED_111 = ("csv_output", "json_output", "tei_output", "xml_output")


def _extract_many(
    htmls: Any,
    kwargs: dict[str, Any],
    threads: Any,
    compat: str,
    raise_errors: bool,
    task_cache: Any = None,
) -> list[Any]:
    """:func:`extract` over `htmls`, in order; errors raised, or returned in their place.

    ``task_cache`` stands for the module-level dedup cache (the datatrove step's per-task one).
    """
    if task_cache is None:
        return _extract_many_in(htmls, kwargs, threads, compat, raise_errors)
    token = _TASK_CACHE.set(task_cache)
    try:
        return _extract_many_in(htmls, kwargs, threads, compat, raise_errors)
    finally:
        _TASK_CACHE.reset(token)


def _extract_many_in(
    htmls: Any, kwargs: dict[str, Any], threads: Any, compat: str, raise_errors: bool
) -> list[Any]:
    mode = _mode(compat)
    _native.check_threads(threads=threads)
    if not isinstance(htmls, (list, tuple)):
        from pyroid._pyroid import InputError

        raise InputError(f"htmls must be a list or tuple, got {type(htmls).__name__}")
    unknown = sorted(set(kwargs) - set(_EXTRACT_DEFAULTS))
    error: Exception | None = None
    if mode == V1_11:
        # 1.11.0's extract() takes **kwargs: unknown keywords (2.3.1's fast too) are ignored
        if any(k in kwargs for k in _DEPRECATED_111):
            error = NameError(
                'Deprecated argument: use output_format instead, e.g. output_format="xml"'
            )
        kwargs = {k: v for k, v in kwargs.items() if k in _EXTRACT_DEFAULTS and k != "fast"}
    elif unknown:
        error = TypeError(f"extract() got an unexpected keyword argument {unknown[0]!r}")
    if error is not None:  # what each extract() call would raise
        if raise_errors:
            raise error
        return [error] * len(htmls)
    args = {**_EXTRACT_DEFAULTS, **kwargs, "filecontent": None}

    def one(html: Any) -> Any:
        return extract(html, **kwargs, compat=mode)

    def loop() -> list[Any]:
        if raise_errors:
            return [one(html) for html in htmls]
        out: list[Any] = []
        for html in htmls:
            try:
                out.append(one(html))
            except Exception as e:
                out.append(e)
        return out

    if not all(type(html) is str for html in htmls):
        return loop()
    if mode == V1_11:
        if _option_reason_111(args, python_format=False) is not None:
            return loop()
        try:
            config = _use_config_111(args["settingsfile"], args["config"])
            settings111 = _Settings111(config, args["date_extraction_params"])
        except Exception as e:
            if raise_errors:
                raise
            return [e] * len(htmls)  # each extract() call would raise it
        if settings111.reason(args["max_tree_size"]) is not None:
            return loop()
        opts = settings111.options(args)
    else:
        if _option_reason(args, python_format=False) is not None or args["no_fallback"]:
            return loop()
        try:
            fast, settings, reason = _prepare_extract(args, mode)
        except Exception as e:
            if raise_errors:
                raise
            return [e] * len(htmls)  # each extract() call would raise it
        if reason is not None:
            return loop()
        opts = settings.options(mode, args, fast)
    cache = _pyroid_cache(args["deduplicate"], mode)
    out: list[Any] = _native.extract_batch(list(htmls), opts, cache, raise_errors, threads=threads)
    return out


def extract_batch(
    htmls: Any,
    *,
    threads: int | None = None,
    compat: str = TRAFILATURA_VERSION,
    **kwargs: Any,
) -> Any:
    """:func:`extract` over many pages, in order, on pyroid's thread pool without the GIL.

    Not part of trafilatura. ``extract_batch(htmls, **kwargs)`` equals
    ``[extract(html, **kwargs) for html in htmls]``, including with ``deduplicate``: with
    ``deduplicate=False`` the pages are extracted in parallel; with ``True`` or an
    :class:`LRUCache` they are parsed in parallel and extracted one after the other in order
    through the cache, so the result (and the cache afterwards) is the loop's. A batch that holds
    a non-str item (``bytes``, as :func:`extract` takes them), or uses a delegated option or
    ``no_fallback``, runs that loop.

    Args:
        htmls: A list or tuple of pages (str), an Arrow string array/column (pyarrow, polars,
            ``pyroid.Column``, ...), or one page (str).
        threads: ``None`` uses pyroid's whole thread pool; ``n`` at most ``n`` tasks at once;
            ``1`` the calling thread only.
        compat: As for :func:`extract`.
        **kwargs: :func:`extract`'s keyword arguments.

    Returns:
        A list (also for a tuple or Arrow input) with one text or ``None`` per page, in order;
        ``None`` for each Arrow null. For a str, ``extract(htmls, **kwargs)``.

    Raises:
        ValueError: As for :func:`extract` (the note names the page); ``threads <= 0``.
        TypeError: An argument :func:`extract` does not take.
        pyroid.InputError: ``htmls`` is not a str, list, tuple or Arrow string column, or a list
            or tuple holds ``None`` (the message gives its index).
    """
    from pyroid._pyroid import InputError

    if isinstance(htmls, str):
        return _extract_many([htmls], kwargs, threads, compat, raise_errors=True)[0]
    if isinstance(htmls, (list, tuple)):
        for i, html in enumerate(htmls):
            if html is None:
                err = InputError(f"expected str at index {i}, got NoneType")
                err.add_note("while processing 'htmls'")
                raise err
        return _extract_many(htmls, kwargs, threads, compat, raise_errors=True)
    if not (hasattr(htmls, "__arrow_c_stream__") or hasattr(htmls, "__arrow_c_array__")):
        _native.check_threads(threads=threads)
        err = InputError(
            "htmls must be a str, a list or tuple of str, or an Arrow string array, got "
            f"{type(htmls).__name__}"
        )
        err.add_note("while processing 'htmls'")
        raise err
    pages = _native.arrow_pages(htmls)
    keep = [i for i, h in enumerate(pages) if h is not None]
    if len(keep) == len(pages):
        return _extract_many(pages, kwargs, threads, compat, raise_errors=True)
    try:
        texts = _extract_many([pages[i] for i in keep], kwargs, threads, compat, raise_errors=True)
    except Exception as e:  # its note names the page by its index among the non-null ones
        notes = getattr(e, "__notes__", [])
        for k, note in enumerate(notes):
            m = _PAGE_NOTE.fullmatch(note) if isinstance(note, str) else None
            if m is not None:
                notes[k] = f"{m[1]}{keep[int(m[2])]}"
        raise
    out: list[str | None] = [None] * len(pages)
    for i, text in zip(keep, texts, strict=True):
        out[i] = text
    return out


#: The note ``_native.extract_batch`` adds to an error, naming the page.
_PAGE_NOTE = re.compile(r"(pyroid: while extracting the document at index )(\d+)")


# ---------------------------------------------------------------------------
# Per-version functions with trafilatura's exact signatures (pyroid.integrations.patch)
# ---------------------------------------------------------------------------


def _api(version: str, config: Any = None) -> dict[str, Callable[..., Any]]:
    """The functions of one trafilatura release, with its exact signatures (no ``compat``).

    2.3.1: ``extract``, ``bare_extraction``, ``extract_with_metadata``; 1.11.0: ``extract``
    and ``bare_extraction``. ``config`` is their default ``config`` (the real trafilatura's
    DEFAULT_CONFIG under a patch).
    """
    mode = _mode(version)
    default = DEFAULT_CONFIG if config is None else config
    if mode == V1_11:
        return _api_111(default if config is not None else DEFAULT_CONFIGS[V1_11])

    def extract(
        filecontent: Any,
        url: str | None = None,
        record_id: str | None = None,
        fast: bool = False,
        no_fallback: bool = False,
        favor_precision: bool = False,
        favor_recall: bool = False,
        include_comments: bool = True,
        output_format: str = "txt",
        tei_validation: bool = False,
        target_language: str | None = None,
        include_tables: bool = True,
        include_images: bool = False,
        include_formatting: bool | None = None,
        include_links: bool = False,
        deduplicate: Any = False,
        date_extraction_params: dict[str, Any] | None = None,
        with_metadata: bool = False,
        only_with_metadata: bool = False,
        max_tree_size: int | None = None,
        url_blacklist: set[str] | None = None,
        author_blacklist: set[str] | None = None,
        settingsfile: str | None = None,
        prune_xpath: str | list[str] | None = None,
        config: configparser.ConfigParser = default,
        options: Any = None,
    ) -> str | None:
        kw = dict(locals())
        kw.pop("mode", None)
        return _extract_fn(**kw, compat=mode)

    def bare_extraction(
        filecontent: Any,
        url: str | None = None,
        fast: bool = False,
        no_fallback: bool = False,
        favor_precision: bool = False,
        favor_recall: bool = False,
        include_comments: bool = True,
        output_format: str = "python",
        target_language: str | None = None,
        include_tables: bool = True,
        include_images: bool = False,
        include_formatting: bool | None = None,
        include_links: bool = False,
        deduplicate: Any = False,
        date_extraction_params: dict[str, Any] | None = None,
        with_metadata: bool = False,
        only_with_metadata: bool = False,
        max_tree_size: int | None = None,
        url_blacklist: set[str] | None = None,
        author_blacklist: set[str] | None = None,
        as_dict: bool = False,
        prune_xpath: str | list[str] | None = None,
        config: configparser.ConfigParser = default,
        options: Any = None,
    ) -> Any:
        kw = dict(locals())
        kw.pop("mode", None)
        return _bare_fn(**kw, compat=mode)

    def extract_with_metadata(
        filecontent: Any,
        url: str | None = None,
        record_id: str | None = None,
        fast: bool = False,
        favor_precision: bool = False,
        favor_recall: bool = False,
        include_comments: bool = True,
        output_format: str = "txt",
        tei_validation: bool = False,
        target_language: str | None = None,
        include_tables: bool = True,
        include_images: bool = False,
        include_formatting: bool | None = None,
        include_links: bool = False,
        deduplicate: Any = False,
        date_extraction_params: dict[str, Any] | None = None,
        url_blacklist: set[str] | None = None,
        author_blacklist: set[str] | None = None,
        settingsfile: str | None = None,
        prune_xpath: str | list[str] | None = None,
        config: configparser.ConfigParser = default,
        options: Any = None,
    ) -> Any:
        kw = dict(locals())
        kw.pop("mode", None)
        return _metadata_fn(**kw, compat=mode)

    fns: dict[str, Callable[..., Any]] = {
        "extract": extract,
        "bare_extraction": bare_extraction,
        "extract_with_metadata": extract_with_metadata,
    }
    for name, fn in fns.items():
        fn.__qualname__ = name
        fn.__doc__ = (
            f"``trafilatura.{name}`` (trafilatura {mode}'s signature), run by "
            f"``pyroid.trafilatura.{name}(..., compat={mode!r})``."
        )
    return fns


def _api_111(default: Any) -> dict[str, Callable[..., Any]]:
    """Trafilatura 1.11.0's ``extract`` and ``bare_extraction``, with its exact signatures."""

    def extract(
        filecontent: Any,
        url: str | None = None,
        record_id: str | None = None,
        no_fallback: bool = False,
        favor_precision: bool = False,
        favor_recall: bool = False,
        include_comments: bool = True,
        output_format: str = "txt",
        tei_validation: bool = False,
        target_language: str | None = None,
        include_tables: bool = True,
        include_images: bool = False,
        include_formatting: bool = False,
        include_links: bool = False,
        deduplicate: Any = False,
        date_extraction_params: dict[str, Any] | None = None,
        with_metadata: bool = False,
        only_with_metadata: bool = False,
        max_tree_size: int | None = None,
        url_blacklist: set[str] | None = None,
        author_blacklist: set[str] | None = None,
        settingsfile: str | None = None,
        prune_xpath: str | list[str] | None = None,
        config: configparser.ConfigParser = default,
        options: Any = None,
        **kwargs: Any,
    ) -> str | None:
        kw = dict(locals())
        extra = kw.pop("kwargs")
        if any(k in extra for k in _DEPRECATED_111):
            raise NameError(
                'Deprecated argument: use output_format instead, e.g. output_format="xml"'
            )
        return _extract_fn(**kw, compat=V1_11)  # other keywords are ignored, as in 1.11.0

    def bare_extraction(
        filecontent: Any,
        url: str | None = None,
        no_fallback: bool = False,
        favor_precision: bool = False,
        favor_recall: bool = False,
        include_comments: bool = True,
        output_format: str = "python",
        target_language: str | None = None,
        include_tables: bool = True,
        include_images: bool = False,
        include_formatting: bool = False,
        include_links: bool = False,
        deduplicate: Any = False,
        date_extraction_params: dict[str, Any] | None = None,
        with_metadata: bool = False,
        only_with_metadata: bool = False,
        max_tree_size: int | None = None,
        url_blacklist: set[str] | None = None,
        author_blacklist: set[str] | None = None,
        as_dict: bool = True,
        prune_xpath: str | list[str] | None = None,
        config: configparser.ConfigParser = default,
        options: Any = None,
    ) -> Any:
        return _bare_fn(**locals(), compat=V1_11)

    fns: dict[str, Callable[..., Any]] = {"extract": extract, "bare_extraction": bare_extraction}
    for name, fn in fns.items():
        fn.__qualname__ = name
        fn.__doc__ = (
            f"``trafilatura.{name}`` (trafilatura 1.11.0's signature), run by "
            f"``pyroid.trafilatura.{name}(..., compat='1.11.0')``."
        )
    return fns


# The signatures shown are 2.3.1's (plus compat): extract's **kwargs (for 1.11.0) and
# bare_extraction's mode-dependent as_dict default are implementation details.
def _shown(fn: Callable[..., Any], **defaults: Any) -> inspect.Signature:
    sig = inspect.signature(fn)
    params = [
        p.replace(default=defaults[p.name]) if p.name in defaults else p
        for p in sig.parameters.values()
        if p.kind is not inspect.Parameter.VAR_KEYWORD
    ]
    return sig.replace(parameters=params)


cast(Any, extract).__signature__ = _shown(extract)
cast(Any, bare_extraction).__signature__ = _shown(bare_extraction, as_dict=False)

_extract_fn = extract
_bare_fn = bare_extraction
_metadata_fn = extract_with_metadata
