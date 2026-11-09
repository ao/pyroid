# Derived from trafilatura 2.3.1 (Apache-2.0), © Adrien Barbaresi; modified.
"""A drop-in, Rust-backed replacement for trafilatura 2.3.1's text extraction.

``extract`` and ``bare_extraction`` take trafilatura's arguments and return the same text, byte
for byte, for HTML given as a str (or UTF-8 bytes): the whole pipeline (libxml2's HTML parser as
lxml 6.1.3 drives it, Trafilatura's cleaning, extractors and fallbacks, jusText 3.0.2) is ported
to Rust. Options the port does not cover are delegated to the installed trafilatura with a
one-time :class:`pyroid.FallbackWarning`, or raise :class:`pyroid.PyroidError` without it. See
``trafilatura.pyi``.
"""

from __future__ import annotations

import configparser
import importlib
import importlib.util
import sys
import threading
import unicodedata
import warnings
import zlib
from collections.abc import Callable, Mapping, Sequence
from html import unescape as _unescape
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pyroid._pyroid import FallbackWarning, PyroidError
from pyroid._pyroid import trafilatura as _native

# "Document" is built on first use (module __getattr__).
__all__ = [  # noqa: F822
    "DEFAULT_CONFIG",
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
    }
)

#: Functions that always run the real trafilatura (it must be installed).
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


def _default_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read_string(_SETTINGS_CFG)
    return config


#: trafilatura.settings.DEFAULT_CONFIG: settings.cfg's values, the default of ``config=``.
DEFAULT_CONFIG = _default_config()


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


def _mode(compat: object) -> str:
    if not isinstance(compat, str) or compat not in TRAFILATURA_VERSIONS:
        choices = ", ".join(map(repr, TRAFILATURA_VERSIONS))
        raise ValueError(f"compat must be one of {choices}, not {compat!r}")
    return compat


# ---------------------------------------------------------------------------
# Delegation
# ---------------------------------------------------------------------------

_WARNED: set[str] = set()
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
    key = f"{name}:{reason}"
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(
            f"pyroid.trafilatura.{name}: {reason} is not supported by the Rust port; "
            "delegating to trafilatura (slow path). This warning is shown once.",
            FallbackWarning,
            stacklevel=_caller_stacklevel(),
        )
    return real


def _is_real_lru(x: object) -> bool:
    module = _loaded("trafilatura.deduplication")
    return module is not None and isinstance(x, module.LRUCache)


def _pyroid_cache(deduplicate: object) -> Any:
    """The pyroid cache a call reads and updates: the argument, :data:`LRU_TEST`, or None."""
    if not deduplicate:
        return None
    if isinstance(deduplicate, LRUCache):
        return deduplicate
    if _is_real_lru(deduplicate):
        return None  # delegated with it
    return LRU_TEST


def _call_real(fn: Callable[..., Any], cache: Any, args: dict[str, Any]) -> Any:
    """``fn(**args)``; a pyroid dedup cache is lent to it as a trafilatura LRUCache."""
    if cache is None:
        return fn(**args)
    real = _real_module()
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
        if args["favor_recall"]:
            focus = "recall"
        elif args["favor_precision"]:
            focus = "precision"
        else:
            focus = "balanced"
        ext, out, out_comm, dupl, reps, max_tree = self.values
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
    return _call_real(real.extract, _pyroid_cache(args["deduplicate"]), args)


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
    config: configparser.ConfigParser = DEFAULT_CONFIG,
    options: Any = None,
    *,
    compat: str = TRAFILATURA_VERSION,
) -> str | None:
    """Extract the main text of an HTML page, like ``trafilatura.extract``.

    See ``trafilatura.pyi`` for the arguments; ``compat`` selects the trafilatura release to
    reproduce.
    """
    args = {k: v for k, v in locals().items() if k != "compat"}
    mode = _mode(compat)
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
    if etree is None:
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
    return element


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


_DOCUMENT: list[type] = []
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


def _document_class() -> type:
    """The class of bare_extraction's results (``Document``), built on first use.

    A subclass of ``trafilatura.settings.Document`` when trafilatura 2.3.1 is installed, else of
    a class with the same fields; ``body`` and ``commentsbody`` are parsed into lxml elements
    when first read.
    """
    if _DOCUMENT:
        return _DOCUMENT[0]
    with _DOCUMENT_LOCK:
        if _DOCUMENT:
            return _DOCUMENT[0]
        base: type = _DocumentFields
        try:
            found = importlib.util.find_spec("trafilatura") is not None
        except (ImportError, ValueError):
            found = False
        if found:
            real = _real_module()
            if real is not None and getattr(real, "__version__", None) == TRAFILATURA_VERSION:
                base = importlib.import_module(real.__name__ + ".settings").Document
        ns = {
            "__module__": __name__,
            "__qualname__": "Document",
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
        _DOCUMENT.append(cls)
        return cls


def __getattr__(name: str) -> Any:
    if name == "Document":
        return _document_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _document(parts: tuple[str, str | None, str, str]) -> Any:
    cls = _document_class()
    text, comments, body, commentsbody = parts
    doc = cls.__new__(cls)
    for slot in _PLAIN_SLOTS:
        setattr(doc, slot, None)
    doc.text = text
    doc.raw_text = text
    doc.comments = comments
    doc.body = _XML(body)
    doc.commentsbody = _XML(commentsbody)
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
    as_dict: bool = False,
    prune_xpath: str | list[str] | None = None,
    config: configparser.ConfigParser = DEFAULT_CONFIG,
    options: Any = None,
    *,
    compat: str = TRAFILATURA_VERSION,
) -> Any:
    """Extract a page's text and trees, like ``trafilatura.bare_extraction``.

    See ``trafilatura.pyi`` for the arguments and the :class:`Document` returned.
    """
    args = {k: v for k, v in locals().items() if k != "compat"}
    mode = _mode(compat)
    reason = _option_reason(args, python_format=True)
    if reason is not None:
        real = _trafilatura("bare_extraction", reason, mode)
        return _call_real(real.bare_extraction, _pyroid_cache(deduplicate), args)
    fast = _check_deprecation(fast, no_fallback, as_dict, max_tree_size)
    settings = _Settings(config, url, date_extraction_params)
    reason = settings.reason() or _bytes_reason(filecontent, settings)
    if reason is not None:
        real = _trafilatura("bare_extraction", reason, mode)
        delegated = {**args, "fast": fast, "no_fallback": False, "as_dict": False}
        doc = _call_real(real.bare_extraction, _pyroid_cache(deduplicate), delegated)
        return doc.as_dict() if as_dict and doc is not None else doc
    if not isinstance(filecontent, (str, bytes)):
        return None  # trafilatura: TypeError("incompatible input type"), caught
    opts = settings.options(mode, args, fast)
    parts = _native.bare_extraction(filecontent, opts, _pyroid_cache(deduplicate))
    if parts is None:
        return None
    doc = _document(parts)
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
    """``trafilatura.extract_with_metadata``: always delegated to trafilatura (metadata)."""
    args = {k: v for k, v in locals().items() if k != "compat"}
    real = _trafilatura("extract_with_metadata", "metadata extraction", _mode(compat))
    return _call_real(real.extract_with_metadata, _pyroid_cache(deduplicate), args)


# ---------------------------------------------------------------------------
# extract_batch
# ---------------------------------------------------------------------------


def _extract_many(
    htmls: Any, kwargs: dict[str, Any], threads: Any, compat: str, raise_errors: bool
) -> list[Any]:
    """:func:`extract` over `htmls`, in order; errors raised, or returned in their place."""
    mode = _mode(compat)
    _native.check_threads(threads=threads)
    if not isinstance(htmls, (list, tuple)):
        from pyroid._pyroid import InputError

        raise InputError(f"htmls must be a list or tuple, got {type(htmls).__name__}")
    unknown = sorted(set(kwargs) - set(_EXTRACT_DEFAULTS))
    if unknown:
        # what each extract() call would raise
        error = TypeError(f"extract() got an unexpected keyword argument {unknown[0]!r}")
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

    if (
        _option_reason(args, python_format=False) is not None
        or args["no_fallback"]
        or not all(type(html) is str for html in htmls)
    ):
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
    cache = _pyroid_cache(args["deduplicate"])
    out: list[Any] = _native.extract_batch(list(htmls), opts, cache, raise_errors, threads=threads)
    return out


def extract_batch(
    htmls: Sequence[str],
    *,
    threads: int | None = None,
    compat: str = TRAFILATURA_VERSION,
    **kwargs: Any,
) -> list[str | None]:
    """:func:`extract` over many pages, on pyroid's thread pool; see ``trafilatura.pyi``."""
    return _extract_many(htmls, kwargs, threads, compat, raise_errors=True)


# ---------------------------------------------------------------------------
# Per-version functions with trafilatura's exact signatures (pyroid.integrations.patch)
# ---------------------------------------------------------------------------


def _api(version: str, config: Any = None) -> dict[str, Callable[..., Any]]:
    """``extract``, ``bare_extraction`` and ``extract_with_metadata`` for one trafilatura release.

    They have trafilatura `version`'s exact signatures (no ``compat``) and reproduce it;
    ``config`` is their default ``config`` (trafilatura's own DEFAULT_CONFIG under a patch).
    """
    mode = _mode(version)
    default = DEFAULT_CONFIG if config is None else config

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


_extract_fn = extract
_bare_fn = bare_extraction
_metadata_fn = extract_with_metadata
