# Derived from PyYAML 6.0.3 (MIT), © Kirill Simonov and Ingy döt Net; modified.
"""A drop-in, Rust-backed replacement for PyYAML 6.0.3's safe loader and dumper.

``import pyroid.yaml as yaml`` and use ``yaml.safe_load``, ``yaml.safe_load_all``,
``yaml.safe_dump``, ``yaml.safe_dump_all``, or ``yaml.load``/``yaml.dump`` with a safe
``Loader``/``Dumper``. Results, dumped text and errors are those of PyYAML's pure-Python
``SafeLoader`` and ``SafeDumper`` (not ``CSafeLoader``, which differs in places).

With PyYAML installed, the exception and ``Mark`` classes are PyYAML's own, and anything the
Rust port does not handle (other loaders' Python tags, customised loaders and dumpers, the
event and token APIs) is delegated to PyYAML with a one-time :class:`pyroid.FallbackWarning`.
Without PyYAML those calls raise :class:`pyroid.PyroidError`.
"""

from __future__ import annotations

import codecs
import datetime
import re
import sys
import warnings
from collections.abc import Iterator
from typing import Any

from pyroid._pyroid import FallbackWarning, PyroidError
from pyroid._pyroid import yaml as _native
from pyroid.yaml import _construct

#: The PyYAML release whose behaviour this module reproduces.
PYYAML_VERSION = "6.0.3"
__version__ = PYYAML_VERSION
__with_libyaml__ = False


def _find_pyyaml() -> Any:
    try:
        import yaml as real
    except ImportError:
        return None
    # Under pyroid.patch("yaml") without PyYAML, `yaml` is pyroid's shim around nothing.
    if getattr(real, "__pyroid__", False):
        real = getattr(real, "__wrapped__", None)
    return real


_yaml: Any = _find_pyyaml()

if _yaml is not None:
    from yaml.composer import ComposerError
    from yaml.constructor import ConstructorError
    from yaml.emitter import EmitterError
    from yaml.error import Mark, MarkedYAMLError, YAMLError
    from yaml.parser import ParserError
    from yaml.reader import ReaderError
    from yaml.representer import RepresenterError
    from yaml.resolver import ResolverError
    from yaml.scanner import ScannerError
    from yaml.serializer import SerializerError
else:
    from pyroid.yaml._errors import (
        ComposerError,
        ConstructorError,
        EmitterError,
        Mark,
        MarkedYAMLError,
        ParserError,
        ReaderError,
        RepresenterError,
        ResolverError,
        ScannerError,
        SerializerError,
        YAMLError,
    )

__all__ = [
    "PYYAML_VERSION",
    "BaseLoader",
    "ComposerError",
    "ConstructorError",
    "Dumper",
    "EmitterError",
    "FullLoader",
    "Loader",
    "Mark",
    "MarkedYAMLError",
    "ParserError",
    "ReaderError",
    "RepresenterError",
    "ResolverError",
    "SafeDumper",
    "SafeLoader",
    "ScannerError",
    "SerializerError",
    "UnsafeLoader",
    "YAMLError",
    "dump",
    "dump_all",
    "full_load",
    "full_load_all",
    "load",
    "load_all",
    "safe_dump",
    "safe_dump_all",
    "safe_load",
    "safe_load_all",
    "unsafe_load",
    "unsafe_load_all",
]

_native._configure(
    {
        "Mark": Mark,
        "ReaderError": ReaderError,
        "ScannerError": ScannerError,
        "ParserError": ParserError,
        "ComposerError": ComposerError,
        "ConstructorError": ConstructorError,
        "RepresenterError": RepresenterError,
        "EmitterError": EmitterError,
        "encodebytes": __import__("base64").encodebytes,
        "nan": _construct.nan_value,
        "date": datetime.date,
        "datetime": datetime.datetime,
        "timedelta": datetime.timedelta,
        "timezone": datetime.timezone,
        "utc": datetime.timezone.utc,
        "surrogates": {0x10F800 + i: 0xD800 + i for i in range(0x800)},
        "helpers": _construct.HELPERS,
    }
)

# ---------------------------------------------------------------------------
# Loader and dumper classes
# ---------------------------------------------------------------------------


class _Placeholder:
    """Stands for one of PyYAML's loader or dumper classes when PyYAML is not installed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise PyroidError(
            f"pyroid.yaml.{type(self).__name__} is only a marker for load()/dump() here: "
            f"PyYAML is not installed (pip install pyyaml=={PYYAML_VERSION})"
        )


if _yaml is not None:
    SafeLoader = _yaml.SafeLoader
    BaseLoader = _yaml.BaseLoader
    FullLoader = _yaml.FullLoader
    Loader = _yaml.Loader
    UnsafeLoader = _yaml.UnsafeLoader
    SafeDumper = _yaml.SafeDumper
    Dumper = _yaml.Dumper
else:

    class SafeLoader(_Placeholder):  # type: ignore[no-redef]
        """PyYAML's ``SafeLoader``: pass it to :func:`load`."""

    class BaseLoader(_Placeholder):  # type: ignore[no-redef]
        """PyYAML's ``BaseLoader`` (needs PyYAML)."""

    class FullLoader(_Placeholder):  # type: ignore[no-redef]
        """PyYAML's ``FullLoader``: pass it to :func:`load`."""

    class Loader(_Placeholder):  # type: ignore[no-redef]
        """PyYAML's ``Loader``: pass it to :func:`load`."""

    class UnsafeLoader(_Placeholder):  # type: ignore[no-redef]
        """PyYAML's ``UnsafeLoader``: pass it to :func:`load`."""

    class SafeDumper(_Placeholder):  # type: ignore[no-redef]
        """PyYAML's ``SafeDumper``: pass it to :func:`dump`."""

    class Dumper(_Placeholder):  # type: ignore[no-redef]
        """PyYAML's ``Dumper``: pass it to :func:`dump`."""


# Which loader classes run natively, and how: "safe" (SafeConstructor) or "full" (the
# loaders whose only extra constructors are for !!python/ tags: native until one shows up).
_LOADER_MODES: dict[Any, str] = {}
# vars(cls).keys() of each stock class at import: anything added later is a customisation.
_STOCK_VARS: dict[Any, frozenset[str]] = {}
_STOCK_CONSTRUCTORS: dict[Any, tuple[dict[Any, Any], dict[Any, Any]]] = {}
_STOCK_RESOLVERS: dict[Any, list[Any]] | None = None


def _snapshot() -> None:
    global _STOCK_RESOLVERS
    if _yaml is None:
        _LOADER_MODES[SafeLoader] = "safe"
        for cls in (FullLoader, Loader, UnsafeLoader):
            _LOADER_MODES[cls] = "full"
        return
    from yaml.resolver import Resolver

    resolvers = Resolver.yaml_implicit_resolvers
    expected = {
        "tag:yaml.org,2002:bool",
        "tag:yaml.org,2002:float",
        "tag:yaml.org,2002:int",
        "tag:yaml.org,2002:merge",
        "tag:yaml.org,2002:null",
        "tag:yaml.org,2002:timestamp",
        "tag:yaml.org,2002:value",
        "tag:yaml.org,2002:yaml",
    }
    tags = {tag for v in resolvers.values() for tag, _ in v}
    if tags != expected or sum(map(len, resolvers.values())) != 54:
        return  # already customised before pyroid.yaml was imported: always delegate
    _STOCK_RESOLVERS = {k: list(v) for k, v in resolvers.items()}
    candidates = [(_yaml.SafeLoader, "safe"), (_yaml.FullLoader, "full")]
    candidates += [(_yaml.Loader, "full"), (_yaml.UnsafeLoader, "full")]
    csafe = getattr(_yaml, "CSafeLoader", None)
    if csafe is not None:
        candidates.append((csafe, "safe"))
    for cls, mode in candidates:
        names = frozenset(vars(cls))
        if names & _CUSTOM_ATTRS or any(not _inert(n) for n in names):
            continue
        _LOADER_MODES[cls] = mode
        _STOCK_VARS[cls] = names
        _STOCK_CONSTRUCTORS[cls] = (dict(cls.yaml_constructors), dict(cls.yaml_multi_constructors))


_CUSTOM_ATTRS = frozenset(
    {
        "yaml_constructors",
        "yaml_multi_constructors",
        "yaml_implicit_resolvers",
        "yaml_path_resolvers",
    }
)
# Names a plain `class X(SafeLoader): pass` has; anything else is an override.
_INERT = frozenset(
    {
        "__module__",
        "__qualname__",
        "__doc__",
        "__dict__",
        "__weakref__",
        "__firstlineno__",
        "__static_attributes__",
        "__annotations__",
        "__annotate__",
        "__annotations_cache__",
        "__orig_bases__",
        "__parameters__",
        "__abstractmethods__",
        "_abc_impl",
    }
)


def _inert(name: str) -> bool:
    return name in _INERT or name == "__init__"


_snapshot()


def _loader_mode(loader_cls: Any) -> str | None:
    """"safe"/"full" when `loader_cls` behaves as the stock class, else None (delegate)."""
    mode = _LOADER_MODES.get(loader_cls)
    if mode is not None:
        if _yaml is None:
            return mode
        if vars(loader_cls).keys() != _STOCK_VARS[loader_cls]:
            return None
        base = loader_cls
    else:
        if not isinstance(loader_cls, type):
            return None
        # A subclass that adds nothing (no registrations, no overrides, no __init__).
        base = None
        for cls in loader_cls.__mro__[1:]:
            if cls in _LOADER_MODES:
                base = cls
                break
        if base is None or _yaml is None:
            return None
        for cls in loader_cls.__mro__:
            if cls is base:
                break
            if any(n not in _INERT for n in vars(cls)):
                return None
        if vars(base).keys() != _STOCK_VARS[base]:
            return None
        mode = _LOADER_MODES[base]
    ctors, multi = _STOCK_CONSTRUCTORS[base]
    if loader_cls.yaml_constructors != ctors or loader_cls.yaml_multi_constructors != multi:
        return None
    if loader_cls.yaml_implicit_resolvers != _STOCK_RESOLVERS or loader_cls.yaml_path_resolvers:
        return None
    return mode


# ---------------------------------------------------------------------------
# Delegation
# ---------------------------------------------------------------------------

_WARNED: set[str] = set()


def _caller_stacklevel() -> int:
    """The ``stacklevel`` that points ``warnings.warn`` at the first frame outside this module."""
    frame = sys._getframe(1)
    level = 1
    while frame.f_back is not None and frame.f_globals.get("__name__") == __name__:
        frame = frame.f_back
        level += 1
    return level


def _pyyaml(name: str, reason: str) -> Any:
    """The real PyYAML for `name`, warning once per function and reason, or PyroidError."""
    if _yaml is None:
        raise PyroidError(
            f"pyroid.yaml.{name}: {reason} is not supported by pyroid's Rust port, and PyYAML "
            f"is not installed to fall back to (pip install pyyaml=={PYYAML_VERSION})"
        )
    if "version" not in _WARNED:
        _WARNED.add("version")
        version = getattr(_yaml, "__version__", None)
        if version != PYYAML_VERSION:
            warnings.warn(
                f"pyroid.yaml is falling back to PyYAML {version}, not {PYYAML_VERSION}",
                RuntimeWarning,
                stacklevel=_caller_stacklevel(),
            )
    key = f"{name}:{reason}"
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(
            f"pyroid.yaml.{name}: {reason} is not supported by the Rust port; delegating to "
            "PyYAML (slow path). This warning is shown once.",
            FallbackWarning,
            stacklevel=_caller_stacklevel(),
        )
    return _yaml


def _loader_reason(loader_cls: Any) -> str:
    name = getattr(loader_cls, "__name__", type(loader_cls).__name__)
    if loader_cls in _LOADER_MODES or any(c in _LOADER_MODES for c in getattr(loader_cls, "__mro__", ())):
        return f"Loader={name} with added constructors, resolvers or overrides"
    return f"Loader={name}"


def _py_load(stream: Any, loader_cls: Any) -> Any:
    """PyYAML's ``yaml.load``."""
    loader = loader_cls(stream)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


def _py_load_all(stream: Any, loader_cls: Any) -> Iterator[Any]:
    """PyYAML's ``yaml.load_all``."""
    loader = loader_cls(stream)
    try:
        while loader.check_data():
            yield loader.get_data()
    finally:
        loader.dispose()


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------

_NON_PRINTABLE = re.compile("[^\x09\x0a\x0d\x20-\x7e\x85\xa0-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")


def _printable_error(text: str, name: Any, offset: int = 0) -> ReaderError | None:
    match = _NON_PRINTABLE.search(text)
    if match is None:
        return None
    return ReaderError(
        name, offset + match.start(), ord(match.group()), "unicode", "special characters are not allowed"
    )


def _decode_bytes(data: bytes, name: str) -> str:
    """PyYAML's reader on a ``bytes`` stream with a UTF-16 BOM."""
    decode = codecs.utf_16_le_decode if data.startswith(codecs.BOM_UTF16_LE) else codecs.utf_16_be_decode
    try:
        text, _ = decode(data, "strict", True)
    except UnicodeDecodeError as exc:
        raise ReaderError(name, exc.start, data[exc.start], exc.encoding, exc.reason) from None
    return text


def _read_all(stream: Any) -> list[Any]:
    """Every ``stream.read(4096)`` result, the final empty one included."""
    reads = []
    while True:
        data = stream.read(4096)
        reads.append(data)
        if not data:
            return reads


def _emulate_reader(reads: list[Any], name: Any) -> tuple[str, ReaderError | None]:
    """PyYAML's ``Reader`` over a file object, reading it all.

    Returns the text PyYAML decodes before its first error and that error (``None`` when the
    whole stream is fine). PyYAML raises the error only once the scanner looks past that text.
    """
    it = iter(reads)
    raw: Any = None
    eof = False
    pointer = 0

    def update_raw() -> None:
        nonlocal raw, eof, pointer
        data = next(it)
        raw = data if raw is None else raw + data
        pointer += len(data)
        if not data:
            eof = True

    while not eof and (raw is None or len(raw) < 2):
        update_raw()
    decode = None
    if isinstance(raw, bytes):
        if raw.startswith(codecs.BOM_UTF16_LE):
            decode = codecs.utf_16_le_decode
        elif raw.startswith(codecs.BOM_UTF16_BE):
            decode = codecs.utf_16_be_decode
        else:
            decode = codecs.utf_8_decode
    parts: list[str] = []
    chars = 0
    while True:
        if not eof:
            update_raw()
        if decode is not None:
            try:
                data, converted = decode(raw, "strict", eof)
            except UnicodeDecodeError as exc:
                character = raw[exc.start]
                position = pointer - len(raw) + exc.start
                return "".join(parts), ReaderError(name, position, character, exc.encoding, exc.reason)
        else:
            data = raw
            converted = len(data)
        error = _printable_error(data, name, chars)
        if error is not None:
            return "".join(parts), error
        parts.append(data)
        chars += len(data)
        raw = raw[converted:]
        if eof:
            return "".join(parts), None


def _max_depth() -> int:
    """How deep PyYAML's recursive composer could nest from here before ``RecursionError``."""
    depth = 0
    frame: Any = sys._getframe(1)
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return max((sys.getrecursionlimit() - depth - 12) // 2, 0)


class _Replay:
    """A file object's ``read(4096)`` results again, for handing the stream to PyYAML."""

    def __init__(self, reads: list[Any], stream: Any) -> None:
        self._reads = iter(reads)
        self._last = reads[-1]
        if hasattr(stream, "name"):
            self.name = stream.name

    def read(self, size: int = -1) -> Any:
        return next(self._reads, self._last)


def _native_load(stream: Any, mode: str) -> tuple[Any, Any]:
    """Load one document natively: ``(result, stream)``. A ``NotImplemented`` result means
    PyYAML must load it, from the returned stream (file objects are replayed)."""
    reads: list[Any] = []
    result = _native_load_one(stream, mode, reads)
    if result is NotImplemented and reads:
        return result, _Replay(reads, stream)
    return result, stream


def _native_load_one(stream: Any, mode: str, reads: list[Any]) -> Any:
    """`reads` receives a file object's read() results."""
    full = mode == "full"
    depth = _max_depth()
    digits = sys.get_int_max_str_digits()
    if isinstance(stream, str):
        name = "<unicode string>"
        try:
            return _native.load(stream, name, True, None, None, depth, digits, full)
        except UnicodeEncodeError:
            error = _printable_error(stream, name)
            if error is None:
                raise
            raise error from None
    if isinstance(stream, bytes):
        name = "<byte string>"
        if stream.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            return _native.load(_decode_bytes(stream, name), name, True, None, None, depth, digits, full)
        return _native.load(stream, name, True, None, None, depth, digits, full)
    name = getattr(stream, "name", "<file>")
    reads.extend(_read_all(stream))
    first = reads[0]
    data = first[:0].join(reads) if isinstance(first, (str, bytes)) else "".join(reads)
    if isinstance(data, bytes) and data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        text, pending = _emulate_reader(reads, name)
        limit = len(text) if pending is not None else None
        return _native.load(text, name, False, limit, pending, depth, digits, full)
    try:
        return _native.load(data, name, False, None, None, depth, digits, full)
    except (ReaderError, UnicodeEncodeError):
        text, pending = _emulate_reader(reads, name)
        if pending is None:
            raise
        return _native.load(text, name, False, len(text), pending, depth, digits, full)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load(stream: Any, Loader: Any) -> Any:  # noqa: N803 - PyYAML's name
    """Parse the first YAML document in `stream` and produce the Python object.

    Like ``yaml.load``. A ``SafeLoader`` (or ``CSafeLoader``, with ``SafeLoader``'s results)
    runs natively; so do ``FullLoader``, ``Loader`` and ``UnsafeLoader`` until a
    ``!!python/`` tag appears, when PyYAML takes over. Customised loaders go to PyYAML.
    """
    mode = _loader_mode(Loader)
    if mode is not None:
        result, again = _native_load(stream, mode)
        if result is not NotImplemented:
            return result
        reason = "a !!python/ tag" if mode == "full" else "a str holding U+10F800..U+10FFFF"
        _pyyaml("load", reason)
        return _py_load(again, Loader)
    _pyyaml("load", _loader_reason(Loader))
    return _py_load(stream, Loader)


def load_all(stream: Any, Loader: Any) -> Iterator[Any]:  # noqa: N803 - PyYAML's name
    """Parse all YAML documents in `stream` and produce the Python objects (a generator).

    Like ``yaml.load_all``. A ``SafeLoader`` runs natively; other loaders go to PyYAML.
    """
    mode = _loader_mode(Loader)
    if mode != "safe":
        _pyyaml("load_all", _loader_reason(Loader))
        yield from _py_load_all(stream, Loader)
        return
    loader = _native_load_all(stream)
    if loader is NotImplemented:
        _pyyaml("load_all", "a str holding U+10F800..U+10FFFF")
        yield from _py_load_all(stream, Loader)
        return
    while loader.check_data():
        yield loader.get_data()


def _native_load_all(stream: Any) -> Any:
    depth = _max_depth()
    digits = sys.get_int_max_str_digits()
    start = _native._LoadAll.start
    if isinstance(stream, str):
        name = "<unicode string>"
        try:
            return start(stream, name, True, None, None, depth, digits)
        except UnicodeEncodeError:
            error = _printable_error(stream, name)
            if error is None:
                raise
            raise error from None
    if isinstance(stream, bytes):
        name = "<byte string>"
        if stream.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            return start(_decode_bytes(stream, name), name, True, None, None, depth, digits)
        return start(stream, name, True, None, None, depth, digits)
    name = getattr(stream, "name", "<file>")
    reads = _read_all(stream)
    first = reads[0]
    data = first[:0].join(reads) if isinstance(first, (str, bytes)) else "".join(reads)
    if isinstance(data, bytes) and data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        text, pending = _emulate_reader(reads, name)
        return start(text, name, False, len(text) if pending else None, pending, depth, digits)
    try:
        return start(data, name, False, None, None, depth, digits)
    except (ReaderError, UnicodeEncodeError):
        text, pending = _emulate_reader(reads, name)
        if pending is None:
            raise
        return start(text, name, False, len(text), pending, depth, digits)


def safe_load(stream: Any) -> Any:
    """Parse the first YAML document in `stream` and produce the Python object.

    Resolve only basic YAML tags (``yaml.safe_load``). `stream` is a ``str``, ``bytes``
    (UTF-8, or UTF-16 with a BOM) or a file object.
    """
    return load(stream, SafeLoader)


def safe_load_all(stream: Any) -> Iterator[Any]:
    """Parse all YAML documents in `stream` and produce the Python objects (``yaml.safe_load_all``)."""
    return load_all(stream, SafeLoader)


def full_load(stream: Any) -> Any:
    """``yaml.full_load``: :func:`load` with ``FullLoader``."""
    return load(stream, FullLoader)


def full_load_all(stream: Any) -> Iterator[Any]:
    """``yaml.full_load_all``: :func:`load_all` with ``FullLoader``."""
    return load_all(stream, FullLoader)


def unsafe_load(stream: Any) -> Any:
    """``yaml.unsafe_load``: :func:`load` with ``UnsafeLoader``."""
    return load(stream, UnsafeLoader)


def unsafe_load_all(stream: Any) -> Iterator[Any]:
    """``yaml.unsafe_load_all``: :func:`load_all` with ``UnsafeLoader``."""
    return load_all(stream, UnsafeLoader)


# ---------------------------------------------------------------------------
# Dumping
# ---------------------------------------------------------------------------

_DUMPER_MODES: dict[Any, str] = {}
_STOCK_REPRESENTERS: dict[Any, tuple[dict[Any, Any], dict[Any, Any]]] = {}


def _snapshot_dumpers() -> None:
    if _yaml is None:
        _DUMPER_MODES[SafeDumper] = "safe"
        _DUMPER_MODES[Dumper] = "full"
        return
    if _STOCK_RESOLVERS is None:
        return
    candidates = [(_yaml.SafeDumper, "safe"), (_yaml.Dumper, "full")]
    for name, mode in (("CSafeDumper", "safe"), ("CDumper", "full")):
        cls = getattr(_yaml, name, None)
        if cls is not None:
            candidates.append((cls, mode))
    for cls, mode in candidates:
        names = frozenset(vars(cls))
        if names & _CUSTOM_DUMPER_ATTRS or any(not _inert(n) for n in names):
            continue
        _DUMPER_MODES[cls] = mode
        _STOCK_VARS[cls] = names
        _STOCK_REPRESENTERS[cls] = (dict(cls.yaml_representers), dict(cls.yaml_multi_representers))


_CUSTOM_DUMPER_ATTRS = frozenset(
    {
        "yaml_representers",
        "yaml_multi_representers",
        "yaml_implicit_resolvers",
        "yaml_path_resolvers",
    }
)

_snapshot_dumpers()


def _dumper_mode(dumper_cls: Any) -> str | None:
    """"safe"/"full" when `dumper_cls` behaves as the stock class, else None (delegate)."""
    mode = _DUMPER_MODES.get(dumper_cls)
    if mode is not None:
        if _yaml is None:
            return mode
        if vars(dumper_cls).keys() != _STOCK_VARS[dumper_cls]:
            return None
        base = dumper_cls
    else:
        if not isinstance(dumper_cls, type) or _yaml is None:
            return None
        base = next((c for c in dumper_cls.__mro__[1:] if c in _DUMPER_MODES), None)
        if base is None:
            return None
        for cls in dumper_cls.__mro__:
            if cls is base:
                break
            if any(n not in _INERT for n in vars(cls)):
                return None
        if vars(base).keys() != _STOCK_VARS[base]:
            return None
        mode = _DUMPER_MODES[base]
    reps, multi = _STOCK_REPRESENTERS[base]
    if dumper_cls.yaml_representers != reps or dumper_cls.yaml_multi_representers != multi:
        return None
    if dumper_cls.yaml_implicit_resolvers != _STOCK_RESOLVERS or dumper_cls.yaml_path_resolvers:
        return None
    return mode


# Codecs whose encode() of the whole text equals the concatenation of encoding each write.
_STATELESS_CODECS = frozenset(
    {
        "utf-8",
        "ascii",
        "latin-1",
        "iso8859-1",
        "utf-16-le",
        "utf-16-be",
        "utf-32-le",
        "utf-32-be",
        "cp1252",
    }
)


def _dump_options(
    default_style: Any,
    default_flow_style: Any,
    canonical: Any,
    indent: Any,
    width: Any,
    allow_unicode: Any,
    line_break: Any,
    explicit_start: Any,
    explicit_end: Any,
    version: Any,
    tags: Any,
    sort_keys: Any,
) -> dict[str, Any] | str:
    """The native options, as PyYAML's Emitter/Serializer/Representer read them, or the reason
    they need PyYAML."""
    if default_style is not None and (type(default_style) is not str or default_style not in ("", '"', "'", "|", ">")):
        return f"default_style={default_style!r}"
    best_indent = 2
    if indent and 1 < indent < 10:
        if type(indent) is not int:
            return f"indent={indent!r}"
        best_indent = indent
    best_width = 80
    if width and width > best_indent * 2:
        if type(width) is int:
            best_width = min(width, 2**64 - 1)
        elif type(width) is float:
            best_width = 2**64 - 1 if width == float("inf") else int(width)
        else:
            return f"width={width!r}"
    if version is not None:
        if not (
            type(version) in (tuple, list)
            and len(version) == 2
            and all(type(v) is int and 0 <= v < 2**63 for v in version)
        ):
            return f"version={version!r}"
        version = tuple(version)
    if tags is not None:
        if type(tags) is not dict or not all(type(k) is str and type(v) is str for k, v in tags.items()):
            return "tags that are not a dict of str"
        tags = sorted(tags.items()) or None
    return {
        "default_style": default_style,
        "default_flow_style": None if default_flow_style is None else bool(default_flow_style),
        "canonical": bool(canonical),
        "best_indent": best_indent,
        "best_width": best_width,
        "allow_unicode": bool(allow_unicode),
        "line_break": line_break if line_break in ("\r", "\n", "\r\n") else "\n",
        "explicit_start": bool(explicit_start),
        "explicit_end": bool(explicit_end),
        "version": version,
        "tags": tags,
        "sort_keys": bool(sort_keys),
    }


def _dump_depth() -> int:
    """How deep PyYAML's recursive representer could nest from here."""
    depth = 0
    frame: Any = sys._getframe(1)
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return max((sys.getrecursionlimit() - depth - 10) // 3, 0)


def _py_dump_all(documents: Any, stream: Any, dumper_cls: Any, **kwds: Any) -> Any:
    """PyYAML's ``yaml.dump_all``."""
    import io

    getvalue = None
    if stream is None:
        stream = io.StringIO() if kwds.get("encoding") is None else io.BytesIO()
        getvalue = stream.getvalue
    dumper = dumper_cls(stream, **kwds)
    try:
        dumper.open()
        for data in documents:
            dumper.represent(data)
        dumper.close()
    finally:
        dumper.dispose()
    if getvalue:
        return getvalue()
    return None


def dump_all(
    documents: Any,
    stream: Any = None,
    Dumper: Any = Dumper,  # noqa: N803 - PyYAML's name
    default_style: Any = None,
    default_flow_style: Any = False,
    canonical: Any = None,
    indent: Any = None,
    width: Any = None,
    allow_unicode: Any = None,
    line_break: Any = None,
    encoding: Any = None,
    explicit_start: Any = None,
    explicit_end: Any = None,
    version: Any = None,
    tags: Any = None,
    sort_keys: Any = True,
) -> Any:
    """Serialize a sequence of Python objects into a YAML stream (``yaml.dump_all``).

    If `stream` is None, return the produced ``str`` (``bytes`` with `encoding`). A
    ``SafeDumper`` runs natively; so does ``Dumper`` for plain data (dicts, lists, tuples,
    sets, str, bytes, numbers, booleans, None, dates). Customised dumpers go to PyYAML.
    """
    kwds = {
        "default_style": default_style,
        "default_flow_style": default_flow_style,
        "canonical": canonical,
        "indent": indent,
        "width": width,
        "allow_unicode": allow_unicode,
        "line_break": line_break,
        "encoding": encoding,
        "explicit_start": explicit_start,
        "explicit_end": explicit_end,
        "version": version,
        "tags": tags,
        "sort_keys": sort_keys,
    }
    mode = _dumper_mode(Dumper)
    if mode is None:
        name = getattr(Dumper, "__name__", type(Dumper).__name__)
        _pyyaml("dump_all", f"Dumper={name}" + ("" if Dumper in _DUMPER_MODES else " (customised)"))
        return _py_dump_all(documents, stream, Dumper, **kwds)
    opts = _dump_options(
        default_style,
        default_flow_style,
        canonical,
        indent,
        width,
        allow_unicode,
        line_break,
        explicit_start,
        explicit_end,
        version,
        tags,
        sort_keys,
    )
    if isinstance(opts, str) or (encoding is not None and type(encoding) is not str) or encoding == "":
        _pyyaml("dump_all", opts if isinstance(opts, str) else f"encoding={encoding!r}")
        return _py_dump_all(documents, stream, Dumper, **kwds)
    # The emitter writes bytes only when the stream has no `encoding` (BytesIO when stream=None).
    effective = encoding if encoding and (stream is None or not hasattr(stream, "encoding")) else None
    per_write = False
    if effective is not None:
        try:
            per_write = codecs.lookup(effective).name not in _STATELESS_CODECS
        except LookupError:
            per_write = True
    it = iter(documents)
    text, writes, error, consumed, reason = _native.dump(
        it,
        full=mode == "full",
        encoding=effective,
        record_writes=per_write,
        max_depth=_dump_depth(),
        **opts,
    )
    if error is NotImplemented:
        _pyyaml("dump_all", reason)
        import itertools

        return _py_dump_all(itertools.chain(consumed, it), stream, Dumper, **kwds)
    if effective is not None and writes is None:
        try:
            encoded = text.encode(effective)
        except UnicodeEncodeError:
            # PyYAML encodes each write: the error comes from the first failing write.
            _, writes, _, _, _ = _native.dump(
                consumed,
                full=mode == "full",
                encoding=effective,
                record_writes=True,
                max_depth=_dump_depth(),
                **opts,
            )
            encoded = None
    else:
        encoded = None
    if stream is None:
        if effective is None:
            if error is not None:
                raise error
            return text if encoding is None else _encode_chunks(text, None, encoding)
        result = encoded if encoded is not None else _encode_chunks(text, writes, effective)
        if error is not None:
            raise error
        return result
    if effective is None:
        if text:
            stream.write(text)
    elif encoded is not None:
        if encoded:
            stream.write(encoded)
    else:
        start = 0
        for end in writes:
            stream.write(text[start:end].encode(effective))
            start = end
    if error is not None:
        raise error
    if hasattr(stream, "flush"):
        stream.flush()
    return None


def _encode_chunks(text: str, writes: list[int] | None, encoding: str) -> bytes:
    if writes is None:
        return text.encode(encoding)
    out = []
    start = 0
    for end in writes:
        out.append(text[start:end].encode(encoding))
        start = end
    return b"".join(out)


def dump(data: Any, stream: Any = None, Dumper: Any = Dumper, **kwds: Any) -> Any:  # noqa: N803
    """Serialize a Python object into a YAML stream (``yaml.dump``)."""
    return dump_all([data], stream, Dumper=Dumper, **kwds)


def safe_dump_all(documents: Any, stream: Any = None, **kwds: Any) -> Any:
    """``yaml.safe_dump_all``: :func:`dump_all` with ``SafeDumper``."""
    return dump_all(documents, stream, Dumper=SafeDumper, **kwds)


def safe_dump(data: Any, stream: Any = None, **kwds: Any) -> Any:
    """``yaml.safe_dump``: :func:`dump_all` of one document with ``SafeDumper``."""
    return dump_all([data], stream, Dumper=SafeDumper, **kwds)


def __getattr__(name: str) -> Any:
    """The rest of PyYAML's API (the event and token APIs, ``YAMLObject``, ...)."""
    if _yaml is not None and not name.startswith("__"):
        return getattr(_yaml, name)
    raise AttributeError(
        f"module 'pyroid.yaml' has no attribute {name!r}"
        + ("" if _yaml is not None else f" (PyYAML provides it: pip install pyyaml=={PYYAML_VERSION})"),
        name=name,
    )
