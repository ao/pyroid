# Derived from json_repair 0.64.9 (MIT), © Stefano Baccianella; modified.
"""A drop-in, Rust-backed replacement for json_repair 0.64.9 (``import pyroid.json_repair``).

``repair_json``, ``loads``, ``load`` and ``from_file`` have json_repair's
signatures and return the same strings and objects. The repair runs in Rust,
linear on most of the inputs that make json_repair quadratic. ``schema=`` calls,
str subclasses that override what json_repair calls on them, a replaced
``json.JSONDecoder`` and the rare inputs the port cannot represent are delegated
to the real json_repair with a one-time :class:`pyroid.FallbackWarning`, or raise
:class:`pyroid.PyroidError` if it is not installed. See ``json_repair.pyi``.
"""

from __future__ import annotations

import contextlib
import json
import json.decoder
import sys
import unicodedata
import warnings
from pathlib import Path
from typing import Any, TextIO

from pyroid import _versions
from pyroid._pyroid import FallbackWarning, PyroidError
from pyroid._pyroid import json_repair as _native

__all__ = [
    "DELEGATED",
    "JSON_REPAIR_VERSION",
    "MAX_DEPTH",
    "JSONReturnType",
    "NestingError",
    "from_file",
    "load",
    "loads",
    "repair_json",
    "repair_json_batch",
]

#: The json_repair release whose behaviour this module reproduces.
JSON_REPAIR_VERSION = "0.64.9"
_versions.check(__name__, "json_repair", f"json-repair=={JSON_REPAIR_VERSION}")

#: Arguments that are always forwarded to the real json_repair (it must be installed).
DELEGATED = ("schema",)

#: Deepest nesting the Rust parser follows; deeper input raises :class:`NestingError`.
MAX_DEPTH: int = _native.MAX_DEPTH

#: Raised for input nested deeper than :data:`MAX_DEPTH` (a ``ValueError`` and a
#: ``RecursionError``: json_repair raises one or the other past its own, lower, limit).
NestingError: type[Exception] = _native.NestingError

JSONReturnType = dict[str, Any] | list[Any] | str | float | int | bool | None


def _unicode_version() -> str:
    """The interpreter's Unicode version, or the newest one the kernel knows."""
    version = unicodedata.unidata_version
    supported = _native.supported_unicode_versions()
    if version in supported:
        return version
    newest = supported[-1]
    warnings.warn(
        f"pyroid.json_repair has no tables for Unicode {version} (this Python's unicodedata); "
        f"using Unicode {newest}. Output can differ from json_repair for characters added "
        f"after {newest}.",
        RuntimeWarning,
        stacklevel=3,
    )
    return newest


_native._set_unicode_version(_unicode_version())

_MODE_STRING = 0
_MODE_OBJECTS = 1
_MODE_LOGGING = 2

_WARNED: set[str] = set()


def _caller_stacklevel() -> int:
    """The ``stacklevel`` that points ``warnings.warn`` at the first frame outside this module."""
    frame = sys._getframe(1)
    level = 1
    while frame.f_back is not None and frame.f_globals.get("__name__") == __name__:
        frame = frame.f_back
        level += 1
    return level


def _json_repair(name: str, reason: str) -> Any:
    """Import the real json_repair for `name`, or raise PyroidError."""
    try:
        import json_repair as real
    except ImportError:
        real = None
    # Under a pyroid patch shim, fall back to the real module it wraps, never to the shim.
    if getattr(real, "__pyroid__", False):
        real = real.__wrapped__
    if real is None:
        raise PyroidError(
            f"pyroid.json_repair.{name}: pyroid's Rust port does not handle {reason}, and "
            f"json_repair is not installed to fall back to (pip install "
            f"json-repair=={JSON_REPAIR_VERSION})"
        )
    return real


def _fallback_warning(name: str, reason: str) -> None:
    """Warn once per `name` and `reason` that the call goes to json_repair."""
    if f"{name}:{reason}" not in _WARNED:
        _WARNED.add(f"{name}:{reason}")
        warnings.warn(
            f"pyroid.json_repair.{name}: the Rust port does not handle {reason}; "
            "delegating to json_repair (slow path). This warning is shown once.",
            FallbackWarning,
            stacklevel=_caller_stacklevel(),
        )


_RESERVED = "a char in U+10F800..U+10FFFF"


def _stdlib_json(name: str) -> Any:
    """``json.<name>`` if it is the standard library's own, else ``None``.

    ``None`` (it was replaced before this module was imported) never matches, so such a
    program always takes the Python route.
    """
    obj = getattr(json, name)
    if isinstance(obj, type):
        own = obj.__module__ == "json.decoder" and obj is vars(json.decoder).get(name)
    else:
        own = getattr(obj, "__globals__", None) is vars(json)
    return obj if own else None


# json_repair looks these up on the json module at each call, so a program that replaces one
# (an orjson-backed json.dumps, say) changes its results. The Rust port reproduces the originals:
# while one of them is replaced, calls take the Python route (json.loads and json.dumps called by
# this module, as json_repair calls them) or are delegated (json.JSONDecoder, json.load).
_JSON_LOADS = _stdlib_json("loads")
_JSON_LOAD = _stdlib_json("load")
_JSON_DUMPS = _stdlib_json("dumps")
_JSON_DECODER = _stdlib_json("JSONDecoder")


def _str_subclass(text: Any) -> str | None:
    """The delegation reason for a str subclass overriding a method json_repair calls on it."""
    name = _native._overridden_str_hook(text)
    return None if name is None else f"a str subclass overriding {name}()"


def _normalize_schema_repair_mode(mode: Any) -> str:
    if mode is None or mode == "standard":
        return "standard"
    if mode == "salvage":
        return "salvage"
    raise ValueError("schema_repair_mode must be one of: standard, salvage.")


#: `json.dumps` keyword arguments the Rust encoder reproduces; others go through `json.dumps`.
_NATIVE_DUMPS = frozenset(
    {"skipkeys", "ensure_ascii", "check_circular", "allow_nan", "indent", "separators", "default"}
    | {"sort_keys", "cls"}
)
_PYTHON_DUMPS = object()
_DUMP_CACHE: dict[tuple[Any, ...], Any] = {}


def _dump_config(kwargs: dict[str, Any]) -> Any:
    """The native dump config for ``json.dumps(**kwargs)``.

    ``None`` stands for the defaults, ``_PYTHON_DUMPS`` for kwargs only ``json.dumps``
    itself can interpret.
    """
    if not kwargs:
        return None
    if not kwargs.keys() <= _NATIVE_DUMPS or kwargs.get("cls") is not None:
        return _PYTHON_DUMPS
    # skipkeys, check_circular and default never matter: keys are str, values JSON types.
    indent = kwargs.get("indent")
    if indent is not None and not isinstance(indent, str):
        if not isinstance(indent, int):
            return _PYTHON_DUMPS
        indent = " " * indent
    separators = kwargs.get("separators")
    if separators is not None:
        try:
            item_separator, key_separator = separators
        except (TypeError, ValueError):
            return _PYTHON_DUMPS
        if type(item_separator) is not str or type(key_separator) is not str:
            return _PYTHON_DUMPS
    elif indent is None:
        item_separator, key_separator = ", ", ": "
    else:
        item_separator, key_separator = ",", ": "
    key = (
        bool(kwargs.get("ensure_ascii", True)),
        bool(kwargs.get("allow_nan", True)),
        bool(kwargs.get("sort_keys", False)),
        indent,
        item_separator,
        key_separator,
    )
    config = _DUMP_CACHE.get(key)
    if config is None:
        config = _native._DumpConfig(
            ensure_ascii=key[0],
            allow_nan=key[1],
            sort_keys=key[2],
            indent=key[3],
            item_separator=key[4],
            key_separator=key[5],
        )
        if len(_DUMP_CACHE) > 64:
            _DUMP_CACHE.clear()
        _DUMP_CACHE[key] = config
    return config


def repair_json(
    json_str: str = "",
    return_objects: bool = False,
    skip_json_loads: bool = False,
    logging: bool = False,
    json_fd: TextIO | None = None,
    chunk_length: int = 0,
    stream_stable: bool = False,
    strict: bool = False,
    schema: Any | None = None,
    schema_repair_mode: str = "standard",
    **json_dumps_args: Any,
) -> Any:
    """Repair a JSON string, like ``json_repair.repair_json``.

    Args:
        json_str: The text to repair.
        return_objects: Return the decoded value instead of a JSON string.
        skip_json_loads: Skip the ``json.loads`` validity check (valid JSON then
            goes through the repair parser too, as in json_repair).
        logging: Return ``(value, repair_log)``; the log is a list of
            ``{"text": ..., "context": ...}`` dicts, empty when nothing was repaired.
        json_fd: Read the text from this file instead (prefer :func:`load`).
        chunk_length: Ignored (json_repair's file chunk size).
        stream_stable: Keep repairs of a growing streamed prefix stable.
        strict: Raise ``ValueError`` on structural problems instead of repairing them.
        schema: A JSON Schema or pydantic model guiding the repair (delegated to
            json_repair; needs json_repair, and jsonschema/pydantic as it does).
        schema_repair_mode: ``"standard"`` or ``"salvage"`` (requires ``schema``).
        **json_dumps_args: Passed to ``json.dumps`` for the string result
            (``ensure_ascii``, ``indent``, ``separators``, ``sort_keys``,
            ``allow_nan`` are reproduced in Rust; others call ``json.dumps``).

    Returns:
        The repaired JSON string (``""`` when nothing was found), the value with
        ``return_objects``, or ``(value, log)`` with ``logging``.

    Raises:
        ValueError: In strict mode, for bad ``schema_repair_mode`` combinations,
            or from ``int()`` limits, as json_repair does.
        NestingError: The input nests deeper than :data:`MAX_DEPTH`.
    """
    mode = _normalize_schema_repair_mode(schema_repair_mode)
    if schema is None and mode == "salvage":
        raise ValueError("schema_repair_mode='salvage' requires schema.")
    if schema is not None and strict:
        raise ValueError("schema and strict cannot be used together.")
    if json_fd is not None and not json_fd.seekable():
        # As json_repair does: a stream that cannot be rewound is read and repaired as a str.
        json_str = json_fd.read()
        json_fd = None
    position = None
    if json_fd is not None:
        position = _tell(json_fd)
        text = json_fd.read()
    else:
        text = json_str
    if (
        schema is not None
        or type(text) is not str
        or (json_fd is not None and not skip_json_loads and json.load is not _JSON_LOAD)
    ):
        reason = _early_reason(text, schema, json_fd, skip_json_loads)
        if reason is not None:
            return _delegate_repair_json(
                reason,
                json_dumps_args,
                position,
                json_str,
                return_objects,
                skip_json_loads,
                logging,
                json_fd,
                chunk_length,
                stream_stable,
                strict,
                schema,
                schema_repair_mode,
            )
    if logging:
        out_mode = _MODE_LOGGING
        dump = None
    elif return_objects:
        out_mode = _MODE_OBJECTS
        dump = None
    else:
        dump = _dump_config(json_dumps_args)
        if dump is _PYTHON_DUMPS:
            out_mode = _MODE_OBJECTS
        elif (
            json.dumps is _JSON_DUMPS
            and json.JSONDecoder is _JSON_DECODER
            and (skip_json_loads or json.loads is _JSON_LOADS)
        ):
            out_mode = _MODE_STRING
        else:
            # A json function json_repair calls is not the standard library's: take its steps
            # here, calling them (the repair in between still runs in Rust).
            out_mode, dump = _MODE_OBJECTS, _PYTHON_DUMPS
    if out_mode != _MODE_STRING and not skip_json_loads:
        # json_repair's own first step. For a value (not a string) CPython's json.loads is the
        # fastest exact answer; only invalid JSON goes on to the Rust parser.
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError):
            skip_json_loads = True
        else:
            if out_mode == _MODE_LOGGING:
                return value, []
            if dump is _PYTHON_DUMPS:
                return json.dumps(value, **json_dumps_args)
            return value
    if out_mode == _MODE_STRING or json.JSONDecoder is _JSON_DECODER:
        result = _native.repair(
            text,
            out_mode,
            bool(skip_json_loads),
            bool(stream_stable),
            bool(strict),
            json_fd is not None,
            sys.get_int_max_str_digits(),
            None if dump is _PYTHON_DUMPS else dump,
        )
        reason = _RESERVED
    else:
        # json_repair decodes the valid stretches of broken input with json.JSONDecoder.
        result = NotImplemented
        reason = "a replaced json.JSONDecoder"
    if result is NotImplemented:
        return _delegate_repair_json(
            reason,
            json_dumps_args,
            position,
            json_str,
            return_objects,
            skip_json_loads,
            logging,
            json_fd,
            chunk_length,
            stream_stable,
            strict,
            schema,
            schema_repair_mode,
        )
    if dump is _PYTHON_DUMPS:
        return "" if result == "" else json.dumps(result, **json_dumps_args)
    return result


def _delegate_repair_json(
    reason: str,
    json_dumps_args: dict[str, Any],
    position: int | None,
    json_str: Any,
    return_objects: bool,
    skip_json_loads: bool,
    logging: bool,
    json_fd: Any,
    chunk_length: int,
    stream_stable: bool,
    strict: bool,
    schema: Any,
    schema_repair_mode: str,
) -> Any:
    """:func:`repair_json`'s call, delegated to json_repair."""
    return _delegate(
        "repair_json",
        reason,
        json_dumps_args,
        position,
        json_str=json_str,
        return_objects=return_objects,
        skip_json_loads=skip_json_loads,
        logging=logging,
        json_fd=json_fd,
        chunk_length=chunk_length,
        stream_stable=stream_stable,
        strict=strict,
        schema=schema,
        schema_repair_mode=schema_repair_mode,
    )


def _early_reason(text: Any, schema: Any, json_fd: Any, skip_json_loads: bool) -> str | None:
    """Why :func:`repair_json` delegates before it starts, if it does."""
    if schema is not None:
        return "schema"
    if not isinstance(text, str):
        return f"a {type(text).__name__} input"
    reason = _str_subclass(text)
    if reason is None and json_fd is not None and not skip_json_loads:
        reason = None if json.load is _JSON_LOAD else "a replaced json.load"
    return reason


def _tell(fd: Any) -> int | None:
    try:
        return int(fd.tell())
    except (AttributeError, OSError, ValueError):
        return None


#: ``repair_json``'s defaults, and the json_repair release that added each argument after
#: ``json_fd`` (``json.dumps`` arguments: ``ensure_ascii`` from 0.26.0, any from 0.47.1). A
#: delegated call passes its arguments by keyword and leaves out those at their default that
#: the installed release does not take, so older releases still run it.
_ARGUMENTS: dict[str, tuple[Any, str]] = {
    "return_objects": (False, ""),
    "skip_json_loads": (False, ""),
    "logging": (False, ""),
    "json_fd": (None, ""),
    "chunk_length": (0, "0.28.4"),
    "stream_stable": (False, "0.44.0"),
    "strict": (False, "0.54"),
    "schema": (None, "0.56.0"),
    "schema_repair_mode": ("standard", "0.58.0"),
}
_DUMPS_RELEASE = "0.47.1"
# The installed repair_json's keyword arguments, and whether it takes ``**kwargs``.
_SIGNATURES: dict[Any, tuple[frozenset[str], bool] | None] = {}


def _accepted(fn: Any) -> tuple[frozenset[str], bool] | None:
    """The argument names `fn` takes by keyword and whether it takes ``**kwargs`` (cached)."""
    try:
        return _SIGNATURES[fn]
    except (KeyError, TypeError):
        pass
    import inspect

    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        accepted = None  # no signature to read: pass everything and let it decide
    else:
        keyword = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        accepted = (
            frozenset(p.name for p in parameters if p.kind in keyword),
            any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters),
        )
    with contextlib.suppress(TypeError):
        _SIGNATURES[fn] = accepted
    return accepted


def _delegate(
    name: str,
    reason: str,
    json_dumps_args: dict[str, Any],
    position: int | None = None,
    **arguments: Any,
) -> Any:
    """Run the installed ``json_repair.repair_json`` with `arguments`, by keyword.

    Raises:
        PyroidError: json_repair is not installed, or its release does not take an argument
            the call sets (one left at its default is not passed).
    """
    real = _json_repair(name, reason)
    fn = real.repair_json
    accepted = _accepted(fn)
    if accepted is not None:
        names, any_keyword = accepted
        for key in [key for key in arguments if key not in names]:
            default, release = _ARGUMENTS.get(key, (None, ""))
            value = arguments.pop(key)
            if value is None if default is None else value == default:
                continue
            raise _too_old(name, reason, key, f"added in json-repair {release}" if release else "")
        if not any_keyword:
            for key in json_dumps_args:
                if key not in names:
                    added = f"json.dumps arguments were added in json-repair {_DUMPS_RELEASE}"
                    raise _too_old(name, reason, key, added)
    _fallback_warning(name, reason)
    json_fd = arguments.get("json_fd")
    if json_fd is not None and position is not None:
        json_fd.seek(position)
    return fn(**arguments, **json_dumps_args)


def _too_old(name: str, reason: str, key: str, added: str) -> PyroidError:
    added = f" ({added})" if added else ""
    return PyroidError(
        f"pyroid.json_repair.{name}: pyroid's Rust port does not handle {reason}, and the "
        f"installed json_repair, which it delegates to, has no {key}= argument{added}. "
        f"Install json-repair=={JSON_REPAIR_VERSION}, the release pyroid reproduces"
    )


def loads(
    json_str: str,
    skip_json_loads: bool = False,
    logging: bool = False,
    stream_stable: bool = False,
    strict: bool = False,
    schema: Any | None = None,
    schema_repair_mode: str = "standard",
) -> Any:
    """Like ``json.loads``, but repairs the text first (``json_repair.loads``)."""
    return repair_json(
        json_str=json_str,
        return_objects=True,
        skip_json_loads=skip_json_loads,
        logging=logging,
        stream_stable=stream_stable,
        strict=strict,
        schema=schema,
        schema_repair_mode=schema_repair_mode,
    )


def load(
    fd: TextIO,
    skip_json_loads: bool = False,
    logging: bool = False,
    chunk_length: int = 0,
    strict: bool = False,
    schema: Any | None = None,
    schema_repair_mode: str = "standard",
) -> Any:
    """Like ``json.load``, but repairs the text first (``json_repair.load``)."""
    return repair_json(
        json_fd=fd,
        chunk_length=chunk_length,
        return_objects=True,
        skip_json_loads=skip_json_loads,
        logging=logging,
        strict=strict,
        schema=schema,
        schema_repair_mode=schema_repair_mode,
    )


def from_file(
    filename: str | Path,
    skip_json_loads: bool = False,
    logging: bool = False,
    chunk_length: int = 0,
    strict: bool = False,
    schema: Any | None = None,
    schema_repair_mode: str = "standard",
) -> Any:
    """Read and repair the JSON in `filename` (``json_repair.from_file``)."""
    with Path(filename).open() as fd:
        return load(
            fd=fd,
            skip_json_loads=skip_json_loads,
            logging=logging,
            chunk_length=chunk_length,
            strict=strict,
            schema=schema,
            schema_repair_mode=schema_repair_mode,
        )


def repair_json_batch(
    texts: Any,
    *,
    threads: int | None = None,
    skip_json_loads: bool = False,
    stream_stable: bool = False,
    strict: bool = False,
    **json_dumps_args: Any,
) -> Any:
    r""":func:`repair_json` over many documents at once, on pyroid's thread pool.

    Not part of json_repair. Each result equals ``json_repair.repair_json(doc,
    skip_json_loads=..., stream_stable=..., strict=..., **json_dumps_args)``;
    a document whose repair equals its input comes back as the original object.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column
            (``string``, ``large_string`` or ``string_view``; nulls stay null).
        threads: ``None`` uses pyroid's shared thread pool; ``n`` runs at most
            ``n`` tasks at once; ``1`` runs on the calling thread.
        skip_json_loads: As for :func:`repair_json`.
        stream_stable: As for :func:`repair_json`.
        strict: As for :func:`repair_json`.
        **json_dumps_args: As for :func:`repair_json`.

    Returns:
        The same kind as ``texts``: str, list, tuple, or a :class:`pyroid.Column`
        of strings (``large_string`` for ``large_string`` input, else ``string``).

    Raises:
        ValueError: A document fails in strict mode: the first one that fails, as in a
            loop over ``repair_json`` (the note names its index or row).
        NestingError: A document nests deeper than :data:`MAX_DEPTH`.
        UnicodeEncodeError: Arrow input with ``ensure_ascii=False``, and a result holds a lone
            surrogate (json_repair keeps an escape such as ``\ud800`` as one), which an Arrow
            column cannot hold. The note names the row; a list gives json_repair's str.
        pyroid.InputError: ``texts`` is not one of the accepted kinds.
    """
    dump = _dump_config(json_dumps_args)
    python = (
        dump is _PYTHON_DUMPS
        or json.dumps is not _JSON_DUMPS
        or (not skip_json_loads and json.loads is not _JSON_LOADS)
        or json.JSONDecoder is not _JSON_DECODER
    )

    def one(doc: Any) -> Any:
        # The Python route, for json.dumps arguments the port does not reproduce, or json functions
        # this program replaced: one repair_json call per document.
        return repair_json(
            doc,
            skip_json_loads=skip_json_loads,
            stream_stable=stream_stable,
            strict=strict,
            **json_dumps_args,
        )

    def fallback(doc: str) -> Any:
        # A document the port does not handle: a char in _RESERVED, or a str subclass.
        return _delegate(
            "repair_json_batch",
            _str_subclass(doc) or _RESERVED,
            json_dumps_args,
            json_str=doc,
            skip_json_loads=skip_json_loads,
            stream_stable=stream_stable,
            strict=strict,
        )

    # Rust calls the delegate in document order, after the native documents ran, so the first
    # document that fails raises, as in a loop.
    return _native.repair_batch(
        texts,
        bool(skip_json_loads),
        bool(stream_stable),
        bool(strict),
        sys.get_int_max_str_digits(),
        None if python else dump,
        threads=threads,
        delegate=one if python else fallback,
        python=python,
    )
