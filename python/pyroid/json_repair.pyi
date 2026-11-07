# Derived from json_repair 0.64.5 (MIT), © Stefano Baccianella; modified.
# Type stubs for pyroid.json_repair: the signatures of json_repair 0.64.5. The module
# docstring is kept here on purpose: the API docs are built from this stub.
"""A drop-in, Rust-backed replacement for json_repair 0.64.5 (``import pyroid.json_repair``).

``repair_json``, ``loads``, ``load`` and ``from_file`` take json_repair's
arguments and return the same strings, objects and repair logs. Valid JSON comes
back as ``json.dumps(json.loads(text))``. Broken JSON (LLM output with missing
quotes, single quotes, comments, trailing commas, truncation, markdown fences,
prose around it, Python literals, ...) is repaired with json_repair's heuristics,
ported line by line to Rust. The port runs in linear time on the inputs where
json_repair's lookaheads are quadratic, and never overflows the stack.

These are delegated to the real json_repair, with a one-time
``pyroid.FallbackWarning``: every ``schema=`` call, non-str input, and the rare
text holding a char in U+10F800..U+10FFFF (the range the port uses to store lone
surrogates). Without json_repair installed they raise ``pyroid.PyroidError``.

Differences from json_repair
----------------------------
- json_repair raises ``ValueError`` past about 140-330 levels of broken input
  (it depends on ``sys.getrecursionlimit()`` and the caller's stack). pyroid
  repairs up to :data:`MAX_DEPTH` levels, and deeper broken input raises
  :class:`NestingError`, which is both a ``ValueError`` and a ``RecursionError``.
- Valid JSON is decoded by ``json.loads`` for objects, as in json_repair, so its
  ``RecursionError`` past about 10,000 levels on 3.12 is the same. For string
  results pyroid decodes it at any depth (json_repair raises there too).
- ``load``/``from_file`` read the file once (``fd.read()``) instead of in
  ``chunk_length`` pieces. The result is the same, ``chunk_length`` is ignored and
  unseekable streams work.
- json_repair's command line (``python -m json_repair``) is not part of this module.
"""

from pathlib import Path
from typing import Any, Final, Literal, TextIO, TypeAlias, overload

from pyroid._pyroid import Column, PyroidError
from pyroid._typing import ArrowStrings

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

JSON_REPAIR_VERSION: Final[str]
DELEGATED: Final[tuple[str, ...]]
MAX_DEPTH: Final[int]

JSONReturnType: TypeAlias = dict[str, Any] | list[Any] | str | float | int | bool | None

class NestingError(PyroidError, ValueError, RecursionError):
    """The input nests deeper than :data:`MAX_DEPTH` levels."""

@overload
def repair_json(
    json_str: str = "",
    return_objects: Literal[False] = False,
    skip_json_loads: bool = False,
    logging: Literal[False] = False,
    json_fd: TextIO | None = None,
    chunk_length: int = 0,
    stream_stable: bool = False,
    strict: bool = False,
    schema: Any | None = None,
    schema_repair_mode: Literal["standard", "salvage"] = "standard",
    **json_dumps_args: Any,
) -> str:
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

@overload
def repair_json(
    json_str: str = "",
    return_objects: Literal[True] = True,
    skip_json_loads: bool = False,
    logging: Literal[False] = False,
    json_fd: TextIO | None = None,
    chunk_length: int = 0,
    stream_stable: bool = False,
    strict: bool = False,
    schema: Any | None = None,
    schema_repair_mode: Literal["standard", "salvage"] = "standard",
    **json_dumps_args: Any,
) -> JSONReturnType: ...
@overload
def repair_json(
    json_str: str = "",
    return_objects: bool = False,
    skip_json_loads: bool = False,
    logging: Literal[True] = True,
    json_fd: TextIO | None = None,
    chunk_length: int = 0,
    stream_stable: bool = False,
    strict: bool = False,
    schema: Any | None = None,
    schema_repair_mode: Literal["standard", "salvage"] = "standard",
    **json_dumps_args: Any,
) -> tuple[JSONReturnType, list[dict[str, str]]]: ...
@overload
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
    schema_repair_mode: Literal["standard", "salvage"] = "standard",
    **json_dumps_args: Any,
) -> str | JSONReturnType | tuple[JSONReturnType, list[dict[str, str]]]: ...
def loads(
    json_str: str,
    skip_json_loads: bool = False,
    logging: bool = False,
    stream_stable: bool = False,
    strict: bool = False,
    schema: Any | None = None,
    schema_repair_mode: Literal["standard", "salvage"] = "standard",
) -> JSONReturnType | tuple[JSONReturnType, list[dict[str, str]]] | str:
    """Like ``json.loads``, but repairs the text first (``json_repair.loads``)."""

def load(
    fd: TextIO,
    skip_json_loads: bool = False,
    logging: bool = False,
    chunk_length: int = 0,
    strict: bool = False,
    schema: Any | None = None,
    schema_repair_mode: Literal["standard", "salvage"] = "standard",
) -> JSONReturnType | tuple[JSONReturnType, list[dict[str, str]]]:
    """Like ``json.load``, but repairs the text first (``json_repair.load``)."""

def from_file(
    filename: str | Path,
    skip_json_loads: bool = False,
    logging: bool = False,
    chunk_length: int = 0,
    strict: bool = False,
    schema: Any | None = None,
    schema_repair_mode: Literal["standard", "salvage"] = "standard",
) -> JSONReturnType | tuple[JSONReturnType, list[dict[str, str]]]:
    """Read and repair the JSON in `filename` (``json_repair.from_file``)."""

@overload
def repair_json_batch(
    texts: str,
    *,
    threads: int | None = None,
    skip_json_loads: bool = False,
    stream_stable: bool = False,
    strict: bool = False,
    **json_dumps_args: Any,
) -> str:
    """:func:`repair_json` over many documents at once, on pyroid's thread pool.

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
        ValueError: A document fails in strict mode (the note names its index).
        NestingError: A document nests deeper than :data:`MAX_DEPTH`.
        pyroid.InputError: ``texts`` is not one of the accepted kinds.
    """

@overload
def repair_json_batch(
    texts: list[str],
    *,
    threads: int | None = None,
    skip_json_loads: bool = False,
    stream_stable: bool = False,
    strict: bool = False,
    **json_dumps_args: Any,
) -> list[str]: ...
@overload
def repair_json_batch(
    texts: tuple[str, ...],
    *,
    threads: int | None = None,
    skip_json_loads: bool = False,
    stream_stable: bool = False,
    strict: bool = False,
    **json_dumps_args: Any,
) -> tuple[str, ...]: ...
@overload
def repair_json_batch(
    texts: ArrowStrings,
    *,
    threads: int | None = None,
    skip_json_loads: bool = False,
    stream_stable: bool = False,
    strict: bool = False,
    **json_dumps_args: Any,
) -> Column[str]: ...
