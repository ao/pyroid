"""Type-exact encoding, comparison and outcome capture for the ``json_repair`` parity suite (stdlib only).

``json_repair.repair_json`` returns a ``str`` (byte-identical parity), an object (``return_objects=True``), an
``(object, log)`` tuple (``logging=True``) or raises. Parity compares all of these exactly:

* **values**: ``same(a, b)`` is deep equality that also requires equal types (``1``, ``1.0`` and ``True`` all
  differ), NaN equal to NaN, ``-0.0`` different from ``0.0``, equal dict key *order*, and lists different from
  tuples;
* **exceptions**: same type name and same message;
* **warnings**: the same warning categories, leaving out ``pyroid.FallbackWarning`` (pyroid's "delegating to
  the incumbent" notice, see ``pyroid_parity.impl``).

``encode`` / ``decode`` round-trip such values through JSON without losing any of that: non-finite floats,
ints outside the int64 range (as hex, so the int-to-str digit limit never applies), tuples, dicts whose keys
are not plain strings, strings with lone surrogates (as UTF-8 ``surrogatepass`` hex, because a JSON decoder
would join an escaped surrogate pair into one character) and long strings with runs of a repeated unit
(``"x" * 5 MiB``, ``"# comment\\n" * 5000`` become ``{"$cat": [prefix, {"$rep": [unit, count]}, ...]}``) are
tagged objects with a single ``"$..."`` key. A plain JSON object never has a key starting with ``$``.

The recorded upstream calls (``parity/data/json_repair-0.64.9/upstream-calls.jsonl``) use this encoding, and
file arguments as placeholders: ``{"$textio": {"text": ..., "pos": n}}`` is an ``io.StringIO`` positioned at
``n`` (``load(fd)`` / ``json_fd=``; with ``"seekable": false``, a stream that cannot seek);
``{"$path": {"text": ..., "type": "Path" | "str"}}`` (or ``"bytes": hex`` when not UTF-8) is a file holding those
bytes (``from_file(filename)``). ``replay`` materialises them.

This module is also loaded by path from ``python/tests`` (no package imports, no third-party imports).
"""

from __future__ import annotations

import io
import json
import math
import re
import tempfile
import warnings
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FORMAT = "pyroid-json_repair-calls/1"
FALLBACK_WARNING = "pyroid.FallbackWarning"
_INT64 = 1 << 63
_SURROGATE = re.compile("[\ud800-\udfff]")
# In strings of >= _RUNS_MIN characters, each run of >= _RUN_REPEATS repeats of a unit of <= 64 characters (and
# >= _RUN_MIN characters) is stored as {"$rep": [unit, count]}: upstream's "x" * n inputs stay small.
_RUNS_MIN = 1024
_RUN_MIN = 256
_RUN_REPEATS = 16
_RUNS_MIN_ITEMS = 32  # lists this long store runs of equal items as {"$runs": [[item, count], ...]}
_RUN = re.compile(rf"(.{{1,64}}?)\1{{{_RUN_REPEATS - 1},}}", re.DOTALL)

# ------------------------------------------------------------------------------------------------ encoding


def _has_surrogate(s: str) -> bool:
    return _SURROGATE.search(s) is not None


def _runs(s: str) -> Any:
    parts: list[Any] = []
    pos = 0
    for m in _RUN.finditer(s):
        if m.end() - m.start() < _RUN_MIN:
            continue
        unit = m.group(1)
        if m.start() > pos:
            parts.append(encode(s[pos : m.start()]))
        parts.append({"$rep": [encode(unit), (m.end() - m.start()) // len(unit)]})
        pos = m.end()
    if not parts:
        return None
    if pos < len(s):
        parts.append(encode(s[pos:]))
    return parts[0] if len(parts) == 1 else {"$cat": parts}


def _list_runs(items: list[Any]) -> Any:
    """``{"$runs": [[item, count], ...]}`` when consecutive equal items make that at most half as long."""
    runs: list[list[Any]] = []
    last = None
    for item in items:
        key = json.dumps(item, sort_keys=False)
        if runs and key == last:
            runs[-1][1] += 1
        else:
            runs.append([item, 1])
            last = key
    return {"$runs": runs} if 2 * len(runs) <= len(items) else items


def encode(x: Any) -> Any:
    """``x`` as a JSON-serialisable value that ``decode`` turns back into an equal (``same``) value."""
    t = type(x)
    if x is None or t is bool:
        return x
    if t is int:
        return x if -_INT64 <= x < _INT64 else {"$int": hex(x)}
    if t is float:
        return x if math.isfinite(x) else {"$float": repr(x)}
    if t is str:
        if len(x) >= _RUNS_MIN and (runs := _runs(x)) is not None:
            return runs
        if _has_surrogate(x):
            return {"$str": x.encode("utf-8", "surrogatepass").hex()}
        return x
    if t is list:
        items = [encode(v) for v in x]
        return _list_runs(items) if len(items) >= _RUNS_MIN_ITEMS else items
    if t is tuple:
        return {"$tuple": [encode(v) for v in x]}
    if t is dict:
        if all(type(k) is str and not k.startswith("$") and not _has_surrogate(k) for k in x):
            return {k: encode(v) for k, v in x.items()}
        return {"$dict": [[encode(k), encode(v)] for k, v in x.items()]}
    if t is bytes:
        return {"$bytes": x.hex()}
    if isinstance(x, TextIOSpec):
        spec = {"text": encode(x.text), "pos": x.pos}
        if not x.seekable:
            spec["seekable"] = False
        return {"$textio": spec}
    if isinstance(x, PathSpec):
        try:  # readable (and compressible: see "$rep") when the bytes are UTF-8
            return {"$path": {"text": encode(x.data.decode("utf-8")), "type": x.kind}}
        except UnicodeDecodeError:
            return {"$path": {"bytes": x.data.hex(), "type": x.kind}}
    # Anything else (a str subclass, a custom object) is kept as an opaque, never-equal-to-a-value marker.
    return {"$object": [f"{t.__module__}.{t.__qualname__}", repr(x)]}


@dataclass(frozen=True)
class TextIOSpec:
    """A text stream argument: ``io.StringIO(text)`` positioned at ``pos`` (characters).

    ``seekable=False``: a stream whose ``seekable()`` is false and whose ``tell()`` raises ``OSError``.
    """

    text: str
    pos: int = 0
    seekable: bool = True


class NonSeekableStringIO(io.StringIO):
    """A ``StringIO`` that reports it cannot seek, like a pipe (json_repair reads such streams whole)."""

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        raise OSError("not seekable")


@dataclass(frozen=True)
class PathSpec:
    """A file name argument: a temporary file with these bytes, passed as ``Path`` or ``str``."""

    data: bytes
    kind: str = "Path"


@dataclass(frozen=True)
class Opaque:
    """A value ``encode`` could not represent (type name and repr). Never ``same`` as a decoded value."""

    type: str
    repr: str


def decode(x: Any) -> Any:
    """Inverse of ``encode``."""
    if isinstance(x, list):
        return [decode(v) for v in x]
    if not isinstance(x, dict):
        return x
    if len(x) == 1:
        ((tag, v),) = x.items()
        if tag.startswith("$"):
            return _decode_tagged(tag, v)
    return {k: decode(v) for k, v in x.items()}


def _decode_tagged(tag: str, v: Any) -> Any:
    if tag == "$int":
        return int(v, 16)
    if tag == "$float":
        return float(v)
    if tag == "$str":
        return bytes.fromhex(v).decode("utf-8", "surrogatepass")
    if tag == "$rep":
        return decode(v[0]) * v[1]
    if tag == "$cat":
        return "".join(decode(p) for p in v)
    if tag == "$runs":
        return [decode(item) for item, n in v for _ in range(n)]
    if tag == "$tuple":
        return tuple(decode(e) for e in v)
    if tag == "$dict":
        return {decode(k): decode(e) for k, e in v}
    if tag == "$bytes":
        return bytes.fromhex(v)
    if tag == "$textio":
        return TextIOSpec(decode(v["text"]), int(v["pos"]), bool(v.get("seekable", True)))
    if tag == "$path":
        data = decode(v["text"]).encode("utf-8") if "text" in v else bytes.fromhex(v["bytes"])
        return PathSpec(data, v["type"])
    if tag == "$object":
        return Opaque(v[0], v[1])
    raise ValueError(f"unknown tag {tag!r}")


# ---------------------------------------------------------------------------------------------- comparison


def same(a: Any, b: Any) -> bool:
    """Deep equality that also requires equal types, NaN == NaN, -0.0 != 0.0 and equal dict key order."""
    ta = type(a)
    if ta is not type(b):
        return False
    if ta is float:
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    if ta is list or ta is tuple:
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if ta is dict:
        return len(a) == len(b) and all(
            same(ka, kb) and same(va, vb) for (ka, va), (kb, vb) in zip(a.items(), b.items(), strict=True)
        )
    return bool(a == b)


def first_difference(a: Any, b: Any, path: str = "$") -> str | None:
    """Where ``a`` and ``b`` first differ (a JSONPath-like string), or None when they are ``same``."""
    if same(a, b):
        return None
    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, (list, tuple)):
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            if (d := first_difference(x, y, f"{path}[{i}]")) is not None:
                return d
        return f"{path}: length {len(a)} vs {len(b)}"
    if isinstance(a, dict):
        for i, ((ka, va), (kb, vb)) in enumerate(zip(a.items(), b.items(), strict=False)):
            if not same(ka, kb):
                return f"{path}: key #{i} {ka!r} vs {kb!r}"
            if (d := first_difference(va, vb, f"{path}[{ka!r}]")) is not None:
                return d
        return f"{path}: {len(a)} keys vs {len(b)}"
    if isinstance(a, str):
        pos = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
        lo = max(0, pos - 20)
        return f"{path}: strings differ at {pos}: {a[lo : pos + 20]!r} vs {b[lo : pos + 20]!r}"
    return f"{path}: {a!r} vs {b!r}"


# ------------------------------------------------------------------------------------------------ outcomes


@dataclass(frozen=True)
class Outcome:
    """``kind`` "ok" (``value`` = the result) or "raise" (``value`` = (exception type name, message))."""

    kind: str
    value: Any
    warnings: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {self.kind: encode(self.value) if self.kind == "ok" else list(self.value)}
        if self.warnings:
            out["warnings"] = list(self.warnings)
        return out

    @classmethod
    def from_json(cls, j: dict[str, Any]) -> Outcome:
        w = tuple(j.get("warnings", ()))
        if "ok" in j:
            return cls("ok", decode(j["ok"]), w)
        name, message = j["raise"]
        return cls("raise", (name, message), w)

    def short(self, limit: int = 300) -> str:
        r = repr(self.value) if self.kind == "ok" else f"{self.value[0]}: {self.value[1]}"
        r = r if len(r) <= limit else r[:limit] + "..."
        return f"{self.kind} {r}" + (f" warnings={list(self.warnings)}" if self.warnings else "")


def is_fallback_warning(category: type[Warning]) -> bool:
    """True for ``pyroid.FallbackWarning`` and its subclasses (matched by name: pyroid need not be installed)."""
    return any(f"{c.__module__}.{c.__qualname__}" == FALLBACK_WARNING for c in category.__mro__)


def capture(fn: Callable[..., Any], *args: Any, scrub: Iterable[str] = (), **kwargs: Any) -> Outcome:
    """Call ``fn`` and return its outcome; every exception is caught (type and message are compared).

    ``scrub``: substrings (temporary paths) replaced by ``<tmp>`` in exception messages.
    Not thread-safe before Python 3.14 (it uses ``warnings.catch_warnings``).
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = Outcome("ok", fn(*args, **kwargs))
        except Exception as e:  # noqa: BLE001  exception types and messages are part of parity
            message = str(e)
            for s in scrub:
                message = message.replace(s, "<tmp>")
            out = Outcome("raise", (type(e).__name__, message))
    cats = tuple(sorted({w.category.__name__ for w in caught if not is_fallback_warning(w.category)}))
    return Outcome(out.kind, out.value, cats)


#: json_repair 0.64.9's chunked file reader (StringFileWrapper) raises this on some inputs when
#: ``chunk_length`` is small: a log-context slice asks for a chunk past the end of the file.
#: pyroid reads the file at once (a documented difference), so for such calls the reference is
#: json_repair with the default ``chunk_length`` (1M characters: the whole file).
CHUNK_BUG = ("IndexError", "Chunk index out of range")


def is_chunk_bug(o: Outcome) -> bool:
    """Whether ``o`` is json_repair's chunked-file-reader IndexError."""
    return o.kind == "raise" and tuple(o.value) == CHUNK_BUG


def same_outcome(a: Outcome, b: Outcome) -> bool:
    if a.kind != b.kind or a.warnings != b.warnings:
        return False
    return same(a.value, b.value) if a.kind == "ok" else tuple(a.value) == tuple(b.value)


def describe_difference(got: Outcome, want: Outcome) -> str:
    if got.kind != want.kind:
        return f"{got.kind} vs {want.kind}"
    if got.warnings != want.warnings:
        return f"warnings {list(got.warnings)} vs {list(want.warnings)}"
    if got.kind == "raise":
        return f"exception {got.value} vs {want.value}"
    return first_difference(got.value, want.value) or "same"


# ------------------------------------------------------------------------------------------- recorded calls


@dataclass
class Call:
    """One recorded public call: ``fn(*args, **kwargs)`` and its outcome under json_repair 0.64.9."""

    fn: str
    args: list[Any]
    kwargs: dict[str, Any]
    outcome: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    test: str = ""
    index: int = 0

    @property
    def want(self) -> Outcome:
        return Outcome.from_json(self.outcome)

    def key(self) -> str:
        return json.dumps([self.fn, self.args, self.kwargs], sort_keys=False, ensure_ascii=True)

    def to_json(self) -> dict[str, Any]:
        return {
            "fn": self.fn,
            "args": self.args,
            "kwargs": self.kwargs,
            "outcome": self.outcome,
            "tags": self.tags,
            "test": self.test,
        }


def read_calls(path: Path) -> tuple[dict[str, Any], list[Call]]:
    """The header line and the calls of an ``upstream-calls.jsonl`` file."""
    with path.open(encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    header = json.loads(lines[0])
    if header.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file")
    calls = []
    for i, line in enumerate(lines[1:]):
        j = json.loads(line)
        calls.append(
            Call(j["fn"], j["args"], j["kwargs"], j["outcome"], j.get("tags", []), j.get("test", ""), i)
        )
    return header, calls


def write_calls(path: Path, header: dict[str, Any], calls: Iterable[Call]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"format": FORMAT, **header}, ensure_ascii=True) + "\n")
        for c in calls:
            f.write(json.dumps(c.to_json(), ensure_ascii=True) + "\n")


def _materialize(x: Any, tmp: Path, counter: Iterator[int]) -> Any:
    if isinstance(x, TextIOSpec):
        fd = io.StringIO(x.text) if x.seekable else NonSeekableStringIO(x.text)
        fd.seek(x.pos)
        return fd
    if isinstance(x, PathSpec):
        p = tmp / f"input-{next(counter)}.json"
        p.write_bytes(x.data)
        return p if x.kind == "Path" else str(p)
    return x


def replay(target: Any, call: Call) -> Outcome:
    """Run ``call`` against ``target`` (a module or adapter with the json_repair API) and return its outcome."""
    args, kwargs = decode(call.args), decode(call.kwargs)
    with tempfile.TemporaryDirectory(prefix="pyroid-jr-") as d:
        tmp, counter = Path(d), iter(range(1 << 30))
        args = [_materialize(a, tmp, counter) for a in args]
        kwargs = {k: _materialize(v, tmp, counter) for k, v in kwargs.items()}
        return capture(getattr(target, call.fn), *args, scrub=(str(tmp),), **kwargs)
