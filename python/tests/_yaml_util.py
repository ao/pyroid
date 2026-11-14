"""Helpers for the pyroid.yaml tests: outcome encodings and Hypothesis strategies."""

from __future__ import annotations

import datetime
import math
import zoneinfo
from collections.abc import Callable
from typing import Any

from hypothesis import strategies as st


def encode(value: Any) -> Any:
    """A comparable rendering of a loaded value: types, key order, and which objects are shared."""
    seen: dict[int, int] = {}

    def enc(x: Any) -> Any:
        if isinstance(x, (list, dict, set, datetime.date, bytes)):
            if id(x) in seen:
                return ("alias", seen[id(x)])
            seen[id(x)] = len(seen)
        t = type(x).__name__
        if isinstance(x, list):
            return (t, [enc(v) for v in x])
        if isinstance(x, tuple):
            return (t, tuple(enc(v) for v in x))
        if isinstance(x, dict):
            return (t, [(enc(k), enc(v)) for k, v in x.items()])
        if isinstance(x, set):
            return (t, sorted(repr(v) for v in x))
        if isinstance(x, float):
            return (t, "nan" if math.isnan(x) else repr(x))
        if isinstance(x, datetime.datetime):
            return (t, repr(x), repr(x.tzinfo))
        return (t, repr(x))

    return enc(value)


def outcome(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """``("ok", value)`` or ``("err", exception class name, str(exception))``."""
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as exc:
        return ("err", type(exc).__name__, str(exc))


def load_outcome(fn: Callable[..., Any], *args: Any) -> Any:
    """Like :func:`outcome`, with the documents of a generator listed and encoded."""
    try:
        result = fn(*args)
        if hasattr(result, "__next__"):
            return ("ok", [encode(d) for d in result])
        return ("ok", encode(result))
    except Exception as exc:
        return ("err", type(exc).__name__, str(exc))


class Chunked:
    """A file object whose ``read`` returns at most `size` units, without a ``name``."""

    def __init__(self, data: Any, size: int) -> None:
        self.data = data
        self.size = size
        self.pos = 0

    def read(self, n: int = -1) -> Any:
        k = self.size if n < 0 else min(n, self.size)
        out = self.data[self.pos : self.pos + k]
        self.pos += len(out)
        return out


# Strings that exercise the resolver, the emitter's style choice and its quoting.
TRICKY = [
    "",
    " a",
    "a ",
    "- x",
    "x: y",
    "a #b",
    "#c",
    "yes",
    "No",
    "on",
    "OFF",
    "~",
    "null",
    "NULL",
    "1",
    "-1",
    "+1",
    "01",
    "08",
    "0x1F",
    "0b101",
    "0o17",
    "1_000",
    "1:20",
    "190:20:30.15",
    "1e5",
    "1.0e+5",
    "1.",
    ".5",
    ".inf",
    "-.Inf",
    ".NaN",
    "2001-12-14",
    "2001-12-14 21:59:43.10 -5",
    "a\nb",
    "a\n",
    "\n",
    "\n\n",
    "\t",
    "a\tb",
    "'",
    '"',
    "\\",
    "\u00e9",
    "\u2028",
    "\x85",
    "\ufeff",
    "\x00",
    "\x07",
    "\x7f",
    "\ud7ff",
    "\ue000",
    "\U0001f600",
    "a" * 120,
    "word " * 30,
    "=",
    "<<",
    "!x",
    "&a",
    "*a",
    "%x",
    "@x",
    "`x",
    "[",
    "{",
    "}",
    ",",
    "?",
    ":",
    "- ",
    "? x",
    "|",
    ">",
    "---",
    "...",
    "--- x",
    "a: b: c",
]

text = st.text(st.characters(exclude_categories=["Cs"]), max_size=40) | st.sampled_from(TRICKY)


def _timezones() -> st.SearchStrategy[datetime.tzinfo]:
    """Fixed offsets everywhere, plus IANA zones where the platform has a tz database (not Windows or Alpine
    without tzdata)."""
    day = datetime.timedelta(hours=23, minutes=59, seconds=59)
    fixed = st.builds(datetime.timezone, st.timedeltas(min_value=-day, max_value=day))
    try:
        zoneinfo.ZoneInfo("UTC")
    except zoneinfo.ZoneInfoNotFoundError:
        return fixed
    return fixed | st.timezones()


scalars = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.integers(-(2**70), 2**70)
    | st.floats()
    | text
    | st.binary(max_size=90)
    | st.dates()
    | st.datetimes()
    | st.datetimes(timezones=_timezones())
)
keys = (
    st.none()
    | st.booleans()
    | st.integers(-1000, 1000)
    | st.floats(allow_nan=False)
    | text
    | st.dates()
)
data = st.recursive(
    scalars,
    lambda c: (
        st.lists(c, max_size=5)
        | st.tuples(c, c)
        | st.dictionaries(keys, c, max_size=5)
        | st.frozensets(keys, max_size=3).map(set)
    ),
    max_leaves=25,
)

dump_options = st.fixed_dictionaries(
    {},
    optional={
        "default_flow_style": st.sampled_from([None, True, False]),
        "default_style": st.sampled_from([None, "", '"', "'", "|", ">"]),
        "canonical": st.booleans(),
        "indent": st.integers(0, 12),
        "width": st.sampled_from([0, 5, 20, 80, 200, float("inf")]),
        "allow_unicode": st.booleans(),
        "line_break": st.sampled_from([None, "\n", "\r", "\r\n", "x"]),
        "explicit_start": st.booleans(),
        "explicit_end": st.booleans(),
        "sort_keys": st.booleans(),
        "version": st.sampled_from([None, (1, 1), (1, 2), (2, 0)]),
        "tags": st.sampled_from([None, {}, {"!e!": "tag:example.com,2000:"}, {"!": "x y"}]),
        "encoding": st.sampled_from([None, "utf-8", "utf-16", "utf-16-le", "ascii"]),
    },
)

# Characters that matter to the scanner, for mutating valid YAML.
YAML_CHARS = [
    *" \t\n\r-?:,[]{}#&*!|>'\"%@`.=<~\\x0",
    "\x85",
    "\u2028",
    "\ufeff",
    "\u00e9",
    "\x07",
]


@st.composite
def yaml_texts(draw: Any, dump: Callable[..., str]) -> str:
    """YAML text: random data dumped with random options, then a few random edits."""
    doc = draw(data)
    opts = draw(dump_options)
    opts.pop("encoding", None)
    opts["version"] = None
    try:
        out = dump(doc, **opts)
    except Exception:
        out = ""
    s = list(out)
    for _ in range(draw(st.integers(0, 4))):
        op = draw(st.sampled_from(["ins", "del", "rep"]))
        if not s and op != "ins":
            op = "ins"
        i = draw(st.integers(0, len(s)))
        if op == "ins":
            s.insert(i, draw(st.sampled_from(YAML_CHARS)))
        elif i < len(s):
            if op == "del":
                del s[i]
            else:
                s[i] = draw(st.sampled_from(YAML_CHARS))
    return "".join(s)
