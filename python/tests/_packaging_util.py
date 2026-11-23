"""Helpers for the pyroid.packaging tests: Hypothesis strategies and outcome encodings."""

from __future__ import annotations

import pickle
from typing import Any

from hypothesis import strategies as st

# --------------------------------------------------------------------------------------------
# Outcomes: what a call returned, or which exception it raised (class name and message)
# --------------------------------------------------------------------------------------------


def outcome(f: Any, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
    """``("ok", value)`` or ``("err", class name, str(exc))`` (``KeyError`` messages included)."""
    try:
        return ("ok", f(*args, **kwargs))
    except Exception as e:
        return ("err", type(e).__name__, str(e))


def listed(it: Any) -> list[Any]:
    """`list(it)` (for outcome(): the iterator's errors surface while listing)."""
    return list(it)


def strings(it: Any) -> list[str]:
    return [str(x) for x in it]


def version_view(v: Any) -> tuple[Any, ...]:
    """Everything a Version shows, as plain values."""
    return (
        str(v),
        repr(v),
        v.epoch,
        v.release,
        v.pre,
        v.post,
        v.dev,
        v.local,
        v.public,
        v.base_version,
        v.is_prerelease,
        v.is_postrelease,
        v.is_devrelease,
        v.major,
        v.minor,
        v.micro,
        v._key,
        hash(v),
        v.__getstate__(),
    )


def spec_view(s: Any) -> tuple[Any, ...]:
    return (str(s), repr(s), s.operator, s.version, s.prereleases, s.__getstate__())


def specset_view(s: Any) -> tuple[Any, ...]:
    return (str(s), repr(s), len(s), [str(x) for x in s], s.prereleases)


def marker_view(m: Any) -> tuple[Any, ...]:
    return (outcome(str, m), outcome(repr, m))


def requirement_view(r: Any) -> tuple[Any, ...]:
    return (
        str(r),
        repr(r),
        r.name,
        r.url,
        sorted(r.extras),
        str(r.specifier),
        repr(r.specifier),
        None if r.marker is None else str(r.marker),
    )


def roundtrip(x: Any) -> Any:
    # Only objects the test itself just made are pickled and loaded back (trusted data).
    return pickle.loads(pickle.dumps(x))


# --------------------------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------------------------

# Characters that exercise the patterns' edges: Python whitespace that is not ASCII, the
# letters re.IGNORECASE folds onto ASCII (dotless i, dotted I, long s, Kelvin sign), other
# digits, a lone surrogate.
ODD = [
    " ", "\u3000", "\u00a0", "\x1c", "\x85", "\u2028", "\t", "\n", "\r", "\x0b", "\x0c",
    "\u0131", "\u0130", "\u017f", "\u212a", "\u0661", "\uff11", "\u00b2", "\ud800", "\u00e9",
    "\u00df", "\u01c5", "\x00", "\u1680",
]  # fmt: skip
SEPARATORS = ["", ".", "-", "_"]
PRE = [
    "a",
    "A",
    "alpha",
    "ALPHA",
    "b",
    "beta",
    "Beta",
    "c",
    "C",
    "rc",
    "RC",
    "pre",
    "preview",
    "Preview",
]
POST = ["post", "POST", "rev", "r", "R"]
NUMBERS = [
    "0",
    "1",
    "2",
    "9",
    "10",
    "00",
    "01",
    "007",
    "12345678901234567890",
    "99999999999999999999999",
]


@st.composite
def version_strings(draw: st.DrawFn) -> str:
    """A PEP 440 version spelled any way the pattern allows, sometimes broken."""
    parts = []
    parts.append(draw(st.sampled_from(["", "", "", " ", "v", "V", "\t", "\u3000"])))
    if draw(st.booleans()) and draw(st.booleans()):
        parts.append(draw(st.sampled_from(NUMBERS[:6])) + "!")
    release = [draw(st.sampled_from(NUMBERS)) for _ in range(draw(st.integers(1, 5)))]
    parts.append(".".join(release))
    if draw(st.booleans()):
        parts.append(draw(st.sampled_from(SEPARATORS)) + draw(st.sampled_from(PRE)))
        parts.append(draw(st.sampled_from([*SEPARATORS, "", ""])))
        parts.append(draw(st.sampled_from(["", "", *NUMBERS[:6]])))
    if draw(st.booleans()):
        if draw(st.booleans()):
            parts.append("-" + draw(st.sampled_from(NUMBERS[:6])))
        else:
            parts.append(draw(st.sampled_from(SEPARATORS)) + draw(st.sampled_from(POST)))
            parts.append(draw(st.sampled_from(SEPARATORS)))
            parts.append(draw(st.sampled_from(["", "", *NUMBERS[:6]])))
    if draw(st.booleans()):
        parts.append(
            draw(st.sampled_from(SEPARATORS)) + draw(st.sampled_from(["dev", "DEV", "Dev"]))
        )
        parts.append(draw(st.sampled_from(SEPARATORS)))
        parts.append(draw(st.sampled_from(["", "", *NUMBERS[:6]])))
    if draw(st.booleans()) and draw(st.booleans()):
        segs = [
            draw(st.sampled_from(["abc", "ABC", "1", "01", "ubuntu", "x9", "0"]))
            for _ in range(draw(st.integers(1, 3)))
        ]
        joiners = [draw(st.sampled_from([".", "-", "_"])) for _ in segs[1:]]
        local = segs[0] + "".join(j + s for j, s in zip(joiners, segs[1:], strict=True))
        parts.append("+" + local)
    parts.append(draw(st.sampled_from(["", "", "", " ", "\n", "\u00a0"])))
    text = "".join(parts)
    return draw(mutated(text))


@st.composite
def mutated(draw: st.DrawFn, text: str) -> str:
    """`text`, or `text` with a few characters inserted, deleted or replaced."""
    if draw(st.integers(0, 3)) != 0:
        return text
    alphabet = st.sampled_from(list("0123456789.!+-_*~=<>; ()[],@'\"abcdeprtvxABCDEPRTVX") + ODD)
    chars = list(text)
    for _ in range(draw(st.integers(1, 3))):
        op = draw(st.integers(0, 2))
        pos = draw(st.integers(0, len(chars)))
        if op == 0 or not chars:
            chars.insert(pos, draw(alphabet))
        elif op == 1:
            del chars[min(pos, len(chars) - 1)]
        else:
            chars[min(pos, len(chars) - 1)] = draw(alphabet)
    return "".join(chars)


OPS = ["==", "!=", "<=", ">=", "<", ">", "~=", "==="]


@st.composite
def specifier_strings(draw: st.DrawFn) -> str:
    op = draw(st.sampled_from(OPS))
    ws = draw(st.sampled_from(["", "", " ", "  ", "\t"]))
    version = draw(version_strings()).strip()
    if op in ("==", "!=") and draw(st.integers(0, 4)) == 0:
        version = version.split("+")[0] + ".*"
    if op == "===" and draw(st.booleans()):
        version = draw(st.sampled_from(["foo", "FOO", "1.0", "1.0.0", "wat", "1.0a1", ""]))
    return draw(mutated(f"{draw(st.sampled_from(['', ' ']))}{op}{ws}{version}"))


@st.composite
def specifier_set_strings(draw: st.DrawFn) -> str:
    specs = [draw(specifier_strings()) for _ in range(draw(st.integers(0, 4)))]
    sep = draw(st.sampled_from([",", ", ", " ,", ",,"]))
    return sep.join(specs)


VARIABLES = [
    "python_version", "python_full_version", "os_name", "os.name", "sys_platform", "sys.platform",
    "platform_release", "platform_system", "platform_version", "platform.version",
    "platform_machine", "platform.machine", "platform_python_implementation",
    "platform.python_implementation", "python_implementation", "implementation_name",
    "implementation_version", "extra", "extras", "dependency_groups",
]  # fmt: skip
VALUES = [
    "3.8", "3", "2.7", "3.12.1", "3.14.0rc1", "linux", "posix", "nt", "win32", "CPython", "cpython",
    "x86_64", "aarch64", "test", "Test_Me", "a.b", "", "5.10", "foo bar", "a'b", 'a"b', "3.*",
    "é", "1.0+local", "!=", "abc def",
]  # fmt: skip
MARKER_OPS = ["==", "!=", "<", "<=", ">", ">=", "~=", "===", "in", "not in", "not  in"]


def quote(value: str, draw: st.DrawFn) -> str:
    if "'" in value and '"' not in value:
        return f'"{value}"'
    if '"' in value and "'" not in value:
        return f"'{value}'"
    q = draw(st.sampled_from(["'", '"']))
    if q in value:
        return "'" + value.replace("'", "\\'") + "'"
    return q + value + q


@st.composite
def marker_atoms(draw: st.DrawFn) -> str:
    var = draw(st.sampled_from(VARIABLES))
    val = quote(draw(st.sampled_from(VALUES)), draw)
    op = draw(st.sampled_from(MARKER_OPS))

    def ws() -> str:
        return draw(st.sampled_from(["", " ", "  ", "\t"]))

    if draw(st.integers(0, 3)) == 0:
        return f"{val}{ws()}{op}{ws()}{var}"
    return f"{var}{ws()}{op}{ws()}{val}"


@st.composite
def marker_strings(draw: st.DrawFn, depth: int = 0) -> str:
    n = draw(st.integers(1, 3))
    parts = []
    for i in range(n):
        if depth < 2 and draw(st.integers(0, 4)) == 0:
            parts.append("(" + draw(marker_strings(depth=depth + 1)) + ")")
        else:
            parts.append(draw(marker_atoms()))
        if i < n - 1:
            parts.append(draw(st.sampled_from([" and ", " or ", "and", " and", "or "])))
    return draw(mutated("".join(parts)))


NAMES = ["requests", "Foo.Bar_baz", "zope.interface", "a", "A-B", "name", "1name", "-name", "x_"]
EXTRAS = ["security", "socks", "Foo_Bar", "test", "a.b"]


@st.composite
def requirement_strings(draw: st.DrawFn) -> str:
    out = draw(st.sampled_from(["", " "])) + draw(st.sampled_from(NAMES))
    if draw(st.booleans()):
        extras = [draw(st.sampled_from(EXTRAS)) for _ in range(draw(st.integers(0, 3)))]
        out += (
            draw(st.sampled_from(["", " "]))
            + "["
            + draw(st.sampled_from([",", ", ", " , "])).join(extras)
            + "]"
        )
    if draw(st.integers(0, 4)) == 0:
        out += (
            draw(st.sampled_from(["", " "]))
            + "@ "
            + draw(
                st.sampled_from(
                    [
                        "https://example.com/x.whl",
                        "file:///tmp/x",
                        "git+https://g/x@v1#egg=x",
                        "url",
                    ]
                )
            )
        )
        if draw(st.booleans()):
            out += " "
    elif draw(st.booleans()):
        spec = draw(specifier_set_strings())
        out += draw(st.sampled_from(["", " "])) + (
            f"({spec})" if draw(st.integers(0, 4)) == 0 else spec
        )
    if draw(st.booleans()):
        out += draw(st.sampled_from([";", "; ", " ; "])) + draw(marker_strings())
    return draw(mutated(out))


ENVIRONMENTS: list[dict[str, Any] | None] = [
    None,
    {},
    {"python_version": "3.7", "python_full_version": "3.7.17"},
    {
        "python_version": "3.14",
        "python_full_version": "3.14.0rc1",
        "implementation_version": "3.14.0rc1",
    },
    {"python_full_version": "3.12.1+"},
    {
        "sys_platform": "win32",
        "os_name": "nt",
        "platform_system": "Windows",
        "platform_machine": "AMD64",
    },
    {"sys_platform": "darwin", "platform_release": "23.1.0", "platform_machine": "arm64"},
    {"platform_release": "5.10.0-28-amd64"},
    {"extra": "test"},
    {"extra": "Test-Me"},
    {"extra": None},
    {"extras": {"Test_Me", "security"}, "dependency_groups": frozenset({"dev"})},
    {"extras": ["a", "b"]},
    {"python_version": 3},
]
CONTEXTS = ["metadata", "requirement", "lock_file"]
