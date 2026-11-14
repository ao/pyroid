"""Differential tests: pyroid.json_repair must equal json_repair 0.64.9 exactly.

Every input is repaired by both under each option set (default, ``return_objects``, ``logging``,
``skip_json_loads``, ``ensure_ascii=False``, ``stream_stable``, ``strict``, ``indent=2``) and the
outcomes must match: ``str`` results byte for byte, objects and ``(object, log)`` tuples
type-exact (``1``, ``1.0`` and ``True`` differ, NaN equals NaN, ``-0.0`` differs from ``0.0``,
dict key order counts, lists differ from tuples), exceptions by type and message, warnings by
category (``pyroid.FallbackWarning`` left out).

Inputs: JSON-ish text from a token grammar (every quote kind, escapes including ``\\uD800``,
literals, odd numbers, comments, fences, unicode spaces, letters and digits of every script, lone
surrogates drawn explicitly), and a 2,000-case slice of the parity suite's generated corpus of
broken LLM outputs (``parity/pyroid_parity/json_repair_corpus.py``, loaded by path; skipped
without ``parity/``). The full differential suite is ``parity/tests/test_json_repair.py``.

``PYROID_JR_IMPL=json_repair`` runs the tests against json_repair itself.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

IMPL = os.environ.get("PYROID_JR_IMPL", "pyroid.json_repair")
try:
    candidate = importlib.import_module(IMPL)
except ImportError as e:  # not built yet, or not in this wheel
    pytest.skip(f"{IMPL} is not importable: {e}", allow_module_level=True)
oracle: Any = pytest.importorskip("json_repair")
if importlib.metadata.version("json-repair") != "0.64.9":
    pytest.skip("needs json-repair 0.64.9", allow_module_level=True)

from hypothesis import example, given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

OPTION_SETS: dict[str, dict[str, Any]] = {
    "default": {},
    "return_objects": {"return_objects": True},
    "logging": {"logging": True},
    "skip_json_loads": {"skip_json_loads": True},
    "ensure_ascii_false": {"ensure_ascii": False},
    "stream_stable": {"stream_stable": True},
    "strict": {"strict": True},
    "indent2": {"indent": 2},
}
CORPUS_PY = (
    Path(__file__).resolve().parents[2] / "parity" / "pyroid_parity" / "json_repair_corpus.py"
)
CORPUS_CASES = 2000


# ---- outcomes -----------------------------------------------------------------------------------


def same(a: Any, b: Any) -> bool:
    """Deep equality that also requires equal types, NaN == NaN, -0.0 != 0.0, equal key order."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return (math.isnan(a) and math.isnan(b)) or (
            a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
        )
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict):
        return len(a) == len(b) and all(
            same(ka, kb) and same(va, vb)
            for (ka, va), (kb, vb) in zip(a.items(), b.items(), strict=True)
        )
    return bool(a == b)


def _is_fallback(category: type[Warning]) -> bool:
    return any(
        f"{c.__module__}.{c.__qualname__}" == "pyroid.FallbackWarning" for c in category.__mro__
    )


def outcome(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, ...]:
    """("ok", result, warnings) or ("raise", type name, message, warnings)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out: tuple[Any, ...] = ("ok", fn(*args, **kwargs))
        except Exception as e:  # exception types and messages are part of parity
            out = ("raise", type(e).__name__, str(e))
    return (*out, sorted({w.category.__name__ for w in caught if not _is_fallback(w.category)}))


#: json_repair 0.64.9's chunked file reader raises this on some inputs when ``chunk_length`` is
#: small (a log-context slice past the last chunk). pyroid reads files at once, so the reference
#: for such calls is json_repair with the default chunk_length (documented difference).
CHUNK_BUG = ("raise", "IndexError", "Chunk index out of range")


def assert_same(name: str, *args: Any, **kwargs: Any) -> None:
    got = outcome(getattr(candidate, name), *args, **kwargs)
    want = outcome(getattr(oracle, name), *args, **kwargs)
    if want[:3] == CHUNK_BUG and "chunk_length" in kwargs:
        want = outcome(getattr(oracle, name), *args, **{**kwargs, "chunk_length": 0})
    assert same(got, want), f"{name}(*{args!r}, **{kwargs!r})\n  got: {got!r}\n want: {want!r}"


# ---- grammar ------------------------------------------------------------------------------------

# Look-alike and invisible characters are spelled as escapes.
QUOTES = (  # straight, curly, low-9, guillemets, fullwidth
    "\"'`\N{LEFT DOUBLE QUOTATION MARK}\N{RIGHT DOUBLE QUOTATION MARK}"
    "\N{DOUBLE LOW-9 QUOTATION MARK}\N{LEFT SINGLE QUOTATION MARK}"
    "\N{RIGHT SINGLE QUOTATION MARK}\xab\xbb\N{FULLWIDTH QUOTATION MARK}"
)
token = st.one_of(
    st.sampled_from(["{", "}", "[", "]", ":", ",", "(", ")", "{}", "[]", '":', ", "]),
    st.sampled_from([*QUOTES, '""', "'''"]),
    st.sampled_from(
        ['\\"', "\\\\", "\\/", "\\n", "\\t", "\\'", "\\", "\\x41", "\\u", "\\u12", "\\u00e9"]
    ),
    st.integers(0xD800, 0xDFFF).map(lambda n: f"\\u{n:04x}"),  # escaped surrogates
    st.sampled_from(
        ["true", "false", "null", "True", "False", "None", "NaN", "Infinity", "true"[:3], "nu"]
    ),
    st.sampled_from(
        ["0", "-0", "1.5", "-.5", "1.", "1e", "1E+5", "1_000", "1,234", "1/2", "0x1F", "007"]
    ),
    # big, overflowing, negative zero; superscript two, Arabic-Indic three, fullwidth one
    st.sampled_from(
        [
            "12345678901234567890",
            "1e400",
            "-0.0",
            "2\N{SUPERSCRIPT TWO}",
            "\N{ARABIC-INDIC DIGIT THREE}",
            "\N{FULLWIDTH DIGIT ONE}",
        ]
    ),
    st.integers(-(10**20), 10**20).map(str),
    st.floats().map(repr),
    st.sampled_from(["# c\n", "// c\n", "/* c */", "/* open", "```json\n", "```", "\n```"]),
    st.sampled_from(
        [
            *(" ", "\t", "\n", "\r\n", "\x0b", "\x1c", "\x1f", "\x85", "\xa0"),
            "\N{IDEOGRAPHIC SPACE}",
            "\N{LINE SEPARATOR}",
        ]
    ),
    st.sampled_from(["\N{ZERO WIDTH NO-BREAK SPACE}", "\N{ZERO WIDTH SPACE}"]),
    st.text("abcdefghijklmnopqrstuvwxyz_$", min_size=1, max_size=6),
    st.text(st.characters(categories=["L"]), min_size=1, max_size=4),  # letters of every script
    st.characters(categories=["N"]),  # digits and numerals of every script
    st.characters(categories=["Cs"]),  # lone surrogates (Hypothesis never draws them unasked)
    st.text(max_size=3),
)
grammar_texts = st.lists(token, max_size=30).map("".join)
values = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(-(10**25), 10**25),
        st.floats(),
        st.text(max_size=8),
        st.characters(categories=["Cs"]),
    ),
    lambda kids: (
        st.lists(kids, max_size=4) | st.dictionaries(st.text(max_size=5), kids, max_size=4)
    ),
    max_leaves=12,
)
near_json = st.builds(
    lambda v, cut, ascii_: json.dumps(v, ensure_ascii=ascii_)[: cut or None],
    values,
    st.integers(-40, 40),
    st.booleans(),
)
texts = st.one_of(grammar_texts, near_json)

EXAMPLES = [
    "",
    "{",
    '{"key": "val\\',
    '{"key": "val\\n123,`key2:value2',
    '{"a": "\\ud800"}',
    '{"a": "\udc00"}',  # a raw lone surrogate
    chr(0xD83D) + chr(0xDE00),  # a surrogate pair as two code points
    "[1_000, 1,234, 1/2, -.5, 1e, 1e400, -0.0, 12345678901234567890]",
    "```json\n{'a': True, 'b': None}\n```",
    "Here is the JSON:\n(1, 2)",
    '{"a": 1, "a": 2}',
    '{"key": , "key2": "value2"}',
    (
        "{\N{LEFT DOUBLE QUOTATION MARK}a\N{RIGHT DOUBLE QUOTATION MARK}: "
        "\N{DOUBLE LOW-9 QUOTATION MARK}b\N{LEFT DOUBLE QUOTATION MARK}}"
    ),
    "{a: b\x1c, c\x85: d\N{IDEOGRAPHIC SPACE}}",
    "// c\n# d\n/* e */ [1, /* f */ 2] // g",
]


def _with_examples(fn: Callable[..., Any]) -> Callable[..., Any]:
    for text in EXAMPLES:
        fn = example(text=text)(fn)
    return given(text=texts)(fn)


# ---- tests --------------------------------------------------------------------------------------


@pytest.mark.thread_unsafe  # warnings.catch_warnings
@pytest.mark.parametrize("option", OPTION_SETS)
@_with_examples
def test_repair_json_matches(option: str, text: str) -> None:
    assert_same("repair_json", text, **OPTION_SETS[option])


flags = st.fixed_dictionaries(
    {}, optional={k: st.booleans() for k in ("skip_json_loads", "logging", "strict")}
)


@pytest.mark.thread_unsafe
@given(text=texts, kwargs=flags, stream_stable=st.booleans())
def test_loads_matches(text: str, kwargs: dict[str, Any], stream_stable: bool) -> None:
    assert_same("loads", text, stream_stable=stream_stable, **kwargs)


@pytest.mark.thread_unsafe
@given(
    text=texts, prefix=st.sampled_from(["", '{"stale": 1}\n']), chunk=st.sampled_from([0, 1, 2, 7])
)
@example(text='\n"a": {"b": "\u4f60\u597d"}}\n', prefix='{"stale": 1}\n', chunk=2)
def test_load_matches(text: str, prefix: str, chunk: int) -> None:
    def run(load: Callable[..., Any], chunk_length: int = chunk) -> Any:
        fd = io.StringIO(prefix + text)
        fd.seek(len(prefix))
        return load(fd, chunk_length=chunk_length, logging=True)

    got, want = outcome(run, candidate.load), outcome(run, oracle.load)
    if want[:3] == CHUNK_BUG:
        want = outcome(run, oracle.load, 0)
    assert same(got, want), (
        f"load({prefix + text!r}, at {len(prefix)})\n  got: {got!r}\n want: {want!r}"
    )


# ---- the generated corpus -----------------------------------------------------------------------


def _corpus() -> list[Any]:
    if not CORPUS_PY.is_file():
        pytest.skip("parity/pyroid_parity/json_repair_corpus.py is not in this checkout")
    name = "pyroid_tests_json_repair_corpus"
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, CORPUS_PY)
        assert spec is not None
        assert spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod  # dataclasses look their module up while the module executes
        spec.loader.exec_module(mod)
    cases: list[Any] = mod.cases(CORPUS_CASES)
    return cases


@pytest.mark.thread_unsafe
@pytest.mark.parametrize("option", OPTION_SETS)
def test_corpus_slice_matches(option: str) -> None:
    kwargs = OPTION_SETS[option]
    diffs = []
    for case in _corpus():
        got = outcome(candidate.repair_json, case.text, **kwargs)
        want = outcome(oracle.repair_json, case.text, **kwargs)
        if not same(got, want):
            diffs.append(
                f"{case.id} ({case.recipe}): {case.text[:200]!r}\n got {got!r}\nwant {want!r}"
            )
    assert not diffs, f"{len(diffs)}/{CORPUS_CASES} differ; first:\n" + "\n".join(diffs[:3])


@pytest.mark.thread_unsafe
def test_from_file_matches_on_corpus_slice() -> None:
    with tempfile.TemporaryDirectory(prefix="pyroid-jr-") as d:
        for case in _corpus()[:100]:
            path = Path(d) / f"{case.id}.json"
            path.write_text(case.text, "utf-8")
            for chunk in (0, 2):
                assert_same("from_file", path, chunk_length=chunk, logging=True)
