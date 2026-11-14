"""json_repair parity: the ``--impl`` candidate vs json_repair 0.64.9 (live) and its recorded outcomes.

    pytest tests/test_json_repair.py                                  # self-check (candidate = json_repair)
    pytest tests/test_json_repair.py --impl pyroid.json_repair        # pyroid vs json_repair
    HYPOTHESIS_PROFILE=ci pytest tests/test_json_repair.py --impl pyroid.json_repair

Outcomes are compared exactly (``pyroid_parity.json_repair_compare``): ``str`` results byte for byte, objects and
``(object, log)`` tuples type-exact (int / float / bool distinct, NaN == NaN, -0.0 != 0.0, dict key order, list
vs tuple), exceptions by type name and message, warnings by category (``pyroid.FallbackWarning`` left out).
Option sets: ``pyroid_parity.specs.JSON_REPAIR_OPTION_SETS`` (default, return_objects, logging, skip_json_loads,
ensure_ascii=False, stream_stable, strict, indent=2).

Checks (suite ``json_repair`` in parity-report.json):
    upstream               every outermost public call json_repair 0.64.9's own test suite makes, replayed
                           (``upstream-calls.jsonl``, recorded by ``python -m pyroid_parity.json_repair_record``)
    upstream-schema        the recorded calls with ``schema=`` (pyroid delegates those to json_repair)
    upstream-<tag>         skipped recorded calls, with the reason (outcome depends on the stack depth / hash seed)
    corpus-<option>        20k generated broken LLM outputs (``json_repair_corpus``): candidate == live json_repair
    corpus-pinned-<option> live json_repair reproduces the outcome digests pinned on CPython 3.12
                           (``corpus-outcomes.json``, one per 250 cases): reference drift across interpreters
    corpus-inputs          the generated corpus is the pinned one (sha256)
    hypothesis-<option>    grammar-generated JSON-ish text (``json_repair_strategies``) under each option set
    hypothesis-loads       ``loads(text, **kw)``
    load                   ``load(StringIO(prefix + text) positioned after prefix, chunk_length=k, **kw)``
    from_file              ``from_file(path, chunk_length=k, **kw)`` on temporary files (corpus slice, Hypothesis)
    api-signature          the four functions' signatures mirror json_repair's (non-reference candidates only)

Without ``--impl`` the candidate is json_repair itself: the live comparisons then check that json_repair is
deterministic, and the recorded / pinned checks that this interpreter reproduces the CPython 3.12 recording.
"""

from __future__ import annotations

import importlib.metadata as md
import importlib.util
import inspect
import io
import json
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import json_repair
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from pyroid_parity import json_repair_compare as C
from pyroid_parity import json_repair_corpus as JC
from pyroid_parity import json_repair_strategies as S
from pyroid_parity.json_repair_record import SKIP_TAGS
from pyroid_parity.plugin import _impl, reference
from pyroid_parity.report import Collector
from pyroid_parity.specs import JSON_REPAIR_OPTION_SETS, JSON_REPAIR_VERSION

SUITE = "json_repair"
DATA = Path(__file__).resolve().parents[1] / "data" / f"json_repair-{JSON_REPAIR_VERSION}"
GOLDEN = DATA / "upstream-calls.jsonl"
HEADER, CALLS = C.read_calls(GOLDEN)
PIN: dict[str, Any] = json.loads(JC.PIN.read_text("utf-8"))
OPTIONS = list(JSON_REPAIR_OPTION_SETS)
FILE_SLICE = 300  # corpus cases written to temporary files for from_file
HAS_JSONSCHEMA = importlib.util.find_spec("jsonschema") is not None


@pytest.fixture(scope="session")
def json_repair_impl(request: pytest.FixtureRequest) -> Any:
    return _impl(request, SUITE)


@pytest.fixture(scope="module", autouse=True)
def _oracle() -> None:
    reference(SUITE, "json_repair", md.version("json-repair"), None)


def check(
    parity: Collector,
    name: str,
    got: C.Outcome,
    want: C.Outcome,
    *,
    case: str | None = None,
    input: Any = None,
) -> None:
    same = parity.record(
        SUITE, name, C.same_outcome(got, want), case=case, input=input, got=got.short(), want=want.short()
    )
    assert same, (
        f"{name} {case or ''}\ninput: {input!r}\n  got: {got.short(2000)}\n want: {want.short(2000)}\n"
        f"where: {C.describe_difference(got, want)}"
    )


def test_live_oracle_is_pinned_json_repair() -> None:
    assert md.version("json-repair") == JSON_REPAIR_VERSION
    assert HEADER["json_repair"] == JSON_REPAIR_VERSION and PIN["json_repair"] == JSON_REPAIR_VERSION


# ---------------------------------------------------------------------------------- upstream test suite


def _upstream_id(c: C.Call) -> str:
    return f"{c.index:04d}-{c.fn}"


@pytest.mark.parametrize("call", CALLS, ids=[_upstream_id(c) for c in CALLS])
def test_upstream(json_repair_impl: Any, parity: Collector, call: C.Call) -> None:
    skip = [t for t in call.tags if t in SKIP_TAGS]
    if skip:
        n = sum(1 for c in CALLS if skip[0] in c.tags)
        parity.skip(SUITE, f"upstream-{skip[0]}", f"{n} recorded calls: {SKIP_TAGS[skip[0]]}")
        pytest.skip(f"{skip[0]}: {SKIP_TAGS[skip[0]]}")
    name = "upstream-schema" if "schema" in call.tags else "upstream"
    if name == "upstream-schema" and not HAS_JSONSCHEMA:
        why = (
            "jsonschema is not installed here (json_repair's schema= needs it; it is in the parity dev group)"
        )
        parity.skip(SUITE, name, why)
        pytest.skip(why)
    check(parity, name, C.replay(json_repair_impl.obj, call), call.want, case=call.test, input=call.args)


# --------------------------------------------------------------------------------------- generated corpus


@cache
def _corpus(limit: int) -> list[JC.Case]:
    return JC.cases(min(limit, PIN["size"]) if limit else PIN["size"])


def test_corpus_inputs_are_pinned(parity: Collector) -> None:
    got = JC.inputs_sha256(_corpus(0))
    same = parity.record(
        SUITE, "corpus-inputs", got == PIN["inputs_sha256"], got=got, want=PIN["inputs_sha256"]
    )
    assert same, "json_repair_corpus generates a different corpus than the one pinned in corpus-outcomes.json"
    assert PIN["option_sets"] == JSON_REPAIR_OPTION_SETS


@pytest.mark.parametrize("option", OPTIONS)
def test_corpus(json_repair_impl: Any, parity: Collector, parity_limit: int, option: str) -> None:
    kw = JSON_REPAIR_OPTION_SETS[option]
    cases, pinned = _corpus(parity_limit), PIN["chunks"][option]
    live = not json_repair_impl.is_reference
    drift, diffs = [], []
    for start in range(0, len(cases), JC.CHUNK):
        chunk = cases[start : start + JC.CHUNK]
        want = [C.capture(json_repair.repair_json, c.text, **kw) for c in chunk]
        if len(chunk) == JC.CHUNK:
            digest = JC.chunk_digest([w.to_json() for w in want])
            ok = parity.record(
                SUITE,
                f"corpus-pinned-{option}",
                digest == pinned[start // JC.CHUNK],
                case=f"cases[{start}:{start + JC.CHUNK}]",
                got=digest,
                want=pinned[start // JC.CHUNK],
            )
            if not ok:
                drift.append(start)
        if not live:
            continue
        for c, w in zip(chunk, want, strict=True):
            g = C.capture(json_repair_impl.obj.repair_json, c.text, **kw)
            if not parity.record(
                SUITE,
                f"corpus-{option}",
                C.same_outcome(g, w),
                case=f"{c.id} {c.recipe}",
                input=c.text,
                got=g.short(),
                want=w.short(),
            ):
                diffs.append(f"{c.id} ({c.recipe}): {C.describe_difference(g, w)}")
    if not live:
        parity.skip(SUITE, f"corpus-{option}", "candidate is json_repair itself (corpus-pinned-* checks it)")
    assert not drift, f"json_repair {option}: chunks starting at {drift} differ from the CPython 3.12 pin"
    assert not diffs, f"{len(diffs)}/{len(cases)} cases differ, first: " + "\n".join(diffs[:10])


# --------------------------------------------------------------------------------- Hypothesis differential

EXAMPLES = (
    "",
    " ",
    "{",
    '{"a": 1',
    '{"key": "val\\',
    '{"key": "val\\n123,`key2:value2',
    '{"a": "\\ud83d\\ude00"}',
    '{"a": "\\ud800"}',
    '{"a": "\udc00"}',  # a raw lone surrogate
    "\ud83d\ude00",  # a surrogate pair as two code points
    "[1_000, 1,234, 1/2, -.5, 1e, 1e400, -0.0, 12345678901234567890]",
    "```json\n{'a': True, 'b': None}\n```",
    "Here is the JSON:\n(1, 2)",
    '{"a": 1} {"b": 2}',
    '{"a": 1, "a": 2}',
    '{"key": , "key2": "value2"}',
    (
        "{\N{LEFT DOUBLE QUOTATION MARK}a\N{RIGHT DOUBLE QUOTATION MARK}: "
        "\N{DOUBLE LOW-9 QUOTATION MARK}b\N{LEFT DOUBLE QUOTATION MARK}}"
    ),
    "[\N{SUPERSCRIPT TWO}, \N{ARABIC-INDIC DIGIT THREE}, \N{FULLWIDTH DIGIT ONE}]",
    "{a: b\x1c, c\x85: d\N{IDEOGRAPHIC SPACE}}",
    "// c\n# d\n/* e */ [1, /* f */ 2] // g",
    '"just a string"',
    "NaN",
    '{"a": "' + "x" * 300 + '", "b": [' + "1," * 200 + "]}",
)


def _hypothesis(fn: Callable[..., Any]) -> Callable[..., Any]:
    for text in EXAMPLES:
        fn = example(text=text)(fn)
    return given(text=S.documents())(fn)


@pytest.mark.parametrize("option", OPTIONS)
@_hypothesis
def test_hypothesis(json_repair_impl: Any, parity: Collector, option: str, text: str) -> None:
    kw = JSON_REPAIR_OPTION_SETS[option]
    check(
        parity,
        f"hypothesis-{option}",
        C.capture(json_repair_impl.obj.repair_json, text, **kw),
        C.capture(json_repair.repair_json, text, **kw),
        input=text,
    )


@given(text=S.documents(), kw=S.options("skip_json_loads", "logging", "stream_stable", "strict"))
@example(text='"a": 1}', kw={"logging": True})
@example(text='"a": 1}', kw={"strict": True})
def test_hypothesis_loads(json_repair_impl: Any, parity: Collector, text: str, kw: dict[str, Any]) -> None:
    check(
        parity,
        "hypothesis-loads",
        C.capture(json_repair_impl.obj.loads, text, **kw),
        C.capture(json_repair.loads, text, **kw),
        input=(text, kw),
    )


CHUNK_LENGTHS = st.sampled_from([0, 1, 2, 3, 7, 64, 1 << 20])


@given(
    text=S.documents(),
    prefix=st.sampled_from(["", " ", '{"stale": true}\n', "x"]),
    chunk_length=CHUNK_LENGTHS,
    kw=S.options("skip_json_loads", "logging", "strict"),
)
@example(text='\n"a": {"b": "你好"}}\n', prefix='{"stale": true}\n', chunk_length=2, kw={"logging": True})
@example(text='{"key": }', prefix='{"stale": true}\n', chunk_length=2, kw={})
@example(text='"a": 1} {"b": 2}', prefix="", chunk_length=2, kw={})
def test_load(
    json_repair_impl: Any, parity: Collector, text: str, prefix: str, chunk_length: int, kw: dict[str, Any]
) -> None:
    def run(load: Callable[..., Any], chunk: int = chunk_length) -> Any:
        fd = io.StringIO(prefix + text)
        fd.seek(len(prefix))
        return load(fd, chunk_length=chunk, **kw)

    want = C.capture(run, json_repair.load)
    if C.is_chunk_bug(want):
        want = C.capture(run, json_repair.load, 0)
    check(
        parity,
        "load",
        C.capture(run, json_repair_impl.obj.load),
        want,
        input=(prefix, text, chunk_length, kw),
    )


def _from_file(fn: Callable[..., Any], path: Path, chunk_length: int, kw: dict[str, Any]) -> C.Outcome:
    out = C.capture(fn, path, chunk_length=chunk_length, scrub=(str(path),), **kw)
    if C.is_chunk_bug(out):
        # json_repair's chunked reader bug (see json_repair_compare.CHUNK_BUG).
        out = C.capture(fn, path, chunk_length=0, scrub=(str(path),), **kw)
    return out


@pytest.fixture(scope="module")
def file_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("json_repair_files")


@pytest.mark.parametrize(("chunk_length", "logging"), [(0, False), (2, True), (7, False)])
def test_from_file_corpus(
    json_repair_impl: Any, parity: Collector, file_dir: Path, chunk_length: int, logging: bool
) -> None:
    diffs = []
    kw = {"logging": True} if logging else {}
    for c in _corpus(FILE_SLICE):
        path = file_dir / f"{c.id}.json"
        path.write_text(c.text, "utf-8")
        got = _from_file(json_repair_impl.obj.from_file, path, chunk_length, kw)
        want = _from_file(json_repair.from_file, path, chunk_length, kw)
        if not parity.record(
            SUITE,
            "from_file",
            C.same_outcome(got, want),
            case=c.id,
            input=c.text,
            got=got.short(),
            want=want.short(),
        ):
            diffs.append(c.id)
    assert not diffs, f"from_file differs on {diffs[:10]}"


@given(text=S.file_texts(), chunk_length=CHUNK_LENGTHS, kw=S.options("skip_json_loads", "logging", "strict"))
def test_from_file_hypothesis(
    json_repair_impl: Any, parity: Collector, file_dir: Path, text: str, chunk_length: int, kw: dict[str, Any]
) -> None:
    path = file_dir / "hypothesis.json"
    path.write_text(text, "utf-8")
    check(
        parity,
        "from_file",
        _from_file(json_repair_impl.obj.from_file, path, chunk_length, kw),
        _from_file(json_repair.from_file, path, chunk_length, kw),
        input=(text, chunk_length, kw),
    )


@pytest.mark.parametrize("data", [b'{"a": "\xed\xa0\x80"}', b"\xff{}", b'{"a": "caf\xc3\xa9"', b""])
def test_from_file_bytes(json_repair_impl: Any, parity: Collector, file_dir: Path, data: bytes) -> None:
    path = file_dir / "bytes.json"
    path.write_bytes(data)
    check(
        parity,
        "from_file",
        _from_file(json_repair_impl.obj.from_file, path, 0, {}),
        _from_file(json_repair.from_file, path, 0, {}),
        input=data,
    )


# -------------------------------------------------------------------------------------------- API mirror


def _shape(fn: Callable[..., Any]) -> list[tuple[str, str, Any]]:
    return [(p.name, p.kind.name, p.default) for p in inspect.signature(fn).parameters.values()]


@pytest.mark.parametrize("name", ["repair_json", "loads", "load", "from_file"])
def test_signature_mirrors_json_repair(json_repair_impl: Any, parity: Collector, name: str) -> None:
    if json_repair_impl.is_reference:
        parity.skip(SUITE, "api-signature", "candidate is json_repair itself")
        pytest.skip("candidate is json_repair itself")
    got = C.capture(lambda: _shape(getattr(json_repair_impl.obj, name)))
    check(parity, "api-signature", got, C.Outcome("ok", _shape(getattr(json_repair, name))), case=name)
