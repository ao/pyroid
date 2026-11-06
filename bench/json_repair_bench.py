#!/usr/bin/env python3
"""json_repair 0.64.5 vs candidates: gated single-doc, pathological-curve and batch timings.

    python bench/json_repair_bench.py --impl json_repair --impl pyroid.json_repair \
        --out /target/jr-bench.json
    python bench/json_repair_bench.py --list                      # the pathological families
    python bench/json_repair_bench.py --sections patho --families truncated_string,table_rows_continued
    python bench/json_repair_bench.py --impl json_repair --impl json_repair --quick   # gate smoke test

Needs the stdlib and ``json-repair==0.64.5`` only. json_repair is always the reference and the
baseline (the first ``--impl json_repair``, or implicitly). Every other ``--impl MODULE`` is an
optional candidate row: a module that does not import (``pyroid.json_repair`` before its wheel is
installed) is skipped with a warning. Repeating ``--impl json_repair`` adds json_repair itself as a
candidate, which exercises the comparison path.

* Equivalence gate, before anything is timed: each candidate's ``repair_json(s)`` must equal
  json_repair's byte for byte, and its ``loads(s)`` must equal json_repair's with type-exact deep
  equality (dict key order, bool vs int, int vs float). Raised exceptions must match in type and
  message. A mismatch aborts the run (exit 1; ``--out`` still gets the partial results).
  Pathological points are gated inside the worker, after its first call and before its timed
  repeats, against fingerprints of the reference's output. There the reference's ``loads`` value is
  ``json.loads`` of its ``repair_json`` string, which json_repair 0.64.5 builds as
  ``json.dumps(loads(s))``. A size json_repair did not reach (it stopped or timed out at a smaller
  size) is reported as unchecked (``*``).
* Sections:
  - ``docs``: ~1 KB and ~1 MB valid and broken docs (seeded LLM/API shapes), ``repair_json(s)`` and
    ``loads(s)``, in-process: one warm-up call, then best and median per call over k samples.
  - ``patho``: every family at doubling sizes, one subprocess per (impl, family, size). A call
    slower than ``--timeout`` s is killed and reported as ``> timeout``; an impl stops doubling on a
    family after a call slower than ``--stop-after`` s. Reports the log-log slope (time exponent).
  - ``batch``: N x ~1 KB broken docs. json_repair in a Python loop is the baseline (on a
    free-threaded build also over a pool of 8 threads, in contiguous chunks); then each candidate in
    a loop and, if it has one, its ``repair_json_batch(docs, threads=t)``. Docs/s.
* Output: a JSON file (``--out``) with machine info and every timing, and a Markdown summary on
  stdout. Numbers measured in the Docker/OrbStack VM are indicative.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata as md
import json
import math
import os
import platform
import queue
import random
import statistics
import subprocess
import sys
import sysconfig
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REFERENCE = "json_repair"
REFERENCE_VERSION = "0.64.5"
LABEL = "Docker/OrbStack VM, indicative"

# ------------------------------------------------------------------------------------------ inputs

_TEXT = (
    "the model returned a long answer about data pipelines and then the user asked for more detail "
    "on caching retries batching vectors embeddings tokens latency throughput memory budgets queues "
    "shards replicas indexes schemas migrations rollouts canaries alerts dashboards"
)
WORDS = _TEXT.split(" ")
NAMES = ["Ana", "Bo", "Chen", "Dana", "\u00c9mile", "Farah", "G\u00fcl", "Hiro", "Ines", "Jo"]
TAGS = ["search", "billing", "ml", "infra", "docs", "ui", "api", "security"]
EXTRAS = [
    "",
    "",
    ' The reviewer said "looks good" and moved on',
    "\nSecond line of the note",
    " \u2014 see the \u201cnotes\u201d section",
    " caf\u00e9 \u6771\u4eac \u2713",
    " 50% off\tthis week",
]


def _prose(nbytes: int, seed: int) -> str:
    """Plain prose (lowercase words, commas and full stops), about ``nbytes`` long."""
    rng = random.Random(seed)
    out: list[str] = []
    size = 0
    while size < nbytes:
        word = rng.choice(WORDS)
        if rng.random() < 0.08:
            word += rng.choice(",.")
        out.append(word)
        size += len(word) + 1
    return " ".join(out)


def _fill(nbytes: int, unit: Callable[[int], str], sep: str = "") -> str:
    """Join ``unit(0), unit(1), ...`` with ``sep`` until the result is at least ``nbytes`` long."""
    parts: list[str] = []
    size = 0
    while size < nbytes or not parts:
        part = unit(len(parts))
        parts.append(part)
        size += len(part) + len(sep)
    return sep.join(parts)


def _escaped(text: str) -> str:
    """``text`` as the body of a JSON string literal (escaped, without the quotes)."""
    return json.dumps(text, ensure_ascii=False)[1:-1]


def _sentence(rng: random.Random, lo: int = 6, hi: int = 30) -> str:
    words = " ".join(rng.choice(WORDS) for _ in range(rng.randint(lo, hi)))
    return words.capitalize() + rng.choice(EXTRAS) + "."


def _title(rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(2, 6)))


def _record(rng: random.Random, i: int) -> dict[str, Any]:
    return {
        "id": i,
        "uuid": f"{rng.getrandbits(64):016x}",
        "title": _title(rng).title(),
        "score": round(rng.uniform(0, 100), 3),
        "active": rng.random() < 0.7,
        "tags": rng.sample(TAGS, rng.randint(0, 3)),
        "author": {
            "name": rng.choice(NAMES),
            "id": rng.randint(1, 10**6),
            "verified": rng.choice([True, False, None]),
        },
        "price": None if rng.random() < 0.3 else round(rng.uniform(1, 500), 2),
        "snippet": _sentence(rng),
    }


def make_valid_doc(nbytes: int, seed: int = 0) -> str:
    """A valid pretty-printed API/LLM payload (records plus usage), about ``nbytes`` long."""
    rng = random.Random(seed)
    data: list[dict[str, Any]] = []
    size = 260
    while size < nbytes or not data:
        rec = _record(rng, len(data))
        data.append(rec)
        size += len(json.dumps(rec, ensure_ascii=False, indent=2)) + 6
    prompt, completion = rng.randint(50, 4000), rng.randint(50, 4000)
    doc = {
        "object": "list",
        "model": "gpt-4o-2024-08-06",
        "request_id": f"req_{seed:08x}",
        "data": data,
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }
    return json.dumps(doc, ensure_ascii=False, indent=2)


def _broken_record(rng: random.Random, i: int) -> str:
    """One record as LLMs break them: mixed key quoting, single quotes, Python literals, comments."""

    def key(name: str) -> str:
        r = rng.random()
        return name if r < 0.25 else (f"'{name}'" if r < 0.35 else f'"{name}"')

    def text(value: str) -> str:  # ``value`` never contains an apostrophe
        return f"'{value}'" if rng.random() < 0.2 else json.dumps(value, ensure_ascii=False)

    tags = rng.sample(TAGS, rng.randint(0, 3))
    trailing = ", " if tags and rng.random() < 0.3 else ""
    tag_list = ", ".join(json.dumps(t) for t in tags) + trailing
    verified = rng.choice(["true", "false", "null", "True", "None"])
    author = f'{{"name": {json.dumps(rng.choice(NAMES))}, "verified": {verified}}}'
    fields = [
        f"{key('id')}: {i}",
        f"{key('title')}: {text(_title(rng))}",
        f"{key('score')}: {round(rng.uniform(0, 100), 3)}",
        f"{key('active')}: {rng.choice(['true', 'false', 'True', 'False'])}",
        f"{key('tags')}: [{tag_list}]",
        f"{key('author')}: {author}",
        f"{key('snippet')}: {json.dumps(_sentence(rng), ensure_ascii=False)}",
    ]
    if rng.random() < 0.1:
        fields.append(f'"note": "he said "{rng.choice(WORDS)}" and left"')
    body = ",\n      ".join(fields) + ("," if rng.random() < 0.3 else "")
    comment = f"      // reviewer note {i}: {rng.choice(WORDS)}\n" if rng.random() < 0.15 else ""
    return "    {\n" + comment + "      " + body + "\n    }"


def make_broken_doc(nbytes: int, seed: int = 0, truncate: bool | None = None) -> str:
    """A typical broken LLM answer, about ``nbytes`` long.

    Prose and a ```json fence around an object with unquoted and single-quoted keys, single-quoted
    strings, Python literals, trailing commas, ``//`` comments, the odd unescaped inner quote and a
    missing closing brace. ``truncate`` (default: a seeded coin flip) cuts the stream off mid-string
    instead of closing the fence.
    """
    rng = random.Random(seed)
    if truncate is None:
        truncate = rng.random() < 0.5
    head = "Sure! Here is the JSON you asked for:\n\n```json\n{\n  \"object\": 'list',\n  data: [\n"
    records: list[str] = []
    size = len(head) + 120
    while size < nbytes or not records:
        rec = _broken_record(rng, len(records))
        records.append(rec)
        size += len(rec) + 2
    body = head + ",\n".join(records) + ",\n"
    if truncate:
        cut = _sentence(rng, 4, 12)
        return body + f'    {{"id": {len(records)}, "snippet": "' + cut[: len(cut) // 2]
    tail = "  ],\n  usage: {'total_tokens': 1234,},\n```\nLet me know if you need anything else!"
    return body + tail


# ---------------------------------------------------------------------------- pathological families


@dataclass(frozen=True)
class Family:
    name: str
    kind: str  # "realistic" | "adversarial" | "control"
    gen: Callable[[int], str]
    desc: str
    cause: str  # the json_repair 0.64.5 code path that makes it super-linear
    kwargs: dict[str, Any] = field(default_factory=dict)


def _code_unit(i: int) -> str:
    return _escaped(f'def step_{i}(x):\n    """Step {i}."""\n    return x + {i}\n\n')


def _markdown_unit(i: int) -> str:
    return _escaped(
        f"## Section {i}\n\nSee [the guide](https://example.com/guide/{i}) and "
        f"[the API reference](https://example.com/api/{i}) for details.\n\n- item {i}\n"
    )


def gen_truncated_string(n: int) -> str:
    return '{"id": "resp_01", "answer": "' + _prose(n, 11)


def gen_escaped_code_string(n: int) -> str:
    return '{"language": "python", "code": "' + _fill(n, _code_unit) + '",}'


def gen_escaped_code_valid(n: int) -> str:
    return '{"language": "python", "code": "' + _fill(n, _code_unit) + '"}'


def gen_markdown_in_array(n: int) -> str:
    return '[{"role": "assistant", "content": "' + _fill(n, _markdown_unit) + '"},]'


def gen_array_unescaped_quotes(n: int) -> str:
    return '{"quotes": [' + _fill(n, lambda i: f'"She said "yes" to offer {i}"', ", ") + "]}"


def gen_single_quoted_apostrophes(n: int) -> str:
    return "[" + _fill(n, lambda i: f"'it's item {i}'", ", ") + "]"


def gen_markdown_paren_list(n: int) -> str:
    lines = _fill(n, lambda i: f"({i + 1}) Check the {WORDS[i % len(WORDS)]} step and go on.\n")
    return "Here is the plan:\n\n" + lines + '\n```json\n{"status": "ok", "steps": 3}\n```\n'


def gen_records_missing_separators(n: int) -> str:
    return "[{" + _fill(n, lambda i: f'"id": {i}, "name": "item {i}" ') + "}]"


def gen_table_rows_continued(n: int) -> str:
    rows = _fill(n, lambda i: f', ["row {i + 1}", {i + 1}]')
    return '{"columns": ["name", "count"], "rows": [["row 0", 0]]' + rows + "}"


def gen_unbalanced_close_braces(n: int) -> str:
    return '{"note": "' + _fill(n, lambda i: "ab} ")


def gen_regex_class_quotes(n: int) -> str:
    return '{"pattern": "[' + _fill(n, lambda i: 'x"')


def gen_prose_fragments(n: int) -> str:
    return _fill(n, lambda i: f'Result {i}: {{"id": {i}, "ok": true, "tags": ["a"]}} and then\n')


_ACC = "parse_string._append_string_content: `state.string_acc += content` per char"
FAMILIES: dict[str, Family] = {
    f.name: f
    for f in [
        Family(
            "truncated_string",
            "realistic",
            gen_truncated_string,
            "streamed answer cut off mid-string (one long string value, no closing quote)",
            _ACC + " (an attribute target, so CPython copies the whole accumulator each time)",
        ),
        Family(
            "escaped_code_string",
            "realistic",
            gen_escaped_code_string,
            'long code string with \\n and \\" escapes; doc broken only by a trailing comma',
            _ACC + ", plus `_replace_trailing_backslash` (`string_acc[:-1]`) per escape",
        ),
        Family(
            "escaped_code_string_skip_loads",
            "realistic",
            gen_escaped_code_valid,
            "the same string in VALID JSON, called with skip_json_loads=True",
            "as escaped_code_string (a string with escapes misses the quoted-string fast path)",
            {"skip_json_loads": True},
        ),
        Family(
            "markdown_in_array",
            "realistic",
            gen_markdown_in_array,
            "markdown (links, \\n) in a string inside an array; broken by a trailing comma",
            "parse_string._scan_string_body: each `]` runs an uncached "
            "`skip_to_character(rstring_delimiter)` to the closing quote",
        ),
        Family(
            "array_unescaped_quotes",
            "realistic",
            gen_array_unescaped_quotes,
            'array of strings with unescaped inner quotes (`"She said "yes" to offer 3"`)',
            "_handle_right_delimiter_candidate, ARRAY branch: each inner quote walks every later "
            "quote pair up to the `]`",
        ),
        Family(
            "single_quoted_apostrophes",
            "realistic",
            gen_single_quoted_apostrophes,
            "Python-style single-quoted list items with apostrophes (`'it's item 3'`)",
            "as array_unescaped_quotes (each apostrophe is a candidate closing quote)",
        ),
        Family(
            "markdown_paren_list",
            "realistic",
            gen_markdown_paren_list,
            "prose with `(1) ...` list lines before a fenced JSON block",
            "parser_parenthesized.top_level_parenthesized_can_start_value: "
            "`json_str[i:].lower()` of the rest of the input for every `(` at a line start",
        ),
        Family(
            "records_missing_separators",
            "realistic",
            gen_records_missing_separators,
            'array of records whose `}, {` separators were dropped (`"name": "a" "id": 1`)',
            "parse_object._split_object_on_duplicate_key rebuilds the input string per record",
        ),
        Family(
            "table_rows_continued",
            "realistic",
            gen_table_rows_continued,
            'table rows continued after the rows array closed early (`"rows": [[..]], [..]`)',
            "parse_object._merge_object_array_continuation rescans all accumulated rows "
            "(`list_lengths`) per continued row",
        ),
        Family(
            "unbalanced_close_braces",
            "adversarial",
            gen_unbalanced_close_braces,
            "truncated string value with many `}` and no later quote or colon",
            "parse_string._scan_string_body: per `}`, an uncached `skip_to_character(':')` "
            "to the end of the input",
        ),
        Family(
            "regex_class_quotes",
            "adversarial",
            gen_regex_class_quotes,
            "string value with an unclosed `[` then many bare quotes, no whitespace",
            "_quote_belongs_to_regex_character_class: per quote, `string_acc[start:]` + "
            "any(isspace) + `skip_to_character(']')` to the end",
        ),
        Family(
            "nested_open_tuples",
            "adversarial",
            lambda n: "[" + "(" * 100 + "1, " * max(1, (n - 101) // 3),
            "100 unclosed nested tuples, then a long flat tail (`[(((... 1, 1, 1`)",
            "parser_parenthesized.parenthesized_is_explicit_tuple scans to the end of the input "
            "from every nested `(`",
        ),
        Family(
            "nested_open_containers",
            "adversarial",
            lambda n: "{[" * 60 + "," * max(1, n - 120),
            "60 unclosed `{[` pairs, then a long run of commas",
            "parse_object._repair_empty_object_result / _classify_empty_object_repair and "
            "_prepare_string_entry rescan the rest of the input at every level",
        ),
        Family(
            "valid_records",
            "control",
            lambda n: make_valid_doc(n, 7),
            "valid JSON records, default args (json.loads fast path)",
            "linear",
        ),
        Family(
            "valid_records_skip_loads",
            "control",
            lambda n: make_valid_doc(n, 7),
            "the same valid JSON with skip_json_loads=True (the parser; short strings)",
            "linear",
            {"skip_json_loads": True},
        ),
        Family(
            "broken_records",
            "control",
            lambda n: make_broken_doc(n, 7, truncate=True),
            "typical broken LLM output (the docs-section generator), truncated",
            "linear",
        ),
        Family(
            "prose_fragments",
            "control",
            gen_prose_fragments,
            "many small JSON objects separated by prose",
            "linear",
        ),
    ]
}

# ------------------------------------------------------------------------------------------ gating


def _outcome(fn: Callable[..., Any], s: str, kwargs: dict[str, Any]) -> tuple[bool, Any]:
    """``(True, fn(s, **kwargs))``, or ``(False, "Type: message")`` when it raises."""
    try:
        return True, fn(s, **kwargs)
    except Exception as e:  # noqa: BLE001 - the exception is part of the compared behaviour
        return False, f"{type(e).__name__}: {e}"


def fingerprint(value: Any) -> str:
    """sha256 of a type-tagged, order-preserving serialization (bool/int/float distinct)."""
    h = hashlib.sha256()
    stack = [value]
    while stack:
        v = stack.pop()
        t = type(v)
        if t is str:
            b = v.encode("utf-8", "surrogatepass")
            h.update(b"s%d:" % len(b))
            h.update(b)
        elif v is None:
            h.update(b"N")
        elif t is bool:
            h.update(b"T" if v else b"F")
        elif t is int:
            h.update(f"i{v};".encode())
        elif t is float:
            h.update(f"f{v!r};".encode())
        elif t is dict:
            h.update(b"{%d:" % len(v))
            for k, x in reversed(list(v.items())):
                stack += [x, k]
        elif t is list:
            h.update(b"[%d:" % len(v))
            stack.extend(reversed(v))
        else:
            h.update(f"?{t.__name__}:{v!r};".encode())
    return h.hexdigest()


def fp_outcome(outcome: tuple[bool, Any]) -> str:
    ok, value = outcome
    return fingerprint(value) if ok else "raise:" + value


def _head(outcome: tuple[bool, Any], n: int = 160) -> str:
    ok, value = outcome
    return repr(value)[:n] if ok else "raise " + value[:n]


def _loads_fn(mod: Any) -> Callable[..., Any]:
    loads = getattr(mod, "loads", None)
    if loads is not None:
        return loads
    return lambda s, **kw: mod.repair_json(s, return_objects=True, **kw)


def _first_diff(a: str, b: str) -> str:
    pairs = enumerate(zip(a, b, strict=False))
    i = next((k for k, (x, y) in pairs if x != y), min(len(a), len(b)))
    return f"first difference at char {i}: {a[i : i + 60]!r} vs {b[i : i + 60]!r} " + (
        f"(len {len(a)} vs {len(b)})"
    )


class GateError(SystemExit):
    pass


# ------------------------------------------------------------------------------------------ timing


def time_call(fn: Callable[[], Any], repeat: int, min_sample_ns: float, max_total_s: float) -> dict[str, Any]:
    """Best and median ns per call.

    One warm-up call, then up to ``repeat`` samples of ``loops`` calls each (``loops`` chosen so a
    sample takes >= ``min_sample_ns``), stopping after ``max_total_s`` once 3 samples exist. The GC
    stays enabled (building the result is part of the work) but is collected before each sample.
    """
    t0 = time.perf_counter_ns()
    fn()
    first = max(1, time.perf_counter_ns() - t0)
    loops = max(1, math.ceil(min_sample_ns / first))
    samples: list[float] = []
    deadline = time.perf_counter() + max_total_s
    while len(samples) < repeat and (len(samples) < 3 or time.perf_counter() < deadline):
        gc.collect()
        t0 = time.perf_counter_ns()
        for _ in range(loops):
            fn()
        samples.append((time.perf_counter_ns() - t0) / loops)
    return {
        "best_ns": min(samples),
        "median_ns": statistics.median(samples),
        "samples": len(samples),
        "loops": loops,
    }


def fmt_ns(ns: float | None) -> str:
    if ns is None:
        return "-"
    if ns < 1e3:
        return f"{ns:.0f} ns"
    if ns < 1e6:
        return f"{ns / 1e3:.1f} us"
    if ns < 1e9:
        return f"{ns / 1e6:.2f} ms"
    return f"{ns / 1e9:.2f} s"


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1 << 20:
        return f"{n / 1024:.0f} KiB"
    return f"{n / (1 << 20):.1f} MiB"


def loglog_slope(points: list[tuple[int, float]]) -> float | None:
    """Least-squares slope of log(time) on log(bytes)."""
    if len(points) < 2:
        return None
    xs = [math.log(x) for x, _ in points]
    ys = [math.log(y) for _, y in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    return num / den if den else None


# ------------------------------------------------------------------------------------------- impls


@dataclass
class Impl:
    label: str
    module: str
    mod: Any
    version: str
    baseline: bool = False


def _version(module: str, mod: Any) -> str:
    v = getattr(mod, "__version__", None)
    if v:
        return str(v)
    try:
        return md.version(module.split(".")[0].replace("_", "-"))
    except md.PackageNotFoundError:
        return "?"


def resolve_impls(names: list[str]) -> tuple[Impl, list[Impl], list[str]]:
    """(baseline, candidates, skipped): the first ``json_repair`` is the baseline."""
    ref_mod = importlib.import_module(REFERENCE)
    baseline = Impl(REFERENCE, REFERENCE, ref_mod, _version(REFERENCE, ref_mod), baseline=True)
    candidates: list[Impl] = []
    skipped: list[str] = []
    seen: dict[str, int] = {REFERENCE: 1}
    baseline_listed = False
    for name in names:
        if name == REFERENCE and not baseline_listed:
            baseline_listed = True
            continue
        try:
            mod = importlib.import_module(name)
        except ImportError as e:
            print(f"warning: --impl {name} skipped: {e}", file=sys.stderr)
            skipped.append(name)
            continue
        seen[name] = seen.get(name, 0) + 1
        label = name if seen[name] == 1 else f"{name}#{seen[name]}"
        candidates.append(Impl(label, name, mod, _version(name, mod)))
    return baseline, candidates, skipped


def gil_enabled() -> bool | None:
    is_enabled = getattr(sys, "_is_gil_enabled", None)
    return bool(is_enabled()) if is_enabled is not None else None


def _loadavg() -> list[float] | None:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return None


def free_threaded_build() -> bool:
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def machine_info(impls: list[Impl]) -> dict[str, Any]:
    affinity = getattr(os, "sched_getaffinity", None)
    process_cpu_count = getattr(os, "process_cpu_count", None)
    return {
        "label": LABEL,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "free_threaded_build": free_threaded_build(),
        "gil_enabled": gil_enabled(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "process_cpu_count": process_cpu_count() if process_cpu_count else None,
        "affinity_cpus": len(affinity(0)) if affinity else None,
        "loadavg_before": _loadavg(),
        "git_sha": os.environ.get("BENCH_GIT_SHA"),
        "impls": {i.label: {"module": i.module, "version": i.version} for i in impls},
    }


# ------------------------------------------------------------------------------ section: documents

OPS: dict[str, Callable[[Any], Callable[..., Any]]] = {
    "repair_json": lambda m: m.repair_json,
    "loads": _loads_fn,
}


def section_docs(baseline: Impl, candidates: list[Impl], args: argparse.Namespace) -> list[dict[str, Any]]:
    docs: list[tuple[str, str]] = []
    for n in args.doc_sizes:
        docs.append((f"valid {fmt_bytes(n)}", make_valid_doc(n, seed=args.seed)))
        docs.append((f"broken {fmt_bytes(n)}", make_broken_doc(n, seed=args.seed, truncate=True)))
    # Gate every doc and op before timing anything.
    for name, s in docs:
        for op, get in OPS.items():
            ref = _outcome(get(baseline.mod), s, {})
            for c in candidates:
                got = _outcome(get(c.mod), s, {})
                if fp_outcome(got) == fp_outcome(ref):
                    continue
                if op == "repair_json" and got[0] and ref[0]:
                    detail = _first_diff(got[1], ref[1])
                else:
                    detail = f"got {_head(got)} want {_head(ref)}"
                raise GateError(f"GATE MISMATCH docs/{name}/{op}: {c.label}: {detail}")
    log(f"gate: docs pass ({len(docs)} docs x {len(OPS)} ops x {len(candidates)} candidates)")
    rows: list[dict[str, Any]] = []
    for name, s in docs:
        for op, get in OPS.items():
            for impl in [baseline, *candidates]:
                fn = get(impl.mod)
                t = time_call(lambda fn=fn, s=s: fn(s), args.repeat, args.min_sample_ms * 1e6, args.max_row_s)
                rows.append({"doc": name, "bytes": len(s), "op": op, "impl": impl.label, **t})
                log(f"  docs {name:>14} {op:<11} {impl.label:<22} {fmt_ns(t['best_ns'])}")
    return rows


# --------------------------------------------------------------------------- section: pathological


def _worker() -> int:
    """Subprocess side of one point: a JSON request on stdin, one JSON line per phase on stdout."""
    req = json.loads(sys.stdin.readline())

    def emit(msg: dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()

    mod = importlib.import_module(req["module"])
    fam = FAMILIES[req["family"]]
    s = fam.gen(req["nbytes"])
    emit({"phase": "ready", "len": len(s), "gil_enabled": gil_enabled()})
    t0 = time.perf_counter_ns()
    out = _outcome(mod.repair_json, s, fam.kwargs)
    times = [time.perf_counter_ns() - t0]
    emit({"phase": "call", "ns": times[0]})
    expect = req.get("expect")
    fp_str = fp_outcome(out)
    if expect is not None:
        fp_obj = fp_outcome(_outcome(_loads_fn(mod), s, fam.kwargs))
        ok = fp_str == expect["str"] and fp_obj == expect["obj"]
        emit({"phase": "gate", "ok": ok, "str_ok": fp_str == expect["str"], "head": _head(out)})
        if not ok:
            return 0
    while len(times) < req["max_runs"] and sum(times) < req["budget_s"] * 1e9:
        t0 = time.perf_counter_ns()
        _outcome(mod.repair_json, s, fam.kwargs)
        times.append(time.perf_counter_ns() - t0)
        emit({"phase": "call", "ns": times[-1]})
    done: dict[str, Any] = {"phase": "done", "times_ns": times}
    if expect is None:
        # json_repair 0.64.5 returns json.dumps(loads(s)) (or "" for ""): this is its loads() value.
        obj = (True, json.loads(out[1]) if out[1] != "" else "") if out[0] else out
        done["fp"] = {"str": fp_str, "obj": fp_outcome(obj)}
        done["head"] = _head(out)
    emit(done)
    return 0


def run_point(
    impl: Impl, family: str, nbytes: int, expect: dict[str, str] | None, args: argparse.Namespace
) -> dict[str, Any]:
    """Run one (impl, family, size) point in a subprocess; kill it when a phase exceeds the timeout."""
    req = {
        "module": impl.module,
        "family": family,
        "nbytes": nbytes,
        "expect": expect,
        "budget_s": args.budget,
        "max_runs": args.patho_runs,
    }
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.close()
    res: dict[str, Any] = {"impl": impl.label, "family": family, "nbytes": nbytes, "status": "ok"}
    times: list[int] = []
    timeout = 120.0  # imports and input generation
    while True:
        try:
            line = lines.get(timeout=timeout)
        except queue.Empty:
            proc.kill()
            res["status"] = "timeout"
            break
        if line is None:
            proc.wait()
            if res.get("phase") not in ("done", "gate"):
                err = (proc.stderr.read() if proc.stderr else "").strip().splitlines()
                res["status"] = "error"
                res["error"] = err[-1] if err else f"exit {proc.returncode}"
            break
        msg = json.loads(line)
        res["phase"] = msg["phase"]
        timeout = args.timeout
        if msg["phase"] == "ready":
            res["len"] = msg["len"]
            res["gil_enabled"] = msg["gil_enabled"]
        elif msg["phase"] == "call":
            times.append(msg["ns"])
        elif msg["phase"] == "gate":
            res.update(gate="pass" if msg["ok"] else "MISMATCH", str_ok=msg["str_ok"])
            res["head"] = msg["head"]
        elif msg["phase"] == "done":
            times = msg["times_ns"]
            res["fp"] = msg.get("fp")
            res["ref_head"] = msg.get("head")
    proc.wait()
    if proc.stderr:
        proc.stderr.close()
    res["times_ns"] = times
    if res["status"] == "ok" and times:
        timed = times[1:] if len(times) >= 3 else times  # with >= 3 calls the first is a warm-up
        res.update(best_ns=min(timed), median_ns=statistics.median(timed), runs=len(timed))
    else:
        res["best_ns"] = None
    return res


def section_patho(baseline: Impl, candidates: list[Impl], args: argparse.Namespace) -> list[dict[str, Any]]:
    sizes = [args.min_bytes << k for k in range(64) if args.min_bytes << k <= args.max_bytes]
    rows: list[dict[str, Any]] = []
    for fam_name in args.families:
        stopped: set[str] = set()
        for nbytes in sizes:
            expect: dict[str, str] | None = None  # the reference's fingerprints at this size
            for impl in [baseline, *candidates]:
                if impl.label in stopped:
                    continue
                res = run_point(impl, fam_name, nbytes, None if impl.baseline else expect, args)
                if impl.baseline:
                    expect = res.get("fp") if res["status"] == "ok" else None
                elif expect is None:
                    res["gate"] = "unchecked"
                rows.append(res)
                if res["status"] == "ok":
                    shown = fmt_ns(res["best_ns"])
                elif res["status"] == "timeout":
                    shown = f"> {args.timeout:.0f} s"
                else:
                    shown = f"error: {res.get('error')}"
                size = fmt_bytes(res.get("len", nbytes))
                log(f"  patho {fam_name:<31} {size:>9} {impl.label:<22} {shown} {res.get('gate', '')}")
                if res.get("gate") == "MISMATCH":
                    raise GateError(
                        f"GATE MISMATCH patho/{fam_name}/{nbytes}: {impl.label} != json_repair "
                        f"(repair_json equal: {res.get('str_ok')}); got {res.get('head')}"
                    )
                if res.get("gil_enabled") and free_threaded_build():
                    raise GateError(f"GIL GUARD: importing {impl.module} re-enabled the GIL")
                if res["status"] != "ok" or res["best_ns"] > args.stop_after * 1e9:
                    stopped.add(impl.label)
            if len(stopped) == 1 + len(candidates):
                break
    return rows


def slopes(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | None]]]:
    """Per family and impl, log-log slopes: ``all`` over the points that took >= 1 ms, ``top3``
    over the three largest measured sizes (the asymptotic exponent; per-char overhead dominates
    small sizes)."""
    pts: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for r in rows:
        per = pts.setdefault(r["family"], {}).setdefault(r["impl"], [])
        if r["status"] == "ok" and r["best_ns"] >= 1e6:
            per.append((r["len"], r["best_ns"]))
    return {
        fam: {impl: {"all": loglog_slope(p), "top3": loglog_slope(p[-3:])} for impl, p in d.items()}
        for fam, d in pts.items()
    }


# ---------------------------------------------------------------------------------- section: batch


def _pool_map(fn: Callable[[str], str], docs: list[str], threads: int) -> list[str]:
    step = math.ceil(len(docs) / threads)
    chunks = [docs[i : i + step] for i in range(0, len(docs), step)]
    with ThreadPoolExecutor(threads) as ex:
        parts = list(ex.map(lambda c: [fn(d) for d in c], chunks))
    return [x for part in parts for x in part]


def section_batch(baseline: Impl, candidates: list[Impl], args: argparse.Namespace) -> list[dict[str, Any]]:
    docs = [make_broken_doc(args.batch_doc_bytes, seed=10_000 + i) for i in range(args.batch_docs)]
    total_bytes = sum(len(d) for d in docs)
    ref_fn = baseline.mod.repair_json
    ref_out = [ref_fn(d) for d in docs]  # also the warm-up
    runs: list[tuple[Impl, str, int, Callable[[], list[str]]]] = [
        (baseline, "loop", 1, lambda: [ref_fn(d) for d in docs])
    ]
    if gil_enabled() is False:
        pool = args.pool_threads
        runs.append((baseline, "thread pool", pool, lambda: _pool_map(ref_fn, docs, pool)))
    for c in candidates:
        fn = c.mod.repair_json
        runs.append((c, "loop", 1, lambda fn=fn: [fn(d) for d in docs]))
        batch = getattr(c.mod, "repair_json_batch", None)
        if batch is not None:
            runs += [
                (c, "repair_json_batch", t, lambda b=batch, t=t: list(b(docs, threads=t)))
                for t in args.batch_threads
            ]
    rows: list[dict[str, Any]] = []
    for impl, mode, threads, fn in runs:
        got = fn()  # gate (and warm up) before timing
        if got != ref_out:
            bad = next(i for i, (a, b) in enumerate(zip(got, ref_out, strict=False)) if a != b)
            raise GateError(
                f"GATE MISMATCH batch/{impl.label}/{mode}/threads={threads}: doc {bad}: "
                + _first_diff(got[bad], ref_out[bad])
            )
        samples = []
        for _ in range(args.batch_repeat):
            gc.collect()
            t0 = time.perf_counter_ns()
            fn()
            samples.append(time.perf_counter_ns() - t0)
        best = min(samples)
        rows.append(
            {
                "impl": impl.label,
                "mode": mode,
                "threads": threads,
                "docs": len(docs),
                "bytes": total_bytes,
                "best_ns": best,
                "median_ns": statistics.median(samples),
                "docs_per_s": len(docs) / (best / 1e9),
                "gate": "pass",
            }
        )
        log(f"  batch {impl.label:<22} {mode:<17} t={threads:<2} {rows[-1]['docs_per_s']:,.0f} docs/s")
    return rows


# ----------------------------------------------------------------------------------------- summary


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _speedups(base: float | None, others: list[float | None]) -> str:
    return " / ".join(f"{base / o:.1f}x" if base and o else "-" for o in others)


def summary(results: dict[str, Any], baseline: Impl, candidates: list[Impl], args: argparse.Namespace) -> str:
    m = results["machine"]
    labels = [baseline.label, *(c.label for c in candidates)]
    vs = f" speedup vs json_repair {baseline.version} |" if candidates else ""
    vs_rule = "---:|" if candidates else ""
    gil = {None: "", True: ", GIL enabled", False: ", GIL disabled"}[m["gil_enabled"]]
    build = " free-threaded" if m["free_threaded_build"] else ""
    versions = ", ".join(f"{k} {v['version']}" for k, v in m["impls"].items())
    out = [
        f"## json_repair benchmark ({LABEL})",
        "",
        (
            f"Python {m['python_version']}{build}{gil}; {m['platform']}; {m['cpu_count']} CPUs; "
            f"{versions}. Load average before/after: {m['loadavg_before']} / "
            f"{results.get('loadavg_after')}. Speedups compare against json_repair on the same "
            "interpreter."
        ),
        "",
    ]
    if results.get("docs"):
        out += [
            "### Single documents (best per call)",
            "",
            f"| doc | bytes | op | {' | '.join(labels)} |{vs}",
            "|---|---:|---|" + "---:|" * len(labels) + vs_rule,
        ]
        by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        for r in results["docs"]:
            by_key.setdefault((r["doc"], r["op"]), {})[r["impl"]] = r
        for (doc, op), per in by_key.items():
            cells = " | ".join(fmt_ns(per[label]["best_ns"]) for label in labels)
            base = per[baseline.label]
            sp = _speedups(base["best_ns"], [per[c.label]["best_ns"] for c in candidates])
            out.append(f"| {doc} | {base['bytes']:,} | {op} | {cells} |" + (f" {sp} |" * bool(vs)))
        out.append("")
    if results.get("patho"):
        out += [
            (
                f"### Pathological curves (best per call; > {args.timeout:.0f} s = killed; "
                "* = unchecked: json_repair did not reach this size)"
            ),
            "",
            f"| family | kind | bytes | {' | '.join(labels)} |{vs}",
            "|---|---|---:|" + "---:|" * len(labels) + vs_rule,
        ]
        by_point: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
        for r in results["patho"]:
            by_point.setdefault((r["family"], r["nbytes"]), {})[r["impl"]] = r

        def cell(r: dict[str, Any] | None) -> str:
            if r is None:
                return "-"
            if r["status"] == "timeout":
                return f"> {args.timeout:.0f} s"
            if r["status"] != "ok":
                return "error"
            return fmt_ns(r["best_ns"]) + ("*" if r.get("gate") == "unchecked" else "")

        for (fam, nbytes), per in by_point.items():
            size = next((r["len"] for r in per.values() if "len" in r), nbytes)
            cells = " | ".join(cell(per.get(label)) for label in labels)
            base_ns = per.get(baseline.label, {}).get("best_ns")
            sp = _speedups(base_ns, [per.get(c.label, {}).get("best_ns") for c in candidates])
            kind = FAMILIES[fam].kind
            out.append(f"| {fam} | {kind} | {size:,} | {cells} |" + (f" {sp} |" * bool(vs)))
        out += [
            "",
            (
                "### Time exponents (log-log slope of best time vs bytes over the points >= 1 ms; "
                "in brackets over the 3 largest sizes)"
            ),
            "",
            f"| family | kind | {' | '.join(labels)} |",
            "|---|---|" + "---:|" * len(labels),
        ]

        def fmt_slope(s: dict[str, float | None] | None) -> str:
            if not s or s["all"] is None:
                return "-"
            return f"{s['all']:.2f}" + ("" if s["top3"] is None else f" ({s['top3']:.2f})")

        for fam, per_impl in results["slopes"].items():
            cells = " | ".join(fmt_slope(per_impl.get(label)) for label in labels)
            out.append(f"| {fam} | {FAMILIES[fam].kind} | {cells} |")
        out.append("")
    if results.get("batch"):
        b0 = results["batch"][0]
        out += [
            (
                f"### Batch: {b0['docs']:,} broken docs of ~{fmt_bytes(args.batch_doc_bytes)} "
                f"({b0['bytes']:,} bytes)"
            ),
            "",
            "| impl | mode | threads | best | docs/s | speedup vs json_repair loop |",
            "|---|---|---:|---:|---:|---:|",
        ]
        out += [
            f"| {r['impl']} | {r['mode']} | {r['threads']} | {fmt_ns(r['best_ns'])} | "
            f"{r['docs_per_s']:,.0f} | {b0['best_ns'] / r['best_ns']:.1f}x |"
            for r in results["batch"]
        ]
        out.append("")
    return "\n".join(out)


# -------------------------------------------------------------------------------------------- main


def _csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    add = p.add_argument
    add("--impl", action="append", default=[], help="module to compare (repeatable)")
    add("--out", type=Path, help="write the JSON results here")
    add("--sections", default="docs,patho,batch", help="comma list of docs,patho,batch")
    add("--families", default="all", help="comma list of pathological families (see --list)")
    add("--list", action="store_true", help="list the pathological families and exit")
    add("--quick", action="store_true", help="small sizes everywhere (a smoke test)")
    add("--seed", type=int, default=1234)
    add("--doc-sizes", type=_csv_ints, default=[1024, 1 << 20], help="docs: sizes in bytes")
    add("--repeat", type=int, default=15, help="docs: max samples per row")
    add("--min-sample-ms", type=float, default=20.0, help="docs: min duration of a sample")
    add("--max-row-s", type=float, default=4.0, help="docs: stop sampling a row after this")
    add("--min-bytes", type=int, default=2048, help="patho: smallest size")
    add("--max-bytes", type=int, default=1 << 20, help="patho: largest size")
    add("--timeout", type=float, default=20.0, help="patho: kill a call after this many s")
    add("--stop-after", type=float, default=10.0, help="patho: stop doubling after a call > s")
    add("--budget", type=float, default=2.0, help="patho: repeat a point while its total < s")
    add("--patho-runs", type=int, default=6, help="patho: max calls per point (1st = warm-up)")
    add("--batch-docs", type=int, default=10_000)
    add("--batch-doc-bytes", type=int, default=1024)
    add("--batch-threads", type=_csv_ints, default=[1, 8])
    add("--batch-repeat", type=int, default=3)
    add("--pool-threads", type=int, default=8, help="batch: json_repair pool (free-threaded)")
    args = p.parse_args(argv)
    if args.quick:
        args.doc_sizes = [1024, 32 * 1024]
        args.min_bytes, args.max_bytes = 1024, 16 * 1024
        args.repeat, args.max_row_s, args.budget = 5, 1.0, 0.5
        args.batch_docs, args.batch_repeat = 300, 2
    args.families = list(FAMILIES) if args.families == "all" else args.families.split(",")
    unknown = [f for f in args.families if f not in FAMILIES]
    if unknown:
        p.error(f"unknown families: {', '.join(unknown)}")
    args.sections = args.sections.split(",")
    return args


def main(argv: list[str] | None = None) -> int:
    if argv is None and sys.argv[1:] == ["--worker"]:
        return _worker()
    args = parse_args(argv)
    if args.list:
        for f in FAMILIES.values():
            kw = f" {f.kwargs}" if f.kwargs else ""
            print(f"{f.name:<31} {f.kind:<11} {f.desc}{kw}\n{'':<43} cause: {f.cause}")
        return 0
    baseline, candidates, skipped = resolve_impls(args.impl)
    if baseline.version != REFERENCE_VERSION:
        log(f"warning: json_repair {baseline.version} is not the pinned {REFERENCE_VERSION}")
    if gil_enabled() and free_threaded_build():
        raise GateError("GIL GUARD: an imported module re-enabled the GIL on a free-threaded build")
    results: dict[str, Any] = {
        "machine": machine_info([baseline, *candidates]),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "skipped_impls": skipped,
    }
    try:
        if "docs" in args.sections:
            results["docs"] = section_docs(baseline, candidates, args)
        if "patho" in args.sections:
            results["patho"] = section_patho(baseline, candidates, args)
            results["slopes"] = slopes(results["patho"])
        if "batch" in args.sections:
            results["batch"] = section_batch(baseline, candidates, args)
    except GateError as e:
        results["aborted"] = str(e)
        log(str(e))
        return 1
    finally:
        # Other sessions may share the VM: the load average before (machine) and after the run.
        results["loadavg_after"] = _loadavg()
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(results, indent=1))
    print(summary(results, baseline, candidates, args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
