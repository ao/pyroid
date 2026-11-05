#!/usr/bin/env python3
"""Per-stage pyperf benchmarks, equivalence-gated before anything is timed (R2 section 5).

    cd bench && uv run --frozen python stages.py --list
    uv run --frozen python stages.py --stage fix_text --docs 1000 --fast -o /target/r/fix_text-d1000.json
    uv run --frozen python stages.py --stage gopher_quality --docs 100 --debug-single-value
    uv run --frozen python stages.py --stage fix_text --impl ftfy=pyroid.ftfy --docs 10000 --rigorous -o ...
    BENCH_THREADS=8 RAYON_NUM_THREADS=8 uv run --frozen python stages.py --stage minhash ...   (3.14t/3.15t only)

Stages: fix_text, lid, one per ``specs.FILTER_CASES`` case, minhash. Each stage has fixed reference rows
(the incumbents) plus one candidate row per ``--impl SUITE=MODULE`` (resolved with ``pyroid_parity.impl``).

* One benchmark value = wall time of ONE call over a batch of ``--docs N`` docs (the first N docs of
  ``--corpus`` after ``--offset``). Per-doc time = value / metadata["docs"]. Sweep N with 1, 100, 1000, 10000.
* The manager (parent) process runs the equivalence gates (``gates.py``) on exactly the timed docs, for every
  row, before pyperf spawns any worker; the gate dict is stored in each benchmark's metadata ("gate").
* GIL guard: on a free-threaded build, abort if any imported extension re-enabled the GIL (rensa 0.5.0 does).
* ``BENCH_*``, ``PYROID_*`` and ``RAYON_NUM_THREADS`` are always forwarded to workers (``--inherit-environ``).
* Reference rows are per-doc loops (``BENCH_THREADS`` > 1 maps them over a thread pool, which only scales on
  free-threaded builds); candidate rows get the whole batch in one call and thread internally.
* Filter rows include tokenization: datatrove's ``split_into_parts`` 5-slot LRU cache is cleared before each
  doc, so a 1-doc batch cannot hit a warm cache across loops (in the pipeline, chained filters share it).
* Peak memory: a SECOND run with pyperf ``--track-memory`` (it replaces the timing values).
"""

from __future__ import annotations

import argparse
import atexit
import importlib
import json
import os
import sys
import sysconfig
import tempfile
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyperf
from pyroid_parity import data as D
from pyroid_parity import impl as I
from pyroid_parity import specs

import gates as G

STAGES = ["fix_text", "lid", *specs.FILTER_CASES, "minhash", "translit"]
SUITE_OF = {
    "fix_text": "ftfy",
    "lid": "fasttext",
    "minhash": "dedup",
    "translit": "translit",
    **dict.fromkeys(specs.FILTER_CASES, "filters"),
}
THREADS = int(os.environ.get("BENCH_THREADS", "1"))
# BENCH_FRESH_STRS=1: every timed call gets new str objects (built untimed), as a pipeline that reads each doc
# once does. CPython caches a non-ASCII str's UTF-8 on first use, so by default (warm) every call after the
# first skips that encoding; extensions that borrow UTF-8 (PyO3 PyBackedStr) pay it only when fresh.
FRESH = os.environ.get("BENCH_FRESH_STRS", "0") == "1"
# Suites that pyroid_parity.impl does not know (no parity contract there yet); resolved here by module path.
LOCAL_SUITES = ("translit",)

Batch = Callable[[list[str]], Any]


@dataclass
class Row:
    """One benchmark row. ``build()`` returns ``(timed, gate)``: the timed batch callable and a thunk that
    gates that same object on the timed docs."""

    name: str
    impl: str
    version: str
    reference: bool
    boundary: str
    build: Callable[[], tuple[Batch, Callable[[], dict[str, Any]]]]
    tokenization: str = "n/a"
    # Untimed conversion of the docs into the timed call's input (e.g. an Arrow array); None = the list itself.
    prep: Callable[[list[str]], Any] | None = None
    input_kind: str = "list[str]"


@dataclass
class Ctx:
    stage: str
    ids: list[str]
    texts: list[str]
    gate_workers: int
    impls: list[str]


# ------------------------------------------------------------------------------------------------ helpers

_POOL: list[ThreadPoolExecutor] = []


def pool() -> ThreadPoolExecutor:
    """Created on first timed call only (never before the gates fork their worker processes)."""
    if not _POOL:
        _POOL.append(ThreadPoolExecutor(THREADS))
    return _POOL[0]


def per_doc(fn: Callable[[str], Any]) -> Batch:
    """Reference rows: a per-doc loop, or ``BENCH_THREADS`` threads mapping it."""
    if THREADS <= 1:
        return lambda docs: [fn(d) for d in docs]
    return lambda docs: list(pool().map(fn, docs))


def per_chunk(fn: Batch) -> Batch:
    """Reference batch APIs (datasketch bulk, rensa from_token_sets): ``BENCH_THREADS`` contiguous chunks."""
    if THREADS <= 1:
        return fn

    def run(docs: list[str]) -> list[Any]:
        step = -(-len(docs) // THREADS)
        return [
            x
            for part in pool().map(fn, [docs[i : i + step] for i in range(0, len(docs), step)])
            for x in part
        ]

    return run


def batch_of(fn: Callable[[str], Any]) -> Batch:
    return lambda docs: [fn(d) for d in docs]


def ver(module: str) -> str:
    try:
        return I.version_of(module) or "?"
    except ImportError:  # e.g. fasttext-predict on 3.14+: its reference rows cannot run there
        return "not installed"


def explicit_suites(spec: str) -> dict[str, str]:
    """``G.explicit_suites`` plus the LOCAL_SUITES (``translit=pyroid.translit`` or a bare ``pyroid.translit``)."""
    suite, sep, module = spec.partition("=")
    if not sep:
        suite, module = spec.rsplit(".", 1)[-1], spec
    if suite in LOCAL_SUITES:
        return {suite: module}
    return G.explicit_suites(spec)


def candidate_specs(ctx: Ctx) -> list[tuple[str, str]]:
    suite = SUITE_OF[ctx.stage]
    return [(m, spec) for spec in ctx.impls for s, m in explicit_suites(spec).items() if s == suite]


def arrow_of(texts: list[str]) -> Any:
    import pyarrow as pa

    return pa.array(texts, pa.string())


def arrow_transform_row(
    ctx: Ctx, module: str, spec: str, label: str, fn: Callable[[Any], Callable[[Any], Any]], gate: Any
) -> Row:
    """A candidate transform fed one pyarrow ``string`` array (built untimed, like the list); it returns a
    ``pyroid.Column``. ``gate(batch_fn, label)`` sees a list-in/list-out wrapper of the same callable."""

    def b() -> tuple[Batch, Callable[[], dict[str, Any]]]:
        timed = fn(importlib.import_module(module))
        name = f"{module} {label} [arrow]"
        return timed, lambda: gate(lambda ts: timed(arrow_of(ts)).to_pylist(), name)

    version = I.version_of(module) or "unknown"
    return Row(
        f"{module} {version} {label} [arrow]",
        spec,
        version,
        False,
        "candidate batch API, one call per batch; input: pyarrow string array (untimed), output: pyroid.Column",
        b,
        prep=arrow_of,
        input_kind="arrow string",
    )


def candidate_row(
    ctx: Ctx,
    module: str,
    spec: str,
    label: str,
    build: Callable[[Any], Batch],
    gate: Callable[[Batch, str], dict[str, Any]],
) -> Row:
    def b() -> tuple[Batch, Callable[[], dict[str, Any]]]:
        obj = I.resolve(SUITE_OF[ctx.stage], module).obj
        timed = build(obj)
        return timed, lambda: gate(timed, f"{module} {label}".strip())

    version = I.version_of(module)
    return Row(
        f"{module} {version or ''} {label}".replace("  ", " ").strip(),
        spec,
        version or "unknown",
        False,
        "candidate batch API: one call per batch (threads via RAYON_NUM_THREADS)",
        b,
    )


# --------------------------------------------------------------------------------------------- stages


def rows_fix_text(ctx: Ctx) -> list[Row]:
    def ref() -> tuple[Batch, Callable[[], dict[str, Any]]]:
        import ftfy

        return per_doc(ftfy.fix_text), lambda: G.gate_ftfy(
            batch_of(ftfy.fix_text), ctx.ids, ctx.texts, "ftfy", ctx.gate_workers, parallel=True
        )

    rows = [
        Row(
            f"ftfy {ver('ftfy')}",
            "ftfy",
            ver("ftfy"),
            True,
            "pure Python ftfy.fix_text, one call per doc",
            ref,
        )
    ]
    for module, spec in candidate_specs(ctx):
        rows.append(
            candidate_row(
                ctx,
                module,
                spec,
                "fix_text_batch",
                lambda o: o.fix_text_batch,
                lambda fn, label: G.gate_ftfy(fn, ctx.ids, ctx.texts, label),
            )
        )
        rows.append(
            arrow_transform_row(
                ctx,
                module,
                spec,
                "fix_text_batch",
                lambda m: m.fix_text_batch,
                lambda fn, label: G.gate_ftfy(fn, ctx.ids, ctx.texts, label),
            )
        )
    return rows


def rows_lid(ctx: Ctx) -> list[Row]:
    v = ver("fasttext")
    rows: list[Row] = []
    for model in specs.LID_MODELS:
        for k in (1, 5):

            def ref(model: str = model, k: int = k) -> tuple[Batch, Callable[[], dict[str, Any]]]:
                import fasttext

                m = fasttext.load_model(G.lid_model_path(model))

                def one(t: str) -> Any:
                    return m.predict(t.replace("\n", " "), k=k, threshold=0.0)

                return per_doc(one), lambda: G.gate_lid(
                    batch_of(one),
                    model,
                    k,
                    ctx.ids,
                    ctx.texts,
                    f"fasttext-predict {model} k={k}",
                    ctx.gate_workers,
                    True,
                )

            rows.append(
                Row(
                    f"fasttext-predict {v} {model} k={k}",
                    "fasttext",
                    v,
                    True,
                    "per-doc Python call into C++ fastText (pybind11); text.replace in Python",
                    ref,
                )
            )

    def lid_filter() -> tuple[Batch, Callable[[], dict[str, Any]]]:
        from datatrove.data import Document

        lf = G.language_filter()
        return per_doc(lambda t: lf.filter(Document(text=t, id=""))), lambda: G.gate_lid_filter(
            lambda ts: G.lid_filter_decide(lf, ts),
            ctx.ids,
            ctx.texts,
            "LanguageFilter",
            ctx.gate_workers,
            True,
        )

    rows.append(
        Row(
            f"datatrove {ver('datatrove')} LanguageFilter(en, 0.65, ft176)",
            "datatrove",
            ver("datatrove"),
            True,
            "datatrove LanguageFilter.filter per doc: fastText predict(k=-1: all labels) + Python dict",
            lid_filter,
        )
    )
    for module, spec in candidate_specs(ctx):
        for model in specs.LID_MODELS:
            for k in (1, 5):

                def build(o: Any, model: str = model, k: int = k) -> Batch:
                    m = o.load(G.lid_model_path(model))
                    return lambda ts: [o.predict(m, t.replace("\n", " "), k) for t in ts]

                def gate(fn: Batch, label: str, model: str = model, k: int = k) -> dict[str, Any]:
                    return G.gate_lid(
                        lambda ts: [G.pairs(p) for p in fn(ts)], model, k, ctx.ids, ctx.texts, label
                    )

                rows.append(candidate_row(ctx, module, spec, f"{model} k={k}", build, gate))
                rows.append(lid_batch_row(module, spec, model, k, ctx))
    return rows


def lid_batch_row(module: str, spec: str, model: str, k: int, ctx: Ctx) -> Row:
    """The candidate's own batch API, ``load_model(p).predict(list, k, threshold)`` (one call per batch, so its
    thread pool runs); the per-doc contract row above can't show threads."""

    def b() -> tuple[Batch, Callable[[], dict[str, Any]]]:
        m = importlib.import_module(module).load_model(G.lid_model_path(model))

        def timed(ts: list[str]) -> Any:
            return m.predict([t.replace("\n", " ") for t in ts], k=k, threshold=0.0)

        def gate() -> dict[str, Any]:
            def pred(ts: list[str]) -> list[Any]:
                labels, probs = timed(ts)
                return list(zip(labels, probs, strict=True))

            return G.gate_lid(pred, model, k, ctx.ids, ctx.texts, f"{module} {model} k={k} batch")

        return timed, gate

    version = I.version_of(module) or "unknown"
    return Row(
        f"{module} {version} {model} k={k} batch",
        spec,
        version,
        False,
        "candidate batch API: model.predict(list[str], k) once per batch; text.replace in Python",
        b,
    )


def rows_filter(ctx: Ctx) -> list[Row]:
    case = ctx.stage
    fn_name, cls, cfg = specs.FILTER_CASES[case]

    def ref() -> tuple[Batch, Callable[[], dict[str, Any]]]:
        from datatrove.data import Document
        from datatrove.utils.text import split_into_parts

        filt = G.datatrove_filter(case)

        def one(t: str) -> Any:
            split_into_parts.cache_clear()  # tokenization included in every doc's time
            return filt.filter(Document(text=t, id=""))

        return per_doc(one), lambda: G.gate_filter(
            case,
            lambda ts: [G.dt_decide(filt, t) for t in ts],
            ctx.ids,
            ctx.texts,
            f"datatrove {case}",
            ctx.gate_workers,
            True,
        )

    args = ",".join(f"{k}={v}" for k, v in cfg.items())
    rows = [
        Row(
            f"datatrove {ver('datatrove')} {cls}({args})",
            "datatrove",
            ver("datatrove"),
            True,
            "datatrove filter() per doc in Python; spaCy (Cython) tokenization included",
            ref,
            "included",
        )
    ]
    for module, spec in candidate_specs(ctx):
        rows.append(
            candidate_row(
                ctx,
                module,
                spec,
                fn_name,
                lambda o: lambda ts: o.run(fn_name, ts, cfg),
                lambda fn, label: G.gate_filter(case, fn, ctx.ids, ctx.texts, label),
            )
        )
    return rows


def rows_minhash(ctx: Ctx) -> list[Row]:
    def datatrove() -> tuple[Batch, Callable[[], dict[str, Any]]]:
        step = G.minhash_step()

        def one(t: str) -> Any:
            sh = step.get_shingles(t)
            return step.get_signature(sh) if sh.size else None

        return per_doc(one), lambda: G.gate_minhash_signatures(
            lambda ts: [G.dt_signature(step, t) for t in ts],
            ctx.ids,
            ctx.texts,
            "datatrove",
            ctx.gate_workers,
            True,
        )

    def jac(a: Any, b: Any) -> float:
        return a.jaccard(b)

    def datasketch() -> tuple[Batch, Callable[[], dict[str, Any]]]:
        return per_chunk(G.datasketch_bulk), lambda: G.gate_estimator(G.datasketch_bulk, jac, "datasketch")

    def rensa_batch() -> tuple[Batch, Callable[[], dict[str, Any]]]:
        return per_chunk(G.rensa_batch), lambda: G.gate_estimator(G.rensa_batch, jac, "rensa from_token_sets")

    def rensa_doc() -> tuple[Batch, Callable[[], dict[str, Any]]]:
        from rensa import RMinHash

        def one(t: str) -> Any:
            m = RMinHash(G.NUM_PERM, 42)
            m.update(list(G.r2_shingles(t)))
            return m

        return per_doc(one), lambda: G.gate_estimator(batch_of(one), jac, "rensa RMinHash.update")

    dv, sv, rv = ver("datatrove"), ver("datasketch"), ver("rensa")
    rows = [
        Row(
            f"datatrove {dv} MinhashDedupSignature",
            "datatrove",
            dv,
            True,
            "Python: simplify_text + spaCy word_tokenize + per-5-gram sha1 loop; numpy: 112 permutations + min",
            datatrove,
            "included",
        ),
        Row(
            f"datasketch {sv} MinHash.bulk(num_perm=112)",
            "datasketch",
            sv,
            True,
            "Python: str.split 5-gram shingles + per-shingle sha1 (hashlib); numpy: permutations + min",
            datasketch,
        ),
        Row(
            f"rensa {rv} RMinHash.from_token_sets(112)",
            "rensa",
            rv,
            True,
            "Python: str.split 5-gram shingles; Rust: hashing + permutations, one FFI call per batch",
            rensa_batch,
        ),
        Row(
            f"rensa {rv} RMinHash.update(112) per doc",
            "rensa",
            rv,
            True,
            "Python: str.split 5-gram shingles; Rust: hashing + permutations, one FFI call per doc",
            rensa_doc,
        ),
    ]
    for module, spec in candidate_specs(ctx):
        rows.append(
            candidate_row(
                ctx,
                module,
                spec,
                "minhash_signatures",
                lambda o: lambda ts: o.signatures(ts, specs.MINHASH_CONFIG),
                lambda fn, label: G.gate_minhash_signatures(fn, ctx.ids, ctx.texts, label),
            )
        )
    return rows


def gate_translit(fn: Batch, ids: list[str], texts: list[str], label: str) -> dict[str, Any]:
    """Byte-identical to the anyascii package (live: it is fast and pure Python, so no pinned ref is needed)."""

    from anyascii import anyascii

    t0 = time.perf_counter()
    got = list(fn(texts))
    diffs = [(i, g, w) for i, g, t in zip(ids, got, texts, strict=True) if g != (w := anyascii(t))]
    if diffs:
        i, g, w = diffs[0]
        raise G.GateError(
            f"[translit] {label}: {len(diffs)}/{len(ids)} docs differ ({i}: got {g[:80]!r} want {w[:80]!r})"
        )
    return {
        "gate": "translit",
        "impl": label,
        "docs": len(ids),
        "pinned": 0,
        "live": len(ids),
        "mismatches": 0,
        "seconds": round(time.perf_counter() - t0, 3),
        "oracle": f"anyascii {ver('anyascii')}",
    }


def rows_translit(ctx: Ctx) -> list[Row]:
    def ref() -> tuple[Batch, Callable[[], dict[str, Any]]]:
        from anyascii import anyascii

        return per_doc(anyascii), lambda: gate_translit(batch_of(anyascii), ctx.ids, ctx.texts, "anyascii")

    rows = [
        Row(
            f"anyascii {ver('anyascii')}",
            "anyascii",
            ver("anyascii"),
            True,
            "pure Python anyascii() per doc (ISC; the data pyroid.translit ships)",
            ref,
        )
    ]
    for module, spec in candidate_specs(ctx):

        def b(module: str = module) -> tuple[Batch, Callable[[], dict[str, Any]]]:
            fn = importlib.import_module(module).anyascii
            return fn, lambda: gate_translit(fn, ctx.ids, ctx.texts, f"{module} anyascii")

        version = I.version_of(module) or "unknown"
        rows.append(
            Row(
                f"{module} {version} anyascii",
                spec,
                version,
                False,
                "candidate batch API: one call per batch (threads via RAYON_NUM_THREADS)",
                b,
            )
        )
        rows.append(
            arrow_transform_row(
                ctx,
                module,
                spec,
                "anyascii",
                lambda m: m.anyascii,
                lambda fn, label: gate_translit(fn, ctx.ids, ctx.texts, label),
            )
        )
    return rows


FILTER_CLASS = {
    "gopher_repetition": "GopherRepetition",
    "gopher_quality": "GopherQuality",
    "c4_quality": "C4Quality",
    "fineweb_quality": "FineWebQuality",
}


def direct_rows(ctx: Ctx) -> list[Row]:
    """Candidate rows that call the module's own batch API with no parity adapter in the timed region (the
    generic dedup adapter, for one, converts every signature value with int() in Python)."""
    rows: list[Row] = []
    for module, spec in candidate_specs(ctx):
        version = I.version_of(module) or "unknown"
        if ctx.stage == "minhash":

            def b(module: str = module) -> tuple[Batch, Callable[[], dict[str, Any]]]:
                m = importlib.import_module(module)

                def timed(ts: list[str]) -> Any:
                    return m.minhash_signatures(ts, **specs.MINHASH_CONFIG)

                return timed, lambda: G.gate_minhash_signatures(timed, ctx.ids, ctx.texts, f"{module} direct")

            rows.append(
                Row(
                    f"{module} {version} minhash_signatures direct",
                    spec,
                    version,
                    False,
                    "candidate batch API, no adapter: minhash_signatures(list) -> list[list[int]]",
                    b,
                )
            )

            def fit(module: str = module) -> tuple[Batch, Callable[[], dict[str, Any]]]:
                m = importlib.import_module(module)

                def timed(ts: list[str]) -> Any:
                    return m.MinHashDedup(**specs.MINHASH_CONFIG).fit(ts)

                # Gated on datatrove's removed set over the whole parity corpus (dev + adversarial).
                return timed, lambda: G.gate_minhash_dedup(
                    lambda ts: [not k for k in timed(ts)], f"{module} MinHashDedup.fit"
                )

            rows.append(
                Row(
                    f"{module} {version} MinHashDedup.fit direct",
                    spec,
                    version,
                    False,
                    "candidate: all 4 dedup stages in one call, MinHashDedup(config).fit(list) -> keep mask",
                    fit,
                )
            )
        elif ctx.stage in specs.FILTER_CASES and spec.startswith(("pyroid", "filters=pyroid")):
            fn_name, _, cfg = specs.FILTER_CASES[ctx.stage]

            def b(
                module: str = module, fn_name: str = fn_name, cfg: dict[str, Any] = cfg
            ) -> tuple[Batch, Callable[[], dict[str, Any]]]:
                m = importlib.import_module(module)
                filt = getattr(m, FILTER_CLASS[fn_name])(**cfg)

                def timed(ts: list[str]) -> Any:
                    return m.apply(ts, [filt])

                def decide(ts: list[str]) -> list[Any]:
                    r = timed(ts)
                    return [
                        ("ok", bool(k), rs, o if o != t else None)
                        for k, rs, o, t in zip(r.keep, r.reason, r.text, ts, strict=True)
                    ]

                return timed, lambda: G.gate_filter(ctx.stage, decide, ctx.ids, ctx.texts, f"{module} direct")

            rows.append(
                Row(
                    f"{module} {version} apply([{FILTER_CLASS[fn_name]}]) direct",
                    spec,
                    version,
                    False,
                    "candidate batch API, no adapter: apply(list, [config]) -> FilterResult",
                    b,
                )
            )
    return rows


def per_doc_rows(ctx: Ctx) -> list[Row]:
    """The candidate's single-str call in a Python loop: ``BENCH_THREADS`` Python threads, each looping over
    a contiguous chunk (no executor future per doc). With the GIL, only work done detached overlaps."""
    rows: list[Row] = []
    for module, spec in candidate_specs(ctx):
        version = I.version_of(module) or "unknown"
        if not module.startswith("pyroid."):
            continue

        def make(
            build_fn: Callable[[Any], Callable[[str], Any]],
            gate: Callable[[Batch, str], dict[str, Any]],
            label: str,
            module: str = module,
            spec: str = spec,
            version: str = version,
        ) -> Row:
            def b() -> tuple[Batch, Callable[[], dict[str, Any]]]:
                fn = build_fn(importlib.import_module(module))
                return per_chunk(batch_of(fn)), lambda: gate(batch_of(fn), f"{module} {label} per doc")

            return Row(
                f"{module} {version} {label} per doc",
                spec,
                version,
                False,
                f"candidate single-str call per doc ({label}), BENCH_THREADS Python threads",
                b,
            )

        if ctx.stage == "fix_text":
            rows.append(
                make(lambda m: m.fix_text, lambda f, lb: G.gate_ftfy(f, ctx.ids, ctx.texts, lb), "fix_text")
            )
        elif ctx.stage == "translit":
            rows.append(
                make(lambda m: m.anyascii, lambda f, lb: gate_translit(f, ctx.ids, ctx.texts, lb), "anyascii")
            )
        elif ctx.stage == "minhash":
            rows.append(
                make(
                    lambda m: lambda t: m.minhash_signatures(t, **specs.MINHASH_CONFIG),
                    lambda f, lb: G.gate_minhash_signatures(f, ctx.ids, ctx.texts, lb),
                    "minhash_signatures",
                )
            )
        elif ctx.stage in specs.FILTER_CASES and ctx.stage != "c4_quality" and not ctx.stage.startswith("c4"):
            fn_name, _, cfg = specs.FILTER_CASES[ctx.stage]

            def reasons(m: Any, fn_name: str = fn_name, cfg: dict[str, Any] = cfg) -> Callable[[str], Any]:
                f = getattr(m, f"{fn_name}_reasons")
                return lambda t: f(t, **cfg)

            def gate(f: Batch, lb: str) -> dict[str, Any]:
                return G.gate_filter(
                    ctx.stage, lambda ts: [("ok", r is None, r, None) for r in f(ts)], ctx.ids, ctx.texts, lb
                )

            rows.append(make(reasons, gate, f"{fn_name}_reasons"))
        elif ctx.stage == "lid":
            for model in specs.LID_MODELS:

                def pred(m: Any, model: str = model) -> Callable[[str], Any]:
                    mdl = m.load_model(G.lid_model_path(model))
                    return lambda t: mdl.predict(t.replace("\n", " "), k=1, threshold=0.0)

                def gate(f: Batch, lb: str, model: str = model) -> dict[str, Any]:
                    return G.gate_lid(f, model, 1, ctx.ids, ctx.texts, lb)

                rows.append(make(pred, gate, f"predict {model} k=1"))
    return rows


def stage_rows(ctx: Ctx) -> list[Row]:
    if ctx.stage == "translit":
        rows = rows_translit(ctx)
    elif ctx.stage == "fix_text":
        rows = rows_fix_text(ctx)
    elif ctx.stage == "lid":
        rows = rows_lid(ctx)
    elif ctx.stage == "minhash":
        rows = rows_minhash(ctx)
    else:
        rows = rows_filter(ctx)
    return rows + direct_rows(ctx) + per_doc_rows(ctx)


def select(rows: list[Row], pattern: str | None) -> list[Row]:
    """``--rows a,b,!c``: rows whose name contains a or b (all rows if no positive key) and not c."""
    if not pattern:
        return rows
    keys = [p.strip().lower() for p in pattern.split(",") if p.strip()]
    want = [k for k in keys if not k.startswith("!")]
    drop = [k[1:] for k in keys if k.startswith("!")]
    return [
        r
        for r in rows
        if (not want or any(k in r.name.lower() for k in want)) and not any(k in r.name.lower() for k in drop)
    ]


def gil_guard(where: str) -> None:
    if sysconfig.get_config_var("Py_GIL_DISABLED") and sys._is_gil_enabled():
        sys.exit(
            f"{where}: free-threaded interpreter but the GIL is enabled (an extension re-enabled it, e.g. "
            "rensa 0.5.0): thread numbers would be invalid. Drop that row with --rows or set PYTHON_GIL=0."
        )


def gil_enabled() -> int:
    return int(getattr(sys, "_is_gil_enabled", lambda: True)())


# ----------------------------------------------------------------------------------------------- main


def forward(cmd: list[str], args: Any) -> None:
    cmd += [
        "--stage",
        args.stage,
        "--docs",
        str(args.docs),
        "--corpus",
        args.corpus,
        "--offset",
        str(args.offset),
    ]
    for spec in args.impl:
        cmd += ["--impl", spec]
    if args.rows:
        cmd += ["--rows", args.rows]
    if args.gate_file:
        cmd += ["--gate-file", args.gate_file]


def list_rows() -> None:
    ctx = Ctx("", [], [], 1, [])
    for stage in STAGES:
        ctx.stage = stage
        print(stage)
        for r in stage_rows(ctx):
            print(f"    {r.name:<60} {r.boundary}")


def compact(d: dict[str, Any]) -> str:
    return json.dumps(d, separators=(",", ":"), sort_keys=True, default=str)


def main() -> None:
    multi = max(THREADS, *(int(os.environ.get(k) or 1) for k in ("RAYON_NUM_THREADS", "PYROID_NUM_THREADS")))
    if multi > 1 and not any(a.startswith("--affinity") for a in sys.argv):
        # pyperf pins workers to the kernel's isolated CPUs when there are any (the OrbStack VM isolates
        # CPU 11), which would run every thread of a multi-thread row on one core. Use the e2e CPU set.
        sys.argv.append(f"--affinity={os.environ.get('BENCH_CPUS', '0-7')}")
    runner = pyperf.Runner(add_cmdline_args=forward)
    p = runner.argparser
    p.add_argument("--stage", choices=STAGES)
    p.add_argument("--docs", type=int, default=100, help="docs per batch (one benchmark value = one batch)")
    p.add_argument("--corpus", default="dev", choices=sorted(G.SEGMENTS))
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--impl", action="append", default=[], metavar="SUITE=MODULE", help="add a candidate row")
    p.add_argument("--rows", help="comma list of substrings: only rows whose name contains one")
    p.add_argument("--gate-workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument(
        "--reuse-gates", metavar="FILE", help="gate results from an earlier identical run (memory pass)"
    )
    p.add_argument("--gate-file", help=argparse.SUPPRESS)
    p.add_argument("--list", action="store_true", help="list stages and rows, then exit")
    args = runner.parse_args()
    if args.list:
        list_rows()
        return
    if not args.stage:
        sys.exit("--stage is required (see --list)")
    keep = set(args.inherit_environ or []) | {"RAYON_NUM_THREADS"}
    keep |= {k for k in os.environ if k.startswith(("BENCH_", "PYROID_", "GLIBC_TUNABLES", "MIMALLOC_"))}
    args.inherit_environ = sorted(keep)

    ids, texts = G.load_slice(args.corpus, args.offset, args.docs, verify=not args.worker)
    ctx = Ctx(args.stage, ids, texts, args.gate_workers, args.impl)
    rows = select(stage_rows(ctx), args.rows)
    if not rows:
        sys.exit(f"--rows {args.rows!r} matches no row of stage {args.stage}")
    names = [r.name for r in rows]
    if len(set(names)) != len(names):
        sys.exit(f"duplicate row names: {names}")

    if not args.worker:
        gate_results = run_gates(args, rows)
    else:
        gate_results = json.loads(Path(args.gate_file).read_text("utf-8")) if args.gate_file else {}

    base = {
        "stage": args.stage,
        "suite": SUITE_OF[args.stage],
        "docs": args.docs,
        "corpus": args.corpus,
        "offset": args.offset,
        "chars": sum(map(len, texts)),
        "input_sha256": D.content_sha256(ids, texts),
        "threads": THREADS,
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS", "unset"),
        "value_unit": "seconds per call over a batch of `docs` docs",
        "gil_enabled": gil_enabled(),
    }
    for task, row in enumerate(rows):
        timed: Batch | None = None
        if args.worker and args.worker_task in (None, task):  # None: a bare --worker runs every row
            timed, _ = row.build()
            gil_guard(row.name)
        md = {
            **base,
            "impl": row.impl,
            "impl_name": row.name,
            "version": row.version,
            "is_reference": int(row.reference),
            "boundary": row.boundary,
            "tokenization": row.tokenization,
            "gate": gate_results.get(row.name, "missing"),
            "input_kind": row.input_kind,
            "fresh_strs": int(FRESH),
        }
        runner.bench_time_func(
            f"{args.stage}/{row.name}/docs={args.docs}/t={THREADS}",
            timer(timed or _unbuilt, row.prep),
            texts,
            metadata=md,
        )


def fresh_copies(texts: list[str]) -> list[str]:
    """New str objects with the same text and no cached UTF-8 (``bytes.decode`` never fills that cache)."""
    return [t.encode("utf-8", "surrogatepass").decode("utf-8", "surrogatepass") for t in texts]


def timer(fn: Batch, prep: Callable[[list[str]], Any] | None) -> Callable[[int, list[str]], float]:
    """pyperf time function: ``loops`` timed calls of ``fn``. Building the input (``prep``, fresh strs) is
    never timed; without BENCH_FRESH_STRS it is done once and reused."""
    cache: list[Any] = []

    def run(loops: int, texts: list[str]) -> float:
        total = 0.0
        for _ in range(loops):
            if FRESH:
                docs = fresh_copies(texts)
                arg = prep(docs) if prep else docs
            else:
                if not cache:
                    cache.append(prep(texts) if prep else texts)
                arg = cache[0]
            t0 = time.perf_counter()
            fn(arg)
            total += time.perf_counter() - t0
        return total

    return run


def _unbuilt(_: Sequence[str]) -> None:
    raise RuntimeError("row callable was not built in this process")


def run_gates(args: Any, rows: list[Row]) -> dict[str, str]:
    """Manager process: gate every row on the timed docs (or reuse a gate file), then hand the results to
    the workers through ``--gate-file``."""
    if args.reuse_gates:
        results = json.loads(Path(args.reuse_gates).read_text("utf-8"))
        missing = [r.name for r in rows if r.name not in results]
        if missing:
            sys.exit(f"--reuse-gates {args.reuse_gates}: no gate for {missing}")
    else:
        results = {}
        for row in rows:
            _, gate = row.build()
            gil_guard(row.name)
            try:
                res = gate()
            except G.GateError as e:
                sys.exit(f"GATE FAILED, not timing {args.stage}: {e}")
            results[row.name] = compact(res)
            print(
                f"gate ok: {row.name}: {res.get('docs', res.get('pairs'))} docs/pairs, {res['seconds']} s",
                file=sys.stderr,
            )
    if args.output:
        path = Path(args.output).with_suffix(".gates.json")
    else:
        fd, name = tempfile.mkstemp(prefix="pyroid-gates-", suffix=".json")
        os.close(fd)
        path = Path(name)
        atexit.register(path.unlink, missing_ok=True)
    path.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", "utf-8")
    args.gate_file = str(path)
    return results


if __name__ == "__main__":
    main()
