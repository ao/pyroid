#!/usr/bin/env python3
"""Equivalence gates (R2 section 5, rule 1): nothing is timed until its output matches the pinned references.

    DX_NAME=harness .orchestration/dx.sh 'cd bench && UV_PROJECT_ENVIRONMENT=/target/venv-bench \
        uv run --frozen python gates.py --workers 8 --out /target/gates.json'          # all suites, dev+adversarial
    ... python gates.py --suite ftfy --impl ftfy=pyroid.ftfy --limit 500                 # + a candidate, 500+500 docs

Gates (each raises ``GateError`` with a clear message, or returns a small dict stored in results metadata):

ftfy      output == pinned ``fixed`` (parity/ftfy-ref.parquet), byte for byte.
lid       same top-k labels and ``|p - p_ref| <= 1e-6`` vs parity/fasttext-ref.parquet (text.replace("\\n", " ")).
          The datatrove ``LanguageFilter`` row is gated on (language, language_score, keep) derived from the
          same reference: keep iff P(en) > 0.65, P(en) = 0 when "en" is outside the pinned top-5.
filters   each case: same (outcome, keep, reason, text_out) as parity/filters-ref.parquet.
minhash   datatrove-equivalent signatures == parity/minhash-ref.parquet ``signature``. datasketch / rensa use
          other hash families, so they are gated on near-duplicate detection instead: on the adversarial
          ``neardup:*`` docs paired with their dev source, ground truth = exact Jaccard of R2-template word
          5-gram shingles >= 0.8, prediction = estimated Jaccard >= 0.8; precision, recall and mean |J_est - J|
          must stay within ``ESTIMATOR_BOUNDS``.

Docs without a pinned reference (headline docs past the first 10k) are compared against the live oracle
(same pinned library) instead; each gate dict reports how many docs were ``pinned`` vs ``live``.

This module also hosts the reference constructors shared by stages.py and pipeline.py (the gated objects
are the timed objects): pinned-corpus slicing, the R2 shingler, and the datatrove LanguageFilter shim for
fasttext-predict (see ``PinnedLanguageFilter``).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from functools import cache
from pathlib import Path
from typing import Any, Self

import numpy as np
from datatrove.data import Document
from datatrove.pipeline.filters import LanguageFilter
from datatrove.utils._import_utils import check_required_dependencies
from datatrove.utils.lid import FT176LID
from datatrove.utils.text import split_into_parts
from pyroid_parity import data as D
from pyroid_parity import impl as I
from pyroid_parity import refs as R
from pyroid_parity import specs

LID_TOL = 1e-6
NEAR_DUP_J = 0.8
NUM_PERM = specs.MINHASH_CONFIG["num_buckets"] * specs.MINHASH_CONFIG["hashes_per_bucket"]  # 112
# Stated bound for estimators with other hash families (datasketch, rensa), on the 600 adversarial near-dup
# pairs. With 112 permutations the estimate's sd at J = 0.8 is sqrt(0.8 * 0.2 / 112) = 0.038, so pairs near
# the cut flip either way. datasketch 2.0.0 / rensa 0.5.0 / datatrove's own signature all measure
# P, R ~0.95-0.97 and mean error ~0.02 here (deterministic seeds); the bound leaves room for another hash
# family's noise but fails a broken shingler or hasher.
ESTIMATOR_BOUNDS = {"min_precision": 0.90, "min_recall": 0.90, "max_mean_abs_err": 0.04}
LID_FILTER = {
    "languages": ["en"],
    "language_threshold": 0.65,
    "backend": "ft176",
}  # datatrove examples/fineweb.py

# Logical corpora -> ordered pinned files. "parity" = dev + adversarial, the exact doc set the refs cover.
SEGMENTS = {
    "dev": [D.CORPORA["dev"]],
    "headline": [D.CORPORA["headline"]],
    "adversarial": [D.CORPORA["adversarial"]],
    "parity": [D.CORPORA["dev"], D.CORPORA["adversarial"]],
}


class GateError(AssertionError):
    """An implementation's output differs from the pinned reference: its timings must not be reported."""


# ------------------------------------------------------------------------------------------- pinned slices


def segment_rows(corpus: str, verify: bool = True) -> list[tuple[Path, int]]:
    """``[(path, num_rows), ...]`` for a logical corpus; ``verify`` checks each file's pinned sha256."""
    import pyarrow.parquet as pq

    return [
        (p, pq.ParquetFile(p).metadata.num_rows) for p in (D.path(rel, verify) for rel in SEGMENTS[corpus])
    ]


def corpus_len(corpus: str, verify: bool = True) -> int:
    return sum(n for _, n in segment_rows(corpus, verify))


def iter_slice(
    segments: Sequence[tuple[Path | str, int]], start: int, stop: int, columns: Sequence[str] = ("id", "text")
) -> Iterator[tuple[Any, ...]]:
    """Rows ``[start, stop)`` of the concatenated segments, reading only the row groups that overlap."""
    import pyarrow.parquet as pq

    base = 0
    for path, n in segments:
        lo, hi = max(start, base), min(stop, base + n)
        if lo < hi:
            pf = pq.ParquetFile(path)
            first = 0
            for g in range(pf.num_row_groups):
                rows = pf.metadata.row_group(g).num_rows
                glo, ghi = base + first, base + first + rows
                if glo < hi and lo < ghi:
                    t = pf.read_row_group(g, columns=list(columns)).slice(
                        max(lo, glo) - glo, min(hi, ghi) - max(lo, glo)
                    )
                    yield from zip(*(t.column(c).to_pylist() for c in columns), strict=True)
                first += rows
        base += n


def load_slice(corpus: str, offset: int, n: int, verify: bool = True) -> tuple[list[str], list[str]]:
    segs = segment_rows(corpus, verify)
    total = sum(k for _, k in segs)
    if offset < 0 or n < 1 or offset + n > total:
        raise SystemExit(f"--offset {offset} --docs {n}: corpus {corpus!r} has {total} docs")
    ids, texts = [], []
    for i, t in iter_slice(segs, offset, offset + n):
        ids.append(i)
        texts.append(t)
    return ids, texts


def sample(n_total: int, limit: int) -> list[int]:
    """Evenly spaced indices (covers every adversarial kind, which are stored in blocks)."""
    if not limit or limit >= n_total:
        return list(range(n_total))
    step = n_total / limit
    return [int(i * step) for i in range(limit)]


# ------------------------------------------------------------------------------------ shared constructors


def r2_shingles(text: str, n: int = 5) -> set[str]:
    """R2 template shingler (bench-hygiene.md section 5): lower-cased whitespace words, word n-grams."""
    w = text.lower().split()
    return {" ".join(w[i : i + n]) for i in range(max(1, len(w) - n + 1))}


def lid_model_path(name: str) -> str:
    return str(D.path(specs.LID_MODELS[name]))


class NumpyProbsModel:
    """Make fasttext-predict's ``predict`` return probabilities as a numpy array, like official fastText.

    fasttext-predict 0.9.2.4 returns ``(labels, probs)`` with ``probs`` a tuple of Python floats, while
    datatrove's ``FastTextLID.predict`` calls ``score.item()`` (written for fasttext / fasttext-numpy2-wheel,
    whose ``predict`` returns ``np.array(probs)``). The values are identical (fastText's float32 widened to
    float64 either way); only the container type changes.
    """

    __slots__ = ("model",)

    def __init__(self, model: Any) -> None:
        self.model = model

    def predict(self, text: str, k: int = 1, threshold: float = 0.0, on_unicode_error: str = "strict") -> Any:
        labels, probs = self.model.predict(text, k, threshold, on_unicode_error)
        return labels, np.asarray(probs)


class PinnedFT176LID(FT176LID):
    """datatrove ``FT176LID`` that loads the sha256-pinned ``/data/models/lid.176.bin`` (lazily, in the worker).

    Upstream downloads the model via ``cached_asset_path_or_download`` (needs network + ``fasteners``).
    ``predict`` is inherited unchanged.
    """

    def __init__(self, languages: list[str] | None, model_path: str, k: int = -1) -> None:
        super().__init__(languages, k)
        self.model_path = model_path

    @property
    def model(self) -> Any:
        if self._model is None:
            from fasttext.FastText import _FastText

            self._model = NumpyProbsModel(_FastText(self.model_path))
        return self._model


class PinnedLanguageFilter(LanguageFilter):
    """datatrove 0.10.1 ``LanguageFilter`` with its ``filter()`` and ``FastTextLID.predict`` unchanged.

    Shims needed to run it on fasttext-predict (the fastText build that has cp313/cp313t wheels):
    1. ``PipelineStep.__new__`` requires the ``fasttext-numpy2-wheel`` distribution and ``fasteners``; we
       check for the ``fasttext`` module from ``fasttext-predict`` instead (``fasteners`` is only used by the
       model download, which we skip).
    2. the model comes from the pinned file (``PinnedFT176LID``) instead of a URL download.
    3. ``NumpyProbsModel`` restores numpy probabilities for ``score.item()``.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        check_required_dependencies(cls.__name__, [("fasttext", "fasttext-predict")])
        return object.__new__(cls)

    def __init__(self, *, model_path: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.backend != "ft176":
            raise ValueError("only backend='ft176' is pinned")
        self.model = PinnedFT176LID(self.languages, model_path)


def language_filter(**kw: Any) -> PinnedLanguageFilter:
    return PinnedLanguageFilter(model_path=lid_model_path("lid.176.bin"), **{**LID_FILTER, **kw})


def datatrove_filter(case: str) -> Any:
    """A fresh datatrove filter instance for ``specs.FILTER_CASES[case]``."""
    from datatrove.pipeline import filters as F

    _, cls, cfg = specs.FILTER_CASES[case]
    return getattr(F, cls)(**cfg)


def dt_decide(filt: Any, text: str, doc_id: str = "") -> tuple[Any, ...]:
    """One datatrove filter decision in the parity format; clears the 5-slot tokenization cache first."""
    from datatrove.pipeline.filters.base_filter import get_filter_result

    split_into_parts.cache_clear()
    doc = Document(text=text, id=doc_id)
    try:
        keep, reason = get_filter_result(filt.filter(doc))
    except Exception as e:  # noqa: BLE001  exception type is part of parity
        return ("raise", type(e).__name__)
    return ("ok", bool(keep), reason, doc.text if doc.text != text else None)


def minhash_step() -> Any:
    import tempfile

    from datatrove.pipeline.dedup import MinhashDedupSignature

    return MinhashDedupSignature(
        output_folder=tempfile.mkdtemp(prefix="pyroid-bench-sig-"), config=R.minhash_config()
    )


def dt_signature(step: Any, text: str) -> list[int] | None:
    sh = step.get_shingles(text)
    return None if sh.size == 0 else [int(x) for b in step.get_signature(sh) for x in b]


# ------------------------------------------------------------------------------------------ pinned refs


@cache
def ftfy_ref() -> dict[str, str]:
    c = D.read_parquet_columns(D.path(R.OUT["ftfy"]), ["id", "fixed"])
    return dict(zip(c["id"], c["fixed"], strict=True))


@cache
def lid_ref() -> dict[tuple[str, str], tuple[list[str], list[float]]]:
    c = D.read_parquet_columns(D.path(R.OUT["fasttext"]), ["model", "id", "labels", "probs"])
    return {(m, i): (lb, p) for m, i, lb, p in zip(c["model"], c["id"], c["labels"], c["probs"], strict=True)}


@cache
def filters_ref() -> dict[tuple[str, str], tuple[Any, ...]]:
    c = D.read_parquet_columns(
        D.path(R.OUT["filters"]), ["case", "id", "outcome", "keep", "reason", "text_out"]
    )
    out: dict[tuple[str, str], tuple[Any, ...]] = {}
    for case, i, o, k, r, t in zip(
        c["case"], c["id"], c["outcome"], c["keep"], c["reason"], c["text_out"], strict=True
    ):
        out[(case, i)] = ("ok", bool(k), r, t) if o == "ok" else ("raise", r)
    return out


@cache
def minhash_ref() -> tuple[dict[str, list[int] | None], dict[str, bool]]:
    c = D.read_parquet_columns(D.path(R.OUT["dedup"]), ["id", "signature", "removed"])
    return dict(zip(c["id"], c["signature"], strict=True)), dict(zip(c["id"], c["removed"], strict=True))


@cache
def near_dup_pairs() -> list[tuple[str, str, str, str]]:
    """``(adv_id, adv_text, src_id, src_text)`` for every adversarial ``neardup:*`` doc."""
    adv = D.read_parquet_columns(D.path(D.CORPORA["adversarial"]), ["id", "text", "kind", "source_id"])
    dev = D.corpus("dev")
    src = dict(zip(dev.ids, dev.texts, strict=True))
    return [
        (i, t, s, src[s])
        for i, t, k, s in zip(adv["id"], adv["text"], adv["kind"], adv["source_id"], strict=True)
        if k.startswith("neardup:")
    ]


# --------------------------------------------------------------------------------------------- helpers

_FORK: dict[int, Callable[[list[Any]], list[Any]]] = {}


def _fork_call(arg: tuple[int, list[Any]]) -> list[Any]:
    key, chunk = arg
    return _FORK[key](chunk)


def fork_map(
    fn: Callable[[list[Any]], list[Any]], items: list[Any], workers: int, chunk: int = 100
) -> list[Any]:
    """Order-preserving chunked map over forked processes (references only: never fork after a Rust/rayon
    candidate initialised its thread pool)."""
    if workers <= 1 or len(items) <= chunk:
        return fn(items)
    key = len(_FORK)
    _FORK[key] = fn
    try:
        chunks = [(key, items[i : i + chunk]) for i in range(0, len(items), chunk)]
        with mp.get_context("fork").Pool(workers) as pool:
            return [x for part in pool.map(_fork_call, chunks) for x in part]
    finally:
        del _FORK[key]


def _expected(
    ids: list[str],
    texts: list[str],
    pinned: Callable[[str], Any],
    live: Callable[[list[str]], list[Any]],
    workers: int,
) -> tuple[list[Any], int]:
    """Pinned reference per doc, falling back to the live oracle for docs the refs do not cover."""
    want: list[Any] = []
    missing: list[int] = []
    for n, i in enumerate(ids):
        try:
            want.append(pinned(i))
        except KeyError:
            want.append(None)
            missing.append(n)
    if missing:
        for n, v in zip(missing, fork_map(live, [texts[n] for n in missing], workers), strict=True):
            want[n] = v
    return want, len(ids) - len(missing)


def _report(
    gate: str,
    label: str,
    ids: list[str],
    diffs: list[tuple[str, Any, Any]],
    pinned: int,
    t0: float,
    **extra: Any,
) -> dict[str, Any]:
    res = {
        "gate": gate,
        "impl": label,
        "docs": len(ids),
        "pinned": pinned,
        "live": len(ids) - pinned,
        "mismatches": len(diffs),
        "seconds": round(time.perf_counter() - t0, 3),
        **extra,
    }
    if diffs:
        shown = "; ".join(f"{i}: got {_short(g)} want {_short(w)}" for i, g, w in diffs[:3])
        raise GateError(f"[{gate}] {label}: {len(diffs)}/{len(ids)} docs differ from the reference ({shown})")
    return res


def _short(x: Any, n: int = 120) -> str:
    s = repr(x)
    return s if len(s) <= n else s[:n] + "..."


# ----------------------------------------------------------------------------------------------- gates


def _ftfy_live(texts: list[str]) -> list[str]:
    import ftfy

    return [ftfy.fix_text(t) for t in texts]


def gate_ftfy(
    fix_batch: Callable[[list[str]], list[str]],
    ids: list[str],
    texts: list[str],
    label: str,
    workers: int = 1,
    parallel: bool = False,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    got = fork_map(fix_batch, texts, workers) if parallel else list(fix_batch(texts))
    want, pinned = _expected(ids, texts, ftfy_ref().__getitem__, _ftfy_live, workers)
    diffs = [(i, g, w) for i, g, w in zip(ids, got, want, strict=True) if g != w]
    return _report("ftfy", label, ids, diffs, pinned, t0)


_LIVE_FT: dict[str, Any] = {}


def _ft_live(model: str, k: int) -> Callable[[list[str]], list[tuple[list[str], list[float]]]]:
    def run(texts: list[str]) -> list[tuple[list[str], list[float]]]:
        import fasttext

        if model not in _LIVE_FT:
            _LIVE_FT[model] = fasttext.load_model(lid_model_path(model))
        out = []
        for t in texts:
            lb, p = _LIVE_FT[model].predict(t.replace("\n", " "), k=k, threshold=0.0)
            out.append((list(lb), [float(x) for x in p]))
        return out

    return run


def gate_lid(
    predict_batch: Callable[[list[str]], list[tuple[Sequence[str], Sequence[float]]]],
    model: str,
    k: int,
    ids: list[str],
    texts: list[str],
    label: str,
    workers: int = 1,
    parallel: bool = False,
) -> dict[str, Any]:
    """``predict_batch(texts)`` gets raw texts and must apply ``replace("\\n", " ")`` itself (as datatrove does)."""
    if k > specs.LID_TOPK:
        raise ValueError(f"refs pin top-{specs.LID_TOPK} only")
    t0 = time.perf_counter()
    got = fork_map(predict_batch, texts, workers) if parallel else list(predict_batch(texts))
    want, pinned = _expected(
        ids, texts, lambda i: lid_ref()[(model, i)], _ft_live(model, specs.LID_TOPK), workers
    )
    diffs, max_dp = [], 0.0
    for i, (gl, gp), (wl, wp) in zip(ids, got, want, strict=True):
        gl, gp, wl, wp = list(gl)[:k], [float(x) for x in gp][:k], list(wl)[:k], list(wp)[:k]
        dp = max((abs(a - b) for a, b in zip(gp, wp, strict=False)), default=0.0)
        max_dp = max(max_dp, dp)
        if gl != wl or len(gp) != len(wp) or dp > LID_TOL:
            diffs.append((i, (gl, gp), (wl, wp)))
    return _report("lid", label, ids, diffs, pinned, t0, model=model, k=k, tol=LID_TOL, max_abs_dp=max_dp)


def lid_filter_expected(labels: Sequence[str], probs: Sequence[float]) -> tuple[bool, str, float]:
    """(keep, language, language_score) of LanguageFilter(en, 0.65, ft176) from a top-k prediction."""
    p_en = next((float(p) for lb, p in zip(labels, probs, strict=True) if lb == "__label__en"), 0.0)
    return p_en > LID_FILTER["language_threshold"], labels[0].removeprefix("__label__"), float(probs[0])


def lid_filter_decide(filt: Any, texts: list[str]) -> list[tuple[bool, str, float]]:
    out = []
    for t in texts:
        doc = Document(text=t, id="")
        keep = bool(filt.filter(doc))
        out.append((keep, doc.metadata["language"], float(doc.metadata["language_score"])))
    return out


def gate_lid_filter(
    decide_batch: Callable[[list[str]], list[tuple[bool, str, float]]],
    ids: list[str],
    texts: list[str],
    label: str,
    workers: int = 1,
    parallel: bool = False,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    got = fork_map(decide_batch, texts, workers) if parallel else list(decide_batch(texts))
    live = _ft_live("lid.176.bin", specs.LID_TOPK)
    want, pinned = _expected(ids, texts, lambda i: lid_ref()[("lid.176.bin", i)], live, workers)
    diffs = []
    for i, g, (wl, wp) in zip(ids, got, want, strict=True):
        w = lid_filter_expected(wl, wp)
        if g[0] != w[0] or g[1] != w[1] or abs(g[2] - w[2]) > LID_TOL:
            diffs.append((i, g, w))
    return _report("lid_filter", label, ids, diffs, pinned, t0, tol=LID_TOL, **LID_FILTER)


def _filter_live(case: str) -> Callable[[list[str]], list[tuple[Any, ...]]]:
    def run(texts: list[str]) -> list[tuple[Any, ...]]:
        f = datatrove_filter(case)
        return [dt_decide(f, t) for t in texts]

    return run


def gate_filter(
    case: str,
    decide_batch: Callable[[list[str]], list[tuple[Any, ...]]],
    ids: list[str],
    texts: list[str],
    label: str,
    workers: int = 1,
    parallel: bool = False,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    got = fork_map(decide_batch, texts, workers) if parallel else list(decide_batch(texts))
    want, pinned = _expected(ids, texts, lambda i: filters_ref()[(case, i)], _filter_live(case), workers)
    diffs = [(i, tuple(g), tuple(w)) for i, g, w in zip(ids, got, want, strict=True) if tuple(g) != tuple(w)]
    kept = sum(1 for w in want if w[0] == "ok" and w[1])
    return _report("filter", label, ids, diffs, pinned, t0, case=case, ref_kept=kept)


def _sig_live(texts: list[str]) -> list[list[int] | None]:
    step = minhash_step()
    return [dt_signature(step, t) for t in texts]


def gate_minhash_signatures(
    sig_batch: Callable[[list[str]], list[Any]],
    ids: list[str],
    texts: list[str],
    label: str,
    workers: int = 1,
    parallel: bool = False,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    got = fork_map(sig_batch, texts, workers) if parallel else list(sig_batch(texts))
    sigs, _ = minhash_ref()
    want, pinned = _expected(ids, texts, sigs.__getitem__, _sig_live, workers)
    norm = [None if g is None else [int(x) for x in g] for g in got]
    diffs = [(i, g, w) for i, g, w in zip(ids, norm, want, strict=True) if g != w]
    return _report("minhash_signature", label, ids, diffs, pinned, t0, num_perm=NUM_PERM)


def gate_minhash_dedup(dup_fn: Callable[[list[str]], list[bool]], label: str) -> dict[str, Any]:
    """Candidate dedup over the full parity set (dev + adversarial) == datatrove's removed mask."""
    t0 = time.perf_counter()
    docs = R.all_docs()
    got = [bool(x) for x in dup_fn(docs.texts)]
    _, removed = minhash_ref()
    diffs = [(i, g, removed[i]) for i, g in zip(docs.ids, got, strict=True) if g != removed[i]]
    return _report("minhash_dedup", label, docs.ids, diffs, len(docs.ids), t0, removed=sum(got))


def estimator_quality(
    sketch_batch: Callable[[list[str]], list[Any]],
    jaccard: Callable[[Any, Any], float],
    shingles: Callable[[str], set[str]] = r2_shingles,
) -> dict[str, Any]:
    nd = near_dup_pairs()
    a = sketch_batch([p[1] for p in nd])
    b = sketch_batch([p[3] for p in nd])
    tp = fp = fn = 0
    err = 0.0
    for (_, ta, _, tb), sa, sb in zip(nd, a, b, strict=True):
        x, y = shingles(ta), shingles(tb)
        exact = len(x & y) / len(x | y) if x | y else 1.0
        est = float(jaccard(sa, sb))
        err += abs(est - exact)
        truth, pred = exact >= NEAR_DUP_J, est >= NEAR_DUP_J
        tp += truth and pred
        fp += pred and not truth
        fn += truth and not pred
    return {
        "pairs": len(nd),
        "positives": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(tp / (tp + fp), 4) if tp + fp else 1.0,
        "recall": round(tp / (tp + fn), 4) if tp + fn else 1.0,
        "mean_abs_err": round(err / len(nd), 4),
        "threshold": NEAR_DUP_J,
        "shingles": "r2: lower().split() 5-grams",
    }


def gate_estimator(
    sketch_batch: Callable[[list[str]], list[Any]], jaccard: Callable[[Any, Any], float], label: str
) -> dict[str, Any]:
    t0 = time.perf_counter()
    q = estimator_quality(sketch_batch, jaccard)
    b = ESTIMATOR_BOUNDS
    ok = (
        q["precision"] >= b["min_precision"]
        and q["recall"] >= b["min_recall"]
        and q["mean_abs_err"] <= b["max_mean_abs_err"]
    )
    res = {
        "gate": "minhash_estimator",
        "impl": label,
        **q,
        "bounds": b,
        "seconds": round(time.perf_counter() - t0, 3),
    }
    if not ok:
        raise GateError(
            f"[minhash_estimator] {label}: precision {q['precision']}, recall {q['recall']}, "
            f"mean |J_est - J| {q['mean_abs_err']} outside {b}"
        )
    return res


def datatrove_estimator_quality() -> dict[str, Any]:
    """Informational: datatrove's own signature as a Jaccard estimator (its shingles differ from the R2 ones)."""
    step = minhash_step()

    def sketch(texts: list[str]) -> list[Any]:
        return [dt_signature(step, t) for t in texts]

    def jac(x: list[int] | None, y: list[int] | None) -> float:
        if x is None or y is None:
            return float(x is None and y is None)
        return sum(p == q for p, q in zip(x, y, strict=True)) / len(x)

    return estimator_quality(sketch, jac)


# ---------------------------------------------------------------------------------------- reference rows


def datasketch_bulk(texts: list[str]) -> list[Any]:
    from datasketch import MinHash

    return MinHash.bulk(
        [[s.encode("utf-8") for s in r2_shingles(t)] for t in texts], num_perm=NUM_PERM, seed=1
    )


def rensa_batch(texts: list[str]) -> list[Any]:
    from rensa import RMinHash

    return RMinHash.from_token_sets([list(r2_shingles(t)) for t in texts], NUM_PERM, 42)


def explicit_suites(spec: str) -> dict[str, str]:
    """Suites an ``--impl`` spec names explicitly: ``pyroid.ftfy`` -> ftfy, ``datatrove`` -> filters + dedup."""
    suite, sep, module = spec.partition("=")
    if sep:
        I.parse_specs([spec])  # validates the suite name
        return {suite: module}
    return {s: m for s, m in I.parse_specs([spec]).items() if m == spec}


def pairs(pred: Sequence[tuple[str, float]]) -> tuple[list[str], list[float]]:
    return [lb for lb, _ in pred], [float(p) for _, p in pred]


def run_reference_gates(
    ids: list[str], texts: list[str], suites: set[str], workers: int
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if "ftfy" in suites:
        out.append(gate_ftfy(_ftfy_live, ids, texts, f"ftfy {I.version_of('ftfy')}", workers, parallel=True))
    if "lid" in suites:
        out += [
            gate_lid(
                _ft_live(m, specs.LID_TOPK),
                m,
                specs.LID_TOPK,
                ids,
                texts,
                f"fasttext-predict {I.version_of('fasttext')} {m}",
                workers,
                parallel=True,
            )
            for m in specs.LID_MODELS
        ]
        lf = language_filter()
        out.append(
            gate_lid_filter(
                lambda ts: lid_filter_decide(lf, ts),
                ids,
                texts,
                f"datatrove {I.version_of('datatrove')} LanguageFilter",
                workers,
                parallel=True,
            )
        )
    if "filters" in suites:
        out += [
            gate_filter(case, _filter_live(case), ids, texts, f"datatrove {case}", workers, parallel=True)
            for case in specs.FILTER_CASES
        ]
    if "minhash" in suites:
        out.append(
            gate_minhash_signatures(_sig_live, ids, texts, "datatrove MinhashDedupSignature", workers, True)
        )
        out.append(
            gate_estimator(
                datasketch_bulk,
                lambda a, b: a.jaccard(b),
                f"datasketch {I.version_of('datasketch')} MinHash.bulk",
            )
        )
        out.append(
            gate_estimator(
                rensa_batch,
                lambda a, b: a.jaccard(b),
                f"rensa {I.version_of('rensa')} RMinHash.from_token_sets",
            )
        )
        out.append(
            {
                "gate": "minhash_estimator_info",
                "impl": "datatrove signature (own shingles, not gated)",
                **datatrove_estimator_quality(),
            }
        )
    return out


def run_candidate_gates(
    spec: str, ids: list[str], texts: list[str], suites: set[str], full_parity: bool
) -> list[dict[str, Any]]:
    """Candidates run in-process (no fork: a Rust thread pool does not survive fork)."""
    out: list[dict[str, Any]] = []
    gate_of = {"ftfy": "ftfy", "fasttext": "lid", "filters": "filters", "dedup": "minhash"}
    for suite, module in explicit_suites(spec).items():
        if gate_of[suite] in suites:
            impl = I.resolve(suite, module)
            out += candidate_suite_gates(
                suite, impl.obj, f"{module} {impl.version or ''}".strip(), ids, texts, full_parity
            )
    return out


def candidate_suite_gates(
    suite: str, o: Any, label: str, ids: list[str], texts: list[str], full_parity: bool = False
) -> list[dict[str, Any]]:
    """Gates for one resolved candidate adapter ``o`` (the ``impl.resolve(...).obj`` contract objects)."""
    if suite == "ftfy":
        return [gate_ftfy(o.fix_text_batch, ids, texts, label)]
    if suite == "fasttext":
        res = []
        for m in specs.LID_MODELS:
            model = o.load(lid_model_path(m))

            def pred(ts: list[str], model: Any = model) -> list[Any]:
                return [pairs(o.predict(model, t.replace("\n", " "), specs.LID_TOPK)) for t in ts]

            res.append(gate_lid(pred, m, specs.LID_TOPK, ids, texts, f"{label} {m}"))
        return res
    if suite == "filters":
        return [
            gate_filter(case, lambda ts, fn=fn, cfg=cfg: o.run(fn, ts, cfg), ids, texts, f"{label} {case}")
            for case, (fn, _, cfg) in specs.FILTER_CASES.items()
        ]
    if suite == "dedup":
        res = [gate_minhash_signatures(lambda ts: o.signatures(ts, specs.MINHASH_CONFIG), ids, texts, label)]
        if full_parity:
            res.append(gate_minhash_dedup(lambda ts: o.duplicates(ts, specs.MINHASH_CONFIG), label))
        return res
    raise ValueError(suite)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="all", help="comma list of ftfy,lid,filters,minhash (default all)")
    ap.add_argument("--corpus", default="parity", choices=sorted(SEGMENTS))
    ap.add_argument(
        "--limit", type=int, default=0, help="evenly spaced sample of N docs per corpus file (0 = all)"
    )
    ap.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1), help="processes for references"
    )
    ap.add_argument(
        "--impl", action="append", default=[], metavar="SUITE=MODULE", help="candidate (repeatable)"
    )
    ap.add_argument("--no-reference", action="store_true", help="only gate the --impl candidates")
    ap.add_argument("--out", help="write the gate dicts as JSON here")
    a = ap.parse_args()
    suites = {"ftfy", "lid", "filters", "minhash"} if a.suite == "all" else set(a.suite.split(","))
    ids: list[str] = []
    texts: list[str] = []
    for path, n in segment_rows(a.corpus):
        part = list(iter_slice([(path, n)], 0, n))
        for k in sample(n, a.limit):
            ids.append(part[k][0])
            texts.append(part[k][1])
    full_parity = a.corpus == "parity" and not a.limit
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    failed: list[str] = []
    jobs: list[Callable[[], list[dict[str, Any]]]] = []
    if not a.no_reference:
        jobs.append(lambda: run_reference_gates(ids, texts, suites, a.workers))
    jobs += [lambda s=s: run_candidate_gates(s, ids, texts, suites, full_parity) for s in a.impl]
    for job in jobs:
        try:
            results += job()
        except GateError as e:
            failed.append(str(e))
            print(f"GATE FAILED: {e}", file=sys.stderr)
    for r in results:
        brief = {
            k: r[k] for k in ("docs", "pinned", "mismatches", "precision", "recall", "mean_abs_err") if k in r
        }
        print(f"ok  {r['gate']:<22} {r['impl']:<52} {brief}")
    doc = {
        "corpus": a.corpus,
        "limit": a.limit,
        "docs": len(ids),
        "seconds": round(time.perf_counter() - t0, 1),
        "ok": not failed,
        "failed": failed,
        "gates": results,
    }
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", "utf-8")
    print(f"{len(results)} gates passed, {len(failed)} failed, {doc['seconds']} s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
