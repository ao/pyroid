"""Reference values for pyroid's port of datasketch 2.0.0's (b, r) searches.

Runs datasketch 2.0.0 (with scipy 1.18.1 and numpy) and writes:

- ``<committed.json>``: the compact reference read by
  crates/pyroid-core/tests/datasketch_params.rs (a sample of probabilities, scipy
  ``_qagse`` outputs, every argmin of the LSH grid, MinHashLSHEnsemble params).
- ``<full_dir>/full_lsh_probs.json``, ``full_ens_probs.json``: every probability the
  grid evaluates (not committed); read by the ignored full-grid test through
  ``DATASKETCH_PARAMS_FULL=<full_dir>``.
- ``<full_dir>/warnings.json``: IntegrationWarning counts per call.

Usage (Docker only): python -I gen_reference.py <committed.json> <full_dir> [--workers N]
Values are written with float.hex.
"""

import argparse
import json
import math
import os
import platform
import random
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import datasketch
import numpy as np
import scipy
from datasketch import MinHashLSHEnsemble, lsh, lshensemble
from scipy.integrate import IntegrationWarning, _quadpack

NUM_PERMS = [2, 3, 4, 8, 16, 32, 64, 100, 128, 200, 256, 512]
WEIGHTS = [(0.5, 0.5), (0.2, 0.8), (0.8, 0.2), (0.4, 0.6), (0.0, 1.0), (1.0, 0.0), (0.3, 0.7)]
THRESHOLDS = (
    [i / 100 for i in range(101)]
    + [1e-9, 0.001, 0.333, 0.999]
    # Accepted by datasketch's range check too.
    + [float("nan"), -0.0, 5e-324, 1 - 2**-53]
)
MAX_PERM = max(NUM_PERMS)
LSH_PAIRS = [(b, r) for b in range(1, MAX_PERM + 1) for r in range(1, MAX_PERM // b + 1)]

ENS_MAIN = [  # what typical MinHashLSHEnsemble constructions compute
    (t, n, part, 8, (0.5, 0.5))
    for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    for n in [128, 256]
    for part in [16, 32]
]
ENS_EXTRA = [
    (t, n, 16, m, w)
    for t in [0.0, 0.1, 0.3, 0.95, 1.0]
    for n in [16, 64, 128]
    for m in [2, 4, 16]
    for w in [(0.5, 0.5), (0.2, 0.8)]
    if m <= n
]
ENS_THRESHOLDS = [*sorted({c[0] for c in ENS_MAIN + ENS_EXTRA}), float("nan")]
ENS_PERM, ENS_MAX_R = 256, 16
ENS_PAIRS = [
    (b, r) for b in range(1, ENS_PERM + 1) for r in range(1, ENS_MAX_R + 1) if b * r <= ENS_PERM
]


def h(x):
    return float.hex(float(x))


def counting(fn, *args):
    """Call fn(*args); return (value, number of IntegrationWarnings, set of messages)."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        value = fn(*args)
    msgs = [str(w.message).split(".")[0] for w in rec if issubclass(w.category, IntegrationWarning)]
    return value, len(msgs), sorted(set(msgs))


def lsh_probs_for_threshold(t):
    fps, fns, nwarn = [], [], 0
    for b, r in LSH_PAIRS:
        fp, k1, _ = counting(lsh._false_positive_probability, t, b, r)
        fn, k2, _ = counting(lsh._false_negative_probability, t, b, r)
        fps.append(h(fp))
        fns.append(h(fn))
        nwarn += k1 + k2
    return h(t), fps, fns, nwarn


XQS = np.exp(np.linspace(-5, 5, 10))


def ens_probs_for_threshold(t):
    out = []
    for xq in XQS:
        fps, fns = [], []
        for b, r in ENS_PAIRS:
            fp, _, _ = counting(lshensemble._false_positive_probability, t, b, r, xq)
            fn, _, _ = counting(lshensemble._false_negative_probability, t, b, r, xq)
            fps.append(h(fp))
            fns.append(h(fn))
        out.append((h(xq), fps, fns))
    return h(t), out


def lsh_opt(task):
    ti, n, wi = task
    t = THRESHOLDS[ti]
    w = WEIGHTS[wi]
    t0 = time.perf_counter()
    opt, nwarn, msgs = counting(lsh._optimal_param, t, n, w[0], w[1])
    return task, list(opt), nwarn, msgs, time.perf_counter() - t0


def ens_construct(case):
    """MinHashLSHEnsemble(...).params; if the constructor raises after computing
    them (e.g. ValueError for int(num_perm / r) < 2), the params come from
    datasketch's own _init_optimal_params and the error is recorded."""
    t, n, part, m, w = case
    error = None
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            e = MinHashLSHEnsemble(threshold=t, num_perm=n, num_part=part, m=m, weights=w)
            assert len(e.indexes) == part
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            e = MinHashLSHEnsemble.__new__(MinHashLSHEnsemble)
            e.threshold, e.h, e.m = t, n, m
            e._init_optimal_params(w)
    nwarn = sum(issubclass(x.category, IntegrationWarning) for x in rec)
    return case, [h(x) for x in e.xqs], [[int(b), int(r)] for b, r in e.params], nwarn, error


DIRECT_XQS = [*XQS.tolist(), 1.0]


def ens_opt_direct(case):
    """lshensemble._optimal_param(t, n, m, xq, *w) for every xq in DIRECT_XQS."""
    t, n, m, w = case
    opts = []
    for xq in DIRECT_XQS:
        opt, _, _ = counting(lshensemble._optimal_param, t, n, m, xq, w[0], w[1])
        opts.extend(opt)
    return case, opts


def qagse_case(kind, t, b, r, xq=None):
    """scipy's _qagse on datasketch's integrand, with the limits datasketch passes."""
    fb, fr = float(b), float(r)
    if kind == "lsh_fp":
        f, lo, hi = (lambda s: 1 - (1 - s**fr) ** fb), 0.0, t
    elif kind == "lsh_fn":
        f, lo, hi = (lambda s: 1 - (1 - (1 - s**fr) ** fb)), t, 1.0
    elif kind == "ens_fp":
        f = lambda u: 1 - (1 - (u / (1 + xq - u)) ** fr) ** fb  # noqa: E731
        lo, hi = 0.0, (t if xq >= t else xq)
    else:
        f = lambda u: 1 - (1 - (1 - (u / (1 + xq - u)) ** fr) ** fb)  # noqa: E731
        lo, hi = t, (1.0 if xq >= 1.0 else xq)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res, err, info, ier = _quadpack._qagse(f, lo, hi, (), 1, 1.49e-8, 1.49e-8, 50)
    last = info["last"]
    limit = 50
    k = last if last <= limit // 2 + 2 else limit + 1 - last
    return {
        "kind": kind,
        "t": h(t),
        "b": b,
        "r": r,
        "xq": None if xq is None else h(xq),
        "a": h(lo),
        "bound": h(hi),
        "result": h(res),
        "abserr": h(err),
        "neval": int(info["neval"]),
        "ier": int(ier),
        "last": int(last),
        "lists": [
            [h(x) for x in info["alist"][:last]],
            [h(x) for x in info["blist"][:last]],
            [h(x) for x in info["rlist"][:last]],
            [h(x) for x in info["elist"][:last]],
            [int(x) for x in info["iord"][:k]],
        ],
    }


def dump_rows(rows):
    return "[\n" + ",\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("committed")
    ap.add_argument("full_dir")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--pow-probes", help="pow_probe.py diff output (inputs where two libms differ)")
    args = ap.parse_args()
    assert datasketch.__version__ == "2.0.0", datasketch.__version__
    assert scipy.__version__ == "1.18.1", scipy.__version__
    os.makedirs(args.full_dir, exist_ok=True)
    meta = {
        "generator": "gen_reference.py",
        "datasketch": datasketch.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
    t_start = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        # 1. Every LSH probability the grid evaluates (b * r <= 512).
        full_lsh = list(pool.map(lsh_probs_for_threshold, THRESHOLDS))
        print(
            f"lsh probabilities: {len(full_lsh)} thresholds x {len(LSH_PAIRS)} pairs "
            f"({time.perf_counter() - t_start:.0f} s)",
            flush=True,
        )
        # 2. Ensemble probabilities over the ensemble thresholds, 10 xqs, b * r <= 256, r <= 16.
        full_ens = list(pool.map(ens_probs_for_threshold, ENS_THRESHOLDS))
        print(f"ensemble probabilities ({time.perf_counter() - t_start:.0f} s)", flush=True)
        # 3. Ensemble constructions and direct ensemble _optimal_param calls.
        ens = list(pool.map(ens_construct, ENS_MAIN + ENS_EXTRA))
        direct_cases = [
            (t, n, m, w)
            for t in [float("nan"), 0.5, 0.0, 1.0]
            for n in [8, 32]
            for m in [1, 3, 8]
            for w in [(0.5, 0.5), (0.0, 1.0)]
        ]
        ens_direct = list(pool.map(ens_opt_direct, direct_cases))
        print(f"ensemble params ({time.perf_counter() - t_start:.0f} s)", flush=True)
        # 4. The LSH argmin grid, largest num_perm first.
        tasks = [
            (ti, n, wi)
            for n in sorted(NUM_PERMS, reverse=True)
            for ti in range(len(THRESHOLDS))
            for wi in range(len(WEIGHTS))
        ]
        opt = {}
        warn = {}
        secs = {}
        for task, o, nwarn, msgs, dt in pool.map(lsh_opt, tasks, chunksize=4):
            opt[task] = o
            warn[task] = (nwarn, msgs)
            secs.setdefault(task[1], []).append(dt)
        print(f"lsh grid: {len(tasks)} calls ({time.perf_counter() - t_start:.0f} s)", flush=True)

    # Full (uncommitted) files.
    with open(os.path.join(args.full_dir, "full_lsh_probs.json"), "w") as fh:
        json.dump({"meta": meta, "pairs": LSH_PAIRS, "rows": full_lsh}, fh, separators=(",", ":"))
    with open(os.path.join(args.full_dir, "full_ens_probs.json"), "w") as fh:
        json.dump({"meta": meta, "pairs": ENS_PAIRS, "rows": full_ens}, fh, separators=(",", ":"))
    by_n = {}
    for (ti, n, _wi), (nwarn, msgs) in warn.items():
        t = THRESHOLDS[ti]
        d = by_n.setdefault(
            n,
            {
                "calls": 0,
                "calls_with_warnings": 0,
                "warnings": 0,
                "thresholds": set(),
                "msgs": set(),
            },
        )
        d["calls"] += 1
        d["warnings"] += nwarn
        if nwarn:
            d["calls_with_warnings"] += 1
            d["thresholds"].add(h(t))
            d["msgs"].update(msgs)
    for d in by_n.values():
        d["thresholds"] = sorted(d["thresholds"])
        d["msgs"] = sorted(d["msgs"])
    lsh_prob_warnings = {row[0]: row[3] for row in full_lsh if row[3]}
    with open(os.path.join(args.full_dir, "warnings.json"), "w") as fh:
        json.dump(
            {
                "lsh_optimal_param_by_num_perm": by_n,
                "lsh_probability_warnings_by_threshold": lsh_prob_warnings,
                "ensemble_construction_warnings": [
                    [list(c[:4]), list(c[4]), nw] for c, _, _, nw, _ in ens if nw
                ],
                "ensemble_construction_errors": [
                    [list(c[:4]), list(c[4]), err] for c, _, _, _, err in ens if err
                ],
                "seconds_per_call_by_num_perm": {
                    n: [min(v), statistics_median(v), max(v)] for n, v in secs.items()
                },
            },
            fh,
            indent=1,
        )

    # Committed sample.
    rng = random.Random(20261107)
    pair_index = {p: i for i, p in enumerate(LSH_PAIRS)}
    fixed = [(1, 1), (1, 512), (512, 1), (2, 256), (256, 2), (5, 25), (25, 5), (16, 32), (32, 16)]
    lsh_rows = []
    for i, (th, fps, fns, _) in enumerate(full_lsh):
        picks = fixed[i % len(fixed) : i % len(fixed) + 2] + rng.sample(LSH_PAIRS, 7)
        for p in picks:
            j = pair_index[p]
            lsh_rows.append([th, p[0], p[1], fps[j], fns[j]])
    ens_rows = []
    for th, per_xq in full_ens:
        for xq_hex, fps, fns in per_xq:
            for j in rng.sample(range(len(ENS_PAIRS)), 4):
                b, r = ENS_PAIRS[j]
                ens_rows.append([th, b, r, xq_hex, fps[j], fns[j]])
    # scipy _qagse outputs on a sample of datasketch's integrands.
    qcases = []
    for _ in range(110):
        t = rng.choice(THRESHOLDS)
        b, r = rng.choice(LSH_PAIRS)
        kind = rng.choice(["lsh_fp", "lsh_fn"])
        if (kind == "lsh_fp" and t == 0.0) or (kind == "lsh_fn" and t == 1.0):
            continue
        qcases.append(qagse_case(kind, t, b, r))
    for _ in range(70):
        t = rng.choice(ENS_THRESHOLDS)
        xq = XQS[rng.randrange(len(XQS))]
        b, r = rng.choice(ENS_PAIRS)
        kind = rng.choice(["ens_fp", "ens_fn"])
        if kind == "ens_fn" and not xq >= t:
            continue
        qcases.append(qagse_case(kind, t, b, r, xq))
    # Keep the work lists for a few multi-interval cases only (size).
    with_lists = 0
    for q in qcases:
        if 2 <= q["last"] <= 20 and with_lists < 14:
            with_lists += 1
        else:
            q["lists"] = None
    opt_rows = [
        [h(t), [x for n in NUM_PERMS for wi in range(len(WEIGHTS)) for x in opt[(ti, n, wi)]]]
        for ti, t in enumerate(THRESHOLDS)
    ]
    ens_rows_c = [
        [h(c[0]), c[1], c[2], c[3], [h(c[4][0]), h(c[4][1])], params, err]
        for c, _, params, _, err in ens
    ]
    xqs_hex = ens[0][1]
    assert all(e[1] == xqs_hex for e in ens)
    ens_direct_rows = [[h(c[0]), c[1], c[2], [h(c[3][0]), h(c[3][1])], o] for c, o in ens_direct]

    parts = [
        '{"meta":' + json.dumps(meta),
        '"weights":' + json.dumps([[h(a), h(b)] for a, b in WEIGHTS]),
        '"lsh_probabilities":' + dump_rows(lsh_rows),
        '"ensemble_probabilities":' + dump_rows(ens_rows),
        '"qagse":' + dump_rows(qcases),
        '"lsh_num_perms":' + json.dumps(NUM_PERMS),
        '"lsh_optimal_param":' + dump_rows(opt_rows),
        '"ensemble_xqs":' + json.dumps(xqs_hex),
        '"ensemble_params":' + dump_rows(ens_rows_c),
        '"ensemble_direct_xqs":' + json.dumps([h(x) for x in DIRECT_XQS]),
        '"ensemble_optimal_param":' + dump_rows(ens_direct_rows),
        '"pow_probes":' + dump_rows(pow_probe_rows(args.pow_probes)),
    ]
    with open(args.committed, "w") as fh:
        fh.write(",\n".join(parts) + "}\n")
    print(
        f"wrote {args.committed} ({os.path.getsize(args.committed) // 1024} KB): "
        f"{len(lsh_rows)} lsh, {len(ens_rows)} ensemble probabilities, {len(qcases)} qagse, "
        f"{len(opt_rows)} x {len(NUM_PERMS)} x {len(WEIGHTS)} argmins, "
        f"{len(ens_rows_c)} ensembles, "
        f"{len(ens_direct_rows)} x {len(DIRECT_XQS)} direct ensemble searches "
        f"({time.perf_counter() - t_start:.0f} s total)",
        flush=True,
    )


def statistics_median(v):
    s = sorted(v)
    return s[len(s) // 2]


def pow_probe_rows(diff_path):
    """[x, y, math.pow(x, y)] on this platform: 300 inputs from diff_path, on which
    glibc's FMA and SSE2 pow differ, and 200 ordinary inputs from pow_probe.py."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "pow_probe", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pow_probe.py")
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    inputs = probe.inputs

    rows = []
    if diff_path:
        with open(diff_path) as fh:
            diff = json.load(fh)
        step = max(1, len(diff) // 300)
        for x_hex, y_hex, _, _ in diff[::step][:300]:
            x, y = float.fromhex(x_hex), float.fromhex(y_hex)
            rows.append([x_hex, int(y), h(math.pow(x, y))])
    for i, (x, y) in enumerate(inputs(20_000)):
        if i % 100 == 0:
            rows.append([h(x), int(y), h(math.pow(x, y))])
    return rows


if __name__ == "__main__":
    sys.exit(main())
