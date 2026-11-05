#!/usr/bin/env python3
"""Deterministic light + dark SVG charts from hyperfine 2.0 JSON (end-to-end) and pyperf JSON (per stage).

    python chart.py e2e    results/.../e2e/hyperfine-e2e.json --out results/.../charts/e2e --title "..."
    python chart.py stages results/.../stages/*.json --out results/.../charts/stages [--docs 1000]
    python chart.py sweep  results/.../stages/*.json --out results/.../charts/sweep

Each command writes ``<out>-light.svg`` and ``<out>-dark.svg``. Rules (R2 section 5, rule 5 and 6): sorted by
median time with the fastest on top and never re-ordered by hand; pyroid / candidate rows in the accent
colour; IQR whiskers; values past ``--xmax`` are drawn to the edge and labelled "> xmax" (never clipped
silently); ``svg.hashsalt`` and no Date metadata, so re-rendering the same JSON gives byte-identical SVGs.
``stages`` draws one small-multiple panel per stage at one batch size (per-doc time, unit chosen per panel);
``sweep`` draws time per 1k characters against docs per call (log-log; the first N docs differ in length, so
each batch is normalised by its characters): per-call / FFI overhead shows up as the small-batch rise.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THEMES = {
    "light": {
        "text": "#333333",
        "muted": "#6e7781",
        "accent": "#2a78d6",
        "base": "#afb8c1",
        "grid": "#d0d7de",
    },
    "dark": {
        "text": "#c9d1d9",
        "muted": "#8b949e",
        "accent": "#3987e5",
        "base": "#57606a",
        "grid": "#30363d",
    },
}
LINE_COLORS = ["#8c959f", "#bf8700", "#1a7f37", "#cf222e", "#8250df", "#0a3069", "#953800"]
UNITS = [(1e-9, "ns"), (1e-6, "\u00b5s"), (1e-3, "ms"), (1.0, "s")]


def setup() -> None:
    plt.rcParams.update(
        {"svg.hashsalt": "pyroid-bench", "svg.fonttype": "path", "font.family": "DejaVu Sans", "font.size": 9}
    )


PREVIEW_PNG: list[bool] = [False]


def save(fig: Any, out: str, theme: str) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}-{theme}.svg", transparent=True, bbox_inches="tight", metadata={"Date": None})
    if PREVIEW_PNG[0]:  # raster preview only (not deterministic across matplotlib/freetype versions)
        bg = "#ffffff" if theme == "light" else "#0d1117"
        fig.savefig(
            f"{out}-{theme}.png", dpi=110, bbox_inches="tight", facecolor=bg, metadata={"Software": None}
        )
    plt.close(fig)


def quartiles(v: list[float]) -> tuple[float, float, float]:
    v = sorted(v)
    if len(v) == 1:
        return v[0], v[0], v[0]
    q1, med, q3 = statistics.quantiles(v, n=4, method="inclusive")
    return q1, med, q3


def is_candidate(name: str, reference: Any = None) -> bool:
    return name.lower().startswith(("pyroid", "candidate")) or reference in (0, False)


def unit_for(seconds: float) -> tuple[float, str]:
    scale, label = UNITS[0]
    for s, lb in UNITS:
        if seconds >= s:
            scale, label = s, lb
    return scale, label


def fmt(v: float, scale: float, unit: str) -> str:
    x = v / scale
    return f"{x:.3g} {unit}" if x < 100 else f"{x:.0f} {unit}"


# --------------------------------------------------------------------------------------------- loaders


def load_hyperfine(paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    for p in paths:
        doc = json.loads(Path(p).read_text("utf-8"))
        if doc.get("schema_version") != 2:
            raise SystemExit(f"{p}: needs hyperfine >= 2.0 --export-json (schema_version 2)")
        for r in doc["results"]:
            ok = [m for m in r["measurements"] if m.get("exit_code", 0) == 0]
            if not ok:
                continue
            q1, med, q3 = quartiles([m["time_wall_clock"]["value"] for m in ok])
            cpu = [m["time_cpu"]["value"] for m in ok if "time_cpu" in m]
            rows.append(
                {
                    "name": r.get("name") or r["command"],
                    "med": med,
                    "q1": q1,
                    "q3": q3,
                    "cpu": statistics.median(cpu) if cpu else None,
                    "n": len(ok),
                }
            )
    return sorted(rows, key=lambda r: r["med"])


def load_pyperf(paths: list[str]) -> list[dict[str, Any]]:
    """Per-doc timing rows from pyperf JSON (``--track-memory`` files are skipped)."""
    import pyperf

    rows = []
    for p in paths:
        if p.endswith(".gates.json"):
            continue
        for b in pyperf.BenchmarkSuite.load(p).get_benchmarks():
            md = b.get_metadata()
            if md.get("unit", "second") != "second" or "stage" not in md:
                continue
            docs = int(md["docs"])
            q1, med, q3 = quartiles([v / docs for v in b.get_values()])
            kchars = max(int(md.get("chars", 0)), 1) / 1000
            rows.append(
                {
                    "stage": md["stage"],
                    "name": md.get("impl_name", b.get_name()),
                    "docs": docs,
                    "med": med,
                    "q1": q1,
                    "q3": q3,
                    "per_kchar": statistics.median(b.get_values()) / kchars,
                    "reference": md.get("is_reference"),
                    "n": len(b.get_values()),
                }
            )
    return rows


# ---------------------------------------------------------------------------------------------- charts


def bars(
    ax: Any, rows: list[dict[str, Any]], c: dict[str, str], xmax: float | None, label: Any, slots: int = 0
) -> None:
    """Horizontal bars, fastest on top; ``slots`` fixes the row pitch so panels with few rows match the others."""
    rows = sorted(rows, key=lambda r: r["med"])
    xmax = xmax or max(r["q3"] for r in rows) * 1.18
    for y, r in enumerate(rows):
        hi = is_candidate(r["name"], r.get("reference"))
        ax.barh(y, min(r["med"], xmax), height=0.55, color=c["accent"] if hi else c["base"])
        if r["q3"] < xmax and r["q3"] > r["q1"]:
            ax.plot([r["q1"], r["q3"]], [y, y], color=c["muted"], lw=1.2, solid_capstyle="butt")
        text = label(r) if r["med"] < xmax else f"> {label({**r, 'med': xmax, 'cpu': None})}"
        ax.text(
            min(max(r["q3"], r["med"]), xmax) + xmax * 0.015,
            y,
            text,
            va="center",
            color=c["text"],
            fontsize=8.5,
            fontweight="bold" if hi else "normal",
        )
    ax.set_yticks(range(len(rows)), [r["name"] for r in rows], color=c["text"], fontsize=8.5)
    ax.set_ylim(max(slots, len(rows)) - 0.5, -0.5)  # inverted: first (fastest) row on top
    ax.set_xlim(0, xmax * 1.35)
    ax.set_xticks([])
    ax.tick_params(left=False)
    ax.spines[:].set_visible(False)
    ax.set_facecolor("none")


def chart_e2e(a: argparse.Namespace) -> None:
    rows = load_hyperfine(a.inputs)
    if not rows:
        raise SystemExit("no successful hyperfine results")

    def label(r: dict[str, Any]) -> str:
        cpu = f"  (CPU {r['cpu']:.0f} s)" if r.get("cpu") is not None else ""
        return f"{r['med']:.1f} s{cpu}"

    for theme, c in THEMES.items():
        fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 0.9))
        bars(ax, rows, c, a.xmax, label)
        ax.set_title(
            a.title or "wall time, median of runs (IQR whiskers); lower is better",
            color=c["text"],
            fontsize=10,
            loc="left",
        )
        save(fig, a.out, theme)


def grid(n: int, ncols: int) -> tuple[int, int]:
    ncols = max(1, min(ncols, n))
    return math.ceil(n / ncols), ncols


def chart_stages(a: argparse.Namespace) -> None:
    rows = load_pyperf(a.inputs)
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_stage[r["stage"]].append(r)
    panels = []
    for stage in sorted(by_stage, key=_stage_order):
        sizes = sorted({r["docs"] for r in by_stage[stage]})
        if a.docs and a.docs not in sizes:
            continue  # only stages measured at the requested batch size
        docs = a.docs or sizes[-1]
        panels.append((stage, docs, [r for r in by_stage[stage] if r["docs"] == docs]))
    if not panels:
        raise SystemExit(f"no pyperf timing benchmarks found{f' at --docs {a.docs}' if a.docs else ''}")
    nrows, ncols = grid(len(panels), a.cols)
    heights = [
        max(len(p[2]) for p in panels[i * ncols : (i + 1) * ncols]) * 0.34 + 0.75 for i in range(nrows)
    ]
    for theme, c in THEMES.items():
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6.6 * ncols, sum(heights)),
            squeeze=False,
            gridspec_kw={"height_ratios": heights, "hspace": 0.55, "wspace": 1.1},
        )
        for k, ax in enumerate(axes.flat):
            if k >= len(panels):
                ax.set_visible(False)
                continue
            stage, docs, pr = panels[k]
            scale, unit = unit_for(max(r["med"] for r in pr))
            slots = max(len(p[2]) for p in panels[(k // ncols) * ncols : (k // ncols + 1) * ncols])
            bars(ax, pr, c, None, lambda r, s=scale, u=unit: fmt(r["med"], s, u) + "/doc", slots)
            ax.set_title(f"{stage}  (batch of {docs} docs)", color=c["text"], fontsize=9.5, loc="left")
        fig.suptitle(
            a.title or "per-doc time, median of pyperf values (IQR whiskers); lower is better",
            color=c["text"],
            fontsize=10.5,
            x=0.01,
            ha="left",
        )
        save(fig, a.out, theme)


def chart_sweep(a: argparse.Namespace) -> None:
    rows = load_pyperf(a.inputs)
    by_stage: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_stage[r["stage"]][r["name"]].append(r)
    stages = sorted(by_stage, key=_stage_order)
    if not stages:
        raise SystemExit("no pyperf timing benchmarks found")
    nrows, ncols = grid(len(stages), a.cols)
    for theme, c in THEMES.items():
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.6 * ncols, 3.1 * nrows),
            squeeze=False,
            gridspec_kw={"hspace": 0.6, "wspace": 0.35},
        )
        for k, ax in enumerate(axes.flat):
            if k >= len(stages):
                ax.set_visible(False)
                continue
            series = by_stage[stages[k]]
            order = sorted(series, key=lambda n: min(r["per_kchar"] for r in series[n]))
            for j, name in enumerate(order):
                pts = sorted(series[name], key=lambda r: r["docs"])
                hi = is_candidate(name, pts[0].get("reference"))
                col = c["accent"] if hi else LINE_COLORS[j % len(LINE_COLORS)]
                ax.plot(
                    [p["docs"] for p in pts],
                    [p["per_kchar"] for p in pts],
                    marker="o",
                    ms=3,
                    lw=1.6 if hi else 1.1,
                    color=col,
                    label=name,
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(stages[k], color=c["text"], fontsize=9.5, loc="left")
            ax.set_xlabel("docs per call", color=c["muted"], fontsize=8)
            ax.set_ylabel("s per 1k chars", color=c["muted"], fontsize=8)
            ax.tick_params(which="both", colors=c["muted"], labelsize=7.5)
            ax.grid(True, which="major", color=c["grid"], lw=0.5)
            ax.spines[:].set_color(c["grid"])
            ax.set_facecolor("none")
            leg = ax.legend(fontsize=6.5, frameon=False, loc="best")
            for t in leg.get_texts():
                t.set_color(c["text"])
        fig.suptitle(
            a.title
            or "time per 1k chars vs docs per call (median; the first N docs differ in length, so the batch's "
            "time is normalised by its characters); lower is better",
            color=c["text"],
            fontsize=10.5,
            x=0.01,
            ha="left",
        )
        save(fig, a.out, theme)


def _stage_order(stage: str) -> tuple[int, str]:
    order = [
        "fix_text",
        "lid",
        "gopher_repetition",
        "gopher_quality",
        "c4_quality",
        "c4_quality@fineweb",
        "fineweb_quality",
        "minhash",
    ]
    return (order.index(stage) if stage in order else len(order), stage)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["e2e", "stages", "sweep"])
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True, help="output prefix; writes <out>-light.svg and <out>-dark.svg")
    ap.add_argument("--title")
    ap.add_argument("--xmax", type=float, help="e2e: clip bars at this many seconds (labelled '> xmax')")
    ap.add_argument("--docs", type=int, help="stages: batch size to plot (default: largest per stage)")
    ap.add_argument("--cols", type=int, default=2, help="small-multiple columns")
    ap.add_argument("--png", action="store_true", help="also write PNG previews next to the SVGs")
    a = ap.parse_args()
    setup()
    PREVIEW_PNG[0] = a.png
    {"e2e": chart_e2e, "stages": chart_stages, "sweep": chart_sweep}[a.kind](a)


if __name__ == "__main__":
    main()
