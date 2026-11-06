#!/usr/bin/env python3
"""Deterministic light + dark SVG charts from hyperfine 2.0 JSON (end-to-end) and pyperf JSON (per stage).

    python chart.py e2e    results/.../e2e/hyperfine-e2e.json --out results/.../charts/e2e --title "..."
    python chart.py stages results/.../stages/*.json --out results/.../charts/stages [--docs 1000]
    python chart.py sweep  results/.../stages/*.json --out results/.../charts/sweep
    python chart.py hero   results/.../e2e/hyperfine-e2e.json results/.../ft315t/hyperfine-chain.json \
                           --run results/... --out results/.../charts/hero

Each command writes ``<out>-light.svg`` and ``<out>-dark.svg``. Rules (R2 section 5, rule 5 and 6): sorted by
median time with the fastest on top and never re-ordered by hand; pyroid / candidate rows in the accent
colour; IQR whiskers; values past ``--xmax`` are drawn to the edge and labelled "> xmax" (never clipped
silently); ``svg.hashsalt`` and no Date metadata, so re-rendering the same JSON gives byte-identical SVGs.
``stages`` draws one small-multiple panel per stage at one batch size (per-doc time, unit chosen per panel);
``sweep`` draws time per 1k characters against docs per call (log-log; the first N docs differ in length, so
each batch is normalised by its characters): per-call / FFI overhead shows up as the small-batch rise.
``hero`` draws the README chart: pyroid's native chain and datatrove at 1 and 8 workers (HERO_ROWS), each bar
labelled with its median and the pyroid rows with their speedup; title and caption come from the run folder's
metadata (``--run``). Its SVG is lean (no metadata, px size) for READMEs and image tools.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

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


def save(fig: Any, out: str, theme: str, lean: bool = False) -> None:
    """``lean`` (the hero): the figure's own size (no tight bbox), no metadata, px size, 2x PNG preview."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    bbox = None if lean else "tight"
    if lean:
        buf = io.StringIO()
        fig.savefig(buf, format="svg", transparent=True, metadata=dict.fromkeys(SVG_METADATA))
        Path(f"{out}-{theme}.svg").write_text(lean_svg(buf.getvalue()), "utf-8")
    else:
        fig.savefig(f"{out}-{theme}.svg", transparent=True, bbox_inches="tight", metadata={"Date": None})
    if PREVIEW_PNG[0]:  # raster preview only (not deterministic across matplotlib/freetype versions)
        bg = "#ffffff" if theme == "light" else "#0d1117"
        fig.savefig(
            f"{out}-{theme}.png",
            dpi=144 if lean else 110,
            bbox_inches=bbox,
            facecolor=bg,
            metadata={"Software": None},
        )
    plt.close(fig)


SVG_METADATA = ("Creator", "Date", "Format", "Type")  # all None: matplotlib writes no <metadata> block


def lean_svg(svg: str) -> str:
    """Smaller, more portable SVG, drawn the same: no DOCTYPE; width/height in px (1 pt = 4/3 CSS px; some tools
    reject ``pt``); path data and translate() offsets to 2 decimals (0.01 pt; matplotlib writes 6) without the
    optional whitespace; glyph ids ``DejaVuSans-Bold-4f`` shortened to ``b4f`` (``r4f`` for regular)."""

    def px(m: re.Match[str]) -> str:
        return re.sub(r'(width|height)="([\d.]+)pt"', lambda d: f'{d[1]}="{float(d[2]) * 4 / 3:.0f}"', m[0])

    def dp2(m: re.Match[str]) -> str:
        return re.sub(r"-?\d+\.\d+", lambda n: f"{float(n[0]):.2f}".rstrip("0").rstrip("."), m[0])

    def path(m: re.Match[str]) -> str:
        d = re.sub(r"\s+", " ", dp2(m)[3:-1]).strip()
        return 'd="' + re.sub(r" ?([A-Za-z]) ?", r"\1", d).replace(" -", "-") + '"'

    svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg, count=1)
    svg = re.sub(r"<svg\b[^>]*>", px, svg, count=1)
    svg = re.sub(r'\bd="[^"]*"', path, svg)
    svg = re.sub(r"translate\([^)]*\)", dp2, svg)
    return re.sub(r"DejaVuSans(-Bold)?-", lambda m: "b" if m[1] else "r", svg)


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


# ------------------------------------------------------------------------------------------------ hero


class HeroRow(NamedTuple):
    label: str  # the row as drawn
    pattern: str  # re.fullmatch against the raw hyperfine run name (hyperfine -n)
    vs: str | None  # label of the row this row's speedup is computed against


# The README hero: pyroid's native chain against datatrove at the same parallelism (threads vs processes).
# Patterns match the raw run names: run_consolidated.sh's ``-n "pyroid chain 3.15t threads=N"``
# (ft315t/hyperfine-chain.json) and run_baselines.sh's ``datatrove-<version>-wN`` (e2e/hyperfine-e2e.json).
# Each must match exactly one result; other results in the inputs (pyroid's stages inside datatrove's executor)
# stay in the e2e chart.
HERO_ROWS = [
    HeroRow("pyroid · 8 threads", r"pyroid chain \S+ threads=8", "datatrove · 8 processes"),
    HeroRow("pyroid · 1 thread", r"pyroid chain \S+ threads=1", "datatrove · 1 process"),
    HeroRow("datatrove · 8 processes", r"datatrove-\S+-w8", None),
    HeroRow("datatrove · 1 process", r"datatrove-\S+-w1", None),
]
# Caption names for run-config.json's e2e_stages (chain order); consecutive filters fold into one step.
HERO_STEPS = {"fix_text": "ftfy", "lid": "fastText language ID", "minhash": "MinHash dedup"}
HERO_FILTERS = {
    "gopher_repetition": "Gopher",
    "gopher_quality": "Gopher",
    "c4_quality": "C4",
    "fineweb_quality": "FineWeb",
}
HERO_CORPORA = {"dev": "FineWeb", "headline": "FineWeb"}  # pipeline.py / chain.py --corpus
# THEMES, with text contrast >= 4.5:1 (WCAG AA) and bars >= 3:1 on GitHub's #ffffff / #0d1117.
HERO_THEMES = {
    "light": {**THEMES["light"], "muted": "#59636e", "base": "#8c959f"},
    "dark": {**THEMES["dark"], "base": "#6e7681"},
}
HERO_PT = {"title": 18.0, "caption": 11.5, "row": 14.0}  # font sizes; >= 11 pt stays legible at 50% scale
HERO_MIN_W = 504.0  # points (7 in, 672 CSS px): a phone shows it at about half size


def secs(v: float) -> str:
    return f"{v:.1f} s" if v < 10 else f"{v:.0f} s"


def times(x: float) -> str:
    return f"{x:.1f}\u00d7" if x < 10 else f"{x:.0f}\u00d7"


def hero_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for spec in HERO_ROWS:
        hits = [r for r in results if re.fullmatch(spec.pattern, r["name"])]
        if len(hits) != 1:
            raise SystemExit(
                f"hero: {spec.label!r} needs exactly one run matching {spec.pattern!r}, found {len(hits)} "
                f"among: {', '.join(r['name'] for r in results)}"
            )
        rows.append({**hits[0], "label": spec.label, "vs": spec.vs})
    med = {r["label"]: r["med"] for r in rows}
    for r in rows:
        r["speedup"] = med[r["vs"]] / r["med"] if r["vs"] else None
    # pyroid on top, then the incumbents; fastest first within each
    return sorted(rows, key=lambda r: (not is_candidate(r["name"]), r["med"]))


def read_json(p: Path) -> Any:
    return json.loads(p.read_text("utf-8")) if p.exists() else None


def machine_label(m: dict[str, Any]) -> str:
    """'Docker/OrbStack VM on Apple Silicon, 12 vCPU, indicative' from machine_info.py's machine.json."""
    cpu_info = m.get("cpu") or {}
    lscpu = cpu_info.get("lscpu") or {}
    model = lscpu.get("Model name", "").strip(" -")
    cpu = m.get("host_cpu") or ("Apple Silicon" if lscpu.get("Vendor ID") == "Apple" else model)
    n = cpu_info.get("cpu_count")
    if not m.get("virtualized"):
        return ", ".join(p for p in (cpu, n and f"{n} CPUs", "bare metal") if p)
    release = (m.get("uname") or {}).get("release", "").lower()
    where = "Docker/OrbStack VM" if "orbstack" in release else ("Docker" if m.get("dockerenv") else "VM")
    return ", ".join(p for p in (f"{where} on {cpu}" if cpu else where, n and f"{n} vCPU", "indicative") if p)


def hero_text(run: Path) -> tuple[str, list[str]]:
    """Title and caption lines from the run folder: run-config.json (corpus, docs, stages), e2e/equivalence.json
    and ft315t/chain-digest.json (the digest gates), machine.json. A missing file drops its part; the output is
    called identical only when both gates passed."""
    cfg = (read_json(run / "run-config.json") or {}).get("resolved", {})
    chain = read_json(run / "ft315t" / "chain-digest.json") or {}
    eq = read_json(run / "e2e" / "equivalence.json") or {}
    docs = cfg.get("e2e_docs")
    if docs and chain.get("docs") and chain["docs"] != docs:
        raise SystemExit(f"hero: chain.py ran {chain['docs']} docs, the datatrove e2e {docs}")
    corpus = cfg.get("e2e_corpus", "")
    corpus = HERO_CORPORA.get(corpus, corpus or "FineWeb")
    title = f"Cleaning {docs:,} {corpus} documents" if docs else f"Cleaning {corpus} documents"
    steps: list[str] = []
    filters: list[str] = []
    for stage in [*cfg.get("e2e_stages", "").split(), ""]:  # "" flushes trailing filters
        if stage in HERO_FILTERS:
            if HERO_FILTERS[stage] not in filters:
                filters.append(HERO_FILTERS[stage])
            continue
        if filters:
            steps.append("/".join(filters) + " filters")
            filters = []
        if stage:
            steps.append(HERO_STEPS.get(stage, stage))
    same = (
        "identical output"
        if eq.get("ok") and chain.get("digest_matches")
        else "output NOT verified identical"
    )
    machine = read_json(run / "machine.json")
    tail = " · ".join([same, *([machine_label(machine)] if machine else [])])
    return title, [" → ".join(steps), tail] if steps else [tail]


def hbar(ax: Any, x: float, y: float, w: float, h: float, color: str, r: float = 3.0) -> None:
    """A bar from x to x + w centred on y (y grows down), square at the baseline, rounded at the data end."""
    r = min(r, w, h / 2)
    t, b, e = y - h / 2, y + h / 2, x + w
    verts = [(x, t), (e - r, t), (e, t), (e, t + r), (e, b - r), (e, b), (e - r, b), (x, b), (x, t)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO]
    codes += [MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, lw=0, clip_on=False))


def chart_hero(a: argparse.Namespace) -> None:
    """Laid out in points on one axes spanning the figure (y down), from measured text: the label column is
    as wide as the longest label, and the bar scale is the largest that keeps every bar's label inside the
    width the title and caption need (at least HERO_MIN_W), so the longest bar and its label end at the edge."""
    if not a.run:
        raise SystemExit("hero: --run <results folder> is required (the title and caption come from it)")
    rows = hero_rows(load_hyperfine(a.inputs))
    title, caption = hero_text(Path(a.run))
    title = a.title or title
    fs, pad, pitch, bar_h, gap, sep = HERO_PT, 1.5, 32.0, 18.0, 7.0, HERO_PT["row"] * 0.9  # points
    for r in rows:
        r["time"] = secs(r["med"])
        r["faster"] = f"{times(r['speedup'])} faster" if r["speedup"] else ""
        r["hi"] = is_candidate(r["name"])

    measure = plt.figure(dpi=72)  # 1 px = 1 pt
    renderer = measure.canvas.get_renderer()

    def width(s: str, size: float, bold: bool = False) -> float:
        prop = FontProperties(size=size, weight="bold" if bold else "normal")
        return float(renderer.get_text_width_height_descent(s, prop, ismath=False)[0])

    for r in rows:
        r["time_w"] = width(r["time"], fs["row"])
        r["value_w"] = r["time_w"] + (sep + width(r["faster"], fs["row"], bold=True) if r["faster"] else 0)
    x0 = pad + max(width(r["label"], fs["row"], r["hi"]) for r in rows) + 16
    heads = [width(title, fs["title"], bold=True), *(width(s, fs["caption"]) for s in caption)]
    plt.close(measure)
    w = math.ceil((max(HERO_MIN_W, *heads) + 2 * pad) / 0.75) * 0.75  # whole CSS px
    scale = min((w - pad - x0 - gap - r["value_w"]) / r["med"] for r in rows)
    lines = [pad + fs["title"] * 1.3 + i * fs["caption"] * 1.4 for i in range(len(caption))]
    top = pad + fs["title"] * 1.3 + len(caption) * fs["caption"] * 1.4 + 12
    h = math.ceil((top + len(rows) * pitch + pad) / 0.75) * 0.75

    for theme, c in HERO_THEMES.items():
        fig = plt.figure(figsize=(w / 72, h / 72), dpi=72)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
        ax.text(pad, pad, title, fontsize=fs["title"], fontweight="bold", color=c["text"], va="top")
        for y, line in zip(lines, caption, strict=True):
            ax.text(pad, y, line, fontsize=fs["caption"], color=c["muted"], va="top")
        for i, r in enumerate(rows):
            y, end = top + (i + 0.5) * pitch, x0 + r["med"] * scale
            weight = "bold" if r["hi"] else "normal"
            ax.text(pad, y, r["label"], fontsize=fs["row"], fontweight=weight, color=c["text"], va="center")
            hbar(ax, x0, y, r["med"] * scale, bar_h, c["accent"] if r["hi"] else c["base"])
            ax.text(end + gap, y, r["time"], fontsize=fs["row"], color=c["text"], va="center")
            if r["faster"]:
                x = end + gap + r["time_w"] + sep
                ax.text(
                    x, y, r["faster"], fontsize=fs["row"], fontweight="bold", color=c["text"], va="center"
                )
        save(fig, a.out, theme, lean=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["e2e", "stages", "sweep", "hero"])
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True, help="output prefix; writes <out>-light.svg and <out>-dark.svg")
    ap.add_argument("--title")
    ap.add_argument("--xmax", type=float, help="e2e: clip bars at this many seconds (labelled '> xmax')")
    ap.add_argument("--docs", type=int, help="stages: batch size to plot (default: largest per stage)")
    ap.add_argument("--cols", type=int, default=2, help="small-multiple columns")
    ap.add_argument("--run", help="hero: the results folder whose metadata names the title and caption")
    ap.add_argument("--png", action="store_true", help="also write PNG previews next to the SVGs")
    a = ap.parse_args()
    setup()
    PREVIEW_PNG[0] = a.png
    {"e2e": chart_e2e, "stages": chart_stages, "sweep": chart_sweep, "hero": chart_hero}[a.kind](a)


if __name__ == "__main__":
    main()
