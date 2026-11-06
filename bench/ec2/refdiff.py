#!/usr/bin/env python3
"""Check regenerated parity references against bench/manifest.json, quantify any delta, and re-pin.

Runs in the parity env (stdlib + pyarrow + ``pyroid_parity``), with ``PYROID_DATA`` (and ``PYROID_MANIFEST``, if
set) pointing where ``python -m pyroid_parity.refs all`` wrote:

    cd parity && PYROID_DATA=/data uv run --frozen --no-default-groups python ../bench/ec2/refdiff.py \\
        diff OLD.parquet NEW.parquet [--kind auto|fasttext|metrics] [--out delta.json]
    cd parity && PYROID_DATA=/data uv run --frozen --no-default-groups python ../bench/ec2/refdiff.py \\
        check --pins DIR --out check.json [--copy-new DIR] [--repin]

diff    Row-by-row comparison of two references of one kind, keyed on its key columns (a repeated key is
        matched by occurrence, in file order):
        fasttext  (model, id) -> labels, probs (top-5). Per model: rows differing, rows whose labels differ,
                  max |dp| (probabilities matched by label) and the max distance in ulps of the column's float
                  type, plus float32 ulps when every differing value is a float32 (fastText computes in float32).
        metrics   (case, key) -> out (rendered results carrying exact floats). Per case: rows differing; when
                  only the floats in ``out`` differ (tokens with a decimal point or an exponent, ints are text),
                  they are compared pairwise: max abs / relative difference and float64 ulps.
        Up to 5 examples per model / case (largest first). Prints a one-line summary, then the JSON unless
        ``--out`` is given.
check   For every reference ``pyroid_parity.refs`` generates on this platform (OUT and OUT_ML) that the
        manifest pins for this platform, hash ``$PYROID_DATA/<rel>`` and compare it with the pin. A mismatch is
        expected when the entry carries ``"platform"`` (floats that depend on the CPU / libm), else it is listed
        under ``unexpected``. Each mismatch records the old and new sha256 + bytes, is copied into
        ``--copy-new`` (if given) and, when ``--pins`` holds a file of the same name whose sha256 is the old
        pin's (the previously pinned bytes), embeds the ``diff`` of old vs new. ``--repin`` re-pins every
        mismatched file with ``pyroid_parity.data.pin``, keeping every other field of its entry (``rows`` is
        re-read from the file), and records each entry before / after. References with no file in
        ``$PYROID_DATA`` or no pin for this platform are listed under ``missing``. Exits 0 unless the tool
        itself fails: read ``reproduced`` / ``unexpected`` in the JSON.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import heapq
import json
import math
import platform
import re
import shutil
import struct
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# kind -> (key columns, value columns); the first key column groups the report (per model / per case).
KINDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "fasttext": (("model", "id"), ("labels", "probs")),
    "metrics": (("case", "key"), ("out",)),
}
EXAMPLES = 5
MAX_TEXT = 2000  # longest ``out`` string kept in an example
MAX_TOKENS = 10  # differing float tokens kept per example
# A float in a rendered result: a decimal point or an exponent (ints stay part of the text), nan or inf.
FLOAT = re.compile(r"(?<![\w.])[-+]?(?:(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+|nan|inf)(?![\w.])")
# struct formats (float, same-width unsigned int) and the sign bit, per float type.
BITS = {"float32": ("<f", "<I", 1 << 31), "float64": ("<d", "<Q", 1 << 63)}


class RefdiffError(RuntimeError):
    """The inputs cannot be compared (unknown schema, different kinds)."""


# ----------------------------------------------------------------------------------------------- floats


def _ordinal(x: float, ftype: str) -> int:
    """``x``'s position on the number line of ``ftype`` values (-0.0 and 0.0 share 0)."""
    fmt, ufmt, sign = BITS[ftype]
    (bits,) = struct.unpack(ufmt, struct.pack(fmt, x))
    return bits if bits < sign else sign - bits


def ulps(a: float, b: float, ftype: str = "float64") -> int | None:
    """Distance between ``a`` and ``b`` in units in the last place of ``ftype`` (None if one side is NaN)."""
    if math.isnan(a) or math.isnan(b):
        return 0 if math.isnan(a) and math.isnan(b) else None
    return abs(_ordinal(a, ftype) - _ordinal(b, ftype))


def is_float32(x: float) -> bool:
    try:
        return math.isnan(x) or struct.unpack("<f", struct.pack("<f", x))[0] == x
    except OverflowError:
        return False


def rel_diff(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale else 0.0


@dataclass
class UlpStats:
    max: int = 0
    counts: Counter[int] = field(default_factory=Counter)
    unknown: int = 0  # NaN on one side only

    def add(self, n: int | None) -> None:
        if n is None:
            self.unknown += 1
            return
        self.max = max(self.max, n)
        self.counts[n] += 1

    def to_json(self) -> dict[str, Any]:
        top = sorted(self.counts.most_common(10))
        out: dict[str, Any] = {"max": self.max, "counts": {str(k): v for k, v in top}}
        if rest := self.counts.total() - sum(v for _, v in top):
            out["counts_other"] = rest
        if self.unknown:
            out["nan_one_side"] = self.unknown
        return out


# ------------------------------------------------------------------------------------------------- diff


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def detect(path: Path) -> str:
    import pyarrow.parquet as pq

    names = set(pq.read_schema(path).names)
    for kind, (key, vals) in KINDS.items():
        if {*key, *vals} <= names:
            return kind
    raise RefdiffError(f"{path}: columns {sorted(names)} match no known kind ({', '.join(KINDS)})")


def read(path: Path, kind: str) -> tuple[Any, dict[str, list[Any]]]:
    import pyarrow.parquet as pq

    cols = [*KINDS[kind][0], *KINDS[kind][1]]
    table = pq.read_table(path, columns=cols)
    return table, {c: table.column(c).to_pylist() for c in cols}


def index(cols: dict[str, list[Any]], key: tuple[str, ...]) -> tuple[dict[tuple[Any, ...], int], int]:
    """``{(*key values, occurrence): row}`` and the number of repeated keys."""
    seen: Counter[tuple[Any, ...]] = Counter()
    out: dict[tuple[Any, ...], int] = {}
    for row, k in enumerate(zip(*(cols[c] for c in key), strict=True)):
        out[(*k, seen[k])] = row
        seen[k] += 1
    return out, sum(n - 1 for n in seen.values())


def key_json(key: tuple[str, ...], k: tuple[Any, ...]) -> dict[str, Any]:
    out = dict(zip(key, k, strict=False))
    if k[-1]:
        out["occurrence"] = k[-1]
    return out


def float_type(table: Any, column: str) -> str:
    import pyarrow as pa

    t = table.schema.field(column).type
    while pa.types.is_list(t) or pa.types.is_large_list(t):
        t = t.value_type
    if pa.types.is_float32(t):
        return "float32"
    if pa.types.is_float64(t):
        return "float64"
    raise RefdiffError(f"{column}: {t} is not a float32/float64 list")


@dataclass
class FtModel:
    rows: int = 0
    differing: int = 0
    label_diffs: int = 0
    probs_differing: int = 0
    max_dp: float = 0.0
    max_rel: float = 0.0
    ulps: UlpStats = field(default_factory=UlpStats)
    ulps32: UlpStats | None = field(default_factory=UlpStats)  # None once a differing value is not a float32
    cands: list[tuple[Any, ...]] = field(default_factory=list)


def fasttext_groups(
    old: dict[str, list[Any]], new: dict[str, list[Any]], pairs: list[tuple[int, int]], ftype: str
) -> dict[str, dict[str, Any]]:
    models: dict[str, FtModel] = {}
    for i, j in pairs:
        g = models.setdefault(old["model"][i], FtModel())
        g.rows += 1
        ol, op, nl, np_ = old["labels"][i], old["probs"][i], new["labels"][j], new["probs"][j]
        if ol == nl and op == np_:
            continue
        g.differing += 1
        labels_differ = ol != nl
        g.label_diffs += labels_differ
        po, pn = dict(zip(ol, op, strict=True)), dict(zip(nl, np_, strict=True))
        row_dp, row_ulps, row_ulps32 = 0.0, 0, 0
        for label, a in po.items():
            b = pn.get(label)
            if b is None or a == b:
                continue
            g.probs_differing += 1
            row_dp = max(row_dp, abs(a - b))
            g.max_rel = max(g.max_rel, rel_diff(a, b))
            u = ulps(a, b, ftype)
            g.ulps.add(u)
            row_ulps = max(row_ulps, u or 0)
            if g.ulps32 is not None and ftype != "float32":
                if is_float32(a) and is_float32(b):
                    u32 = ulps(a, b, "float32")
                    g.ulps32.add(u32)
                    row_ulps32 = max(row_ulps32, u32 or 0)
                else:
                    g.ulps32 = None
        g.max_dp = max(g.max_dp, row_dp)
        g.cands.append((labels_differ, row_dp, -i, i, j, row_ulps, row_ulps32))
    out = {}
    for model, g in models.items():
        examples = []
        for labels_differ, row_dp, _, i, j, row_ulps, row_ulps32 in heapq.nlargest(EXAMPLES, g.cands):
            ex = {
                "id": old["id"][i],
                "labels_differ": labels_differ,
                "max_abs_dp": row_dp,
                "max_ulps": row_ulps,
                "old": {"labels": old["labels"][i], "probs": old["probs"][i]},
                "new": {"labels": new["labels"][j], "probs": new["probs"][j]},
            }
            if g.ulps32 is not None and ftype != "float32":
                ex["max_ulps_float32"] = row_ulps32
            examples.append(ex)
        ulp_stats = {ftype: g.ulps.to_json()}
        if g.ulps32 is not None and ftype != "float32":
            ulp_stats["float32"] = g.ulps32.to_json()
        out[model] = {
            "rows": g.rows,
            "rows_differing": g.differing,
            "label_differences": g.label_diffs,
            "probs_differing": g.probs_differing,
            "max_abs_dp": g.max_dp,
            "max_rel_dp": g.max_rel,
            "dtype": ftype,
            "max_ulps": g.ulps.max,
            # float32 is listed when every differing probability is a float32 value
            "ulps": ulp_stats,
            "examples": examples,
        }
    return out


def split_floats(s: str) -> tuple[list[str], list[str]]:
    """``(text around the float tokens, float tokens)`` of a rendered result."""
    text, toks, pos = [], [], 0
    for m in FLOAT.finditer(s):
        text.append(s[pos : m.start()])
        toks.append(m.group())
        pos = m.end()
    text.append(s[pos:])
    return text, toks


def first_difference(a: str, b: str) -> int:
    return next((n for n, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))


@dataclass
class MetricsCase:
    rows: int = 0
    differing: int = 0
    numeric_only: int = 0
    text_differing: int = 0
    floats_differing: int = 0
    max_abs: float = 0.0
    max_rel: float = 0.0
    ulps: UlpStats = field(default_factory=UlpStats)
    cands: list[tuple[Any, ...]] = field(default_factory=list)


def metrics_groups(
    old: dict[str, list[Any]], new: dict[str, list[Any]], pairs: list[tuple[int, int]]
) -> dict[str, dict[str, Any]]:
    cases: dict[str, MetricsCase] = {}
    for i, j in pairs:
        g = cases.setdefault(old["case"][i], MetricsCase())
        g.rows += 1
        a, b = old["out"][i], new["out"][j]
        if a == b:
            continue
        g.differing += 1
        if a is None or b is None:
            g.text_differing += 1
            g.cands.append((True, 0, -i, i, j, []))
            continue
        text_a, toks_a = split_floats(a)
        text_b, toks_b = split_floats(b)
        if text_a != text_b or len(toks_a) != len(toks_b):
            g.text_differing += 1
            g.cands.append((True, 0, -i, i, j, []))
            continue
        g.numeric_only += 1
        diffs, row_ulps = [], 0
        for x, y in zip(toks_a, toks_b, strict=True):
            if x == y:
                continue
            fx, fy = float(x), float(y)
            u = ulps(fx, fy)
            g.floats_differing += 1
            g.ulps.add(u)
            row_ulps = max(row_ulps, u or 0)
            d = r = None
            if math.isfinite(fx) and math.isfinite(fy):
                d, r = abs(fx - fy), rel_diff(fx, fy)
                g.max_abs, g.max_rel = max(g.max_abs, d), max(g.max_rel, r)
            diffs.append({"old": x, "new": y, "abs": d, "rel": r, "ulps": u})
        g.cands.append((False, row_ulps, -i, i, j, diffs))
    out = {}
    for case, g in cases.items():
        examples = []
        for text_differs, _, _, i, j, diffs in heapq.nlargest(EXAMPLES, g.cands, key=lambda c: c[:3]):
            a, b = old["out"][i], new["out"][j]
            ex: dict[str, Any] = {"key": old["key"][i], "text_differs": text_differs}
            if a is not None and b is not None:
                ex["first_difference_at"] = first_difference(a, b)
            ex["old"] = a if a is None else a[:MAX_TEXT]
            ex["new"] = b if b is None else b[:MAX_TEXT]
            if diffs:
                ex["floats"] = diffs[:MAX_TOKENS]
            examples.append(ex)
        out[case] = {
            "rows": g.rows,
            "rows_differing": g.differing,
            "rows_floats_only": g.numeric_only,
            "rows_text_differing": g.text_differing,
            "floats_differing": g.floats_differing,
            "max_abs": g.max_abs,
            "max_rel": g.max_rel,
            "max_ulps": g.ulps.max,
            "ulps": {"float64": g.ulps.to_json()},
            "examples": examples,
        }
    return out


def describe(path: Path, rows: int) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size, "rows": rows}


def diff(old_path: Path, new_path: Path, kind: str = "auto") -> dict[str, Any]:
    """The delta between two references of one kind (see the module docstring)."""
    if kind == "auto":
        kinds = {detect(old_path), detect(new_path)}
        if len(kinds) != 1:
            raise RefdiffError(f"{old_path} and {new_path} are different kinds: {sorted(kinds)}")
        (kind,) = kinds
    key, _ = KINDS[kind]
    t_old, old = read(old_path, kind)
    t_new, new = read(new_path, kind)
    idx_old, dup_old = index(old, key)
    idx_new, dup_new = index(new, key)
    pairs = [(i, idx_new[k]) for k, i in idx_old.items() if k in idx_new]
    only_old = [k for k in idx_old if k not in idx_new]
    only_new = [k for k in idx_new if k not in idx_old]
    out: dict[str, Any] = {
        "kind": kind,
        "key": list(key),
        "old": describe(old_path, t_old.num_rows),
        "new": describe(new_path, t_new.num_rows),
    }
    out["identical_bytes"] = out["old"]["sha256"] == out["new"]["sha256"]
    out["schema_equal"] = t_old.schema.remove_metadata().equals(t_new.schema.remove_metadata())
    if kind == "fasttext":
        groups = fasttext_groups(old, new, pairs, float_type(t_old, "probs"))
    else:
        groups = metrics_groups(old, new, pairs)
    out |= {
        "rows_common": len(pairs),
        "rows_only_old": len(only_old),
        "rows_only_new": len(only_new),
        "rows_differing": sum(g["rows_differing"] for g in groups.values()),
        "duplicate_keys": {"old": dup_old, "new": dup_new},
        "examples_only_old": [key_json(key, k) for k in only_old[:EXAMPLES]],
        "examples_only_new": [key_json(key, k) for k in only_new[:EXAMPLES]],
        "models" if kind == "fasttext" else "cases": groups,
    }
    out["summary"] = summary(out)
    return out


def summary(d: dict[str, Any]) -> str:
    head = (
        f"{d['kind']}: {d['old']['rows']} vs {d['new']['rows']} rows "
        f"({d['rows_only_old']} only old, {d['rows_only_new']} only new), "
        f"{d['rows_differing']} of {d['rows_common']} common rows differ"
    )
    parts = [head]
    if d["kind"] == "fasttext":
        for model, g in d["models"].items():
            s = f"{model} {g['rows_differing']}/{g['rows']}"
            if g["rows_differing"]:
                s += f", {g['label_differences']} label diffs, max |dp| {g['max_abs_dp']:.3g}"
                # float32 ulps when the values are float32 (fastText computes in float32), else the column's
                ftype = "float32" if "float32" in g["ulps"] else g["dtype"]
                s += f" (rel {g['max_rel_dp']:.3g}), max {g['ulps'][ftype]['max']} {ftype} ulps"
            parts.append(s)
    else:
        for case, g in d["cases"].items():
            if not g["rows_differing"]:
                continue
            s = f"{case} {g['rows_differing']}/{g['rows']}"
            if g["floats_differing"]:
                s += f" (max {g['max_ulps']} ulps, rel {g['max_rel']:.3g})"
            if g["rows_text_differing"]:
                s += f" ({g['rows_text_differing']} beyond floats)"
            parts.append(s)
        if len(parts) == 1:
            parts.append("every case identical")
    return "; ".join(parts)


# ------------------------------------------------------------------------------------------------ check


def environment(platform_key: str) -> dict[str, Any]:
    env: dict[str, Any] = {
        "platform_key": platform_key,
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "libc": " ".join(platform.libc_ver()).strip(),
    }
    try:
        info = Path("/proc/cpuinfo").read_text("utf-8", "replace")
    except OSError:
        return env
    fields: dict[str, str] = {}
    for line in info.splitlines():
        name, sep, value = line.partition(":")
        if sep:
            fields.setdefault(name.strip(), value.strip())
    env["cpu"] = fields.get("model name") or fields.get("CPU part")
    flags = set((fields.get("flags") or fields.get("Features") or "").split())
    interesting = {"sse4_2", "avx", "avx2", "fma", "avx512f", "asimd", "fphp", "sve", "sve2"}
    env["cpu_flags"] = sorted(flags & interesting)
    return env


def check(pins: Path, copy_new: Path | None, repin: bool) -> dict[str, Any]:
    import pyarrow.parquet as pq
    from pyroid_parity import data as D
    from pyroid_parity import refs as R

    here = D.platform_key()
    files = D.load_manifest()["files"]
    before_all = copy.deepcopy(D._load_all()["files"])
    result: dict[str, Any] = {
        "environment": environment(here),
        "manifest": str(D.MANIFEST),
        "manifest_sha256": sha256(D.MANIFEST),
        "data": str(D.DATA),
        "pins": str(pins),
        "reproduced": False,
        "checked": [],
        "mismatched": [],
        "unexpected": [],
        "missing": [],
        "repinned": [],
    }
    # OUT, plus the per-language refs (OUT_ML) on branches that have them: everything `refs all` writes.
    for rel in sorted(set(R.OUT.values()) | set(getattr(R, "OUT_ML", {}).values())):
        entry = files.get(rel)
        path = D.DATA / rel
        if entry is None:
            result["missing"].append({"rel": rel, "reason": f"not pinned in the manifest for {here}"})
            continue
        specific = "platform" in entry
        if not path.is_file():
            result["missing"].append(
                {"rel": rel, "reason": f"{path} does not exist", "platform_specific": specific}
            )
            continue
        old = {"sha256": entry["sha256"], "bytes": entry["bytes"]}
        new = {"sha256": sha256(path), "bytes": path.stat().st_size}
        result["checked"].append({"rel": rel, "reproduced": old == new, **new})
        if old == new:
            continue
        rec: dict[str, Any] = {
            "rel": rel,
            "platform": entry.get("platform"),
            "platform_specific": specific,
            "expected": specific,
            "old": old,
            "new": new,
        }
        if copy_new is not None:
            copy_new.mkdir(parents=True, exist_ok=True)
            rec["copied_to"] = str(shutil.copy2(path, copy_new / path.name))
        old_file = pins / path.name
        if old_file.is_file() and sha256(old_file) == old["sha256"]:
            rec["old_file"] = str(old_file)
            try:
                rec["delta"] = diff(old_file, path)
            except RefdiffError as e:
                rec["delta"], rec["delta_error"] = None, str(e)
        else:
            rec["old_file"] = rec["delta"] = None
            rec["delta_error"] = f"no {path.name} with the old pin's sha256 in {pins}"
        result["mismatched"].append(rec)
    result["unexpected"] = [r["rel"] for r in result["mismatched"] if not r["platform_specific"]]
    result["reproduced"] = not result["mismatched"] and not result["missing"]
    if repin:
        for rec in result["mismatched"]:
            rel = rec["rel"]
            before = copy.deepcopy(files[rel])
            extra = {k: v for k, v in before.items() if k not in ("sha256", "bytes")}
            if "rows" in extra:
                extra["rows"] = pq.read_metadata(D.DATA / rel).num_rows
            after = D.pin(rel, **extra)
            D._verified.cache_clear()
            D.path(rel)  # raises DataError unless the loaders now accept the file
            changed = sorted(k for k in {*before, *after} if before.get(k) != after.get(k))
            result["repinned"].append({"rel": rel, "changed": changed, "before": before, "after": after})
        after_all = D._load_all()["files"]
        result["manifest_entries_changed"] = sorted(
            rel for rel in {*before_all, *after_all} if before_all.get(rel) != after_all.get(rel)
        )
        result["manifest_sha256_after"] = sha256(D.MANIFEST)
    return result


def check_summary(r: dict[str, Any]) -> list[str]:
    lines = []
    for rec in r["mismatched"]:
        tag = f"expected, {rec['platform']}" if rec["platform_specific"] else "UNEXPECTED"
        delta = rec["delta"]["summary"] if rec.get("delta") else rec.get("delta_error")
        lines.append(
            f"MISMATCH ({tag}) {rec['rel']}: {rec['old']['sha256'][:12]} {rec['old']['bytes']} B -> "
            f"{rec['new']['sha256'][:12]} {rec['new']['bytes']} B; {delta}"
        )
    lines += [f"MISSING {m['rel']}: {m['reason']}" for m in r["missing"]]
    lines += [f"REPINNED {p['rel']}: changed {', '.join(p['changed'])}" for p in r["repinned"]]
    lines.append(
        f"check {r['environment']['platform_key']}: {len(r['checked'])} checked, "
        f"{sum(c['reproduced'] for c in r['checked'])} reproduced, {len(r['mismatched'])} mismatched "
        f"({len(r['unexpected'])} unexpected), {len(r['missing'])} missing, {len(r['repinned'])} repinned; "
        f"reproduced={r['reproduced']}"
    )
    return lines


# ------------------------------------------------------------------------------------------------- main


def write_json(obj: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", "utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diff", help="row-by-row delta between two references of one kind")
    d.add_argument("old", type=Path)
    d.add_argument("new", type=Path)
    d.add_argument("--kind", choices=["auto", *KINDS], default="auto")
    d.add_argument("--out", type=Path, help="write the JSON here (default: stdout, after the summary)")
    c = sub.add_parser(
        "check", help="hash the regenerated references against the manifest; optionally re-pin"
    )
    c.add_argument("--pins", type=Path, required=True, help="previously pinned files, by basename")
    c.add_argument("--out", type=Path, required=True, help="JSON report")
    c.add_argument("--copy-new", type=Path, help="copy every mismatched (new) file here")
    c.add_argument("--repin", action="store_true", help="re-pin the mismatched files in the manifest")
    a = ap.parse_args()
    if a.cmd == "diff":
        try:
            delta = diff(a.old, a.new, a.kind)
        except RefdiffError as e:
            sys.exit(f"refdiff: {e}")
        print(delta["summary"])
        if a.out:
            write_json(delta, a.out)
        else:
            print(json.dumps(delta, indent=2, ensure_ascii=False))
        return
    if a.copy_new is not None and a.copy_new.resolve() == a.pins.resolve():
        ap.error("--copy-new must not be the --pins directory (it would overwrite the old pins)")
    result = check(a.pins, a.copy_new, a.repin)
    write_json(result, a.out)
    print("\n".join(check_summary(result)))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
