"""``parity-report.json``: collector, writer, validator and Markdown renderer.

    python -m pyroid_parity.report validate parity-report.json
    python -m pyroid_parity.report render parity-report.json [--out compat.md]
    python -m pyroid_parity.report merge a.json b.json -o merged.json   # e.g. one report per interpreter

The schema lives in ``parity/schema/parity-report.schema.json`` (JSON Schema 2020-12). Status per check:
``pass`` (>=1 comparison, 0 diffs), ``fail`` (any diff), ``skip`` (0 comparisons). A suite fails if any check fails.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import sysconfig
import threading
from pathlib import Path
from typing import Any

from pyroid_parity import data as D

SCHEMA_VERSION = 1
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "parity-report.schema.json"
MAX_EXAMPLES = 5
MAX_REPR = 300


def _short(x: Any) -> Any:
    r = x if isinstance(x, (int, float, bool)) or x is None else repr(x)
    return r[:MAX_REPR] + "…" if isinstance(r, str) and len(r) > MAX_REPR else r


class Collector:
    """Thread-safe tally of comparisons; one per pytest session (free-threaded runs record from many threads)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.checks: dict[tuple[str, str], dict[str, Any]] = {}
        self.impls: dict[str, dict[str, Any]] = {}
        self.references: dict[str, dict[str, Any]] = {}
        self.inputs: dict[str, set[str]] = {}
        self.skips: dict[tuple[str, str], str] = {}

    def record(
        self,
        suite: str,
        check: str,
        same: bool,
        *,
        case: str | None = None,
        input: Any = None,
        got: Any = None,
        want: Any = None,
    ) -> bool:
        with self.lock:
            c = self.checks.setdefault((suite, check), {"same": 0, "diff": 0, "examples": []})
            c["same" if same else "diff"] += 1
            if not same and len(c["examples"]) < MAX_EXAMPLES:
                c["examples"].append(
                    {"case": case, "input": _short(input), "got": _short(got), "want": _short(want)}
                )
        return same

    def skip(self, suite: str, check: str, reason: str) -> None:
        with self.lock:
            self.skips[(suite, check)] = reason

    def uses(self, suite: str, *rels: str) -> None:
        with self.lock:
            self.inputs.setdefault(suite, set()).update(rels)

    def build(self) -> dict[str, Any]:
        manifest = D.load_manifest()
        suites: dict[str, Any] = {}
        names = {s for s, _ in self.checks} | {s for s, _ in self.skips} | set(self.impls)
        for s in sorted(names):
            checks: dict[str, Any] = {}
            for (suite, check), c in sorted(self.checks.items()):
                if suite == s:
                    checks[check] = {"status": "fail" if c["diff"] else "pass", **c}
            for (suite, check), why in sorted(self.skips.items()):
                if suite == s and check not in checks:
                    checks[check] = {
                        "status": "skip",
                        "same": 0,
                        "diff": 0,
                        "examples": [],
                        "skip_reason": why,
                    }
            statuses = {c["status"] for c in checks.values()}
            status = "fail" if "fail" in statuses else "pass" if "pass" in statuses else "skip"
            suites[s] = {
                "status": status,
                "candidate": self.impls.get(s, {}),
                "reference": self.references.get(s, {}),
                "inputs": {
                    rel: manifest["files"].get(rel, {}).get("sha256")
                    for rel in sorted(self.inputs.get(s, ()))
                },
                "checks": checks,
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "tool": {"name": "pyroid-parity", "git_sha": _git_sha()},
            "environment": environment(),
            "manifest_sha256": hashlib.sha256(D.MANIFEST.read_bytes()).hexdigest()
            if D.MANIFEST.exists()
            else None,
            "status": _overall(suites),
            "suites": suites,
        }


def _overall(suites: dict[str, Any]) -> str:
    st = {v["status"] for v in suites.values()}
    return "fail" if "fail" in st else "pass" if "pass" in st else "skip"


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment() -> dict[str, Any]:
    ft = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "free_threaded": ft,
        "gil_enabled": sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "in_docker": Path("/.dockerenv").exists(),
    }


def write(report: dict[str, Any], out: Path) -> None:
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as f:
            f.write(render(report) + "\n")


def validate(report: dict[str, Any]) -> list[str]:
    """Validate against the JSON Schema; returns error messages (empty = valid)."""
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text("utf-8"))
    v = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in v.iter_errors(report)]


def render(report: dict[str, Any]) -> str:
    env = report["environment"]
    py = f"{env['implementation']} {env['python']}{'t' if env['free_threaded'] else ''} ({env['machine']})"
    lines = [
        f"### Parity: **{report['status']}** · {py}",
        "",
        "| suite | candidate | reference | check | status | same | diff |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for name, s in report["suites"].items():
        cand = f"{s['candidate'].get('module', '?')} {s['candidate'].get('version') or ''}".strip()
        ref = f"{s['reference'].get('name', '?')} {s['reference'].get('version') or ''}".strip()
        for check, c in s["checks"].items():
            mark = {"pass": "✅", "fail": "❌", "skip": "⏭️"}[c["status"]]
            lines.append(
                f"| {name} | {cand} | {ref} | {check} | {mark} {c['status']} | {c['same']} | {c['diff']} |"
            )
    fails = [
        (n, ch, c) for n, s in report["suites"].items() for ch, c in s["checks"].items() if c["examples"]
    ]
    for n, ch, c in fails:
        lines += ["", f"<details><summary>{n}/{ch}: first {len(c['examples'])} diffs</summary>", "", "```"]
        lines += [json.dumps(e, ensure_ascii=False) for e in c["examples"]] + ["```", "</details>"]
    return "\n".join(lines)


def merge(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine reports from several interpreters into ``{"schema_version", "runs": [...]}`` keyed by env."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": _overall({str(i): r for i, r in enumerate(reports)}),
        "runs": reports,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pyroid_parity.report")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("report")
    r = sub.add_parser("render")
    r.add_argument("report")
    r.add_argument("--out")
    m = sub.add_parser("merge")
    m.add_argument("reports", nargs="+")
    m.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "validate":
        errs = validate(json.loads(Path(a.report).read_text("utf-8")))
        print("\n".join(errs) or "valid")
        return 1 if errs else 0
    if a.cmd == "render":
        md = render(json.loads(Path(a.report).read_text("utf-8")))
        if a.out:
            Path(a.out).write_text(md + "\n", "utf-8")
        else:
            print(md)
        return 0
    merged = merge([json.loads(Path(p).read_text("utf-8")) for p in a.reports])
    Path(a.out).write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
