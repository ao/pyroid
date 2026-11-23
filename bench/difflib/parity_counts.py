#!/usr/bin/env python3
"""Count the comparisons behind docs/compatibility.md's pyroid.difflib table, for this Python.

    PYROID_DIFFLIB_EXAMPLES=20000 python bench/difflib/parity_counts.py --out counts.json

Runs python/tests/test_difflib_parity.py with Hypothesis statistics (``--examples`` per test)
and python/tests/test_difflib_cpython.py, and records how many examples and CPython tests passed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "python" / "tests"


def run(args: list[str], env: dict[str, str]) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=TESTS,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.exit(proc.stdout[-5000:] + proc.stderr[-2000:])
    return proc.stdout


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--examples", type=int, default=20_000)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    env = {**os.environ, "PYROID_DIFFLIB_EXAMPLES": str(args.examples), "HYPOTHESIS_PROFILE": "ci"}
    # No per-test timeout: thousands of examples of one test take longer than the suite's 300 s.
    out = run(["-q", "--timeout=0", "test_difflib_parity.py", "--hypothesis-show-statistics"], env)
    fuzz: dict[str, int] = {}
    test = None
    for line in out.splitlines():
        m = re.match(r"test_difflib_parity\.py::(\w+):", line.strip())
        if m:
            test = m.group(1)
        m = re.search(r"(\d+) passing(?: examples)?,? ", line)
        if m and test:
            fuzz[test] = fuzz.get(test, 0) + int(m.group(1))
    out = run(["-q", "test_difflib_cpython.py", "-rs"], env)
    m = re.search(r"(\d+) passed", out)
    s = re.search(r"(\d+) skipped", out)
    result = {
        "python": sys.version.split()[0]
        + ("t" if not getattr(sys, "_is_gil_enabled", lambda: True)() else ""),
        "fuzz_examples": fuzz,
        "fuzz_total": sum(fuzz.values()),
        "cpython_tests_passed": int(m.group(1)) if m else 0,
        "cpython_tests_skipped": int(s.group(1)) if s else 0,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
