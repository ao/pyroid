"""Compare a consumer's results with sqlparse (A) and with pyroid.patch("sqlparse") (B).

    python compare.py json A.json B.json        # outputs of django_paths.py / toolbar_paths.py
    python compare.py junit A.xml B.xml          # pytest --junitxml reports: per-test outcomes
    python compare.py unittest A.log B.log       # Django / unittest -v 2 logs: per-test outcomes
    python compare.py probe LOG                  # probe/sitecustomize.py's per-process lines

Prints the differences (at most 20) and a summary line; exits with 1 if they differ.
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any


def short(x: Any, n: int = 300) -> str:
    s = json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else repr(x)
    return s if len(s) <= n else s[:n] + f"... ({len(s)} chars)"


def diff_json(a: Any, b: Any, path: str, out: list[str]) -> int:
    """Count the leaves that differ; record the first ones in `out`."""
    if type(a) is not type(b):
        out.append(f"{path}: {short(a)} != {short(b)}")
        return 1
    if isinstance(a, dict):
        n = 0
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: only in {'A' if k in a else 'B'}")
                n += 1
            else:
                n += diff_json(a[k], b[k], f"{path}.{k}", out)
        return n
    if isinstance(a, list):
        n = 0
        if len(a) != len(b):
            out.append(f"{path}: {len(a)} items != {len(b)} items")
            n += 1
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            n += diff_json(x, y, f"{path}[{i}]", out)
        return n
    if a != b:
        out.append(f"{path}: {short(a)} != {short(b)}")
        return 1
    return 0


def leaves(x: Any) -> int:
    if isinstance(x, dict):
        return sum(leaves(v) for v in x.values())
    if isinstance(x, list):
        return sum(leaves(v) for v in x)
    return 1


def cmp_json(a_path: str, b_path: str) -> int:
    a, b = (json.loads(Path(p).read_text()) for p in (a_path, b_path))
    # Which functions ran is expected to differ (that is the point): report it, don't count it.
    fa, fb = a.pop("functions", None), b.pop("functions", None)
    print(f"A functions: {fa}\nB functions: {fb}")
    out: list[str] = []
    n = diff_json(a, b, "", out)
    for line in out[:20]:
        print("DIFF", line)
    same_bytes = json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    print(f"{n} of {leaves(a)} values differ; byte-identical: {same_bytes}; calls: {a.get('calls')}")
    return 1 if n else 0


def junit_outcomes(path: str) -> dict[str, str]:
    res: dict[str, str] = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        name = f"{case.get('classname')}::{case.get('name')}"
        kind = "passed"
        for child in case:
            if child.tag in ("failure", "error", "skipped"):
                kind = child.tag
                if child.tag == "skipped" and child.get("type") == "pytest.xfail":
                    kind = "xfailed"
                break
        # A test that fails in teardown gets a second entry: keep the worst.
        if res.get(name) not in (None, "passed"):
            continue
        res[name] = kind
    return res


UNITTEST_RESULT = re.compile(
    r"^(?P<test>\w+ \([\w.]+\))(?:\n.*?)?? \.\.\. "
    r"(?P<outcome>ok|FAIL|ERROR|skipped|expected failure|unexpected success)",
    re.MULTILINE,
)


def unittest_outcomes(path: str) -> dict[str, str]:
    text = Path(path).read_text(errors="replace")
    res: dict[str, str] = {}
    for m in UNITTEST_RESULT.finditer(text):
        res[m["test"]] = m["outcome"]
    ran = re.findall(r"^Ran (\d+) tests?", text, re.MULTILINE)
    summary = re.findall(r"^(OK|FAILED)\b.*$", text, re.MULTILINE)
    print(
        f"{path}: Ran {ran[-1] if ran else '?'}; {summary[-1] if summary else '?'}; "
        f"{len(res)} results parsed: {dict(Counter(res.values()))}"
    )
    return res


def cmp_outcomes(a: dict[str, str], b: dict[str, str]) -> int:
    print(f"A: {len(a)} tests {dict(Counter(a.values()))}")
    print(f"B: {len(b)} tests {dict(Counter(b.values()))}")
    diffs = [(k, a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    for k, x, y in diffs[:20]:
        print(f"DIFF {k}: {x} != {y}")
    print(f"{len(diffs)} tests differ")
    return 1 if diffs else 0


def probe_summary(path: str) -> int:
    lines = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    used = [x for x in lines if x["format"] is not None]
    via: Counter[str] = Counter(x["format"] for x in used)
    calls: Counter[str] = Counter()
    for x in lines:
        calls.update(x["counts"])
    print(
        f"{path}: {len(lines)} processes, {len(used)} imported sqlparse, sqlparse.format from "
        f"{dict(via)}; native calls: {dict(calls.most_common())}"
    )
    return 0


def main(argv: list[str]) -> int:
    if argv[0] == "probe":
        return probe_summary(argv[1])
    mode, a, b = argv
    if mode == "json":
        return cmp_json(a, b)
    if mode == "junit":
        return cmp_outcomes(junit_outcomes(a), junit_outcomes(b))
    if mode == "unittest":
        return cmp_outcomes(unittest_outcomes(a), unittest_outcomes(b))
    raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
