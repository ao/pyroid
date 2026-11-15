#!/usr/bin/env python3
"""Count pyroid.jmespath's comparisons with jmespath 1.1.0, for docs/compatibility.md.

    python bench/jmespath/parity_counts.py [--fuzz 200000] [--custom 100000] [--out FILE]

Runs what python/tests/test_jmespath_parity.py runs, at a larger scale, and counts each
comparison: the compliance suite (2 option sets, plus the AST), jmespath.py's unit tests, the
comparator grid, Hypothesis expressions over random data (plain and ``dict_cls``) and with
custom functions, and the nesting depths. Prints JSON; exits 1 on any difference.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import sysconfig
import unittest
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fuzz", type=int, default=200_000)
    ap.add_argument("--custom", type=int, default=100_000)
    ap.add_argument("--out")
    args = ap.parse_args()
    warnings.simplefilter("ignore", PendingDeprecationWarning)
    t = _load("test_jmespath_parity", ROOT / "python" / "tests" / "test_jmespath_parity.py")
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    counts: dict[str, list[int]] = {}

    def tally(key: str, ok: bool) -> None:
        c = counts.setdefault(key, [0, 0])
        c[0 if ok else 1] += 1

    def check(key: str, fn: Any, *a: Any) -> None:
        try:
            fn(*a)
        except AssertionError:
            tally(key, False)
        else:
            tally(key, True)

    for param in t.CASES:
        given_, case = param.values
        check("compliance", t.compare, case["expression"], given_)
        check(
            "compliance",
            t.compare,
            case["expression"],
            given_,
            t.jmespath.Options(dict_cls=OrderedDict),
            t.pj.Options(dict_cls=OrderedDict),
        )
    for name in t.UPSTREAM:
        suite = unittest.TestLoader().loadTestsFromModule(t._load_upstream(name))
        result = unittest.TestResult()
        suite.run(result)
        bad = len(result.errors) + len(result.failures)
        counts.setdefault("upstream", [0, 0])
        counts["upstream"][0] += result.testsRun - bad
        counts["upstream"][1] += bad
    for op in ["==", "!=", "<", "<=", ">", ">="]:
        for a in t.SCALARS:
            for b in t.SCALARS:
                check("comparators", t.compare, f"a {op} b", {"a": a, "b": b})
                check("comparators", t.compare, f"[?a {op} b] | length(@)", [{"a": a, "b": b}])
                check("comparators", t.compare, f"!(a {op} b)", {"a": a, "b": b})
    for case in t.REAL_CASES:
        check("real", t.compare, case["expression"], t._decode(case["data"]))
    for kind in sorted(t.NESTING):
        build = t.NESTING[kind]
        tally("nesting", t._deepest(t.pj.search, build) == t._deepest(t.jmespath.search, build))

    u = t._u
    quiet = {"deadline": None, "suppress_health_check": list(HealthCheck), "database": None}

    @settings(max_examples=args.fuzz, **quiet)
    @given(u.expressions, u.data, st.booleans())
    def fuzz(expression: str, value: Any, ordered: bool) -> None:
        if ordered:
            check(
                "fuzz",
                t.compare,
                expression,
                value,
                t.jmespath.Options(dict_cls=OrderedDict),
                t.pj.Options(dict_cls=OrderedDict),
            )
        else:
            check("fuzz", t.compare, expression, value)

    @settings(max_examples=args.custom, **quiet)
    @given(
        t.custom_expressions, u.data, st.sampled_from(["plus", "dispatch", "plain"]), st.booleans()
    )
    def custom(expression: str, value: Any, kind: str, jbase: bool) -> None:
        check("custom", t.test_fuzz_custom_functions.hypothesis.inner_test, expression, value, kind, jbase)

    if args.fuzz:
        fuzz()
    if args.custom:
        custom()
    out = {
        "python": sys.version.split()[0] + ("t" if sysconfig.get_config_var("Py_GIL_DISABLED") else ""),
        "counts": {k: {"same": v[0], "diff": v[1]} for k, v in counts.items()},
    }
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    return 1 if any(v[1] for v in counts.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
