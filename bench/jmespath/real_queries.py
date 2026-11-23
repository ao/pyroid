#!/usr/bin/env python3
"""Real JMESPath queries: compare pyroid.jmespath with jmespath on them.

The expressions are botocore's paginator ``result_key``, ``more_results``, ``output_token`` and
``non_aggregate_keys`` expressions and waiter acceptor ``argument`` expressions (from botocore's
service data), and the ``--query`` arguments of the AWS CLI's documentation examples (aws-cli v1's
``awscli/examples/**/*.rst``). Each runs on responses generated from the operation's output shape
(seeded), and on the example's own output where the CLI example shows one.

    python bench/jmespath/real_queries.py [--per-op 3] [--sample-out FILE]

Needs botocore and awscli installed (no AWS calls). Prints counts; ``--sample-out`` writes a
small deterministic sample (expressions and data) for python/tests/test_jmespath_parity.py.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import random
import re
import shlex
import sys
import warnings
from pathlib import Path
from typing import Any

import botocore.session
from botocore import xform_name

HERE = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location(
    "_jmespath_util", HERE.parents[2] / "python" / "tests" / "_jmespath_util.py"
)
assert spec is not None and spec.loader is not None
util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(util)

import pyroid.jmespath as pj

import jmespath

WORDS = ["running", "stopped", "pending", "available", "ACTIVE", "in-use", "true", "Name", "web", "db-1", ""]


def generate(shape: Any, rng: random.Random, depth: int = 0) -> Any:
    """A response value for a botocore shape, as botocore's parser would return it."""
    t = shape.type_name
    if t == "structure":
        out = {}
        for name, member in shape.members.items():
            if depth < 7 and rng.random() < 0.85:
                out[name] = generate(member, rng, depth + 1)
        return out
    if t == "list":
        if depth >= 7:
            return []
        return [generate(shape.member, rng, depth + 1) for _ in range(rng.choice([0, 1, 2, 3]))]
    if t == "map":
        if depth >= 7:
            return {}
        return {
            generate(shape.key, rng, depth + 1): generate(shape.value, rng, depth + 1)
            for _ in range(rng.choice([0, 1, 2]))
        }
    if t == "string":
        enum = shape.enum
        if enum:
            return rng.choice(enum)
        return rng.choice(WORDS) + (str(rng.randint(0, 99)) if rng.random() < 0.5 else "")
    if t in ("integer", "long"):
        return rng.choice([0, 1, rng.randint(-5, 5000)])
    if t in ("float", "double"):
        return rng.choice([0.0, 1.5, rng.uniform(-10, 10)])
    if t == "boolean":
        return rng.random() < 0.5
    if t == "timestamp":
        return datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC) + datetime.timedelta(
            seconds=rng.randint(0, 10**7)
        )
    if t == "blob":
        return b"blob"
    return None


def as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def botocore_expressions(session: Any) -> list[tuple[str, str, str, str]]:
    """(source, service, operation, expression) from paginators and waiters."""
    loader = session.get_component("data_loader")
    out = []
    for service in session.get_available_services():
        try:
            pag = loader.load_service_model(service, "paginators-1")
        except Exception:  # noqa: BLE001 - not every service has paginators
            pag = {}
        for op, cfg in pag.get("pagination", {}).items():
            keys = ("result_key", "more_results", "output_token", "non_aggregate_keys")
            out.extend((f"paginator.{key}", service, op, e) for key in keys for e in as_list(cfg.get(key)))
        try:
            wait = loader.load_service_model(service, "waiters-2")
        except Exception:  # noqa: BLE001
            wait = {}
        out.extend(
            ("waiter.argument", service, cfg["operation"], acc["argument"])
            for cfg in wait.get("waiters", {}).values()
            for acc in cfg.get("acceptors", [])
            if acc.get("matcher") in ("path", "pathAll", "pathAny")
        )
    return out


QUERY = re.compile(r"--query[\s=]+('(?:[^']|'\"'\"')*'|\"(?:[^\"\\]|\\.)*\"|[^\s'\"\\|<>]+)")
CLI_SERVICE = {"s3api": "s3", "configservice": "config"}


def cli_expressions(session: Any, root: Path) -> list[tuple[str, str, str, str, Any]]:
    """(source, service, operation, expression, shown output or None) from aws-cli's examples."""
    services = set(session.get_available_services())
    out = []
    for rst in sorted(root.glob("*/*.rst")):
        service = CLI_SERVICE.get(rst.parent.name, rst.parent.name)
        if service not in services:
            continue
        model = session.get_service_model(service)
        ops = {xform_name(o, "-"): o for o in model.operation_names}
        op = ops.get(rst.stem)
        text = rst.read_text(encoding="utf-8", errors="replace")
        for m in QUERY.finditer(text):
            try:
                expr = shlex.split(m.group(1))[0]
            except ValueError:
                continue
            if expr != "<value>":
                out.append(("cli.--query", service, op or "", expr, None))
    return out


AWS_GUIDE = "https://docs.aws.amazon.com/cli/latest/userguide/cli-usage-filter.html"
ANSIBLE_GUIDE = (
    "https://raw.githubusercontent.com/ansible-collections/community.general/main/"
    "docs/docsite/rst/filter_guide_selecting_json_data.rst"
)


def doc_cases() -> list[tuple[str, str, list[Any]]]:
    """(source, expression, data values) from the AWS CLI user guide's filtering page and the
    Ansible json_query guide, fetched now (their text is not copied into the repository)."""
    import html
    import urllib.request

    def fetch(url: str) -> str:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read().decode("utf-8")

    out: list[tuple[str, str, list[Any]]] = []
    page = fetch(AWS_GUIDE)
    blocks = [
        html.unescape(re.sub(r"<[^>]+>", "", b))
        for b in re.findall(r"<pre[^>]*>(.*?)</pre>", page, re.DOTALL)
    ]
    data = []
    for b in blocks:
        start = b.find("{")
        if start < 0:
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(b[start:])
        except ValueError:
            continue
        if value not in data:
            data.append(value)
    out.extend(
        ("docs.aws-cli-guide", shlex.split(m.group(1))[0], data) for b in blocks for m in QUERY.finditer(b)
    )
    rst = fetch(ANSIBLE_GUIDE)
    value = json.loads(rst.split(".. code-block:: yaml+jinja", 2)[1].split("\n\nTo ", 1)[0])
    value = value["domain_definition"]
    out.extend(
        ("docs.ansible-json_query", m.group(1).replace("''", "'"), [value])
        for m in re.finditer(r"json_query\('((?:[^']|'')*)'\)", rst)
    )
    out.extend(
        ("docs.ansible-json_query", m.group(1), [value]) for m in re.finditer(r"_query: \"([^\"]*)\"", rst)
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-op", type=int, default=3, help="generated responses per expression")
    ap.add_argument("--sample-out", help="write a deterministic sample for the tests")
    ap.add_argument("--sample-rate", type=float, default=0.04, help="share of expressions sampled")
    ap.add_argument("--docs", action="store_true", help="also fetch the AWS CLI and Ansible guides")
    ap.add_argument(
        "--examples",
        help="aws-cli's awscli/examples directory (default: the installed awscli's; the v2 "
        "branch on GitHub has many more)",
    )
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    session = botocore.session.get_session()
    import awscli

    examples = Path(args.examples) if args.examples else Path(awscli.__file__).parent / "examples"
    entries = [(*e, None) for e in botocore_expressions(session)] + cli_expressions(session, examples)
    rng = random.Random(20261115)
    counts: dict[str, list[int]] = {}
    sample = []
    diffs = []
    shapes: dict[tuple[str, str], Any] = {}
    for source, service, op, expr, _ in entries:
        key = (service, op)
        if key not in shapes:
            shape = None
            if op:
                try:
                    shape = session.get_service_model(service).operation_model(op).output_shape
                except Exception:  # noqa: BLE001
                    shape = None
            shapes[key] = shape
        shape = shapes[key]
        values: list[Any] = [{}]
        if shape is not None:
            values += [generate(shape, rng) for _ in range(args.per_op)]
        c = counts.setdefault(source, [0, 0, 0, 0])  # expressions, evaluations, non-null, differences
        c[0] += 1
        for v in values:
            ids = util.index(v)
            a = util.outcome(jmespath.search, expr, v, None, ids)
            b = util.outcome(pj.search, expr, v, None, ids)
            c[1] += 1
            if a[0] == "ok" and a[1] != ("NoneType", "None"):
                c[2] += 1
            if a != b:
                c[3] += 1
                if len(diffs) < 20:
                    diffs.append((source, service, op, expr, a, b))
        if args.sample_out and rng.random() < args.sample_rate and shape is not None:
            sample.append(
                {"source": source, "service": service, "operation": op, "expression": expr, "data": values[1]}
            )
    if args.docs:
        for source, expr, values in doc_cases():
            c = counts.setdefault(source, [0, 0, 0, 0])
            c[0] += 1
            for v in values:
                ids = util.index(v)
                a = util.outcome(jmespath.search, expr, v, None, ids)
                b = util.outcome(pj.search, expr, v, None, ids)
                c[1] += 1
                if a[0] == "ok" and a[1] != ("NoneType", "None"):
                    c[2] += 1
                if a != b:
                    c[3] += 1
                    diffs.append((source, "", "", expr, a, b))
    for d in diffs:
        print("DIFF", *d, sep="\n  ")
    total = [sum(c[i] for c in counts.values()) for i in range(4)]
    for source, c in sorted(counts.items()):
        print(f"{source:28s} expressions {c[0]:6d} evaluations {c[1]:7d} non-null {c[2]:7d} different {c[3]}")
    print(
        f"{'total':28s} expressions {total[0]:6d} evaluations {total[1]:7d} non-null {total[2]:7d} different {total[3]}"
    )
    print(
        "botocore",
        botocore.__version__,
        "awscli",
        awscli.__version__,
        "jmespath",
        jmespath.__version__,
        "examples",
        examples,
    )
    if args.sample_out:

        def default(o: Any) -> Any:
            if isinstance(o, datetime.datetime):
                return {"$datetime": o.isoformat()}
            if isinstance(o, bytes):
                return {"$bytes": o.decode()}
            raise TypeError(type(o))

        Path(args.sample_out).write_text(json.dumps(sample, default=default, sort_keys=False) + "\n")
        print("sample:", len(sample), "cases ->", args.sample_out)
    return 1 if total[3] else 0


if __name__ == "__main__":
    sys.exit(main())
