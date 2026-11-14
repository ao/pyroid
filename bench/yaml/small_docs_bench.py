#!/usr/bin/env python3
"""Per-call time of pyroid.yaml and PyYAML on small documents (the overhead of one call).

    python -P bench/yaml/small_docs_bench.py

Interleaves the implementations in one process and keeps the best of 7 rounds of 2,000 calls.
"""

from __future__ import annotations

import time

import pyroid.yaml as py

import yaml

DOCS = {
    "5 bytes": "a: 1\n",
    "300 bytes": "name: test\nversion: 1.2.3\ndeps: [a, b, c]\nopts: {x: true, y: null}\n" * 4,
    "2 KB workflow": """name: CI
on:
  push:
    branches: [main]
  pull_request:
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - run: pip install -e .[test]
      - run: pytest -q
      - name: Upload coverage
        if: matrix.os == 'ubuntu-latest'
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
"""
    * 3,
}


def best(f: object, n: int = 2000, rounds: int = 7) -> float:
    out = float("inf")
    for _ in range(rounds):
        t = time.perf_counter()
        for _ in range(n):
            f()  # type: ignore[operator]
        out = min(out, (time.perf_counter() - t) / n)
    return out


def main() -> None:
    print("| Document | pyroid load | CSafeLoader | SafeLoader | pyroid dump | CSafeDumper | SafeDumper |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, text in DOCS.items():
        data = yaml.safe_load(text)
        cells = [
            best(lambda t=text: py.safe_load(t)),
            best(lambda t=text: yaml.load(t, Loader=yaml.CSafeLoader)),
            best(lambda t=text: yaml.load(t, Loader=yaml.SafeLoader), n=200),
            best(lambda d=data: py.safe_dump(d)),
            best(lambda d=data: yaml.dump(d, Dumper=yaml.CSafeDumper)),
            best(lambda d=data: yaml.safe_dump(d), n=200),
        ]
        print(f"| {name} | " + " | ".join(f"{c * 1e6:.1f} µs" for c in cells) + " |")


if __name__ == "__main__":
    main()
