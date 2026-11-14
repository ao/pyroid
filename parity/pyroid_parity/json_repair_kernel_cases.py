"""Write the Rust kernel's golden file from json_repair's recorded upstream calls.

    python -m pyroid_parity.json_repair_kernel_cases           # rewrite the golden file
    python -m pyroid_parity.json_repair_kernel_cases --check   # regenerate and compare

The golden file is ``crates/pyroid-core/src/json_repair/testdata/upstream_cases.json``, which
``cargo test -p pyroid-core json_repair`` replays without Python. Its inputs are every distinct ``str`` that
json_repair's own test suite passes to ``repair_json`` (without ``json_fd``) or ``loads``, schema calls included
(repaired here without the schema), up to 1,000 characters and without lone surrogates, in the order of
``upstream-calls.jsonl``. Each input gets json_repair's outcome under four option sets: the defaults,
``skip_json_loads``, ``strict`` and ``stream_stable``. Run it in the parity env (CPython 3.12, whose Unicode 15.0
tables the Rust test uses), after ``json_repair_record``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

from pyroid_parity import json_repair_compare as C
from pyroid_parity.json_repair_record import GOLDEN

OUT = Path(__file__).resolve().parents[2] / "crates/pyroid-core/src/json_repair/testdata/upstream_cases.json"
MAX_LEN = 1000
OPTION_SETS: dict[str, dict[str, bool]] = {
    "default": {},
    "skip": {"skip_json_loads": True},
    "strict": {"strict": True},
    "stream": {"stream_stable": True},
}


def _text(fn: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
    if fn == "loads":
        return args[0] if args else kwargs.get("json_str")
    if fn == "repair_json":
        if (len(args) > 4 and args[4] is not None) or kwargs.get("json_fd") is not None:
            return None
        return args[0] if args else kwargs.get("json_str", "")
    return None


def inputs(calls: Path = GOLDEN) -> list[str]:
    """The distinct inputs, in call order."""
    seen: dict[str, None] = {}
    with calls.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            d = json.loads(line)
            text = _text(d["fn"], C.decode(d["args"]), C.decode(d["kwargs"]))
            if not isinstance(text, str) or len(text) > MAX_LEN:
                continue
            try:
                text.encode("utf-8")
            except UnicodeEncodeError:
                continue
            seen.setdefault(text)
    return list(seen)


def render() -> str:
    import json_repair

    def outcome(text: str, kw: dict[str, bool]) -> dict[str, str]:
        try:
            return {"ok": json_repair.repair_json(text, **kw)}
        except ValueError as e:
            return {"err": "ValueError", "msg": str(e)}

    cases = [{"in": t, **{name: outcome(t, kw) for name, kw in OPTION_SETS.items()}} for t in inputs()]
    doc = {"json_repair": version("json-repair"), "python": platform.python_version(), "cases": cases}
    return json.dumps(doc, ensure_ascii=False, indent=0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--check", action="store_true", help="compare with the committed file instead of writing it"
    )
    a = p.parse_args(argv)
    text = render()
    if a.check:
        # The cases and the json_repair version; the header's Python version may differ (CI runs 3.13).
        got, want = json.loads(text), json.loads(OUT.read_text(encoding="utf-8"))
        same = all(got[k] == want[k] for k in ("json_repair", "cases"))
        print(f"{OUT}: {'same cases' if same else 'DIFFERENT'} ({len(got['cases'])} cases)")
        return 0 if same else 1
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({len(json.loads(text)['cases'])} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
