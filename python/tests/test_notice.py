"""NOTICE attributes every project pyroid ports code from.

Each ported file names its source project in its first lines ("Derived from X", "port of X",
"Ports X"); NOTICE, which ships in every wheel, must name each of them. The check needs the
source tree (a checkout or the sdist) and skips when the tests run from a copy elsewhere.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HEADER = re.compile(r"(?i)\b(?:derived from|port of|ports) ([A-Za-z][\w.-]*[\w])")


def test_notice_names_every_ported_project() -> None:
    notice = ROOT / "NOTICE"
    trees = [ROOT / "crates", ROOT / "python" / "pyroid"]
    if not notice.is_file() or not all(t.is_dir() for t in trees):
        pytest.skip("needs the source tree (a checkout or the sdist)")
    text = notice.read_text(encoding="utf-8").lower()
    missing: dict[str, str] = {}
    for tree in trees:
        for path in sorted(tree.rglob("*")):
            if path.suffix not in {".rs", ".py", ".pyi"} or "target" in path.parts:
                continue
            with path.open(encoding="utf-8", errors="replace") as f:
                head = f.readline() + f.readline()
            for name in HEADER.findall(head):
                if name.lower() not in text:
                    missing.setdefault(name, str(path.relative_to(ROOT)))
    assert not missing, f"projects ported but not in NOTICE: {missing}"
