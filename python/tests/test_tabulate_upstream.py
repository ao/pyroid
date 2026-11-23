"""tabulate 0.10.0's own test suite, run against pyroid.tabulate.

``parity/data/tabulate-0.10.0/test`` is tabulate's ``test/`` directory, unmodified; the
``conftest.py`` next to it patches ``tabulate`` with pyroid's, and ``cli/tabulate/__init__.py``
stands in for the script ``test_cli.py`` runs. The suite runs in a child pytest; every test must
pass or be skipped for a missing optional dependency, as it does against tabulate itself.
Without ``parity/`` (an sdist) the module is skipped.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "parity" / "data" / "tabulate-0.10.0"
if not (ROOT / "test" / "test_output.py").is_file():
    pytest.skip("parity/data/tabulate-0.10.0 is not in this checkout", allow_module_level=True)

# tabulate skips one test unconditionally; the others need numpy, pandas and wcwidth.
ALWAYS_SKIPPED = "It detects all values as floats but there are strings and integers."
OPTIONAL = ("numpy", "pandas", "wcwidth")


def _run(impl: str) -> str:
    env = dict(os.environ, PYROID_TABULATE_IMPL=impl, PYTHONSAFEPATH="1")
    env.pop("PYTEST_ADDOPTS", None)
    args = ["-q", "-rs", "-p", "no:cacheprovider", "-c", str(ROOT / "pytest.ini")]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *args, "--rootdir", str(ROOT), str(ROOT / "test")],
        cwd=ROOT / "cli",
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    return proc.stdout + proc.stderr


@pytest.mark.thread_unsafe  # a child process
def test_upstream_suite_passes() -> None:
    out = _run("pyroid")
    summary = out.strip().splitlines()[-1]
    assert " passed" in summary, out[-4000:]
    assert "failed" not in summary, out[-6000:]
    assert "error" not in summary, out[-6000:]
    for line in out.splitlines():
        if line.startswith("SKIPPED"):
            assert ALWAYS_SKIPPED in line or any(d in line for d in OPTIONAL), line
    match = re.search(r"(\d+) passed", summary)
    assert match is not None
    if all(importlib.util.find_spec(d) for d in OPTIONAL):
        assert int(match.group(1)) == 318, summary
