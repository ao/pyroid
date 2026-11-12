"""The ftfy suite has teeth: a candidate with ``fix_surrogates`` forced off must fail it.

Such a candidate passes ftfy's entire golden corpus, and the pinned parquet corpora cannot hold
surrogates (Arrow strings are UTF-8), so only generated input catches it. Both are asserted here.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import ftfy
import pytest
from hypothesis import Phase, given, seed, settings

from pyroid_parity import strategies as S
from pyroid_parity.mutants import ftfy_nosurrogates

PARITY = Path(__file__).resolve().parents[1]
MUTANT = "pyroid_parity.mutants.ftfy_nosurrogates"
SURROGATE = re.compile("[\ud800-\udfff]")
SURROGATE_REPR = re.compile(r"\\ud[89a-f][0-9a-f]{2}")


def test_mutant_fails_the_ftfy_suite(tmp_path: Path) -> None:
    report = tmp_path / "parity-report.json"
    env = {
        **os.environ,
        "HYPOTHESIS_PROFILE": "dev",
        "HYPOTHESIS_STORAGE_DIRECTORY": str(tmp_path / "hypothesis"),
    }
    env.pop("PYTEST_ADDOPTS", None)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_ftfy.py",
        "--impl",
        f"ftfy={MUTANT}",
        "-p",
        "no:cacheprovider",
        "--parity-report",
        str(report),
        "--parity-limit",
        "50",
        "-q",
        "--no-header",
    ]
    proc = subprocess.run(cmd, cwd=PARITY, env=env, capture_output=True, text=True, timeout=600, check=False)
    tail = proc.stdout[-3000:] + proc.stderr[-2000:]
    assert proc.returncode == pytest.ExitCode.TESTS_FAILED, tail

    suite = json.loads(report.read_text("utf-8"))["suites"]["ftfy"]
    checks = suite["checks"]
    assert suite["status"] == "fail", checks
    assert suite["candidate"]["module"] == MUTANT
    assert not suite["candidate"]["is_reference"]
    hyp = checks["hypothesis-fix_text"]
    assert hyp["status"] == "fail"
    assert hyp["diff"] > 0
    assert any(SURROGATE_REPR.search(str(e["input"])) for e in hyp["examples"]), hyp["examples"]
    # The golden corpus alone does not catch this mutant, nor can the parquet corpora.
    assert checks["golden-fix_text"]["status"] == "pass", checks["golden-fix_text"]
    assert checks["golden-fix_encoding"]["status"] == "pass", checks["golden-fix_encoding"]
    for name in ("corpus-dev", "corpus-adversarial"):
        if name in checks:
            assert checks[name]["status"] == "pass", checks[name]
    assert checks["api-signature"]["status"] == "pass"  # the mutant mirrors ftfy's API


@pytest.mark.parametrize("seed_", [0, 1, 2])
def test_generated_input_alone_finds_the_mutant(seed_: int) -> None:
    """No explicit examples: generation (no shrinking) must hit a surrogate input within 200 examples."""
    found: list[str] = []

    @seed(seed_)
    @settings(max_examples=200, database=None, deadline=None, phases=[Phase.generate])
    @given(S.documents())
    def prop(text: str) -> None:
        if ftfy_nosurrogates.fix_text(text) != ftfy.fix_text(text):
            found.append(text)
            raise AssertionError(text)

    with pytest.raises(AssertionError):
        prop()
    assert found
    assert SURROGATE.search(found[0]), found[0]
