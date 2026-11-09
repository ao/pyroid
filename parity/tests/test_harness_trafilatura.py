"""The trafilatura suite has teeth: ``pyroid_parity.trafilatura_mutant``, a deliberately broken candidate, must
fail every check it can fail, in the mode checked ($PYROID_TRAFILATURA_COMPAT, default 2.3.1).

The mutant wraps the installed trafilatura and the suite reads the pinned data, so this skips without either
(the parity-data job runs it after building the data). It runs the suite on 25 pages per file and 25 of the
recorded tests: about two minutes.
"""

from __future__ import annotations

import importlib.metadata as md
import importlib.util

import pytest

from pyroid_parity import data as D
from pyroid_parity import specs

MODE = specs.trafilatura_mode()


def _missing() -> str | None:
    if importlib.util.find_spec("trafilatura") is None:
        return "trafilatura is not installed (the mutant wraps it)"
    if (found := md.version("trafilatura")) != MODE:
        return f"trafilatura {found} is installed; the {MODE} mode needs {MODE} (group {specs.TRAFILATURA_GROUPS[MODE]})"
    files = specs.trafilatura_files(MODE)
    for rel in (*specs.TRAFILATURA_CORPORA, files["ref"], files["calls"], files["blobs"], files["tests"]):
        if not D.available(rel):
            return f"{rel} is not available (bench/fetch.py, python -m pyroid_parity.refs)"
    return None


@pytest.mark.timeout(1800)
def test_mutant_fails_every_check() -> None:
    if why := _missing():
        pytest.skip(why)
    from pyroid_parity import trafilatura_mutant as M

    p = M.prove()
    assert p.checks, p.output
    assert not p.not_run, (p.not_run, p.output)
    assert not p.not_caught, {name: p.checks[name] for name in p.not_caught}
    assert p.returncode == 1, p.output
    assert {name for name, c in p.checks.items() if c["status"] != "fail"} <= M.EXEMPT
