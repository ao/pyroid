"""The warc suite has teeth: ``pyroid_parity.warc_mutant`` with every defect on must fail each check it targets.

It wraps warcio and datatrove and reads the pinned inputs, so this skips without them (the parity-data job runs it
after building the data). It runs the suite on 20 records per Common Crawl slice, without the ``big`` synthetic
family.
"""

from __future__ import annotations

import importlib.util

import pytest

from pyroid_parity import warc_corpus as W


def _missing() -> str | None:
    for mod in ("warcio", "datatrove", "cchardet"):
        if importlib.util.find_spec(mod) is None:
            return f"{mod} is not installed (the mutant wraps warcio and datatrove; parity group `warc`)"
    if gone := W.missing():
        return f"{len(gone)} pinned warc inputs are missing (bench/fetch.py --only warc)"
    return None


@pytest.mark.timeout(3600)
def test_mutant_fails_every_check() -> None:
    if why := _missing():
        pytest.skip(why)
    from pyroid_parity import warc_mutant as M

    p, missed = M.prove()
    assert p.checks, p.output
    assert not missed, {name: p.checks[name] for name in missed}
    assert p.returncode == 1, p.output
