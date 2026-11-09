"""A deliberately broken trafilatura candidate: ``--impl trafilatura=pyroid_parity.trafilatura_mutant``.

    python -m pyroid_parity.trafilatura_mutant [N]   # the suite on N pages per file (default 50) against it;
                                                     # exit 1 unless every check fails

It wraps trafilatura 2.3.1 with one defect per kind of check, each of the sort a port could ship:

* ``deduplicate=True`` is ignored (stateful corpus and batch checks, the repeated-paragraph Hypothesis example);
* ``include_tables=False`` is ignored (``corpus-no_tables``, upstream calls that drop tables);
* RIGHT SINGLE QUOTATION MARK in the output becomes an ASCII apostrophe (every corpus and bare check);
* a str with a lone surrogate or a NUL returns None (only generated input has those: ``hypothesis-*``);
* lxml trees raise TypeError instead of being handed to trafilatura (``edge-tree_input``, ``upstream-delegated``);
* delegated options go to trafilatura without a FallbackWarning, and markdown comes back as txt (``delegated-*``);
* ``max_tree_size`` returns None instead of raising, ``target_language`` warns twice (``edge-*``);
* ``favor_recall`` warns as if it were delegated (``native-no-fallback``);
* ``extract_batch`` loses its last result and takes ``threads`` positionally, ``extract`` takes everything after
  the input by keyword only, ``reset_caches`` has a parameter (``batch-*``, ``api-*``).

Not provable with a mutant: ``inputs`` (data integrity), ``delegated-no-oracle`` (it needs an environment
without trafilatura, where this wrapper cannot run) and ``api-shim`` (it checks ``pyroid.integrations``).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import trafilatura
from trafilatura.meta import reset_caches as _trafilatura_reset


class FallbackWarning(RuntimeWarning):
    """Stands in for pyroid.FallbackWarning (same qualified name, which is what the suite matches)."""


FallbackWarning.__module__ = "pyroid"

_SURROGATE_OR_NUL = re.compile("[\ud800-\udfff\x00]")
_DELEGATED_FLAGS = (
    "include_formatting",
    "include_links",
    "include_images",
    "with_metadata",
    "only_with_metadata",
)


def _delegated(fn: str, kw: dict[str, Any]) -> bool:
    fmt = kw.get("output_format", "txt" if fn == "extract" else "python")
    if fmt != ("txt" if fn == "extract" else "python"):
        return True
    return any(kw.get(k) for k in (*_DELEGATED_FLAGS, "url_blacklist", "author_blacklist"))


def _mangle(result: Any) -> Any:
    fix = "\N{RIGHT SINGLE QUOTATION MARK}"
    if isinstance(result, str):
        return result.replace(fix, "'")
    if result is not None and hasattr(result, "text"):
        for attr in ("text", "raw_text"):
            v = getattr(result, attr, None)
            if isinstance(v, str):
                setattr(result, attr, v.replace(fix, "'"))
    return result


def _call(fn: str, filecontent: Any, kw: dict[str, Any]) -> Any:
    real = getattr(trafilatura, fn)
    if _delegated(fn, kw):
        if kw.get("output_format") == "markdown":
            kw = {**kw, "output_format": "txt"}
        return real(filecontent, **kw)  # no FallbackWarning
    if not isinstance(filecontent, (str, bytes)) and filecontent is not None:
        raise TypeError("the mutant takes str or bytes")
    if kw.get("max_tree_size"):
        return None
    if kw.get("target_language"):
        for _ in range(2):
            warnings.warn("target_language: delegating", FallbackWarning, stacklevel=3)
    if kw.get("favor_recall"):
        warnings.warn("favor_recall: delegating", FallbackWarning, stacklevel=3)
    if isinstance(filecontent, str) and _SURROGATE_OR_NUL.search(filecontent):
        return None
    if kw.get("deduplicate") is True:
        kw = {**kw, "deduplicate": False}
    if kw.get("include_tables") is False:
        kw = {**kw, "include_tables": True}
    return _mangle(real(filecontent, **kw))


def extract(filecontent: Any, **kwargs: Any) -> str | None:
    return _call("extract", filecontent, kwargs)


def bare_extraction(filecontent: Any, **kwargs: Any) -> Any:
    return _call("bare_extraction", filecontent, kwargs)


def extract_batch(htmls: list[Any], threads: int | None = None, **kwargs: Any) -> list[str | None]:
    out = [extract(h, **kwargs) for h in htmls]
    return [*out[:-1], None] if out else out


def reset_caches(collect: bool = True) -> None:
    _trafilatura_reset()


# ------------------------------------------------------------------------------------------------ the proof

MODULE = "pyroid_parity.trafilatura_mutant"  # not __name__: that is "__main__" under python -m
#: Checks a mutant cannot fail (see the module docstring).
EXEMPT = {"inputs", "delegated-no-oracle", "api-shim"}


def expected_checks() -> set[str]:
    """Every check the suite records against a non-reference candidate where trafilatura is installed."""
    from pyroid_parity import specs

    out = {"inputs", "upstream", "upstream-delegated", "api-signature", "api-batch", "api-reset"}
    for o in specs.TRAFILATURA_OPTION_SETS:
        out |= {f"corpus-{o}", f"batch-{o}", f"hypothesis-{o}"}
    out |= {f"bare-{o}" for o in specs.TRAFILATURA_BARE_SETS} | {"hypothesis-bare"}
    for g in ("delegated", "edge", "native"):
        out.add(f"{g}-result")
    return out | {"delegated-warning", "edge-warning", "native-no-fallback"}


def main(argv: list[str] | None = None) -> int:
    parity = Path(__file__).resolve().parents[1]
    args = argv if argv is not None else sys.argv[1:]
    limit = args[0] if args else "50"
    with tempfile.TemporaryDirectory(prefix="pyroid-traf-mutant-") as d:
        report = Path(d) / "parity-report.json"
        env = {
            **os.environ,
            "HYPOTHESIS_PROFILE": "dev",
            "HYPOTHESIS_STORAGE_DIRECTORY": str(Path(d) / "hyp"),
        }
        env.pop("PYTEST_ADDOPTS", None)
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_trafilatura.py",
            "--impl",
            f"trafilatura={MODULE}",
            "--parity-limit",
            limit,
            "--parity-report",
            str(report),
            "-p",
            "no:cacheprovider",
            "-q",
            "--no-header",
        ]
        proc = subprocess.run(cmd, cwd=parity, env=env, capture_output=True, text=True, check=False)
        if not report.exists():
            print(proc.stdout[-4000:], proc.stderr[-4000:])
            return 2
        checks = json.loads(report.read_text("utf-8"))["suites"]["trafilatura"]["checks"]
    bad = []
    for name, c in sorted(checks.items()):
        verdict = "ok (failed, as it must)" if c["status"] == "fail" else "NOT CAUGHT"
        if name in EXEMPT or name.startswith("upstream-hash-seed"):
            verdict = "exempt"
        elif c["status"] != "fail":
            bad.append(name)
        print(f"{name:32} {c['status']:5} same={c['same']:<7} diff={c['diff']:<7} {verdict}")
    missing = sorted(expected_checks() - set(checks))
    for name in missing:
        print(f"{name:32} -     not recorded                 NOT RUN")
    print(f"pytest exit {proc.returncode}; {len(checks)} checks, not caught: {bad}, not run: {missing}")
    return 1 if bad or missing or proc.returncode != 1 else 0


if __name__ == "__main__":
    sys.exit(main())
