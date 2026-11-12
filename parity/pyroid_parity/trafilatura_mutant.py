"""A deliberately broken trafilatura candidate: ``--impl trafilatura=pyroid_parity.trafilatura_mutant``.

    python -m pyroid_parity.trafilatura_mutant [N]   # the suite on N pages per file and N recorded tests
                                                     # (default 25) against it; exit 1 unless every check fails

It runs in the current mode ($PYROID_TRAFILATURA_COMPAT) and wraps whichever trafilatura is installed, so run it
in that mode's group. ``tests/test_harness_trafilatura.py`` calls ``prove()``.

It wraps trafilatura 2.3.1 with one defect per kind of check, each of the sort a port could ship:

* ``deduplicate=True`` is ignored (stateful corpus and batch checks, the repeated-paragraph Hypothesis example);
* ``include_tables=False`` is ignored (``corpus-no_tables``, upstream calls that drop tables);
* RIGHT SINGLE QUOTATION MARK in the output becomes an ASCII apostrophe, and a final period is dropped (every
  corpus, bare and upstream check);
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
from dataclasses import dataclass
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


def _bad_text(text: str) -> str:
    text = text.replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")
    return text.removesuffix(".")


def _mangle(result: Any) -> Any:
    if isinstance(result, str):
        return _bad_text(result)
    if isinstance(result, dict):  # bare_extraction(as_dict=True), 1.11.0's default
        return {
            k: _bad_text(v) if k in ("text", "raw_text") and isinstance(v, str) else v
            for k, v in result.items()
        }
    if result is not None and hasattr(result, "text"):
        for attr in ("text", "raw_text"):
            v = getattr(result, attr, None)
            if isinstance(v, str):
                setattr(result, attr, _bad_text(v))
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


#: Pages per corpus file (and recorded tests) the proof runs on: enough for every defect to show.
PROOF_LIMIT = 25


def expected_checks(mode: str | None = None) -> set[str]:
    """Every check the suite records against a non-reference candidate where the mode's trafilatura is installed."""
    from pyroid_parity import specs

    mode = mode or specs.trafilatura_mode()
    out = {"inputs", "upstream", "upstream-delegated", "api-signature", "api-batch", "api-reset"}
    for o in specs.TRAFILATURA_OPTION_SETS_BY_MODE[mode]:
        out |= {f"corpus-{o}", f"batch-{o}", f"hypothesis-{o}"}
    out |= {f"bare-{o}" for o in specs.TRAFILATURA_BARE_SETS_BY_MODE[mode]} | {"hypothesis-bare"}
    for g in ("delegated", "edge", "native"):
        out.add(f"{g}-result")
    return out | {"delegated-warning", "edge-warning", "native-no-fallback"}


@dataclass
class Proof:
    """The suite's verdicts on the mutant: ``ok`` when every check it can fail failed."""

    mode: str
    returncode: int
    checks: dict[str, Any]
    not_caught: list[str]
    not_run: list[str]
    output: str

    @property
    def ok(self) -> bool:
        return bool(self.checks) and not self.not_caught and not self.not_run and self.returncode == 1


def prove(limit: int = PROOF_LIMIT) -> Proof:
    """Run tests/test_trafilatura.py against this mutant in a subprocess, in the current mode."""
    from pyroid_parity import specs

    parity = Path(__file__).resolve().parents[1]
    mode = specs.trafilatura_mode()
    with tempfile.TemporaryDirectory(prefix="pyroid-traf-mutant-") as d:
        report = Path(d) / "parity-report.json"
        env = {
            **os.environ,
            specs.TRAFILATURA_COMPAT_ENV: mode,
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
            str(limit),
            "--parity-report",
            str(report),
            "-p",
            "no:cacheprovider",
            "-q",
            "--no-header",
        ]
        proc = subprocess.run(cmd, cwd=parity, env=env, capture_output=True, text=True, check=False)
        output = proc.stdout[-4000:] + proc.stderr[-4000:]
        if not report.exists():
            return Proof(mode, proc.returncode, {}, [], sorted(expected_checks(mode)), output)
        checks = json.loads(report.read_text("utf-8"))["suites"]["trafilatura"]["checks"]
    not_caught = [
        n
        for n, c in sorted(checks.items())
        if c["status"] != "fail" and n not in EXEMPT and not n.startswith("upstream-hash-seed")
    ]
    not_run = sorted(expected_checks(mode) - set(checks))
    return Proof(mode, proc.returncode, checks, not_caught, not_run, output)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    p = prove(int(args[0]) if args else PROOF_LIMIT)
    if not p.checks:
        print(p.output)
    for name, c in sorted(p.checks.items()):
        verdict = (
            "exempt"
            if name in EXEMPT
            else "NOT CAUGHT"
            if name in p.not_caught
            else "ok (failed, as it must)"
        )
        print(f"{name:32} {c['status']:5} same={c['same']:<7} diff={c['diff']:<7} {verdict}")
    for name in p.not_run:
        print(f"{name:32} -     not recorded                 NOT RUN")
    print(
        f"mode {p.mode}; pytest exit {p.returncode}; {len(p.checks)} checks, not caught: {p.not_caught}, not run: {p.not_run}"
    )
    return 0 if p.ok else 1


if __name__ == "__main__":
    sys.exit(main())
