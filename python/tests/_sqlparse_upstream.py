"""pytest plugin: run sqlparse 0.6.1's own test suite with ``pyroid.patch("sqlparse")`` on.

``python -m pytest -p _sqlparse_upstream parity/data/sqlparse-0.6.1/tests`` (with this
directory on ``sys.path``). With sqlparse installed, its ``parse``, ``parsestream``, ``format``,
``split`` and ``lexer.tokenize`` are pyroid's; without it, ``sqlparse`` and its submodules are
pyroid.sqlparse's own.
"""

from __future__ import annotations

import os
import warnings

import pyroid.integrations
from pyroid.integrations import _patch

# test_cli runs `python -m sqlparse.cli`: patch child processes too, as `pyroid run` does.
os.environ[_patch.ENV_VAR] = "sqlparse"
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (_patch.HOOK_DIR, os.environ.get("PYTHONPATH", "")) if p
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", pyroid.integrations.PatchWarning)
    pyroid.integrations.patch("sqlparse")
