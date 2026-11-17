"""pytest plugin: run langdetect 1.0.9's own tests with ``pyroid.patch("langdetect")`` on.

``python -m pytest -p _langdetect_upstream parity/data/langdetect-1.0.9/tests`` (with this
directory on ``sys.path``). Set ``PYROID_BLOCK_LANGDETECT=1`` to block the installed langdetect,
so the tests import pyroid's classes from the shims.
"""

from __future__ import annotations

import os
import sys
import warnings

if os.environ.get("PYROID_BLOCK_LANGDETECT") == "1":
    sys.modules["langdetect"] = None  # type: ignore[assignment]

import pyroid.integrations

with warnings.catch_warnings():
    warnings.simplefilter("ignore", pyroid.integrations.PatchWarning)
    pyroid.integrations.patch("langdetect")
