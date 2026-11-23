"""pytest plugin: run packaging 26.3's own test suite with ``pyroid.patch("packaging")`` on.

``python -m pytest -p _packaging_upstream parity/data/packaging-26.3/tests`` (with this directory
on ``sys.path``): packaging's tests then import pyroid's classes as ``packaging.version.Version``
and so on, and packaging's unported modules (ranges, metadata, pylock) run on pyroid's objects.
"""

from __future__ import annotations

import warnings

import pyroid.integrations

with warnings.catch_warnings():
    warnings.simplefilter("ignore", pyroid.integrations.PatchWarning)
    pyroid.integrations.patch("packaging")
