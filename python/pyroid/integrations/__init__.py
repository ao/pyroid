"""Plug pyroid into the tools people already run.

Nothing here is imported by ``import pyroid``, and nothing changes behaviour
until you ask for it.

- :mod:`pyroid.integrations.datatrove`: drop-in datatrove pipeline steps
  (``FTFYFormatter``, ``LanguageFilter``, the Gopher/C4/FineWeb filters,
  ``MinhashDedupSignature``) that subclass datatrove's own classes and give
  the same output.
- :mod:`pyroid.integrations.recipes`: the FineWeb cleaning chain over Arrow
  batches (what ``pyroid clean --recipe fineweb`` runs).
- :mod:`pyroid.integrations.hf`: helpers for Hugging Face ``datasets``
  (``Dataset.map(batched=True)`` / ``.filter`` on Arrow batches).
- :func:`patch` / :func:`unpatch` / :func:`patching`: explicit, reversible ``sys.modules`` shims
  so that existing ``import ftfy`` / ``import fasttext`` code gets pyroid's
  drop-ins without edits (also ``pyroid run -p ftfy,fasttext script.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyroid.integrations._patch import PatchWarning, patch, patched, patching, run, unpatch

__all__ = ["PatchWarning", "patch", "patched", "patching", "run", "unpatch"]

_PATCH_NAMES = frozenset(__all__)


def __getattr__(name: str) -> Any:
    # Lazy: importing pyroid.integrations (e.g. for .datatrove) never loads the patch machinery.
    if name in _PATCH_NAMES:
        from pyroid.integrations import _patch

        return getattr(_patch, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
