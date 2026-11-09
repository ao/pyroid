"""pyroid for datatrove and NeMo Curator pipelines, Hugging Face ``datasets`` and existing imports.

``import pyroid`` imports none of this, and nothing changes behaviour until you
call it.

- :mod:`pyroid.integrations.datatrove`: 15 drop-in datatrove pipeline steps
  (``FTFYFormatter``, ``LanguageFilter``, the Gopher/C4/FineWeb quality filters,
  ``C4BadWordsFilter``, ``URLFilter``, ``PIIFormatter``,
  ``SymbolLinesFormatter``, ``MinhashDedupSignature`` and the four sentence-dedup
  steps) that subclass datatrove's own classes and give the same output. The
  steps that tokenize run in Rust for every language pyroid tokenizes natively.
- :mod:`pyroid.integrations.nemo_curator`: NeMo Curator 1.4.0's ``ScoreFilter``, ``Score``,
  ``Filter`` and ``Modify`` stages, its 23 heuristic filters, two fastText filters,
  ``UnicodeReformatter`` and string modifiers, subclassing NeMo's classes, with the same output,
  one pyroid call per batch.
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
