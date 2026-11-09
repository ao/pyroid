# Derived from NeMo Curator 1.4.0 (Apache-2.0), © NVIDIA CORPORATION; modified.
"""NeMo Curator 1.4.0's CPU text heuristics in Rust, with the same scores (``pyroid.nemo``).

One function per heuristic ``DocumentFilter`` (its ``score_document``) and per string
``DocumentModifier`` (its ``modify_document``), plus the fastText stages' scoring. NeMo Curator itself
is not needed. Types and docs are in ``nemo.pyi``; the drop-in pipeline stages are in
:mod:`pyroid.integrations.nemo_curator`.
"""

from __future__ import annotations

from pyroid._pyroid import nemo as _native

#: The NeMo Curator release whose results this module reproduces.
NEMO_CURATOR_VERSION = "1.4.0"

# --- filter scores and fastText helpers -------------------------------------------------------

# --- string modifiers ---------------------------------------------------------------------------

__all__ = [
    "NEMO_CURATOR_VERSION",
    # filter scores and fastText helpers
    # string modifiers
]
