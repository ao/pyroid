# Derived from NeMo Curator 1.4.0 (Apache-2.0), © NVIDIA CORPORATION; modified.
"""NeMo Curator 1.4.0's CPU text heuristics in Rust, with the same scores (``pyroid.nemo``).

One function per heuristic ``DocumentFilter`` (its ``score_document``) and per string
``DocumentModifier`` (its ``modify_document``), plus the fastText stages' scoring. NeMo
Curator itself is not needed. Types and docs are in ``nemo.pyi``; the drop-in pipeline
stages are in :mod:`pyroid.integrations.nemo_curator`.
"""

from __future__ import annotations

from typing import Any

from pyroid._pyroid import nemo as _native

#: The NeMo Curator release whose results this module reproduces.
NEMO_CURATOR_VERSION = "1.4.0"

# --- filter scores and fastText helpers -------------------------------------------------------

# --- string modifiers ---------------------------------------------------------------------------

remove_boilerplate_strings = _native.remove_boilerplate_strings
remove_markdown = _native.remove_markdown
normalize_newlines = _native.normalize_newlines
remove_quotation = _native.remove_quotation
remove_urls = _native.remove_urls
remove_lines = _native.remove_lines

# Python slice indices beyond this are the same as this, for any text pyroid can hold.
_INDEX_LIMIT = 1 << 62


def _slice_bound(value: object) -> tuple[int | None, str | None]:
    # Slicer's own isinstance checks, in its order (bool is an int).
    if isinstance(value, int):
        return max(-_INDEX_LIMIT, min(_INDEX_LIMIT, value)), None
    if isinstance(value, str):
        return None, value
    return None, None


def slice_text(
    x: Any,
    /,
    left: int | str | None = 0,
    right: int | str | None = None,
    include_left: bool = True,
    include_right: bool = True,
    strip: bool = True,
    *,
    threads: int | None = None,
) -> Any:
    """Slice each document as ``Slicer(...).modify_document`` does (see ``nemo.pyi``)."""
    li, ls = _slice_bound(left)
    ri, rs = _slice_bound(right)
    return _native._slice_text(
        x, li, ls, ri, rs, bool(include_left), bool(include_right), bool(strip), threads=threads
    )


def fasttext_label(x: Any, /, label: object, *, threads: int | None = None) -> Any:
    """Prefix each document with ``label`` as ``FastTextLabelModifier`` does (see ``nemo.pyi``)."""
    # FastTextLabelModifier builds f"{label} {text}".
    return _native._fasttext_label(x, format(label, ""), threads=threads)


__all__ = [
    "NEMO_CURATOR_VERSION",
    # filter scores and fastText helpers
    # string modifiers
    "fasttext_label",
    "normalize_newlines",
    "remove_boilerplate_strings",
    "remove_lines",
    "remove_markdown",
    "remove_quotation",
    "remove_urls",
    "slice_text",
]
