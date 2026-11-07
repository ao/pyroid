"""ftfy with ``fix_surrogates`` forced off, and no ``fix_text_batch``: ``--impl ftfy=pyroid_parity.mutants.ftfy_nosurrogates``.

A candidate like this passes ftfy's whole golden corpus; only generated surrogate input catches it.
The API (signatures, ``TextFixerConfig``) mirrors ftfy 6.3.1 exactly, so the API-mirror checks pass.
"""

from __future__ import annotations

from typing import Any

import ftfy
from ftfy import TextFixerConfig

__all__ = ["TextFixerConfig", "fix_encoding", "fix_text"]


def fix_text(text: str, config: TextFixerConfig | None = None, **kwargs: Any) -> str:
    return ftfy.fix_text(text, config, **{**kwargs, "fix_surrogates": False})


def fix_encoding(text: str, config: TextFixerConfig | None = None, **kwargs: Any) -> str:
    return ftfy.fix_encoding(text, config, **{**kwargs, "fix_surrogates": False})
