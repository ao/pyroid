"""pyroid.dedup behind the ``dedup`` contract: its keep masks become the suite's removed masks.

``config`` may carry ``language`` (datatrove ``MinhashDedupSignature``'s); a language pyroid does not tokenize
natively (``NotImplementedError``) is retried with ``python_tokenizer=True`` (datatrove's own word tokenizer, in
Python; its ``pyroid.FallbackWarning`` is silenced).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any


class PyroidDedup:
    notes = (
        "pyroid.dedup returns keep masks (True = keep); the adapter inverts them to removed masks",
        "languages without a native tokenizer: python_tokenizer=True",
    )

    def __init__(self) -> None:
        import pyroid.dedup

        self.mod: Any = pyroid.dedup
        self.python_languages: set[str] = set()

    def _call(self, fn: Callable[..., Any], texts: list[str], config: dict[str, Any]) -> Any:
        language = str(config.get("language", "eng"))
        if language not in self.python_languages:
            try:
                return fn(list(texts), **config)
            except NotImplementedError:
                self.python_languages.add(language)
        import pyroid

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pyroid.FallbackWarning)
            return fn(list(texts), **config, python_tokenizer=True)

    def signatures(self, texts: list[str], config: dict[str, Any]) -> list[list[int] | None]:
        return [
            None if s is None else [int(x) for x in s]
            for s in self._call(self.mod.minhash_signatures, texts, config)
        ]

    def duplicates(self, texts: list[str], config: dict[str, Any]) -> list[bool]:
        return [not k for k in self._call(self.mod.minhash_dedup, texts, config)]


def make() -> PyroidDedup:
    return PyroidDedup()
