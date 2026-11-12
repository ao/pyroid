"""pyroid.filters behind the ``filters`` contract: ``apply()`` with one config gives keep, reason and text.

pyroid's per-filter functions return keep masks only, so the adapter runs ``pyroid.filters.apply(texts, [Config])``
and turns its ``FilterResult`` into the suite's per-document decisions. A batch that raises is re-run one document
at a time (``GenericFilters.run``), so the exception lands on the document that caused it, as in datatrove.

A ``language`` pyroid does not tokenize natively (it raises ``NotImplementedError``) is retried with
``python_tokenizer=True``, pyroid's explicit opt-in to datatrove's own word tokenizer in Python (its
``pyroid.FallbackWarning`` is silenced: it says which path ran, not what the result is).
"""

from __future__ import annotations

import warnings
from typing import Any

from pyroid_parity.impl import Decision, GenericFilters


class PyroidFilters(GenericFilters):
    notes = (
        "pyroid.filters.apply(texts, [Config(**config)]) -> keep, reason, text",
        "languages without a native tokenizer: apply(..., python_tokenizer=True)",
    )

    def __init__(self) -> None:
        import pyroid.filters

        super().__init__(pyroid.filters)
        f = self.mod
        self.configs = {
            "gopher_repetition": f.GopherRepetition,
            "gopher_quality": f.GopherQuality,
            "c4_quality": f.C4Quality,
            "fineweb_quality": f.FineWebQuality,
        }
        self.python_languages: set[str] = set()

    def _apply(self, texts: list[str], cfg: Any, language: str) -> Any:
        if language not in self.python_languages:
            try:
                return self.mod.apply(texts, [cfg])
            except NotImplementedError:
                self.python_languages.add(language)
        import pyroid

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pyroid.FallbackWarning)
            return self.mod.apply(texts, [cfg], python_tokenizer=True)

    def _call(self, name: str, texts: list[str], config: dict[str, Any]) -> list[Decision]:
        cfg = self.configs[name](**config)
        r = self._apply(list(texts), cfg, str(config.get("language", "eng")))
        return [
            ("ok", bool(k), reason, out if out != text else None)
            for k, reason, out, text in zip(r.keep, r.reason, r.text, texts, strict=True)
        ]


def make() -> PyroidFilters:
    return PyroidFilters()
