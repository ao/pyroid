"""pyroid.filters behind the ``filters`` contract: ``apply()`` with one config gives keep, reason and text.

pyroid's per-filter functions return keep masks only, so the adapter runs ``pyroid.filters.apply(texts, [Config])``
and turns its ``FilterResult`` into the suite's per-document decisions. A batch that raises is re-run one document
at a time (``GenericFilters.run``), so the exception lands on the document that caused it, as in datatrove.
"""

from __future__ import annotations

from typing import Any

from pyroid_parity.impl import Decision, GenericFilters


class PyroidFilters(GenericFilters):
    notes = ("pyroid.filters.apply(texts, [Config(**config)]) -> keep, reason, text",)

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

    def _call(self, name: str, texts: list[str], config: dict[str, Any]) -> list[Decision]:
        r = self.mod.apply(list(texts), [self.configs[name](**config)])
        return [
            ("ok", bool(k), reason, out if out != text else None)
            for k, reason, out, text in zip(r.keep, r.reason, r.text, texts, strict=True)
        ]


def make() -> PyroidFilters:
    return PyroidFilters()
