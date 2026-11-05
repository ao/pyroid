"""datatrove 0.10.1 filters behind the ``filters`` contract: one ``filter(Document)`` call per doc."""

from __future__ import annotations

import json
from typing import Any

from pyroid_parity import specs
from pyroid_parity.impl import Decision

CLASSES = {fn: cls for fn, cls, _ in specs.FILTER_CASES.values()}


class DatatroveFilters:
    notes = ("reference: datatrove BaseFilter.filter(doc) per document",)

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], Any] = {}

    def _filter(self, name: str, config: dict[str, Any]) -> Any:
        key = (name, json.dumps(config, sort_keys=True))
        if key not in self._cache:
            from datatrove.pipeline import filters as F

            self._cache[key] = getattr(F, CLASSES[name])(**config)
        return self._cache[key]

    def run(self, name: str, texts: list[str], config: dict[str, Any]) -> list[Decision]:
        from datatrove.data import Document
        from datatrove.pipeline.filters.base_filter import get_filter_result

        f = self._filter(name, config)
        out: list[Decision] = []
        for n, text in enumerate(texts):
            doc = Document(text=text, id=str(n))
            try:
                keep, reason = get_filter_result(f.filter(doc))
            except Exception as e:  # noqa: BLE001  exception type is part of parity
                out.append(("raise", type(e).__name__))
                continue
            out.append(("ok", bool(keep), reason, doc.text if doc.text != text else None))
        return out


def make() -> DatatroveFilters:
    return DatatroveFilters()
