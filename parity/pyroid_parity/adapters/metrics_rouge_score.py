"""The ``metrics`` reference: rouge-score's ``rouge_scorer`` and sacrebleu, behind the suite contract.

sacrebleu is imported on first use only: ``import sacrebleu`` loads lxml (via ``sacrebleu.dataset``), which
re-enables the GIL on free-threaded builds, so ROUGE-only runs keep the GIL off.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
from typing import Any

from pyroid_parity.metrics_cases import Metrics


class ReferenceMetrics(Metrics):
    def __init__(self) -> None:
        super().__init__(importlib.import_module("rouge_score.rouge_scorer"), None)
        vs = {}
        for d in ("rouge-score", "sacrebleu", "nltk"):
            try:
                vs[d] = md.version(d)
            except md.PackageNotFoundError:
                vs[d] = "not installed"
        self.notes = [", ".join(f"{d} {v}" for d, v in vs.items())]
        self._sacrebleu: Any = None

    @property  # type: ignore[override]
    def sacrebleu(self) -> Any:
        if self._sacrebleu is None:
            self._sacrebleu = importlib.import_module("sacrebleu")
        return self._sacrebleu

    @sacrebleu.setter
    def sacrebleu(self, value: Any) -> None:
        self._sacrebleu = value


def make() -> ReferenceMetrics:
    return ReferenceMetrics()
