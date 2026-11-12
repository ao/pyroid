"""datatrove 0.10.1 MinHash behind the ``dedup`` contract (signature maths + the real 3-stage dedup chain).

``config`` is a ``specs.MINHASH_CONFIG``-shaped dict, optionally with ``language`` (``MinhashDedupSignature``'s
word-tokenizer language, ``"eng"`` by default).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any


class DatatroveDedup:
    notes = (
        "reference: MinhashDedupSignature.get_shingles/get_signature; Signature->Buckets->Cluster executors",
    )

    def __init__(self) -> None:
        self._steps: dict[str, Any] = {}

    def _sig_step(self, config: dict[str, Any]) -> Any:
        key = json.dumps(config, sort_keys=True)
        if key not in self._steps:
            from datatrove.pipeline.dedup import MinhashDedupSignature

            from pyroid_parity.refs import minhash_config

            cfg = dict(config)
            language = cfg.pop("language", "eng")
            self._steps[key] = MinhashDedupSignature(
                output_folder=tempfile.mkdtemp(prefix="pyroid-sig-"),
                config=minhash_config(cfg),
                language=language,
            )
        return self._steps[key]

    def signatures(self, texts: list[str], config: dict[str, Any]) -> list[list[int] | None]:
        step = self._sig_step(config)
        out: list[list[int] | None] = []
        for t in texts:
            sh = step.get_shingles(t)
            out.append(None if sh.size == 0 else [int(x) for b in step.get_signature(sh) for x in b])
        return out

    def duplicates(self, texts: list[str], config: dict[str, Any]) -> list[bool]:
        from pyroid_parity.refs import datatrove_minhash_removed

        cfg = dict(config)
        language = cfg.pop("language", "eng")
        with tempfile.TemporaryDirectory(prefix="pyroid-minhash-") as tmp:
            return datatrove_minhash_removed(
                texts, [str(i) for i in range(len(texts))], Path(tmp), cfg, language
            )


def make() -> DatatroveDedup:
    return DatatroveDedup()
