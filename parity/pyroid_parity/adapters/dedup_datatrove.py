"""datatrove 0.10.1 MinHash behind the ``dedup`` contract (signature maths + the real 3-stage dedup chain)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from pyroid_parity import specs


class DatatroveDedup:
    notes = (
        "reference: MinhashDedupSignature.get_shingles/get_signature; Signature->Buckets->Cluster executors",
    )

    def __init__(self) -> None:
        self._step: Any = None

    def _sig_step(self, config: dict[str, Any]) -> Any:
        assert config == specs.MINHASH_CONFIG, (
            "the datatrove adapter only implements the pinned FineWeb config"
        )
        if self._step is None:
            from datatrove.pipeline.dedup import MinhashDedupSignature

            from pyroid_parity.refs import minhash_config

            self._step = MinhashDedupSignature(
                output_folder=tempfile.mkdtemp(prefix="pyroid-sig-"), config=minhash_config()
            )
        return self._step

    def signatures(self, texts: list[str], config: dict[str, Any]) -> list[list[int] | None]:
        step = self._sig_step(config)
        out: list[list[int] | None] = []
        for t in texts:
            sh = step.get_shingles(t)
            out.append(None if sh.size == 0 else [int(x) for b in step.get_signature(sh) for x in b])
        return out

    def duplicates(self, texts: list[str], config: dict[str, Any]) -> list[bool]:
        from pyroid_parity.refs import datatrove_minhash_removed

        assert config == specs.MINHASH_CONFIG
        with tempfile.TemporaryDirectory(prefix="pyroid-minhash-") as tmp:
            return datatrove_minhash_removed(texts, [str(i) for i in range(len(texts))], Path(tmp))


def make() -> DatatroveDedup:
    return DatatroveDedup()
