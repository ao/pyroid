"""pyroid.dedup behind the ``dedup`` contract: its keep masks become the suite's removed masks."""

from __future__ import annotations

from typing import Any


class PyroidDedup:
    notes = ("pyroid.dedup returns keep masks (True = keep); the adapter inverts them to removed masks",)

    def __init__(self) -> None:
        import pyroid.dedup

        self.mod: Any = pyroid.dedup

    def signatures(self, texts: list[str], config: dict[str, Any]) -> list[list[int] | None]:
        return [
            None if s is None else [int(x) for x in s]
            for s in self.mod.minhash_signatures(list(texts), **config)
        ]

    def duplicates(self, texts: list[str], config: dict[str, Any]) -> list[bool]:
        return [not k for k in self.mod.minhash_dedup(list(texts), **config)]


def make() -> PyroidDedup:
    return PyroidDedup()
