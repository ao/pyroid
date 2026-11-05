"""Select an implementation by module path and wrap it in the suite's normalized contract.

``--impl ftfy`` / ``--impl pyroid.ftfy`` / ``--impl filters=pyroid.filters`` all resolve here. Resolution:

1. ``suite=module`` is explicit. A bare ``module`` maps to a suite by its last dotted component
   (``pyroid.ftfy`` -> ``ftfy``), and ``datatrove`` maps to both ``filters`` and ``dedup``.
2. If ``pyroid_parity.adapters.<suite>_<module with dots as underscores>`` exists it wraps the module (this is
   how the reference packages, whose APIs differ from the contract, are plugged in).
3. Otherwise the module itself must satisfy the suite contract below (``Generic*`` adapters).

Suite contracts (what a product module must export to plug in with no adapter):

ftfy      ``fix_text(text, config=None, **kwargs) -> str``, ``fix_encoding(text, config=None, **kwargs) -> str``,
          optional ``fix_text_batch(texts, **kwargs) -> list[str]``.
fasttext  ``load_model(path) -> model`` with ``model.predict(text, k=1, threshold=0.0) -> (labels, probs)``
          exactly like fastText's Python API (labels keep the ``__label__`` prefix).
filters   one callable per filter name in ``pyroid_parity.specs.FILTER_CASES``:
          ``name(texts: list[str], **config) -> (keep, reasons) | (keep, reasons, texts_out)``, or an object
          with ``.keep`` / ``.reason`` (/ ``.text``) sequences. ``texts_out`` is required for filters that rewrite
          the text (``c4_quality``). ``reasons[i]`` is datatrove's reason string, or ``None`` when kept.
dedup     ``minhash_signatures(texts, **config) -> list[list[int] | None]`` (``None`` = no shingles) and
          ``minhash_dedup(texts, **config) -> list[bool]`` (True = removed), with ``config`` from
          ``pyroid_parity.specs.MINHASH_CONFIG``.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import importlib.util
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

SUITES = ("ftfy", "fasttext", "filters", "dedup")
REFERENCE = {"ftfy": "ftfy", "fasttext": "fasttext", "filters": "datatrove", "dedup": "datatrove"}
# Distribution that provides a module, for version reporting when it differs from the module name.
DIST_HINT = {"fasttext": ("fasttext-predict", "fasttext-predict-v2", "fasttext", "fasttext-wheel")}


class ImplError(ValueError):
    pass


def parse_specs(specs: Sequence[str]) -> dict[str, str]:
    """``["pyroid.ftfy", "filters=x.y"]`` -> ``{"ftfy": "pyroid.ftfy", "filters": "x.y"}``; others default."""
    chosen = dict(REFERENCE)
    for spec in specs:
        suite, sep, module = spec.partition("=")
        if sep:
            suites = [suite]
        else:
            module = spec
            last = module.rsplit(".", 1)[-1]
            suites = ["filters", "dedup"] if module == "datatrove" else [last]
        for s in suites:
            if s not in SUITES:
                raise ImplError(
                    f"--impl {spec!r}: cannot tell which suite it is for; use SUITE=MODULE, SUITE in {SUITES}"
                )
            chosen[s] = module
    return chosen


def version_of(module_name: str) -> str | None:
    top = module_name.split(".")[0]
    for dist in (*DIST_HINT.get(top, ()), top, top.replace("_", "-")):
        try:
            return md.version(dist)
        except md.PackageNotFoundError:
            continue
    mod = importlib.import_module(module_name)
    return getattr(mod, "__version__", None)


def dist_of(module_name: str) -> str | None:
    top = module_name.split(".")[0]
    for dist in (*DIST_HINT.get(top, ()), top, top.replace("_", "-")):
        try:
            md.version(dist)
            return dist
        except md.PackageNotFoundError:
            continue
    return None


@dataclass
class Impl:
    suite: str
    module: str
    adapter: str
    obj: Any
    is_reference: bool
    version: str | None = None
    dist: str | None = None
    notes: list[str] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            "spec": self.module,
            "module": self.module,
            "dist": self.dist,
            "version": self.version,
            "adapter": self.adapter,
            "is_reference": self.is_reference,
            "notes": self.notes,
        }


def resolve(suite: str, module: str) -> Impl:
    adapter_name = f"pyroid_parity.adapters.{suite}_{module.replace('.', '_')}"
    if (
        importlib.util.find_spec("pyroid_parity.adapters")
        and importlib.util.find_spec(adapter_name) is not None
    ):
        ad = importlib.import_module(adapter_name)
        obj = ad.make()
        adapter = adapter_name
    else:
        mod = importlib.import_module(module)
        obj = GENERIC[suite](mod)
        adapter = "generic"
    return Impl(
        suite,
        module,
        adapter,
        obj,
        module == REFERENCE[suite],
        version_of(module),
        dist_of(module),
        list(getattr(obj, "notes", [])),
    )


# ---------------------------------------------------------------- generic adapters (the contracts above)


class GenericFtfy:
    def __init__(self, mod: Any) -> None:
        self.mod = mod
        self.fix_text: Callable[..., str] = mod.fix_text
        self.fix_encoding: Callable[..., str] = mod.fix_encoding
        self.has_batch = hasattr(mod, "fix_text_batch")
        self.notes = [] if self.has_batch else ["no fix_text_batch: batch checks map fix_text"]

    def fix_text_batch(self, texts: list[str], **kw: Any) -> list[str]:
        if self.has_batch:
            return list(self.mod.fix_text_batch(texts, **kw))
        return [self.fix_text(t, **kw) for t in texts]


class GenericFasttext:
    def __init__(self, mod: Any) -> None:
        self.mod = mod

    def load(self, path: str) -> Any:
        return self.mod.load_model(path)

    def predict(self, model: Any, text: str, k: int) -> list[tuple[str, float]]:
        labels, probs = model.predict(text, k=k, threshold=0.0)
        return [(str(lbl), float(p)) for lbl, p in zip(labels, probs, strict=True)]


Decision = tuple[Any, ...]  # ("ok", keep, reason, text_out | None) or ("raise", ExceptionTypeName)


class GenericFilters:
    def __init__(self, mod: Any) -> None:
        self.mod = mod

    def _call(self, name: str, texts: list[str], config: dict[str, Any]) -> list[Decision]:
        res = getattr(self.mod, name)(texts, **config)
        if hasattr(res, "keep"):
            keep, reason, out = res.keep, res.reason, getattr(res, "text", None)
        else:
            keep, reason, out = (*res, None)[:3]
        out = [None] * len(texts) if out is None else list(out)
        return [
            ("ok", bool(k), r, o if (o is not None and o != t) else None)
            for k, r, o, t in zip(keep, reason, out, texts, strict=True)
        ]

    def run(self, name: str, texts: list[str], config: dict[str, Any]) -> list[Decision]:
        try:
            return self._call(name, texts, config)
        except Exception:  # noqa: BLE001  localize: per-doc calls tell which doc raised
            out: list[Decision] = []
            for t in texts:
                try:
                    out.extend(self._call(name, [t], config))
                except Exception as e:  # noqa: BLE001
                    out.append(("raise", type(e).__name__))
            return out


class GenericDedup:
    def __init__(self, mod: Any) -> None:
        self.mod = mod

    def signatures(self, texts: list[str], config: dict[str, Any]) -> list[list[int] | None]:
        return [
            None if s is None else [int(x) for x in s] for s in self.mod.minhash_signatures(texts, **config)
        ]

    def duplicates(self, texts: list[str], config: dict[str, Any]) -> list[bool]:
        return [bool(x) for x in self.mod.minhash_dedup(texts, **config)]


GENERIC: dict[str, Callable[[Any], Any]] = {
    "ftfy": GenericFtfy,
    "fasttext": GenericFasttext,
    "filters": GenericFilters,
    "dedup": GenericDedup,
}
