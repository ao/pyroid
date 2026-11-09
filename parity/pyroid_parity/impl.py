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
metrics   a package with ``rouge_scorer`` (rouge-score's ``RougeScorer``; optional ``score_batch``) and ``sacrebleu``
          (``BLEU``/``CHRF`` with ``corpus_score``/``get_signature``, ``sentence_bleu``/``sentence_chrf``; optional
          ``sentence_bleu_batch``/``sentence_chrf_batch``) submodules; see ``pyroid_parity.metrics_cases``.
json_repair  json_repair 0.64.5's own API: ``repair_json(json_str="", return_objects=False,
          skip_json_loads=False, logging=False, json_fd=None, chunk_length=0, stream_stable=False, strict=False,
          schema=None, schema_repair_mode="standard", **json_dumps_args)``, ``loads``, ``load`` and ``from_file``
          with json_repair's signatures. Results must be identical: ``str`` byte for byte, objects and repair logs
          type-exact (``pyroid_parity.json_repair_compare.same``), exceptions with the same type and message.
datasketch  datasketch 2.0.0's own API: ``MinHash`` (with ``bulk``), ``LeanMinHash``, ``MinHashLSH``,
          ``MinHashLSHForest`` and ``MinHashLSHEnsemble``; optional pyroid-style extensions
          ``minhash_batch(texts, num_perm=, seed=, scheme=, ngram=)`` and ``MinHashLSH.query_batch``.
jieba     jieba 0.42.1's own module API (``lcut``, ``lcut_for_search``, ``tokenize``, ``Tokenizer``, ``dt``);
          a candidate whose ``lcut`` also takes a list of str (one list of words per document) gets batch checks.
trafilatura  trafilatura 2.3.1's ``extract`` and ``bare_extraction`` with its signatures plus one keyword-only
          ``compat="2.3.1"`` (the version mode) as the last parameter (``config`` may default to ``None`` for
          trafilatura's ``DEFAULT_CONFIG``), and ``reset_caches()``, which empties the module-level dedup cache that
          ``deduplicate=True`` reads and updates (``trafilatura.meta.reset_caches``; the suites call it before every
          stateful sequence). Optional ``extract_batch(htmls, *, threads=None, compat="2.3.1", **kwargs) ->
          list[str | None]``: the same results as ``extract`` in order, the dedup cache carried through the batch in
          order. ``<top>.integrations.patch("trafilatura")``, if the package has it, must give functions with
          the installed trafilatura's exact signatures (no ``compat``).
          ``bare_extraction`` returns an object with trafilatura's ``Document`` attributes (a dict where the
          release returns one: 1.11.0's ``as_dict=True`` default); ``body`` and ``commentsbody`` may be lxml
          elements or ``etree.tostring(element, encoding="unicode")`` strings
          (``pyroid_parity.trafilatura_compare.normalize``). Options pyroid does not port
          (``specs.trafilatura_cases(mode, "delegated")``) go to trafilatura with exactly one
          ``pyroid.FallbackWarning``. The suite checks one mode (``--trafilatura-compat``): ``compat=<mode>`` is
          passed to every function that takes ``compat``, and in that mode the candidate must behave as that
          release, whose signature may differ (1.11.0 has no ``fast``; its ``extract`` ignores unknown keywords).
nltk      NLTK 3.10.4's tokenizer API: ``sent_tokenize``, ``word_tokenize``, ``PunktTokenizer``,
          ``PunktSentenceTokenizer``, ``PunktParameters``, ``TreebankWordTokenizer``, ``NLTKWordTokenizer``,
          ``TreebankWordDetokenizer``, ``RegexpTokenizer``, ``WordPunctTokenizer``, ``WhitespaceTokenizer``,
          ``BlanklineTokenizer``, ``wordpunct_tokenize`` (``nltk`` itself, or ``nltk.tokenize``); a candidate whose
          ``sent_tokenize`` also takes a list of str gets batch checks.

warc      warcio 1.8.2's reader API: ``ArchiveIterator(fileobj, no_record_parse=False, verify_http=False,
          arc2warc=False, ensure_http_headers=False, block_size=16384, check_digests=False)``, ``WARCIterator``,
          ``ARCIterator``, ``StatusAndHeaders``, ``ArcWarcRecord``, ``DigestChecker``, ``ArchiveLoadFailed`` and
          ``WARCIO_VERSION == "1.8.2"``; records, streams, offsets, messages (stderr, logging) and exceptions as
          warcio's (``pyroid_parity.warc_compare``). ``iter_documents(source, *, compression="infer")`` yields
          ``(record index, dict)`` exactly as datatrove 0.10.1's ``WarcReader.read_file`` + ``process_record``
          (``pyroid_parity.warc_datatrove.reference_documents``); optional ``iter_batches(source, *,
          batch_size=1000, compression="infer")`` gives Arrow-exportable batches (id, url, date, mime_type,
          content_type, html, text). ``<top>.integrations.datatrove.WarcReader``, if the package has it, is checked
          against datatrove's ``WarcReader`` (same arguments, same Documents, same exceptions); a module attribute
          ``datatrove_reader`` names another class to check (``pyroid_parity.warc_mutant`` has one).

Warnings: where a suite compares warnings (the ftfy ``errors`` check, every json_repair check), a candidate's
``pyroid.FallbackWarning`` (its "delegating to the incumbent" notice, shown once per process; see
``docs/dev/MODULES.md``) is left out, so the result does not depend on test order. Every other warning
category must match the reference's.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import importlib.util
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

SUITES = (
    "ftfy",
    "fasttext",
    "filters",
    "dedup",
    "metrics",
    "json_repair",
    "datasketch",
    "jieba",
    "trafilatura",
    "nltk",
    "warc",
)
REFERENCE = {
    "ftfy": "ftfy",
    "fasttext": "fasttext",
    "filters": "datatrove",
    "dedup": "datatrove",
    "metrics": "rouge_score",
    "json_repair": "json_repair",
    "datasketch": "datasketch",
    "jieba": "jieba",
    "trafilatura": "trafilatura",
    "nltk": "nltk",
    "warc": "warcio",
}
# Distribution that provides a module, for version reporting when it differs from the module name.
DIST_HINT = {"fasttext": ("fasttext-predict", "fasttext-predict-v2", "fasttext", "fasttext-wheel")}


class ImplError(ValueError):
    pass


#: pyroid's delegation warning (``pyroid.FallbackWarning``), matched by name so pyroid need not be installed.
FALLBACK_WARNING = "pyroid.FallbackWarning"


def is_fallback_warning(category: type[Warning]) -> bool:
    """True for ``pyroid.FallbackWarning`` and its subclasses."""
    return any(f"{c.__module__}.{c.__qualname__}" == FALLBACK_WARNING for c in category.__mro__)


def warning_categories(caught: Iterable[warnings.WarningMessage]) -> list[str]:
    """Sorted names of the warning categories in ``caught`` that parity compares (all but FALLBACK_WARNING)."""
    return sorted({w.category.__name__ for w in caught if not is_fallback_warning(w.category)})


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


def _generic_metrics(mod: Any) -> Any:
    from pyroid_parity.metrics_cases import Metrics

    return Metrics.from_package(mod)


class GenericJsonRepair:
    """json_repair's four public functions, taken from the module as they are."""

    FUNCTIONS = ("repair_json", "loads", "load", "from_file")

    def __init__(self, mod: Any) -> None:
        self.mod = mod
        self.repair_json: Callable[..., Any] = mod.repair_json
        self.loads: Callable[..., Any] = mod.loads
        self.load: Callable[..., Any] = mod.load
        self.from_file: Callable[..., Any] = mod.from_file


class GenericDatasketch:
    """A module with datasketch's classes, taken as they are (``mod.MinHash``, ``mod.MinHashLSH``, ...)."""

    def __init__(self, mod: Any) -> None:
        self.mod = mod
        self.notes = [] if hasattr(mod, "minhash_batch") else ["no minhash_batch: those checks skip"]


class GenericJieba:
    """jieba's module API, taken from the module as it is; ``has_batch`` if ``lcut`` takes a list."""

    def __init__(self, mod: Any) -> None:
        self.mod = mod
        try:
            self.has_batch = mod.lcut(["中文", ""]) == [mod.lcut("中文"), []]
        except Exception:  # noqa: BLE001  jieba itself raises AttributeError for a list
            self.has_batch = False
        self.notes = [] if self.has_batch else ["lcut takes no list: batch checks skipped"]


class GenericTrafilatura:
    """trafilatura's ``extract`` / ``bare_extraction``, the dedup reset and a batch call, taken from the module.

    The mode checked (``specs.trafilatura_mode()``) is passed as ``compat=`` to every function that takes it
    (pyroid.trafilatura), and to none that does not (trafilatura itself, which is one release). For trafilatura
    itself ``reset_caches`` is ``trafilatura.meta.reset_caches``. Without ``extract_batch`` the batch checks map
    ``extract`` in order (and say so in the report).
    """

    def __init__(self, mod: Any) -> None:
        from pyroid_parity import specs

        self.mod = mod
        self.mode = specs.trafilatura_mode()
        self.extract: Callable[..., Any] = self._in_mode(mod.extract)
        self.bare_extraction: Callable[..., Any] = self._in_mode(mod.bare_extraction)
        reset = getattr(mod, "reset_caches", None)
        if reset is None and mod.__name__ == "trafilatura":
            reset = importlib.import_module("trafilatura.meta").reset_caches
        self._reset: Callable[[], Any] | None = reset
        self.has_reset = reset is not None
        self.has_batch = hasattr(mod, "extract_batch")
        self._batch = self._in_mode(mod.extract_batch) if self.has_batch else None
        self.takes_compat = self.extract is not mod.extract
        self.notes = [f"trafilatura mode {self.mode}"]
        if self.takes_compat:
            self.notes.append(f"called with compat={self.mode!r}")
        if not self.has_batch:
            self.notes.append("no extract_batch: batch checks map extract")
        if not self.has_reset:
            self.notes.append("no reset_caches(): the stateful (deduplicate=True) checks cannot run")

    def _in_mode(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        import functools
        import inspect

        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return fn
        return functools.partial(fn, compat=self.mode) if "compat" in params else fn

    def reset_caches(self) -> None:
        if self._reset is None:
            raise ImplError(f"{self.mod.__name__} has no reset_caches()")
        self._reset()

    def extract_batch(self, htmls: list[Any], threads: int | None = None, **kw: Any) -> list[str | None]:
        if self._batch is not None:
            return list(self._batch(htmls, threads=threads, **kw))
        return [self.extract(h, **kw) for h in htmls]


class GenericNltk:
    """NLTK's tokenizer API, taken from the module as it is; ``has_batch`` if ``sent_tokenize`` takes a list."""

    def __init__(self, mod: Any) -> None:
        self.mod = mod
        try:
            self.has_batch = mod.sent_tokenize(["A b. C d.", ""]) == [mod.sent_tokenize("A b. C d."), []]
        except Exception:  # noqa: BLE001  NLTK itself raises TypeError for a list
            self.has_batch = False
        self.notes = [] if self.has_batch else ["sent_tokenize takes no list: batch checks skipped"]


class GenericWarc:
    """warcio's reader API taken from the module; ``iter_documents`` / ``iter_batches`` / the datatrove step if present."""

    def __init__(self, mod: Any) -> None:
        from pyroid_parity.warc_compare import exc_namer

        self.mod = mod
        top = mod.__name__.split(".")[0]
        self.exc_name = exc_namer(top)
        self.has_documents = hasattr(mod, "iter_documents")
        self.has_batches = hasattr(mod, "iter_batches")
        self.reader_cls: Any = getattr(mod, "datatrove_reader", None)
        if self.reader_cls is None:
            try:
                self.reader_cls = importlib.import_module(f"{top}.integrations.datatrove").WarcReader
            except (ImportError, AttributeError):
                self.reader_cls = None
        self.notes = []
        if not self.has_documents:
            self.notes.append("no iter_documents: those checks skip")
        if not self.has_batches:
            self.notes.append("no iter_batches: those checks skip")
        if self.reader_cls is None:
            self.notes.append(f"no {top}.integrations.datatrove.WarcReader: the WarcReader checks skip")


GENERIC: dict[str, Callable[[Any], Any]] = {
    "ftfy": GenericFtfy,
    "fasttext": GenericFasttext,
    "filters": GenericFilters,
    "dedup": GenericDedup,
    "metrics": _generic_metrics,
    "json_repair": GenericJsonRepair,
    "datasketch": GenericDatasketch,
    "jieba": GenericJieba,
    "trafilatura": GenericTrafilatura,
    "nltk": GenericNltk,
    "warc": GenericWarc,
}
