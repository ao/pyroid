"""Compare a ``nemo`` suite candidate with NeMo Curator 1.4.0: pinned references, live NeMo and NeMo's stages.

``tests/test_nemo.py`` drives this. The checks against the pinned references of RepeatingTopNGramsFilter cases need
the hash seed they were recorded with; when the test process has another one, they run here in a subprocess:

    PYTHONHASHSEED=0 python -m pyroid_parity.nemo_compare --impl pyroid.nemo --case repeating_top_ngrams@2 \
        --what fn integration [--limit N]

which prints one JSON object, the ``Tally`` of the checks it ran.

Outcomes compared per document:
    filters    ``("ok", type name, repr of the score)`` or ``("raise", exception type)``; the integration's
               ``keep_document`` too. Batch calls: the array's dtype (float64 / int64) and every element.
    modifiers  ``("ok", digest, length)`` of the output text, or ``("raise", exception type)``.
    fastText   NeMo's score string ``str([score, lang])`` (FastTextLangId), ``repr`` of the score (quality).
"""

from __future__ import annotations

import argparse
import ast
import functools
import importlib
import json
import os
import subprocess
import sys
import types
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path
from typing import Any

from pyroid_parity import data as D
from pyroid_parity import nemo_cases as C

Outcome = tuple[Any, ...]
MAX_EXAMPLES = 20
#: Documents per corpus for the stage checks (the stock stages run NeMo's Python per document).
STAGE_DOCS = 1000
INT_FNS = {"word_count", "long_word", "common_english_words", "pornographic_urls", "substring"}
DTYPE = {"float": "float64", "int": "int64"}


# --------------------------------------------------------------------------------------------- tally


class Tally:
    """Comparisons per check, recorded here and replayed into the pytest collector (also across processes)."""

    def __init__(self) -> None:
        self.checks: dict[str, dict[str, Any]] = {}
        self.skips: dict[str, str] = {}

    def add(
        self,
        check: str,
        same: bool,
        *,
        case: Any = None,
        input: Any = None,
        got: Any = None,
        want: Any = None,
    ) -> bool:
        c = self.checks.setdefault(check, {"same": 0, "diff": 0, "examples": []})
        c["same" if same else "diff"] += 1
        if not same and len(c["examples"]) < MAX_EXAMPLES:
            c["examples"].append(
                {"case": case, "input": _short(input), "got": _short(got), "want": _short(want)}
            )
        return same

    def skip(self, check: str, reason: str) -> None:
        self.skips[check] = reason

    def diff(self, check: str) -> int:
        return self.checks.get(check, {}).get("diff", 0)

    def summary(self, check: str) -> str:
        c = self.checks.get(check)
        if c is None:
            return f"{check}: skipped ({self.skips.get(check)})"
        first = [e["case"] for e in c["examples"][:5]]
        return f"{check}: {c['diff']} of {c['same'] + c['diff']} differ, first {first}"

    def to_json(self) -> str:
        return json.dumps({"checks": self.checks, "skips": self.skips}, ensure_ascii=True)

    @classmethod
    def from_json(cls, text: str) -> Tally:
        d = json.loads(text)
        t = cls()
        t.checks, t.skips = d["checks"], d["skips"]
        return t

    def merge(self, other: Tally) -> None:
        for k, c in other.checks.items():
            mine = self.checks.setdefault(k, {"same": 0, "diff": 0, "examples": []})
            mine["same"] += c["same"]
            mine["diff"] += c["diff"]
            mine["examples"] = (mine["examples"] + c["examples"])[:MAX_EXAMPLES]
        self.skips.update(other.skips)

    def replay(self, collector: Any, suite: str = C.SUITE) -> None:
        """Record every comparison in a ``pyroid_parity.report.Collector``."""
        for check, c in self.checks.items():
            for e in c["examples"]:
                collector.record(suite, check, False, **e)
            for _ in range(c["diff"] - len(c["examples"])):
                collector.record(suite, check, False)
            for _ in range(c["same"]):
                collector.record(suite, check, True)
        for check, why in self.skips.items():
            if check not in self.checks:
                collector.skip(suite, check, why)


def _short(x: Any) -> Any:
    if isinstance(x, str) and len(x) > 300:
        return x[:300] + "…"
    return x if isinstance(x, (str, int, float, bool)) or x is None else _short(repr(x))


# ------------------------------------------------------------------------------------------- candidate


FILTER_FN_CLASS = {c.fn: c.nemo for c in C.FILTER_CASES.values()}
MODIFIER_FN_CLASS = {c.fn: c.nemo for c in C.MODIFIER_CASES.values() if c.fn != "ftfy.fix_text"}


class OracleApi:
    """The suite contract implemented with NeMo's own classes: the candidate of the self-check (no ``--impl``)."""

    def __getattr__(self, name: str) -> Any:
        if name in FILTER_FN_CLASS:
            return functools.partial(self._score, name)
        if name in MODIFIER_FN_CLASS:
            return functools.partial(self._modify, name)
        raise AttributeError(name)

    @staticmethod
    def _texts(texts: Any) -> list[str]:
        return texts.to_pylist() if hasattr(texts, "to_pylist") else list(texts)

    def _score(self, name: str, texts: Any, **kw: Any) -> Any:
        import numpy as np

        obj = C.nemo_class(FILTER_FN_CLASS[name])(**kw)
        if isinstance(texts, str):
            return obj.score_document(texts)
        out = []
        for i, t in enumerate(self._texts(texts)):
            try:
                out.append(obj.score_document(t))
            except Exception as e:
                e.add_note(f"document {i}")
                raise
        return np.array(out, dtype=np.int64 if name in INT_FNS else np.float64)

    def _modify(self, name: str, texts: Any, **kw: Any) -> Any:
        obj = C.nemo_class(MODIFIER_FN_CLASS[name])(**kw)
        if isinstance(texts, str):
            return obj.modify_document(texts)
        return [obj.modify_document(t) for t in self._texts(texts)]

    @cache  # noqa: B019  one filter per model file
    def _filter(self, cls: str, path: str) -> Any:
        obj = C.nemo_class(cls)(model_path=path)
        obj.load_model()
        return obj

    def fasttext_lang_id(self, model: str, texts: Sequence[str]) -> tuple[Any, list[str]]:
        import numpy as np

        f = self._filter("FastTextLangId", model)
        pairs = [ast.literal_eval(f.score_document(t)) for t in texts]
        return np.array([p[0] for p in pairs], dtype=np.float64), [p[1] for p in pairs]

    def lang_id_strings(self, scores: Any, langs: Sequence[str]) -> list[str]:
        return [str([float(s), lang]) for s, lang in zip(scores, langs, strict=True)]

    def fasttext_quality(self, model: str, texts: Sequence[str], label: str = C.QUALITY_LABEL) -> Any:
        import numpy as np

        f = self._filter("FastTextQualityFilter", model)
        f._label = label
        return np.array([f.score_document(t) for t in texts], dtype=np.float64)


class Candidate:
    """A ``nemo`` suite candidate: its functions, its ftfy and fastText, and its NeMo integration classes."""

    def __init__(self, mod: Any) -> None:
        self.mod = mod
        self.name = mod.__name__
        top = self.name.split(".")[0]
        self.is_nemo = top == "nemo_curator"
        self.notes: list[str] = []
        self.integration: Any = None
        self.integration_reason = ""
        if self.is_nemo:
            self.api: Any = OracleApi()
            self.ftfy = _optional("ftfy")
            self._ft_load: Callable[[str], Any] | None = str  # OracleApi takes the model path
            self.integration = types.SimpleNamespace(**{n: C.nemo_class(n) for n in C.INTEGRATION_CLASSES})
            self.integration_name = "nemo_curator (its own classes)"
        else:
            self.api = mod
            self.ftfy = _optional(f"{top}.ftfy")
            ft = _optional(f"{top}.fasttext")
            self._ft_load = getattr(ft, "load_model", None)
            self.integration_name = f"{top}.integrations.nemo_curator"
            try:
                self.integration = importlib.import_module(self.integration_name)
            except ImportError as e:
                self.integration_reason = f"{self.integration_name} does not import here: {e}"
                self.notes.append(self.integration_reason)
        missing = sorted(
            {c.fn for c in (*C.FILTER_CASES.values(), *C.MODIFIER_CASES.values()) if self.fn(c.fn) is None}
            | {f for f in ("fasttext_lang_id", "lang_id_strings", "fasttext_quality") if self.fn(f) is None}
        )
        if missing:
            self.notes.append(f"no {', '.join(missing)}: those checks skip")
        if not C.hash_seed_ok():
            self.notes.append(
                f"{C.HASH_SEED_ENV} is not {C.HASH_SEED}: RepeatingTopNGramsFilter checks against the pinned "
                "references run in a subprocess with it set"
            )

    def fn(self, name: str) -> Callable[..., Any] | None:
        if name == "ftfy.fix_text":
            return getattr(self.ftfy, "fix_text", None)
        return getattr(self.api, name, None)

    def fn_batch(self, name: str) -> Callable[..., Any] | None:
        """The list form: the function itself, or ``fix_text_batch`` for ftfy (``fix_text`` mapped without it)."""
        if name != "ftfy.fix_text":
            return self.fn(name)
        batch = getattr(self.ftfy, "fix_text_batch", None)
        fix = self.fn(name)
        if batch is not None:
            return lambda texts, **kw: list(batch(list(texts), **kw))
        return None if fix is None else (lambda texts, **kw: [fix(t, **kw) for t in texts])

    def missing(self, name: str) -> str:
        where = f"{self.ftfy.__name__ if self.ftfy else 'ftfy'}" if name == "ftfy.fix_text" else self.name
        return f"{where} has no {name.removeprefix('ftfy.')}"

    @cache  # noqa: B019  one model per file per candidate
    def fasttext_model(self, path: str) -> Any:
        if self._ft_load is None:
            raise LookupError(f"{self.name}'s package has no fasttext.load_model")
        return self._ft_load(path)


def _optional(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError:
        return None


# ------------------------------------------------------------------------------------------- references


class RefError(LookupError):
    pass


@cache
def ref_table() -> Any:
    import pyarrow.parquet as pq

    try:
        path = D.path(C.nemo_ref(D.platform_key()))
    except D.DataError as e:
        raise RefError(str(e)) from None
    return pq.read_table(path)


@cache
def ref_rows(case: str) -> dict[str, dict[str, Any]]:
    import pyarrow.compute as pc

    t = ref_table()
    d = t.filter(pc.equal(t["case"], case)).to_pydict()
    return {i: {k: d[k][n] for k in d} for n, i in enumerate(d["id"])}


@cache
def all_docs() -> D.Corpus:
    """dev + adversarial, in the order of the references (``refs.all_docs``)."""
    dev, adv = D.corpus("dev"), D.corpus("adversarial")
    return D.Corpus("dev+adversarial", dev.ids + adv.ids, dev.texts + adv.texts, dev.meta + adv.meta)


@cache
def doc_index(case: str, limit: int) -> tuple[int, ...]:
    """Indices into ``all_docs()`` of the case's documents; with ``limit``, its first ``limit`` of each corpus
    and, always, the adversarial set's 32 edge cases (empty, whitespace-only, ...: where NeMo raises)."""
    docs = all_docs()
    idx = C.doc_indices(C.ALL_CASES[case].docs, docs.meta)
    if not limit:
        return tuple(idx)
    n_dev = len(D.corpus("dev"))
    dev, adv = [i for i in idx if i < n_dev], [i for i in idx if i >= n_dev]
    edge = [i for i in adv[limit:] if docs.meta[i] == "edge"]
    return tuple(dev[:limit] + adv[:limit] + edge)


def docs_for(case: str, limit: int) -> tuple[list[str], list[str]]:
    docs = all_docs()
    idx = doc_index(case, limit)
    return [docs.ids[i] for i in idx], [docs.texts[i] for i in idx]


def want_score(row: dict[str, Any], keep: bool = False) -> Outcome:
    if row["outcome"] == "raise":
        return ("raise", row["error"])
    return ("ok", row["type"], row["score"], row["keep"]) if keep else ("ok", row["type"], row["score"])


def want_text(row: dict[str, Any]) -> Outcome:
    return ("raise", row["error"]) if row["outcome"] == "raise" else ("ok", row["digest"], row["length"])


# --------------------------------------------------------------------------------------------- outcomes


def score_outcome(fn: Callable[..., Any], text: str, kw: dict[str, Any]) -> Outcome:
    try:
        v = fn(text, **kw)
    except Exception as e:  # noqa: BLE001  the exception type is part of parity
        return ("raise", type(e).__name__)
    return ("ok", *C.score_cell(v))


def text_outcome(fn: Callable[..., Any], text: str, kw: dict[str, Any]) -> Outcome:
    try:
        out = fn(text, **kw)
    except Exception as e:  # noqa: BLE001
        return ("raise", type(e).__name__)
    if not isinstance(out, str):
        return ("ok", f"<{type(out).__name__}>", None)
    return ("ok", C.digest(out), len(out))


def _element(arr: Any, k: int) -> Outcome:
    x = arr[k]
    kind = getattr(getattr(arr, "dtype", None), "kind", "")
    return (
        str(getattr(arr, "dtype", type(arr).__name__)),
        repr(float(x)) if kind == "f" else repr(int(x)) if kind in "iu" else repr(x),
    )


def _zh_unsupported(c: Any, got: Outcome) -> bool:
    return got == ("raise", "NotImplementedError") and str(c.kwargs.get("lang", "")).lower() in ("zh", "ja")


# --------------------------------------------------------------------------------- pinned: functions


def filter_fn(cand: Candidate, case: str, tally: Tally, limit: int) -> None:
    """Checks ``<case>`` (one str call per document) and ``batch-<case>`` (list and Arrow inputs)."""
    c = C.FILTER_CASES[case]
    single, batch = case, f"batch-{case}"
    fn = cand.fn(c.fn)
    if fn is None:
        for k in (single, batch):
            tally.skip(k, cand.missing(c.fn))
        return
    ids, texts = docs_for(case, limit)
    ref = ref_rows(case)
    want = [want_score(ref[i]) for i in ids]
    got_all = [score_outcome(fn, t, c.kwargs) for t in texts]
    if any(_zh_unsupported(c, g) for g in got_all):
        for k in (single, batch):
            tally.skip(k, f"{cand.name}.{c.fn} raises NotImplementedError for lang={c.kwargs['lang']!r}")
        return
    for i, t, g, w in zip(ids, texts, got_all, want, strict=True):
        tally.add(single, g == w, case=i, input=t, got=g, want=w)
    import pyarrow as pa

    for form, data in (("list", list(texts)), ("arrow", pa.array(texts, pa.string()))):
        _batch_scores(fn, c.kwargs, form, data, ids, texts, want, tally, batch)


def _batch_scores(
    fn: Callable[..., Any],
    kw: dict[str, Any],
    form: str,
    data: Any,
    ids: list[str],
    texts: list[str],
    want: list[Outcome],
    tally: Tally,
    check: str,
) -> None:
    raising = [k for k, w in enumerate(want) if w[0] == "raise"]
    if raising:
        k0 = raising[0]
        try:
            fn(data, **kw)
            got: Outcome = ("ok",)
        except Exception as e:  # noqa: BLE001
            got = ("raise", type(e).__name__)
        tally.add(check, got == want[k0], case=f"{form}:first-raise:{ids[k0]}", got=got, want=want[k0])
        keep = [k for k in range(len(ids)) if k not in set(raising)]
        data = [texts[k] for k in keep] if form == "list" else _arrow([texts[k] for k in keep])
    else:
        keep = list(range(len(ids)))
    if not keep:
        return
    try:
        arr = fn(data, **kw)
    except Exception as e:  # noqa: BLE001
        tally.add(check, False, case=f"{form}:call", got=("raise", type(e).__name__, str(e)), want="an array")
        return
    if len(arr) != len(keep):
        tally.add(check, False, case=f"{form}:length", got=len(arr), want=len(keep))
        return
    for n, k in enumerate(keep):
        w = (DTYPE.get(want[k][1], want[k][1]), want[k][2])
        g = _element(arr, n)
        tally.add(check, g == w, case=f"{form}:{ids[k]}", input=texts[k], got=g, want=w)


def _arrow(texts: list[str]) -> Any:
    import pyarrow as pa

    return pa.array(texts, pa.string())


def modifier_fn(cand: Candidate, case: str, tally: Tally, limit: int) -> None:
    """Checks ``<case>`` (str calls) and ``batch-<case>`` (list, and Arrow for pyroid.nemo functions)."""
    c = C.MODIFIER_CASES[case]
    single, batch = case, f"batch-{case}"
    fn, fnb = cand.fn(c.fn), cand.fn_batch(c.fn)
    if fn is None:
        for k in (single, batch):
            tally.skip(k, cand.missing(c.fn))
        return
    ids, texts = docs_for(case, limit)
    ref = ref_rows(case)
    want = [want_text(ref[i]) for i in ids]
    for i, t, w in zip(ids, texts, want, strict=True):
        g = text_outcome(fn, t, c.kwargs)
        tally.add(single, g == w, case=i, input=t, got=g, want=w)
    forms: list[tuple[str, Any]] = [("list", list(texts))]
    if c.fn != "ftfy.fix_text":
        forms.append(("arrow", _arrow(texts)))
    for form, data in forms:
        try:
            out = fnb(data, **c.kwargs)
        except Exception as e:  # noqa: BLE001
            tally.add(
                batch, False, case=f"{form}:call", got=("raise", type(e).__name__, str(e)), want="a list"
            )
            continue
        out = out.to_pylist() if hasattr(out, "to_pylist") else list(out)
        if len(out) != len(texts):
            tally.add(batch, False, case=f"{form}:length", got=len(out), want=len(texts))
            continue
        for i, t, o, w in zip(ids, texts, out, want, strict=True):
            g = ("ok", C.digest(o), len(o)) if isinstance(o, str) else ("ok", f"<{type(o).__name__}>", None)
            tally.add(batch, g == w, case=f"{form}:{i}", input=t, got=g, want=w)


def fasttext_fn(cand: Candidate, case: str, tally: Tally, limit: int) -> None:
    """``lang_id@<model>``: fasttext_lang_id + lang_id_strings vs NeMo's score strings; ``quality``: the scores."""
    c = C.FASTTEXT_CASES[case]
    if "lang" in c.nemo_kwargs:
        tally.skip(
            case, "the scores of lang_id@lid.176.bin; this case's keep_document is in its integration check"
        )
        return
    needs = ("fasttext_lang_id", "lang_id_strings") if c.nemo == "FastTextLangId" else ("fasttext_quality",)
    if missing := [f for f in needs if cand.fn(f) is None]:
        tally.skip(case, cand.missing(missing[0]))
        return
    try:
        model = cand.fasttext_model(str(D.path(c.model)))
    except (LookupError, D.DataError) as e:
        tally.skip(case, str(e))
        return
    ids, texts = docs_for(case, limit)
    ref = ref_rows(case)
    want = [want_score(ref[i]) for i in ids]
    if any(w[0] == "raise" for w in want):
        tally.add(case, False, case="reference", got="NeMo raised on some documents", want="no exception")
        return
    if c.nemo == "FastTextLangId":
        scores, langs = cand.fn("fasttext_lang_id")(model, texts)
        strings = cand.fn("lang_id_strings")(scores, langs)
        got = [(str(scores.dtype), s) for s in strings]
        expect = [("float64", w[2]) for w in want]
    else:
        scores = cand.fn("fasttext_quality")(model, texts, label=c.nemo_kwargs["label"])
        got = [_element(scores, k) for k in range(len(scores))]
        expect = [("float64", w[2]) for w in want]
    for i, t, g, w in zip(ids, texts, got, expect, strict=True):
        tally.add(case, g == w, case=i, input=t, got=g, want=w)


# ------------------------------------------------------------------------------ pinned: integration


def integration_doc(cand: Candidate, case: str, tally: Tally, limit: int) -> None:
    """``integration-<case>``: the integration class, document by document, against the pinned references."""
    check = f"integration-{case}"
    c = C.ALL_CASES[case]
    if cand.integration is None:
        tally.skip(check, cand.integration_reason)
        return
    cls = getattr(cand.integration, c.nemo, None)
    if cls is None:
        tally.skip(check, f"{cand.integration_name} has no {c.nemo}")
        return
    ids, texts = docs_for(case, limit)
    ref = ref_rows(case)
    kind = C.kind_of(case)
    if kind == "modifier":
        obj = c.make(cls)
        for i, t in zip(ids, texts, strict=True):
            g, w = text_outcome(lambda x: obj.modify_document(x), t, {}), want_text(ref[i])
            tally.add(check, g == w, case=i, input=t, got=g, want=w)
        return
    if case == "quality":
        _quality_keep(cls, c, ids, texts, ref, tally, check, limit)
        return
    if kind == "fasttext":
        obj = c.make(cls, model_path=str(D.path(c.model)))
        obj.load_model()
    else:
        obj = c.make(cls)
    for i, t in zip(ids, texts, strict=True):
        try:
            v = obj.score_document(t)
            g: Outcome = ("ok", *C.score_cell(v), bool(obj.keep_document(v)))
        except Exception as e:  # noqa: BLE001
            g = ("raise", type(e).__name__)
        w = want_score(ref[i], keep=True)
        tally.add(check, g == w, case=i, input=t, got=g, want=w)


def _quality_keep(
    cls: type,
    c: Any,
    ids: list[str],
    texts: list[str],
    ref: dict[str, Any],
    tally: Tally,
    check: str,
    limit: int,
) -> None:
    """Scores, then keep_document once per document in the recorded order (dev, then adversarial).

    The draws come from numpy's global RNG, seeded by the constructor. With ``--parity-limit N`` only the first N
    documents of that order are checked: later documents' draws depend on every earlier one.
    """
    if limit:
        n_dev = sum(1 for i in ids if not i.startswith("adv-"))
        ids, texts = ids[:n_dev], texts[:n_dev]
    obj = c.make(cls, model_path=str(D.path(c.model)))  # seeds the global RNG, as in NeMo
    obj.load_model()
    scores = []
    for t in texts:
        try:
            scores.append(obj.score_document(t))
        except Exception as e:  # noqa: BLE001
            scores.append(e)
    for i, t, s in zip(ids, texts, scores, strict=True):
        g: Outcome = (
            ("raise", type(s).__name__)
            if isinstance(s, Exception)
            else ("ok", *C.score_cell(s), bool(obj.keep_document(s)))
        )
        w = want_score(ref[i], keep=True)
        tally.add(check, g == w, case=i, input=t, got=g, want=w)


#: Extra constructor parameters seen on integration classes (keyword-only, with a default), per class.
EXTRA_PARAMETERS: dict[str, list[str]] = {}


def integration_classes(cand: Candidate, tally: Tally) -> None:
    """``integration-classes``: every class exists, subclasses NeMo's, and takes NeMo's constructor arguments.

    Parameters are compared by name, kind and default, in order (annotations are left out: a string annotation
    and the type it names are the same parameter). NeMo's stages are dataclasses whose ``__init__`` NeMo wraps
    as ``(*args, **kwargs)``, so their init fields are compared instead. A class may add parameters only as
    keyword-only ones with a default (every NeMo call then means the same); those are listed in
    ``EXTRA_PARAMETERS``, and the check's case names them.
    """
    check = "integration-classes"
    if cand.integration is None:
        tally.skip(check, cand.integration_reason)
        return
    for name in C.INTEGRATION_CLASSES:
        nemo = C.nemo_class(name)
        cls = getattr(cand.integration, name, None)
        if cls is None:
            tally.add(check, False, case=name, got=None, want=f"{nemo.__module__}.{name}")
            continue
        sub = isinstance(cls, type) and issubclass(cls, nemo)
        want = parameters(nemo)
        got = parameters(cls)
        names = {p[0] for p in want}
        extra = [p for p in got if p[0] not in names and p[1] == "KEYWORD_ONLY" and p[2] is not None]
        if extra:
            EXTRA_PARAMETERS[name] = [f"{p[0]}={p[2]}" for p in extra]
            got = [p for p in got if p not in extra]
        case = f"{name} (+ {', '.join(EXTRA_PARAMETERS[name])})" if extra else name
        tally.add(check, sub and got == want, case=case, got=(sub, got), want=(True, want))


def parameters(cls: type) -> list[tuple[str, str, str | None]]:
    """``(name, kind, repr of the default or None)`` of each constructor parameter."""
    import dataclasses
    import inspect

    if dataclasses.is_dataclass(cls):
        out = []
        for f in dataclasses.fields(cls):
            if not f.init:
                continue
            if f.default is not dataclasses.MISSING:
                default: str | None = repr(f.default)
            elif f.default_factory is not dataclasses.MISSING:
                default = f"<factory {f.default_factory()!r}>"
            else:
                default = None
            out.append((f.name, "KEYWORD_ONLY" if f.kw_only is True else "POSITIONAL_OR_KEYWORD", default))
        return out
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError) as e:
        return [(f"<{type(e).__name__}>", "", None)]
    return [
        (p.name, p.kind.name, None if p.default is p.empty else repr(p.default))
        for p in sig.parameters.values()
    ]


# ------------------------------------------------------------------------------- live: NeMo's stages


@cache
def _reader_frame() -> Any:
    import pandas as pd
    from nemo_curator.stages.text.io.reader.parquet import ParquetReaderStage

    reader = ParquetReaderStage()
    frames = [
        reader.read_data([str(D.path(D.CORPORA[name]))], {}, ["id", "text"])
        for name in ("dev", "adversarial")
    ]
    return pd.concat(frames, ignore_index=True)


def _stage_frame(case: str, limit: int) -> Any:
    """The case's first ``limit`` (default STAGE_DOCS) documents of dev and of adversarial as NeMo's
    ParquetReaderStage reads them (``string[pyarrow]`` columns), with a fresh RangeIndex."""
    df = _reader_frame()
    ids, _ = docs_for(case, limit or STAGE_DOCS)
    return df[df["id"].isin(set(ids))].reset_index(drop=True)


def _batch(df: Any) -> Any:
    from nemo_curator.tasks import DocumentBatch

    return DocumentBatch(dataset_name="parity", data=df.copy())


def _run(make_stage: Callable[[], Any], df: Any) -> Outcome:
    stage = make_stage()
    try:
        stage.setup()
        out = stage.process(_batch(df))
    except Exception as e:  # noqa: BLE001
        return ("raise", type(e).__name__)
    return ("ok", out.to_pandas())


def frame_diff(got: Outcome, want: Outcome) -> str | None:
    """None if the two stage outcomes are identical (columns, dtypes, index, every value), else the first difference."""
    if got[0] != want[0] or got[0] == "raise":
        return None if got == want else f"{got} vs {want}"
    g, w = got[1], want[1]
    if list(g.columns) != list(w.columns):
        return f"columns {list(g.columns)} vs {list(w.columns)}"
    gd, wd = {k: str(v) for k, v in g.dtypes.items()}, {k: str(v) for k, v in w.dtypes.items()}
    if gd != wd:
        return f"dtypes {gd} vs {wd}"
    if list(g.index) != list(w.index):
        return f"index: {len(g)} rows vs {len(w)}, first {list(g.index)[:5]} vs {list(w.index)[:5]}"
    for col in g.columns:
        gv, wv = g[col].tolist(), w[col].tolist()
        bad = [k for k, (a, b) in enumerate(zip(gv, wv, strict=True)) if _value(a) != _value(b)]
        if bad:
            k = bad[0]
            return f"column {col!r}: {len(bad)} rows differ, first index {g.index[k]}: {gv[k]!r} vs {wv[k]!r}"
    return None


def _value(x: Any) -> tuple[str, str]:
    return type(x).__name__, x if isinstance(x, str) else repr(x)


Factory = Callable[[], Any]


def _stage_variants(
    c: Any, P: Any, model: str | None
) -> tuple[dict[str, tuple[Factory, str]], dict[str, Factory]]:
    """``({variant: (candidate stage factory, stock key)}, {stock key: stock stage factory})``.

    "candidate" is the integration's class, "stock" NeMo's; a variant needs the classes it names. Each factory
    builds its filter when called, so a FastTextQualityFilter seeds numpy's global RNG right before its stage.
    """
    kind = C.kind_of(c.case)
    Pc, Sc = getattr(P, c.nemo, None), C.nemo_class(c.nemo)

    def obj(cls: type) -> Any:
        return c.make(cls, model_path=model) if kind == "fasttext" else c.make(cls)

    variants: dict[str, tuple[Factory, str]] = {}
    if kind == "modifier":
        PM, SM = getattr(P, "Modify", None), C.nemo_class("Modify")
        stocks = {"Modify": lambda: SM(obj(Sc)), "Modify/out": lambda: SM(obj(Sc), output_fields="out")}
        if PM is not None and Pc is not None:
            variants["Modify(candidate modifier)"] = (lambda: PM(obj(Pc)), "Modify")
            variants["Modify(candidate modifier, output_fields)"] = (
                lambda: PM(obj(Pc), output_fields="out"),
                "Modify/out",
            )
        if PM is not None:
            variants["Modify(stock modifier)"] = (lambda: PM(obj(Sc)), "Modify")
        if Pc is not None:
            variants["stock Modify(candidate modifier)"] = (lambda: SM(obj(Pc)), "Modify")
        return variants, stocks
    PSF, SSF = getattr(P, "ScoreFilter", None), C.nemo_class("ScoreFilter")
    PS, SS = getattr(P, "Score", None), C.nemo_class("Score")
    stocks = {
        "ScoreFilter/score": lambda: SSF(obj(Sc), score_field="score"),
        "ScoreFilter": lambda: SSF(obj(Sc)),
        "ScoreFilter/invert": lambda: SSF(obj(Sc), score_field="score", invert=True),
        "Score": lambda: SS(obj(Sc), score_field="score"),
    }
    if PSF is not None and Pc is not None:
        variants["ScoreFilter(candidate filter, score_field)"] = (
            lambda: PSF(obj(Pc), score_field="score"),
            "ScoreFilter/score",
        )
        variants["ScoreFilter(candidate filter)"] = (lambda: PSF(obj(Pc)), "ScoreFilter")
        variants["ScoreFilter(candidate filter, invert)"] = (
            lambda: PSF(obj(Pc), score_field="score", invert=True),
            "ScoreFilter/invert",
        )
    if PSF is not None:
        variants["ScoreFilter(stock filter, score_field)"] = (
            lambda: PSF(obj(Sc), score_field="score"),
            "ScoreFilter/score",
        )
    if Pc is not None:
        variants["stock ScoreFilter(candidate filter, score_field)"] = (
            lambda: SSF(obj(Pc), score_field="score"),
            "ScoreFilter/score",
        )
    if PS is not None and Pc is not None:
        variants["Score(candidate filter)"] = (lambda: PS(obj(Pc), score_field="score"), "Score")
    if PS is not None:
        variants["Score(stock filter)"] = (lambda: PS(obj(Sc), score_field="score"), "Score")
    return variants, stocks


def stage(cand: Candidate, case: str, tally: Tally, limit: int) -> None:
    """``stage-<case>``: NeMo's stages with the candidate's classes vs the stock ones, on the same DocumentBatch.

    The variants mix candidate and stock stages and filters (the candidate's stages also take stock NeMo
    filters). Documents on which NeMo's filter raises are checked apart: a batch with the first of them must make
    both pipelines raise the same exception type, and the main batch leaves them out.
    """
    check = f"stage-{case}"
    if cand.integration is None:
        tally.skip(check, cand.integration_reason)
        return
    c = C.ALL_CASES[case]
    model = str(D.path(c.model)) if isinstance(c, C.FastTextCase) else None
    variants, stocks = _stage_variants(c, cand.integration, model)
    if not variants:
        tally.skip(check, f"{cand.integration_name} has neither the stages nor {c.nemo}")
        return
    df = _stage_frame(case, limit)
    stock = c.make(model_path=model) if model else c.make()
    if model:
        stock.load_model()
    one = text_outcome if C.kind_of(case) == "modifier" else score_outcome
    method = stock.modify_document if C.kind_of(case) == "modifier" else stock.score_document
    raising = [k for k, t in enumerate(df["text"]) if one(method, t, {})[0] == "raise"]
    if raising:
        rest = [k for k in range(len(df)) if k not in set(raising)][:3]
        small = df.iloc[[raising[0], *rest]].reset_index(drop=True)
        for name, (mk_got, key) in variants.items():
            g, w = _run(mk_got, small), _run(stocks[key], small)
            diff = frame_diff(g, w)
            want = w if w[0] == "raise" else "the stock frame"
            tally.add(
                check, diff is None, case=f"{name} [raise]", input=df["id"][raising[0]], got=diff, want=want
            )
        df = df.drop(index=raising).reset_index(drop=True)
    want_of: dict[str, Outcome] = {}
    for name, (mk_got, key) in variants.items():
        if key not in want_of:
            want_of[key] = _run(stocks[key], df)
        diff = frame_diff(_run(mk_got, df), want_of[key])
        tally.add(
            check, diff is None, case=name, input=f"{len(df)} documents", got=diff, want="the stock frame"
        )
    if C.kind_of(case) == "filter" and getattr(cand.integration, "Filter", None) is not None:
        _filter_stage(c, cand.integration, df, tally, check)


def _filter_stage(c: Any, P: Any, df: Any, tally: Tally, check: str) -> None:
    """Filter (keep_document on an existing score column), candidate vs stock, on stock Score's output."""
    SS, SF, PF = C.nemo_class("Score"), C.nemo_class("Filter"), P.Filter
    scored = _run(lambda: SS(c.make(), score_field="score"), df)
    if scored[0] != "ok":
        return
    Pc = getattr(P, c.nemo, None) or C.nemo_class(c.nemo)
    g = _run(lambda: PF(c.make(Pc), filter_field="score"), scored[1])
    w = _run(lambda: SF(c.make(), filter_field="score"), scored[1])
    diff = frame_diff(g, w)
    tally.add(
        check,
        diff is None,
        case="Filter(candidate filter)",
        input=f"{len(df)} documents",
        got=diff,
        want="the stock frame",
    )


def stage_on_texts(cand: Candidate, case: str, texts: Sequence[str], tally: Tally, check: str) -> None:
    """The stage variants of ``stage`` on a DocumentBatch whose text column is object dtype holding ``texts``
    (strs an Arrow column cannot hold, such as lone surrogates), against NeMo's stock stages."""
    import pandas as pd

    c = C.ALL_CASES[case]
    variants, stocks = _stage_variants(c, cand.integration, None)
    df = pd.DataFrame(
        {"id": [f"#{k}" for k in range(len(texts))], "text": pd.Series(list(texts), dtype=object)}
    )
    want_of: dict[str, Outcome] = {}
    for name, (mk_got, key) in variants.items():
        if key not in want_of:
            want_of[key] = _run(stocks[key], df)
        diff = frame_diff(_run(mk_got, df), want_of[key])
        tally.add(
            check, diff is None, case=f"stage:{name}", input=list(texts), got=diff, want="the stock frame"
        )


# ------------------------------------------------------------------------------------- subprocess entry


def isolated(impl: str, case: str, what: Sequence[str], limit: int) -> Tally:
    """Run checks for ``case`` in a fresh interpreter with the references' hash seed."""
    env = {**os.environ, C.HASH_SEED_ENV: C.HASH_SEED}
    cmd = [
        sys.executable,
        "-m",
        "pyroid_parity.nemo_compare",
        "--impl",
        impl,
        "--case",
        case,
        "--limit",
        str(limit),
    ]
    proc = subprocess.run(
        [*cmd, "--what", *what],
        env=env,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(f"{' '.join(cmd)} exited {proc.returncode}: {proc.stderr[-2000:]}")
    return Tally.from_json(proc.stdout.strip().splitlines()[-1])


CHECKS: dict[str, Callable[[Candidate, str, Tally, int], None]] = {
    "fn": lambda cand, case, t, n: {"filter": filter_fn, "modifier": modifier_fn, "fasttext": fasttext_fn}[
        C.kind_of(case)
    ](cand, case, t, n),
    "integration": integration_doc,
    "stage": stage,
}


def run(cand: Candidate, case: str, what: Sequence[str], limit: int) -> Tally:
    t = Tally()
    for w in what:
        CHECKS[w](cand, case, t, limit)
    return t


def main(argv: list[str] | None = None) -> int:
    from pyroid_parity import impl as I

    ap = argparse.ArgumentParser(prog="python -m pyroid_parity.nemo_compare")
    ap.add_argument("--impl", required=True)
    ap.add_argument("--case", required=True, choices=list(C.ALL_CASES))
    ap.add_argument("--what", nargs="+", choices=list(CHECKS), default=["fn", "integration"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    if C.ALL_CASES[a.case].hash_seeded and not C.hash_seed_ok():
        sys.exit(f"{a.case} needs {C.HASH_SEED_ENV}={C.HASH_SEED}")
    cand = I.resolve(C.SUITE, a.impl).obj
    print(run(cand, a.case, a.what, a.limit).to_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
