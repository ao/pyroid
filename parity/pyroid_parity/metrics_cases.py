"""The ``metrics`` suite: pinned eval data, the cases, and one serialization shared by the refs and the checks.

ROUGE (rouge-score 0.1.2) runs on summarization data, BLEU and chrF (sacrebleu 2.6.0) on machine translation:

- ``metrics/cnndm-3.0.0-test.parquet``: CNN/DailyMail test. Target = the highlights; prediction = the lead-3
  sentences of the article (the classic extractive baseline), both newline-separated so rougeLsum sees sentences.
  The ``article`` cases score the whole article (long targets) against the highlights.
- ``metrics/summeval.parquet``: the 16 SummEval systems' summaries of 100 CNN/DM articles, 11 references each
  (``score_multi``).
- ``metrics/wmt21.parquet``: WMT21 newstest2021 system outputs for 8 language pairs (141 systems, 163k segments),
  with every published reference (up to 3).

Every result is compared as a string that carries the exact floats (``repr``), the result types (``0`` vs ``0.0``),
``str()``, ``format()`` and the signature, so "same" means the same bytes a user would see.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Sequence
from functools import cache
from typing import Any

from pyroid_parity import data as D

CNNDM = "metrics/cnndm-3.0.0-test.parquet"
SUMMEVAL = "metrics/summeval.parquet"
WMT21 = "metrics/wmt21.parquet"
REF = "parity/metrics-ref.parquet"
ORACLES = ("rouge-score", "sacrebleu", "nltk")

# ------------------------------------------------------------------------------------------------- the cases

# case -> (data, RougeScorer kwargs, how). how: "score" (target, prediction) or "multi" (score_multi).
ROUGE_CASES: dict[str, tuple[str, dict[str, Any], str]] = {
    "rouge@cnndm": ("cnndm", {"rouge_types": ["rouge1", "rouge2", "rougeL", "rougeLsum"]}, "score"),
    "rouge@cnndm-stem": (
        "cnndm",
        {"rouge_types": ["rouge1", "rouge2", "rouge3", "rouge4", "rougeL", "rougeLsum"], "use_stemmer": True},
        "score",
    ),
    "rougeLsum@cnndm-stem": ("cnndm", {"rouge_types": ["rougeLsum"], "use_stemmer": True}, "score"),
    "rouge@article-stem": (
        "article",
        {"rouge_types": ["rouge1", "rouge2", "rougeL", "rougeLsum"], "use_stemmer": True},
        "score",
    ),
    "rouge@summeval": ("summeval", {"rouge_types": ["rouge1", "rouge2", "rougeL", "rougeLsum"]}, "score"),
    "rouge@summeval-multi-stem": (
        "summeval-multi",
        {"rouge_types": ["rouge1", "rouge2", "rougeL", "rougeLsum"], "use_stemmer": True},
        "multi",
    ),
}
ARTICLE_PAIRS = 2000  # the article-length cases use the first 2,000 CNN/DM pairs (long LCS tables)

# BLEU/chrF corpus cases: case -> (metric class, kwargs, references). Scored per (language pair, system).
# references: "all" (every published reference), "A" (the first), "holes" (all + a copy of the last reference
# with None at every 5th segment: a variable number of references per segment).
CORPUS_CASES: dict[str, tuple[str, dict[str, Any], str]] = {
    "bleu": ("BLEU", {}, "all"),
    "bleu@refA": ("BLEU", {}, "A"),
    "bleu@trg_lang": ("BLEU", {"trg_lang": "<tgt>"}, "all"),
    "bleu@intl": ("BLEU", {"tokenize": "intl"}, "all"),
    "bleu@char": ("BLEU", {"tokenize": "char"}, "A"),
    "bleu@none": ("BLEU", {"tokenize": "none"}, "A"),
    "bleu@zh": ("BLEU", {"tokenize": "zh"}, "all"),
    "bleu@lowercase": ("BLEU", {"lowercase": True}, "all"),
    "bleu@floor": ("BLEU", {"smooth_method": "floor", "smooth_value": 0.1}, "A"),
    "bleu@add-k": ("BLEU", {"smooth_method": "add-k", "smooth_value": 1}, "A"),
    "bleu@nosmooth": ("BLEU", {"smooth_method": "none"}, "A"),
    "bleu@eff-order": ("BLEU", {"effective_order": True}, "A"),
    "bleu@max-order-2": ("BLEU", {"max_ngram_order": 2}, "A"),
    "bleu@holes": ("BLEU", {}, "holes"),
    "chrf": ("CHRF", {}, "all"),
    "chrf@refA": ("CHRF", {}, "A"),
    "chrf++": ("CHRF", {"word_order": 2}, "all"),
    "chrf@whitespace": ("CHRF", {"whitespace": True}, "A"),
    "chrf@beta1": ("CHRF", {"beta": 1}, "A"),
    "chrf@lowercase": ("CHRF", {"lowercase": True}, "A"),
    "chrf@eps": ("CHRF", {"eps_smoothing": True}, "all"),
    "chrf@holes": ("CHRF", {"word_order": 2}, "holes"),
}
# Sentence-level cases over every WMT21 segment: case -> (compat function, kwargs, references).
SENTENCE_CASES: dict[str, tuple[str, dict[str, Any], str]] = {
    "sentence_bleu": ("sentence_bleu", {}, "A"),
    "sentence_bleu@multi": ("sentence_bleu", {}, "all"),
    "sentence_chrf": ("sentence_chrf", {}, "all"),
    "sentence_chrf++": ("sentence_chrf", {"word_order": 2}, "A"),
}
# trg_lang values whose tokenizer the Rust port does not implement (MeCab): left out of "bleu@trg_lang".
DELEGATED_LANGS = ("ja", "ko")

# ----------------------------------------------------------------------------------------------------- data

_SENT_END = re.compile(r"(?<=[.!?])\s+")


def lead3(article: str) -> str:
    """The first three sentences of ``article`` (split after . ! ? + whitespace), newline-separated."""
    return "\n".join(_SENT_END.split(article.strip())[:3])


def sentences(article: str) -> str:
    return "\n".join(_SENT_END.split(article.strip()))


@cache
def rouge_pairs(data: str) -> tuple[list[str], list[Any], list[str]]:
    """``(keys, targets, predictions)``; for ``summeval-multi`` each target is a list of 11 references."""
    if data in ("cnndm", "article"):
        t = D.read_parquet_columns(D.path(CNNDM), ["id", "article", "highlights"])
        if data == "cnndm":
            return t["id"], t["highlights"], [lead3(a) for a in t["article"]]
        n = ARTICLE_PAIRS
        return t["id"][:n], [sentences(a) for a in t["article"][:n]], t["highlights"][:n]
    t = D.read_parquet_columns(D.path(SUMMEVAL), ["id", "machine_summaries", "human_summaries"])
    keys, targets, preds = [], [], []
    for doc, machine, human in zip(t["id"], t["machine_summaries"], t["human_summaries"], strict=True):
        for k, summary in enumerate(machine):
            keys.append(f"{doc}/{k}")
            targets.append(list(human) if data == "summeval-multi" else human[0])
            preds.append(summary)
    return keys, targets, preds


@cache
def wmt21() -> dict[str, list[Any]]:
    return D.read_parquet_columns(D.path(WMT21), ["lp", "system", "seg", "hyp", "refs"])


def refs_for(refs: list[list[str]], which: str) -> list[list[str | None]]:
    """Reference *streams* (``streams[k][i]``) for one system's segments ``refs[i]``."""
    if which == "A":
        return [[r[0] for r in refs]]
    streams: list[list[str | None]] = [list(s) for s in zip(*refs, strict=True)]
    if which == "holes":
        streams.append([None if i % 5 == 0 else r for i, r in enumerate(streams[-1])])
    return streams


@cache
def systems() -> list[tuple[str, list[str], list[list[str]]]]:
    """``[(key "lp/system", hyps, refs per segment)]`` in file order."""
    w = wmt21()
    out: dict[str, tuple[list[str], list[list[str]]]] = {}
    for lp, system, hyp, refs in zip(w["lp"], w["system"], w["hyp"], w["refs"], strict=True):
        h, r = out.setdefault(f"{lp}/{system}", ([], []))
        h.append(hyp)
        r.append(list(refs))
    return [(k, h, r) for k, (h, r) in out.items()]


def segment_keys() -> list[str]:
    w = wmt21()
    return [f"{lp}/{s}/{i}" for lp, s, i in zip(w["lp"], w["system"], w["seg"], strict=True)]


# ------------------------------------------------------------------------------------------ serialization


def ser_rouge(result: dict[str, Any]) -> str:
    """``repr`` of the result dict: types, order, field names and shortest-repr floats."""
    return repr(result)


def ser_bleu(score: Any, signature: str | None = None) -> str:
    fields = [
        score.score,
        score.counts,
        score.totals,
        score.precisions,
        score.bp,
        score.sys_len,
        score.ref_len,
    ]
    out = "|".join(map(repr, fields)) + "|" + str(score)
    if signature is not None:
        out += (
            "|"
            + signature
            + "|"
            + score.format(signature=signature)
            + "|"
            + score.format(signature=signature, is_json=True)
        )
    return out


def ser_chrf(score: Any, signature: str | None = None) -> str:
    fields = [score.score, score.char_order, score.word_order, score.beta]
    out = "|".join(map(repr, fields)) + "|" + str(score)
    if signature is not None:
        out += (
            "|"
            + signature
            + "|"
            + score.format(signature=signature)
            + "|"
            + score.format(signature=signature, is_json=True)
        )
    return out


SER = {"BLEU": ser_bleu, "CHRF": ser_chrf, "sentence_bleu": ser_bleu, "sentence_chrf": ser_chrf}


def outcome(fn: Callable[[], str]) -> str:
    """The serialized result, or ``raise:<ExceptionType>`` (an exception is a result too)."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"raise:{type(e).__name__}"


# ----------------------------------------------------------------------------------- the suite contract


class Metrics:
    """What the suite calls. ``mod`` is a package with ``rouge_scorer`` and ``sacrebleu`` (``pyroid.metrics``);
    the reference adapter passes rouge_score's and sacrebleu's own modules instead. Batch entry points
    (``RougeScorer.score_batch``, ``sentence_bleu_batch``, ``sentence_chrf_batch``) are used where present."""

    def __init__(self, rouge_scorer: Any, sacrebleu: Any, notes: Sequence[str] = ()) -> None:
        self.rouge_scorer = rouge_scorer
        self.sacrebleu = sacrebleu
        self.notes = list(notes)

    @classmethod
    def from_package(cls, mod: Any) -> Metrics:
        name = mod.__name__
        return cls(
            importlib.import_module(f"{name}.rouge_scorer"), importlib.import_module(f"{name}.sacrebleu")
        )

    # ROUGE
    def rouge(self, case: str, targets: list[Any], preds: list[str], batch: bool = False) -> list[str]:
        _, kwargs, how = ROUGE_CASES[case]
        scorer = self.rouge_scorer.RougeScorer(**kwargs)
        if how == "multi":
            return [
                outcome(lambda t=t, p=p: ser_rouge(scorer.score_multi(t, p)))
                for t, p in zip(targets, preds, strict=True)
            ]
        if batch and hasattr(scorer, "score_batch"):
            return [ser_rouge(r) for r in scorer.score_batch(targets, preds)]
        return [
            outcome(lambda t=t, p=p: ser_rouge(scorer.score(t, p)))
            for t, p in zip(targets, preds, strict=True)
        ]

    # BLEU / chrF, one corpus per (language pair, system)
    def corpus(self, case: str, key: str, hyps: list[str], refs: list[list[str]]) -> str:
        cls_name, kwargs, which = CORPUS_CASES[case]
        kwargs = dict(kwargs)
        if kwargs.get("trg_lang") == "<tgt>":
            kwargs["trg_lang"] = key.split("/")[0].split("-")[1]

        def run() -> str:
            metric = getattr(self.sacrebleu, cls_name)(**kwargs)
            score = metric.corpus_score(hyps, refs_for(refs, which))
            return SER[cls_name](score, str(metric.get_signature()))

        return outcome(run)

    # sentence level, every segment
    def sentence(self, case: str, hyps: list[str], refs: list[list[str]], batch: bool = False) -> list[str]:
        fn_name, kwargs, which = SENTENCE_CASES[case]
        ser = SER[fn_name]
        batch_fn = getattr(self.sacrebleu, f"{fn_name}_batch", None)
        if batch and batch_fn is not None:
            # One batch call per run of segments with the same number of references (streams can't have holes
            # here: element i must equal the per-segment call with refs[i]).
            out: list[str] = []
            i = 0
            while i < len(hyps):
                n = 1 if which == "A" else len(refs[i])
                j = i
                while j < len(hyps) and (which == "A" or len(refs[j]) == n):
                    j += 1
                streams = [[r[k] for r in refs[i:j]] for k in range(n)]
                out += [ser(s) for s in batch_fn(hyps[i:j], streams, **kwargs)]
                i = j
            return out
        fn = getattr(self.sacrebleu, fn_name)
        return [
            outcome(lambda h=h, r=r: ser(fn(h, r[:1] if which == "A" else r, **kwargs)))
            for h, r in zip(hyps, refs, strict=True)
        ]


# ---------------------------------------------------------------------------------- reference generation


def _rouge_chunk(job: tuple[str, list[Any], list[str]]) -> list[str]:
    from pyroid_parity.adapters.metrics_rouge_score import make

    case, targets, preds = job
    return make().rouge(case, targets, preds)


def _corpus_chunk(job: tuple[str, list[tuple[str, list[str], list[list[str]]]]]) -> list[str]:
    from pyroid_parity.adapters.metrics_rouge_score import make

    case, items = job
    impl = make()
    return [impl.corpus(case, key, h, r) for key, h, r in items]


def _sentence_chunk(job: tuple[str, list[str], list[list[str]]]) -> list[str]:
    from pyroid_parity.adapters.metrics_rouge_score import make

    case, hyps, refs = job
    return make().sentence(case, hyps, refs)


def corpus_case_applies(case: str, key: str) -> bool:
    """``bleu@trg_lang`` skips the MeCab languages (their tokenizer is delegated, not ported)."""
    return not (case == "bleu@trg_lang" and key.split("/")[0].split("-")[1] in DELEGATED_LANGS)


def build_reference(workers: int, pmap: Callable[..., list[Any]]) -> Any:
    """The reference table: ``case, key, out`` for every ROUGE pair, corpus and segment case."""
    import pyarrow as pa

    cases: list[str] = []
    keys: list[str] = []
    outs: list[str] = []
    chunk = 250
    for case, (data, _, _) in ROUGE_CASES.items():
        k, t, p = rouge_pairs(data)
        jobs = [(case, t[i : i + chunk], p[i : i + chunk]) for i in range(0, len(k), chunk)]
        res = [x for part in pmap(_rouge_chunk, jobs, workers) for x in part]
        cases += [case] * len(k)
        keys += k
        outs += res
    sys_list = systems()
    for case in CORPUS_CASES:
        items = [s for s in sys_list if corpus_case_applies(case, s[0])]
        jobs = [(case, items[i : i + 4]) for i in range(0, len(items), 4)]
        res = [x for part in pmap(_corpus_chunk, jobs, workers) for x in part]
        cases += [case] * len(items)
        keys += [s[0] for s in items]
        outs += res
    w = wmt21()
    seg_keys = segment_keys()
    for case in SENTENCE_CASES:
        hyps, refs = w["hyp"], w["refs"]
        jobs = [(case, hyps[i : i + 2000], refs[i : i + 2000]) for i in range(0, len(hyps), 2000)]
        res = [x for part in pmap(_sentence_chunk, jobs, workers) for x in part]
        cases += [case] * len(hyps)
        keys += seg_keys
        outs += res
    return pa.table({"case": cases, "key": keys, "out": outs})


def _pmap_jobs(fn: Callable[[Any], list[Any]], jobs: list[Any], workers: int) -> list[list[Any]]:
    """Order-preserving process map over prepared jobs (fork, so workers share the loaded data)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        return list(ex.map(fn, jobs))


def ref_metrics(pin: bool, workers: int) -> None:
    """``python -m pyroid_parity.refs metrics``: rouge-score + sacrebleu on the pinned eval data."""
    from pyroid_parity import refs as R

    table = build_reference(workers, _pmap_jobs)
    R.write(
        table,
        REF,
        pin,
        oracle=R.versions(*ORACLES),
        corpora=[CNNDM, SUMMEVAL, WMT21],
        cases={
            "rouge": {c: {"data": d, "kwargs": k, "call": h} for c, (d, k, h) in ROUGE_CASES.items()},
            "corpus": {c: {"class": m, "kwargs": k, "refs": r} for c, (m, k, r) in CORPUS_CASES.items()},
            "sentence": {c: {"call": f, "kwargs": k, "refs": r} for c, (f, k, r) in SENTENCE_CASES.items()},
        },
    )


@cache
def reference_table() -> dict[str, dict[str, list[str]]]:
    """Pinned reference outputs: ``{case: {"key": [...], "out": [...]}}``."""
    t = D.read_parquet_columns(D.path(REF), ["case", "key", "out"])
    out: dict[str, dict[str, list[str]]] = {}
    for case, key, res in zip(t["case"], t["key"], t["out"], strict=True):
        c = out.setdefault(case, {"key": [], "out": []})
        c["key"].append(key)
        c["out"].append(res)
    return out
