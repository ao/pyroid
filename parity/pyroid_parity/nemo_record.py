"""Record NeMo Curator 1.4.0's outputs on dev + adversarial: the ``nemo`` step of ``pyroid_parity.refs``.

    PYROID_DATA=/target/parity/data python -m pyroid_parity.refs nemo --pin --workers 4   # NeMo env, Linux

Runs in the NeMo env (README, "The nemo suite"), not in parity/pyproject.toml's groups, so it is not part of
``refs all``. The step re-runs itself with ``PYTHONHASHSEED=0`` when the variable is not set to that:
RepeatingTopNGramsFilter breaks ties among the most frequent n-grams by set iteration order, which follows the
string hash seed.

For every case of ``pyroid_parity.nemo_cases`` and every document of the case (dev 10k + adversarial 3,232, in
that order; the ``lang="zh"`` cases on the adversarial set's Mandarin and edge-case documents), one row:

- filters: ``score_document``'s value (``type`` and exact ``score`` text) and ``keep_document`` of it, or the
  exception (``error`` type and ``message``);
- modifiers: a digest, the length and ``changed`` of ``modify_document``'s output;
- FastTextLangId (lid.176.bin, lid.176.ftz): the score string; FastTextQualityFilter: the score, and ``keep``
  from one filter built with ``seed=42`` and called once per document in order (numpy's global RNG).

The quality model is trained here with fasttext 0.9.3 on the dev slice (``nemo_cases.QUALITY_RECIPE`` and
``QUALITY_LABEL_RULE``: the labels are NeMo's own heuristic pipeline decisions) and pinned as
``models/nemo-quality-dev10k.bin``. Without ``--pin`` the model is retrained and compared with the pin, and
the pinned model scores the documents.
"""

from __future__ import annotations

import importlib.metadata as md
import importlib.resources
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from pyroid_parity import data as D
from pyroid_parity import nemo_cases as C

# Corpus texts, set before the process pool forks so that workers share them.
_TEXTS: list[str] = []
_OBJ: dict[str, Any] = {}


def _log(msg: str) -> None:
    from pyroid_parity.refs import log

    log(msg)


def check_oracle() -> dict[str, str]:
    """The installed oracle versions; exits unless they are ``nemo_cases.ORACLE``."""
    got = {d: md.version(d) for d in C.ORACLE}
    if bad := {d: v for d, v in got.items() if v != C.ORACLE[d]}:
        sys.exit(f"[refs] the nemo references need {C.ORACLE}, found {bad}")
    return {**got, "python": platform.python_version()}


def check_yaml() -> None:
    """``nemo_cases.HEURISTIC_PIPELINE`` is NeMo's heuristic_filter_english_pipeline.yaml, stage for stage."""
    from omegaconf import OmegaConf

    path = (
        importlib.resources.files("nemo_curator")
        / "config"
        / "text"
        / "heuristic_filter_english_pipeline.yaml"
    )
    cfg = OmegaConf.load(str(path))
    got = []
    for stage in cfg.stages:
        if stage["_target_"].endswith(".ScoreFilter"):
            obj = OmegaConf.to_container(stage.filter_obj)
            got.append((obj.pop("_target_").rsplit(".", 1)[1], obj))
    want = [(c.nemo, c.nemo_kwargs) for c in C.HEURISTIC_PIPELINE]
    if got != want:
        sys.exit(f"[refs] nemo_cases.HEURISTIC_PIPELINE differs from {path}: {got} vs {want}")


def _obj(case: str) -> Any:
    if case not in _OBJ:
        c = C.ALL_CASES[case]
        if isinstance(c, C.FastTextCase):
            obj = c.make(model_path=str(D.path(c.model)))
            obj.load_model()
        else:
            obj = c.make()
        _OBJ[case] = obj
    return _OBJ[case]


Row = tuple[Any, ...]  # nemo_cases.REF_COLUMNS[2:]


def outcome(case: str, text: str, keep: bool = True) -> Row:
    """One reference row (without ``case`` and ``id``) for ``text``."""
    obj = _obj(case)
    kind = C.kind_of(case)
    try:
        if kind == "modifier":
            out = obj.modify_document(text)
            return ("ok", type(out).__name__, None, None, None, None, C.digest(out), len(out), out != text)
        value = obj.score_document(text)
        k = bool(obj.keep_document(value)) if keep else None
    except Exception as e:  # noqa: BLE001  the exception is part of the reference
        return ("raise", None, None, None, type(e).__name__, str(e), None, None, None)
    t, s = C.score_cell(value)
    return ("ok", t, s, k, None, None, None, None, None)


def _chunk(job: list[tuple[str, int]]) -> list[Row]:
    # FastTextQualityFilter.keep_document draws from numpy's global RNG: keep is drawn in order afterwards.
    return [outcome(case, _TEXTS[i], keep=case != "quality") for case, i in job]


def _train_quality(dev_texts: list[str], labels: list[bool], out: Path) -> None:
    import fasttext
    from nemo_curator.stages.text.modifiers.fasttext import FastTextLabelModifier

    hq, lq = FastTextLabelModifier(C.QUALITY_LABEL), FastTextLabelModifier("__label__lq")
    with tempfile.TemporaryDirectory(prefix="pyroid-nemo-") as tmp:
        train = Path(tmp) / "train.txt"
        with train.open("w", encoding="utf-8") as f:
            for text, good in zip(dev_texts, labels, strict=True):
                f.write((hq if good else lq).modify_document(text) + "\n")
        model = fasttext.train_supervised(input=str(train), **C.QUALITY_RECIPE)
        model.quantize(input=str(train), **C.QUALITY_QUANTIZE)
        model.save_model(str(out))


def quality_model(pin: bool, dev_texts: list[str], labels: list[bool]) -> Path:
    """Train the quality model; pin it (``--pin``) or check it against the pin. Returns the model to score with."""
    from pyroid_parity import refs

    rel = C.QUALITY_MODEL
    dest = D.DATA / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    trained = dest.with_suffix(".trained")
    _log(f"training {rel}: {sum(labels)} of {len(labels)} dev docs labelled hq, {C.QUALITY_RECIPE}")
    _train_quality(dev_texts, labels, trained)
    sha = D.sha256_file(trained)
    n_hq = sum(labels)
    if pin:
        trained.replace(dest)
        D.pin(
            rel,
            oracle={"fasttext": md.version("fasttext"), "nemo-curator": md.version("nemo-curator")},
            recipe={"train_supervised": C.QUALITY_RECIPE, "quantize": C.QUALITY_QUANTIZE},
            labels=C.QUALITY_LABEL_RULE,
            label_counts={C.QUALITY_LABEL: n_hq, "__label__lq": len(labels) - n_hq},
            train_docs=D.CORPORA["dev"],
            trained_on=D.platform_key(),
            generator="pyroid_parity.nemo_record:quality_model",
        )
        D._verified.cache_clear()
        _log(f"pinned {rel}: sha256 {sha}, {n_hq} of {len(labels)} dev docs labelled hq")
        return dest
    entry = D.load_manifest()["files"].get(rel)
    if entry is None or entry["sha256"] != sha:
        refs.MISMATCHED.append(rel)
        _log(f"MISMATCH {rel}: retrained model has sha256 {sha}, manifest has {entry and entry['sha256']}")
    if entry is not None and D.available(rel):
        trained.unlink()
        return D.path(rel)
    trained.replace(dest)  # nothing pinned to fall back on: score with the model trained here
    return dest


def ref_nemo(pin: bool, workers: int) -> None:
    """The ``nemo`` step: re-runs itself with ``PYTHONHASHSEED=0`` unless this process already has it."""
    from pyroid_parity import refs

    rel = C.nemo_ref(D.platform_key())
    if not C.hash_seed_ok():
        _log(f"nemo: re-running with {C.HASH_SEED_ENV}={C.HASH_SEED}")
        cmd = [sys.executable, "-m", "pyroid_parity.refs", "nemo", "--workers", str(workers)]
        env = {**os.environ, C.HASH_SEED_ENV: C.HASH_SEED}
        proc = subprocess.run([*cmd, *(["--pin"] if pin else [])], env=env, check=False)
        if proc.returncode:
            refs.MISMATCHED.append(rel)
        return
    record(pin, workers)


def record(pin: bool, workers: int) -> None:
    import pyarrow as pa

    from pyroid_parity import refs

    oracle = check_oracle()
    check_yaml()
    docs = refs.all_docs()
    n_dev = len(D.corpus("dev"))
    _TEXTS[:] = docs.texts
    by_case: dict[str, list[int]] = {c: C.doc_indices(C.ALL_CASES[c].docs, docs.meta) for c in C.ALL_CASES}
    first = [c for c in C.ALL_CASES if c != "quality"]
    jobs = [(c, i) for c in first for i in by_case[c]]
    _log(f"nemo: {len(first)} cases, {len(jobs)} documents on {workers} workers")
    results: dict[str, list[Row]] = {c: [] for c in C.ALL_CASES}
    for (c, _), r in zip(jobs, refs.pmap(_chunk, jobs, workers, chunk=250), strict=True):
        results[c].append(r)

    # Quality model: labels from the recorded heuristic pipeline decisions on the dev slice.
    labels = [
        all(results[c.case][i][0] == "ok" and results[c.case][i][3] for c in C.HEURISTIC_PIPELINE)
        for i in range(n_dev)
    ]
    model = quality_model(pin, docs.texts[:n_dev], labels)
    _OBJ.pop("quality", None)
    jobs = [("quality", i) for i in by_case["quality"]]
    results["quality"] = refs.pmap(_chunk, jobs, workers, chunk=250)
    keeper = C.FASTTEXT_CASES["quality"].make(model_path=str(model))  # seeds numpy's global RNG
    results["quality"] = [
        (*r[:3], bool(keeper.keep_document(float(r[2]))), *r[4:]) if r[0] == "ok" else r
        for r in results["quality"]
    ]

    cols: dict[str, list[Any]] = {k: [] for k in C.REF_COLUMNS}
    raises: dict[str, dict[str, int]] = {}
    for c in C.ALL_CASES:
        for i, r in zip(by_case[c], results[c], strict=True):
            for k, v in zip(C.REF_COLUMNS, (c, docs.ids[i], *r), strict=True):
                cols[k].append(v)
        if errs := Counter(r[4] for r in results[c] if r[0] == "raise"):
            raises[c] = dict(sorted(errs.items()))
    table = pa.table(
        {
            **cols,
            "keep": pa.array(cols["keep"], pa.bool_()),
            "length": pa.array(cols["length"], pa.int64()),
            "changed": pa.array(cols["changed"], pa.bool_()),
        }
    )
    manifest = D.load_manifest()["files"]
    refs.write(
        table,
        C.nemo_ref(D.platform_key()),
        pin,
        oracle=oracle,
        hash_seed=C.HASH_SEED,
        platform=D.platform_key(),
        corpora=["dev", "adversarial"],
        models={
            Path(rel).name: (D.sha256_file(model) if rel == C.QUALITY_MODEL else manifest[rel]["sha256"])
            for rel in (*C.LID_MODELS.values(), C.QUALITY_MODEL)
        },
        cases={
            c: {"class": v.nemo, "kwargs": v.nemo_kwargs, "docs": v.docs, "rows": len(by_case[c])}
            for c, v in C.ALL_CASES.items()
        },
        raises=raises,
        quality_keep=C.QUALITY_KEEP_ORDER,
        generator="pyroid_parity.nemo_record:record",
    )
