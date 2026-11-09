"""warc parity: the ``--impl`` candidate vs warcio 1.8.2 and datatrove 0.10.1 (live) and their pinned digests.

    pytest tests/test_warc.py                                  # self-check (candidate = warcio)
    pytest tests/test_warc.py --impl pyroid.warc                # pyroid.warc vs warcio + datatrove
    pytest tests/test_warc.py --impl pyroid.warc --parity-limit 0   # every record of the Common Crawl slices

Inputs (``pyroid_parity.warc_corpus``, ``warc_synth``): 34 Common Crawl slices (WARC 2013-2026, WET, WAT, the 2012
ARC crawl; whole members, truncated tails, a start inside a member, an empty file), warcio's 16 test files, and
about 1,100 seeded synthetic files in 13 families (framing, truncation, bit flips, lengths, headers, HTTP,
chunked, encodings, record types, digests, ARC, charsets, big records). Each runs under a set of
``warc_compare.Config`` s (iterator options, read plans, openers). Traces are compared event for event
(``warc_compare``): records, reads, offsets, stderr and log messages, exceptions and what follows them.

Checks (suite ``warc`` in parity-report.json):
    inputs                    pinned files match bench/manifest.json; the synthetic set matches its pinned sha256
    oracle                    warcio 1.8.2, datatrove 0.10.1, brotlicffi 1.2.1.0, faust-cchardet 3.2.0,
                              python-magic 0.4.27 where installed
    records-<group>           candidate == live warcio, per job (main events)
    io                        the same jobs' ``io`` events: ``fh.tell()`` after each record, the reads the
                              ``logged`` opener sees
    records-<group>-pinned    the trace digests == ``data/warcio-1.8.2/ref-digests.json`` (warcio itself in the
                              self-check; the candidate where warcio is not installed)
    digest-other-algorithms   digests with algorithms hashlib knows but pyroid does not port: warcio's outcome,
                              or warcio's outcome with the algorithm treated as unknown (pyroid's documented one)
    hypothesis-records        grammar-generated WARC streams (``warc_strategies``) under a drawn config
    documents-<group>         datatrove's ``process_record`` over the candidate's ``ArchiveIterator`` (on what
                              WarcReader opens) == over warcio's
    iter-documents-<group>    ``iter_documents(path)`` == datatrove's read path (``warc_datatrove``)
    iter-documents-fileobj    ``iter_documents(file object, compression="gzip" / None)``
    documents-<group>-pinned  datatrove's read path == the pinned digests (same libmagic only)
    batches-documents         ``iter_batches`` rows with text == the documents; ``batches-columns``: their html,
                              mime_type and content_type; ``batches-rows``: the rows without text
    warcreader                ``<top>.integrations.datatrove.WarcReader(folder, **kw).run()`` == datatrove's
    warcreader-executor       the same through ``LocalPipelineExecutor`` + ``JsonlWriter``
    hypothesis-documents      grammar-generated streams through both datatrove paths
    api-*                     names, ``WARCIO_VERSION``, signatures; ``StatusAndHeaders``, ``DigestChecker`` and
                              ``ArchiveLoadFailed`` behave as warcio's (Hypothesis on header lists)
    mutant                    (self-check only) deliberately broken warcio wrappers are caught

Without ``--impl`` the candidate is warcio itself: live checks are skipped (they would compare warcio with
itself), the pinned checks verify that this interpreter reproduces the pins, and the mutant check proves the
comparison catches differences. ``--parity-limit N`` reads the first N records of each Common Crawl slice
(default 100, the pinned head; 0 = all). Set ``PYROID_PARITY_REPROS=DIR`` to copy every input with a diff there,
with a JSON note of the config and the first differing event.
"""

from __future__ import annotations

import bz2
import contextlib
import hashlib
import importlib.metadata as md
import inspect
import io
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pyroid_parity import data as D
from pyroid_parity import warc_compare as C
from pyroid_parity import warc_corpus as W
from pyroid_parity import warc_datatrove as T
from pyroid_parity import warc_pins as P
from pyroid_parity import warc_strategies as WS
from pyroid_parity import warc_synth as S
from pyroid_parity.plugin import _impl, reference
from pyroid_parity.report import Collector

SUITE = "warc"
# The digest algorithms pyroid.warc ports (RustCrypto); hashlib may know more names (documented difference).
PORTED_HASHES = ("md5", "sha1", "sha224", "sha256", "sha384", "sha512")
# What the record traces depend on besides warcio: CPython's gzip module (the GzipFile openers), zlib (the synthetic
# gzip bytes, its error messages) and brotlicffi (the synthetic br bytes). The pinned checks run only where they match.
RECORD_ENV = ("python", "zlib", "brotlicffi")
ORACLE = {
    "warcio": "1.8.2",
    "datatrove": "0.10.1",
    "brotlicffi": "1.2.1.0",
    "faust-cchardet": "3.2.0",
    "python-magic": "0.4.27",
}
MAX_DETAIL = 8
GROUPS = P.groups()
CC_NAMES = [f.rel for f in W.cc_files()]
REPROS = os.environ.get("PYROID_PARITY_REPROS")


@pytest.fixture(scope="session")
def warc_impl(request: pytest.FixtureRequest) -> Any:
    return _impl(request, SUITE)


@cache
def oracle() -> Any | None:
    """warcio behind the contract, if it is installed here."""
    try:
        from pyroid_parity.adapters import warc_warcio

        return warc_warcio.make()
    except ImportError:
        return None


@cache
def has_datatrove() -> bool:
    try:
        import datatrove.pipeline.readers.warc  # noqa: F401
    except ImportError:
        return False
    T.ensure_magic()
    try:
        import cchardet  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(scope="module", autouse=True)
def _oracle_info() -> None:
    try:
        v = md.version("warcio")
    except md.PackageNotFoundError:
        v = "1.8.2 (pinned)"
    reference(SUITE, "warcio", v, None)


def record_limit(request: pytest.FixtureRequest) -> int:
    raw = request.config.getoption("--parity-limit")
    return P.HEAD if raw is None else int(raw)


def skip(parity: Collector, check: str, why: str) -> None:
    parity.skip(SUITE, check, why)
    pytest.skip(why)


def save_repro(job_name: str, path: Path, config: str, check: str, diff: C.Diff | None) -> None:
    if not REPROS:
        return
    out = Path(REPROS)
    out.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in job_name)
    target = out / (safe + "".join(path.suffixes[-2:]))
    if not target.exists():
        shutil.copyfile(path, target)
    note = out / (safe + ".json")
    notes = json.loads(note.read_text("utf-8")) if note.exists() else []
    notes.append({"check": check, "config": config, "diff": None if diff is None else diff.short()})
    note.write_text(json.dumps(notes, indent=1, ensure_ascii=False) + "\n", "utf-8")


# ------------------------------------------------------------------------------------------------ inputs


def test_oracle_versions(parity: Collector) -> None:
    missing, bad = [], []
    for dist, want in ORACLE.items():
        try:
            got = md.version(dist)
        except md.PackageNotFoundError:
            missing.append(dist)
            continue
        if got != want:
            bad.append(f"{dist} {got} (pinned {want})")
    parity.record(SUITE, "oracle", not bad, got=bad, want=ORACLE)
    assert not bad, f"oracle versions differ from the pins: {bad}"
    if missing:
        parity.skip(SUITE, "oracle-missing", f"not installed here: {missing}")


def test_inputs(parity: Collector) -> None:
    if not P.HAVE_DATATROVE:
        parity.skip(
            SUITE,
            "datatrove-opener",
            "datatrove is not installed: configs reading through DataFolder.open are left out",
        )
    gone = W.missing()
    if gone:
        skip(
            parity,
            "inputs",
            f"{len(gone)} pinned warc inputs missing (bench/fetch.py --only warc): {gone[:3]}",
        )
    parity.uses(SUITE, *(i.name for i in W.pinned_inputs()))
    pins = P.load()
    assert pins is not None, f"{P.PINS} is missing"
    if why := P.mismatch(pins, ("zlib", "brotlicffi")):
        skip(parity, "inputs", f"the synthetic gzip and brotli bytes depend on the library: {why}")
    got = S.inputs_sha256(S.cases())
    same = parity.record(SUITE, "inputs", got == pins["synth_sha256"], got=got, want=pins["synth_sha256"])
    assert same, (
        "warc_synth generates other bytes than the pinned set (zlib or brotlicffi differ?): "
        f"{P.env()} vs {pins['env']}"
    )
    assert sorted(i.name for i in W.pinned_inputs()) == pins["inputs"]


# ------------------------------------------------------------------------------------------------ records


def compare_traces(
    parity: Collector, check: str, job: P.Job, got: list[C.Event], want: list[C.Event], diffs: list[str]
) -> None:
    gm, gi = C.split(got)
    wm, wi = C.split(want)
    case = f"{job.name} [{job.config.name}]"
    d = C.first_diff(gm, wm)
    if not parity.record(
        SUITE, check, d is None, case=case, input=job.path.name, got=d and d.got, want=d and d.want
    ):
        diffs.append(f"{case}: event {d.index}: got {d.got!r:.300} want {d.want!r:.300}")  # type: ignore[union-attr]
        save_repro(job.name, job.path, job.config.name, check, d)
    di = C.first_diff(gi, wi)
    if not parity.record(SUITE, "io", di is None, case=case, got=di and di.got, want=di and di.want):
        save_repro(job.name, job.path, job.config.name, "io", di)


def _pinned(
    parity: Collector, check: str, case: str, got_pair: str, want_pair: str | None, bad: list[str]
) -> None:
    """Record a (main, io) digest pair against its pin; main diffs go to ``bad``."""
    g = got_pair.split(",")
    w = (want_pair or ",").split(",")
    if not parity.record(SUITE, f"{check}-pinned", g[0] == w[0], case=case, got=g[0], want=w[0]):
        bad.append(case)
    parity.record(SUITE, "io-pinned", g[1] == w[1], case=case, got=g[1], want=w[1])


def run_group(parity: Collector, impl: Any, group: str, limit: int, jobs: list[P.Job]) -> None:
    """Live comparison (candidate != warcio) or pinned comparison (self-check, or warcio missing)."""
    o = oracle()
    cand = impl.obj
    pins = P.load()
    check = f"records-{group}"
    n = limit if P.limited(group) else 0
    diffs: list[str] = []
    pinned_diffs: list[str] = []
    live = o is not None and not impl.is_reference
    if not live:
        why = "candidate is warcio itself (the -pinned check covers it)" if impl.is_reference else "no warcio"
        parity.skip(SUITE, check, why)
        if pins is None:
            skip(parity, f"{check}-pinned", f"{P.PINS} is missing")
        if why := P.mismatch(pins, RECORD_ENV):
            skip(parity, f"{check}-pinned", why)
        if (key := P.pin_key(group, limit)) is None:
            skip(parity, f"{check}-pinned", f"pinned for {P.HEAD} records per slice or all, not {limit}")
    folded: dict[str, list[str]] = {}
    for job in jobs:
        got = C.iter_events(cand.mod, job.path, job.config, n, cand.exc_name)
        if live:
            want = C.iter_events(o.mod, job.path, job.config, n, o.exc_name)
            compare_traces(parity, check, job, got, want, diffs)
        elif P.per_input(group):
            entry = pins["records"].get(group, {}).get(job.name, {}).get(job.config.name)  # type: ignore[index]
            want_pair = P.unpack(entry).get(key) if entry else None
            _pinned(parity, check, f"{job.name} [{job.config.name}]", P.pair(got), want_pair, pinned_diffs)
        else:
            folded.setdefault(job.config.name, []).append(P.pair(got))
    for name, parts in folded.items():
        entry = pins["records"].get(group, {}).get(name)  # type: ignore[index]
        _pinned(
            parity,
            check,
            f"{group} [{name}]",
            P.fold(parts),
            P.unpack(entry).get(key) if entry else None,
            pinned_diffs,
        )
    assert not diffs, f"{len(diffs)} jobs differ from warcio, first:\n" + "\n".join(diffs[:MAX_DETAIL])
    assert not pinned_diffs, f"{len(pinned_diffs)} differ from the pins: {pinned_diffs[:MAX_DETAIL]}"


def _need_inputs(parity: Collector, check: str) -> None:
    if W.missing():
        skip(
            parity, check, f"pinned warc inputs missing here (bench/fetch.py --only warc): {W.missing()[:3]}"
        )
    if P.load() is None:
        skip(parity, check, f"{P.PINS} is missing")


@pytest.mark.parametrize("rel", CC_NAMES)
def test_records_cc(warc_impl: Any, parity: Collector, request: pytest.FixtureRequest, rel: str) -> None:
    _need_inputs(parity, "records-cc")
    jobs = [j for j in P.jobs("cc") if j.name == rel]
    run_group(parity, warc_impl, "cc", record_limit(request), jobs)


@pytest.mark.parametrize("group", [g for g in GROUPS if g != "cc"])
def test_records(warc_impl: Any, parity: Collector, group: str) -> None:
    _need_inputs(parity, f"records-{group}")
    jobs = list(P.jobs(group))
    run_group(parity, warc_impl, group, 0, jobs)


@cache
def _tags_by_name() -> dict[str, frozenset[str]]:
    return {c.name: c.tags for c in S.cases()}


def _tags(name: str) -> frozenset[str]:
    return _tags_by_name().get(name, frozenset())


def test_digest_other_algorithms(warc_impl: Any, parity: Collector) -> None:
    """Algorithms hashlib knows but pyroid does not port: warcio's outcome or the "unknown" one."""
    check = "digest-other-algorithms"
    o = oracle()
    if o is None or warc_impl.is_reference:
        skip(parity, check, "candidate is warcio itself" if o else "warcio is not installed")
    from warcio import utils as wu

    jobs = list(P.jobs("synth-digest", other_digests=True))
    diffs = []
    for job in jobs:
        got = C.split(C.iter_events(warc_impl.obj.mod, job.path, job.config, 0, warc_impl.obj.exc_name))[0]
        want = C.split(C.iter_events(o.mod, job.path, job.config, 0, o.exc_name))[0]
        if got == want:
            parity.record(SUITE, check, True, case=job.name)
            continue
        real = wu.Digester.__init__

        def unknown(self: Any, type_: str = "sha1", real: Any = real) -> None:
            if hashlib.new(type_).name not in PORTED_HASHES:
                raise ValueError(f"unsupported hash type {type_}")
            real(self, type_)

        wu.Digester.__init__ = unknown  # type: ignore[method-assign]
        try:
            alt = C.split(C.iter_events(o.mod, job.path, job.config, 0, o.exc_name))[0]
        finally:
            wu.Digester.__init__ = real  # type: ignore[method-assign]
        d = C.first_diff(got, alt)
        if not parity.record(
            SUITE,
            check,
            d is None,
            case=f"{job.name} [{job.config.name}]",
            got=d and d.got,
            want=d and d.want,
        ):
            diffs.append(f"{job.name} [{job.config.name}]: {d.short()}")  # type: ignore[union-attr]
    assert not diffs, "\n".join(diffs[:MAX_DETAIL])


# ------------------------------------------------------------------------------------------- Hypothesis


@pytest.fixture(scope="module")
def hyp_dir() -> Iterator[Path]:
    d = Path(tempfile.mkdtemp(prefix="pyroid-warc-hyp-"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@settings(
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
        HealthCheck.data_too_large,
    ]
)
@given(f=WS.warc_files(), which=st.integers(0, len(C.SYNTH_CONFIGS) - 1))
def test_hypothesis_records(
    warc_impl: Any, parity: Collector, hyp_dir: Path, f: tuple[bytes, str], which: int
) -> None:
    check = "hypothesis-records"
    o = oracle()
    if o is None:
        skip(parity, check, "warcio is not installed")
    data, suffix = f
    gz = suffix.endswith(".gz")
    configs = P.configs_for(gz, C.SYNTH_CONFIGS)
    cfg = configs[which % len(configs)]
    path = hyp_dir / f"h{suffix}"
    path.write_bytes(data)
    job = P.Job("hypothesis", "hypothesis", path, gz, cfg)
    got = C.iter_events(warc_impl.obj.mod, path, cfg, 0, warc_impl.obj.exc_name)
    want = C.iter_events(o.mod, path, cfg, 0, o.exc_name)
    diffs: list[str] = []
    compare_traces(parity, check, job, got, want, diffs)
    assert not diffs, f"{cfg.name} on {data!r:.400}: {diffs[0]}"


# ------------------------------------------------------------------------------------------- datatrove


def _need_datatrove(parity: Collector, check: str) -> None:
    if not has_datatrove():
        skip(parity, check, "datatrove 0.10.1 + faust-cchardet are not installed here (parity group `warc`)")


def compare_events(
    parity: Collector,
    check: str,
    case: str,
    path: Path,
    got: list[C.Event],
    want: list[C.Event],
    diffs: list[str],
) -> None:
    d = C.first_diff(got, want)
    if not parity.record(
        SUITE, check, d is None, case=case, input=path.name, got=d and d.got, want=d and d.want
    ):
        diffs.append(f"{case}: event {d.index}: got {d.got!r:.300} want {d.want!r:.300}")  # type: ignore[union-attr]
        save_repro(case, path, "documents", check, d)


@pytest.mark.parametrize("group", GROUPS)
def test_documents(warc_impl: Any, parity: Collector, request: pytest.FixtureRequest, group: str) -> None:
    _need_inputs(parity, f"documents-{group}")
    _need_datatrove(parity, f"documents-{group}")
    cand = warc_impl.obj
    limit = record_limit(request) if P.limited(group) else 0
    ref_name = T.default_exc_name()
    inputs = P.document_inputs(group)
    diffs: list[str] = []
    if warc_impl.is_reference:
        parity.skip(SUITE, f"documents-{group}", "candidate is warcio itself (the -pinned check covers it)")
        skip(parity, f"iter-documents-{group}", "candidate is warcio itself")
    if not cand.has_documents:
        parity.skip(SUITE, f"iter-documents-{group}", "the candidate has no iter_documents")
    for name, path in inputs:
        want = T.events_of(lambda path=path: T.reference_documents(path), T.doc_view, ref_name, limit)
        got = T.events_of(
            lambda path=path: T.via_process_record(cand.mod, path), T.doc_view, cand.exc_name, limit
        )
        compare_events(parity, f"documents-{group}", name, path, got, want, diffs)
        if cand.has_documents:
            got = T.events_of(
                lambda path=path: cand.mod.iter_documents(path), T.doc_view, cand.exc_name, limit
            )
            compare_events(parity, f"iter-documents-{group}", name, path, got, want, diffs)
            if limit == 0:
                comp = "gzip" if path.name.endswith(".gz") else None
                with open(path, "rb") as f1, open(path, "rb") as f2:
                    got = T.events_of(
                        lambda f1=f1, comp=comp: cand.mod.iter_documents(f1, compression=comp),
                        T.doc_view,
                        cand.exc_name,
                    )
                    want2 = T.events_of(
                        lambda f2=f2, comp=comp: T.reference_documents(f2, compression=comp),
                        T.doc_view,
                        ref_name,
                    )
                compare_events(parity, "iter-documents-fileobj", name, path, got, want2, diffs)
    assert not diffs, f"{len(diffs)} differ, first:\n" + "\n".join(diffs[:MAX_DETAIL])


@pytest.mark.parametrize("group", GROUPS)
def test_documents_pinned(parity: Collector, request: pytest.FixtureRequest, group: str) -> None:
    check = f"documents-{group}-pinned"
    _need_inputs(parity, check)
    _need_datatrove(parity, check)
    pins = P.load()
    assert pins is not None
    if why := P.mismatch(pins, (*RECORD_ENV, "libmagic", "cchardet")):
        skip(parity, check, why)
    limit = record_limit(request)
    key = P.pin_key(group, limit)
    if key is None:
        skip(parity, check, f"pinned for {P.HEAD} records per slice or all, not {limit}")
    n = P.HEAD if key == "head" else 0
    ref_name = T.default_exc_name()
    bad = []
    if P.per_input(group):
        for name, path in P.document_inputs(group):
            got = P.document_digest(ref_name, path, n)
            want = P.document_pin(pins["documents"][group][name], key)
            if not parity.record(SUITE, check, got == want, case=name, got=got, want=want):
                bad.append(name)
    else:
        got = P.fold_one([P.document_digest(ref_name, p, n) for _, p in P.document_inputs(group)])
        want = P.document_pin(pins["documents"][group], key)
        if not parity.record(SUITE, check, got == want, case=group, got=got, want=want):
            bad.append(group)
    assert not bad, f"datatrove's read path differs from the pins on {bad[:MAX_DETAIL]}"


def _reference_rows(path: Path) -> list[list[Any]] | None:
    try:
        return list(T.reference_rows(path))
    except Exception:  # noqa: BLE001
        return None


BATCH_GROUPS = [
    "cc",
    "warcio-tests",
    "synth-charset",
    "synth-encoding",
    "synth-big",
    "synth-type",
    "synth-length",
]


@pytest.mark.parametrize("group", BATCH_GROUPS)
@pytest.mark.parametrize("batch_size", [1000, 7])
def test_batches(
    warc_impl: Any, parity: Collector, request: pytest.FixtureRequest, group: str, batch_size: int
) -> None:
    """Whole files; for the Common Crawl slices every fifth one unless ``--parity-limit 0``."""
    _need_inputs(parity, "batches-documents")
    _need_datatrove(parity, "batches-documents")
    cand = warc_impl.obj
    if not cand.has_batches:
        skip(parity, "batches-documents", "the candidate has no iter_batches")
    ref_name = T.default_exc_name()
    diffs: list[str] = []
    inputs = P.document_inputs(group)
    if P.limited(group) and record_limit(request) != 0:
        inputs = inputs[::5]
    for name, path in inputs:
        got = T.batch_events(cand.mod, path, batch_size, cand.exc_name)
        rows = [r for e in got if e[0] == "item" for r in e[4]]
        got_end = [e for e in got if e[0] != "item"]
        docs = T.events_of(lambda path=path: T.reference_documents(path), T.doc_view, ref_name)
        want_docs = [e for e in docs if e[0] == "item"]
        want_end = [e for e in docs if e[0] not in ("item",)]
        # Rows with text, in order, are the documents (id, url, date, text).
        with_text = [[r[0], r[1], r[2], r[6]] for r in rows if r[6] is not None]
        want_text = [[e[5]["id"], e[5]["url"], e[5]["date"], e[4]] for e in want_docs]
        ok = with_text == want_text[: len(with_text)] and (
            len(with_text) == len(want_text) or any(e[1] == "raise" for e in want_end if len(e) > 1)
        )
        if not parity.record(
            SUITE,
            "batches-documents",
            ok,
            case=f"{name} [{batch_size}]",
            got=with_text[:3],
            want=want_text[:3],
        ):
            diffs.append(
                f"{name} [{batch_size}]: {len(with_text)} rows with text, {len(want_text)} documents"
            )
        g_exc = [e[2:4] for e in got_end if len(e) > 1 and e[1] == "raise" and e[0] == "next"]
        w_exc = [e[2:4] for e in want_end if len(e) > 1 and e[1] == "raise" and e[0] == "next"]
        if not parity.record(SUITE, "batches-exception", g_exc == w_exc, case=name, got=g_exc, want=w_exc):
            diffs.append(f"{name}: exception {g_exc} vs {w_exc}")
        ref_rows = _reference_rows(path)
        if ref_rows is None:  # it raises: the documents check covers the exception
            continue
        full = [r for r in ref_rows if r[6] is not None]
        got_full = [r for r in rows if r[6] is not None]
        if not parity.record(
            SUITE, "batches-columns", got_full == full, case=name, got=got_full[:2], want=full[:2]
        ):
            diffs.append(f"{name}: columns of the rows with text differ")
        parity.record(
            SUITE, "batches-rows", len(rows) == len(ref_rows), case=name, got=len(rows), want=len(ref_rows)
        )
    assert not diffs, "\n".join(diffs[:MAX_DETAIL])


# ------------------------------------------------------------------------------------------- WarcReader


@cache
def reader_folders() -> dict[str, tuple[Path, dict[str, Any]]]:
    """name -> (folder, WarcReader kwargs). Synthetic families get one folder each."""
    out: dict[str, tuple[Path, dict[str, Any]]] = {}
    if not W.missing():
        out["cc"] = (D.path(W.cc_files()[0].rel).parent, {})
        out["cc-limit-skip"] = (
            D.path(W.cc_files()[0].rel).parent,
            {"limit": 50, "skip": 7, "glob_pattern": "CC-MAIN-201*"},
        )
        out["cc-no-compression"] = (
            D.path(W.cc_files()[0].rel).parent,
            {"compression": None, "glob_pattern": "*2026*"},
        )
        out["warcio-tests"] = (D.path(W.warcio_rel("example.warc")).parent, {})
    root = Path(tempfile.mkdtemp(prefix="pyroid-warc-readers-"))
    import atexit

    atexit.register(shutil.rmtree, root, True)
    for fam in P.families():
        d = root / fam
        d.mkdir()
        for c, p in P.synth_inputs(fam):
            os.link(p, d / c.filename)
        out[f"synth-{fam}"] = (d, {})
    other = root / "other-compressions"
    other.mkdir()
    sample = S.cases()[0].data
    (other / "a.warc.bz2").write_bytes(bz2.compress(sample))
    (other / "b.warc.gz").write_bytes(S.gz(sample))
    (other / "c.warc").write_bytes(sample)
    out["other-compressions"] = (other, {})
    out["gzip-forced"] = (other, {"compression": "gzip"})
    return out


READERS = [
    "cc",
    "cc-limit-skip",
    "cc-no-compression",
    "warcio-tests",
    *(f"synth-{f}" for f in P.families()),
    "other-compressions",
    "gzip-forced",
]


@pytest.mark.parametrize("folder", READERS)
def test_warcreader(warc_impl: Any, parity: Collector, folder: str) -> None:
    check = "warcreader"
    _need_datatrove(parity, check)
    cand = warc_impl.obj
    if warc_impl.is_reference:
        skip(parity, check, "candidate is warcio itself: both readers are datatrove's")
    if cand.reader_cls is None:
        skip(parity, check, "the candidate package has no integrations.datatrove.WarcReader")
    folders = reader_folders()
    if folder not in folders:
        skip(parity, check, "pinned warc inputs missing here")
    from datatrove.pipeline.readers import WarcReader

    path, kw = folders[folder]
    got = T.reader_events(cand.reader_cls, path, cand.exc_name, **kw)
    want = T.reader_events(WarcReader, path, T.default_exc_name(), **kw)
    diffs: list[str] = []
    compare_events(parity, check, f"{folder} {kw}", path, got, want, diffs)
    assert not diffs, diffs[0]


@pytest.mark.parametrize("folder", ["warcio-tests", "synth-charset", "synth-frame"])
def test_warcreader_executor(warc_impl: Any, parity: Collector, folder: str, tmp_path: Path) -> None:
    check = "warcreader-executor"
    _need_datatrove(parity, check)
    cand = warc_impl.obj
    if warc_impl.is_reference or cand.reader_cls is None:
        skip(
            parity,
            check,
            "candidate is warcio itself"
            if warc_impl.is_reference
            else "no integrations.datatrove.WarcReader",
        )
    folders = reader_folders()
    if folder not in folders:
        skip(parity, check, "pinned warc inputs missing here")
    from datatrove.pipeline.readers import WarcReader

    path, _ = folders[folder]
    got = T.executor_output(cand.reader_cls, path, tmp_path / "cand")
    want = T.executor_output(WarcReader, path, tmp_path / "ref")
    ok = parity.record(
        SUITE, check, got == want, case=folder, got=got[1] or len(got[0]), want=want[1] or len(want[0])
    )
    assert ok, f"{folder}: {len(got[0])} vs {len(want[0])} documents, errors {got[1]} vs {want[1]}"


@settings(
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
        HealthCheck.data_too_large,
    ]
)
@given(f=WS.warc_files())
def test_hypothesis_documents(warc_impl: Any, parity: Collector, hyp_dir: Path, f: tuple[bytes, str]) -> None:
    check = "hypothesis-documents"
    _need_datatrove(parity, check)
    if warc_impl.is_reference:
        skip(parity, check, "candidate is warcio itself")
    cand = warc_impl.obj
    data, suffix = f
    path = hyp_dir / f"d{suffix}"
    path.write_bytes(data)
    want = T.events_of(lambda: T.reference_documents(path), T.doc_view, T.default_exc_name())
    diffs: list[str] = []
    got = T.events_of(lambda: T.via_process_record(cand.mod, path), T.doc_view, cand.exc_name)
    compare_events(parity, check, "process_record", path, got, want, diffs)
    if cand.has_documents:
        got = T.events_of(lambda: cand.mod.iter_documents(path), T.doc_view, cand.exc_name)
        compare_events(parity, check, "iter_documents", path, got, want, diffs)
    assert not diffs, f"{data!r:.400}: {diffs[0]}"


# ------------------------------------------------------------------------------------------------- API

NAMES = (
    "ArchiveIterator",
    "WARCIterator",
    "ARCIterator",
    "StatusAndHeaders",
    "ArcWarcRecord",
    "DigestChecker",
    "ArchiveLoadFailed",
)


def test_api_names(warc_impl: Any, parity: Collector) -> None:
    mod = warc_impl.obj.mod
    missing = [n for n in NAMES if not hasattr(mod, n)]
    parity.record(SUITE, "api-names", not missing, got=missing, want=[])
    assert not missing
    v = getattr(mod, "WARCIO_VERSION", None)
    assert parity.record(SUITE, "api-names", v == "1.8.2", case="WARCIO_VERSION", got=v, want="1.8.2")
    assert parity.record(
        SUITE,
        "api-names",
        issubclass(mod.ArchiveLoadFailed, Exception),
        case="ArchiveLoadFailed is an Exception",
    )


def _params(obj: Any) -> Any:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError) as e:
        return f"no signature: {e}"
    return [(p.name, p.kind.name, repr(p.default)) for p in sig.parameters.values()]


@pytest.mark.parametrize(
    "name", ["ArchiveIterator", "WARCIterator", "ARCIterator", "iter_documents", "iter_batches"]
)
def test_api_signature(warc_impl: Any, parity: Collector, name: str) -> None:
    check = "api-signature"
    o = oracle()
    if warc_impl.is_reference or o is None:
        skip(parity, check, "candidate is warcio itself" if o else "warcio is not installed")
    mod = warc_impl.obj.mod
    empty = repr(inspect.Parameter.empty)
    full = _params(o.mod.ArchiveIterator)
    if name == "iter_documents":
        if not hasattr(mod, name):
            skip(parity, check, "no iter_documents")
        want: Any = [("source", "POSITIONAL_OR_KEYWORD", empty), ("compression", "KEYWORD_ONLY", "'infer'")]
        got = _params(mod.iter_documents)
        ok = got == want
    elif name == "iter_batches":
        if not hasattr(mod, name):
            skip(parity, check, "no iter_batches")
        got = _params(mod.iter_batches)
        want = "source first; batch_size=1000 and compression='infer' keyword-only"
        ok = (
            isinstance(got, list)
            and got[0][:2] == ("source", "POSITIONAL_OR_KEYWORD")
            and ("batch_size", "KEYWORD_ONLY", "1000") in got
            and ("compression", "KEYWORD_ONLY", "'infer'") in got
        )
    else:
        # warcio's WARCIterator / ARCIterator take (*args, **kwargs); ArchiveIterator's own signature is fine too.
        want = _params(getattr(o.mod, name))
        got = _params(getattr(mod, name))
        ok = got in (want, full)
    assert parity.record(SUITE, check, ok, case=name, got=got, want=want)


# StatusAndHeaders: a port of warcio's (Apache-2.0). Hypothesis over header lists and method calls.
HEADER_NAMES = st.sampled_from(
    ["Content-Type", "content-type", "X-A", "Set-Cookie", "WARC-Type", "K", "K", "é", ""]
)
HEADER_VALUES = st.one_of(
    st.sampled_from(["text/html", "", " v ", "a; b=c", 'attachment; filename="é.txt"', "naïve", "x" * 30]),
    st.text(max_size=12),
)


def _sah_outcome(
    cls: Any,
    statusline: str,
    headers: list[tuple[str, str]],
    protocol: str,
    http_req: bool,
    calls: list[tuple[str, Any]],
) -> list[Any]:
    out: list[Any] = []

    def do(label: str, fn: Callable[[], Any]) -> None:
        try:
            out.append([label, "ok", repr(fn())])
        except Exception as e:  # noqa: BLE001
            out.append([label, "raise", type(e).__name__, str(e)])

    try:
        h = cls(statusline, list(headers), protocol=protocol, is_http_request=http_req)
    except Exception as e:  # noqa: BLE001
        return [["init", "raise", type(e).__name__, str(e)]]
    for name, arg in calls:
        if name == "get_header":
            do(f"get_header({arg!r})", lambda arg=arg: h.get_header(arg))
        elif name == "get":
            do(f"get({arg!r})", lambda arg=arg: h.get(arg, "dflt"))
        elif name == "getitem":
            do(f"[{arg!r}]", lambda arg=arg: h[arg])
        elif name == "contains":
            do(f"in {arg!r}", lambda arg=arg: arg in h)
        elif name == "replace":
            do(f"replace_header({arg!r})", lambda arg=arg: h.replace_header(arg, "new"))
        elif name == "remove":
            do(f"remove_header({arg!r})", lambda arg=arg: h.remove_header(arg))
        elif name == "add":
            do(f"add_header({arg!r})", lambda arg=arg: h.add_header(arg, "added"))
        elif name == "setitem":
            do(f"[{arg!r}] = x", lambda arg=arg: h.__setitem__(arg, "x"))
        elif name == "delitem":
            do(f"del [{arg!r}]", lambda arg=arg: h.__delitem__(arg))
        elif name == "statuscode":
            do("get_statuscode", h.get_statuscode)
        elif name == "validate":
            do("validate_statusline", lambda: h.validate_statusline("500 Error"))
        elif name == "range":
            do("add_range", lambda: repr(h.add_range(0, 10, 100)))
        elif name == "percent":
            do("percent_encode", lambda: h.percent_encode_non_ascii_headers())
        do("headers", lambda: h.headers)
    for label, fn in (
        ("to_str", h.to_str),
        ("to_bytes", h.to_bytes),
        ("to_ascii_bytes", h.to_ascii_bytes),
        ("repr", lambda: repr(h)),
        ("str", lambda: str(h)),
        ("bool", lambda: bool(h)),
        ("protocol", lambda: h.protocol),
        ("statusline", lambda: h.statusline),
        ("total_len", lambda: h.total_len),
        ("eq-self-copy", lambda: h == cls(h.statusline, list(h.headers), protocol=h.protocol)),
        ("ne-empty", lambda: h != cls("", [])),
        ("eq-none", lambda: h == None),
    ):
        do(label, fn)
    return out


CALLS = st.lists(
    st.tuples(
        st.sampled_from(
            [
                "get_header",
                "get",
                "getitem",
                "contains",
                "replace",
                "remove",
                "add",
                "setitem",
                "delitem",
                "statuscode",
                "validate",
                "range",
                "percent",
            ]
        ),
        HEADER_NAMES,
    ),
    max_size=6,
)


@given(
    statusline=st.sampled_from(["200 OK", "", "404", "abc def", "GET / HTTP/1.1", "-1 x", "0 zero", "٣ x"]),
    headers=st.lists(st.tuples(HEADER_NAMES, HEADER_VALUES), max_size=5),
    protocol=st.sampled_from(["", "HTTP/1.0", "WARC/1.0"]),
    http_req=st.booleans(),
    calls=CALLS,
)
def test_api_status_and_headers(
    warc_impl: Any,
    parity: Collector,
    statusline: str,
    headers: list[tuple[str, str]],
    protocol: str,
    http_req: bool,
    calls: list[tuple[str, Any]],
) -> None:
    check = "api-status-and-headers"
    o = oracle()
    if o is None or warc_impl.is_reference:
        skip(parity, check, "candidate is warcio itself" if o else "warcio is not installed")
    got = _sah_outcome(warc_impl.obj.mod.StatusAndHeaders, statusline, headers, protocol, http_req, calls)
    want = _sah_outcome(o.mod.StatusAndHeaders, statusline, headers, protocol, http_req, calls)
    d = C.first_diff(got, want)
    assert parity.record(
        SUITE,
        check,
        d is None,
        input=(statusline, headers, protocol, http_req, calls),
        got=d and d.got,
        want=d and d.want,
    )


def _checker_outcome(mod: Any, kind: Any) -> list[Any]:
    out: list[Any] = []
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        dc = mod.DigestChecker(kind)
        out.append(["initial", repr(dc.passed), repr(list(dc.problems)), repr(dc.kind)])
        dc.passed = True
        out.append(["set", repr(dc.passed)])
        try:
            dc.problem("first problem")
            out.append(["problem", repr(dc.passed), repr(list(dc.problems))])
            dc.problem("second", passed=None)
            out.append(["problem2", repr(dc.passed), repr(list(dc.problems))])
        except Exception as e:  # noqa: BLE001
            out.append(["raise", type(e).__name__, str(e), repr(getattr(e, "msg", None))])
    out.append(["stderr", buf.getvalue()])
    e = mod.ArchiveLoadFailed("reason")
    out.append(["ArchiveLoadFailed", str(e), repr(e.msg), repr(e.args)])
    return out


@pytest.mark.parametrize("kind", [None, False, True, "log", "raise"])
def test_api_digest_checker(warc_impl: Any, parity: Collector, kind: Any) -> None:
    check = "api-digest-checker"
    o = oracle()
    if o is None or warc_impl.is_reference:
        skip(parity, check, "candidate is warcio itself" if o else "warcio is not installed")
    got = _checker_outcome(warc_impl.obj.mod, kind)
    want = _checker_outcome(o.mod, kind)
    assert parity.record(SUITE, check, got == want, case=repr(kind), got=got, want=want)


# ------------------------------------------------------------------------------------------------ mutant


def _mutants(o: Any) -> dict[str, Any]:
    import types

    from warcio.archiveiterator import ArchiveIterator

    class DropsLastByte(ArchiveIterator):
        def _next_record(self, next_line: Any) -> Any:
            rec = super()._next_record(next_line)
            real = rec.content_stream

            def content_stream() -> Any:
                s = real()

                class Wrap:
                    def read(self, n: Any = None) -> bytes:
                        b = s.read() if n is None else s.read(n)
                        return b[:-1] if len(b) > 1000 and n is None else b

                    def readline(self, n: Any = None) -> bytes:
                        return s.readline(n)

                return Wrap()

            rec.content_stream = content_stream
            return rec

    class Quiet(ArchiveIterator):
        def _consume_blanklines(self) -> Any:
            old = sys.stderr
            sys.stderr = io.StringIO()
            try:
                return super()._consume_blanklines()
            finally:
                sys.stderr = old

    class OffsetPlusOne(ArchiveIterator):
        def read_to_end(self, record: Any = None) -> Any:
            r = super().read_to_end(record)
            if self.member_info:
                self.member_info = (self.member_info[0], self.member_info[1] + 1)
            return r

    out = {}
    for name, cls in (
        ("drops-last-byte", DropsLastByte),
        ("no-stderr", Quiet),
        ("length-plus-one", OffsetPlusOne),
    ):
        ns = types.SimpleNamespace(**{k: getattr(o.mod, k) for k in NAMES}, __name__="mutant")
        ns.ArchiveIterator = cls
        out[name] = ns
    return out


def test_mutant(warc_impl: Any, parity: Collector) -> None:
    """Self-check only: each broken wrapper of warcio must show diffs in the record traces."""
    check = "mutant"
    o = oracle()
    if not warc_impl.is_reference or o is None:
        skip(parity, check, "only in the self-check" if o else "warcio is not installed")
    _need_inputs(parity, check)
    jobs = [
        j
        for g in ("warcio-tests", "synth-frame", "synth-length")
        for j in P.jobs(g)
        if j.config.name in ("content/file", "offsets/bytesio")
    ]
    cc = [j for j in P.jobs("cc") if j.name.endswith("2019-35.warc.gz") and j.config.name == "content/file"]
    caught = {}
    for name, mut in _mutants(o).items():
        n = 0
        for job in [*cc, *jobs]:
            got = C.split(C.iter_events(mut, job.path, job.config, 30, o.exc_name))[0]
            want = C.split(C.iter_events(o.mod, job.path, job.config, 30, o.exc_name))[0]
            n += got != want
        caught[name] = n
        parity.record(SUITE, check, n > 0, case=name, got=n)
    assert all(caught.values()), f"mutants not caught: {caught}"
