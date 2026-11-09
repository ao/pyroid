"""trafilatura parity: the ``--impl`` candidate vs trafilatura 2.3.1 (+ lxml 6.1.3, justext 3.0.2).

    pytest tests/test_trafilatura.py                                   # self-check (candidate = trafilatura)
    pytest tests/test_trafilatura.py --impl pyroid.trafilatura         # pyroid vs trafilatura
    pytest tests/test_trafilatura.py --impl pyroid.trafilatura --parity-limit 0   # every pinned page

Outcomes are compared exactly (``pyroid_parity.trafilatura_compare``): ``extract`` text byte for byte (``None``
distinct from ``""``), ``bare_extraction`` field by field with ``body``/``commentsbody`` as lxml's XML
serialization, exceptions by type and message, warnings by category (``pyroid.FallbackWarning`` left out).

Checks (suite ``trafilatura`` in parity-report.json):
    corpus-<option>     ``extract(html, **option)`` on the pinned Common Crawl pages of 3 crawls
                        (``specs.TRAFILATURA_CORPORA``) == the pinned digests. Options: ``specs.TRAFILATURA_OPTION_SETS``;
                        ``datatrove`` and ``dedup`` are stateful: each file runs in order after ``reset_caches()``
    batch-<option>      the same through ``extract_batch`` (64 pages per call, the dedup cache carried across calls)
    bare-<option>       ``bare_extraction(html, **option)`` on the same pages (``specs.TRAFILATURA_BARE_SETS``)
    inputs              the pinned pages and references are the ones in bench/manifest.json
    upstream            calls recorded from Trafilatura's own tests whose options pyroid ports, replayed test by
                        test (``pyroid_parity.trafilatura_record``)
    upstream-delegated  the other recorded calls (formats, metadata, trees, configs, ...): pyroid delegates them
    hypothesis-<option> grammar-generated pages (``trafilatura_strategies``): candidate == live trafilatura
    delegated-result, delegated-warning   ``specs.TRAFILATURA_DELEGATED`` in a fresh process: trafilatura's
                        result, and exactly one ``pyroid.FallbackWarning`` over the case's calls
    edge-result, edge-warning   ``specs.TRAFILATURA_EDGE_OPTIONS``: trafilatura's result, at most one warning
    native-result, native-no-fallback   the corpus option sets on the same pages: no FallbackWarning at all
    delegated-no-oracle where trafilatura is not installed, delegated options raise ``pyroid.PyroidError``
    api-signature, api-batch, api-reset   trafilatura's signatures plus ``compat="2.3.1"`` (keyword-only, last);
                        ``extract_batch(htmls, *, threads=None, compat="2.3.1", **kwargs)``; ``reset_caches()``
    api-shim            ``pyroid.integrations.patch("trafilatura")``: trafilatura's exact signatures, no ``compat``

Without ``--impl`` the candidate is trafilatura itself: the pinned checks then verify that this interpreter
reproduces the references, on the first 100 pages of each file unless ``--parity-limit`` says otherwise.
Where trafilatura is not installed (the free-threaded CI legs) the live checks skip and say so; the pinned ones
still run against the candidate.
"""

from __future__ import annotations

import concurrent.futures
import importlib.metadata as md
import inspect
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from hypothesis import example, given

from pyroid_parity import data as D
from pyroid_parity import specs
from pyroid_parity import trafilatura_compare as T
from pyroid_parity import trafilatura_record as R
from pyroid_parity import trafilatura_strategies as S
from pyroid_parity.impl import FALLBACK_WARNING
from pyroid_parity.plugin import _impl, need, reference
from pyroid_parity.report import Collector

SUITE = "trafilatura"
PARITY = Path(__file__).resolve().parents[1]
REF = "parity/trafilatura-ref.parquet"
OPTIONS = list(specs.TRAFILATURA_OPTION_SETS)
BARE = list(specs.TRAFILATURA_BARE_SETS)
CORPORA = list(specs.TRAFILATURA_CORPORA)
SELF_CHECK_PAGES = 100
BATCH = 64
MAX_DETAIL = 5  # diffs per check explained against the live oracle


def _crawl(rel: str) -> str:
    return Path(rel).stem


@cache
def oracle() -> Any:
    """trafilatura 2.3.1 if it is installed here (imported only then: lxml re-enables the GIL on 3.14t)."""
    try:
        import trafilatura
    except ImportError:
        return None
    for dist, want in specs.TRAFILATURA_ORACLE.items():
        got = md.version(dist)
        assert got == want, f"the trafilatura oracle needs {dist}=={want}, found {got}"
    return trafilatura


def oracle_reset() -> None:
    if (o := oracle()) is not None:
        o.meta.reset_caches()


def need_oracle(parity: Collector, check: str) -> Any:
    o = oracle()
    if o is None:
        why = "trafilatura is not installed here (parity group `trafilatura`; not on free-threaded legs)"
        parity.skip(SUITE, check, why)
        pytest.skip(why)
    return o


@pytest.fixture(scope="session")
def trafilatura_impl(request: pytest.FixtureRequest) -> Any:
    return _impl(request, SUITE)


@pytest.fixture(scope="module", autouse=True)
def _oracle_info() -> None:
    o = oracle()
    reference(SUITE, "trafilatura", o.__version__ if o else f"{specs.TRAFILATURA_VERSION} (pinned)", None)


def page_limit(request: pytest.FixtureRequest, impl: Any) -> int:
    raw = request.config.getoption("--parity-limit")
    if raw is None:
        return SELF_CHECK_PAGES if impl.is_reference else 0
    return int(raw)


def pages(rel: str, limit: int) -> Iterator[tuple[int, str, str, str]]:
    """(index, id, url, html) of a pinned corpus file in order, streamed in batches."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(need(rel))
    n = 0
    for batch in pf.iter_batches(batch_size=256, columns=["id", "url", "html"]):
        cols = [batch.column(c).to_pylist() for c in ("id", "url", "html")]
        for doc_id, url, html in zip(*cols, strict=True):
            yield n, doc_id, url, html
            n += 1
            if limit and n >= limit:
                return


@cache
def refs() -> dict[tuple[str, str, str], list[tuple[str, str, str, int | None, tuple[str, ...]]]]:
    """(corpus, fn, option) -> [(id, status, digest, length, warnings)] in page order."""
    names = ["corpus", "fn", "option", "id", "status", "digest", "length", "warnings"]
    cols = D.read_parquet_columns(need(REF), names)
    out: dict[tuple[str, str, str], list[Any]] = {}
    for c, fn, o, i, st, dg, ln, w in zip(*cols.values(), strict=True):
        out.setdefault((c, fn, o), []).append((i, st, dg, ln, tuple(w or ())))
    return out


def _explain(fn: str, kw: dict[str, Any], html: str, got: T.Outcome, stateful: bool) -> str:
    o = oracle()
    if o is None or stateful:
        return "(the live oracle is not available for a stateful set: compare with the reference run)"
    want = T.capture_call(getattr(o, fn), html, **kw)
    if got.kind == want.kind == "ok" and isinstance(got.value, str) and isinstance(want.value, str):
        return T.first_text_difference(got.value, want.value)
    return T.describe_difference(got, want)


def _check_pages(
    parity: Collector,
    check: str,
    rel: str,
    fn: str,
    kw: dict[str, Any],
    stateful: bool,
    outcomes: Iterator[tuple[int, str, str, str, T.Outcome]],
    ref: list[Any],
) -> list[str]:
    diffs: list[str] = []
    for n, doc_id, url, html, got in outcomes:
        want_id, status, dig, length, _ = ref[n]
        assert want_id == doc_id, f"{REF} is not aligned with {rel} at page {n}"
        same = T.digest(got) == dig
        want = f"{status} length={length} sha256={dig[:16]}"
        parity.record(
            SUITE, check, same, case=f"{_crawl(rel)}#{n} {doc_id}", input=url, got=got.short(), want=want
        )
        if not same:
            detail = _explain(fn, kw, html, got, stateful) if len(diffs) < MAX_DETAIL else ""
            diffs.append(f"{_crawl(rel)}#{n} {url}: got {got.short(120)}, want {want} {detail}")
    return diffs


def _start(impl: Any, parity: Collector, check: str, kw: dict[str, Any]) -> bool:
    stateful = specs.trafilatura_stateful(kw)
    if stateful:
        if not impl.obj.has_reset:
            why = f"{impl.module} has no reset_caches(), so the dedup cache cannot be emptied first"
            parity.skip(SUITE, check, why)
            pytest.fail(why)
        impl.obj.reset_caches()
        if not impl.is_reference:
            oracle_reset()  # in case the candidate hands some calls to trafilatura
    return stateful


# --------------------------------------------------------------------------------------------- pinned corpus


def test_inputs_are_pinned(parity: Collector) -> None:
    manifest = D.load_manifest()["files"]
    for rel in CORPORA:
        need(rel)
        parity.uses(SUITE, rel)
        want = manifest[rel]["content_sha256"]
        cols = D.read_parquet_columns(D.path(rel), ["id", "html"])
        got = D.content_sha256(cols["id"], cols["html"])
        del cols
        parity.record(SUITE, "inputs", got == want, case=rel, got=got, want=want)
        assert got == want, f"{rel}: content sha256 differs from the manifest"
    need(REF)
    parity.uses(SUITE, REF)
    pinned = manifest[REF].get("corpora", {})
    same = pinned == {rel: manifest[rel]["content_sha256"] for rel in CORPORA}
    parity.record(SUITE, "inputs", same, case=REF, got=pinned)
    assert same, f"{REF} was made from other inputs than the pinned corpus"
    assert manifest[REF]["option_sets"] == specs.TRAFILATURA_OPTION_SETS
    assert manifest[REF]["bare_sets"] == specs.TRAFILATURA_BARE_SETS


@pytest.mark.parametrize("option", OPTIONS)
@pytest.mark.parametrize("rel", CORPORA, ids=_crawl)
def test_corpus(
    trafilatura_impl: Any, parity: Collector, request: pytest.FixtureRequest, rel: str, option: str
) -> None:
    impl, kw, check = trafilatura_impl, specs.TRAFILATURA_OPTION_SETS[option], f"corpus-{option}"
    ref = refs()[rel, "extract", option]
    stateful = _start(impl, parity, check, kw)
    outcomes = (
        (n, i, u, h, T.capture_call(impl.obj.extract, h, **kw))
        for n, i, u, h in pages(rel, page_limit(request, impl))
    )
    diffs = _check_pages(parity, check, rel, "extract", kw, stateful, outcomes, ref)
    assert not diffs, f"{len(diffs)} pages differ, first:\n" + "\n".join(diffs[:10])


def _batched(
    impl: Any, rel: str, limit: int, kw: dict[str, Any]
) -> Iterator[tuple[int, str, str, str, T.Outcome]]:
    chunk: list[tuple[int, str, str, str]] = []

    def flush() -> Iterator[tuple[int, str, str, str, T.Outcome]]:
        res = T.capture(lambda: impl.obj.extract_batch([h for *_, h in chunk], threads=None, **kw))
        if res.kind == "ok" and isinstance(res.value, list) and len(res.value) == len(chunk):
            for (n, i, u, h), v in zip(chunk, res.value, strict=True):
                yield n, i, u, h, T.Outcome("ok", T.normalize(v))
        else:
            for n, i, u, h in chunk:
                yield n, i, u, h, T.Outcome("raise", ("extract_batch", res.short(200)))

    for item in pages(rel, limit):
        chunk.append(item)
        if len(chunk) == BATCH:
            yield from flush()
            chunk = []
    if chunk:
        yield from flush()


@pytest.mark.parametrize("option", OPTIONS)
@pytest.mark.parametrize("rel", CORPORA, ids=_crawl)
def test_batch(
    trafilatura_impl: Any, parity: Collector, request: pytest.FixtureRequest, rel: str, option: str
) -> None:
    impl, kw, check = trafilatura_impl, specs.TRAFILATURA_OPTION_SETS[option], f"batch-{option}"
    ref = refs()[rel, "extract", option]
    stateful = _start(impl, parity, check, kw)
    # extract_batch reports no warnings per page: compare against the reference's own warning categories.
    raw = _batched(impl, rel, page_limit(request, impl), kw)
    outcomes = (
        (n, i, u, h, T.Outcome(o.kind, o.value, ref[n][4] if o.kind == "ok" else ())) for n, i, u, h, o in raw
    )
    diffs = _check_pages(parity, check, rel, "extract", kw, stateful, outcomes, ref)
    assert not diffs, f"{len(diffs)} pages differ, first:\n" + "\n".join(diffs[:10])


@pytest.mark.parametrize("option", BARE)
@pytest.mark.parametrize("rel", CORPORA, ids=_crawl)
def test_bare(
    trafilatura_impl: Any, parity: Collector, request: pytest.FixtureRequest, rel: str, option: str
) -> None:
    impl, kw, check = trafilatura_impl, specs.TRAFILATURA_BARE_SETS[option], f"bare-{option}"
    ref = refs()[rel, "bare_extraction", option]
    stateful = _start(impl, parity, check, kw)
    outcomes = (
        (n, i, u, h, T.capture_call(impl.obj.bare_extraction, h, **kw))
        for n, i, u, h in pages(rel, page_limit(request, impl))
    )
    diffs = _check_pages(parity, check, rel, "bare_extraction", kw, stateful, outcomes, ref)
    assert not diffs, f"{len(diffs)} pages differ, first:\n" + "\n".join(diffs[:10])


# ------------------------------------------------------------------------------------ Trafilatura's own tests

_NATIVE_VALUES: dict[str, tuple[Any, ...] | None] = {
    # None: any value. Otherwise the encoded values a native call may pass.
    "url": None,
    "record_id": None,
    "fast": None,
    "no_fallback": None,
    "favor_precision": None,
    "favor_recall": None,
    "include_comments": None,
    "include_tables": None,
    "tei_validation": None,
    "deduplicate": (True, False),
    "include_formatting": (None, False),
    "include_links": (False,),
    "include_images": (False,),
    "with_metadata": (False,),
    "only_with_metadata": (False,),
    "target_language": (None,),
    "date_extraction_params": (None,),
    "max_tree_size": (None,),
    "url_blacklist": (None,),
    "author_blacklist": (None,),
    "settingsfile": (None,),
    "prune_xpath": (None,),
    "options": (None,),
    "as_dict": (False,),
    "config": ({"$config": "default"},),
}


def upstream_native(call: dict[str, Any]) -> bool:
    """Whether a recorded call uses only what pyroid ports (str or bytes input, the corpus options, txt)."""
    names = [n for n, _ in specs.TRAFILATURA_SIGNATURES[call["fn"]]]
    bound = dict(zip(names, call["args"], strict=False)) | call["kwargs"]
    first = bound.pop("filecontent", None)
    if isinstance(first, dict) and not ({"$blob", "$str", "$bytes", "$cat", "$rep"} & set(first)):
        return False
    fmt = bound.pop("output_format", "txt" if call["fn"] == "extract" else "python")
    if fmt != ("txt" if call["fn"] == "extract" else "python"):
        return False
    for k, v in bound.items():
        allowed = _NATIVE_VALUES.get(k, ())
        if allowed is not None and v not in allowed:
            return False
    return True


@pytest.mark.timeout(3600)  # one test replays all 4,553 calls (about 5 minutes for trafilatura itself)
def test_upstream(trafilatura_impl: Any, parity: Collector, tmp_path: Path) -> None:
    impl = trafilatura_impl
    need(R.CALLS)
    need(R.BLOBS)
    need(R.UPSTREAM_TESTS)
    parity.uses(SUITE, R.CALLS, R.BLOBS, R.UPSTREAM_TESTS)
    R.materialize_tests(tmp_path)  # some calls pass settingsfile= paths into tests/resources/
    calls, blobs = R.load(tmp_path)
    has_oracle = oracle() is not None
    if not impl.obj.has_reset:
        pytest.fail(f"{impl.module} has no reset_caches(): the recorded sequences cannot be replayed")
    skipped: dict[str, int] = {}
    diffs: list[str] = []
    group = None
    for c in calls:
        if c["test"] != group:
            group = c["test"]
            impl.obj.reset_caches()
            if not impl.is_reference:
                oracle_reset()
        name = "upstream" if upstream_native(c) else "upstream-delegated"
        encoded = json.dumps([c["args"], c["kwargs"]])
        needs_objects = any(f'"{tag}"' in encoded for tag in ("$tree", "$lru", "$extractor"))
        if not has_oracle and (name == "upstream-delegated" or needs_objects):
            skipped["no-oracle"] = skipped.get("no-oracle", 0) + 1
            continue
        args = [T.decode_arg(a, blobs) for a in c["args"]]
        kwargs = {k: T.decode_arg(v, blobs) for k, v in c["kwargs"].items()}
        got = T.capture_call(getattr(impl.obj, c["fn"]), *args, **kwargs)
        skip = [t for t in c["tags"] if t in R.SKIP_TAGS]
        if skip:  # run anyway, so the rest of the test's sequence sees the same state
            skipped[skip[0]] = skipped.get(skip[0], 0) + 1
            continue
        same = T.outcome_json(got) == c["outcome"]
        want = T.Outcome.from_json(json.loads(c["outcome"]))
        parity.record(
            SUITE,
            name,
            same,
            case=f"{c['index']} {c['test']}",
            input=(c["fn"], c["kwargs"]),
            got=got.short(),
            want=want.short(),
        )
        if not same:
            diffs.append(
                f"{c['index']} {c['test']} {c['fn']}{c['kwargs']}: {T.describe_difference(got, want)}"
            )
    for tag, n in skipped.items():
        why = R.SKIP_TAGS.get(
            tag, "trafilatura is not installed here, and pyroid delegates these calls to it"
        )
        parity.skip(SUITE, f"upstream-{tag}", f"{n} recorded calls not compared: {why}")
    assert not diffs, f"{len(diffs)}/{len(calls)} recorded calls differ, first:\n" + "\n".join(diffs[:10])


# ---------------------------------------------------------------------------------- Hypothesis differential

EXAMPLES = (
    "",
    " ",
    "<html></html>",
    "<p>x</p>",
    "<html><body><p>" + "word " * 60 + "</p></body></html>",
    "<html><body><p>lone \ud800 surrogate " + "text " * 60 + "</p></body></html>",
    "<html><body><p>a NUL \x00 and \x01 controls " + "text " * 60 + "</p></body></html>",
    "<html><body><article><p>" + "x" * 5000 + "</p></article></body></html>",
    "<html><body>"
    + "<p>The same paragraph, repeated on purpose, long enough to be checked.</p>" * 6
    + "</body></html>",
    "\N{ZERO WIDTH NO-BREAK SPACE}<html><body><div class='comments'><p>"
    + "c " * 200
    + "</p></div></body></html>",
)


def _calls(option: str) -> tuple[str, dict[str, Any]]:
    return ("bare_extraction", {}) if option == "bare" else ("extract", specs.TRAFILATURA_OPTION_SETS[option])


@pytest.mark.parametrize("option", [*OPTIONS, "bare"])
@given(html=S.documents())
@example(html=EXAMPLES[0])
@example(html=EXAMPLES[1])
@example(html=EXAMPLES[2])
@example(html=EXAMPLES[3])
@example(html=EXAMPLES[4])
@example(html=EXAMPLES[5])
@example(html=EXAMPLES[6])
@example(html=EXAMPLES[7])
@example(html=EXAMPLES[8])
@example(html=EXAMPLES[9])
def test_hypothesis(trafilatura_impl: Any, parity: Collector, option: str, html: str) -> None:
    o = need_oracle(parity, f"hypothesis-{option}")
    impl = trafilatura_impl
    fn, kw = _calls(option)
    stateful = specs.trafilatura_stateful(kw)
    if stateful:
        impl.obj.reset_caches()
    got = T.capture_call(getattr(impl.obj, fn), html, **kw)
    if stateful:
        o.meta.reset_caches()
    want = T.capture_call(getattr(o, fn), html, **kw)
    same = parity.record(
        SUITE,
        f"hypothesis-{option}",
        T.same_outcome(got, want),
        input=html,
        got=got.short(),
        want=want.short(),
    )
    assert same, f"input: {html[:2000]!r}\n  got: {got.short(2000)}\n want: {want.short(2000)}"


# ----------------------------------------------------------------------- delegation, edge options, native sets

CASES = (
    [("delegated", c) for c in specs.TRAFILATURA_DELEGATED]
    + [("edge", c) for c in specs.TRAFILATURA_EDGE_OPTIONS]
    + [("native", c) for c in OPTIONS]
    + [("native", f"bare:{c}") for c in BARE]
)


def _run_case(module: str, group: str, case: str) -> dict[str, Any]:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    cmd = [sys.executable, "-m", "pyroid_parity.trafilatura_delegate", module, group, case]
    proc = subprocess.run(cmd, cwd=PARITY, env=env, capture_output=True, text=True, timeout=600, check=False)
    if proc.returncode != 0:
        return {"error": proc.stderr[-3000:]}
    return json.loads(proc.stdout)


@pytest.fixture(scope="session")
def case_results(trafilatura_impl: Any) -> dict[tuple[str, str], dict[str, Any]]:
    """Every case, each in its own fresh process (several at a time)."""
    if oracle() is None:
        return {}
    with concurrent.futures.ThreadPoolExecutor(min(8, os.cpu_count() or 2)) as ex:
        futures = {k: ex.submit(_run_case, trafilatura_impl.module, *k) for k in CASES}
        return {k: f.result() for k, f in futures.items()}


@pytest.mark.parametrize(("group", "case"), CASES, ids=[f"{g}-{c}" for g, c in CASES])
def test_case(
    trafilatura_impl: Any, parity: Collector, case_results: dict[tuple[str, str], Any], group: str, case: str
) -> None:
    wcheck = {"delegated": "delegated-warning", "edge": "edge-warning", "native": "native-no-fallback"}[group]
    if oracle() is None:
        parity.skip(SUITE, wcheck, "trafilatura is not installed here: nothing to delegate to")
    need_oracle(parity, f"{group}-result")
    r = case_results[group, case]
    assert "error" not in r, f"{group} {case}: the runner failed:\n{r.get('error')}"
    diffs = []
    for c in r["calls"]:
        same = c["got"] == c["want"]
        got, want = T.Outcome.from_json(json.loads(c["got"])), T.Outcome.from_json(json.loads(c["want"]))
        parity.record(
            SUITE, f"{group}-result", same, case=f"{case} doc {c['doc']}", got=got.short(), want=want.short()
        )
        if not same:
            diffs.append(f"doc {c['doc']}: {T.describe_difference(got, want)}")
    fallbacks = [n for c in r["calls"] for n in c["warnings"] if n == FALLBACK_WARNING]
    near = [n for c in r["calls"] for n in c["warnings"] if n != FALLBACK_WARNING and "Fallback" in n]
    if trafilatura_impl.is_reference and group != "native":
        parity.skip(SUITE, wcheck, "the candidate is trafilatura itself, which does not delegate")
    else:
        if group == "delegated":
            ok = len(fallbacks) == 1 and FALLBACK_WARNING in r["calls"][0]["warnings"] and not near
            want_w = "exactly one pyroid.FallbackWarning, on the first call"
        elif group == "edge":
            ok, want_w = len(fallbacks) <= 1 and not near, "at most one pyroid.FallbackWarning"
        else:
            ok, want_w = not fallbacks and not near, "no FallbackWarning"
        got_w = [c["warnings"] for c in r["calls"]]
        parity.record(SUITE, wcheck, ok, case=case, got=got_w, want=want_w)
        if not ok:
            diffs.append(f"warnings {got_w}, want {want_w}")
    assert not diffs, f"{group} {case}: " + "; ".join(diffs)


@pytest.mark.parametrize("case", list(specs.TRAFILATURA_DELEGATED))
def test_delegated_without_trafilatura(trafilatura_impl: Any, parity: Collector, case: str) -> None:
    """Without trafilatura installed, pyroid cannot delegate: it raises PyroidError (docs/dev/MODULES.md)."""
    if oracle() is not None:
        parity.skip(
            SUITE, "delegated-no-oracle", "trafilatura is installed here (the delegated-* checks run)"
        )
        pytest.skip("trafilatura is installed")
    if trafilatura_impl.is_reference:
        pytest.skip("the candidate is trafilatura itself")
    kw = dict(specs.TRAFILATURA_DELEGATED[case])
    fn = kw.pop("$fn", "extract")
    from pyroid_parity.trafilatura_delegate import DOCS

    got = T.capture_call(getattr(trafilatura_impl.obj, fn), DOCS[0], **kw)
    ok = got.kind == "raise" and got.value[0] == "PyroidError"
    parity.record(SUITE, "delegated-no-oracle", ok, case=case, got=got.short(), want="raise PyroidError")
    assert ok, f"{case}: {got.short()}"


# -------------------------------------------------------------------------------------------- API mirror


#: The one parameter pyroid.trafilatura adds to trafilatura's functions: the version mode, keyword-only, last
#: (on extract_batch: last before ``**kwargs``).
COMPAT = ("compat", "KEYWORD_ONLY", specs.TRAFILATURA_VERSION)


def _shape(fn: Any) -> list[tuple[str, str, Any]]:
    out = []
    for p in inspect.signature(fn).parameters.values():
        d = p.default
        if p.name == "config" and (d is None or type(d).__name__ == "ConfigParser"):
            d = "DEFAULT_CONFIG"  # None, or trafilatura's own object (checked against the oracle below)
        elif d is inspect.Parameter.empty:
            d = "<required>"
        out.append((p.name, p.kind.name, d))
    return out


def _want_shape(name: str, compat: bool = False) -> list[tuple[str, str, Any]]:
    shape = [(n, "POSITIONAL_OR_KEYWORD", d) for n, d in specs.TRAFILATURA_SIGNATURES[name]]
    return [*shape, COMPAT] if compat else shape


def test_oracle_signatures_are_pinned() -> None:
    o = oracle()
    if o is None:
        pytest.skip("trafilatura is not installed here")
    from trafilatura.settings import DEFAULT_CONFIG

    for name in specs.TRAFILATURA_SIGNATURES:
        fn = getattr(o, name)
        assert inspect.signature(fn).parameters["config"].default is DEFAULT_CONFIG
        assert _shape(fn) == _want_shape(name)


@pytest.mark.parametrize("name", [*specs.TRAFILATURA_SIGNATURES, "extract_batch", "reset_caches"])
def test_api(trafilatura_impl: Any, parity: Collector, name: str) -> None:
    """trafilatura's signatures plus ``compat="2.3.1"`` (keyword-only, last); ``reset_caches()`` takes nothing."""
    impl = trafilatura_impl
    check = {"extract_batch": "api-batch", "reset_caches": "api-reset"}.get(name, "api-signature")
    if impl.is_reference:
        parity.skip(SUITE, check, "the candidate is trafilatura itself")
        pytest.skip("the candidate is trafilatura itself")
    fn = getattr(impl.obj.mod, name, None)
    got: Any = None
    try:
        sig = None if fn is None else inspect.signature(fn)
    except (TypeError, ValueError) as e:  # a builtin without a text signature
        sig, got = None, f"no signature: {e}"
    if name == "reset_caches":
        want: Any = "()"
        got = str(sig) if sig is not None else got
    elif name == "extract_batch":
        want = [("htmls", "POSITIONAL_OR_KEYWORD", "<required>"), ("threads", "KEYWORD_ONLY", None), COMPAT]
        want.append(("kwargs", "VAR_KEYWORD", "<required>"))
        got = _shape(fn) if sig is not None else got
    else:
        want = _want_shape(name, compat=True)
        got = _shape(fn) if sig is not None else got
    ok = got == want
    parity.record(SUITE, check, ok, case=name, got=got, want=want)
    assert ok, f"{name}: {got} != {want}"


def test_api_shim(trafilatura_impl: Any, parity: Collector) -> None:
    """``<top>.integrations.patch("trafilatura")`` gives a module whose functions have trafilatura's exact
    signatures: no ``compat``, and ``config`` defaulting to trafilatura's own ``DEFAULT_CONFIG`` object (``None``
    only where trafilatura is not installed). Checked in a fresh process, which the patch then does not leak from.
    """
    impl = trafilatura_impl
    if impl.is_reference:
        parity.skip(SUITE, "api-shim", "the candidate is trafilatura itself")
        pytest.skip("the candidate is trafilatura itself")
    top = impl.module.split(".")[0]
    cmd = [sys.executable, "-m", "pyroid_parity.trafilatura_delegate", "--shim", top]
    proc = subprocess.run(cmd, cwd=PARITY, capture_output=True, text=True, timeout=600, check=False)
    assert proc.returncode == 0, proc.stderr[-3000:]
    r = json.loads(proc.stdout)
    if "skip" in r:
        parity.skip(SUITE, "api-shim", r["skip"])
        pytest.skip(r["skip"])
    diffs = []
    for name in specs.TRAFILATURA_SIGNATURES:
        got = [tuple(x) for x in r["shapes"][name]] if r["shapes"].get(name) is not None else None
        want = _want_shape(name)
        ok = parity.record(SUITE, "api-shim", got == want, case=name, got=got, want=want)
        if not ok:
            diffs.append(f"{name}: {got} != {want}")
    ok = parity.record(SUITE, "api-shim", r["pyroid"] is True, case="__pyroid__", got=r["pyroid"], want=True)
    if not ok:
        diffs.append("the patched module has no __pyroid__ = True")
    assert not diffs, "; ".join(diffs)
