# mypy: disallow-subclassing-any=False
"""pyroid.integrations.datatrove.Trafilatura against datatrove 0.10.1's own Trafilatura step.

The stock step runs trafilatura in a child process (forked from here, so trafilatura is imported
here first, with its import-time warnings silenced). Both steps get the same pages and arguments;
the documents out, ``doc.text``, and every statistic except time must be equal. It runs with
trafilatura 2.3.1 or 1.11.0 installed (stock datatrove installs 1.11.0), and the step follows the
installed release. The other ``compat`` branches: trafilatura reported as not installed (pyroid
runs 2.3.1) and as a release pyroid does not reproduce (datatrove's own step runs, with one
FallbackWarning).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import inspect
import json
import os
import pickle
import struct
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("datatrove")
pytest.importorskip("regex")  # datatrove.pipeline.filters imports regex and tokenizers,
pytest.importorskip("tokenizers")  # which datatrove does not declare
pytest.importorskip("numpy")
FREE_THREADED = not getattr(sys, "_is_gil_enabled", lambda: True)()
if FREE_THREADED:
    pytest.skip(
        "datatrove's step needs trafilatura (lxml re-enables the GIL)", allow_module_level=True
    )
INSTALLED = (
    importlib.metadata.version("trafilatura")
    if importlib.util.find_spec("trafilatura") is not None
    else None
)
if INSTALLED not in ("2.3.1", "1.11.0"):
    pytest.skip(
        f"needs trafilatura 2.3.1 or 1.11.0 for datatrove's own step, found {INSTALLED}",
        allow_module_level=True,
    )

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # urllib3's DependencyWarning (SOCKS) when trafilatura loads
    import trafilatura  # type: ignore[import-not-found,import-untyped,unused-ignore]
    from datatrove.data import Document  # type: ignore[import-not-found,import-untyped,unused-ignore]
    from datatrove.pipeline.extractors import Trafilatura as StockTrafilatura  # type: ignore[import-not-found,import-untyped,unused-ignore]

import pyroid  # noqa: E402
import pyroid.trafilatura as pt  # noqa: E402
from pyroid.integrations import datatrove as pdt  # noqa: E402

pytestmark = [
    pytest.mark.thread_unsafe(reason="child processes and process-wide state"),
    # datatrove's step forks its child after pyroid's thread pool has started (harmless: pyroid
    # rebuilds its pool in a forked child)
    pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning"),
]

RAW = Path(os.environ.get("PYROID_TRAFILATURA_PAGES", "/target/cc/dev/raw.bin"))


def load_pages() -> Any:
    name = "_pyroid_test_trafilatura_pages"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("trafilatura_pages.py")
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


tp = load_pages()


def pages(n: int = 150) -> list[str]:
    out = [p for p in tp.PAGES.values() if p.strip()] * 2  # repeats exercise the dedup cache
    out.append(f"<html><body><p>{tp.PARAGRAPH} &amp;#{'1' * 4400};</p></body></html>")  # raises
    if RAW.is_file():
        data = RAW.read_bytes()
        off = 4
        for _ in range(min(n, struct.unpack_from("<I", data, 0)[0])):
            (ln,) = struct.unpack_from("<I", data, off)
            out.append(data[off + 4 : off + 4 + ln].decode("utf-8", "surrogatepass"))
            off += 4 + ln
    return out


def docs(texts: list[str]) -> list[Any]:
    return [Document(text=t, id=str(i), metadata={}) for i, t in enumerate(texts)]


def run(step: Any, texts: list[str]) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    out = [(d.id, d.text) for d in step.run(docs(texts))]
    return out, step.stats.to_dict()["stats"]


STOCK: dict[str, Any] = {}


def stock(kwargs: dict[str, Any], texts: list[str]) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    key = json.dumps([kwargs, texts], sort_keys=True)
    if key not in STOCK:
        # datatrove's child process is forked from this one: empty trafilatura's module-level
        # dedup cache first, as in a datatrove task (whose parent never extracts).
        importlib.import_module("trafilatura.meta").reset_caches()
        # timeout=60: datatrove's 1 s default would skip slow pages under load (pyroid has none)
        STOCK[key] = run(StockTrafilatura(timeout=60, **kwargs), texts)
    return STOCK[key]  # type: ignore[no-any-return]


def test_subclass_and_signature() -> None:
    assert issubclass(pdt.Trafilatura, StockTrafilatura)
    assert pdt.Trafilatura.__name__ == "Trafilatura"
    base = list(inspect.signature(StockTrafilatura).parameters.values())
    params = list(inspect.signature(pdt.Trafilatura).parameters.values())
    assert [p for p in params if p.kind is not p.VAR_KEYWORD][: len(base) - 1] == base[:-1]
    extra = {p.name: p.default for p in params if p.kind is p.KEYWORD_ONLY}
    assert extra == {"compat": None, "batch_size": pdt.TRAFILATURA_BATCH_SIZE, "threads": 1}
    assert params[-1].kind is inspect.Parameter.VAR_KEYWORD
    step = pdt.Trafilatura()
    assert (step.favour_precision, step.include_images, step.timeout, step.deduplicate) == (
        True,
        False,
        1,
        True,
    )
    with pytest.raises(NotImplementedError):
        pdt.Trafilatura(include_images=True)
    with pytest.raises(ValueError, match="compat must be one of"):
        pdt.Trafilatura(compat="1.0")
    with pytest.raises(ValueError, match="batch_size"):
        pdt.Trafilatura(batch_size=0)
    # our own freshly built object: nothing untrusted is unpickled
    assert type(pickle.loads(pickle.dumps(pdt.Trafilatura(threads=None)))) is pdt.Trafilatura


CASES: dict[str, dict[str, Any]] = {
    "fineweb": {},  # FineWeb's Trafilatura(favour_precision=True): datatrove's defaults
    "recall_side": {"favour_precision": False},
    "no_dedup": {"deduplicate": False},
    "no_tables": {"include_tables": False},
    "fast": {"fast": True},  # 2.3.1's switch; 1.11.0's extract ignores the keyword
    "no_fallback_111": {"no_fallback": True},  # 1.11.0's switch (2.3.1 warns: not compared)
}


@pytest.mark.timeout(900)
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("threads", [1, None])
def test_matches_datatrove(case: str, threads: int | None) -> None:
    if case == "no_fallback_111" and INSTALLED != "1.11.0":
        pytest.skip("no_fallback warns (DeprecationWarning) in 2.3.1")
    texts = pages()
    kwargs = CASES[case]
    want = stock(kwargs, texts)
    got = run(pdt.Trafilatura(timeout=60, threads=threads, batch_size=17, **kwargs), texts)
    assert got[0] == want[0]
    assert got[1] == want[1]
    assert {"total", "extracted", "forwarded", "dropped", "doc_len"} <= set(got[1])
    assert got[1]["broken_process"] == 1  # the 4400-digit reference kills the child


def test_warm_up_failure_times_out() -> None:
    texts = [tp.PAGES["article"], tp.PAGES["table"]]
    # a keyword trafilatura.extract rejects (2.3.1: TypeError; 1.11.0: NameError for this one):
    # datatrove's child dies in its warm-up call extract(""), so every page waits for the
    # timeout (1 s here)
    want = run(StockTrafilatura(timeout=1, csv_output=True), texts)
    got = run(pdt.Trafilatura(timeout=1, csv_output=True), texts)
    assert got == want
    assert got[1]["timeout"] == 2


def test_unknown_keyword() -> None:
    texts = [tp.PAGES["article"], tp.PAGES["table"]]
    # 2.3.1 rejects it (every page times out, as above); 1.11.0's extract(**kwargs) ignores it
    want = run(StockTrafilatura(timeout=1, bogus=1), texts)
    assert run(pdt.Trafilatura(timeout=1, bogus=1), texts) == want
    if INSTALLED == "1.11.0":
        assert want[1]["forwarded"] == 2


def test_raising_page_restarts_the_cache() -> None:
    bad = f"<html><body><p>{tp.PARAGRAPH} &amp;#{'1' * 4400};</p></body></html>"
    texts = [tp.PAGES["repeated"], bad, tp.PAGES["repeated"], tp.PAGES["repeated"], bad]
    texts += [tp.PAGES["repeated"]] * 3
    for batch_size in (1, 3, 100):
        got = run(pdt.Trafilatura(timeout=60, batch_size=batch_size), texts)
        assert got == stock({}, texts), batch_size
    assert got[1]["broken_process"] == 2


def test_delegated_option_runs_trafilatura(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt, "_WARNED", set())
    texts = pages(40)
    want = stock({"include_links": True}, texts)
    with pytest.warns(pyroid.FallbackWarning, match="include_links"):
        got = run(pdt.Trafilatura(timeout=60, include_links=True), texts)
    assert got == want


def test_fresh_cache_per_run() -> None:
    texts = [tp.PAGES["repeated"]] * 4 + [tp.PAGES["article"]] * 4
    before = list(pt.LRU_TEST.cache.items())
    step = pdt.Trafilatura()
    first = run(step, texts)[0]
    assert "3" not in [d for d, _ in first]  # the 4th copy is a duplicate
    assert [(d.id, d.text) for d in step.run(docs(texts))] == first  # a new task, a new cache
    assert list(pt.LRU_TEST.cache.items()) == before  # the module cache is not used
    assert first == stock({}, texts)[0]


def test_compat_follows_installed_version(monkeypatch: pytest.MonkeyPatch) -> None:
    texts = pages(30)
    want = stock({}, texts)
    monkeypatch.setattr(pt, "_WARNED", set())
    monkeypatch.setattr(pdt, "_WARNED", set())
    # the installed release (stock datatrove installs 1.11.0): pyroid reproduces it, no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert run(pdt.Trafilatura(timeout=60), texts) == want
        assert run(pdt.Trafilatura(timeout=60, compat=INSTALLED), texts) == want  # explicit
    # not installed: pyroid's default release, 2.3.1
    default = run(pdt.Trafilatura(timeout=60, compat="2.3.1"), texts)
    monkeypatch.setattr(pdt, "_installed_trafilatura", lambda: None)
    assert run(pdt.Trafilatura(timeout=60), texts) == default
    if INSTALLED == "2.3.1":
        assert default == want
    # a release pyroid does not reproduce: datatrove's own step, here running the installed one
    # in its child process
    monkeypatch.setattr(pdt, "_installed_trafilatura", lambda: "9.9.9")
    step = pdt.Trafilatura(timeout=60)
    with pytest.warns(
        pyroid.FallbackWarning,
        match=r"trafilatura 9\.9\.9 \(pyroid reproduces 2\.3\.1, 1\.11\.0\)",
    ):
        assert run(step, texts) == want
    assert step._datatrove is False


def test_single_document_api() -> None:
    step = pdt.Trafilatura(deduplicate=False)
    want = trafilatura.extract(tp.PAGES["article"], favor_precision=True, include_comments=False)
    assert step.extract(tp.PAGES["article"]) == want


@pytest.mark.timeout(900)
@pytest.mark.parametrize("workers", [1, 2])
def test_executor_end_to_end(tmp_path: Path, workers: int) -> None:
    """LocalPipelineExecutor, 2 tasks (deep-copied at 1 worker, pickled at 2): same JSONL out."""
    pytest.importorskip("orjson")
    from datatrove.executor import LocalPipelineExecutor  # type: ignore[import-not-found,import-untyped,unused-ignore]
    from datatrove.pipeline.readers import JsonlReader  # type: ignore[import-not-found,import-untyped,unused-ignore]
    from datatrove.pipeline.writers import JsonlWriter  # type: ignore[import-not-found,import-untyped,unused-ignore]

    src = tmp_path / "in"
    src.mkdir()
    texts = pages(60)
    for shard in range(2):
        with open(src / f"{shard}.jsonl", "w", encoding="utf-8") as f:
            for i, t in enumerate(texts[shard::2]):
                f.write(json.dumps({"text": t, "id": f"{shard}-{i}"}) + "\n")

    def pipeline(step: Any, name: str) -> list[str]:
        LocalPipelineExecutor(
            pipeline=[JsonlReader(str(src)), step, JsonlWriter(str(tmp_path / name))],
            tasks=2,
            workers=workers,
            logging_dir=str(tmp_path / f"{name}-logs"),
        ).run()
        import gzip

        lines = []
        for p in sorted((tmp_path / name).glob("*.jsonl*")):
            with gzip.open(p, "rt", encoding="utf-8") as f:
                lines += [json.loads(line)["text"] for line in f]
        return lines

    want = pipeline(StockTrafilatura(timeout=60), "stock")
    got = pipeline(pdt.Trafilatura(), "ours")
    assert got == want
    assert want
