"""pytest plugin (loaded with ``-p pyroid_parity.plugin``): ``--impl`` selection, fixtures, report writing.

    pytest parity/tests --impl pyroid.ftfy                    # ftfy suite vs pyroid.ftfy, other suites self-check
    pytest parity/tests -m ftfy --impl pyroid.ftfy             # only the ftfy suite
    pytest parity/tests --impl filters=pyroid.filters --impl dedup=pyroid.dedup
    pytest parity/tests --parity-limit 500                     # quick run on the first 500 docs of each corpus
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from pyroid_parity import data as D
from pyroid_parity import impl as I
from pyroid_parity.report import Collector, write

_COLLECTOR = Collector()


def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup("pyroid-parity")
    g.addoption("--impl", action="append", default=[], metavar="[SUITE=]MODULE",
                help="implementation under test, e.g. ftfy, pyroid.ftfy, filters=pyroid.filters (repeatable)")
    g.addoption("--parity-report", default="parity-report.json", metavar="PATH", help="where to write the report")
    g.addoption("--parity-limit", type=int, default=0, metavar="N", help="only the first N docs of each corpus")


def pytest_configure(config: pytest.Config) -> None:
    for s in I.SUITES:
        config.addinivalue_line("markers", f"{s}: {s} parity suite")
    config._parity_impls = I.parse_specs(config.getoption("--impl"))  # type: ignore[attr-defined]


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        stem = Path(str(item.fspath)).stem.removeprefix("test_")
        if stem in I.SUITES:
            item.add_marker(getattr(pytest.mark, stem))


def pytest_sessionfinish(session: pytest.Session) -> None:
    if _COLLECTOR.checks or _COLLECTOR.skips:
        out = Path(session.config.getoption("--parity-report"))
        write(_COLLECTOR.build(), out)


def pytest_terminal_summary(terminalreporter: Any) -> None:
    if not _COLLECTOR.checks:
        return
    terminalreporter.section("parity")
    for (suite, check), c in sorted(_COLLECTOR.checks.items()):
        terminalreporter.write_line(f"{suite:9} {check:32} same={c['same']:<8} diff={c['diff']}")


@pytest.fixture(scope="session")
def parity() -> Collector:
    """``parity.record(suite, check, same, case=..., input=..., got=..., want=...)`` -> same."""
    return _COLLECTOR


@pytest.fixture(scope="session")
def parity_limit(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--parity-limit"))


def _impl(request: pytest.FixtureRequest, suite: str) -> I.Impl:
    module = request.config._parity_impls[suite]  # type: ignore[attr-defined]
    explicit = any(spec == module or spec.endswith("=" + module) for spec in request.config.getoption("--impl"))
    try:
        impl = I.resolve(suite, module)
    except ImportError as e:
        if explicit:
            raise
        _COLLECTOR.skip(suite, "import", f"reference {module!r} not importable here: {e}")
        pytest.skip(f"{suite}: reference implementation {module!r} is not installed on this interpreter ({e})")
    _COLLECTOR.impls[suite] = impl.describe()
    return impl


@pytest.fixture(scope="session")
def ftfy_impl(request: pytest.FixtureRequest) -> I.Impl:
    return _impl(request, "ftfy")


@pytest.fixture(scope="session")
def fasttext_impl(request: pytest.FixtureRequest) -> I.Impl:
    return _impl(request, "fasttext")


@pytest.fixture(scope="session")
def filters_impl(request: pytest.FixtureRequest) -> I.Impl:
    return _impl(request, "filters")


@pytest.fixture(scope="session")
def dedup_impl(request: pytest.FixtureRequest) -> I.Impl:
    return _impl(request, "dedup")


def reference(suite: str, name: str, version: str | None, tolerance: float | None = None) -> None:
    """Record which oracle (and tolerance) a suite's pinned outputs came from."""
    _COLLECTOR.references[suite] = {"name": name, "version": version, "tolerance": tolerance}


def need(rel: str) -> Path:
    """Pinned file or skip (``bench/fetch.py`` / ``pyroid_parity.refs`` not run on this machine)."""
    try:
        p = D.path(rel)
    except D.DataError as e:
        pytest.skip(str(e))
    return p


def need_corpus(name: str, limit: int = 0) -> D.Corpus:
    need(D.CORPORA[name])
    c = D.corpus(name)
    return c.head(limit) if limit else c
