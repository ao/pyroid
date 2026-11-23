"""dateutil parity: the ``--impl`` candidate's ``dateutil.parser`` vs the installed python-dateutil (live).

    pytest tests/test_dateutil.py                                    # self-check (candidate = dateutil)
    pytest tests/test_dateutil.py --impl pyroid.dateutil             # pyroid vs python-dateutil
    PYROID_DATEUTIL_CORPUS=/target/dateutil-corpus pytest tests/test_dateutil.py --impl pyroid.dateutil

The candidate is a package with a ``parser`` submodule (``dateutil``, ``pyroid.dateutil``). Outcomes are
compared exactly: the ``datetime`` (value, type, ``fold``), its ``tzinfo`` type and ``repr``, the skipped
tokens of ``fuzzy_with_tokens``, the exception type and message and the type of its ``__cause__``, and the
warnings other than ``pyroid.FallbackWarning`` (category and message). Option sets:
``pyroid_parity.specs.DATEUTIL_OPTION_SETS``.

Checks (suite ``dateutil`` in parity-report.json):
    upstream-<release>     python-dateutil's own tests/test_parser.py and test_isoparser.py for the installed
                           release (``data/dateutil-<release>/upstream_tests/``), run once against dateutil and
                           once with ``from dateutil.parser import`` pointing at the candidate: the same outcome
                           (passed, failed, xfailed, skipped) for every test
    issues-<option>        747 strings reported in dateutil's and pandas' issue trackers
                           (``data/dateutil/issue-strings.json``) under each option set
    issues-recorded        the same strings with the keywords their issue used
    isoparse-issues        ``isoparse`` of the same strings
    hypothesis-parse       generated date-like token strings, every keyword, ``default`` values
    hypothesis-isoparse    generated ISO-like strings, each ``sep``
    batch                  ``parse_batch`` / ``isoparse_batch`` (when the candidate has them) equal the
                           candidate's own per-string results
    corpus-<option>        real strings (``$PYROID_DATEUTIL_CORPUS``: one ``<source>.txt`` per source), each
                           under each corpus option set; skipped when the variable is unset
"""

from __future__ import annotations

import datetime
import importlib
import importlib.metadata as md
import json
import os
import re
import shutil
import subprocess
import sys
import warnings
import xml.etree.ElementTree as ET
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pyroid_parity.plugin import _impl, reference
from pyroid_parity.report import Collector
from pyroid_parity.specs import DATEUTIL_CORPUS_OPTION_SETS, DATEUTIL_OPTION_SETS

dateutil_parser = pytest.importorskip("dateutil.parser")
tz = pytest.importorskip("dateutil.tz")

SUITE = "dateutil"
DATA = Path(__file__).resolve().parents[1] / "data"
INSTALLED = md.version("python-dateutil")
ISSUES: list[dict[str, Any]] = json.loads((DATA / "dateutil" / "issue-strings.json").read_text("utf-8"))[
    "strings"
]


@pytest.fixture(scope="session")
def dateutil_impl(request: pytest.FixtureRequest) -> Any:
    return _impl(request, SUITE)


@pytest.fixture(scope="module", autouse=True)
def _oracle() -> None:
    reference(SUITE, "python-dateutil", INSTALLED, None)


def _options(name: str) -> dict[str, Any]:
    """The keyword arguments of option set `name` (tzinfos and defaults built here)."""
    kw = dict(DATEUTIL_OPTION_SETS[name])
    if "default" in kw:
        kw["default"] = datetime.datetime(*kw["default"])
    if kw.get("tzinfos") == "names":
        kw["tzinfos"] = {
            "BRST": -10800,
            "EST": tz.tzoffset("EST", -18000),
            "EDT": -14400,
            "CET": 3600,
            "PST": "PST8PDT",
            "IST": 19800,
        }
    return kw


def rep(r: Any) -> Any:
    if isinstance(r, tuple):
        return ("tuple", rep(r[0]), r[1])
    if isinstance(r, datetime.datetime):
        try:
            iso = r.isoformat()
        except ValueError:  # a tzoffset of a day or more: datetime cannot use it
            iso = repr(r)
        return (type(r).__name__, iso, r.fold, repr(r.tzinfo), type(r.tzinfo).__name__)
    return ("other", repr(r))


def outcome(f: Any, s: Any, kw: dict[str, Any]) -> Any:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out: Any = ("ok", rep(f(s, **kw)))
        except Exception as e:  # noqa: BLE001 - compared
            out = ("err", type(e).__name__, str(e), type(e.__cause__).__name__)
    seen = tuple((x.category.__name__, str(x.message)) for x in w if x.category.__name__ != "FallbackWarning")
    return out, seen


def check(parity: Collector, name: str, got: Any, want: Any, *, case: str | None = None, input: Any = None):
    same = parity.record(SUITE, name, got == want, case=case, input=input, got=got, want=want)
    assert same, f"{name} {case or ''}\ninput: {input!r}\n  got: {got}\n want: {want}"


def test_installed_release_is_reproduced(dateutil_impl: Any) -> None:
    if dateutil_impl.is_reference:
        return
    versions = getattr(dateutil_impl.obj.parser, "DATEUTIL_VERSIONS", ())
    assert INSTALLED in versions, f"python-dateutil {INSTALLED} is not one the candidate reproduces"


# ------------------------------------------------------------------------------- upstream test suite


def _junit(path: Path) -> dict[str, str]:
    # The JUnit XML that our own pytest subprocess wrote (trusted input).
    out = {}
    for tc in ET.parse(path).iter("testcase"):
        name = tc.get("classname", "").split(".", 1)[-1] + "::" + tc.get("name", "")
        kind = "passed"
        for k in ("failure", "error", "skipped"):
            if tc.find(k) is not None:
                kind = k
        out[name] = kind
    return out


@cache
def _upstream(module: str, tmp: str) -> dict[str, str]:
    """Outcome per test of python-dateutil's own parser tests, with `module` as dateutil.parser."""
    src = DATA / f"dateutil-{INSTALLED}" / "upstream_tests"
    work = Path(tmp) / module.replace(".", "_")
    shutil.copytree(src, work / "tests")
    for f in ("test_parser.py", "test_isoparser.py"):
        p = work / "tests" / f
        text = p.read_text("utf-8")
        if module != "dateutil.parser":
            text = re.sub(r"^from dateutil\.parser import", f"from {module} import", text, flags=re.MULTILINE)
        p.write_text(text, "utf-8")
    xml = work / "junit.xml"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
        "-W",
        "ignore::DeprecationWarning",
        f"--junitxml={xml}",
        "tests/test_parser.py",
        "tests/test_isoparser.py",
    ]
    env = {**os.environ, "PYTHONWARNINGS": "ignore"}
    subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, check=False, timeout=600)
    return _junit(xml)


def test_upstream(dateutil_impl: Any, parity: Collector, tmp_path_factory: pytest.TempPathFactory) -> None:
    if not (DATA / f"dateutil-{INSTALLED}").is_dir():
        pytest.skip(f"no vendored tests for python-dateutil {INSTALLED}")
    tmp = str(tmp_path_factory.mktemp("du-upstream"))
    want = _upstream("dateutil.parser", tmp)
    got = _upstream(dateutil_impl.obj.parser.__name__, tmp)
    assert want, "the upstream tests did not run"
    for test in sorted(set(want) | set(got)):
        check(parity, f"upstream-{INSTALLED}", got.get(test), want.get(test), case=test)


# -------------------------------------------------------------------------------------- issue strings


@pytest.mark.parametrize("option", list(DATEUTIL_OPTION_SETS))
def test_issues(dateutil_impl: Any, parity: Collector, option: str) -> None:
    kw = _options(option)
    for item in ISSUES:
        s = item["string"]
        want = outcome(dateutil_parser.parse, s, kw)
        got = outcome(dateutil_impl.obj.parser.parse, s, kw)
        check(parity, f"issues-{option}", got, want, case=item["url"], input=s)


def test_issues_recorded(dateutil_impl: Any, parity: Collector) -> None:
    for item in ISSUES:
        kw = item["kwargs"]
        want = outcome(dateutil_parser.parse, item["string"], kw)
        got = outcome(dateutil_impl.obj.parser.parse, item["string"], kw)
        check(parity, "issues-recorded", got, want, case=item["url"], input=(item["string"], kw))


def test_isoparse_issues(dateutil_impl: Any, parity: Collector) -> None:
    for item in ISSUES:
        want = outcome(dateutil_parser.isoparse, item["string"], {})
        got = outcome(dateutil_impl.obj.parser.isoparse, item["string"], {})
        check(parity, "isoparse-issues", got, want, case=item["url"], input=item["string"])


# ------------------------------------------------------------------------------------------- Hypothesis

PIECES = (
    [str(i) for i in range(61)]
    + ["%02d" % i for i in range(13)]  # noqa: UP031
    + ["1999", "2003", "99", "00", "100", "9999", "10000", "123456", "20030925", "200309251049"]
    + ["20030925104941", "19990101T23", "235959.5", "1.5", "10.25", "3,5", "12,5", "0.000001"]
    + ["Jan", "January", "feb", "Sep", "Sept", "Mar.", "May", "dec", "Thu", "Thursday", "sun"]
    + ["am", "pm", "AM", "PM", "a", "p", "A.M.", "P.M.", "a.m.", "m", "UTC", "GMT", "Z", "z"]
    + ["BRST", "EST", "CET", "PST", "XYZ", "ABCDEF", "utc", "h", "s", "hours", "minute"]
    + ["of", "at", "on", "and", "ad", "T", "t", "st", "nd", "rd", "th", " ", "  ", "\t", "-"]
    + ["/", ":", ".", ",", ";", "'", "+", "(", ")", "_", "\x00", "\n", "　", "inf", "nan"]
    + ["١٢", "٣", "²", "½", "é", "ß", "İ", "K"]
    + ["\ud800", "\udfff", "\U0010f801"]
)
text = st.lists(
    st.one_of(st.sampled_from(PIECES), st.text(min_size=1, max_size=3)), min_size=0, max_size=14
).map("".join)


@given(s=text, option=st.sampled_from(list(DATEUTIL_OPTION_SETS)), fuzzy=st.booleans())
def test_hypothesis_parse(dateutil_impl: Any, parity: Collector, s: str, option: str, fuzzy: bool) -> None:
    kw = _options(option)
    if fuzzy:
        kw["fuzzy"] = True
    want = outcome(dateutil_parser.parse, s, kw)
    got = outcome(dateutil_impl.obj.parser.parse, s, kw)
    check(parity, "hypothesis-parse", got, want, case=option, input=s)


iso_text = st.lists(
    st.sampled_from(
        [
            *["2003", "-", "09", "25", "T", "10", ":", "49", "41", ".", ",", "5", "123456789", "Z"],
            *["+", "05", "30", "W", "39", "4", "268", " ", "_", "24", "00", "60", "0", "z", "é"],
        ]
    ),
    max_size=12,
).map("".join)


@given(s=iso_text, sep=st.sampled_from([None, "T", " "]))
def test_hypothesis_isoparse(dateutil_impl: Any, parity: Collector, s: str, sep: str | None) -> None:
    want = outcome(dateutil_parser.isoparser(sep).isoparse, s, {})
    got = outcome(dateutil_impl.obj.parser.isoparser(sep).isoparse, s, {})
    check(parity, "hypothesis-isoparse", got, want, input=(s, sep))


# --------------------------------------------------------------------------------------------- batches


def test_batch(dateutil_impl: Any, parity: Collector) -> None:
    p = dateutil_impl.obj.parser
    if not hasattr(p, "parse_batch"):
        parity.skip(SUITE, "batch", "the candidate has no parse_batch")
        pytest.skip("no parse_batch")
    strings = [item["string"] for item in ISSUES if "\x00" not in item["string"]]
    strings = [s for s in strings if not re.search(r"[\ud800-\udfff]", s)]
    for option in ("default", "dayfirst", "fuzzy"):
        kw = _options(option)
        want = []
        keep = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for s in strings:
                try:
                    want.append(rep(p.parse(s, **kw)))
                except (ValueError, OverflowError):
                    want.append(None)
                except ArithmeticError:  # decimal.InvalidOperation: not coerced, and raised as in parse
                    continue
                keep.append(s)
            got = [None if r is None else rep(r) for r in p.parse_batch(keep, errors="coerce", **kw)]
        for s, g, w in zip(keep, got, want, strict=True):
            check(parity, "batch", g, w, case=option, input=s)
    want = []
    for s in strings:
        try:
            want.append(rep(p.isoparse(s)))
        except ValueError:
            want.append(None)
    got = [None if r is None else rep(r) for r in p.isoparse_batch(strings, errors="coerce")]
    for s, g, w in zip(strings, got, want, strict=True):
        check(parity, "batch", g, w, case="isoparse", input=s)


# ---------------------------------------------------------------------------------------- real strings

CORPUS = os.environ.get("PYROID_DATEUTIL_CORPUS")
CORPUS_DEFAULT_LIMIT = 2000


def _corpus_files() -> list[Path]:
    if not CORPUS or not Path(CORPUS).is_dir():
        return []
    return sorted(Path(CORPUS).glob("*.txt"))


@pytest.mark.parametrize("option", list(DATEUTIL_CORPUS_OPTION_SETS))
def test_corpus(dateutil_impl: Any, parity: Collector, request: pytest.FixtureRequest, option: str) -> None:
    files = _corpus_files()
    if not files:
        parity.skip(
            SUITE, f"corpus-{option}", "PYROID_DATEUTIL_CORPUS is not set (real strings: see the docstring)"
        )
        pytest.skip("no corpus")
    kw = _options(option)
    cand = dateutil_impl.obj.parser.parse
    # The whole corpus (9 million strings) takes hours here; by default each source's first
    # CORPUS_DEFAULT_LIMIT strings (--parity-limit N picks N, and 0 every string).
    raw = request.config.getoption("--parity-limit")
    limit = CORPUS_DEFAULT_LIMIT if raw is None else int(raw)
    for f in files:
        lines = [x for x in f.read_text("utf-8", errors="replace").split("\n") if x]
        if limit > 0:
            lines = lines[:limit]
        for s in lines:
            want = outcome(dateutil_parser.parse, s, kw)
            got = outcome(cand, s, kw)
            check(parity, f"corpus-{option}", got, want, case=f.stem, input=s)


def test_candidate_module(dateutil_impl: Any) -> None:
    assert importlib.import_module(dateutil_impl.obj.parser.__name__) is dateutil_impl.obj.parser
