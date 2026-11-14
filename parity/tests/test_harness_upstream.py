"""Self-tests of the upstream watch (pyroid_parity.upstream, .github/workflows/upstream.yml), offline.

PyPI's JSON is faked: newer stable releases, pre-releases, yanked releases, malformed version strings and project
URLs. The last tests read the real pyproject files and dependabot.yml, so a new exact pin must be classified.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from pyroid_parity import impl as I
from pyroid_parity import plugin, specs
from pyroid_parity import upstream as U

REPO = Path(__file__).resolve().parents[2]


def file(date: str = "2026-11-07T10:00:00.123456Z", yanked: bool = False) -> dict[str, Any]:
    return {"packagetype": "sdist", "upload_time_iso_8601": date, "yanked": yanked}


def pypi(name: str, releases: dict[str, list[dict[str, Any]]], urls: dict[str, str] | None = None) -> dict:
    return {"info": {"name": name, "project_urls": urls, "home_page": None}, "releases": releases}


ROOT_PYPROJECT = """
[project]
name = "fake"

[dependency-groups]
test = [
  "pytest>=9",
  "json-repair==0.64.9",
  "nltk==3.10.4",
  "regex==2026.9.29",
  "numpy>=1.26",
  "fasttext-predict==0.9.2.4; python_version < '3.14'",
]
test-gil = ["trafilatura==2.3.1", "lxml==6.1.3"]
test-trafilatura-111 = ["trafilatura==1.11.0", "lxml==6.1.3", "courlan==1.4.0"]
dev = [{ include-group = "test" }]
"""
PARITY_PYPROJECT = """
[project]
name = "fake-parity"
dependencies = ["json-repair==0.64.5", "pytest>=9.2", "nltk==3.10.4"]

[dependency-groups]
oracle = ["datatrove==0.10.1", "regex>=2024.11"]
trafilatura = ["trafilatura==2.3.1", "lxml==6.1.3"]
trafilatura111 = ["trafilatura==1.11.0", "lxml==6.1.3", "courlan==1.4.0"]
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "parity" / "pyroid_parity").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(ROOT_PYPROJECT, "utf-8")
    (tmp_path / "parity" / "pyproject.toml").write_text(PARITY_PYPROJECT, "utf-8")
    (tmp_path / "parity" / "pyroid_parity" / "nemo_cases.py").write_text('NEMO_VERSION = "1.4.0"\n', "utf-8")
    return tmp_path


FAKE = {
    "json-repair": pypi(
        "json_repair",
        {"0.64.5": [file()], "0.64.9": [file()], "0.65.0": [file(), file("2026-11-12T01:02:03Z")]},
        {"Homepage": "https://github.com/mangiucugna/json_repair/"},
    ),
    "nltk": pypi("nltk", {"3.10.4": [file()], "3.10.5": [file("2026-11-10T00:00:00Z")]}),
    "regex": pypi("regex", {"2026.9.29": [file()], "2026.11.9": [file("2026-11-09T00:39:39Z")]}),
    "datatrove": pypi("datatrove", {"0.10.1": [file()], "0.11.0": [file()]}),
    "lxml": pypi("lxml", {"6.1.3": [file()], "6.2.0": [file()]}),
    "trafilatura": pypi("trafilatura", {"1.11.0": [file()], "1.12.2": [file()], "2.3.1": [file()]}),
    "courlan": pypi("courlan", {"1.4.0": [file()]}),
    "fasttext-predict": pypi("fasttext-predict", {"0.9.2.4": [file()], "0.9.3rc1": [file()]}),
    "nemo-curator": pypi("nemo_curator", {"1.4.0": [file()], "1.5.0": [file()]}),
}


def fake_fetch(name: str) -> dict[str, Any]:
    if name not in FAKE:
        raise U.UpstreamError(f"{name}: HTTP 404")
    return FAKE[name]


# ---------------------------------------------------------------------------------------- releases


def test_newer_stable_release_found():
    rel = U.releases_after(FAKE["json-repair"], "0.64.5")
    assert (rel.latest, rel.newer, rel.released) == ("0.65.0", "0.65.0", "2026-11-07")
    assert rel.prerelease == rel.yanked == [] and rel.malformed == 0


def test_only_prereleases_are_newer():
    data = pypi(
        "x", {"1.0": [file()], "1.1rc1": [file()], "1.1.dev3": [file()], "1.1a2": [file()], "0.9": []}
    )
    rel = U.releases_after(data, "1.0")
    assert rel.newer is None and rel.latest == "1.0"
    assert rel.prerelease == ["1.1.dev3", "1.1a2", "1.1rc1"]


def test_yanked_and_empty_releases_are_skipped():
    data = pypi(
        "x",
        {
            "1.0": [file()],
            "1.1": [file(yanked=True), file()],  # one file left: installable
            "1.2": [file(yanked=True), file(yanked=True)],
            "1.3": [],
        },
    )
    rel = U.releases_after(data, "1.0")
    assert (rel.newer, rel.latest) == ("1.1", "1.1")
    assert rel.yanked == ["1.2", "1.3"]


def test_post_release_counts_as_newer():
    rel = U.releases_after(pypi("x", {"1.2.1.0": [file()], "1.2.1.0.post1": [file()]}), "1.2.1.0")
    assert rel.newer == "1.2.1.0.post1"


def test_pin_already_latest():
    rel = U.releases_after(pypi("x", {"0.42.1": [file()], "0.42.0": [file()]}), "0.42.1")
    assert rel.newer is None and rel.latest == "0.42.1"


MALFORMED = [
    "9.0; rm -rf /",
    "9.0\n",
    "9.0 ",
    "$(id)",
    "`id`",
    "v9.0",
    "9.0+local",
    "1!9.0",
    "09.0",  # not canonical (leading zero)
    "9.0-beta",  # not canonical (9.0b0)
    "9.0.0.0.0.0.0",  # more than 6 release components
    "９.0",  # FULLWIDTH DIGIT NINE
    "9" * 13,
    "",
]


@pytest.mark.parametrize("bad", MALFORMED)
def test_malformed_version_rejected(bad):
    assert U.parse_version(bad) is None
    with pytest.raises(ValueError, match="not a stable canonical version"):
        U.checked_version(bad)


def test_malformed_release_keys_are_ignored():
    data = pypi("x", {"1.0": [file()], **{bad: [file()] for bad in MALFORMED}, "2.0": "not a list"})
    rel = U.releases_after(data, "1.0")
    assert rel.newer is None and rel.malformed == len(MALFORMED) + 1


@pytest.mark.parametrize("ok", ["0.64.9", "2026.11.9", "1.2.1.0", "20261015", "1.0.post2"])
def test_canonical_versions_accepted(ok):
    assert U.checked_version(ok) == ok


@pytest.mark.parametrize("pre", ["1.0rc1", "1.0a1", "1.0b2", "1.0.dev0", "1.0.post1.dev1"])
def test_prereleases_are_not_stable(pre):
    assert U.parse_version(pre) is not None
    with pytest.raises(ValueError, match="stable"):
        U.checked_version(pre)


def test_releases_missing():
    with pytest.raises(U.UpstreamError, match="no 'releases'"):
        U.releases_after({"info": {"name": "x"}}, "1.0")


@pytest.mark.parametrize(
    ("urls", "home", "want"),
    [
        (
            {"Changelog": "https://example.org/CHANGES.md", "Source": "https://github.com/o/r"},
            None,
            "https://example.org/CHANGES.md",
        ),
        (
            {"Release notes": "https://github.com/explosion/spaCy/releases", "Homepage": "https://spacy.io"},
            None,
            "https://github.com/explosion/spaCy/releases",
        ),
        (
            {"Homepage": "https://github.com/mangiucugna/json_repair/"},
            None,
            "https://github.com/mangiucugna/json_repair/releases",
        ),
        (
            {"source": "https://github.com/PyThaiNLP/pythainlp.git"},
            None,
            "https://github.com/PyThaiNLP/pythainlp/releases",
        ),
        ({}, "https://github.com/o/r", "https://github.com/o/r/releases"),
        (None, None, "https://pypi.org/project/x/#history"),
        ({"Changelog": "javascript:alert(1)"}, None, "https://pypi.org/project/x/#history"),
        ({"Changelog": "http://example.org/changes"}, None, "https://pypi.org/project/x/#history"),
        (
            {"Changelog": "https://example.org/a)](https://evil.example"},
            None,
            "https://pypi.org/project/x/#history",
        ),
        ({"Changelog": "https://example.org/a b"}, None, "https://pypi.org/project/x/#history"),
        ({"Changelog": "https://example.org/a|b"}, None, "https://pypi.org/project/x/#history"),
    ],
)
def test_changes_url(urls, home, want):
    assert U.changes_url({"project_urls": urls, "home_page": home}, "x") == want


# -------------------------------------------------------------------------------------------- pins


def test_pins_disagree_between_files(repo):
    choice = U.choose_pin("json-repair", U.read_pins(repo))
    assert choice is not None and choice.disagree and choice.version == "0.64.9"
    assert {p.where() for p in choice.pins} == {"pyproject.toml [test]", "parity/pyproject.toml [project]"}


def test_compat_group_pins(repo):
    pins = U.read_pins(repo)
    traf = U.choose_pin("trafilatura", pins)
    assert (
        traf is not None and traf.version == "2.3.1" and not traf.disagree
    )  # 1.11.0 is the compat reference
    courlan = U.choose_pin("courlan", pins)
    assert courlan is not None and courlan.version == "1.4.0" and all(p.compat for p in courlan.pins)
    nemo = U.choose_pin("nemo-curator", pins)
    assert nemo is not None and nemo.version == "1.4.0"
    assert nemo.pins[0].where() == "parity/pyroid_parity/nemo_cases.py [NEMO_VERSION]"
    assert U.choose_pin("anyascii", pins) is None
    # Floors (>=) and include-group entries are not pins; markers are dropped.
    assert {p.name for p in pins} >= {"fasttext-predict"} and "numpy" not in {p.name for p in pins}
    assert not any(p.name == "regex" and p.file == "parity/pyproject.toml" for p in pins)


# ------------------------------------------------------------------------------------------ detect


def by_name(found: dict) -> dict[str, dict]:
    return {e["library"]: e for e in found["libraries"]}


def test_detect_and_suite_mapping(repo):
    found = U.detect(repo, fake_fetch)
    lib = by_name(found)
    jr = lib["json-repair"]
    assert (jr["pinned"], jr["newer"], jr["released"], jr["disagree"]) == (
        "0.64.9",
        "0.65.0",
        "2026-11-07",
        True,
    )
    assert jr["pypi"] == "https://pypi.org/project/json-repair/0.65.0/"
    assert jr["changes"] == "https://github.com/mangiucugna/json_repair/releases"
    assert lib["trafilatura"]["newer"] is None  # 1.12.2 is newer than the compat pin only
    assert lib["courlan"]["newer"] is None and lib["courlan"]["error"] is None
    assert lib["fasttext-predict"]["newer"] is None and lib["fasttext-predict"]["prerelease"] == ["0.9.3rc1"]
    assert lib["nemo-curator"]["newer"] == "1.5.0"
    assert lib["anyascii"]["error"].startswith("not pinned")
    matrix = {(m["library"], m["suite"]): m for m in found["matrix"]["include"]}
    assert set(matrix) == {
        ("json-repair", "json_repair"),
        ("nltk", "nltk"),
        ("nltk", "metrics"),
        ("regex", "nltk"),
        ("regex", "metrics"),
        ("datatrove", "filters"),
        ("datatrove", "dedup"),
        ("datatrove", "warc"),
        ("lxml", "trafilatura"),
        ("lxml", "trafilatura@1.11.0"),
    }  # nemo-curator has no suite here
    assert matrix[("json-repair", "json_repair")] == {
        "library": "json-repair",
        "pinned": "0.64.9",
        "version": "0.65.0",
        "suite": "json_repair",
        "data": False,
    }
    assert matrix[("datatrove", "warc")]["data"] is True


def test_detect_reports_fetch_errors_and_other_projects(repo):
    def fetch(name):
        if name == "nltk":
            raise U.UpstreamError("https://pypi.org/pypi/nltk/json: HTTP 503")
        if name == "regex":
            return pypi("regex2", {"2027.1.1": [file()]})
        return fake_fetch(name)

    lib = by_name(U.detect(repo, fetch))
    assert "HTTP 503" in lib["nltk"]["error"] and lib["nltk"]["newer"] is None
    assert lib["regex"]["error"] == "PyPI's JSON is for another project"
    assert lib["json-repair"]["newer"] == "0.65.0"


def test_errors_cannot_start_a_workflow_command():
    assert U._command_data("x\n::add-mask::y %0A\r") == "x%0A::add-mask::y %250A%0D"
    found = {"libraries": [{**by_name(U.detect(Path(REPO), fake_fetch))["ftfy"], "error": "a|b\n`c`"}]}
    found["unclassified"] = []
    assert "error: a b  c " in U.detect_summary(found)


def test_watch_suites_are_parity_suites():
    for name, w in U.WATCH.items():
        assert U.NAME_RE.fullmatch(name), name
        assert set(w.suites) <= U.KNOWN_SUITES, name
    for suite in U.KNOWN_SUITES:
        base, _, mode = suite.partition("@")
        assert base in I.SUITES and (not mode or mode in specs.TRAFILATURA_MODES)
    assert not set(U.WATCH) & set(U.NOT_WATCHED)


def test_cli_detect_writes_outputs(repo, tmp_path, monkeypatch):
    out, summary = tmp_path / "gh-output", tmp_path / "gh-summary"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert (
        U.main(["detect", "--root", str(repo), "--out", str(tmp_path / "found.json")], fetch=fake_fetch) == 0
    )
    outputs = dict(line.split("=", 1) for line in out.read_text("utf-8").splitlines())
    assert json.loads(outputs["matrix"]) == json.loads((tmp_path / "found.json").read_text("utf-8"))["matrix"]
    assert (outputs["parity"], outputs["found"]) == ("true", "true")
    # Every watched library the fake repo does not pin is an error, as a missing pin would be.
    assert int(outputs["errors"]) == sum(
        1 for e in by_name(U.detect(repo, fake_fetch)).values() if e["error"]
    )
    assert "| json-repair | 0.64.9 | 0.65.0 |" in summary.read_text("utf-8")


# ------------------------------------------------------------------------------------------ render


def write_results(root: Path, library: str, suite: str, status: dict, impl: dict | None, self_: dict | None):
    d = root / f"upstream-parity-{library}-{suite}"
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps(status), "utf-8")
    for name, rep in (("parity-report.json", impl), ("parity-self.json", self_)):
        if rep is not None:
            (d / name).write_text(json.dumps(rep), "utf-8")


def report(suite: str, checks: dict[str, tuple[str, int, int]]) -> dict:
    return {
        "suites": {
            suite: {"checks": {k: {"status": s, "same": a, "diff": b} for k, (s, a, b) in checks.items()}}
        }
    }


ENV = {"GITHUB_REPOSITORY": "ao/pyroid", "GITHUB_RUN_ID": "123", "GITHUB_SHA": "ab" * 20}


def test_render_issue(repo, tmp_path):
    found = U.detect(repo, fake_fetch)
    results = tmp_path / "results"
    write_results(
        results,
        "json-repair",
        "json_repair",
        {"install": 0, "impl": 1, "self": 1, "data": "unused"},
        report("json_repair", {"upstream": ("pass", 672, 0), "corpus-default": ("fail", 19988, 12)}),
        report("json_repair", {"upstream": ("fail", 670, 2), "x](https://evil)": ("fail", 0, 1)}),
    )
    rows = U.render(found, results, tmp_path / "issues", ENV)
    titles = {lib: t for lib, t, _ in rows}
    assert titles["json-repair"] == "upstream: json-repair 0.65.0 released"
    assert titles["nemo-curator"] == "upstream: nemo-curator 1.5.0 released"
    index = (tmp_path / "issues" / "index.tsv").read_text("utf-8").splitlines()
    assert len(index) == len(rows) and all(len(line.split("\t")) == 3 for line in index)
    body = (tmp_path / "issues" / "json-repair.md").read_text("utf-8")
    assert "json-repair 0.65.0 is on PyPI. pyroid is checked against 0.64.9" in body
    assert "| New | 0.65.0, released 2026-11-07 |" in body
    assert "| Run | <https://github.com/ao/pyroid/actions/runs/123> (commit abababababab) |" in body
    assert "The pins disagree: pyproject.toml [test] 0.64.9, parity/pyproject.toml [project] 0.64.5." in body
    assert "| `json_repair` | fail (checks: 1 pass, 1 fail, 0 skip; 20,660 same, 12 different) |" in body
    assert "`corpus-default` (12 of 20,000)" in body and "`upstream` (2 of 672)" in body
    assert "evil" not in body and "`?` (1 of 1)" in body
    assert "- [ ] Move the pin in parity/pyproject.toml, pyproject.toml" in body
    # Suites whose job did not finish or whose release did not install, and libraries without a suite.
    lxml = (tmp_path / "issues" / "lxml.md").read_text("utf-8")
    assert "| `trafilatura` | no result: the parity job did not finish (see the run) | |" in lxml
    nemo = (tmp_path / "issues" / "nemo-curator.md").read_text("utf-8")
    assert "No parity suite runs for this library here." in nemo


def test_render_install_failure_and_cache_miss(repo, tmp_path):
    found = U.detect(repo, fake_fetch)
    results = tmp_path / "results"
    write_results(
        results, "nltk", "nltk", {"install": 1, "impl": None, "self": None, "data": "miss"}, None, None
    )
    write_results(
        results,
        "nltk",
        "metrics",
        {"install": 0, "impl": 0, "self": 4, "data": "miss"},
        report("metrics", {"rouge": ("pass", 5, 0), "bleu": ("skip", 0, 0)}),
        None,
    )
    U.render(found, results, tmp_path / "issues", ENV)
    body = (tmp_path / "issues" / "nltk.md").read_text("utf-8")
    assert "| `nltk` | 3.10.5 did not install over the pin (see the run) | not run |" in body
    assert (
        "| `metrics` | pass (checks: 1 pass, 0 fail, 1 skip; 5 same, 0 different) | no report (pytest: usage error) |"
        in body
    )
    assert "skipped in: metrics." in body


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("newer", "0.65.0;id"),
        ("library", "json repair"),
        ("suites", ["json_repair", "evil"]),
        ("changes", "http://x"),
    ],
)
def test_render_rechecks_found_json(repo, tmp_path, key, value):
    found = U.detect(repo, fake_fetch)
    by_name(found)["json-repair"][key] = value
    with pytest.raises(ValueError, match=r"not a|unknown suite|bad changes URL"):
        U.render(found, tmp_path, tmp_path / "issues", ENV)


# ------------------------------------------------------------------------------- plugin: --upstream-oracle


def test_oracle_pin(monkeypatch):
    monkeypatch.delenv(plugin.UPSTREAM_ENV, raising=False)
    assert plugin.oracle_pin("json-repair", "0.64.9") == "0.64.9"
    monkeypatch.setenv(plugin.UPSTREAM_ENV, "json_repair==0.65.0 Regex==2026.11.9")
    assert plugin.oracle_pin("json-repair", "0.64.9") == "0.65.0"
    assert plugin.oracle_pin("regex", "2026.9.29") == "2026.11.9"
    assert plugin.oracle_pin("nltk", "3.10.4") == "3.10.4"


@pytest.mark.parametrize(
    "bad", ["json-repair", "json-repair=0.65.0", "json-repair==0.65.0rc1", "==1.0", "x==1;id"]
)
def test_upstream_oracle_rejects(bad):
    with pytest.raises(ValueError, match=r"expected DIST==VERSION|not a stable canonical version"):
        plugin.parse_upstream_oracle(bad)


# ------------------------------------------------------------------------------------ the real repository


def test_repo_pins_are_classified_and_agree():
    pins = U.read_pins(REPO)
    assert U.unclassified(pins) == [], (
        "classify these exact pins in pyroid_parity/upstream.py (WATCH or NOT_WATCHED)"
    )
    for name in U.WATCH:
        choice = U.choose_pin(name, pins)
        assert choice is not None, f"{name} is in WATCH but pinned nowhere"
        assert not choice.disagree, (
            f"{name}: the pins disagree: {[(p.where(), p.version) for p in choice.pins]}"
        )


def test_dependabot_ignores_are_classified():
    text = (REPO / ".github" / "dependabot.yml").read_text("utf-8")
    ignored = re.findall(r"^\s*- dependency-name: ([A-Za-z0-9._-]+)\s*$", text, re.MULTILINE)
    assert "json-repair" in ignored
    assert [n for n in ignored if n not in U.WATCH and n not in U.NOT_WATCHED] == []
