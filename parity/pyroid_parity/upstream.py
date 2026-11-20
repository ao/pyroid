"""The upstream watch: newer PyPI releases of the libraries pyroid reproduces, and one issue text per library.

    python -I parity/pyroid_parity/upstream.py detect --root . --out found.json
    python -I parity/pyroid_parity/upstream.py render --found found.json --results DIR --out DIR

``.github/workflows/upstream.yml`` runs ``detect`` weekly. For each library with a release newer than its pin it
runs the library's parity suites against that release (``--upstream-oracle``, see ``pyroid_parity.plugin``), and
``render`` turns the parity reports into one issue body per library.

The pins are the exact (``==``) requirements of the root pyproject.toml's dependency groups and of
parity/pyproject.toml, plus NeMo Curator's version in ``nemo_cases.py``. ``WATCH`` maps each watched library to
the suites that exercise it, ``NOT_WATCHED`` says why the other exact pins are left out; every exact pin must be in
one of the two (``tests/test_harness_upstream.py`` checks the real files).

Everything read from PyPI is untrusted: names, versions, dates and URLs must match the patterns below before they
reach a title, the job matrix, a shell or ``uv``. Needs the standard library and ``packaging``; runs as a script
(``python -I``), so it imports nothing from pyroid_parity.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

SCHEMA = 1


@dataclass(frozen=True)
class Watch:
    """A watched library: the parity suites that exercise it (``SUITE`` or ``SUITE@MODE``) and why."""

    suites: tuple[str, ...]
    why: str


_T111 = (
    'Part of pyroid.trafilatura\'s compat="1.11.0" reference, what a fresh datatrove[processing]==0.10.1 install '
    "resolves; only that mode pins it."
)
_BOTH_MODES = ("trafilatura", "trafilatura@1.11.0")
_TOKENIZERS = ("filters", "dedup")

WATCH: dict[str, Watch] = {
    "ftfy": Watch(("ftfy",), "pyroid.ftfy reproduces ftfy."),
    "json-repair": Watch(("json_repair",), "pyroid.json_repair reproduces json_repair."),
    "datasketch": Watch(("datasketch",), "pyroid.datasketch reproduces datasketch."),
    "jieba": Watch(("jieba",), "pyroid.jieba reproduces jieba (also datatrove's Chinese tokenizer)."),
    "warcio": Watch(("warc",), "pyroid.warc reproduces warcio's reader."),
    "trafilatura": Watch(
        ("trafilatura",),
        'pyroid.trafilatura reproduces trafilatura 2.x, its default mode. The compat="1.11.0" mode is not '
        "compared with newer 1.x releases: datatrove 0.10.1's processing extra requires trafilatura<1.12, so "
        "that mode moves with datatrove.",
    ),
    "nltk": Watch(
        ("nltk", "metrics"),
        "pyroid.nltk reproduces NLTK's tokenizers; pyroid.metrics' ROUGE stems with NLTK's Porter stemmer, as "
        "rouge-score does.",
    ),
    "rouge-score": Watch(("metrics",), "pyroid.metrics reproduces rouge-score."),
    "sacrebleu": Watch(("metrics",), "pyroid.metrics reproduces sacrebleu."),
    "fasttext-predict": Watch(
        ("fasttext",), "pyroid.fasttext reproduces fastText's predict path (fasttext-predict)."
    ),
    "datatrove": Watch(
        ("filters", "dedup", "warc"),
        "pyroid.filters, pyroid.dedup and pyroid.warc's datatrove level reproduce datatrove. A new release can "
        "also change which trafilatura its processing extra allows (pyroid.trafilatura's 1.11.0 mode).",
    ),
    "anyascii": Watch(
        (),
        "pyroid.translit.anyascii reproduces anyascii. No parity suite covers it here: "
        "python/tests/test_translit_parity.py compares against anyascii 0.3.3 only and skips on other releases.",
    ),
    "nemo-curator": Watch(
        (),
        "pyroid.nemo reproduces NeMo Curator's filters and modifiers. Its parity suite needs the env of ci.yml's "
        "nemo job (NeMo without its dependencies, fasttext built from source), so this workflow does not run it.",
    ),
    # Libraries an oracle depends on whose behaviour pyroid reproduces too.
    "lxml": Watch(_BOTH_MODES, "pyroid.trafilatura reproduces trafilatura on lxml's (libxml2's) parse tree."),
    "justext": Watch(_BOTH_MODES, "pyroid.trafilatura reproduces jusText, trafilatura's fallback extractor."),
    "lxml-html-clean": Watch(("trafilatura@1.11.0",), _T111),
    "charset-normalizer": Watch(("trafilatura@1.11.0",), _T111),
    "courlan": Watch(("trafilatura@1.11.0",), _T111),
    "htmldate": Watch(("trafilatura@1.11.0",), _T111),
    "regex": Watch(
        ("nltk", "metrics"),
        "pyroid.nltk and pyroid.metrics' sacrebleu intl tokenizer reproduce this release's Unicode tables.",
    ),
    "brotlicffi": Watch(
        ("warc",), "warcio decodes Content-Encoding: br only with brotlicffi; pyroid.warc follows."
    ),
    "spacy": Watch(_TOKENIZERS, "pyroid.filters and pyroid.dedup reproduce spaCy's tokenizer (datatrove's)."),
    "sudachipy": Watch(
        _TOKENIZERS, "pyroid ports SudachiPy, datatrove's Japanese tokenizer (jpn_Jpan checks)."
    ),
    "pyvi": Watch(_TOKENIZERS, "pyroid ports pyvi, datatrove's Vietnamese tokenizer (vie_Latn checks)."),
    "pythainlp": Watch(_TOKENIZERS, "pyroid ports PyThaiNLP's newmm and crfcut, datatrove's Thai tokenizer."),
    "python-crfsuite": Watch(_TOKENIZERS, "pyroid ports the CRF tagging PyThaiNLP's crfcut runs on it."),
    "tldextract": Watch(("filters",), "pyroid.filters' URL filter reproduces tldextract's URL splitting."),
}

# Exact pins that are not watched, and why.
NOT_WATCHED: dict[str, str] = {
    "sudachidict-core": "pyroid reads the installed SudachiDict, so pyroid and SudachiPy see the same dictionary",
    "faust-cchardet": "pyroid.warc calls the installed faust-cchardet, as datatrove does",
    "python-magic": "pyroid.warc calls the installed python-magic, as datatrove does",
    "pyahocorasick": "datatrove's URLFilter matches exact substrings with it; pyroid's matches do not depend on it",
    "ruff": "lint tool, pinned together with .pre-commit-config.yaml",
    "mypy": "lint tool, pinned together with .pre-commit-config.yaml",
    "prek": "lint tool, pinned together with .pre-commit-config.yaml",
    "zensical": "docs build",
    "mkdocs-material": "docs build",
    "mkdocstrings-python": "docs build",
}

# Where pins are read: [dependency-groups] of the root pyproject.toml; parity/pyproject.toml's [project]
# dependencies and [dependency-groups]. Pins in a compat group count only for libraries pinned nowhere else
# (trafilatura's 1.11.0 is the compat reference, not the pin to compare with PyPI).
ROOT_PYPROJECT = "pyproject.toml"
PARITY_PYPROJECT = "parity/pyproject.toml"
COMPAT_GROUPS = {(ROOT_PYPROJECT, "test-trafilatura-111"), (PARITY_PYPROJECT, "trafilatura111")}
# Pins kept in Python source: (file, constant, pattern with one group).
SOURCE_PINS: dict[str, tuple[str, str, str]] = {
    "nemo-curator": ("parity/pyroid_parity/nemo_cases.py", "NEMO_VERSION", r'^NEMO_VERSION = "([^"\n]+)"$'),
}

# Suites that read the pinned parity data (the parity-data cache); json_repair's data is in the repository.
SUITES_WITHOUT_DATA = {"json_repair"}
# The suites this workflow runs: entries of ci.yml's parity matrix (SUITE@MODE is a suite in another mode).
KNOWN_SUITES = frozenset(s for w in WATCH.values() for s in w.suites)

# ------------------------------------------------------------------------------------------ untrusted input

_N = r"(?:0|[1-9][0-9]{0,11})"
#: A canonical PEP 440 public version without an epoch: release, optional a/b/rc, .postN, .devN. ASCII only.
VERSION_RE = re.compile(rf"{_N}(?:\.{_N}){{0,5}}(?:(?:a|b|rc){_N})?(?:\.post{_N})?(?:\.dev{_N})?", re.ASCII)
#: A normalized (PEP 503) project name.
NAME_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,98}[a-z0-9])?", re.ASCII)
DATE_RE = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2})T[0-9]{2}:[0-9]{2}:[0-9]{2}", re.ASCII)
#: An https URL without characters that could break out of a Markdown autolink or table cell.
URL_RE = re.compile(
    r"https://[A-Za-z0-9.-]{1,253}(?::[0-9]{1,5})?(?:[/?#][A-Za-z0-9._~%!$&'*+,;=:@/?#-]*)?", re.ASCII
)
GITHUB_REPO_RE = re.compile(
    r"https://github\.com/(?P<owner>[A-Za-z0-9-]{1,39})/(?P<repo>[A-Za-z0-9._-]{1,100}?)(?:\.git)?/?",
    re.ASCII,
)
CHECK_RE = re.compile(r"[A-Za-z0-9_.:@/=+-]{1,120}", re.ASCII)
MAX_URL = 300


class UpstreamError(Exception):
    """PyPI could not be read, or sent something this module does not accept."""


def parse_version(s: object) -> Version | None:
    """``s`` as a Version if it is a canonical version string ``VERSION_RE`` accepts, else None."""
    if not isinstance(s, str) or len(s) > 64 or not VERSION_RE.fullmatch(s):
        return None
    try:
        v = Version(s)
    except InvalidVersion:
        return None
    return v if str(v) == s else None


def checked_version(s: object) -> str:
    """``s`` if it is a stable (not pre-release, not dev) canonical version; raises ValueError otherwise."""
    v = parse_version(s)
    if v is None or v.is_prerelease:
        raise ValueError(f"not a stable canonical version: {s!r}")
    return str(s)


def checked_name(s: object) -> str:
    if not isinstance(s, str) or not NAME_RE.fullmatch(s):
        raise ValueError(f"not a normalized project name: {s!r}")
    return s


def safe_url(s: object) -> str | None:
    return s if isinstance(s, str) and len(s) <= MAX_URL and URL_RE.fullmatch(s) else None


# ------------------------------------------------------------------------------------------------ pins


@dataclass(frozen=True)
class Pin:
    name: str  # normalized
    version: str
    file: str
    group: str

    @property
    def compat(self) -> bool:
        return (self.file, self.group) in COMPAT_GROUPS

    def where(self) -> str:
        return f"{self.file} [{self.group}]"


def _requirement_pins(file: str, group: str, entries: Iterable[object]) -> Iterable[Pin]:
    for entry in entries:
        if not isinstance(entry, str):  # {include-group = ...}
            continue
        try:
            req = Requirement(entry)
        except InvalidRequirement as e:
            raise UpstreamError(f"{file} [{group}]: {entry!r}: {e}") from None
        specs = list(req.specifier)
        if len(specs) == 1 and specs[0].operator == "==" and "*" not in specs[0].version:
            yield Pin(canonicalize_name(req.name), specs[0].version, file, group)


def read_pins(root: Path) -> list[Pin]:
    """Every exact pin of the two pyproject files, plus ``SOURCE_PINS``."""
    pins: list[Pin] = []
    rp = tomllib.loads((root / ROOT_PYPROJECT).read_text("utf-8"))
    for group, entries in rp.get("dependency-groups", {}).items():
        pins += _requirement_pins(ROOT_PYPROJECT, group, entries)
    pp = tomllib.loads((root / PARITY_PYPROJECT).read_text("utf-8"))
    pins += _requirement_pins(PARITY_PYPROJECT, "project", pp.get("project", {}).get("dependencies", []))
    for group, entries in pp.get("dependency-groups", {}).items():
        pins += _requirement_pins(PARITY_PYPROJECT, group, entries)
    for name, (file, constant, pattern) in SOURCE_PINS.items():
        m = re.search(pattern, (root / file).read_text("utf-8"), re.MULTILINE)
        if not m:
            raise UpstreamError(f"{file}: no {constant}")
        pins.append(Pin(name, m.group(1), file, constant))
    return pins


@dataclass(frozen=True)
class PinChoice:
    """The pins of one library that count, and the version compared with PyPI."""

    pins: tuple[Pin, ...]
    version: str  # the newest of them
    disagree: bool  # the files (or groups) pin different versions


def choose_pin(name: str, pins: list[Pin]) -> PinChoice | None:
    mine = [p for p in pins if p.name == name]
    chosen = [p for p in mine if not p.compat] or mine
    if not chosen:
        return None
    versions = {p.version for p in chosen}
    for v in versions:
        checked_version(v)
    newest = max(versions, key=Version)
    return PinChoice(tuple(chosen), newest, len(versions) > 1)


# ------------------------------------------------------------------------------------------------- PyPI

PYPI_JSON = "https://pypi.org/pypi/{name}/json"
USER_AGENT = "pyroid-upstream-watch (+https://github.com/ao/pyroid/blob/main/.github/workflows/upstream.yml)"
MAX_BYTES = 64 * 1024 * 1024
Fetch = Callable[[str], dict[str, Any]]


def fetch_pypi(name: str, attempts: int = 3) -> dict[str, Any]:
    """PyPI's JSON for one project (read-only GET, retried on network errors and 5xx)."""
    url = PYPI_JSON.format(name=checked_name(name))
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    last: Exception | None = None
    for attempt in range(attempts):
        if attempt:
            time.sleep(5 * attempt)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read(MAX_BYTES + 1)
        except urllib.error.HTTPError as e:
            if e.code < 500:
                raise UpstreamError(f"{url}: HTTP {e.code}") from None
            last = e
            continue
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            continue
        if len(body) > MAX_BYTES:
            raise UpstreamError(f"{url}: response over {MAX_BYTES} bytes")
        try:
            data = json.loads(body)
        except ValueError as e:
            raise UpstreamError(f"{url}: not JSON ({e})") from None
        if not isinstance(data, dict):
            raise UpstreamError(f"{url}: not a JSON object")
        return data
    raise UpstreamError(f"{url}: {last}")


@dataclass
class Release:
    """What PyPI says about the releases after a pin."""

    latest: str | None  # the newest stable release with files, whatever the pin
    newer: str | None  # the newest one newer than the pin
    released: str | None  # its upload date (YYYY-MM-DD)
    prerelease: list[str]  # pre-releases and dev releases newer than the pin (ignored)
    yanked: list[str]  # stable releases newer than the pin with no file left that is not yanked (ignored)
    malformed: int  # release keys that are not canonical version strings (ignored)


def _upload_date(files: list[dict[str, Any]]) -> str | None:
    dates = []
    for f in files:
        m = DATE_RE.match(str(f.get("upload_time_iso_8601") or f.get("upload_time") or ""))
        if m:
            dates.append(m.group(1))
    return min(dates) if dates else None


def releases_after(data: dict[str, Any], pinned: str) -> Release:
    """The newest stable release newer than ``pinned`` in PyPI's JSON, skipping pre/dev, yanked and malformed."""
    releases = data.get("releases")
    if not isinstance(releases, dict):
        raise UpstreamError("PyPI's JSON has no 'releases'")
    pin = Version(checked_version(pinned))
    out = Release(None, None, None, [], [], 0)
    stable: list[tuple[Version, str, list[dict[str, Any]]]] = []
    for key, files in releases.items():
        v = parse_version(key)
        if v is None or not isinstance(files, list):
            out.malformed += 1
            continue
        live = [f for f in files if isinstance(f, dict) and f.get("yanked") is not True]
        if v.is_prerelease:  # a/b/rc and .devN
            if v > pin:
                out.prerelease.append(key)
            continue
        if not live:
            if v > pin:
                out.yanked.append(key)
            continue
        stable.append((v, key, live))
    out.prerelease.sort(key=Version)
    out.yanked.sort(key=Version)
    if stable:
        v, key, live = max(stable, key=lambda t: t[0])
        out.latest = key
        if v > pin:
            out.newer, out.released = key, _upload_date(live)
    return out


def changes_url(info: dict[str, Any], name: str) -> str:
    """A changelog link from the project URLs, else the GitHub releases page, else PyPI's release history."""
    urls = info.get("project_urls")
    urls = urls if isinstance(urls, dict) else {}
    ranked: list[tuple[int, str]] = []
    keys = ("changelog", "changes", "releasenotes", "history", "news", "whatsnew", "releases")
    for label, url in urls.items():
        u = safe_url(url)
        if u is None or not isinstance(label, str):
            continue
        flat = re.sub(r"[^a-z]", "", label.lower())
        ranked += [(rank, u) for rank, key in enumerate(keys) if key in flat][:1]
    if ranked:
        return min(ranked)[1]
    for url in [*urls.values(), info.get("home_page")]:
        if isinstance(url, str) and (m := GITHUB_REPO_RE.fullmatch(url)):
            return f"https://github.com/{m['owner']}/{m['repo']}/releases"
    return f"https://pypi.org/project/{name}/#history"


# ----------------------------------------------------------------------------------------------- detect


def unclassified(pins: list[Pin]) -> list[str]:
    """Exact pins that are in neither ``WATCH`` nor ``NOT_WATCHED``."""
    return sorted({p.name for p in pins} - set(WATCH) - set(NOT_WATCHED))


def detect(root: Path, fetch: Fetch = fetch_pypi) -> dict[str, Any]:
    """Look every watched library up on PyPI; the result is ``found.json`` (and the job matrix in it)."""
    pins = read_pins(root)
    libraries = []
    for name, watch in WATCH.items():
        entry: dict[str, Any] = {
            "library": name,
            "suites": list(watch.suites),
            "why": watch.why,
            "pinned": None,
            "pins": [],
            "disagree": False,
            "latest": None,
            "newer": None,
            "released": None,
            "pypi": None,
            "changes": None,
            "prerelease": [],
            "yanked": [],
            "malformed": 0,
            "error": None,
        }
        libraries.append(entry)
        try:
            choice = choose_pin(name, pins)
            if choice is None:
                raise UpstreamError(f"not pinned in {ROOT_PYPROJECT} or {PARITY_PYPROJECT}")
            entry["pinned"] = choice.version
            entry["pins"] = [{"where": p.where(), "version": p.version} for p in choice.pins]
            entry["disagree"] = choice.disagree
            data = fetch(name)
            info = data.get("info")
            if not isinstance(info, dict) or canonicalize_name(str(info.get("name", ""))) != name:
                raise UpstreamError("PyPI's JSON is for another project")
            rel = releases_after(data, choice.version)
        except (UpstreamError, ValueError, OSError) as e:
            entry["error"] = str(e)[:300]
            continue
        entry.update(latest=rel.latest, prerelease=rel.prerelease, yanked=rel.yanked, malformed=rel.malformed)
        if rel.newer:
            entry.update(
                newer=rel.newer,
                released=rel.released,
                pypi=f"https://pypi.org/project/{name}/{rel.newer}/",
                changes=changes_url(info, name),
            )
    matrix = [
        {
            "library": e["library"],
            "pinned": e["pinned"],
            "version": e["newer"],
            "suite": suite,
            "data": suite not in SUITES_WITHOUT_DATA,
        }
        for e in libraries
        if e["newer"]
        for suite in e["suites"]
    ]
    return {
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "unclassified": unclassified(pins),
        "libraries": libraries,
        "matrix": {"include": matrix},
    }


def detect_summary(found: dict[str, Any]) -> str:
    lines = [
        "### Upstream releases",
        "",
        "| library | pinned | newest stable on PyPI | suites | result |",
        "|---|---|---|---|---|",
    ]
    for e in found["libraries"]:
        if e["error"]:
            result = "error: " + re.sub(r"[|\r\n`<>]", " ", e["error"])
        elif e["newer"]:
            result = f"**{e['newer']}** is newer (released {e['released'] or '?'})"
        else:
            result = "up to date"
        if e["disagree"]:
            result += "; the pins disagree: " + ", ".join(f"{p['where']} {p['version']}" for p in e["pins"])
        suites = ", ".join(e["suites"]) or "none"
        lines.append(
            f"| {e['library']} | {e['pinned'] or '?'} | {e['latest'] or '?'} | {suites} | {result} |"
        )
    if found["unclassified"]:
        lines += ["", "Exact pins in neither WATCH nor NOT_WATCHED: " + ", ".join(found["unclassified"])]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------- render

PYTEST_EXIT = {
    0: "all tests passed",
    1: "some tests failed",
    2: "interrupted",
    3: "internal error",
    4: "usage error",
    5: "no tests collected",
}


def _int(x: object) -> int:
    return x if isinstance(x, int) and not isinstance(x, bool) and x >= 0 else 0


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


@dataclass
class PassResult:
    """One pytest pass of a suite: pytest's exit status and the suite's checks from its parity-report.json."""

    exit: int | None
    status: str | None = None  # pass / fail / skip
    checks: dict[str, int] | None = None  # pass / fail / skip counts
    same: int = 0
    diff: int = 0
    failing: list[tuple[str, int, int]] | None = None  # (check, same, diff)

    def cell(self) -> str:
        if self.checks is None:
            if self.exit is None:
                return "not run"
            return f"no report (pytest: {PYTEST_EXIT.get(self.exit, f'exit {self.exit}')})"
        c = self.checks
        text = (
            f"{self.status} (checks: {c['pass']} pass, {c['fail']} fail, {c['skip']} skip; "
            f"{self.same:,} same, {self.diff:,} different)"
        )
        if self.exit not in (0, 1):
            text += f"; pytest: {PYTEST_EXIT.get(self.exit or 0, f'exit {self.exit}')}"
        return text


def read_pass(report: dict[str, Any] | None, suite: str, exit_code: object) -> PassResult:
    code = exit_code if isinstance(exit_code, int) and not isinstance(exit_code, bool) else None
    s = (report or {}).get("suites", {}).get(suite.split("@", 1)[0])
    if not isinstance(s, dict) or not isinstance(s.get("checks"), dict):
        return PassResult(code)
    counts = {"pass": 0, "fail": 0, "skip": 0}
    same = diff = 0
    failing = []
    for name, c in s["checks"].items():
        if not isinstance(c, dict) or c.get("status") not in counts:
            continue
        counts[c["status"]] += 1
        same += _int(c.get("same"))
        diff += _int(c.get("diff"))
        if c["status"] == "fail":
            failing.append(
                (name if CHECK_RE.fullmatch(str(name)) else "?", _int(c.get("same")), _int(c.get("diff")))
            )
    status = "fail" if counts["fail"] else "pass" if counts["pass"] else "skip"
    return PassResult(code, status, counts, same, diff, failing)


def _checked_entry(e: dict[str, Any]) -> dict[str, Any]:
    """Re-check what reaches the issue (found.json comes from another job of the same run)."""
    if checked_name(e["library"]) not in WATCH:
        raise ValueError(f"not watched: {e['library']!r}")
    checked_version(e["pinned"])
    checked_version(e["newer"])
    for suite in e["suites"]:
        if suite not in KNOWN_SUITES:
            raise ValueError(f"unknown suite {suite!r}")
    if e["released"] is not None and not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", str(e["released"])):
        raise ValueError(f"bad date {e['released']!r}")
    for key in ("pypi", "changes"):
        if safe_url(e[key]) is None:
            raise ValueError(f"bad {key} URL {e[key]!r}")
    for p in e["pins"]:
        checked_version(p["version"])
        if not re.fullmatch(r"[A-Za-z0-9_./-]+ \[[A-Za-z0-9_-]+\]", str(p["where"])):
            raise ValueError(f"bad pin location {p['where']!r}")
    for v in [*e["yanked"], *e["prerelease"]]:
        if parse_version(v) is None:
            raise ValueError(f"bad version {v!r}")
    return {**e, "why": WATCH[e["library"]].why}


def title(library: str, version: str) -> str:
    return f"upstream: {library} {version} released"


def render_issue(e: dict[str, Any], results: Path, env: dict[str, str]) -> str:
    """The issue body for one library with a newer release; ``results`` holds the parity jobs' artifacts."""
    lib, old, new = e["library"], e["pinned"], e["newer"]
    server = env.get("GITHUB_SERVER_URL", "https://github.com")
    repo, run_id, sha = (
        env.get("GITHUB_REPOSITORY", ""),
        env.get("GITHUB_RUN_ID", ""),
        env.get("GITHUB_SHA", ""),
    )
    run = (
        f"{server}/{repo}/actions/runs/{run_id}"
        if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) and run_id.isdigit()
        else None
    )
    sha = sha[:12] if re.fullmatch(r"[0-9a-f]{40}", sha) else None
    pins = ", ".join(sorted({p["where"].split(" [")[0] for p in e["pins"]}))
    lines = [
        f"{lib} {new} is on PyPI. pyroid is checked against {old}, the version pinned in {pins}.",
        "",
        "| | |",
        "|---|---|",
        f"| Pinned | {old} |",
        f"| New | {new}, released {e['released'] or 'on an unknown date'} |",
        f"| PyPI | <{e['pypi']}> |",
        f"| Changes | <{e['changes']}> |",
    ]
    if run:
        lines.append(f"| Run | <{run}>" + (f" (commit {sha})" if sha else "") + " |")
    if e["disagree"]:
        lines += [
            "",
            "The pins disagree: " + ", ".join(f"{p['where']} {p['version']}" for p in e["pins"]) + ".",
        ]
    if e["yanked"] or e["prerelease"]:
        skipped = [*(f"{v} (yanked)" for v in e["yanked"]), *(f"{v} (pre-release)" for v in e["prerelease"])]
        lines += ["", "Not considered: " + ", ".join(skipped) + "."]
    lines += ["", f"### Parity against {lib} {new}", "", e["why"], ""]
    if not e["suites"]:
        lines.append("No parity suite runs for this library here.")
    else:
        lines += [
            f"| suite | pyroid against {new} | {new} against the pinned references |",
            "|---|---|---|",
        ]
        details, misses = [], []
        for suite in e["suites"]:
            art = results / f"upstream-parity-{lib}-{suite}"
            status = _read_json(art / "status.json")
            if status is None:
                lines.append(f"| `{suite}` | no result: the parity job did not finish (see the run) | |")
                continue
            if status.get("install") != 0:
                lines.append(f"| `{suite}` | {new} did not install over the pin (see the run) | not run |")
                continue
            if status.get("data") == "miss":
                misses.append(suite)
            impl = read_pass(_read_json(art / "parity-report.json"), suite, status.get("impl"))
            self_ = read_pass(_read_json(art / "parity-self.json"), suite, status.get("self"))
            lines.append(f"| `{suite}` | {impl.cell()} | {self_.cell()} |")
            for label, p in (
                (f"pyroid against {new}", impl),
                (f"{new} against the pinned references", self_),
            ):
                if p.failing:
                    shown = ", ".join(
                        f"`{n}` ({diff:,} of {same + diff:,})" for n, same, diff in p.failing[:12]
                    )
                    more = f" and {len(p.failing) - 12} more" if len(p.failing) > 12 else ""
                    details.append(f"- `{suite}`, {label}: {shown}{more}")
        if details:
            lines += ["", "Failing checks (different of compared):", "", *details]
        lines += [
            "",
            (
                f"The first column runs `--impl pyroid.<suite>` with {lib} {new} installed over the pin: its live "
                f"checks compare pyroid with {new}, its pinned checks compare pyroid with the references of {old}. "
                f"The second column runs the suite with {new} itself as the candidate (no `--impl`): its failures "
                f"are where {new} changes the output of {old} on the pinned inputs. The parity reports are "
                "artifacts of the run."
            ),
        ]
        if misses:
            lines += [
                "",
                (
                    "The parity data cache was not found for this commit (ci.yml's parity-data job builds it), "
                    f"so the checks that read pinned files were skipped in: {', '.join(misses)}."
                ),
            ]
    lines += [
        "",
        "### Porting",
        "",
        f"- [ ] Read the changes from {old} to {new} and port those that change output (module and tests).",
        f"- [ ] Move the pin in {pins}, and in parity/uv.lock (`uv lock --upgrade-package {lib}` in parity/).",
        "- [ ] Re-record or re-pin the references that change (parity/README.md, `python -m pyroid_parity.refs`).",
        (
            "- [ ] Update the version wherever it is named: README.md, docs/ (compatibility page, "
            "docs/data/*.json), the stubs and NOTICE."
        ),
        "",
        "This issue is kept up to date by .github/workflows/upstream.yml while a newer release is out.",
    ]
    return "\n".join(lines) + "\n"


def render(
    found: dict[str, Any], results: Path, out: Path, env: dict[str, str]
) -> list[tuple[str, str, str]]:
    """Write ``out/<library>.md`` and ``out/index.tsv`` (library, title, file); returns the index rows."""
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in found["libraries"]:
        if not e.get("newer") or e.get("error"):
            continue
        e = _checked_entry(e)
        name = f"{e['library']}.md"
        (out / name).write_text(render_issue(e, results, env), "utf-8")
        rows.append((e["library"], title(e["library"], e["newer"]), name))
    (out / "index.tsv").write_text("".join("\t".join(r) + "\n" for r in rows), "utf-8")
    return rows


# ------------------------------------------------------------------------------------------------- CLI


def _command_data(s: str) -> str:
    """Escape a message for a GitHub Actions workflow command (one line, no command of its own)."""
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _append(var: str, text: str) -> None:
    if path := os.environ.get(var):
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def main(argv: list[str] | None = None, fetch: Fetch = fetch_pypi) -> int:
    ap = argparse.ArgumentParser(prog="upstream.py", description="the upstream watch (upstream.yml)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect", help="look the watched libraries up on PyPI")
    d.add_argument("--root", type=Path, default=Path("."))
    d.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("render", help="write the issue bodies from found.json and the parity artifacts")
    r.add_argument("--found", type=Path, required=True)
    r.add_argument("--results", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "detect":
        found = detect(a.root, fetch)
        a.out.write_text(json.dumps(found, indent=1) + "\n", "utf-8")
        summary = detect_summary(found)
        print(summary)
        _append("GITHUB_STEP_SUMMARY", summary)
        errors = [e for e in found["libraries"] if e["error"]]
        for e in errors:
            print(f"::error title=upstream {e['library']}::{_command_data(e['error'])}")
        for e in found["libraries"]:
            if e["disagree"]:
                print(f"::warning title=upstream {e['library']}::the pins disagree, see the job summary")
        if found["unclassified"]:
            print(
                f"::error title=upstream::not classified in upstream.py: {', '.join(found['unclassified'])}"
            )
        newer = [e for e in found["libraries"] if e["newer"]]
        _append(
            "GITHUB_OUTPUT",
            f"matrix={json.dumps(found['matrix'], separators=(',', ':'))}\n"
            f"parity={'true' if found['matrix']['include'] else 'false'}\n"
            f"found={'true' if newer else 'false'}\n"
            f"errors={len(errors) + len(found['unclassified'])}",
        )
        return 0
    found = json.loads(a.found.read_text("utf-8"))
    rows = render(found, a.results, a.out, dict(os.environ))
    for lib, t, name in rows:
        print(f"{lib}: {t}")
        _append(
            "GITHUB_STEP_SUMMARY",
            f"<details><summary>{t}</summary>\n\n{(a.out / name).read_text('utf-8')}\n</details>",
        )
    _append("GITHUB_OUTPUT", f"issues={len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
