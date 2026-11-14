"""pyroid.VersionMismatchWarning: the installed incumbent is not the release pyroid reproduces.

Every case runs in a fresh interpreter (``-I -S``) that sees no installed package except pyroid
and numpy: the incumbents it finds are stand-ins in a temporary directory, a
``<dist>-<version>.dist-info/METADATA`` and a top-level module that raises if it is imported.
So each case controls exactly which release is "installed", and shows that the check never
imports it. The Thai and Japanese checks, which need the real data files, are in
test_filters_vithai.py and test_filters_japanese.py.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import subprocess
import sys
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import pyroid

pytestmark = pytest.mark.thread_unsafe(reason="subprocesses")

Run = Callable[..., subprocess.CompletedProcess[str]]
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
SILENCE = 'warnings.simplefilter("ignore", pyroid.VersionMismatchWarning)'


def _version(module: str, attr: str) -> Callable[[], str]:
    """The version `module` reproduces, read from pyroid when a test needs it."""

    def get() -> str:
        return str(getattr(importlib.import_module(module), attr))

    return get


@dataclass(frozen=True)
class Case:
    id: str
    statement: str  # the user's line that triggers the check
    library: str  # the incumbent's top-level module
    dist: str  # its distribution
    version: Callable[[], str]  # the release pyroid reproduces
    subject: str  # the pyroid module the message names (the first to check)
    checks: tuple[str, ...]  # every library the statement checks
    patch: str | None = None  # the pyroid.patch() name, if there is one
    numpy: bool = False  # the pyroid module needs numpy


def _fixed(version: str) -> Callable[[], str]:
    return lambda: version


CASES = [
    Case(
        "ftfy",
        "import pyroid.ftfy",
        "ftfy",
        "ftfy",
        _version("pyroid.ftfy", "FTFY_VERSION"),
        "pyroid.ftfy",
        ("ftfy",),
        patch="ftfy",
    ),
    Case(
        "fasttext",
        "import pyroid.fasttext",
        "fasttext",
        "fasttext-predict",
        _fixed("0.9.2.4"),
        "pyroid.fasttext",
        ("fasttext",),
        patch="fasttext",
        numpy=True,
    ),
    Case(
        "anyascii",
        "import pyroid.translit",
        "anyascii",
        "anyascii",
        _version("pyroid.translit", "ANYASCII_VERSION"),
        "pyroid.translit",
        ("anyascii",),
        patch="anyascii",
    ),
    Case(
        "datasketch",
        "import pyroid.datasketch",
        "datasketch",
        "datasketch",
        _version("pyroid.datasketch", "DATASKETCH_VERSION"),
        "pyroid.datasketch",
        ("datasketch",),
        patch="datasketch",
        numpy=True,
    ),
    Case(
        "trafilatura",
        "import pyroid.trafilatura",
        "trafilatura",
        "trafilatura",
        _version("pyroid.trafilatura", "TRAFILATURA_VERSION"),
        "pyroid.trafilatura",
        ("trafilatura",),
        patch="trafilatura",
    ),
    Case(
        "nltk",
        "import pyroid.nltk",
        "nltk",
        "nltk",
        _version("pyroid.nltk", "NLTK_VERSION"),
        "pyroid.nltk",
        ("nltk",),
        patch="nltk",
    ),
    Case(
        "warcio",
        "import pyroid.warc",
        "warcio",
        "warcio",
        _version("pyroid.warc", "WARCIO_VERSION"),
        "pyroid.warc",
        ("warcio", "datatrove"),
        patch="warcio",
    ),
    Case(
        "datatrove-warc",
        "import pyroid.warc",
        "datatrove",
        "datatrove",
        _version("pyroid.filters", "DATATROVE_VERSION"),
        "pyroid.warc",
        ("warcio", "datatrove"),
    ),
    Case(
        "datatrove-filters",
        "import pyroid.filters",
        "datatrove",
        "datatrove",
        _version("pyroid.filters", "DATATROVE_VERSION"),
        "pyroid.filters",
        ("datatrove",),
    ),
    Case(
        "datatrove-dedup",
        "import pyroid.dedup",
        "datatrove",
        "datatrove",
        _version("pyroid.dedup", "DATATROVE_VERSION"),
        "pyroid.filters",  # pyroid.dedup imports pyroid.filters first
        ("datatrove",),
    ),
    Case(
        "jieba",
        "import pyroid.jieba",
        "jieba",
        "jieba",
        _version("pyroid.jieba", "JIEBA_VERSION"),
        "pyroid.jieba",
        ("jieba",),
    ),
    Case(
        "json_repair",
        "import pyroid.json_repair",
        "json_repair",
        "json-repair",
        _version("pyroid.json_repair", "JSON_REPAIR_VERSION"),
        "pyroid.json_repair",
        ("json_repair",),
    ),
    Case(
        "rouge-score",
        "from pyroid.metrics import rouge_scorer",
        "rouge_score",
        "rouge-score",
        _version("pyroid.metrics.rouge_scorer", "ROUGE_SCORE_VERSION"),
        "pyroid.metrics.scoring",  # rouge_scorer imports scoring first
        ("rouge_score", "nltk"),
    ),
    Case(
        "nltk-rouge",
        "from pyroid.metrics import rouge_scorer",
        "nltk",
        "nltk",
        _version("pyroid.nltk", "NLTK_VERSION"),
        "pyroid.metrics.tokenizers",  # the Porter stemmer of DefaultTokenizer(use_stemmer=True)
        ("rouge_score", "nltk"),
    ),
    Case(
        "sacrebleu",
        "import pyroid.metrics.sacrebleu",
        "sacrebleu",
        "sacrebleu",
        _version("pyroid.metrics.sacrebleu", "__version__"),
        "pyroid.metrics.sacrebleu",
        ("sacrebleu",),
    ),
    Case(
        "nemo-curator",
        "import pyroid.nemo",
        "nemo_curator",
        "nemo-curator",
        _version("pyroid.nemo", "NEMO_CURATOR_VERSION"),
        "pyroid.nemo",
        ("nemo_curator",),
    ),
]
PATCHED = [c for c in CASES if c.patch is not None]


def _params(cases: Sequence[Case]) -> list[Any]:
    marks = pytest.mark.skipif(not HAS_NUMPY, reason="needs numpy")
    return [pytest.param(c, id=c.id, marks=marks if c.numpy else ()) for c in cases]


def message(case: Case, found: str, *, dist: str | None = None, todo: str | None = None) -> str:
    """The warning's text when `found` of `dist` (default: the case's) is installed."""
    version = case.version()
    todo = todo or f"Install {case.dist}=={version} to compare like for like"
    return (
        f"{case.subject} reproduces {case.dist} {version}, but {dist or case.dist} {found} is "
        f"installed: results can differ from it. {todo} (or silence this with {SILENCE})."
    )


def stand_in(root: Path, library: str, dist: str | None, version: str | None) -> None:
    """A stand-in `library` under `root`: a module that raises if imported, and its metadata.

    ``dist=None`` leaves the metadata out (a source checkout); ``version=None`` the module.
    """
    if version is not None or dist is None:
        package = root / library
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text(
            f"raise RuntimeError('the stand-in {library} was imported')\n", encoding="utf-8"
        )
    if dist is not None and version is not None:
        info = root / f"{dist.replace('-', '_')}-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n", encoding="utf-8"
        )


def _site(module: str) -> str | None:
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.submodule_search_locations:
        return None
    return os.path.dirname(next(iter(spec.submodule_search_locations)))


# Only pyroid and numpy come from the real environment; sys.path is the stdlib (-S) and `root`.
PRELUDE = """\
import importlib.machinery, json, sys, warnings
sys.path.insert(0, {root!r})
class _Only:
    @staticmethod
    def find_spec(name, path=None, target=None):
        where = {sites!r}.get(name)
        if path is None and where:
            return importlib.machinery.PathFinder.find_spec(name, [where])
        return None
sys.meta_path.insert(0, _Only())
"""

REPORT = """
print(json.dumps({{
    "warnings": [
        [w.category.__name__, str(w.message), w.filename, w.lineno]
        for w in caught
        if w.category.__name__ in ("VersionMismatchWarning", "PatchWarning")
    ],
    "imported": sorted(m for m in {libraries!r} if m in sys.modules),
    "metadata": "importlib.metadata" in sys.modules,
}}))
"""


@dataclass
class Child:
    """What the child reported, and the line number of each statement it ran."""

    warnings: list[list[Any]]
    imported: list[str]
    metadata: bool
    lines: dict[str, int]

    def of(self, category: str) -> list[list[Any]]:
        return [w for w in self.warnings if w[0] == category]


def run_child(
    run_python: Run,
    root: Path,
    statements: Sequence[str],
    libraries: Sequence[str],
    *,
    count_frames: bool = False,
    setup: Sequence[str] = (),
) -> Child:
    """Run `statements` (recording warnings) after `setup`, with `root`'s stand-ins installed."""
    sites = {name: site for name in ("pyroid", "numpy") if (site := _site(name)) is not None}
    lines = [PRELUDE.format(root=str(root), sites=sites)]
    if count_frames:  # count frames for stacklevel, as on CPython 3.11
        lines.append("import pyroid._versions as _v; _v._COUNT_FRAMES = True\n")
    lines.extend(f"{s}\n" for s in setup)
    lines.append("with warnings.catch_warnings(record=True) as caught:\n")
    lines.append('    warnings.simplefilter("always")\n')
    lines.extend(f"    {s}\n" for s in statements)
    lines.append(REPORT.format(libraries=list(libraries)))
    code = "".join(lines)
    assert textwrap.dedent(code) == code  # run_python dedents: the line numbers stay
    numbers = {s: code.splitlines().index(f"    {s}") + 1 for s in statements}
    proc = run_python(code, args=("-S",), timeout=180)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    return Child(out["warnings"], out["imported"], out["metadata"], numbers)


# ---------------------------------------------------------------------------
# The class
# ---------------------------------------------------------------------------


def test_warning_class() -> None:
    w = pyroid.VersionMismatchWarning
    assert issubclass(w, UserWarning)
    assert not issubclass(w, (RuntimeWarning, pyroid.FallbackWarning, pyroid.PyroidError))
    assert w.__module__ == "pyroid"
    assert w.__qualname__ == "VersionMismatchWarning"
    assert "VersionMismatchWarning" in pyroid.__all__
    assert w is pyroid._pyroid.VersionMismatchWarning
    clone = pickle.loads(pickle.dumps(w("ftfy 6.2.0")))
    assert type(clone) is w
    assert clone.args == ("ftfy 6.2.0",)


# ---------------------------------------------------------------------------
# Every library
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _params(CASES))
def test_another_release_warns_once_at_the_import(
    run_python: Run, tmp_path: Path, case: Case
) -> None:
    stand_in(tmp_path, case.library, case.dist, "0.0.1")
    child = run_child(run_python, tmp_path, [case.statement], case.checks)
    assert child.of("VersionMismatchWarning") == [
        ["VersionMismatchWarning", message(case, "0.0.1"), "<string>", child.lines[case.statement]]
    ]
    assert child.imported == []  # the stand-in was never imported


@pytest.mark.parametrize("case", _params(CASES))
@pytest.mark.parametrize("state", ["same release", "no metadata", "absent"])
def test_silent(run_python: Run, tmp_path: Path, case: Case, state: str) -> None:
    if state == "same release":
        stand_in(tmp_path, case.library, case.dist, case.version())
    elif state == "no metadata":  # importable, not installed: a source checkout
        stand_in(tmp_path, case.library, None, None)
    child = run_child(run_python, tmp_path, [case.statement], case.checks)
    assert child.warnings == []
    assert child.imported == []
    if state == "absent":  # no incumbent: not even the metadata machinery is loaded
        assert child.metadata is False


@pytest.mark.parametrize("case", _params(PATCHED))
def test_patch_warns_at_its_line_and_shares_the_check(
    run_python: Run, tmp_path: Path, case: Case
) -> None:
    stand_in(tmp_path, case.library, case.dist, "0.0.1")
    patch = f"pyroid.patch({case.patch!r})"
    statements = ["import pyroid", patch, case.statement, f"pyroid.unpatch(); {patch}"]
    child = run_child(run_python, tmp_path, statements, ())
    assert child.of("VersionMismatchWarning") == [
        ["VersionMismatchWarning", message(case, "0.0.1"), "<string>", child.lines[patch]]
    ]
    assert [w[0] for w in child.warnings] == ["VersionMismatchWarning", "PatchWarning"]
    # The other way round: the import warns, patch() does not warn again.
    child = run_child(run_python, tmp_path, [case.statement, "import pyroid", patch], ())
    assert child.of("VersionMismatchWarning") == [
        ["VersionMismatchWarning", message(case, "0.0.1"), "<string>", child.lines[case.statement]]
    ]


@pytest.mark.parametrize(
    "statement",
    [
        "import pyroid.ftfy",
        "from pyroid import ftfy",
        "importlib.import_module('pyroid.ftfy')",
        "pyroid.patch('ftfy')",
        "pyroid.integrations.patch('ftfy')",
        "pyroid.integrations._patch.patching('ftfy').__enter__()",
    ],
)
@pytest.mark.parametrize("count_frames", [False, True], ids=["skip_file_prefixes", "stacklevel"])
def test_points_at_the_callers_line(
    run_python: Run, tmp_path: Path, statement: str, count_frames: bool
) -> None:
    """The first frame outside pyroid (and importlib), with either way of finding it.

    ``stacklevel`` is what CPython 3.11 uses (it has no ``skip_file_prefixes``); it counts the
    frames as ``warnings.warn`` does, passing over importlib's bootstrap frames.
    """
    case = CASES[0]
    stand_in(tmp_path, "ftfy", "ftfy", "6.2.0")
    setup = ["import importlib, pyroid, pyroid.integrations, pyroid.integrations._patch"]
    child = run_child(run_python, tmp_path, [statement], (), count_frames=count_frames, setup=setup)
    assert child.of("VersionMismatchWarning") == [
        ["VersionMismatchWarning", message(case, "6.2.0"), "<string>", child.lines[statement]]
    ]


def test_silenced_by_its_class(run_python: Run, tmp_path: Path) -> None:
    stand_in(tmp_path, "ftfy", "ftfy", "6.2.0")
    statements = [
        f"import pyroid; {SILENCE}",
        "import pyroid.ftfy",
        "pyroid.patch('ftfy')",
    ]
    child = run_child(run_python, tmp_path, statements, ("ftfy",))
    assert [w[0] for w in child.warnings] == ["PatchWarning"]


def test_one_check_per_library(run_python: Run, tmp_path: Path) -> None:
    """Every module that reproduces datatrove shares one check; nltk is checked on its own."""
    stand_in(tmp_path, "datatrove", "datatrove", "0.11.0")
    stand_in(tmp_path, "nltk", "nltk", "3.9.0")
    statements = [
        "import pyroid.warc",
        "import pyroid.dedup",
        "import pyroid.filters",
        "import pyroid.nltk",
        "from pyroid.metrics import rouge_scorer",
    ]
    child = run_child(run_python, tmp_path, statements, ("datatrove", "nltk"))
    got = [(w[1].split(" reproduces ")[0], w[3]) for w in child.of("VersionMismatchWarning")]
    assert got == [
        ("pyroid.warc", child.lines["import pyroid.warc"]),
        ("pyroid.nltk", child.lines["import pyroid.nltk"]),
    ]
    assert child.imported == []


# ---------------------------------------------------------------------------
# Distributions named differently from their module, and other builds
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_NUMPY, reason="pyroid.fasttext needs numpy")
@pytest.mark.parametrize(
    ("dist", "version", "warned"),
    [
        ("fasttext-predict", "0.9.2.4", False),
        ("fasttext", "0.9.3", False),  # NeMo Curator's fastText: the same results
        ("fasttext-predict", "0.9.2.3", True),
        ("fasttext", "0.9.2", True),
        ("fasttext-wheel", "0.9.2", True),
        ("fasttext-numpy2-wheel", "0.9.2.1", True),
    ],
)
def test_fasttext_distributions(
    run_python: Run, tmp_path: Path, dist: str, version: str, warned: bool
) -> None:
    case = CASES[1]
    stand_in(tmp_path, "fasttext", dist, version)
    child = run_child(run_python, tmp_path, ["import pyroid.fasttext"], ("fasttext",))
    want = [message(case, version, dist=dist)] if warned else []
    assert [w[1] for w in child.of("VersionMismatchWarning")] == want
    assert child.imported == []


@pytest.mark.parametrize(
    ("library", "dist", "statement", "subject"),
    [
        ("rouge_score", "rouge-score", "import pyroid.metrics.tokenize", "pyroid.metrics.tokenize"),
        ("nemo_curator", "nemo-curator", "import pyroid.nemo", "pyroid.nemo"),
        ("json_repair", "json-repair", "import pyroid.json_repair", "pyroid.json_repair"),
    ],
)
def test_normalized_distribution_names(
    run_python: Run, tmp_path: Path, library: str, dist: str, statement: str, subject: str
) -> None:
    """A dist-info named ``rouge_score-...`` (as wheels name it) is ``rouge-score``'s."""
    stand_in(tmp_path, library, dist, "0.0.1")
    assert (tmp_path / f"{library}-0.0.1.dist-info").is_dir()
    child = run_child(run_python, tmp_path, [statement], (library,))
    got = child.of("VersionMismatchWarning")
    assert len(got) == 1, child.warnings
    assert got[0][1].startswith(f"{subject} reproduces {dist} ")
    assert f", but {dist} 0.0.1 is installed: " in got[0][1]


# ---------------------------------------------------------------------------
# trafilatura: the releases compat= reproduces
# ---------------------------------------------------------------------------


def _other_trafilatura_mode() -> str:
    import pyroid.trafilatura as pt

    return next(v for v in pt.TRAFILATURA_VERSIONS if v != pt.TRAFILATURA_VERSION)


def test_trafilatura_suggests_compat(run_python: Run, tmp_path: Path) -> None:
    mode = _other_trafilatura_mode()
    stand_in(tmp_path, "trafilatura", "trafilatura", mode)
    child = run_child(run_python, tmp_path, ["import pyroid.trafilatura"], ("trafilatura",))
    want = message(CASES[4], mode, todo=f'Pass compat="{mode}" to reproduce it')
    assert [w[1] for w in child.of("VersionMismatchWarning")] == [want]


@pytest.mark.parametrize(
    "statement",
    [
        "pyroid.patch('trafilatura')",
        'exec(\'with pyroid._versions.follow_installed("trafilatura"): '
        "import pyroid.trafilatura')",
    ],
    ids=["patch", "follow_installed"],
)
def test_trafilatura_mode_chosen_for_the_user(
    run_python: Run, tmp_path: Path, statement: str
) -> None:
    """patch() (and datatrove's step, through follow_installed) reproduce the installed mode."""
    stand_in(tmp_path, "trafilatura", "trafilatura", _other_trafilatura_mode())
    setup = ["import pyroid, pyroid._versions"]
    child = run_child(
        run_python, tmp_path, [statement, "import pyroid.trafilatura"], (), setup=setup
    )
    assert child.of("VersionMismatchWarning") == []
    # A release pyroid does not reproduce still warns there.
    root = tmp_path / "other"
    stand_in(root, "trafilatura", "trafilatura", "1.10.0")
    child = run_child(run_python, root, [statement], (), setup=setup)
    assert [w[1] for w in child.of("VersionMismatchWarning")] == [message(CASES[4], "1.10.0")]


# ---------------------------------------------------------------------------
# Threads, and process trees
# ---------------------------------------------------------------------------

THREADS = """
import sys, threading
sys.path.insert(0, {root!r})
import pyroid._versions as v
barrier = threading.Barrier(16)
def work():
    barrier.wait()
    v.check("pyroid.ftfy", "ftfy", "ftfy==6.3.1")
threads = [threading.Thread(target=work) for _ in range(16)]
for t in threads:
    t.start()
for t in threads:
    t.join()
"""


def test_once_from_many_threads(run_python: Run, tmp_path: Path) -> None:
    stand_in(tmp_path, "ftfy", "ftfy", "6.2.0")
    # The stand-in metadata is found first, before any installed ftfy's.
    proc = run_python(THREADS.format(root=str(tmp_path)), args=("-W", "always"))
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr.count("VersionMismatchWarning: pyroid.ftfy reproduces ftfy 6.3.1") == 1


SCRIPT = """\
import subprocess, sys
import ftfy  # the shim (the stand-in itself raises)
print("parent", ftfy.__pyroid__, flush=True)
child = "import ftfy; print('child', ftfy.__pyroid__)"
subprocess.run([sys.executable, "-c", child], check=True)
"""


def test_pyroid_run_warns_once_per_process_tree(tmp_path: Path) -> None:
    """The start-up hook patches the child, quietly: the parent already warned."""
    stand_in(tmp_path / "site", "ftfy", "ftfy", "6.2.0")
    script = tmp_path / "script.py"
    script.write_text(SCRIPT, encoding="utf-8")
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONPATH", "PYROID_PATCH", "_PYROID_PATCH_WARNED", "PYTHONWARNINGS")
    }
    env["PYTHONPATH"] = str(tmp_path / "site")
    run = "import sys; from pyroid.integrations._patch import run; sys.exit(run(sys.argv[1:]))"
    proc = subprocess.run(
        [sys.executable, "-c", run, "-p", "ftfy", str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["parent", "True", "child", "True"]
    assert proc.stderr.count("VersionMismatchWarning: ") == 1, proc.stderr
    assert "pyroid.ftfy reproduces ftfy 6.3.1, but ftfy 6.2.0 is installed" in proc.stderr
