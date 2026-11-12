"""Shared configuration for the pyroid test suite.

The tests run against the *installed* wheel (see test_smoke.py), so this file
must not import anything from the source tree.

Hypothesis profiles: ``default`` (quick, for local runs) and ``ci`` (more
examples, no example database). Pick one with ``HYPOTHESIS_PROFILE=ci``.
"""

import dataclasses
import hashlib
import importlib
import os
import subprocess
import sys
import textwrap
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from hypothesis import settings

# deadline=None: the extension may be a debug build, and the first call in a
# process builds the thread pool; neither says anything about correctness.
settings.register_profile("default", max_examples=60, deadline=None)
settings.register_profile("ci", max_examples=300, deadline=None, database=None, print_blob=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))

# Environment variables that change pyroid's (or Python's) threading. They are
# removed from every child interpreter unless a test sets them explicitly.
_THREADING_ENV = ("PYTHON_GIL", "PYROID_NUM_THREADS", "RAYON_NUM_THREADS")


@pytest.fixture(scope="session")
def pa() -> Any:
    """pyarrow, or skip the test (pyroid itself never needs it)."""
    return pytest.importorskip("pyarrow")


@pytest.fixture(scope="session")
def pl() -> Any:
    """polars, or skip the test (it has no free-threaded wheels yet)."""
    return pytest.importorskip("polars")


@pytest.fixture(scope="session")
def run_python() -> Callable[..., subprocess.CompletedProcess[str]]:
    """Run ``code`` in a fresh, isolated interpreter (``python -I``).

    Returns the completed process; a child that outlives ``timeout`` seconds is
    killed and the test fails with its output, so a deadlock cannot hang pytest.
    ``env`` entries add (str) or remove (None) environment variables.
    """

    def run(
        code: str,
        *,
        env: Mapping[str, str | None] | None = None,
        args: Sequence[str] = (),
        timeout: float = 120.0,
    ) -> subprocess.CompletedProcess[str]:
        child_env = {k: v for k, v in os.environ.items() if k not in _THREADING_ENV}
        for key, value in (env or {}).items():
            if value is None:
                child_env.pop(key, None)
            else:
                child_env[key] = value
        # The child's stdio and our decoding are UTF-8 on every OS: on Windows before 3.15 both
        # would be the ANSI code page (cp1252 on the runners), which cannot print CJK text. -I
        # ignores PYTHONIOENCODING, so this is UTF-8 mode (the default from 3.15).
        cmd = [sys.executable, "-I", "-X", "utf8", *args, "-c", textwrap.dedent(code)]
        try:
            return subprocess.run(
                cmd,
                env=child_env,
                capture_output=True,
                encoding="utf-8",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            pytest.fail(
                f"child interpreter did not finish within {timeout} s (deadlock?)\n"
                f"stdout:\n{exc.stdout!r}\nstderr:\n{exc.stderr!r}"
            )

    return run


# ---- NLTK's punkt_tab models (pyroid.nltk's tests) ------------------------------------------

#: NLTK's ``punkt_tab`` package (``nltk.download('punkt_tab')``). It has no licence, so it is
#: never committed: the tests use the installed copy, or download this one (pinned by sha256).
PUNKT_TAB_URL = (
    "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip"
)
PUNKT_TAB_SHA256 = "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"
PUNKT_TAB_FILES = ("collocations.tab", "sent_starters.txt", "abbrev_types.txt", "ortho_context.tab")


@dataclasses.dataclass(frozen=True)
class PunktTab:
    """Where this session's ``punkt_tab`` models are (the ``punkt_tab`` fixture)."""

    root: Path
    """An ``nltk_data`` directory on ``NLTK_DATA`` and ``nltk.data.path``."""
    zip_path: Path | None
    """``<root>/tokenizers/punkt_tab.zip``, if the models are there as NLTK's zip."""
    dir_path: Path | None
    """``<root>/tokenizers/punkt_tab``, if they are there unpacked."""

    def files(self, lang: str = "english") -> dict[str, bytes]:
        """The four files of `lang`'s model, to build other layouts in a test's tmp dir."""
        if self.dir_path is not None and (self.dir_path / lang).is_dir():
            return {n: (self.dir_path / lang / n).read_bytes() for n in PUNKT_TAB_FILES}
        if self.zip_path is None:
            pytest.skip(f"no punkt_tab model files for {lang!r} under {self.root}")
        with zipfile.ZipFile(self.zip_path) as zf:
            return {n: zf.read(f"punkt_tab/{lang}/{n}") for n in PUNKT_TAB_FILES}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _nltk_data_roots() -> list[Path]:
    """``nltk.data.path`` if NLTK imports, else NLTK's default search path."""
    try:
        nltk_data: Any = importlib.import_module("nltk.data")
    except ImportError:
        env = os.environ.get("NLTK_DATA", "").split(os.pathsep)
        prefix = Path(sys.prefix)
        return [
            *(Path(os.path.expanduser(d)) for d in env if d),
            Path(os.path.expanduser("~/nltk_data")),
            *(prefix / sub for sub in ("nltk_data", "share/nltk_data", "lib/nltk_data")),
            *(Path(d, "nltk_data") for d in ("/usr/share", "/usr/local/share", "/usr/lib")),
            Path("/usr/local/lib/nltk_data"),
        ]
    return [Path(p) for p in nltk_data.path]


def _punkt_layout(root: Path) -> PunktTab | None:
    zip_path = root / "tokenizers" / "punkt_tab.zip"
    dir_path = root / "tokenizers" / "punkt_tab"
    z = zip_path if zip_path.is_file() else None
    d = dir_path if (dir_path / "english").is_dir() else None
    return PunktTab(root, z, d) if z is not None or d is not None else None


def _nltk_finds_punkt_tab() -> bool | None:
    """Whether ``nltk.data.find`` finds the English model (None: NLTK is not installed)."""
    try:
        nltk_data: Any = importlib.import_module("nltk.data")
    except ImportError:
        return None
    try:
        nltk_data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        return False
    return True


def _punkt_cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Where to download punkt_tab (the first writable of these).

    ``$PYROID_NLTK_DATA`` (CI caches it), ``$PYROID_DATA/nltk_data``, else a session tmp dir.
    """
    nltk_data = os.environ.get("PYROID_NLTK_DATA")
    data = os.environ.get("PYROID_DATA")
    if nltk_data or data:
        d = Path(nltk_data) if nltk_data else Path(str(data)) / "nltk_data"
        try:
            (d / "tokenizers").mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        else:
            if os.access(d / "tokenizers", os.W_OK):
                return d
    return tmp_path_factory.mktemp("nltk_data")


def _download_punkt_tab(dest: Path) -> None:
    """Download NLTK's punkt_tab.zip to `dest` and check its sha256 (skip if offline)."""
    import urllib.request

    part = dest.with_name(f"{dest.name}.{os.getpid()}.part")
    try:
        with urllib.request.urlopen(PUNKT_TAB_URL, timeout=60) as r, part.open("wb") as f:
            while block := r.read(1 << 20):
                f.write(block)
    except OSError as exc:  # URLError, timeouts, DNS: offline
        part.unlink(missing_ok=True)
        pytest.skip(
            f"NLTK's punkt_tab models are not installed and {PUNKT_TAB_URL} could not be "
            f"downloaded ({type(exc).__name__}: {exc}); install them with "
            "`python -m nltk.downloader punkt_tab`"
        )
    digest = _sha256(part)
    if digest != PUNKT_TAB_SHA256:
        part.unlink()
        pytest.fail(f"{PUNKT_TAB_URL} has sha256 {digest}, not the pinned {PUNKT_TAB_SHA256}")
    os.replace(part, dest)


def _expose_nltk_data(root: Path) -> None:
    """Put `root` first on ``NLTK_DATA`` (child processes, pyroid without NLTK) and
    ``nltk.data.path``, and drop the tokenizers already loaded from elsewhere."""
    env = [p for p in os.environ.get("NLTK_DATA", "").split(os.pathsep) if p]
    if str(root) not in env:
        os.environ["NLTK_DATA"] = os.pathsep.join([str(root), *env])
    try:
        nltk_data: Any = importlib.import_module("nltk.data")
    except ImportError:
        pass
    else:
        if str(root) not in nltk_data.path:
            nltk_data.path.insert(0, str(root))
    for name in ("pyroid.nltk", "nltk.tokenize"):
        cached = getattr(sys.modules.get(name), "_get_punkt_tokenizer", None)
        if cached is not None:
            cached.cache_clear()


@pytest.fixture(scope="session")
def punkt_tab(tmp_path_factory: pytest.TempPathFactory) -> PunktTab:
    """NLTK's ``punkt_tab`` models, visible to NLTK and to pyroid.nltk; skip if unavailable.

    The installed models if ``nltk.data.find("tokenizers/punkt_tab/english/")`` finds them
    (without NLTK: if NLTK's default search path has them). Otherwise punkt_tab.zip is
    downloaded once into ``$PYROID_NLTK_DATA``, ``$PYROID_DATA/nltk_data`` or a session tmp dir,
    its sha256
    checked, and that directory put first on ``NLTK_DATA`` and ``nltk.data.path``. Skips if
    the models are not installed and the download fails (offline).
    """
    found = _nltk_finds_punkt_tab()
    if found is not False:
        for root in _nltk_data_roots():
            layout = _punkt_layout(root)
            if layout is not None:
                _expose_nltk_data(root)
                return layout
    root = _punkt_cache_dir(tmp_path_factory)
    dest = root / "tokenizers" / "punkt_tab.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file() or _sha256(dest) != PUNKT_TAB_SHA256:
        _download_punkt_tab(dest)
    _expose_nltk_data(root)
    if _nltk_finds_punkt_tab() is False:
        pytest.fail(f"NLTK does not find the punkt_tab models in {dest}")
    return PunktTab(root, dest, None)


def pytest_terminal_summary(terminalreporter: Any, config: pytest.Config) -> None:
    """Print the largest pyroid-vs-fastText |dp| that test_fasttext.py's live-oracle tests saw."""
    deltas: dict[tuple[str, int], float] | None = getattr(
        config, "_pyroid_fasttext_oracle_deltas", None
    )
    if not deltas:
        return
    worst = sorted(deltas.items(), key=lambda kv: kv[1], reverse=True)
    shown = ", ".join(f"{name} k={k}: {dp:.3g}" for (name, k), dp in worst[:5] if dp > 0)
    terminalreporter.write_line(
        f"fastText live oracle on {sys.platform}: max |dp| {worst[0][1]:.3g} over "
        f"{len(deltas)} (model, k) pairs" + (f" ({shown})" if shown else "")
    )
