"""Shared configuration for the pyroid test suite.

The tests run against the *installed* wheel (see test_smoke.py), so this file
must not import anything from the source tree.

Hypothesis profiles: ``default`` (quick, for local runs) and ``ci`` (more
examples, no example database). Pick one with ``HYPOTHESIS_PROFILE=ci``.
"""

import os
import subprocess
import sys
import textwrap
from collections.abc import Callable, Mapping, Sequence
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
        cmd = [sys.executable, "-I", *args, "-c", textwrap.dedent(code)]
        try:
            return subprocess.run(
                cmd, env=child_env, capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired as exc:
            pytest.fail(
                f"child interpreter did not finish within {timeout} s (deadlock?)\n"
                f"stdout:\n{exc.stdout!r}\nstderr:\n{exc.stderr!r}"
            )

    return run


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
