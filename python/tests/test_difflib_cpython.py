"""CPython's own ``Lib/test/test_difflib.py`` (with its doctests) against pyroid.difflib.

The test file of the running interpreter's version (``parity/data/cpython-difflib/<version>``)
runs three times per test: against the standard library (the oracle: a test it fails here is
skipped), with ``difflib`` being ``pyroid.difflib``, and against the standard library under
``pyroid.patch("difflib")``. Its ``load_tests`` adds ``doctest.DocTestSuite(difflib)``, so the
docstring examples run against each module too.
"""

from __future__ import annotations

import contextlib
import difflib
import doctest
import functools
import os
import subprocess
import sys
import textwrap
import types
import unittest
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

import pyroid.difflib as pd
from pyroid import FallbackWarning, integrations

DATA = Path(__file__).resolve().parents[2] / "parity" / "data" / "cpython-difflib"
HERE = DATA / f"{sys.version_info[0]}.{sys.version_info[1]}"

pytestmark = pytest.mark.thread_unsafe(reason="HtmlDiff's prefix counter and patching are global")


@contextlib.contextmanager
def _force_color(color: bool) -> Iterator[None]:
    # test.support.force_color, without test.support.
    import _colorize  # type: ignore[import-not-found]  # 3.13+, only the 3.15 tests use it

    names = ("FORCE_COLOR", "NO_COLOR", "PYTHON_COLORS")
    saved_env = {k: os.environ.get(k) for k in names}
    saved = _colorize.can_colorize
    try:
        _colorize.can_colorize = lambda *args, **kwargs: color
        for k in names:
            os.environ.pop(k, None)
        os.environ["FORCE_COLOR" if color else "NO_COLOR"] = "1"
        yield
    finally:
        _colorize.can_colorize = saved
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _support(module_name: str) -> dict[str, types.ModuleType]:
    """A ``test.support`` with what test_difflib uses; ``module_name`` is what ``difflib`` is."""
    test = types.ModuleType("test")
    support = types.ModuleType("test.support")
    helper = types.ModuleType("test.support.import_helper")
    test.__path__ = []
    support.__path__ = []
    test.support = support  # type: ignore[attr-defined]
    support.import_helper = helper  # type: ignore[attr-defined]

    def findfile(name: str, subdir: str | None = None) -> str:
        return str(HERE / name)

    def force_colorized(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with _force_color(True):
                return func(*args, **kwargs)

        return wrapper

    def ensure_lazy_imports(imported: str, blocked: set[str]) -> None:
        name = module_name if imported == "difflib" else imported
        script = textwrap.dedent(f"""
            import sys
            blocked = {sorted(blocked)!r}
            import {name}
            found = [m for m in blocked if m in sys.modules]
            assert not found, found
        """)
        subprocess.run([sys.executable, "-c", script], check=True, timeout=120)

    support.findfile = findfile  # type: ignore[attr-defined]
    support.force_colorized = force_colorized  # type: ignore[attr-defined]
    support.cpython_only = lambda f: f  # type: ignore[attr-defined]
    helper.ensure_lazy_imports = ensure_lazy_imports  # type: ignore[attr-defined]
    return {"test": test, "test.support": support, "test.support.import_helper": helper}


def _load(mode: str, target: types.ModuleType, target_name: str) -> types.ModuleType:
    """The test file as a module whose ``import difflib`` gave `target`."""
    path = HERE / "test_difflib.py"
    module = types.ModuleType(f"_cpython_test_difflib_{mode}")
    module.__file__ = str(path)
    replaced = {"difflib": target, **_support(target_name)}
    saved = {k: sys.modules.get(k) for k in replaced}
    try:
        sys.modules.update(replaced)
        code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
        exec(code, module.__dict__)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return module


def _flatten(suite: unittest.TestSuite) -> Iterator[unittest.TestCase]:
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from _flatten(test)
        else:
            yield test


def _name(module: types.ModuleType, case: unittest.TestCase) -> str:
    if isinstance(case, doctest.DocTestCase):  # "difflib.X" or "pyroid.difflib.X"
        return "doctest:" + case.id().removeprefix("pyroid.").removeprefix("difflib.")
    return case.id().removeprefix(module.__name__ + ".")


def _cases(module: types.ModuleType) -> dict[str, unittest.TestCase]:
    suite = unittest.TestLoader().loadTestsFromModule(module)
    return {_name(module, case): case for case in _flatten(suite)}


if HERE.is_dir():
    STD = _load("std", difflib, "difflib")
    PYROID = _load("pyroid", pd, "pyroid.difflib")
    STD_CASES = _cases(STD)
    PYROID_CASES = _cases(PYROID)
else:  # a copy of python/tests without the repository's parity data
    STD_CASES = PYROID_CASES = {}


def _run(module: types.ModuleType, case: unittest.TestCase) -> unittest.TestResult:
    module.setUpModule()
    result = unittest.TestResult()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FallbackWarning)
        case(result)
    return result


def _check(result: unittest.TestResult, what: str) -> None:
    problems = result.failures + result.errors
    if problems:
        detail = "\n".join(text for _, text in problems)
        pytest.fail(f"{what}:\n{detail}", pytrace=False)
    if result.skipped:
        pytest.skip(result.skipped[0][1])


def _oracle(name: str) -> None:
    if name not in STD_CASES:  # a doctest of the standard library's names that only pyroid lists
        return
    result = _run(STD, STD_CASES[name])
    if result.failures or result.errors or result.skipped:
        pytest.skip(f"the standard library's difflib does not pass {name} here either")


@pytest.mark.skipif(not STD_CASES, reason="needs parity/data/cpython-difflib")
@pytest.mark.parametrize("name", sorted(PYROID_CASES))
def test_pyroid_difflib(name: str) -> None:
    _oracle(name)
    _check(_run(PYROID, PYROID_CASES[name]), f"{name} with difflib = pyroid.difflib")


@pytest.mark.skipif(not STD_CASES, reason="needs parity/data/cpython-difflib")
@pytest.mark.parametrize("name", sorted(STD_CASES))
def test_patched_difflib(name: str) -> None:
    _oracle(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", integrations.PatchWarning)
        with integrations.patching("difflib"):
            assert difflib.SequenceMatcher.ratio is pd.SequenceMatcher.ratio
            result = _run(STD, STD_CASES[name])
    _check(result, f"{name} under pyroid.patch('difflib')")


def test_the_suite_is_complete() -> None:
    if not STD_CASES:
        pytest.skip("needs parity/data/cpython-difflib")
    # Every test, and every docstring example of the standard library's difflib except those of
    # the functions pyroid.difflib re-exports unchanged (restore, IS_LINE_JUNK, ...).
    unchanged = {f"doctest:{name}" for name in ("restore", "IS_CHARACTER_JUNK", "IS_LINE_JUNK")}
    assert set(PYROID_CASES) == set(STD_CASES) - unchanged
