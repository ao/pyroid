"""Smoke tests for the installed pyroid wheel.

Run these against an installed wheel from a directory outside the source tree,
so that ``import pyroid`` cannot pick up ``python/pyroid`` from the checkout.
"""

import importlib.metadata
import json
import os
import pickle
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest
from packaging.version import Version

import pyroid
from pyroid import _pyroid

FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def _editable_install() -> bool:
    """True for `maturin develop` / `pip install -e` installs (PEP 610 direct_url.json)."""
    raw = importlib.metadata.distribution("pyroid").read_text("direct_url.json")
    if not raw:
        return False
    return bool(json.loads(raw).get("dir_info", {}).get("editable", False))


def test_imports_installed_extension_not_source_tree() -> None:
    if _editable_install():
        pytest.skip("editable install: the extension lives in the source tree")
    ext = Path(_pyroid.__file__ or "")
    assert ext.suffix in {".so", ".pyd"}, ext
    assert "site-packages" in ext.parts, f"not the installed wheel: {ext}"


def test_version_matches_distribution_metadata() -> None:
    assert pyroid.__version__ == _pyroid.__version__
    # Cargo keeps SemVer (1.0.0-dev); the wheel metadata is PEP 440 (1.0.0.dev0).
    assert Version(pyroid.__version__) == Version(importlib.metadata.version("pyroid"))


def test_public_api() -> None:
    assert pyroid.__all__ == [
        "Column",
        "FallbackWarning",
        "InputError",
        "PyroidError",
        "__version__",
    ]
    assert pyroid.PyroidError is _pyroid.PyroidError
    assert pyroid.FallbackWarning is _pyroid.FallbackWarning
    assert pyroid.InputError is _pyroid.InputError
    assert pyroid.Column is _pyroid.Column


def test_exception_hierarchy() -> None:
    err = pyroid.PyroidError
    assert issubclass(err, Exception)
    assert err.__module__ == "pyroid"
    assert err.__qualname__ == "PyroidError"
    assert err.__doc__ == "Base class for all pyroid errors."
    with pytest.raises(pyroid.PyroidError, match="boom"):
        raise err("boom")


def test_fallback_warning() -> None:
    # The one warning every module emits when it delegates to the incumbent (docs/dev/MODULES.md).
    w = pyroid.FallbackWarning
    assert w.__mro__[1:] == RuntimeWarning.__mro__
    assert not issubclass(w, pyroid.PyroidError)
    assert w.__module__ == "pyroid"
    assert w.__qualname__ == "FallbackWarning"
    clone = pickle.loads(pickle.dumps(w("slow path")))
    assert type(clone) is w
    assert clone.args == ("slow path",)


def test_exception_pickles_by_public_name() -> None:
    clone = pickle.loads(pickle.dumps(pyroid.PyroidError("x")))
    assert type(clone) is pyroid.PyroidError
    assert clone.args == ("x",)


@pytest.mark.skipif(not FREE_THREADED, reason="needs a free-threaded CPython build")
@pytest.mark.thread_unsafe(reason="runs a subprocess")
def test_import_keeps_gil_disabled() -> None:
    # Fresh interpreter, PYTHON_GIL unset: if the module did not declare
    # Py_MOD_GIL_NOT_USED, importing it would re-enable the GIL.
    env = {k: v for k, v in os.environ.items() if k != "PYTHON_GIL"}
    code = "import sys, pyroid; print(sys._is_gil_enabled())"
    out = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False", out.stderr
