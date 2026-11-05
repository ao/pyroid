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
    assert pyroid.__all__ == ["PyroidError", "__version__"]
    assert pyroid.PyroidError is _pyroid.PyroidError


def test_exception_hierarchy() -> None:
    err = pyroid.PyroidError
    assert issubclass(err, Exception)
    assert err.__module__ == "pyroid"
    assert err.__qualname__ == "PyroidError"
    assert err.__doc__ == "Base class for all pyroid errors."
    with pytest.raises(pyroid.PyroidError, match="boom"):
        raise err("boom")


def test_exception_pickles_by_public_name() -> None:
    clone = pickle.loads(pickle.dumps(pyroid.PyroidError("x")))
    assert type(clone) is pyroid.PyroidError
    assert clone.args == ("x",)


@pytest.mark.parametrize(
    "text",
    ["", "pyroid", "naïve café", "€", "😀", "a\x00b", "\r\n\u2028\x85", "日本語テキスト" * 100],
)
def test_utf8_len_matches_python(text: str) -> None:
    assert _pyroid.utf8_len(text) == len(text.encode("utf-8"))


def test_utf8_len_is_positional_only() -> None:
    with pytest.raises(TypeError):
        _pyroid.utf8_len(text="x")  # type: ignore[call-arg]


def test_utf8_len_rejects_non_str() -> None:
    with pytest.raises(TypeError):
        _pyroid.utf8_len(b"bytes")  # type: ignore[arg-type]


def test_utf8_len_lone_surrogate_raises_like_encode() -> None:
    # A lone surrogate has no UTF-8 encoding; Python's str.encode raises too.
    with pytest.raises(UnicodeEncodeError):
        "\ud800".encode("utf-8")
    with pytest.raises(UnicodeEncodeError):
        _pyroid.utf8_len("\ud800")


@pytest.mark.skipif(not FREE_THREADED, reason="needs a free-threaded CPython build")
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
