"""pyroid.patch("packaging"): packaging's modules use pyroid's classes, and unpatch undoes it."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any

import pytest

import pyroid.integrations as integrations

pytest.importorskip("packaging")
pytestmark = pytest.mark.thread_unsafe(reason="patches sys.modules and packaging's globals")

DATA = Path(__file__).resolve().parents[2] / "parity" / "data" / "packaging-26.3"


def _patched() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", integrations.PatchWarning)
        integrations.patch("packaging")


def test_patch_replaces_and_unpatch_restores() -> None:
    import packaging.markers
    import packaging.requirements
    import packaging.specifiers
    import packaging.utils
    import packaging.version

    # Any: the patch makes packaging's names pyroid's objects, which the type checker can't see.
    rm: Any = importlib.import_module("pyroid.packaging.markers")
    rr: Any = importlib.import_module("pyroid.packaging.requirements")
    rs: Any = importlib.import_module("pyroid.packaging.specifiers")
    ru: Any = importlib.import_module("pyroid.packaging.utils")
    rv: Any = importlib.import_module("pyroid.packaging.version")

    originals = (
        packaging.version.Version,
        packaging.specifiers.SpecifierSet,
        packaging.requirements.Requirement,
    )
    _patched()
    try:
        assert packaging.version.Version is rv.Version
        assert packaging.specifiers.Version is rv.Version  # type: ignore[attr-defined]
        assert packaging.specifiers.SpecifierSet is rs.SpecifierSet
        assert packaging.markers.Marker is rm.Marker
        assert packaging.requirements.Requirement is rr.Requirement
        assert packaging.requirements.SpecifierSet is rs.SpecifierSet  # type: ignore[attr-defined]
        assert packaging.utils.canonicalize_name is ru.canonicalize_name
        if hasattr(packaging, "ranges"):
            import packaging.ranges

            assert packaging.ranges.Version is rv.Version  # type: ignore[attr-defined]
        v = packaging.version.Version("1.0")
        assert type(v) is rv.Version
        assert v.__class__ is rv.Version
        # no claim under the patch
        # Objects made by code that held packaging's class before the patch still compare.
        old = originals[0]("1.0")
        assert old == v
        assert hash(old) == hash(v)
        assert len({old, v}) == 1
    finally:
        integrations.unpatch("packaging")
    assert (
        packaging.version.Version,
        packaging.specifiers.SpecifierSet,
        packaging.requirements.Requirement,
    ) == originals
    assert packaging.specifiers.Version is originals[0]  # type: ignore[attr-defined]


def test_patch_warning_names_the_release() -> None:
    integrations.unpatch("packaging")
    integrations._patch._WARNED.discard("packaging")
    with pytest.warns(integrations.PatchWarning, match="from packaging.version import Version"):
        integrations.patch("packaging")
    integrations.unpatch("packaging")


def test_importing_pyroid_patches_nothing() -> None:
    code = textwrap.dedent(
        """
        import pyroid, pyroid.packaging, pyroid.integrations, packaging.version
        import pyroid.packaging.version as rv
        assert packaging.version.Version is not rv.Version
        assert pyroid.integrations.patched() == ()
        print("ok")
        """
    )
    env = {k: v for k, v in os.environ.items() if k != "PYROID_PATCH"}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    assert out.stdout.strip() == "ok"


def test_pyroid_run(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="the pyroid command line needs pyarrow (pyroid[cli])")
    script = tmp_path / "probe.py"
    script.write_text(
        "import packaging.version, packaging.specifiers, pyroid.packaging.version as rv\n"
        "assert packaging.version.Version is rv.Version\n"
        "assert packaging.specifiers.SpecifierSet('>=1').contains('2.0')\n"
        "print('patched')\n"
    )
    out = subprocess.run(
        [sys.executable, "-m", "pyroid", "run", "-p", "packaging", str(script)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "patched"


def test_without_packaging_installed() -> None:
    code = textwrap.dedent(
        """
        import sys, warnings
        sys.modules["packaging"] = None  # as if it were not installed
        import pyroid.integrations
        from pyroid.packaging.version import Version, InvalidVersion
        assert InvalidVersion.__module__ == "pyroid.packaging.version"
        assert Version("1.0").__class__ is Version
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pyroid.integrations.patch("packaging")
        from packaging.version import Version as V
        from packaging.specifiers import SpecifierSet
        from packaging.requirements import Requirement
        from packaging.utils import parse_wheel_filename
        assert V is Version and SpecifierSet(">=1").contains("1.5")
        assert str(Requirement("a[b]>=1; os_name == 'posix'")) == 'a[b]>=1; os_name == "posix"'
        assert parse_wheel_filename("foo-1.0-py3-none-any.whl")[0] == "foo"
        try:
            SpecifierSet(">=1").to_range()
        except pyroid.PyroidError:
            print("ok")
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "ok"


@pytest.mark.skipif(not (DATA / "tests").is_dir(), reason="needs parity/data/packaging-26.3")
@pytest.mark.skipif(
    importlib.metadata.version("packaging") != "26.3", reason="packaging's tests are 26.3's"
)
@pytest.mark.timeout(1800)
def test_packagings_own_test_suite() -> None:
    """packaging 26.3's whole test suite, unchanged, with pyroid.patch("packaging") on."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent), env.get("PYTHONPATH", "")])
    args = [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        os.devnull,
        "-p",
        "no:cacheprovider",
        "-p",
        "_packaging_upstream",
    ]
    args += ["-q", "-W", "ignore::pytest.PytestUnknownMarkWarning", "-p", "no:randomly", "tests"]
    if os.environ.get("PYROID_PACKAGING_UPSTREAM") != "all":
        args += ["--ignore=tests/property"]
    out = subprocess.run(args, cwd=DATA, capture_output=True, text=True, env=env)
    tail = "\n".join(out.stdout.splitlines()[-30:])
    assert out.returncode == 0, tail
    assert " passed" in tail, tail
    assert "failed" not in tail, tail
