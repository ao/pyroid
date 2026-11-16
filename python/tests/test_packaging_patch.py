"""pyroid.patch("packaging"): packaging's modules use pyroid's classes, and unpatch undoes it."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import pickle
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


def test_objects_made_before_the_patch_keep_working() -> None:
    """packaging's objects from before the patch work with each other and with pyroid's after."""
    import packaging.markers as pm
    import packaging.requirements as pr
    import packaging.specifiers as ps
    import packaging.version as pv

    ss, v, spec = ps.SpecifierSet(">=1.0"), pv.Version("1.5"), ps.Specifier(">=1.0")
    m, r = pm.Marker("os_name == 'posix'"), pr.Requirement("a>=1; os_name == 'posix'")
    candidates = [v, "1.6", "0.5", pv.Version("2.0a1")]

    def observe() -> tuple[Any, ...]:
        return (
            ss.contains(v),
            list(ss.filter(candidates)),
            spec.contains(v),
            list(spec.filter(candidates)),
            ss.contains("1.2"),
            m.evaluate({"os_name": "posix"}),
            r.specifier.contains(v),
            v < pv.Version("2"),
        )

    want = observe()
    _patched()
    try:
        assert observe() == want
        new = pv.Version("1.5")
        assert type(new) is not type(v)
        assert new == v and v == new and hash(new) == hash(v)
        assert pv.Version("1.0") < v < pv.Version("2.0")
        assert pv.Version(v) == v  # taken by value while patched
        assert ps.SpecifierSet(">=1").contains(v)
        assert list(ps.SpecifierSet(">=1").filter([v, "0.1"])) == [v]
        both = ss & ps.SpecifierSet("<2")
        assert both.contains(v) and not both.contains("2.5")
        assert pm.Marker("os_name == 'posix'") == m
        assert pr.Requirement("a>=1; os_name == 'posix'") == r
    finally:
        integrations.unpatch("packaging")
    assert observe() == want
    with pytest.raises(pv.InvalidVersion):
        importlib.import_module("pyroid.packaging.version").Version(v)  # as packaging's own


def test_pickles_of_objects_made_before_the_patch() -> None:
    import packaging.markers as pm
    import packaging.requirements as pr
    import packaging.specifiers as ps
    import packaging.version as pv

    objs: list[Any] = [
        pv.Version("1.5.post1+x"),
        ps.Specifier(">=1", prereleases=True),
        ps.SpecifierSet(">=1,<2,!=1.5.*", prereleases=False),
        pm.Marker("os_name == 'nt' or extra == 'Test_Me'"),
        pr.Requirement("a[b]>=1; os_name == 'nt'"),
    ]
    classes = [type(o) for o in objs]
    _patched()
    try:
        for o in objs:
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                back = pickle.loads(pickle.dumps(o, protocol))
                assert back == o
                assert str(back) == str(o)
                # The class packaging's name gives now: pyroid's.
                assert type(back) is getattr(sys.modules[type(o).__module__], type(o).__name__)
        assert pickle.loads(pickle.dumps(objs[1])).prereleases is True
        assert pickle.loads(pickle.dumps(objs[2])).prereleases is False
    finally:
        integrations.unpatch("packaging")
    for o, cls in zip(objs, classes):
        assert type(pickle.loads(pickle.dumps(o))) is cls
        assert "__reduce_ex__" not in vars(cls)


def test_patch_warning_names_the_release() -> None:
    integrations.unpatch("packaging")
    integrations._patch._WARNED.discard("packaging")
    with pytest.warns(integrations.PatchWarning, match="from packaging.version import Version") as w:
        integrations.patch("packaging")
    integrations.unpatch("packaging")
    unported = [m for m in ("pylock", "ranges", "dependency groups") if m in str(w[0].message)]
    present = [
        m
        for m in ("pylock", "ranges", "dependency_groups")
        if importlib.util.find_spec(f"packaging.{m}") is not None
    ]
    assert len(unported) == len(present)


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
