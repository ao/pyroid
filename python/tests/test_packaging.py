"""pyroid.packaging: API details, interop with packaging's own objects, threads."""

from __future__ import annotations

import copy
import itertools
import pickle
import threading
import warnings
from collections.abc import Iterator
from typing import Any, cast

import pytest

import pyroid.packaging as rp
from pyroid.packaging.markers import (
    InvalidMarker,
    Marker,
    UndefinedEnvironmentName,
    default_environment,
)
from pyroid.packaging.requirements import InvalidRequirement, Requirement
from pyroid.packaging.specifiers import InvalidSpecifier, Specifier, SpecifierSet
from pyroid.packaging.utils import canonicalize_name, canonicalize_version
from pyroid.packaging.version import InvalidVersion, Version, parse

packaging_version: Any = pytest.importorskip("packaging.version")
import packaging.markers as pm  # noqa: E402
import packaging.requirements as pr  # noqa: E402
import packaging.specifiers as ps  # noqa: E402

PACKAGING_26 = getattr(__import__("packaging"), "__version__", "").startswith("26.")


def test_version_basics() -> None:
    v = Version("1!2.0.3rc4.post5.dev6+Ubuntu.1")
    assert str(v) == "1!2.0.3rc4.post5.dev6+ubuntu.1"
    assert repr(v) == "<Version('1!2.0.3rc4.post5.dev6+ubuntu.1')>"
    assert (v.epoch, v.release, v.pre, v.post, v.dev, v.local) == (
        1,
        (2, 0, 3),
        ("rc", 4),
        5,
        6,
        "ubuntu.1",
    )
    assert v.public == "1!2.0.3rc4.post5.dev6"
    assert v.base_version == "1!2.0.3"
    assert Version("1.0") == Version("1.0.0")
    assert hash(Version("1.0")) == hash(Version("1.0.0"))
    assert parse("1.0a1") < Version("1.0")
    with pytest.raises(InvalidVersion, match=r"^Invalid version: 'foo'$"):
        Version("foo")
    with pytest.raises(InvalidVersion, match=r"^Invalid version: None$"):
        Version(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'version'"):
        Version()  # type: ignore[call-arg]
    assert Version(version="2.0") == Version("2")
    assert Version.from_parts(release=(1, 2), pre=("ALPHA", 1)) == Version("1.2a1")


def test_version_huge_numbers_raise_as_int_does() -> None:
    import sys

    old = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(4300)
    try:
        with pytest.raises(ValueError, match="Exceeds the limit"):
            Version("1." + "1" * 5000)
        assert Version("1." + "9" * 100).release[1] == int("9" * 100)
    finally:
        sys.set_int_max_str_digits(old)


def test_errors_are_packagings_own_classes() -> None:
    assert InvalidVersion is packaging_version.InvalidVersion
    assert InvalidSpecifier is cast(Any, ps.InvalidSpecifier)
    assert InvalidMarker is cast(Any, pm.InvalidMarker)
    assert InvalidRequirement is cast(Any, pr.InvalidRequirement)
    with pytest.raises(packaging_version.InvalidVersion):
        Version("1..0")


@pytest.mark.skipif(not PACKAGING_26, reason="the __class__ claim is for packaging 26")
def test_isinstance_with_packagings_classes() -> None:
    assert isinstance(Version("1.0"), packaging_version.Version)
    assert isinstance(SpecifierSet(">=1"), ps.SpecifierSet)
    assert isinstance(Specifier(">=1"), ps.Specifier)
    assert isinstance(SpecifierSet(">=1"), ps.BaseSpecifier)
    assert isinstance(Marker("os_name == 'nt'"), pm.Marker)
    assert isinstance(Requirement("a>=1"), pr.Requirement)
    assert type(Version("1.0")) is Version
    # packaging's own code accepts pyroid's objects.
    v2, v05, v15 = (cast(Any, Version(s)) for s in ("2.0", "0.5", "1.5"))
    assert ps.SpecifierSet(">=1").contains(v2)
    assert not ps.SpecifierSet(">=1").contains(v05)
    assert ps.Specifier("~=1.4").contains(v15)


def test_mixed_sets_sorts_and_comparisons() -> None:
    ours = [Version(s) for s in ("1.0", "1.0.0", "2.0a1", "1.0+ABC", "0.9")]
    theirs = [packaging_version.Version(s) for s in ("1.0", "2.0a1", "1.0+abc", "3")]
    assert len(set(ours) | set(theirs)) == len(set(theirs) | {packaging_version.Version("0.9")})
    for a, b in itertools.product(ours, theirs):
        assert (a < b) == (packaging_version.Version(str(a)) < b) == (b > a)
        assert (a == b) == (b == a)
        if a == b:
            assert hash(a) == hash(b)
    assert [str(v) for v in sorted(ours + theirs)] == [
        str(v) for v in sorted(packaging_version.Version(str(v)) for v in ours + theirs)
    ]
    assert SpecifierSet(">=1,<2") == ps.SpecifierSet("<2.0,>=1.0") == SpecifierSet("<2.0,>=1.0")
    assert hash(SpecifierSet(">=1,<2")) == hash(ps.SpecifierSet("<2,>=1"))
    assert Requirement("Foo[Bar]>=1.0") == pr.Requirement("foo[bar]>=1.0.0")
    assert hash(Marker("os_name=='a'")) == hash(pm.Marker('os_name == "a"'))


def test_pickle_and_copy() -> None:
    objs = [
        Version("1.0rc1+x"),
        Specifier(">=1", prereleases=True),
        SpecifierSet(">=1,<2", prereleases=False),
        Marker("python_version > '3' and extra == 'Test_Me'"),
        Requirement("a[b]>=1; os_name=='posix'"),
    ]
    for obj in objs:
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            back = pickle.loads(pickle.dumps(obj, protocol))
            assert type(back) is type(obj)
            assert str(back) == str(obj)
            assert back == obj
        assert copy.copy(obj) == obj
        assert copy.deepcopy(obj) == obj
    assert pickle.loads(pickle.dumps(objs[1])).prereleases is True
    assert pickle.loads(pickle.dumps(objs[4])).specifier._prereleases is None


def test_packagings_pickle_states_load() -> None:
    v = Version.__new__(Version)
    v.__setstate__(packaging_version.Version("1.0.post1").__getstate__())
    assert v == Version("1.0.post1")
    s = Specifier.__new__(Specifier)
    s.__setstate__(ps.Specifier(">=1.0", prereleases=True).__getstate__())
    assert str(s) == ">=1.0"
    assert s.prereleases is True
    m = Marker.__new__(Marker)
    m.__setstate__(pm.Marker("os_name == 'nt'").__getstate__())
    assert str(m) == 'os_name == "nt"'


def test_subclasses() -> None:
    class MyVersion(Version):
        def __init__(self, raw: str) -> None:
            super().__init__(raw.removeprefix("release-"))
            self.raw = raw

    m = MyVersion("release-1.2")
    assert str(m) == "1.2"
    assert m.raw == "release-1.2"
    assert repr(m) == "<MyVersion('1.2')>"
    assert m.__class__ is MyVersion
    assert m > Version("1")
    assert type(m.__replace__(local="x")) is MyVersion

    class MyRequirement(Requirement):
        def __init__(self, s: str) -> None:
            super().__init__(s)
            self.unsafe_name = self.name

    assert MyRequirement("Foo>=1").unsafe_name == "Foo"


def test_version_can_be_changed_in_place() -> None:
    v = Version("2.3.4")
    with pytest.warns(DeprecationWarning, match="is private"):
        parts = v._version
    with pytest.warns(DeprecationWarning, match="is private"):
        v._version = parts._replace(post=("post", 1))
    assert str(v) == "2.3.4.post1"


def test_filter_is_lazy() -> None:
    def forever() -> Iterator[str]:
        for i in itertools.count():
            yield f"1.{i}"

    it = SpecifierSet(">=1.5").filter(forever())
    assert next(it) == "1.5"
    assert next(it) == "1.6"
    it = Specifier(">=1.0", prereleases=False).filter(forever())
    assert next(it) == "1.0"


def test_prerelease_settings_are_live() -> None:
    s = Specifier(">=1.0")
    ss = SpecifierSet([s])
    assert ss.contains("1.5a1")
    assert ss.prereleases is None
    s.prereleases = True
    assert ss.prereleases is True
    ss.prereleases = False
    assert not ss.contains("1.5a1")


def test_canonicalisation_reorders_iteration() -> None:
    ss = SpecifierSet(">=1,>=1.0,<2")
    assert len(ss) == 3
    assert [str(x) for x in ss] == [">=1", ">=1.0", "<2"]
    assert str(ss) == "<2,>=1"
    assert len(ss) == 2
    assert [str(x) for x in ss] == ["<2", ">=1"]


def test_markers() -> None:
    m = Marker("python_version >= '3.8' and (os_name == 'nt' or extra == 'Test_Me')")
    assert str(m) == 'python_version >= "3.8" and (os_name == "nt" or extra == "test-me")'
    assert m.evaluate({"python_version": "3.12", "os_name": "posix", "extra": "TEST_ME"})
    with pytest.raises(UndefinedEnvironmentName):
        Marker("extra == 'x'").evaluate(context="requirement")
    with pytest.raises(KeyError):
        Marker("extra == 'x'").evaluate(context="requirement")
    with pytest.raises(InvalidMarker, match="Expected a marker variable or quoted string"):
        Marker("python_version >")
    assert default_environment() == default_environment()
    assert default_environment() is not default_environment()


def test_requirements() -> None:
    r = Requirement("requests[security,socks]>=2.8.1,==2.8.*; python_version < '2.7'")
    assert (r.name, sorted(r.extras), str(r.specifier), str(r.marker)) == (
        "requests",
        ["security", "socks"],
        "==2.8.*,>=2.8.1",
        'python_version < "2.7"',
    )
    r.marker = None
    assert str(r) == "requests[security,socks]==2.8.*,>=2.8.1"
    with pytest.raises(InvalidRequirement) as info:
        Requirement("name >=1.0 extra")
    assert str(info.value).startswith("Expected comma (within version specifier), semicolon")
    assert type(info.value.__cause__).__name__ == "ParserSyntaxError"


def test_utils() -> None:
    assert canonicalize_name("Foo.Bar__baz") == "foo-bar-baz"
    assert canonicalize_version("1.0.0") == "1"
    assert canonicalize_version(Version("1.4.0.0")) == "1.4"
    assert canonicalize_version("foo bar") == "foo bar"


# --------------------------------------------------------------------------------------------
# Threads (pytest-run-parallel runs these on free-threaded builds too)
# --------------------------------------------------------------------------------------------


@pytest.mark.thread_unsafe(reason="starts its own threads")
def test_shared_objects_from_many_threads() -> None:
    ss = SpecifierSet(">=1.0,!=1.5.*,<3,!=2.2")
    v = Version("2.1.0rc1+local")
    versions = [f"{a}.{b}{s}" for a in range(4) for b in range(10) for s in ("", "a1", ".post1")]
    want = [ss.contains(x) for x in versions]
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def work(k: int) -> None:
        try:
            barrier.wait(timeout=30)
            for _ in range(50):
                assert [ss.contains(x) for x in versions] == want
                assert hash(v) == hash(Version("2.1.0rc1+local"))
                assert str(ss) == "!=1.5.*,!=2.2,<3,>=1.0"
                if k == 0:
                    ss.prereleases = None
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=work, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not errors, errors


def test_batch_threads_agree() -> None:
    versions = [f"{a}.{b}.{c}" for a in range(5) for b in range(20) for c in range(20)]
    base = rp.sort_versions(versions, threads=1)
    for n in (None, 2, 8):
        assert rp.sort_versions(versions, threads=n) == base
        assert rp.contains_batch(">=2,<4", versions, threads=n) == rp.contains_batch(
            ">=2,<4", versions, threads=1
        )


def test_no_warning_on_import() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        import importlib

        importlib.reload(rp)
