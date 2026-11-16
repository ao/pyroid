"""pyroid.packaging: API details, interop with packaging's own objects, threads."""

from __future__ import annotations

import copy
import itertools
import pickle
import sys
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
PACKAGING_263 = getattr(__import__("packaging"), "__version__", "") == "26.3"


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
    assert type(m) is MyVersion
    assert isinstance(m, MyVersion)
    assert m > Version("1")
    assert type(m.__replace__(local="x")) is MyVersion

    class MyRequirement(Requirement):
        def __init__(self, s: str) -> None:
            super().__init__(s)
            self.unsafe_name = self.name

    r = MyRequirement("Foo>=1")
    assert r.unsafe_name == "Foo"
    assert repr(r) == "<MyRequirement('Foo>=1')>"


@pytest.mark.skipif(not PACKAGING_26, reason="the __class__ claim is for packaging 26")
def test_packagings_code_accepts_subclass_objects() -> None:
    """Unpatched, packaging's own code takes objects of subclasses of pyroid's classes."""
    import packaging.utils as pu

    class RV(Version):
        pass

    class RS(Specifier):
        pass

    class RSS(SpecifierSet):
        pass

    class RM(Marker):
        pass

    class RR(Requirement):
        pass

    v: Any = RV("1.5.0")
    assert v.__class__ is packaging_version.Version
    assert ps.SpecifierSet(">=1.0").contains(v)
    assert list(ps.SpecifierSet(">=1.0").filter([v, "0.5"])) == [v]
    assert pu.canonicalize_version(v) == "1.5"
    assert ps.Specifier("~=1.4").contains(v)
    for obj, cls in (
        (RS(">=1"), ps.Specifier),
        (RSS(">=1"), ps.SpecifierSet),
        (RM("os_name == 'nt'"), pm.Marker),
        (RR("a>=1"), pr.Requirement),
    ):
        assert isinstance(obj, cls)
        assert type(obj).__name__.startswith("R")
    assert (ps.SpecifierSet(">=1") & cast(Any, RSS("<2"))).contains("1.5")


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


# --------------------------------------------------------------------------------------------
# Edge cases from an outside evaluation (each compared with packaging itself)
# --------------------------------------------------------------------------------------------


def _outcome(f: Any) -> tuple[str, str]:
    try:
        return ("ok", repr(f()))
    except Exception as e:  # noqa: BLE001
        return (type(e).__name__, str(e))


def test_none_arguments_raise_as_packaging_does() -> None:
    for build in (
        lambda m: m.SpecifierSet(None),
        lambda m: m.Specifier(None),
        lambda m: m.SpecifierSet(None, prereleases=True),
    ):
        mine, theirs = _outcome(lambda: build(rp.specifiers)), _outcome(lambda: build(ps))
        assert mine == theirs
        assert mine[0] == "TypeError"
    assert str(SpecifierSet()) == "" and len(SpecifierSet()) == 0
    with pytest.raises(InvalidSpecifier, match="^Invalid specifier: ''$"):
        Specifier()


@pytest.mark.parametrize(
    ("marker", "environment"),
    [
        ("os_name == 'posix'", {"extra": 1}),
        ("extra == 'x'", {"extra": 1}),
        ("os_name == 'posix'", {"extra": b"x"}),
        ("os_name == 'posix'", {"python_full_version": None}),
        ("os_name == 'posix'", {"python_full_version": 3}),
        ("os_name == 'posix'", {"extra": None}),
        ("extra == ''", {"extra": None}),
        ("extra == ''", {"extra": 0}),
        ("python_full_version >= '3'", {"python_full_version": "3.12.0+"}),
    ],
)
def test_marker_environment_is_prepared_first(marker: str, environment: dict[str, Any]) -> None:
    """packaging canonicalises `extra` and repairs `python_full_version` before evaluating."""
    env = {"os_name": "posix", **environment}
    mine = _outcome(lambda: Marker(marker).evaluate(env))
    assert mine == _outcome(lambda: pm.Marker(marker).evaluate(env))


@pytest.mark.thread_unsafe(reason="changes sys.set_int_max_str_digits")
def test_digit_limit_applies_when_printing() -> None:
    """A Version made with the int-string limit off prints (or not) as packaging's does."""
    import sys

    import packaging.utils as pu

    text = "1." + "7" * 5000 + "+" + "3" * 5000
    old = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        mine, theirs = Version(text), packaging_version.Version(text)
    finally:
        sys.set_int_max_str_digits(4300)
    try:
        checks: list[Any] = [
            lambda v: len(str(v)),
            lambda v: v.release[1] % 97,
            lambda v: v.local,
            lambda v: v.public,
            lambda v: v.base_version,
            lambda v: repr(v)[:20],
            lambda v: isinstance(hash(v), int),
            lambda v: v == theirs,
        ]
        for check in checks:
            assert _outcome(lambda c=check: c(mine)) == _outcome(lambda c=check: c(theirs))
        assert _outcome(lambda: str(mine))[0] == "ValueError"
        assert _outcome(lambda: canonicalize_version(mine)) == _outcome(
            lambda: pu.canonicalize_version(theirs)
        )
    finally:
        sys.set_int_max_str_digits(old)


def test_from_parts_requires_release() -> None:
    assert _outcome(lambda: Version.from_parts()) == _outcome(  # type: ignore[call-arg]
        lambda: packaging_version.Version.from_parts()
    )
    assert Version.from_parts(release=None) == packaging_version.Version.from_parts(release=None)


@pytest.mark.skipif(not PACKAGING_263, reason="pyroid has packaging 26.3's attributes")
def test_packagings_class_attributes() -> None:
    assert Specifier._operators == ps.Specifier._operators  # type: ignore[attr-defined]
    for mine, theirs in (
        (Version, packaging_version.Version),
        (Specifier, ps.Specifier),
        (SpecifierSet, ps.SpecifierSet),
        (Marker, pm.Marker),
    ):
        assert getattr(mine, "__slots__", None) == getattr(theirs, "__slots__", None)
    assert Version._regex.pattern == packaging_version.Version._regex.pattern  # type: ignore[attr-defined]


DEEP = 100_000


def test_deep_markers_do_not_use_the_stack() -> None:
    """Nesting is limited by memory, not by the native stack (packaging raises RecursionError)."""
    deep = "(" * DEEP + "os_name == 'nt'" + ")" * DEEP
    m = Marker(deep)
    assert str(m) == 'os_name == "nt"'
    assert m.evaluate({"os_name": "nt"})
    assert m == Marker("os_name == 'nt'")
    assert hash(m) == hash(Marker("os_name == 'nt'"))
    r = Requirement("foo; " + deep)
    assert str(r) == 'foo; os_name == "nt"'
    assert pickle.loads(pickle.dumps(m)) == m
    with pytest.raises(InvalidMarker) as info:
        Marker("(" * DEEP)
    assert str(info.value).startswith("Expected a marker variable or quoted string")
    with pytest.raises(InvalidMarker):
        Marker("(" * DEEP + "os_name == 'nt'")
    # A group at every level: printed with all its parentheses.
    chained = "(os_name == 'a' or " * 5000 + "os_name == 'nt'" + ")" * 5000
    m = Marker(chained)
    assert str(m).count("(") == 4999
    assert m.evaluate({"os_name": "nt"})
    # packaging's _markers: nested lists as deep as the marker, read back the same way.
    nested = Marker(deep)._markers
    back = Marker.__new__(Marker)
    back._markers = nested
    assert str(back) == 'os_name == "nt"'
    del nested, back


@pytest.mark.thread_unsafe(reason="sets its own thread stack size")
@pytest.mark.parametrize("stack", [256 << 10, 1 << 20])
def test_deep_markers_in_a_thread_with_a_small_stack(stack: int) -> None:
    deep = "(" * 20_000 + "os_name == 'nt'" + ")" * 20_000
    results: list[Any] = []

    def work() -> None:
        try:
            m = Marker(deep)
            results.append((m.evaluate({"os_name": "nt"}), str(m), hash(m) == hash(m)))
            results.append(str(Requirement("foo; " + deep)))
            m2, other = Marker("os_name == 'a'"), Marker("os_name == 'b'")
            for _ in range(20_000):
                m2 = m2 & other
            results.append(m2.evaluate({"os_name": "nt"}))
        except BaseException as e:  # noqa: BLE001
            results.append(e)

    old = threading.stack_size(stack)
    try:
        t = threading.Thread(target=work)
        t.start()
        t.join()
    finally:
        threading.stack_size(old)
    assert results == [(True, 'os_name == "nt"', True), 'foo; os_name == "nt"', False]


def test_combining_markers_shares_the_groups() -> None:
    """`m & other` is constant time and `m & m` holds m once, as packaging's lists do."""
    import time

    other = Marker("os_name == 'b'")
    m = Marker("os_name == 'a'")
    start = time.perf_counter()
    for _ in range(50_000):
        m = m & other
    assert time.perf_counter() - start < 5
    assert m.evaluate({"os_name": "a"}) is False
    assert m.evaluate({"os_name": "b"}) is False
    assert str(m).startswith("(" * 49_999 + 'os_name == "a" and os_name == "b")')
    m = Marker("os_name == 'a'")
    for _ in range(18):
        m = m | m
    tree = m._markers
    assert tree[0] is tree[2]
    assert m.evaluate({"os_name": "a"})
    assert (m | Marker("os_name == 'c'")).evaluate({"os_name": "c"})
    # A packaging marker whose groups are shared stays small when pyroid reads it (2**40 leaves).
    p = pm.Marker("os_name == 'a'")
    if hasattr(p, "__and__"):
        for _ in range(40):
            p = p & p
        both = (Marker("os_name == 'x'") | cast(Any, p))._markers
        assert both[2][0] is both[2][2]


def test_a_marker_list_that_holds_itself() -> None:
    loop: list[Any] = []
    loop.append(loop)
    m = Marker.__new__(Marker)
    with pytest.raises(RecursionError):
        m._markers = loop


def test_and_with_packagings_specifier_set() -> None:
    for ours, theirs in (
        (SpecifierSet(">=1.0"), ps.SpecifierSet("<3")),
        (SpecifierSet(">=1.0", prereleases=True), ps.SpecifierSet("<3")),
        (SpecifierSet(">=1.0"), ps.SpecifierSet("<3,===foo")),
    ):
        both = ours & cast(Any, theirs)
        assert type(both) is SpecifierSet
        expected = ps.SpecifierSet(str(ours)) & theirs
        if ours._prereleases is not None:
            expected = ps.SpecifierSet(str(ours), prereleases=ours._prereleases) & theirs
        assert str(both) == str(expected)
        assert both.prereleases == expected.prereleases
        for v in ("0.5", "1.0", "2.5a1", "3.0", "foo"):
            assert both.contains(v) == expected.contains(v)
    with pytest.raises(ValueError, match="Cannot combine"):
        SpecifierSet(">=1", prereleases=True) & cast(Any, ps.SpecifierSet("<3", prereleases=False))
    assert SpecifierSet(">=1").__and__(1) is NotImplemented  # type: ignore[operator]


def test_specifier_set_of_packagings_specifiers() -> None:
    items = [ps.Specifier(">=1.0"), ps.Specifier("!=1.5.*"), ps.Specifier("<3", prereleases=True)]
    ours, theirs = SpecifierSet(cast(Any, items)), ps.SpecifierSet(items)
    candidates = ["0.9", "1.0", "1.5.2", "2.0a1", "2.9", "3.0", "nope"]
    assert list(ours) == items  # the caller's objects, as given (until printed)
    assert str(ours) == str(theirs)
    assert [ours.contains(v) for v in candidates] == [theirs.contains(v) for v in candidates]
    assert list(ours.filter(candidates)) == list(theirs.filter(candidates))
    assert ours.prereleases == theirs.prereleases
    assert list(ours) == list(theirs)
    assert rp.filter_batch(ours, candidates) == list(theirs.filter(candidates))
    # Strings are not specifiers: the same AttributeError as packaging's.
    assert _outcome(lambda: SpecifierSet(cast(Any, [">=1"])).contains("1.5")) == _outcome(
        lambda: ps.SpecifierSet(cast(Any, [">=1"])).contains("1.5")
    )


def test_fallback_warning_points_at_the_caller() -> None:
    import packaging.ranges  # noqa: F401  # the code these calls hand over to

    from pyroid import FallbackWarning
    from pyroid.packaging import _compat

    _compat._WARNED.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = sys._getframe().f_lineno + 1
        SpecifierSet(">=1").to_range()
        SpecifierSet(">=1").is_subset(SpecifierSet(">=0"))
        SpecifierSet(">=1").is_superset(ps.SpecifierSet(">=2"))
        SpecifierSet(">=1").is_disjoint(SpecifierSet("<1"))
    got = [(w.filename, w.lineno, str(w.message).split(" runs")[0]) for w in caught]
    assert all(w.category is FallbackWarning for w in caught)
    assert got == [
        (__file__, line + i, f"pyroid.packaging: SpecifierSet.{name}()")
        for i, name in enumerate(("to_range", "is_subset", "is_superset", "is_disjoint"))
    ]
