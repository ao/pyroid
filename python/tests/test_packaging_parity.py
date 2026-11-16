"""pyroid.packaging against packaging 26.3, differentially.

Hypothesis builds versions, specifiers, markers and requirements from the PEP 440 and PEP 508
grammars (every spelling, separator and case the patterns allow), plus mutations that insert,
delete or replace characters (Python whitespace, the letters re.IGNORECASE folds onto ASCII,
other digits, a lone surrogate). Each comparison covers the result (strings, properties,
hashes, equality across the two libraries, pickles) or the exception (class and message).
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import itertools
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


def _helpers() -> Any:
    """python/tests/_packaging_util.py (CI imports tests with --import-mode=importlib)."""
    name = "_packaging_util"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("_packaging_util.py"))
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_u = _helpers()

pytest.importorskip("packaging")
if importlib.metadata.version("packaging") != "26.3":
    pytest.skip("needs packaging 26.3", allow_module_level=True)

import packaging.markers as pm  # noqa: E402
import packaging.requirements as pr  # noqa: E402
import packaging.specifiers as ps  # noqa: E402
import packaging.utils as pu  # noqa: E402
import packaging.version as pv  # noqa: E402

import pyroid.integrations  # noqa: E402
import pyroid.packaging.markers as rm  # noqa: E402
import pyroid.packaging.requirements as rr  # noqa: E402
import pyroid.packaging.specifiers as rs  # noqa: E402
import pyroid.packaging.utils as ru  # noqa: E402
import pyroid.packaging.version as rv  # noqa: E402

# PYROID_PACKAGING_EXAMPLES=20000 for a long run.
PROFILE = settings(
    max_examples=int(os.environ.get("PYROID_PACKAGING_EXAMPLES", "300")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

pytestmark = pytest.mark.skipif(
    "packaging" in pyroid.integrations.patched(), reason="compares against the unpatched packaging"
)


def same(a: Any, b: Any) -> None:
    assert a == b, f"pyroid: {a!r}\npackaging: {b!r}"


# --------------------------------------------------------------------------------------------
# Versions
# --------------------------------------------------------------------------------------------


@PROFILE
@given(_u.version_strings())
def test_version(text: str) -> None:
    ours = _u.outcome(lambda: _u.version_view(rv.Version(text)))
    theirs = _u.outcome(lambda: _u.version_view(pv.Version(text)))
    same(ours, theirs)
    same(_u.outcome(rv.parse, text)[0], _u.outcome(pv.parse, text)[0])
    if ours[0] == "ok":
        a, b = rv.Version(text), pv.Version(text)
        assert a == b and b == a and hash(a) == hash(b)
        assert isinstance(a, pv.Version)
        same(_u.version_view(_u.roundtrip(a)), ours[1])
        same(ru.canonicalize_version(a), pu.canonicalize_version(b))
    for strip in (True, False):
        same(
            _u.outcome(ru.canonicalize_version, text, strip_trailing_zero=strip),
            _u.outcome(pu.canonicalize_version, text, strip_trailing_zero=strip),
        )


@PROFILE
@given(st.lists(_u.version_strings(), min_size=2, max_size=12))
def test_version_ordering(texts: list[str]) -> None:
    ours, theirs = [], []
    for t in texts:
        try:
            theirs.append(pv.Version(t))
        except pv.InvalidVersion:
            continue
        ours.append(rv.Version(t))
    same([str(v) for v in sorted(ours)], [str(v) for v in sorted(theirs)])
    for (a, b), (c, d) in zip(itertools.product(ours, ours), itertools.product(theirs, theirs), strict=True):
        same((a < b, a <= b, a == b, a != b, a >= b, a > b), (c < d, c <= d, c == d, c != d, c >= d, c > d))
        # Across the two classes, both ways.
        same((a < d, c < b, a == d, c == b), (c < d, c < d, c == d, c == d))
    same(len(set(ours) | set(theirs)), len(set(theirs)))


@PROFILE
@given(
    st.dictionaries(
        st.sampled_from(["epoch", "release", "pre", "post", "dev", "local", "other"]),
        st.one_of(
            st.none(),
            st.integers(-2, 3),
            st.sampled_from([(1,), (1, 0), (), (0,), [1], ("a", 1), ("RC", 2), ("x", 1), ("b", -1)]),
            st.sampled_from(["abc", "a.B", "", "a..b", "1", "é"]),
            st.booleans(),
        ),
    )
)
def test_from_parts_and_replace(parts: dict[str, Any]) -> None:
    kw = {k: v for k, v in parts.items() if k != "other" and not isinstance(v, bool)}
    if "release" not in kw:
        kw["release"] = (1, 2)
    same(
        _u.outcome(lambda: _u.version_view(rv.Version.from_parts(**kw))),
        _u.outcome(lambda: _u.version_view(pv.Version.from_parts(**kw))),
    )
    same(
        _u.outcome(lambda: _u.version_view(rv.Version("1.2.3").__replace__(**kw))),
        _u.outcome(lambda: _u.version_view(pv.Version("1.2.3").__replace__(**kw))),
    )


# --------------------------------------------------------------------------------------------
# Specifiers
# --------------------------------------------------------------------------------------------

CANDIDATES = st.lists(st.one_of(_u.version_strings(), st.sampled_from(["foo", "1.0", "FOO", "2.0a1"])), max_size=10)


@PROFILE
@given(_u.specifier_strings(), CANDIDATES, st.sampled_from([None, True, False]))
def test_specifier(text: str, items: list[str], pre: bool | None) -> None:
    ours = _u.outcome(rs.Specifier, text, prereleases=pre)
    theirs = _u.outcome(ps.Specifier, text, prereleases=pre)
    same(ours[0], theirs[0])
    if ours[0] == "err":
        same(ours, theirs)
        return
    a, b = ours[1], theirs[1]
    same(_u.spec_view(a), _u.spec_view(b))
    same(_u.outcome(hash, a)[0], _u.outcome(hash, b)[0])
    if _u.outcome(hash, b)[0] == "ok":
        assert hash(a) == hash(b) and a == b and b == a
    for item in items:
        for p in (None, True, False):
            same(_u.outcome(a.contains, item, prereleases=p), _u.outcome(b.contains, item, prereleases=p))
        same(_u.outcome(lambda: item in a), _u.outcome(lambda: item in b))
    for p in (None, True, False):
        same(_u.outcome(lambda: list(a.filter(items, prereleases=p))), _u.outcome(lambda: list(b.filter(items, prereleases=p))))


@PROFILE
@given(_u.specifier_set_strings(), CANDIDATES, st.sampled_from([None, True, False]), st.booleans())
def test_specifier_set(text: str, items: list[str], pre: bool | None, objects: bool) -> None:
    ours = _u.outcome(rs.SpecifierSet, text, prereleases=pre)
    theirs = _u.outcome(ps.SpecifierSet, text, prereleases=pre)
    same(ours[0], theirs[0])
    if ours[0] == "err":
        same(ours, theirs)
        return
    a, b = ours[1], theirs[1]
    same(_u.outcome(_u.specset_view, a), _u.outcome(_u.specset_view, b))
    same(_u.outcome(lambda: a.is_unsatisfiable()), _u.outcome(lambda: b.is_unsatisfiable()))
    if objects:
        mine = [_u.outcome(rv.Version, i)[1] if _u.outcome(pv.Version, i)[0] == "ok" else i for i in items]
        theirs_items = [_u.outcome(pv.Version, i)[1] if _u.outcome(pv.Version, i)[0] == "ok" else i for i in items]
    else:
        mine = theirs_items = items
    for x, y in zip(mine, theirs_items, strict=True):
        for p in (None, True, False):
            for installed in (None, True):
                same(
                    _u.outcome(a.contains, x, prereleases=p, installed=installed),
                    _u.outcome(b.contains, y, prereleases=p, installed=installed),
                )
    for p in (None, True, False):
        same(
            _u.outcome(lambda: [str(v) for v in a.filter(mine, prereleases=p)]),
            _u.outcome(lambda: [str(v) for v in b.filter(theirs_items, prereleases=p)]),
        )
    same(_u.outcome(lambda: str(a & ">=1.0")), _u.outcome(lambda: str(b & ">=1.0")))
    if _u.outcome(hash, b)[0] == "ok":
        assert hash(a) == hash(b) and a == b and b == a


# --------------------------------------------------------------------------------------------
# Markers and requirements
# --------------------------------------------------------------------------------------------


@PROFILE
@given(_u.marker_strings(), st.sampled_from(_u.ENVIRONMENTS), st.sampled_from(_u.CONTEXTS))
def test_marker(text: str, env: dict[str, Any] | None, context: str) -> None:
    ours = _u.outcome(rm.Marker, text)
    theirs = _u.outcome(pm.Marker, text)
    same(ours[0], theirs[0])
    if ours[0] == "err":
        same(ours, theirs)
        return
    a, b = ours[1], theirs[1]
    same(_u.marker_view(a), _u.marker_view(b))
    same(_u.outcome(lambda: a.evaluate(env, context)), _u.outcome(lambda: b.evaluate(env, context)))
    same(_u.outcome(lambda: a.evaluate(env)), _u.outcome(lambda: b.evaluate(env)))
    if _u.outcome(hash, b)[0] == "ok":
        assert hash(a) == hash(b) and a == b and b == a
        same(str(a & a), str(b & b))
        same(str(a | rm.Marker("os_name == 'nt'")), str(b | pm.Marker("os_name == 'nt'")))


@PROFILE
@given(_u.requirement_strings())
def test_requirement(text: str) -> None:
    ours = _u.outcome(lambda: _u.requirement_view(rr.Requirement(text)))
    theirs = _u.outcome(lambda: _u.requirement_view(pr.Requirement(text)))
    same(ours, theirs)
    if ours[0] == "ok":
        a, b = rr.Requirement(text), pr.Requirement(text)
        assert a == b and b == a and hash(a) == hash(b)
        same(_u.requirement_view(_u.roundtrip(a)), ours[1])


# --------------------------------------------------------------------------------------------
# packaging.utils
# --------------------------------------------------------------------------------------------

FILENAME_PARTS = st.lists(
    st.sampled_from(["foo", "Foo.Bar", "foo_bar", "1.0", "1.0rc1", "x", "", "1", "1a", "py3", "py2.py3", "none", "any", "cp312", "manylinux_2_17_x86_64", "-", "__", "é"]),
    min_size=1,
    max_size=6,
)


@PROFILE
@given(FILENAME_PARTS, st.sampled_from([".whl", ".tar.gz", ".zip", "", ".whl\n"]), st.booleans())
def test_filenames(parts: list[str], ext: str, validate_order: bool) -> None:
    name = "-".join(parts) + ext

    def wheel(f: Any) -> Any:
        n, v, b, tags = f(name, validate_order=validate_order)
        return (n, str(v), b, sorted(str(t) for t in tags))

    same(_u.outcome(wheel, ru.parse_wheel_filename), _u.outcome(wheel, pu.parse_wheel_filename))
    same(
        _u.outcome(lambda: tuple(map(str, ru.parse_sdist_filename(name)))),
        _u.outcome(lambda: tuple(map(str, pu.parse_sdist_filename(name)))),
    )


@PROFILE
@given(st.text(st.sampled_from(list("aZ09._-") + _u.ODD), max_size=12), st.booleans())
def test_names(name: str, validate: bool) -> None:
    same(_u.outcome(ru.canonicalize_name, name, validate=validate), _u.outcome(pu.canonicalize_name, name, validate=validate))
    same(ru.is_normalized_name(name), pu.is_normalized_name(name))
