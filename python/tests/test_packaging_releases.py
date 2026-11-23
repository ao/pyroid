# The stand-in releases below are derived from packaging 25.0 and 26.2 (Apache-2.0 OR
# BSD-2-Clause), © Donald Stufft and individual contributors; reduced to what these tests need.
"""pyroid.packaging with a packaging release other than 26.3 installed.

Each test writes a small stand-in for an older (or unknown) packaging release to a temporary
directory and runs a child process with it first on ``sys.path``. The stand-ins keep what
pyroid depends on: packaging 25.0's ``_BaseVersion`` protocol and comparison key (a tuple with
``_structures.Infinity``), packaging 26.2's ``UndefinedEnvironmentName`` (a ``ValueError`` that
a missing key never raises), and the modules that 24.2 lacks.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.thread_unsafe(reason="starts child processes with their own packaging")

_INFINITY = """
class InfinityType:
    def __repr__(self):
        return "Infinity"

    def __hash__(self):
        return hash(repr(self))

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return False

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __gt__(self, other):
        return True

    def __ge__(self, other):
        return True

    def __neg__(self):
        return NegativeInfinity


Infinity = InfinityType()


class NegativeInfinityType:
    def __repr__(self):
        return "-Infinity"

    def __hash__(self):
        return hash(repr(self))

    def __lt__(self, other):
        return True

    def __le__(self, other):
        return True

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return False

    def __neg__(self):
        return Infinity


NegativeInfinity = NegativeInfinityType()
"""

# packaging 25.0's version module, without the properties these tests do not read.
_VERSION_25 = r'''
import itertools
import re

from ._structures import Infinity, NegativeInfinity

VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?
        (?P<release>[0-9]+(?:\.[0-9]+)*)
        (?P<pre>
            [-_\.]?
            (?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?
"""


class InvalidVersion(ValueError):
    pass


class _BaseVersion:
    def __hash__(self):
        return hash(self._key)

    def __lt__(self, other):
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key < other._key

    def __le__(self, other):
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key <= other._key

    def __eq__(self, other):
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key == other._key

    def __ge__(self, other):
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key >= other._key

    def __gt__(self, other):
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key > other._key

    def __ne__(self, other):
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key != other._key


def _letter(letter, number):
    if letter:
        if number is None:
            number = 0
        letter = letter.lower()
        letter = {"alpha": "a", "beta": "b", "c": "rc", "pre": "rc", "preview": "rc",
                  "rev": "post", "r": "post"}.get(letter, letter)
        return letter, int(number)
    if number:
        return "post", int(number)
    return None


def _local(local):
    if local is not None:
        return tuple(
            part.lower() if not part.isdigit() else int(part)
            for part in re.split(r"[\._-]", local)
        )
    return None


def _cmpkey(epoch, release, pre, post, dev, local):
    _release = tuple(reversed(list(itertools.dropwhile(lambda x: x == 0, reversed(release)))))
    if pre is None and post is None and dev is not None:
        _pre = NegativeInfinity
    elif pre is None:
        _pre = Infinity
    else:
        _pre = pre
    _post = NegativeInfinity if post is None else post
    _dev = Infinity if dev is None else dev
    if local is None:
        _local_key = NegativeInfinity
    else:
        _local_key = tuple((i, "") if isinstance(i, int) else (NegativeInfinity, i) for i in local)
    return epoch, _release, _pre, _post, _dev, _local_key


class Version(_BaseVersion):
    _regex = re.compile(r"^\s*" + VERSION_PATTERN + r"\s*$", re.VERBOSE | re.IGNORECASE)

    def __init__(self, version):
        match = self._regex.search(version)
        if not match:
            raise InvalidVersion(f"Invalid version: {version!r}")
        self._parts = (
            int(match.group("epoch")) if match.group("epoch") else 0,
            tuple(int(i) for i in match.group("release").split(".")),
            _letter(match.group("pre_l"), match.group("pre_n")),
            _letter(match.group("post_l"), match.group("post_n1") or match.group("post_n2")),
            _letter(match.group("dev_l"), match.group("dev_n")),
            _local(match.group("local")),
        )
        self._key = _cmpkey(*self._parts)

    def __str__(self):
        epoch, release, pre, post, dev, local = self._parts
        out = ".".join(map(str, release))
        if epoch:
            out = f"{epoch}!{out}"
        if pre is not None:
            out += "".join(map(str, pre))
        if post is not None:
            out += f".post{post[1]}"
        if dev is not None:
            out += f".dev{dev[1]}"
        if local is not None:
            out += "+" + ".".join(map(str, local))
        return out

    def __repr__(self):
        return f"<Version({str(self)!r})>"


def parse(version):
    return Version(version)
'''

# A release whose key pyroid does not build: the string itself.
_VERSION_ODD = _VERSION_25.replace(
    "        self._key = _cmpkey(*self._parts)", '        self._key = ("odd", str(self))'
)

# packaging 26.0's specifiers, as far as these tests read them: a set is a frozenset of
# specifiers, hashed and compared as one (26.1 and later compare _canonical_specs()).
_SPECIFIERS_260 = """
import abc


class InvalidSpecifier(ValueError):
    pass


class BaseSpecifier(metaclass=abc.ABCMeta):
    pass


class Specifier(BaseSpecifier):
    def __init__(self, spec="", prereleases=None):
        op = next(o for o in ("~=", "==", "!=", "<=", ">=", "<", ">") if spec.startswith(o))
        self._spec = (op, spec[len(op):].strip())
        self._prereleases = prereleases

    @property
    def _canonical_spec(self):
        return self._spec

    def __str__(self):
        return "".join(self._spec)

    def __hash__(self):
        return hash(self._canonical_spec)

    def __eq__(self, other):
        if isinstance(other, str):
            other = Specifier(other)
        elif not isinstance(other, Specifier):
            return NotImplemented
        return self._canonical_spec == other._canonical_spec


class SpecifierSet(BaseSpecifier):
    def __init__(self, specifiers="", prereleases=None):
        parts = [s.strip() for s in specifiers.split(",") if s.strip()]
        self._specs = frozenset(map(Specifier, parts))
        self._prereleases = prereleases

    def __hash__(self):
        return hash(self._specs)

    def __eq__(self, other):
        if not isinstance(other, SpecifierSet):
            return NotImplemented
        return self._specs == other._specs

    def __and__(self, other):
        if not isinstance(other, SpecifierSet):
            return NotImplemented
        out = SpecifierSet()
        out._specs = frozenset(self._specs | other._specs)
        return out
"""

# packaging 25.0's markers, as far as these tests read them: a marker hashes as
# (class name, str) and compares by str (26.0 and later hash str alone).
_MARKERS_25 = """
class InvalidMarker(ValueError):
    pass


class UndefinedComparison(ValueError):
    pass


class UndefinedEnvironmentName(ValueError):
    pass


class Marker:
    def __init__(self, marker):
        self._text = marker

    def __str__(self):
        return self._text

    def __hash__(self):
        return hash((self.__class__.__name__, str(self)))

    def __eq__(self, other):
        if not isinstance(other, Marker):
            return NotImplemented
        return str(self) == str(other)
"""

# packaging 26.2's markers module, as far as pyroid reads it: its exceptions.
_MARKERS_262 = '''
class InvalidMarker(ValueError):
    pass


class UndefinedComparison(ValueError):
    pass


class UndefinedEnvironmentName(ValueError):
    """Raised when evaluating a marker that references a missing environment key."""
'''


def _release(root: Path, version: str, files: dict[str, str]) -> Path:
    pkg = root / "packaging"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(f"__version__ = {version!r}\n")
    for name, text in files.items():
        (pkg / name).write_text(textwrap.dedent(text))
    return root


def _run(root: Path, code: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(root), env.get("PYTHONPATH", "")])
    env.pop("PYROID_PATCH", None)
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        cwd=root,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    return out


def test_versions_mix_with_a_release_using_the_old_key(tmp_path: Path) -> None:
    """packaging 26.0 and earlier: the key holds Infinity objects; pyroid builds the same."""
    root = _release(tmp_path, "25.0", {"_structures.py": _INFINITY, "version.py": _VERSION_25})
    out = _run(
        root,
        """
        import itertools, warnings
        warnings.simplefilter("error")
        import packaging.version as pv
        from pyroid.packaging import _compat
        from pyroid.packaging.version import Version
        assert _compat.KEY_FORMAT == "legacy", _compat.KEY_FORMAT
        texts = ["1.0", "1.0.0", "1", "1.0a1", "1.0b2.post3", "1.0rc1.dev4", "1.0.dev5", "0.9",
                 "1.0.post6", "1.0.post7.dev8", "1!0.5", "1.0+abc.5", "1.0+5.abc", "1.0+ABC",
                 "2.0", "1.0a1.dev1", "1.0.dev1+x", "1." + "9" * 25]
        ours = [Version(t) for t in texts]
        theirs = [pv.Version(t) for t in texts]
        for a, b in itertools.product(ours, theirs):
            want = pv.Version(str(a))
            assert (a == b) == (want == b) == (b == a), (a, b)
            assert (a < b) == (want < b) == (b > a), (a, b)
            assert (a <= b) == (want <= b) == (b >= a), (a, b)
            if a == b:
                assert hash(a) == hash(b), (a, b)
            assert a._key == want._key
        assert len(set(ours) | set(theirs)) == len(set(theirs))
        assert [str(v) for v in sorted(ours + theirs)] == [
            str(v) for v in sorted(pv.Version(str(v)) for v in ours + theirs)
        ]
        assert not isinstance(Version("1.0"), pv.Version)  # no claim before 26
        print("ok")
        """,
    )
    assert out.stdout.strip() == "ok"


def test_specifier_sets_hash_as_a_release_before_261(tmp_path: Path) -> None:
    root = _release(
        tmp_path,
        "26.0",
        {"_structures.py": _INFINITY, "version.py": _VERSION_25, "specifiers.py": _SPECIFIERS_260},
    )
    out = _run(
        root,
        """
        import packaging.specifiers as ps
        from pyroid.packaging.specifiers import Specifier, SpecifierSet
        for text in (">=1.5,<2", "<2,>=1.5", ">=1.5", "", "~=1.4.2,!=1.5"):
            mine, theirs = SpecifierSet(text), ps.SpecifierSet(text)
            assert mine == theirs and theirs == mine, text
            assert hash(mine) == hash(theirs), text
            assert len({mine, theirs}) == 1, text
        assert SpecifierSet(">=1.5") != ps.SpecifierSet(">=1.6")
        assert SpecifierSet(">=1.5,<2") == SpecifierSet("<2,>=1.5")
        assert hash(SpecifierSet(">=1.5,<2")) == hash(SpecifierSet("<2,>=1.5"))
        assert Specifier(">=1.5") == ps.Specifier(">=1.5")
        assert hash(Specifier(">=1.5")) == hash(ps.Specifier(">=1.5"))
        # 26.0 is claimed: its own code reads pyroid's _specs as its frozenset.
        assert isinstance(SpecifierSet(">=1"), ps.SpecifierSet)
        both = ps.SpecifierSet(">=1.5") & SpecifierSet("<2")
        assert both == SpecifierSet(">=1.5,<2") == both
        print("ok")
        """,
    )
    assert out.stdout.strip() == "ok"


def test_markers_hash_as_a_release_before_26(tmp_path: Path) -> None:
    root = _release(
        tmp_path,
        "25.0",
        {"_structures.py": _INFINITY, "version.py": _VERSION_25, "markers.py": _MARKERS_25},
    )
    out = _run(
        root,
        """
        import packaging.markers as pm
        from pyroid.packaging.markers import Marker
        mine, theirs = Marker("os_name=='a'"), pm.Marker('os_name == "a"')
        assert mine == theirs and theirs == mine
        assert hash(mine) == hash(theirs) and len({mine, theirs}) == 1
        assert hash(Marker("os_name == 'a'")) == hash(mine)
        print("ok")
        """,
    )
    assert out.stdout.strip() == "ok"


def test_a_release_with_an_unknown_key_is_kept_apart(tmp_path: Path) -> None:
    root = _release(tmp_path, "26.9", {"_structures.py": _INFINITY, "version.py": _VERSION_ODD})
    out = _run(
        root,
        """
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import packaging.version as pv
            from pyroid.packaging import _compat
            from pyroid.packaging.version import Version
            from pyroid.packaging.specifiers import SpecifierSet
        messages = [str(w.message) for w in caught if w.category is RuntimeWarning]
        assert _compat.KEY_FORMAT == "apart"
        assert len(messages) == 1 and "packaging 26.9" in messages[0], messages
        mine, theirs = Version("1.0"), pv.Version("1.0")
        assert mine != theirs and not (mine == theirs) and len({mine, theirs}) == 2
        try:
            mine < theirs
        except TypeError:
            pass
        else:
            raise AssertionError("compared")
        assert not isinstance(mine, pv.Version)  # no claim: packaging's code is not fooled
        assert Version("1.0") == Version("1.0.0") and Version("1.0") < Version("2")
        assert SpecifierSet(">=1").contains(theirs)  # read by value
        print("ok")
        """,
    )
    assert out.stdout.strip() == "ok"


def test_a_missing_key_raises_a_keyerror_with_a_release_before_263(tmp_path: Path) -> None:
    """packaging 26.2 raises a plain KeyError; 26.3 its UndefinedEnvironmentName (a KeyError)."""
    root = _release(
        tmp_path,
        "26.2",
        {"_structures.py": _INFINITY, "version.py": _VERSION_25, "markers.py": _MARKERS_262},
    )
    out = _run(
        root,
        """
        import packaging.markers as pm
        from pyroid.packaging.markers import Marker, UndefinedEnvironmentName
        assert issubclass(UndefinedEnvironmentName, KeyError)
        assert issubclass(UndefinedEnvironmentName, pm.UndefinedEnvironmentName)
        assert UndefinedEnvironmentName.__module__ == "pyroid.packaging.markers"
        for catch in (KeyError, pm.UndefinedEnvironmentName, UndefinedEnvironmentName):
            try:
                Marker('extra == "x"').evaluate({}, "requirement")
            except catch as e:
                assert str(e) == "'extra'", str(e)
            else:
                raise AssertionError("no error")
        print("ok")
        """,
    )
    assert out.stdout.strip() == "ok"


def test_the_patch_warning_lists_only_the_installed_modules(tmp_path: Path) -> None:
    """packaging 24.2 has no pylock, ranges or dependency_groups."""
    root = _release(tmp_path, "24.2", {"_structures.py": _INFINITY, "version.py": _VERSION_25})
    (root / "packaging" / "metadata.py").write_text("")
    (root / "packaging" / "tags.py").write_text("")
    out = _run(
        root,
        """
        import warnings
        import pyroid.integrations
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pyroid.integrations.patch("packaging")
        text = " ".join(str(w.message) for w in caught)
        assert "(tags, metadata; they now use pyroid's classes)" in text, text
        for name in ("pylock", "ranges", "dependency groups"):
            assert name not in text, text
        print("ok")
        """,
    )
    assert out.stdout.strip() == "ok"
