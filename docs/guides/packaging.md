# `pyroid.packaging`: drop-in packaging

`pyroid.packaging` is a Rust port of [packaging](https://github.com/pypa/packaging) 26.3
(Apache-2.0 OR BSD-2-Clause), the PyPA library for PEP 440 versions and specifiers and PEP 508
markers and requirements that pip, setuptools and hatchling use. Its version parser, specifier
pattern and range engine, marker and requirement parsers and name helpers are ported function by
function, so pyroid gives the same parsing, ordering, hashes, strings and exceptions as packaging
26.3 ([parity](../compatibility.md#packaging)). Its objects mix with packaging's: they compare
and hash equal with any packaging release from 24.2 to 26.3, and packaging 26's own code accepts
them ([interop](#interop)).

On a dedicated AWS EC2 c7i.4xlarge (Intel Xeon, CPython 3.12), pyroid parses versions 3 to 8×
faster than packaging 26.3, answers `SpecifierSet.contains` 8× faster, filters a project's release
list 17× faster, and parses markers 21× and requirements 6 to 7× faster; `sort_versions` sorts
1,000,000 version strings 13× faster than `sorted(key=Version)`
([benchmarks](../benchmarks.md#pyroidpackaging)).

```console
$ pip install pyroid
```

## Change one import

```diff
-from packaging.version import Version
-from packaging.specifiers import SpecifierSet
-from packaging.requirements import Requirement
+from pyroid.packaging.version import Version
+from pyroid.packaging.specifiers import SpecifierSet
+from pyroid.packaging.requirements import Requirement

 Version("1.0rc1") < Version("1.0")  # True
 SpecifierSet(">=1.0,!=1.5.*").contains("1.6.2")  # True
 req = Requirement("requests[socks]>=2.8; python_version >= '3.9'")
 str(req.specifier), req.marker.evaluate()  # ('>=2.8', True) on Python 3.9 and later
```

Everything else stays as it was: `except InvalidVersion`, `parse(...)`, `sorted(versions,
key=Version)`, `Marker(...).evaluate(environment)`, `canonicalize_name(...)`,
`parse_wheel_filename(...)`. With packaging installed, the exception classes are packaging's own,
so `except packaging.version.InvalidVersion` catches pyroid's errors too. packaging is not
required: without it, pyroid has copies with the same names.

The ported modules and their names:

- `pyroid.packaging.version`: `Version`, `parse`, `InvalidVersion`, `VERSION_PATTERN`,
  `normalize_pre`;
- `pyroid.packaging.specifiers`: `Specifier`, `SpecifierSet`, `InvalidSpecifier`,
  `BaseSpecifier`;
- `pyroid.packaging.markers`: `Marker`, `default_environment`, `InvalidMarker`,
  `UndefinedComparison`, `UndefinedEnvironmentName`;
- `pyroid.packaging.requirements`: `Requirement`, `InvalidRequirement`;
- `pyroid.packaging.utils`: `canonicalize_name`, `canonicalize_version`, `is_normalized_name`,
  `parse_wheel_filename`, `parse_sdist_filename`, `InvalidName`, `InvalidWheelFilename`,
  `InvalidSdistFilename`.

packaging's `tags`, `metadata`, `pylock`, `ranges`, `dependency_groups`, `direct_url` and
`licenses` are not ported; use packaging's. Under the patch they run on pyroid's classes.

To make code you don't own use pyroid, patch it:

```python
import pyroid

pyroid.patch("packaging")  # from packaging.version import Version gives pyroid's class
```

`pyroid run -p packaging script.py` does the same for a whole program and the processes it starts
([pyroid.patch()](patch.md)). The details are [below](#patch).

## What is the same

Versions parse exactly where packaging's pattern matches, with surrounding whitespace, a leading
`v`, the `.`, `_` and `-` separators, every letter spelling (`alpha`, `c`, `pre`, `preview`,
`rev`, `r`), the implicit post-release `1.0-1` and any case. `str`, `repr`, the properties
(`epoch`, `release`, `pre`, `post`, `dev`, `local`, `public`, `base_version`, `is_prerelease`,
`major`, ...), `from_parts`, `__replace__`, the order and the hash are packaging's. A run of more
than 19 digits goes through `int()`, so the interpreter's limit on integer string conversion
(`sys.get_int_max_str_digits()`) raises the same `ValueError`.

Specifiers follow packaging 26.3's pattern (its case folding included) and its range engine:
`contains` and `in`, `filter` with 26.3's PEP 440 pre-release rule ([Versions](#versions) has
what changed in 26.0), `prereleases`, `&`, `is_unsatisfiable`, `str` and `hash`. `filter`
returns a lazy iterator, like packaging's generators. packaging's quirks come along: a
`SpecifierSet` drops duplicate specifiers and reorders the rest the first time it is printed,
hashed or compared, so its `len()` can change.

Markers are tokenized and parsed with packaging's rules, so an invalid marker raises
`InvalidMarker` with packaging's message and the same `~~~^` span under the source.
`evaluate()` takes the same `environment` and `context` (`"metadata"`, `"lock_file"`,
`"requirement"`), normalises extra names, and raises `UndefinedComparison` and
`UndefinedEnvironmentName` where packaging does. `default_environment()` returns a new copy of
the environment computed once per process.

`Requirement` is a Python class, copied from packaging's, over the native parser: `name`,
`url`, `extras`, `specifier` and `marker` are plain attributes you can set, and `str`, `==` and
`hash` are packaging's. The `utils` functions return what packaging's return; the wheel tags are
packaging's own `Tag` objects when packaging is installed, else copies.

Errors have packaging's classes and messages:

```python
>>> Version("1.0.0-final")
packaging.version.InvalidVersion: Invalid version: '1.0.0-final'
```

## Mixing pyroid's and packaging's objects {#interop}

pyroid's objects meet packaging's in your code, in packaging's own modules (`ranges`, `metadata`,
`pylock`) and in libraries that imported packaging first. These rules hold:

1. They compare, hash and deduplicate with packaging's. `<` works in either order, so sets, dict
   keys and sorted lists can mix the two. packaging orders versions by a comparison key
   (`Version._key`) whose format changed in 26.1, and pyroid builds it in the installed release's
   format: tuples of ints for 26.1 and later, a tuple holding packaging's `Infinity` objects for
   26.0 and earlier. At import pyroid checks its keys and hashes against the installed release's
   on versions that take every branch of the key. With a release whose key matches neither
   format, pyroid's objects and packaging's do not compare with each other at all (`==` is
   false, `<` raises `TypeError`) and pyroid warns once, with a `RuntimeWarning`.
2. With a packaging 26 release installed and not patched, pyroid's objects, and those of your
   subclasses of pyroid's classes, report packaging's class as `__class__`.
   `isinstance(v, packaging.version.Version)` is then true, and packaging's own code accepts
   them: `packaging.specifiers.SpecifierSet(">=1").contains(v)` would otherwise be `False`.
   `type(v)` stays your class; in a subclass, make new objects with `type(self)(...)`, since
   `self.__class__(...)` makes one of packaging's.
3. They have the private attributes that packaging's modules read (`Version._key`, `_epoch`,
   `_release`, ..., `Specifier._spec`, `SpecifierSet._specs`, `Marker._markers` and others),
   and packaging's class attributes `Version._regex` and `Specifier._operators`.
4. A `SpecifierSet` takes packaging's `Specifier` objects as items and `&` takes packaging's
   `SpecifierSet`, of any release; markers combine with packaging's markers.
5. With packaging installed, the exception classes are packaging's own. Without it, pyroid has
   copies with the same names and bases. Before 26.3 a missing marker environment key raises
   `KeyError` (the release's `UndefinedEnvironmentName` exists but is not raised): with one of
   those releases, pyroid raises a subclass of both, so either `except` catches it.
6. They pickle through `pyroid.packaging._restore` with packaging 26.3's state tuples and
   unpickle as pyroid's objects, so loading them needs pyroid. `__setstate__` accepts every state
   format packaging writes, so under the patch packaging's own pickles load into pyroid's
   classes.
7. Subclasses work, including `super().__init__(version)`. `Requirement` is a Python class with
   writable attributes, so pkg_resources-style subclasses work. Assigning `Version._version` (as
   hatchling does) works, with packaging's `DeprecationWarning`.

```python
import packaging.specifiers
import packaging.version

from pyroid.packaging.version import Version

v = Version("2.0.post1")
v == packaging.version.Version("2.0.post1")  # True
len({v, packaging.version.Version("2.0.post1")})  # 1
isinstance(v, packaging.version.Version)  # True (packaging 26 installed, not patched)
type(v)  # <class 'pyroid.packaging.version.Version'>
packaging.specifiers.SpecifierSet(">=2").contains(v)  # True, in packaging's own code
```

## Patching code you don't own {#patch}

```python
import pyroid

pyroid.patch("packaging")

from packaging.requirements import Requirement  # pyroid.packaging.requirements.Requirement
from packaging.version import Version  # pyroid.packaging.version.Version
```

`pyroid.patch("packaging")` imports packaging and its submodules. Then, in every loaded
`packaging.*` module, it replaces each global that is one of packaging's ported objects:
`Version`, `parse`, `Specifier`, `SpecifierSet`, `Marker`, `default_environment` (and its cache),
`Requirement`, `canonicalize_name`, `canonicalize_version`, `is_normalized_name`,
`parse_wheel_filename` and `parse_sdist_filename`. So imports made after the patch get pyroid's
classes, and so do packaging's unported modules (`ranges`, `metadata`, `pylock`,
`dependency_groups`), which hold their own references to these names. While patched, packaging's
names are pyroid's classes, so the `__class__` claim of rule 2 is off. `unpatch("packaging")`
puts every global back. Without packaging installed, `packaging` and its `version`,
`specifiers`, `markers`, `requirements` and `utils` submodules are shims holding
pyroid.packaging's names.

Code that imported packaging's classes before the patch keeps them, and objects made before the
patch stay packaging's. They keep working with each other and with pyroid's: they compare and
hash equal, pyroid's constructors take them by value while patched (so packaging's own code,
which converts what `isinstance` does not recognise, accepts them), and pickling one while
patched writes the pyroid object it equals. `isinstance(old_object, packaging.version.Version)` is
false once that name is pyroid's class. Patch before importing the libraries you want redirected,
or use `pyroid run -p packaging`. pip uses its own vendored copy (`pip._vendor.packaging`), which
the patch does not touch.

Importing pyroid never patches anything (pyroid's own build tooling uses packaging). With a
packaging release other than 26.3 installed, `import pyroid.packaging` and the patch give one
[`VersionMismatchWarning`](../compatibility.md#installed-versions): the patched names give 26.3's
results, and the rest is the installed release's ([Versions](#versions)).

## Many versions at once {#batch}

`parse_batch`, `sort_versions`, `contains_batch` and `filter_batch` are pyroid's additions. Each
gives what the one-at-a-time calls give.

```python
import pyroid.packaging as pp

releases = ["1.0", "2.0rc1", "1.10", "1.9.post1"]
pp.parse_batch([*releases, "nightly"], errors="none")
# [<Version('1.0')>, <Version('2.0rc1')>, <Version('1.10')>, <Version('1.9.post1')>, None]
pp.sort_versions(releases)  # ['1.0', '1.9.post1', '1.10', '2.0rc1']
pp.contains_batch(">=1.5", releases)  # [False, True, True, True]
pp.filter_batch(">=1.5", releases)  # ['1.10', '1.9.post1']
```

- `parse_batch(versions, *, errors="raise", threads=None)` returns one `Version` per item. With
  `errors="raise"` the first invalid version raises `InvalidVersion`, with a note naming its
  index; with `errors="none"` it is `None`.
- `sort_versions(versions, *, reverse=False, threads=None)` returns the objects you passed in the
  order of `sorted(versions, key=Version)`. It is stable, and the items can be strings or
  `Version` objects, pyroid's or packaging's.
- `contains_batch(specifiers, versions, *, prereleases=None, installed=None, threads=None)` is
  `[specifiers.contains(v, prereleases, installed) for v in versions]`: a list of bools, or a
  tuple for a tuple.
- `filter_batch(specifiers, versions, *, prereleases=None, threads=None)` is
  `list(specifiers.filter(versions, prereleases))`.

`specifiers` is a `SpecifierSet`, a `Specifier` or a specifier string. `versions` is a list or
tuple, or an Arrow string array ([inputs](../concepts/inputs.md)). For Arrow input,
`contains_batch` returns a [`pyroid.Column`](../reference/pyroid.md#pyroid.Column) of bools with
nulls kept, `parse_batch` gives `None` for a null, and `sort_versions` and `filter_batch` return
new `str` objects. The input is read and parsed 65,536 items at a time, on pyroid's shared thread
pool (a smaller batch on the calling thread), so `filter_batch` and `contains_batch` hold one
chunk of parsed versions at most. Ctrl-C is checked between chunks, and `sort_versions` sorts
runs of that size and merges them, checking between steps. `threads=n` runs at most n tasks at
once.

## What runs in Rust

These run in Rust: version parsing, ordering, hashing and `str`; the specifier pattern and
26.3's range engine (`contains`, `filter`, `&`, `is_unsatisfiable`); the marker and requirement
tokenizer and parser, with their messages and spans; marker evaluation; `canonicalize_name` and
`canonicalize_version`. Python runs only where packaging's result is Python's:
`ast.literal_eval` for a quoted marker string that holds a backslash, a line break or a NUL,
`repr()` in messages, Python's operators for marker environment values that are not `str` or
sets, and `int()` for digit runs longer than 19 characters.

## Fallbacks {#fallbacks}

`SpecifierSet.to_range()`, `is_subset()`, `is_superset()` and `is_disjoint()` run packaging
26.3's own `VersionRange` code on pyroid's objects, with one
[`pyroid.FallbackWarning`](../concepts/fallbacks.md) per function, at the caller's line. A test
suite that turns warnings into errors (`-W error`, or pytest's `filterwarnings = ["error"]`) needs
`-W ignore::pyroid.FallbackWarning` for these. Without packaging they raise
[`pyroid.PyroidError`](../reference/pyroid.md#pyroid.PyroidError).

A `SpecifierSet` built from items that are not `Specifier` objects (setuptools passes strings)
fails in `contains` and `filter` with the same `AttributeError` as packaging's.

## Versions

pyroid.packaging reproduces packaging 26.3, the newest release. A probe of 32,316 observations
(3,043 version strings; 40 specifiers against 106 candidates with each `prereleases` setting; 37
markers in 9 environments and 3 contexts; 35 requirements; names, wheel and sdist filenames; the
sorted order), recorded with each release from 24.2 to 26.3, found that versions parse, print and
sort the same in all of them. The rest differs:

| Release | Observations that differ from 26.3 | What changed since |
|---|---:|---|
| 24.2 | 7,486 | as 25.0, and `Marker.evaluate` has no `context=` and no `extras` or `dependency_groups` variables |
| 25.0 | 7,102 | 26.0's PEP 440 pre-release rule: `SpecifierSet(">=1.0").contains("1.5a1")` was `False` and is `True`; `filter` returns pre-releases when nothing else matches; an invalid version in `contains` or `filter` raised `InvalidVersion` and now gives `False` or is skipped; `===foo` raised `InvalidVersion`; and the rows below |
| 26.0 | 322 | `>V.devN` and `<V.postN`, which 26.1 narrowed (`SpecifierSet(">1.0.dev1").contains("1.0.post1")` was `False`); and the rows below |
| 26.1, 26.2 | 171 each | 26.3's marker and requirement fixes: `UndefinedEnvironmentName`, a `KeyError` subclass, instead of a plain `KeyError`; `"X_Y" in extras`, and an `extra` literal inside parentheses, normalised at parse time; a trailing newline rejected; "Invalid quoted string" instead of a leaked `SyntaxError`; the `.*` suffix message; `is_normalized_name("a--b")` is `False`; wheel and sdist filenames with an empty name or bad tags rejected |

pyroid reproduces 26.3 only. The 25.0 differences are real (a resolver built on 25.0 sees another
pre-release policy), but 26.0 made them on purpose to follow PEP 440, and 26.3 is what a new install
gets. Supporting 25.0 as well would take a second specifier engine and a second set of error
messages. With another release installed, `import pyroid.packaging` and `pyroid.patch("packaging")`
say so once with a `VersionMismatchWarning`, and a release whose comparison key pyroid does not
reproduce gets a `RuntimeWarning` too (rule 1 [above](#interop)). With a release older than 26
installed, pyroid's objects do not claim packaging's class (rule 2), because those releases' private
attributes differ; they still compare and hash with its objects.

## Differences from packaging {#differences-from-packaging}

- CPython's own messages about pyroid's objects name the class with its module:
  `'pyroid.packaging.version.Version' object is not iterable`, and `'<' not supported between
  instances of 'pyroid.packaging.version.Version' and 'int'`, where packaging's say `'Version'`.
  The messages pyroid writes itself are packaging's.
- Pickles of pyroid's objects need pyroid to load.
- `Version.from_parts` and `__replace__` given `True` or `False` for a number store `1` or `0`.
  packaging keeps the bool and prints, for example, `True!1.0`.
- Calling `__init__` again on an existing `Version`, `Specifier`, `SpecifierSet` or `Marker`
  keeps the first value, where packaging parses the new one in place. Assigning `_version` and
  calling `__setstate__` do replace it.
- A `str` subclass that overrides `split` or `lower` is read as its string data, except where a
  version is parsed (`Version()`, `parse`, a specifier's `contains` and `filter`, the batch
  functions): there `__iter__` and `split` overrides count, as in packaging.
- A marker string literal whose escapes produce a lone surrogate (`'\ud800'`) raises
  `InvalidMarker` ("Invalid quoted string"). packaging accepts it.
- Markers nest as deep as memory allows. packaging parses, evaluates and prints them by
  recursion, so it raises `RecursionError` past a few hundred levels of parentheses (fewer in a
  thread with a small stack), or for `str()` or `evaluate()` after `m = m & other` has run some
  500 times. pyroid gives the result, and for a string of only `(` the `InvalidMarker` the
  grammar calls for. `Marker._markers` builds a new list each time it is read.
- `filter()` returns an iterator, not a generator object.
- Unpatched, `functools.singledispatch` dispatches pyroid's objects as packaging's, since it
  looks at `__class__`, so a handler registered for pyroid's own class is not found. Under the
  patch the two are the same class.

## Parity

packaging 26.3's own tests of versions, specifiers, markers, requirements, utils, ranges,
metadata, pylock and dependency groups, with its property tests, pass unchanged with
`pyroid.patch("packaging")` on: 62,499 tests, so packaging's unported modules are exercised on
pyroid's classes too. They run with packaging's pytest settings, which turn warnings into errors,
plus `-W ignore::pyroid.FallbackWarning` for the notice `to_range()` gives
([fallbacks](#fallbacks)). A Hypothesis differential fuzzer builds versions, specifiers, markers,
requirements, names and filenames from the PEP 440 and PEP 508 grammars, plus mutations (Python
whitespace, the letters `re.IGNORECASE` folds onto ASCII, other digits, lone surrogates), and
compares each result or exception (class and message) with packaging 26.3's. Its long run of
180,000 cases found no differences. Details: [Compatibility](../compatibility.md#packaging).

The same comparison ran on PyPI's metadata for the top 9,996 projects (their JSON API, fetched
on 2026-11-16): all 828,241 release versions (parse, `str`, properties, key, hash,
`canonicalize_version`, each project's sort order), the 25,930 distinct `Requires-Python` values
(parsed, and `contains` of 61 CPython versions with each `prereleases` setting), and 69,162
distinct `Requires-Dist` values (parsed and printed, `==` and `hash` against packaging's objects,
each marker in 4 environments, 3 extras and 3 contexts, each specifier filtering the dependency's
own release list). That is 8,725,233 comparisons, with no differences
(`bench/packaging/pypi_corpus.py`).

API: [`pyroid.packaging` reference](../reference/packaging.md).
