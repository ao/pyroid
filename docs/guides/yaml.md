# `pyroid.yaml`: drop-in PyYAML

`pyroid.yaml` is a Rust port of [PyYAML](https://github.com/yaml/pyyaml) 6.0.3 (MIT), the
library behind `yaml.safe_load` and `yaml.safe_dump`. It reproduces PyYAML's pure-Python
`SafeLoader` and `SafeDumper` method by method: the same Python objects from the same YAML, the
same text from the same objects for every dump option, and the same exceptions with the same
messages and marks ([parity](../compatibility.md#yaml)).

`yaml.safe_load` uses the pure-Python loader unless you pass `Loader=yaml.CSafeLoader`, and
`CSafeLoader` gives different results and errors on some inputs. pyroid follows the pure loader,
and is faster than either.

```console
$ pip install pyroid
```

## Change one import

```diff
-import yaml
+import pyroid.yaml as yaml

 config = yaml.safe_load(open("config.yaml"))
 docs = list(yaml.safe_load_all(text))
 yaml.safe_dump(config, sort_keys=False)
 yaml.load(f, Loader=yaml.SafeLoader)
```

Everything else stays as it was: `except yaml.YAMLError`, `yaml.SafeLoader`, `yaml.Dumper`,
`yaml.scan`, `yaml.add_constructor`. With PyYAML installed, those names *are* PyYAML's own
objects (pyroid.yaml forwards the names it does not define), so errors from pyroid and from
PyYAML are the same classes.

PyYAML is not required. Without it, pyroid.yaml has its own copies of the exception and `Mark`
classes, and its loader and dumper classes are markers: they can be passed to `load()` and
`dump()` and subclassed, but a registration on them or on a subclass (`add_constructor`,
`add_multi_constructor`, `add_representer`, `add_multi_representer`, `add_implicit_resolver`,
`add_path_resolver`) raises `pyroid.PyroidError` with the `pip install pyyaml==6.0.3` line. The
names it does not port, on the module or on those classes (`yaml.scan`, `yaml.add_constructor`,
`SafeLoader.yaml_constructors`), raise `AttributeError`.

To make code you don't own use pyroid, patch it:

```python
import pyroid

pyroid.patch("yaml")  # yaml.safe_load, yaml.load, yaml.dump, ... are pyroid.yaml's
```

`pyroid run -p yaml script.py` does the same for a whole program and the processes it starts
([pyroid.patch()](patch.md)).

## What is the same

Loading gives PyYAML's YAML 1.1 types, as `SafeConstructor` builds them: `yes`/`no`/`on`/`off`
booleans, `0x`, `0b`, `0`-octal and `1:20` (sexagesimal) ints, `.inf` and `.nan` (PyYAML's own
shared `nan` object, `SafeConstructor.nan_value`, so containers holding one compare equal to
PyYAML's; one shared object of pyroid's without PyYAML), timestamps as `date` and `datetime` with `tzinfo`, `!!binary` as
`bytes`, `!!set`, `!!omap` and `!!pairs`, merge keys (`<<`), duplicate keys (the last value wins,
in the first key's place), and aliases as shared objects (recursive ones included). An unusual
explicitly tagged value, like `!!int " 0x1F "`, `!!float` with Arabic-Indic digits or
`!!bool maybe`, gives PyYAML's result or its exception (`KeyError`, `ValueError`, ...), because
pyroid runs PyYAML's own code for the scalar forms it does not handle natively.

Input can be a `str`, `bytes` (UTF-8, or UTF-16 with a BOM, as PyYAML's reader decodes it) or a
file object. A file object is read as PyYAML reads it: 4096 units at a time, and only as far as
the parser has got. So `safe_load_all` on a pipe or socket yields each document once the next
one starts, memory stays at about one document, and a bad byte or a failed `read()` further on
is raised where PyYAML raises it: the documents before it still load, and a YAML error earlier
in the stream wins.

Errors have PyYAML's classes (`ScannerError`, `ParserError`, `ComposerError`,
`ConstructorError`, `ReaderError`, ...), messages, `context`/`problem` and marks (`name`,
`line`, `column`, `index`, and `buffer`/`pointer` for the snippet), so `str(error)` is the same
text:

```python
>>> yaml.safe_load("a: b: c")
yaml.scanner.ScannerError: mapping values are not allowed here
  in "<unicode string>", line 1, column 5:
    a: b: c
        ^
```

Without PyYAML installed, the class is pyroid's copy, shown as `pyroid.yaml.ScannerError`, with
the same message.

Dumping gives the same text byte for byte, for every option: `default_flow_style`,
`default_style`, `sort_keys`, `indent`, `width` (including `float("inf")`), `allow_unicode`,
`line_break`, `canonical`, `explicit_start`, `explicit_end`, `version`, `tags` and `encoding`.
Strings get PyYAML's style and quoting, floats PyYAML's text (`1.0e+17`), and objects that
appear twice get `&id001` anchors and `*id001` aliases. With `encoding="utf-16"` the output has a
BOM before each piece PyYAML writes, as Python's codec gives PyYAML.

## What runs in Rust, and what goes to PyYAML

| Call | Runs |
|---|---|
| `safe_load`, `safe_load_all` | in Rust |
| `load`/`load_all` with `SafeLoader`, `CSafeLoader` (with `SafeLoader`'s results), or a subclass that only subclasses | in Rust |
| `load` with `FullLoader`, `Loader`, `UnsafeLoader`; `full_load`, `unsafe_load` | in Rust, until a `!!python/` tag appears; then PyYAML loads the document |
| `safe_dump`, `safe_dump_all`; `dump`/`dump_all` with `SafeDumper` or `CSafeDumper` (with `SafeDumper`'s text) | in Rust |
| `dump`/`dump_all` with `Dumper` (the default) | in Rust for plain data (dicts, lists, tuples as `!!python/tuple`, sets, `str`, `bytes`, numbers, booleans, `None`, dates); other objects go to PyYAML |
| A loader or dumper with `add_constructor`, `add_representer`, `add_implicit_resolver` or `add_path_resolver` registrations, or overriding a method (`construct_mapping`, `ignore_aliases`, `increase_indent`, ...) | PyYAML |
| `BaseLoader`, `load_all` with the full loaders | PyYAML |
| `scan`, `parse`, `compose`, `emit`, `serialize`, `YAMLObject`, `add_constructor` and the other names | PyYAML's own (not ported) |

Calls that go to PyYAML return PyYAML's result and emit one [`pyroid.FallbackWarning`](../concepts/fallbacks.md)
per function and reason. Without PyYAML they raise `pyroid.PyroidError` instead. pyroid checks
the loader or dumper class on every call, so a registration made after import is seen. It does
not see methods or attributes (such as `nan_value`) changed on PyYAML's own base classes
(`yaml.constructor.SafeConstructor` and the like), which is rare.

A program that changes `SafeLoader` itself sends every call to PyYAML: yamllint, for one, replaces
`SafeLoader`'s implicit resolvers when it is imported, so in a yamllint process every load falls
back (the results are the same, without the speed-up).

In a sample of public Python code (Sourcegraph's index, November 2026), calls with customised
safe loaders and dumpers are about 1 to 2% of `safe_load`, `load` and `dump` calls.

## Speed

On 11,878 real YAML files (343 MB: GitHub Actions workflows, Kubernetes and Helm files, OpenAPI
specs, model and dataset cards), `safe_load` runs 10x faster than PyYAML with `CSafeLoader` and
44x faster than the default `SafeLoader`, and `safe_dump` 5.4x faster than `CSafeDumper` and 21x
faster than `SafeDumper`. On single files of 1 to 100 MB, loading is 5.1x to 7.7x faster than
`CSafeLoader`. A 300-byte document loads in 19 µs (`CSafeLoader`: 105 µs). Measured on a
dedicated AWS EC2 c7i.4xlarge, CPython 3.12; the [benchmarks page](../benchmarks.md#pyroidyaml)
has the tables.

`safe_load` releases the GIL while it parses a document of 4 KiB or more; building the Python
objects needs the interpreter. pyroid.yaml runs on free-threaded CPython without enabling the GIL.

## Versions {#versions}

pyroid.yaml reproduces PyYAML 6.0.3 (September 2025), the newest release. PyYAML 6.0 (October
2021), 6.0.1 and 6.0.2 have the same Python code apart from `__version__`: they changed only how
PyYAML is built and which wheels it ships. So pyroid gives the same results with any of the four
installed, and does not warn about them.

PyYAML 5.x (5.1 to 5.4.1, the last of them from January 2021) differs from 6.0 in what pyroid
replaces:

- `yaml.load(stream)` and `yaml.load_all(stream)` work without a `Loader`: they use `FullLoader`
  and emit a `YAMLLoadWarning`. Since 6.0 the argument is required.
- The float resolver also matches plain scalars that start with `._`: `._1` loads as `0.1` (and
  `._` raises `ValueError`), and the string `"._1"` is dumped in quotes.
- Two scanner error messages misspell "hexadecimal".
- 5.1 and 5.2 load a timestamp with an offset as a naive `datetime` converted to UTC; 5.3 and
  later give it a `tzinfo`.

`pyroid.patch("yaml")` refuses PyYAML 5.x: it raises `ImportError` naming the release it needs
and changes nothing, and `pyroid run -p yaml` stops with that error. A program written for 5.x
keeps its own PyYAML then, instead of a `TypeError` from `yaml.load(stream)` or different results
for the inputs above (accepting 5.x's call forms alone would still give 6.0's results). Upgrade with
`pip install pyyaml==6.0.3`. `import pyroid.yaml` works with 5.x installed: its functions take
6.0's arguments and give 6.0.3's results, it warns once that results can differ, and the classes it
forwards and the calls it hands to PyYAML are 5.x's. PyYAML 5.x has no wheels for Python 3.10 or
later, and its source distribution builds only with `cython<3` and `--no-build-isolation`, so it is
rare on the Pythons pyroid supports.

Without PyYAML, pyroid.yaml gives 6.0.3's results, with its own copies of the exception and `Mark`
classes ([above](#change-one-import)).

## Differences from PyYAML {#differences-from-pyyaml}

- Deep nesting: PyYAML's composer and representer are recursive, so it raises `RecursionError`
  at a depth that depends on `sys.getrecursionlimit()` and on how deep the caller's stack is.
  pyroid computes the same limit from both and raises `RecursionError` at that nesting depth,
  within three levels of PyYAML's (about 490 levels from a shallow stack at the default limit;
  dumping, about 328). pyroid itself does not recurse, so deep nesting cannot overflow the
  native stack, whatever the recursion limit or the thread's stack size.
- A stream given to `dump` gets the whole text in one `write()` and one `flush()` at the end.
  PyYAML makes many small writes and flushes after each document and at the end; pyroid makes
  PyYAML's writes only for codecs that need them, such as `utf-16`.
- Dumping a `str` with lone surrogates goes to PyYAML.
- Exceptions raised from pyroid have no `__context__` (PyYAML raises some of them while
  handling another exception).

## Differences between `CSafeLoader` and `SafeLoader`

pyroid follows the pure-Python `SafeLoader` and `SafeDumper`. Code that passed
`Loader=yaml.CSafeLoader` or `Dumper=yaml.CSafeDumper` gets their pure-Python counterparts'
results from pyroid. On the corpus and test suites used for parity (below), libyaml's classes
differ from the pure-Python ones in these ways:

- Errors: libyaml's marks carry no snippet, so `str(error)` lacks the source line and caret for
  every error; many scanner and parser messages are worded differently ("mapping values are not
  allowed in this context" for "... allowed here", "did not find expected node content" for
  "expected the node content, but found '-'", "invalid leading UTF-8 octet" for "invalid start
  byte"), sometimes at another column; and two of the yaml-test-suite's cases raise another
  class (`ParserError` against `ScannerError`, `ConstructorError` against `ScannerError`).
- Valid and invalid input: libyaml accepts tabs as separators where the pure loader raises
  "found character '\t' that cannot start any token" (2 of the 11,989 corpus files, and 16
  test-suite cases load only with `CSafeLoader`), and rejects some plain scalars the pure loader
  reads (10 test-suite cases load only with `SafeLoader`; libyaml says "found unexpected ':'"
  for some of them).
- Values: a lone `!` tag gives `None` with `SafeLoader` and `""` with `CSafeLoader`.
- Dumping: libyaml folds long double-quoted strings at spaces where PyYAML breaks them with `\`
  continuations (also inside CJK text), and does not end a document that is a top-level plain
  scalar with `...`. 3,738 of the 11,989 corpus files dump differently in at least one of six
  option sets (3,300 with the default options).
