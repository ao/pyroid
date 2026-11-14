# Derived from PyYAML 6.0.3 (MIT), © Kirill Simonov and Ingy döt Net; modified.
"""A drop-in for PyYAML 6.0.3's ``safe_load`` and ``safe_dump``, in Rust.

```python
import pyroid.yaml as yaml          # was: import yaml

config = yaml.safe_load(open("config.yaml"))
docs = list(yaml.safe_load_all(text))
text = yaml.safe_dump(config, sort_keys=False)
data = yaml.load(f, Loader=yaml.SafeLoader)
```

Results, dumped text and errors are those of PyYAML's pure-Python ``SafeLoader`` and
``SafeDumper``: the same objects (YAML 1.1 booleans, sexagesimal and octal ints, timestamps,
``!!binary``, ``!!set``, ``!!omap``, merge keys, aliases as shared objects), the same text
byte for byte for every option, and the same exception classes, messages and marks. With
PyYAML installed the exception, ``Mark``, loader and dumper classes *are* PyYAML's, so
``except yaml.YAMLError`` catches errors from both.

What runs in Rust:

- ``safe_load``, ``safe_load_all``, ``load``/``load_all`` with ``SafeLoader``, ``CSafeLoader``
  (with ``SafeLoader``'s results) or a subclass that adds nothing;
- ``load`` with ``FullLoader``, ``Loader`` or ``UnsafeLoader``, and ``full_load``, until a
  ``!!python/`` tag appears;
- ``safe_dump``, ``safe_dump_all``, ``dump``/``dump_all`` with ``SafeDumper`` (or
  ``CSafeDumper``, with ``SafeDumper``'s text), and with ``Dumper`` for plain data.

Everything else is handed to PyYAML with a one-time :class:`pyroid.FallbackWarning` (or raises
:class:`pyroid.PyroidError` without PyYAML): loaders and dumpers with added constructors,
representers or resolvers or overridden methods, ``BaseLoader``, ``!!python/`` tags, objects
other than plain data under ``Dumper``, and dumping a ``str`` with lone surrogates. Any other
name (``scan``, ``parse``, ``compose``, ``emit``, ``YAMLObject``, ``add_constructor``, ...)
is PyYAML's, through the module's ``__getattr__``.

Input is a ``str``, ``bytes`` (UTF-8, or UTF-16 with a BOM) or a file object, read as
PyYAML reads it, including where a decoding error further on is raised. Deeply nested
documents raise ``RecursionError`` where PyYAML's recursive composer would, within three
levels.
"""

from collections.abc import Iterable, Iterator
from typing import Any, Final

__all__ = [
    "PYYAML_VERSION",
    "BaseLoader",
    "ComposerError",
    "ConstructorError",
    "Dumper",
    "EmitterError",
    "FullLoader",
    "Loader",
    "Mark",
    "MarkedYAMLError",
    "ParserError",
    "ReaderError",
    "RepresenterError",
    "ResolverError",
    "SafeDumper",
    "SafeLoader",
    "ScannerError",
    "SerializerError",
    "UnsafeLoader",
    "YAMLError",
    "dump",
    "dump_all",
    "full_load",
    "full_load_all",
    "load",
    "load_all",
    "safe_dump",
    "safe_dump_all",
    "safe_load",
    "safe_load_all",
    "unsafe_load",
    "unsafe_load_all",
]

#: The PyYAML release whose behaviour this module reproduces.
PYYAML_VERSION: Final[str]
__version__: Final[str]
__with_libyaml__: Final[bool]

class Mark:
    """A position in a YAML stream (PyYAML's ``yaml.error.Mark``).

    ``index``, ``line`` and ``column`` are 0-based and count code points. ``buffer`` and
    ``pointer`` are set for ``str`` and ``bytes`` input (``str()`` then shows the line with a
    caret), ``None`` for file objects.
    """

    name: Any
    index: int
    line: int
    column: int
    buffer: str | None
    pointer: int | None
    def __init__(
        self, name: Any, index: int, line: int, column: int, buffer: str | None, pointer: int | None
    ) -> None: ...
    def get_snippet(self, indent: int = 4, max_length: int = 75) -> str | None: ...

class YAMLError(Exception):
    """Base class of the YAML errors."""

class MarkedYAMLError(YAMLError):
    """An error with a context, a problem and where they are in the stream."""

    context: str | None
    context_mark: Mark | None
    problem: str | None
    problem_mark: Mark | None
    note: str | None
    def __init__(
        self,
        context: str | None = None,
        context_mark: Mark | None = None,
        problem: str | None = None,
        problem_mark: Mark | None = None,
        note: str | None = None,
    ) -> None: ...

class ReaderError(YAMLError):
    """The stream holds bytes that do not decode or characters YAML does not allow."""

    name: Any
    character: Any
    position: int
    encoding: str
    reason: str
    def __init__(
        self, name: Any, position: int, character: Any, encoding: str, reason: str
    ) -> None: ...

class ScannerError(MarkedYAMLError):
    """The scanner could not make a token."""

class ParserError(MarkedYAMLError):
    """The tokens do not form a YAML document."""

class ComposerError(MarkedYAMLError):
    """Anchors, aliases or the document count are wrong."""

class ConstructorError(MarkedYAMLError):
    """A node could not be turned into a Python object."""

class ResolverError(YAMLError):
    """A bad resolver registration."""

class RepresenterError(YAMLError):
    """An object could not be represented as YAML."""

class SerializerError(YAMLError):
    """The serializer was used out of order."""

class EmitterError(YAMLError):
    """The emitter got events it cannot write."""

#: PyYAML's ``SafeLoader`` (a marker for :func:`load` when PyYAML is not installed).
SafeLoader: type[Any]
#: PyYAML's ``BaseLoader`` (always handed to PyYAML).
BaseLoader: type[Any]
#: PyYAML's ``FullLoader``: native until a ``!!python/`` tag appears.
FullLoader: type[Any]
#: PyYAML's ``Loader``: native until a ``!!python/`` tag appears.
Loader: type[Any]
#: PyYAML's ``UnsafeLoader``: native until a ``!!python/`` tag appears.
UnsafeLoader: type[Any]
#: PyYAML's ``SafeDumper``.
SafeDumper: type[Any]
#: PyYAML's ``Dumper``: native for plain data.
Dumper: type[Any]

def load(stream: Any, Loader: Any) -> Any:  # noqa: N803
    """Parse the first YAML document in `stream` and produce the Python object (``yaml.load``).

    Args:
        stream: A ``str``, ``bytes`` (UTF-8, or UTF-16 with a BOM) or a file object.
        Loader: A loader class. ``SafeLoader``, ``CSafeLoader`` and unmodified subclasses run
            natively with ``SafeLoader``'s results; ``FullLoader``, ``Loader`` and
            ``UnsafeLoader`` too until a ``!!python/`` tag appears.

    Raises:
        YAMLError: The stream is not valid YAML, or a node cannot be constructed (a subclass
            such as ``ScannerError`` or ``ConstructorError``, with PyYAML's message and marks).
    """

def load_all(stream: Any, Loader: Any) -> Iterator[Any]:  # noqa: N803
    """Parse all YAML documents in `stream` and produce the Python objects (``yaml.load_all``).

    A generator, as in PyYAML: documents come one at a time, and an error in a later
    document is raised when the generator gets there. Only ``SafeLoader`` (and
    ``CSafeLoader``) run natively here.
    """

def safe_load(stream: Any) -> Any:
    """Parse the first YAML document in `stream`, with only basic tags (``yaml.safe_load``).

    >>> safe_load("a: [1, 2.5, yes, ~, 2001-12-14]")
    {'a': [1, 2.5, True, None, datetime.date(2001, 12, 14)]}
    """

def safe_load_all(stream: Any) -> Iterator[Any]:
    """Parse all YAML documents in `stream` (``yaml.safe_load_all``)."""

def full_load(stream: Any) -> Any:
    """``yaml.full_load``: :func:`load` with ``FullLoader``."""

def full_load_all(stream: Any) -> Iterator[Any]:
    """``yaml.full_load_all``: :func:`load_all` with ``FullLoader``."""

def unsafe_load(stream: Any) -> Any:
    """``yaml.unsafe_load``: :func:`load` with ``UnsafeLoader``."""

def unsafe_load_all(stream: Any) -> Iterator[Any]:
    """``yaml.unsafe_load_all``: :func:`load_all` with ``UnsafeLoader``."""

def dump_all(
    documents: Iterable[Any],
    stream: Any = None,
    Dumper: Any = ...,  # noqa: N803
    default_style: str | None = None,
    default_flow_style: bool | None = False,
    canonical: bool | None = None,
    indent: int | None = None,
    width: float | None = None,
    allow_unicode: bool | None = None,
    line_break: str | None = None,
    encoding: str | None = None,
    explicit_start: bool | None = None,
    explicit_end: bool | None = None,
    version: tuple[int, int] | None = None,
    tags: dict[str, str] | None = None,
    sort_keys: bool = True,
) -> Any:
    r"""Serialize a sequence of Python objects into a YAML stream (``yaml.dump_all``).

    Args:
        documents: The documents, iterated once.
        stream: A file object to write to; ``None`` returns the text (``bytes`` with
            `encoding`). A stream gets the text in one ``write`` (PyYAML writes it in many
            pieces), and what PyYAML would have written before an error.
        Dumper: ``SafeDumper`` or ``CSafeDumper`` (``SafeDumper``'s text), ``Dumper`` (plain
            data natively, tuples as ``!!python/tuple``), or a class that goes to PyYAML.
        default_style: ``None``, ``"'"``, ``'"'``, ``"|"`` or ``">"``.
        default_flow_style: ``False`` (block), ``True`` (flow) or ``None`` (flow for
            collections of plain scalars).
        canonical: Write canonical YAML.
        indent: 2 to 9 (other values give 2, as in PyYAML).
        width: Preferred line width; ``float("inf")`` never wraps.
        allow_unicode: Write non-ASCII characters instead of escapes.
        line_break: ``"\n"``, ``"\r"`` or ``"\r\n"``.
        encoding: Encode the text (as PyYAML, per write: ``"utf-16"`` repeats its BOM).
        explicit_start: Start each document with ``---``.
        explicit_end: End each document with ``...``.
        version: A ``%YAML`` directive, e.g. ``(1, 1)``.
        tags: ``%TAG`` directives, handle to prefix.
        sort_keys: Sort mapping keys (keys of mixed types stay in order).

    Raises:
        RepresenterError: ``SafeDumper`` met an object it cannot represent.
        EmitterError: A bad `version` or `tags`.
    """

def dump(data: Any, stream: Any = None, Dumper: Any = ..., **kwds: Any) -> Any:  # noqa: N803
    """Serialize one Python object into a YAML stream (``yaml.dump``); see :func:`dump_all`."""

def safe_dump_all(documents: Iterable[Any], stream: Any = None, **kwds: Any) -> Any:
    """``yaml.safe_dump_all``: :func:`dump_all` with ``SafeDumper``."""

def safe_dump(data: Any, stream: Any = None, **kwds: Any) -> Any:
    r"""Serialize one Python object with ``SafeDumper`` (``yaml.safe_dump``).

    >>> safe_dump({"b": [1, 2], "a": "x"})
    'a: x\nb:\n- 1\n- 2\n'
    """

def __getattr__(name: str) -> Any:
    """The rest of PyYAML's API (``scan``, ``parse``, ``compose``, ``YAMLObject``, ...)."""
