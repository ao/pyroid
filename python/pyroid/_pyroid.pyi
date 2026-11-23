# Type stubs for the pyroid._pyroid extension module (crates/pyroid-py).
# Hand-written: keep in sync with crates/pyroid-py/src/. Checked by stubtest.
import sys
from collections.abc import Iterator
from types import ModuleType
from typing import Final, Generic, Literal, SupportsIndex, TypeVar, final, overload

_T_co = TypeVar("_T_co", covariant=True)

__all__ = [
    "Column",
    "FallbackWarning",
    "InputError",
    "PyroidError",
    "VersionMismatchWarning",
    "__version__",
    "_numpy_ready",
    "_testing",
    "datasketch",
    "dedup",
    "fasttext",
    "filters",
    "ftfy",
    "jieba",
    "json_repair",
    "metrics",
    "nemo",
    "nltk",
    "text",
    "trafilatura",
    "translit",
    "warc",
]

__version__: Final[str]

# Native submodules; their public faces are pyroid/<name>.py + .pyi.
text: ModuleType
ftfy: ModuleType
fasttext: ModuleType
filters: ModuleType
dedup: ModuleType
translit: ModuleType
metrics: ModuleType
json_repair: ModuleType
datasketch: ModuleType
jieba: ModuleType
nltk: ModuleType
nemo: ModuleType
trafilatura: ModuleType
warc: ModuleType
_testing: ModuleType

def _numpy_ready() -> None:
    """Look NumPy's C API up now (for modules that import numpy at import)."""

class PyroidError(Exception):
    """Base class for all pyroid errors."""

class InputError(PyroidError, TypeError):
    """An argument has the wrong type or shape for this pyroid function."""

class FallbackWarning(RuntimeWarning):
    """A pyroid function delegated this call to the incumbent library (slow path)."""

class VersionMismatchWarning(UserWarning):
    """The installed incumbent library is not the release pyroid reproduces: results can differ."""

@final
class Column(Generic[_T_co]):
    """A read-only Arrow column produced by pyroid (one or more chunks).

    pyroid returns a ``Column`` when it is given Arrow input. It implements the
    Arrow PyCapsule interface, so any Arrow library can take it without a copy:
    ``pyarrow.chunked_array(col)``, ``pyarrow.array(col)``,
    ``polars.Series(col)``, or pass it back to pyroid. pyroid itself needs no
    Arrow library.

    It is also a read-only sequence of its values, which come back as
    :meth:`to_pylist` gives them (``str``, ``bool`` or ``bytes``, and ``None``
    for a null): ``len(col)``, ``col[i]`` (negative ``i`` counts from the end),
    ``for value in col``, ``reversed(col)``, ``value in col``, :meth:`index` and
    :meth:`count`. Iteration reads each row when it gets to it, so it never
    builds a list of the whole column. ``col[a:b]`` is a ``Column`` over the
    same buffers, without a copy; a slice with a step (``col[::2]``) copies the
    rows it selects into a new ``Column``.
    """

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
        """Export as an Arrow C stream (PyCapsule interface)."""

    def __arrow_c_array__(self, requested_schema: object | None = None) -> tuple[object, object]:
        """Export as one Arrow C array, concatenating chunks if needed."""

    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, index: SupportsIndex, /) -> _T_co | None: ...
    @overload
    def __getitem__(self, index: slice, /) -> Column[_T_co]: ...
    def __iter__(self) -> Iterator[_T_co | None]: ...
    def __reversed__(self) -> Iterator[_T_co | None]: ...
    def __contains__(self, value: object, /) -> bool: ...
    def index(
        self, value: object, start: SupportsIndex = 0, stop: SupportsIndex = sys.maxsize, /
    ) -> int:
        """Return the first row in ``start:stop`` whose value equals ``value``.

        Rows compare as in ``list.index``: a row matches if it is ``value`` or
        ``== value``, so ``None`` finds the first null.

        Raises:
            ValueError: No row in ``start:stop`` equals ``value``.
        """

    def count(self, value: object, /) -> int:
        """Return the number of rows whose value equals ``value``."""

    @property
    def type(self) -> Literal["string", "large_string", "bool", "large_binary"]:
        """Arrow type name."""

    @property
    def name(self) -> str:
        """Field name, carried over from the input column (may be empty)."""

    @property
    def null_count(self) -> int:
        """Number of nulls."""

    @property
    def num_chunks(self) -> int:
        """Number of chunks."""

    def to_pylist(self) -> list[_T_co | None]:
        """The values as a Python list (``None`` for nulls)."""
