# Type stubs for the pyroid._pyroid extension module (crates/pyroid-py).
# Hand-written: keep in sync with crates/pyroid-py/src/. Checked by stubtest.
from types import ModuleType
from typing import Final, Generic, Literal, TypeVar, final

_T_co = TypeVar("_T_co", covariant=True)

__all__ = [
    "Column",
    "FallbackWarning",
    "InputError",
    "PyroidError",
    "__version__",
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

class PyroidError(Exception):
    """Base class for all pyroid errors."""

class InputError(PyroidError, TypeError):
    """An argument has the wrong type or shape for this pyroid function."""

class FallbackWarning(RuntimeWarning):
    """A pyroid function delegated this call to the incumbent library (slow path)."""

@final
class Column(Generic[_T_co]):
    """A read-only Arrow column produced by pyroid (one or more chunks).

    pyroid returns a ``Column`` when it is given Arrow input. It implements the
    Arrow PyCapsule interface, so any Arrow library can take it without a copy:
    ``pyarrow.chunked_array(col)``, ``pyarrow.array(col)``,
    ``polars.Series(col)``, or pass it back to pyroid. pyroid itself needs no
    Arrow library.
    """

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
        """Export as an Arrow C stream (PyCapsule interface)."""

    def __arrow_c_array__(self, requested_schema: object | None = None) -> tuple[object, object]:
        """Export as one Arrow C array, concatenating chunks if needed."""

    def __len__(self) -> int: ...
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
