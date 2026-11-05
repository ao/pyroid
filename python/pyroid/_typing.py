"""Typing helpers shared by pyroid's stubs. Not a public API."""

from typing import Protocol, TypeAlias

__all__ = ["ArrowArrayExportable", "ArrowStreamExportable", "ArrowStrings"]


class ArrowArrayExportable(Protocol):
    """An object with ``__arrow_c_array__`` (Arrow PyCapsule interface)."""

    def __arrow_c_array__(
        self, requested_schema: object | None = None, /
    ) -> tuple[object, object]: ...


class ArrowStreamExportable(Protocol):
    """An object with ``__arrow_c_stream__`` (Arrow PyCapsule interface)."""

    def __arrow_c_stream__(self, requested_schema: object | None = None, /) -> object: ...


#: Arrow string input: a pyarrow ``Array``/``ChunkedArray``, a polars
#: ``Series``, a :class:`pyroid.Column`, or any other PyCapsule producer.
ArrowStrings: TypeAlias = ArrowArrayExportable | ArrowStreamExportable
