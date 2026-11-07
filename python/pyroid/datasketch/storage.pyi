# Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
# Type stubs for pyroid.datasketch.storage. The module docstring is kept here on purpose:
# the API docs are built from this stub.
"""Storage containers (``pyroid.datasketch.storage``), as ``datasketch.storage``.

``Storage``, ``OrderedStorage``, ``UnorderedStorage``, ``DictListStorage``,
``DictSetStorage``, ``ordered_storage`` and ``unordered_storage`` are
datasketch's in-memory classes and factories. ``"redis"`` and ``"cassandra"``
configurations are passed to datasketch's factories (a
``pyroid.FallbackWarning``).

``MinHashLSH`` keeps its index in Rust; its ``keys`` and ``hashtables[i]`` are
the read-only views :class:`DictListStorageView` and
:class:`DictSetStorageView`, with the same read API. Their values are new
lists and sets on every call (a snapshot).
"""

from abc import ABC, ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Hashable, Iterator
from typing import Any

__all__ = [
    "DictListStorage",
    "DictListStorageView",
    "DictSetStorage",
    "DictSetStorageView",
    "OrderedStorage",
    "Storage",
    "UnorderedStorage",
    "ordered_storage",
    "unordered_storage",
]

def ordered_storage(config: dict[str, Any], name: bytes | None = None) -> OrderedStorage:
    """Return an ordered storage (values are lists) for ``config``.

    Raises:
        ValueError: ``"Unknown storage type: ..."``.
    """

def unordered_storage(config: dict[str, Any], name: bytes | None = None) -> UnorderedStorage:
    """Return an unordered storage (values are sets) for ``config``.

    Raises:
        ValueError: ``"Unknown storage type: ..."``.
    """

class Storage(ABC):
    """Base class for key, value containers where the values are sequences."""

    def __getitem__(self, key: Any) -> Any: ...
    def __delitem__(self, key: Any) -> Any: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __contains__(self, item: Any) -> bool: ...
    @abstractmethod
    def keys(self) -> Any:
        """Return the keys."""
    @abstractmethod
    def get(self, key: Any) -> Any:
        """Return the values of ``key``, empty if it is absent."""
    def getmany(self, *keys: Any) -> list[Any]:
        """Return ``[self.get(key) for key in keys]``."""
    @abstractmethod
    def insert(self, key: Any, *vals: Any, **kwargs: Any) -> None:
        """Add ``vals`` to the values of ``key``."""
    @abstractmethod
    def remove(self, *keys: Any, **kwargs: Any) -> None:
        """Remove ``keys``."""
    @abstractmethod
    def remove_val(self, key: Any, val: Any) -> None:
        """Remove ``val`` from the values of ``key``."""
    @abstractmethod
    def size(self) -> int:
        """Return the number of keys."""
    @abstractmethod
    def itemcounts(self, **kwargs: Any) -> dict[Any, int]:
        """Return the number of values of each key."""
    @abstractmethod
    def has_key(self, key: Any) -> bool:
        """Return whether ``key`` is stored."""
    @property
    def buffer_size(self) -> int:
        """The buffer size (used by the Redis and Cassandra storages)."""
    @buffer_size.setter
    def buffer_size(self, value: int) -> None: ...
    def status(self) -> dict[str, Any]:
        """Return ``{"keyspace_size": len(self)}``."""
    def empty_buffer(self) -> None:
        """Flush buffered writes (nothing to do in memory)."""
    def add_to_select_buffer(self, keys: Any) -> None:
        """Look up ``keys`` and buffer the results."""
    def collect_select_buffer(self) -> list[Any]:
        """Return the buffered results and clear the buffer."""

class OrderedStorage(Storage, metaclass=ABCMeta):
    """A storage whose values are lists (``defaultdict(list)``)."""

class UnorderedStorage(Storage, metaclass=ABCMeta):
    """A storage whose values are sets (``defaultdict(set)``)."""

class DictListStorage(OrderedStorage):
    """``defaultdict(list)`` with the :class:`Storage` API."""

    _dict: defaultdict[Any, Any]
    def __init__(self, config: Any) -> None: ...
    def keys(self) -> Any:
        """Return the keys (a dict keys view)."""
    def get(self, key: Any) -> Any:
        """Return the list stored under ``key``, ``[]`` if there is none."""
    def remove(self, *keys: Any, **kwargs: Any) -> None:
        """Delete ``keys`` (``KeyError`` if one is absent)."""
    def remove_val(self, key: Any, val: Any, **kwargs: Any) -> None:
        """Remove ``val`` from the values of ``key``."""
    def insert(self, key: Any, *vals: Any, **kwargs: Any) -> None:
        """Add ``vals`` to the values of ``key``."""
    def size(self) -> int:
        """Return the number of keys."""
    def itemcounts(self, **kwargs: Any) -> dict[Any, int]:
        """Return ``{key: number of values}``."""
    def has_key(self, key: Any) -> bool:
        """Return whether ``key`` is stored."""

class DictSetStorage(UnorderedStorage, DictListStorage):
    """``defaultdict(set)`` with the :class:`Storage` API."""

    def __init__(self, config: Any) -> None: ...
    def get(self, key: Any) -> Any:
        """Return the set stored under ``key``, ``set()`` if there is none."""
    def insert(self, key: Any, *vals: Any, **kwargs: Any) -> None:
        """Add ``vals`` to the set of ``key``."""

class DictListStorageView(OrderedStorage):
    """Read-only view of ``MinHashLSH.keys``: key -> its band keys.

    It has ``DictListStorage``'s API. ``insert``, ``remove`` and
    ``remove_val`` raise ``NotImplementedError``.
    """

    _index: Any
    def __init__(self, index: Any) -> None: ...
    def keys(self) -> list[Hashable]:
        """Return the keys in insertion order (a new list)."""
    def get(self, key: Hashable) -> list[bytes]:
        """Return the band keys of ``key`` (a new list), ``[]`` if it is absent."""
    def insert(self, key: Any, *vals: Any, **kwargs: Any) -> None:
        """Raise ``NotImplementedError``: the view is read-only."""
    def remove(self, *keys: Any, **kwargs: Any) -> None:
        """Raise ``NotImplementedError``: the view is read-only."""
    def remove_val(self, key: Any, val: Any, **kwargs: Any) -> None:
        """Raise ``NotImplementedError``: the view is read-only."""
    def size(self) -> int:
        """Return the number of keys."""
    def itemcounts(self, **kwargs: Any) -> dict[Hashable, int]:
        """Return ``{key: number of band keys}``."""
    def has_key(self, key: Hashable) -> bool:
        """Return whether ``key`` is in the index."""

class DictSetStorageView(UnorderedStorage):
    """Read-only view of one band table, ``MinHashLSH.hashtables[i]``.

    It has ``DictSetStorage``'s API: band key (bytes) -> the set of keys in
    that bucket. Buckets iterate in no particular order. ``insert``,
    ``remove`` and ``remove_val`` raise ``NotImplementedError``.
    """

    _index: Any
    _table: int
    def __init__(self, index: Any, table: int) -> None: ...
    def keys(self) -> list[bytes]:
        """Return the band keys of the buckets (a new list)."""
    def get(self, key: Any) -> set[Hashable]:
        """Return the keys in bucket ``key`` (a new set), ``set()`` if it is absent."""
    def insert(self, key: Any, *vals: Any, **kwargs: Any) -> None:
        """Raise ``NotImplementedError``: the view is read-only."""
    def remove(self, *keys: Any, **kwargs: Any) -> None:
        """Raise ``NotImplementedError``: the view is read-only."""
    def remove_val(self, key: Any, val: Any, **kwargs: Any) -> None:
        """Raise ``NotImplementedError``: the view is read-only."""
    def size(self) -> int:
        """Return the number of buckets."""
    def itemcounts(self, **kwargs: Any) -> dict[bytes, int]:
        """Return ``{band key: number of keys in the bucket}``."""
    def has_key(self, key: Any) -> bool:
        """Return whether there is a bucket ``key``."""

def _random_name(length: int) -> bytes: ...
