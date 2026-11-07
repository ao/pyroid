# Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
"""Storage containers, as ``datasketch.storage``. Types and docs are in ``storage.pyi``.

The in-memory containers and the ``Storage`` base classes are datasketch's.
``MinHashLSH`` keeps its state in Rust and shows it through the read-only
views at the end of this module. Redis and Cassandra storage come from
datasketch itself.
"""

from __future__ import annotations

import random
import string
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Hashable, Iterator
from typing import Any

from pyroid.datasketch import _common

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


def _datasketch_storage(function: str, tp: str) -> Any:
    real = _common.fallback(f"storage.{function}", f"storage type {tp!r}")
    return real.storage


def ordered_storage(config: dict[str, Any], name: bytes | None = None) -> OrderedStorage:
    """Return an ordered storage container (``defaultdict(list)``-like) for ``config``."""
    tp = config["type"]
    if tp == "dict":
        return DictListStorage(config)
    if tp in ("redis", "cassandra"):
        return _datasketch_storage("ordered_storage", tp).ordered_storage(config, name=name)  # type: ignore[no-any-return]
    raise ValueError(f"Unknown storage type: {tp}")


def unordered_storage(config: dict[str, Any], name: bytes | None = None) -> UnorderedStorage:
    """Return an unordered storage container (``defaultdict(set)``-like) for ``config``."""
    tp = config["type"]
    if tp == "dict":
        return DictSetStorage(config)
    if tp in ("redis", "cassandra"):
        return _datasketch_storage("unordered_storage", tp).unordered_storage(config, name=name)  # type: ignore[no-any-return]
    raise ValueError(f"Unknown storage type: {tp}")


class Storage(ABC):
    """Base class for key, value containers where the values are sequences."""

    def __getitem__(self, key: Any) -> Any:
        return self.get(key)

    def __delitem__(self, key: Any) -> Any:
        return self.remove(key)

    def __len__(self) -> int:
        return self.size()

    def __iter__(self) -> Iterator[Any]:
        yield from self.keys()

    def __contains__(self, item: Any) -> bool:
        return self.has_key(item)

    @abstractmethod
    def keys(self) -> Any:
        """Return an iterator on keys in storage."""
        return []

    @abstractmethod
    def get(self, key: Any) -> Any:
        """Return the values stored under ``key``, empty if it is absent."""

    def getmany(self, *keys: Any) -> list[Any]:
        """Return ``[self.get(key) for key in keys]``."""
        return [self.get(key) for key in keys]

    @abstractmethod
    def insert(self, key: Any, *vals: Any, **kwargs: Any) -> None:
        """Add `val` to storage against `key`."""

    @abstractmethod
    def remove(self, *keys: Any, **kwargs: Any) -> None:
        """Remove `keys` from storage."""

    @abstractmethod
    def remove_val(self, key: Any, val: Any) -> None:
        """Remove `val` from list of values under `key`."""

    @abstractmethod
    def size(self) -> int:
        """Return size of storage with respect to number of keys."""

    @abstractmethod
    def itemcounts(self, **kwargs: Any) -> dict[Any, int]:
        """Return the number of values stored under each key."""

    @abstractmethod
    def has_key(self, key: Any) -> bool:
        """Return whether ``key`` is in the storage."""

    @property
    def buffer_size(self) -> int:
        """The buffer size (used by the Redis and Cassandra storages)."""
        return getattr(self, "_buffer_size", 50000)

    @buffer_size.setter
    def buffer_size(self, value: int) -> None:
        self._buffer_size = value

    def status(self) -> dict[str, Any]:
        """Return ``{"keyspace_size": len(self)}``."""
        return {"keyspace_size": len(self)}

    def empty_buffer(self) -> None:  # noqa: B027
        """Flush buffered writes (nothing to do in memory)."""

    def add_to_select_buffer(self, keys: Any) -> None:
        """Query keys and add them to internal buffer."""
        if not hasattr(self, "_select_buffer"):
            self._select_buffer = self.getmany(*keys)
        else:
            self._select_buffer.extend(self.getmany(*keys))

    def collect_select_buffer(self) -> list[Any]:
        """Return buffered query results."""
        if not hasattr(self, "_select_buffer"):
            return []
        buffer = list(self._select_buffer)
        del self._select_buffer[:]
        return buffer


class OrderedStorage(Storage):
    """A storage whose values are lists (``defaultdict(list)``)."""


class UnorderedStorage(Storage):
    """A storage whose values are sets (``defaultdict(set)``)."""


class DictListStorage(OrderedStorage):
    """``defaultdict(list)`` with the :class:`Storage` API."""

    def __init__(self, config: Any) -> None:
        self._dict: defaultdict[Any, Any] = defaultdict(list)

    def keys(self) -> Any:
        """Return the keys (a dict keys view)."""
        return self._dict.keys()

    def get(self, key: Any) -> Any:
        """Return the list stored under ``key``, ``[]`` if there is none."""
        return self._dict.get(key, [])

    def remove(self, *keys: Any, **kwargs: Any) -> None:
        """Delete ``keys`` (``KeyError`` if one is absent)."""
        for key in keys:
            del self._dict[key]

    def remove_val(self, key: Any, val: Any, **kwargs: Any) -> None:
        """Remove ``val`` from the values of ``key``."""
        self._dict[key].remove(val)

    def insert(self, key: Any, *vals: Any, **kwargs: Any) -> None:
        """Add ``vals`` to the values of ``key``."""
        self._dict[key].extend(vals)

    def size(self) -> int:
        """Return the number of keys."""
        return len(self._dict)

    def itemcounts(self, **kwargs: Any) -> dict[Any, int]:
        """Return ``{key: number of values}``."""
        return {k: len(v) for k, v in self._dict.items()}

    def has_key(self, key: Any) -> bool:
        """Return whether ``key`` is stored."""
        return key in self._dict


class DictSetStorage(UnorderedStorage, DictListStorage):
    """``defaultdict(set)`` with the :class:`Storage` API."""

    def __init__(self, config: Any) -> None:
        self._dict = defaultdict(set)

    def get(self, key: Any) -> Any:
        """Return the set stored under ``key``, ``set()`` if there is none."""
        return self._dict.get(key, set())

    def insert(self, key: Any, *vals: Any, **kwargs: Any) -> None:
        """Add ``vals`` to the set of ``key``."""
        self._dict[key].update(vals)


def _random_name(length: int) -> bytes:
    # For use with Redis, we return bytes
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length)).encode("utf8")


def _read_only(name: str) -> NotImplementedError:
    return NotImplementedError(
        f"pyroid.datasketch: MinHashLSH storage views are read-only ({name}); "
        "use MinHashLSH.insert, remove or merge"
    )


class DictListStorageView(OrderedStorage):
    """Read-only view of ``MinHashLSH.keys``, with ``DictListStorage``'s API.

    Values are new lists (a snapshot); the mutating methods raise
    ``NotImplementedError``.
    """

    def __init__(self, index: Any) -> None:
        self._index = index

    def keys(self) -> list[Hashable]:
        """Return the keys, in insertion order (a new list)."""
        return self._index.keys_list()  # type: ignore[no-any-return]

    def get(self, key: Hashable) -> list[bytes]:
        """Return the band keys of ``key`` (a new list), ``[]`` if it is absent."""
        hs = self._index.keys_get(key)
        return [] if hs is None else hs

    def insert(self, key: Any, *vals: Any, **kwargs: Any) -> None:
        """Raise ``NotImplementedError``: the view is read-only."""
        raise _read_only("insert")

    def remove(self, *keys: Any, **kwargs: Any) -> None:
        """Raise ``NotImplementedError``: the view is read-only."""
        raise _read_only("remove")

    def remove_val(self, key: Any, val: Any, **kwargs: Any) -> None:
        """Raise ``NotImplementedError``: the view is read-only."""
        raise _read_only("remove_val")

    def size(self) -> int:
        """Return the number of keys."""
        return len(self._index)

    def itemcounts(self, **kwargs: Any) -> dict[Hashable, int]:
        """Return ``{key: number of band keys}``."""
        return self._index.keys_counts()  # type: ignore[no-any-return]

    def has_key(self, key: Hashable) -> bool:
        """Return whether ``key`` is in the index."""
        return self._index.contains(key)  # type: ignore[no-any-return]


class DictSetStorageView(UnorderedStorage):
    """Read-only view of one band table, ``MinHashLSH.hashtables[i]``.

    It has ``DictSetStorage``'s API: band keys (bytes) map to the set of keys
    in that bucket. Values are new sets (a snapshot); the mutating methods
    raise ``NotImplementedError``.
    """

    def __init__(self, index: Any, table: int) -> None:
        self._index = index
        self._table = table

    def keys(self) -> list[bytes]:
        """Return the band keys of the buckets (a new list, in no particular order)."""
        return self._index.table_keys(self._table)  # type: ignore[no-any-return]

    def get(self, key: Any) -> set[Hashable]:
        """Return the keys in bucket ``key`` (a new set), ``set()`` if it is absent."""
        if not isinstance(key, bytes):
            hash(key)
            return set()
        found = self._index.table_get(self._table, key)
        return set() if found is None else found

    def insert(self, key: Any, *vals: Any, **kwargs: Any) -> None:
        """Raise ``NotImplementedError``: the view is read-only."""
        raise _read_only("insert")

    def remove(self, *keys: Any, **kwargs: Any) -> None:
        """Raise ``NotImplementedError``: the view is read-only."""
        raise _read_only("remove")

    def remove_val(self, key: Any, val: Any, **kwargs: Any) -> None:
        """Raise ``NotImplementedError``: the view is read-only."""
        raise _read_only("remove_val")

    def size(self) -> int:
        """Return the number of buckets."""
        return self._index.table_len(self._table)  # type: ignore[no-any-return]

    def itemcounts(self, **kwargs: Any) -> dict[bytes, int]:
        """Return ``{band key: number of keys in the bucket}``."""
        return self._index.table_counts(self._table)  # type: ignore[no-any-return]

    def has_key(self, key: Any) -> bool:
        """Return whether there is a bucket ``key``."""
        if not isinstance(key, bytes):
            hash(key)
            return False
        return self._index.table_contains(self._table, key)  # type: ignore[no-any-return]
