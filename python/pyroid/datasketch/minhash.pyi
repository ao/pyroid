# Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
# Type stubs for pyroid.datasketch.minhash. The API docs are built from this stub.
"""datasketch's ``MinHash`` (``datasketch.minhash``), with the hashing in Rust.

``from pyroid.datasketch import MinHash`` replaces ``from datasketch import MinHash``.
A pyroid ``MinHash`` has datasketch 2.0.0's attributes (``hashvalues``,
``permutations``, ``seed``, ``scheme``, ``hashfunc``, ...), the same hash values
for all three permutation schemes (``"affine32"``, the default, ``"affine64"`` and
``"legacy"``), the same pickled state and the same errors.

What runs in Rust: drawing the permutation parameters (numpy's ``RandomState``,
reproduced and cached per seed), SHA1 hashing with the default hash functions,
and applying the permutations. :meth:`MinHash.bulk` and :meth:`MinHash.generator`
hash many documents in parallel (``threads=``), and :func:`minhash_batch` returns
a signature matrix for token lists or the word n-grams of texts.

Example:
    ```python
    from pyroid.datasketch import MinHash

    m1, m2 = MinHash(num_perm=128), MinHash(num_perm=128)
    m1.update_batch([b"minhash", b"is", b"a", b"probabilistic", b"data", b"structure"])
    m2.update_batch([b"minhash", b"is", b"a", b"probability", b"data", b"structure"])
    m1.jaccard(m2)  # 0.65625, as with datasketch

    sigs = MinHash.bulk([[b"a", b"b"], [b"b", b"c"]], num_perm=64, threads=8)
    ```

Differences from datasketch
---------------------------
- A ``hashfunc`` other than ``sha1_hash32``/``sha1_hash64`` (pyroid's or
  datasketch's) is called from Python for every token, as in datasketch, with a
  one-time :class:`pyroid.FallbackWarning`; the permutations still run in Rust
  when it returns ints in range.
- Tokens that are not ``bytes`` (``bytearray``, ``memoryview``, numpy scalars),
  and hash values or permutations that are not the scheme's numpy dtype, run
  datasketch's numpy code (same results, datasketch's speed).
- ``gpu_mode`` is accepted, but pyroid always computes on the CPU; the values are
  the same. ``gpu_mode="always"`` still raises ``RuntimeError`` without CuPy and a
  CUDA device, as datasketch does.
- CuPy is imported only when a MinHash with ``gpu_mode`` other than
  ``"disable"`` updates, not when the module is imported.
"""

from collections.abc import Callable, Generator, Iterable, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyroid._typing import ArrowStrings

__all__ = ["MinHash", "minhash_batch"]

hashvalue_byte_size: int

_Scheme: TypeAlias = Literal["affine32", "affine64", "legacy"]

def _fmix(hv: Any, width: int) -> Any: ...

class MinHash:
    """MinHash: estimate the Jaccard similarity and cardinality of sets.

    Args:
        num_perm: Number of random permutation functions. Ignored if
            `hashvalues` is given.
        seed: The random seed of the permutation functions.
        gpu_mode: ``"disable"``, ``"detect"`` or ``"always"``. pyroid computes on
            the CPU in every mode; ``"always"`` raises ``RuntimeError`` if CuPy and
            a CUDA device are not available, as in datasketch.
        hashfunc: The hash function: takes a value given to :meth:`update` and
            returns an int of 32 bits (``"affine32"``, ``"legacy"``) or 64 bits
            (``"affine64"``). Default: ``sha1_hash32``, or ``sha1_hash64`` for
            ``"affine64"``. Other functions run in Python (slow path).
        hashobj: Deprecated since datasketch 1.4.0; a no-op that warns.
        hashvalues: Initial hash values (another MinHash's :attr:`hashvalues`).
            `scheme` must then be given.
        permutations: Permutation parameters (another MinHash's
            :attr:`permutations`). `scheme` must then be given.
        scheme: ``"affine32"`` (the default), ``"affine64"`` or ``"legacy"``
            (datasketch before 2.0.0). MinHash of different schemes cannot be
            compared, merged or unioned.

    Raises:
        ValueError: As datasketch: an unknown scheme, `num_perm` out of
            range, `hashvalues`/`permutations` without `scheme`, mismatched
            lengths, values out of the scheme's range.
    """

    scheme: str
    seed: Any
    num_perm: int
    hashfunc: Callable[[Any], Any]
    hashvalues: NDArray[Any]
    permutations: Any
    _gpu_mode: str
    _a_gpu: Any
    _b_gpu: Any
    def __init__(
        self,
        num_perm: int = 128,
        seed: int = 1,
        gpu_mode: Literal["disable", "detect", "always"] = "disable",
        hashfunc: Callable[[Any], Any] | None = None,
        hashobj: object | None = None,
        hashvalues: ArrayLike | None = None,
        permutations: tuple[ArrayLike, ArrayLike] | ArrayLike | None = None,
        scheme: _Scheme | None = None,
    ) -> None: ...
    def update(self, b: Any) -> None:
        """Add one value to the set.

        With the default hash function the value must be bytes-like, e.g.
        ``m.update("text".encode("utf-8"))``.

        Args:
            b: The value, passed to :attr:`hashfunc`.
        """

    def update_batch(self, b: Iterable[Any]) -> None:
        """Add many values to the set (faster than :meth:`update` per value).

        Args:
            b: The values, each passed to :attr:`hashfunc`. An empty batch
                leaves the hash values unchanged.

        Raises:
            RuntimeError: ``gpu_mode="always"`` without CuPy and a CUDA device.
        """

    def jaccard(self, other: MinHash) -> float:
        """Estimate the Jaccard similarity with `other`.

        Raises:
            ValueError: Different schemes, seeds or numbers of permutations.
        """

    def count(self) -> float:
        """Estimate the cardinality of the set (a ``numpy.float64``)."""

    def merge(self, other: MinHash) -> None:
        """Make this MinHash the union of itself and `other`.

        Raises:
            ValueError: Different schemes, seeds or numbers of permutations.
        """

    def digest(self) -> NDArray[Any]:
        """Return a copy of :attr:`hashvalues`."""

    def is_empty(self) -> bool:
        """Return whether the MinHash is in its initial state."""

    def clear(self) -> None:
        """Reset the hash values to the initial state."""

    def copy(self) -> MinHash:
        """Return a copy (the permutations are shared, as in datasketch)."""

    def __len__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    @classmethod
    def union(cls, *mhs: MinHash) -> MinHash:
        """Return a new MinHash, the union of `mhs` (at least two).

        Raises:
            ValueError: Fewer than two, or different seeds, sizes or schemes.
        """

    @classmethod
    def bulk(
        cls, b: Iterable[Iterable[Any]], *, threads: int | None = None, **minhash_kwargs: Any
    ) -> list[MinHash]:
        """Compute one MinHash per document of `b`.

        Equal to ``[m.copy() updated with update_batch(doc) for doc in b]`` with
        ``m = MinHash(**minhash_kwargs)``, as in datasketch. Documents that are
        lists or tuples of ``bytes`` are hashed in parallel on pyroid's thread
        pool; others run datasketch's code.

        Args:
            b: The documents, each an iterable of values for :attr:`hashfunc`.
            threads: pyroid extension: at most this many threads (``None``:
                the whole pool, ``1``: the calling thread).
            **minhash_kwargs: Arguments for :class:`MinHash`, used for all.

        Example:
            ```python
            data = [[b"token1", b"token2", b"token3"], [b"token4", b"token5"]]
            minhashes = MinHash.bulk(data, num_perm=64)
            ```
        """

    @classmethod
    def generator(
        cls, b: Iterable[Iterable[Any]], *, threads: int | None = None, **minhash_kwargs: Any
    ) -> Generator[MinHash]:
        """Yield one MinHash per document of `b` (see :meth:`bulk`).

        pyroid reads up to 2,048 documents ahead to hash them in parallel; the
        MinHash objects and any error come out in datasketch's order.
        """

    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...

def minhash_batch(
    docs: str | list[str] | tuple[str, ...] | ArrowStrings | Sequence[Sequence[bytes | str]],
    *,
    num_perm: int = 128,
    seed: int = 1,
    scheme: _Scheme | None = None,
    ngram: int = 1,
    threads: int | None = None,
) -> NDArray[np.uint32] | NDArray[np.uint64]:
    """Compute MinHash signatures for many documents, as a matrix.

    Row ``i`` equals ``MinHash(num_perm, seed, scheme=scheme)`` updated with the
    tokens of document ``i``, i.e. its :attr:`MinHash.hashvalues` (``uint32`` for
    ``"affine32"``, ``uint64`` otherwise). Documents are hashed in parallel on
    pyroid's thread pool. A pyroid extension.

    Args:
        docs: Token lists (a list or tuple of lists/tuples of ``bytes`` or
            ``str`` tokens; ``str`` is hashed as UTF-8), or texts (``str``,
            ``list[str]``, ``tuple[str, ...]`` or an Arrow string array). A text's
            tokens are its word `ngram`-grams: ``" ".join(words[i:i + ngram])``
            over ``text.split()``, UTF-8 encoded; a text with fewer words has no
            tokens. Arrow nulls give the empty signature.
        num_perm: Number of permutations.
        seed: The permutation seed.
        scheme: ``"affine32"`` (default), ``"affine64"`` or ``"legacy"``.
        ngram: Words per shingle, for texts only.
        threads: At most this many threads (``None``: the whole pool).

    Returns:
        A ``(len(docs), num_perm)`` array, or ``(num_perm,)`` for a single ``str``.

    Raises:
        pyroid.InputError: A document or token of an unsupported type.
        ValueError: `ngram` below 1, or as :class:`MinHash` for bad parameters.
        UnicodeEncodeError: A ``str`` with a lone surrogate.

    Example:
        ```python
        from pyroid.datasketch import MinHashLSH, minhash_batch

        sigs = minhash_batch(texts, ngram=5, threads=8)  # (len(texts), 128) uint32
        lsh = MinHashLSH(threshold=0.8)
        lsh.insert_batch(range(len(texts)), sigs)
        ```
    """
