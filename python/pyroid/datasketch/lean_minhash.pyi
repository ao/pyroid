# Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
# Type stubs for pyroid.datasketch.lean_minhash.
"""datasketch's ``LeanMinHash`` (``datasketch.lean_minhash``).

A frozen :class:`~pyroid.datasketch.MinHash` without permutations or hash
function, for storage and indexing. :meth:`LeanMinHash.serialize` writes
datasketch 2.0.0's byte format (the pre-2.0.0 format for ``"legacy"``), so the
bytes are interchangeable with datasketch's in both directions; pickles hold
the same bytes.

Example:
    ```python
    from pyroid.datasketch import LeanMinHash

    lean = LeanMinHash(minhash)
    buf = bytearray(lean.bytesize())
    lean.serialize(buf)
    LeanMinHash.deserialize(buf) == lean  # True
    ```
"""

from collections.abc import Iterable
from typing import Any

from typing_extensions import Buffer

from pyroid.datasketch.minhash import MinHash

__all__ = ["LeanMinHash"]

class LeanMinHash(MinHash):
    """A MinHash with its state frozen: no :meth:`update`.

    Args:
        minhash: The MinHash to take the seed, hash values and scheme from.
        seed: With `hashvalues` and `scheme`: the seed of the permutations.
        hashvalues: With `seed` and `scheme`: the hash values.
        scheme: The scheme of `hashvalues` (required with them; ``"legacy"`` for
            values from datasketch before 2.0.0). With `minhash` it may only
            repeat the MinHash's scheme.

    Raises:
        ValueError: Neither `minhash` nor `seed` and `hashvalues`; no or
            conflicting `scheme`; empty `hashvalues` for an affine scheme.
    """

    __slots__ = ("hashvalues", "scheme", "seed")

    def __init__(
        self,
        minhash: MinHash | None = None,
        seed: int | None = None,
        hashvalues: Iterable[int] | None = None,
        scheme: str | None = None,
    ) -> None: ...
    def update(self, b: Any) -> None:
        """Raise ``TypeError``: a LeanMinHash cannot be updated."""

    def copy(self) -> LeanMinHash:
        """Return a copy."""

    def bytesize(self, byteorder: str = "@") -> int:
        """Return the size of the serialized form in bytes.

        Args:
            byteorder: A `struct` byte order character: ``@`` (native, the
                default), ``=``, ``<``, ``>`` or ``!``.
        """

    def serialize(self, buf: Buffer, byteorder: str = "@") -> None:
        """Write this LeanMinHash into the writable buffer `buf`, from offset 0.

        Layout (``"legacy"``, as before datasketch 2.0.0): seed (8 bytes), the
        number of hash values (4 bytes), then each value in 4 bytes. Affine
        schemes: seed, the *negated* number of values, a scheme code byte (1 for
        ``"affine32"``, 2 for ``"affine64"``), then each value in 4 or 8 bytes.

        Raises:
            ValueError: `buf` is shorter than :meth:`bytesize`.
        """

    @classmethod
    def deserialize(cls, buf: Buffer, byteorder: str = "@") -> LeanMinHash:
        """Read a LeanMinHash written by :meth:`serialize` (pyroid's or datasketch's).

        Buffers written before datasketch 2.0.0 come back with
        ``scheme="legacy"``.

        Raises:
            ValueError: An unknown scheme code.
        """

    def __getstate__(self) -> bytearray: ...  # type: ignore[override]
    def __setstate__(self, buf: Buffer) -> None: ...  # type: ignore[override]
    def __hash__(self) -> int: ...
    @classmethod
    def union(cls, *lmhs: LeanMinHash) -> LeanMinHash:  # type: ignore[override]
        """Return a new LeanMinHash, the union of `lmhs` (at least two)."""
