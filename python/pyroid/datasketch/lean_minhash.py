# Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
# Error messages keep datasketch's %-formatting, so they stay byte-identical and the code
# stays easy to diff against upstream.
# ruff: noqa: UP031, E501, UP045
"""datasketch's ``LeanMinHash`` (``datasketch.lean_minhash``): same byte format and pickles.

A frozen MinHash without permutations or hash function. Types and docs are in
``lean_minhash.pyi``.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable
from typing import Optional

import numpy as np

from pyroid._pyroid import datasketch as _native
from pyroid.datasketch.minhash import (
    _SCHEME_AFFINE32,
    _SCHEME_AFFINE64,
    _SCHEME_CODES,
    _SCHEME_CODES_INV,
    _SCHEME_LEGACY,
    _VALID_SCHEMES,
    MinHash,
)

__all__ = ["LeanMinHash"]

# Byte-format notes: legacy payloads have no scheme field and are identified
# by a non-negative number-of-hash-values field, while the affine formats
# store the negated number followed by a scheme code byte. This keeps legacy
# sketches bit-identical to (and readable by) versions before 2.0.0.
# struct format character of one hash value, per scheme.
_SCHEME_VALUE_FMTS = {
    _SCHEME_LEGACY: "I",
    _SCHEME_AFFINE32: "I",
    _SCHEME_AFFINE64: "Q",
}


_BYTE_ORDERS = ("@", "=", "<", ">", "!")
_INT64 = (-(1 << 63), 1 << 63)


def _native_unpack(buf, byteorder):
    """``LeanMinHash.deserialize`` in Rust for bytes/bytearray, or None (then ``struct`` runs)."""
    if (type(buf) is bytes or type(buf) is bytearray) and byteorder in _BYTE_ORDERS:
        res = _native.lean_unpack(buf, byteorder)
        if res is not None:
            seed, scheme, hashvalues = res
            lmh = object.__new__(LeanMinHash)
            lmh._initialize_slots(seed, hashvalues, scheme)
            return lmh
    return None


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

    def _initialize_slots(self, seed, hashvalues, scheme=_SCHEME_LEGACY):
        """Initialize the slots of the LeanMinHash.

        Args:
            seed (int): The random seed controls the set of random
                permutation functions generated for this LeanMinHash.
            hashvalues (Iterable): The hash values is the internal state of the LeanMinHash.
            scheme (str): The permutation scheme of the hash values.

        """
        if scheme not in _VALID_SCHEMES:
            raise ValueError(
                "scheme must be one of %s, got %r" % (", ".join(_VALID_SCHEMES), scheme)
            )
        self.seed = seed
        self.scheme = scheme
        self.hashvalues = self._parse_hashvalues(hashvalues)
        if scheme != _SCHEME_LEGACY and len(self.hashvalues) == 0:
            # An empty sketch would serialize with a hash value count of 0,
            # which the deserializer cannot tell apart from the legacy format
            # (identified by a non-negative count).
            raise ValueError("hashvalues must not be empty")

    def __init__(
        self,
        minhash: MinHash = None,
        seed: Optional[int] = None,
        hashvalues: Optional[Iterable] = None,
        scheme: Optional[str] = None,
    ):
        if minhash is not None:
            if scheme is not None and scheme != minhash.scheme:
                raise ValueError(
                    "scheme %r conflicts with the scheme %r of the given MinHash"
                    % (scheme, minhash.scheme)
                )
            self._initialize_slots(minhash.seed, minhash.hashvalues, minhash.scheme)
        elif hashvalues is not None and seed is not None:
            if scheme is None:
                # Hash values carry no trace of the scheme that produced
                # them, so a default here would silently mislabel pre-2.0.0
                # values and defeat the cross-scheme comparison guards.
                raise ValueError(
                    "scheme must be specified explicitly when initializing from existing "
                    "hash values: pass the scheme of the MinHash they came from, or "
                    "scheme='legacy' for hash values created by datasketch before 2.0.0."
                )
            self._initialize_slots(seed, hashvalues, scheme)
        else:
            raise ValueError(
                "Init parameters cannot be None: make sure to set either minhash or both of hash values and seed"
            )

    def update(self, b) -> None:
        """Raise ``TypeError``: a LeanMinHash cannot be updated."""
        raise TypeError("Cannot update a LeanMinHash")

    def copy(self) -> LeanMinHash:
        """Return a copy."""
        lmh = object.__new__(LeanMinHash)
        lmh._initialize_slots(self.seed, self.hashvalues, self.scheme)
        return lmh

    def _value_fmt(self) -> str:
        return _SCHEME_VALUE_FMTS[self.scheme]

    def _native_pack(self, byteorder):
        # The serialized bytes from Rust, or None where `struct` would raise or the
        # state is not canonical (then datasketch's `struct` code runs).
        seed = self.seed
        if type(seed) is not int or not _INT64[0] <= seed < _INT64[1]:
            return None
        if byteorder not in _BYTE_ORDERS or self.scheme not in _SCHEME_VALUE_FMTS:
            return None
        return _native.lean_pack(self.scheme, seed, self.hashvalues, byteorder)

    def bytesize(self, byteorder="@") -> int:
        """Return the size of the serialized form in bytes.

        Args:
            byteorder: A `struct` byte order character: ``@`` (native, the
                default), ``=``, ``<``, ``>`` or ``!``.
        """
        if self.scheme == _SCHEME_LEGACY:
            # 8 bytes for the seed, 4 bytes for the number of hash values,
            # and 4 bytes for each hash value.
            return struct.calcsize("%sqi%dI" % (byteorder, len(self)))
        # The affine formats add a 1-byte scheme code after the number of
        # hash values, and store each hash value in 4 ("affine32") or
        # 8 ("affine64") bytes.
        return struct.calcsize("%sqiB%d%s" % (byteorder, len(self), self._value_fmt()))

    def serialize(self, buf, byteorder="@") -> None:
        """Write this LeanMinHash into the writable buffer `buf`, from offset 0.

        Layout (``"legacy"``, as before datasketch 2.0.0): seed (8 bytes), the
        number of hash values (4 bytes), then each value in 4 bytes. Affine
        schemes: seed, the *negated* number of values, a scheme code byte (1 for
        ``"affine32"``, 2 for ``"affine64"``), then each value in 4 or 8 bytes.

        Raises:
            ValueError: `buf` is shorter than :meth:`bytesize`.
        """
        if len(buf) < self.bytesize(byteorder):
            raise ValueError("The buffer does not have enough space for holding this MinHash.")
        if type(buf) is bytearray:
            data = self._native_pack(byteorder)
            if data is not None:
                buf[: len(data)] = data
                return
        if self.scheme == _SCHEME_LEGACY:
            fmt = "%sqi%dI" % (byteorder, len(self))
            struct.pack_into(fmt, buf, 0, self.seed, len(self), *self.hashvalues)
        else:
            fmt = "%sqiB%d%s" % (byteorder, len(self), self._value_fmt())
            struct.pack_into(
                fmt, buf, 0, self.seed, -len(self), _SCHEME_CODES[self.scheme], *self.hashvalues
            )

    @classmethod
    def deserialize(cls, buf, byteorder="@") -> LeanMinHash:
        """Read a LeanMinHash written by :meth:`serialize` (pyroid's or datasketch's).

        Buffers written before datasketch 2.0.0 come back with
        ``scheme="legacy"``.

        Raises:
            ValueError: An unknown scheme code.
        """
        lmh = _native_unpack(buf, byteorder)
        if lmh is not None:
            return lmh
        fmt_seed_size = "%sqi" % byteorder
        try:
            seed, num_perm = struct.unpack_from(fmt_seed_size, buf, 0)
        except TypeError:
            buf = memoryview(buf)
            seed, num_perm = struct.unpack_from(fmt_seed_size, buf, 0)
        if num_perm >= 0:
            scheme = _SCHEME_LEGACY
            offset = struct.calcsize(fmt_seed_size)
        else:
            num_perm = -num_perm
            (scheme_code,) = struct.unpack_from(
                byteorder + "B", buf, struct.calcsize(fmt_seed_size)
            )
            if scheme_code not in _SCHEME_CODES_INV:
                raise ValueError("Unknown permutation scheme code: %d" % scheme_code)
            scheme = _SCHEME_CODES_INV[scheme_code]
            # The 0-count value entry aligns the offset without consuming data
            # (only relevant for the native byte order "@").
            offset = struct.calcsize("%sqiB0%s" % (byteorder, _SCHEME_VALUE_FMTS[scheme]))
        fmt_hash = "%s%d%s" % (byteorder, num_perm, _SCHEME_VALUE_FMTS[scheme])
        hashvalues = struct.unpack_from(fmt_hash, buf, offset)
        lmh = object.__new__(LeanMinHash)
        lmh._initialize_slots(seed, hashvalues, scheme)
        return lmh

    def __getstate__(self):
        data = self._native_pack("@")
        if data is not None:
            return bytearray(data)
        buf = bytearray(self.bytesize())
        if self.scheme == _SCHEME_LEGACY:
            fmt = "qi%dI" % len(self)
            struct.pack_into(fmt, buf, 0, self.seed, len(self), *self.hashvalues)
        else:
            fmt = "qiB%d%s" % (len(self), self._value_fmt())
            struct.pack_into(
                fmt, buf, 0, self.seed, -len(self), _SCHEME_CODES[self.scheme], *self.hashvalues
            )
        return buf

    def __setstate__(self, buf):
        lmh = _native_unpack(buf, "@")
        if lmh is not None:
            self._initialize_slots(lmh.seed, lmh.hashvalues, lmh.scheme)
            return
        try:
            seed, num_perm = struct.unpack_from("qi", buf, 0)
        except TypeError:
            buf = memoryview(buf)
            seed, num_perm = struct.unpack_from("qi", buf, 0)
        if num_perm >= 0:
            scheme = _SCHEME_LEGACY
            offset = struct.calcsize("qi")
        else:
            num_perm = -num_perm
            (scheme_code,) = struct.unpack_from("B", buf, struct.calcsize("qi"))
            if scheme_code not in _SCHEME_CODES_INV:
                raise ValueError("Unknown permutation scheme code: %d" % scheme_code)
            scheme = _SCHEME_CODES_INV[scheme_code]
            offset = struct.calcsize("qiB0%s" % _SCHEME_VALUE_FMTS[scheme])
        hashvalues = struct.unpack_from(
            "%d%s" % (num_perm, _SCHEME_VALUE_FMTS[scheme]), buf, offset
        )
        self._initialize_slots(seed, hashvalues, scheme)

    def __hash__(self) -> int:
        return hash((self.scheme, self.seed, tuple(self.hashvalues)))

    @classmethod
    def union(cls, *lmhs: LeanMinHash) -> LeanMinHash:
        """Return a new LeanMinHash, the union of `lmhs` (at least two)."""
        if len(lmhs) < 2:
            raise ValueError("Cannot union less than 2 MinHash")
        num_perm = len(lmhs[0])
        seed = lmhs[0].seed
        scheme = lmhs[0].scheme
        if any((seed != m.seed or num_perm != len(m) or scheme != m.scheme) for m in lmhs):
            raise ValueError(
                "The unioning MinHash must have the same seed, number of permutation functions and scheme."
            )
        hashvalues = np.minimum.reduce([m.hashvalues for m in lmhs])

        lmh = object.__new__(LeanMinHash)
        lmh._initialize_slots(seed, hashvalues, scheme)
        return lmh
