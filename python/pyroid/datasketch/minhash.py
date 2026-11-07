# Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
# Error messages keep datasketch's %-formatting, so they stay byte-identical and the code
# stays easy to diff against upstream.
# ruff: noqa: UP031
"""datasketch's ``MinHash`` (``datasketch.minhash``), with the hashing in Rust.

Attributes, pickles, errors and every hash value are datasketch 2.0.0's.
Hashing with the default SHA1 functions, the permutations and the minimum
run natively; anything else (a custom ``hashfunc``, tokens that are not
``bytes``, hash values or permutations of another dtype) runs datasketch's
own numpy code. Types and docs are in ``minhash.pyi``.
"""

from __future__ import annotations

import copy
import itertools
import operator
import sys
import threading
import warnings
from collections.abc import Generator, Iterable
from typing import Any

import numpy as np

from pyroid._pyroid import datasketch as _native
from pyroid.datasketch._common import (
    _SCHEME_AFFINE32,
    _SCHEME_AFFINE64,
    _SCHEME_LEGACY,
    _VALID_SCHEMES,
    _check_scheme_consistency,  # noqa: F401  (datasketch.minhash has it; lsh.py imports it)
    warn_once,
)
from pyroid.datasketch.hashfunc import sha1_hash32, sha1_hash64

__all__ = ["MinHash", "minhash_batch"]

# The size of a hash value in number of bytes
hashvalue_byte_size = len(bytes(np.int64(42).data))

# http://en.wikipedia.org/wiki/Mersenne_prime
_mersenne_prime = np.uint64((1 << 61) - 1)
_max_hash = np.uint64((1 << 32) - 1)
_hash_range = 1 << 32

# Per-scheme storage dtype and maximum hash value (also the "empty" sentinel).
_SCHEME_DTYPES = {
    _SCHEME_LEGACY: np.uint64,
    _SCHEME_AFFINE32: np.uint32,
    _SCHEME_AFFINE64: np.uint64,
}
_SCHEME_MAX_HASHES = {
    _SCHEME_LEGACY: _max_hash,
    _SCHEME_AFFINE32: np.uint32((1 << 32) - 1),
    _SCHEME_AFFINE64: np.uint64((1 << 64) - 1),
}
_SCHEME_WIDTHS = {
    _SCHEME_AFFINE32: 32,
    _SCHEME_AFFINE64: 64,
}

# Scheme codes used in the LeanMinHash and bBitMinHash byte formats. Legacy
# payloads have no scheme field, so there is no code for the legacy scheme.
_SCHEME_CODES = {
    _SCHEME_AFFINE32: 1,
    _SCHEME_AFFINE64: 2,
}
_SCHEME_CODES_INV = {code: scheme for scheme, code in _SCHEME_CODES.items()}

# MurmurHash3 finalizer constants per width:
# (shift1, multiplier1, shift2, multiplier2, shift3).
_FMIX_CONSTANTS = {
    32: (
        np.uint32(16),
        np.uint32(0x85EBCA6B),
        np.uint32(13),
        np.uint32(0xC2B2AE35),
        np.uint32(16),
    ),
    64: (
        np.uint64(33),
        np.uint64(0xFF51AFD7ED558CCD),
        np.uint64(33),
        np.uint64(0xC4CEB9FE1A85EC53),
        np.uint64(33),
    ),
}

_NATIVE_SCHEMES = frozenset(_VALID_SCHEMES)

# Documents per native round of MinHash.bulk / MinHash.generator.
_BULK_CHUNK = 2048


def _fmix(hv: Any, width: int) -> Any:
    """Apply the MurmurHash3 finalizer, a fixed avalanching bijection on [0, 2^width)."""
    s1, m1, s2, m2, s3 = _FMIX_CONSTANTS[width]
    hv = hv ^ (hv >> s1)
    hv = hv * m1
    hv = hv ^ (hv >> s2)
    hv = hv * m2
    return hv ^ (hv >> s3)


_GPU_OK_CACHE: bool | None = None


def _gpu_available() -> bool:
    """Return whether CuPy and a CUDA device are available (imports CuPy on first use)."""
    global _GPU_OK_CACHE
    if _GPU_OK_CACHE is not None:
        return _GPU_OK_CACHE
    try:
        import cupy as cp
    except ImportError:
        _GPU_OK_CACHE = False
        return False
    try:
        _GPU_OK_CACHE = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        _GPU_OK_CACHE = False
    return bool(_GPU_OK_CACHE)


def _sha1_bits(hashfunc: Any) -> int:
    """32 or 64 for pyroid's or datasketch's ``sha1_hash32``/``sha1_hash64``, else 0."""
    if hashfunc is sha1_hash32:
        return 32
    if hashfunc is sha1_hash64:
        return 64
    real = sys.modules.get("datasketch.hashfunc")
    if real is not None and getattr(hashfunc, "__module__", None) == "datasketch.hashfunc":
        if hashfunc is getattr(real, "sha1_hash32", None):
            return 32
        if hashfunc is getattr(real, "sha1_hash64", None):
            return 64
    return 0


def _native_bits(scheme: Any, hashfunc: Any) -> int:
    """The SHA1 width the native kernels hash with for this state, or 0."""
    if scheme not in _NATIVE_SCHEMES:
        return 0
    bits = _sha1_bits(hashfunc)
    if bits == 64 and scheme == _SCHEME_AFFINE32:
        return 0  # datasketch raises (or not) per value: its own code decides
    return bits


def _warn_custom_hashfunc() -> None:
    warn_once(
        "MinHash:hashfunc",
        "pyroid.datasketch.MinHash: a custom hashfunc is called from Python for every "
        "token (slow path); only the permutations run in Rust. Pass no hashfunc (or "
        "sha1_hash32 / sha1_hash64) for the native path. This warning is shown once.",
    )


# The permutation parameters per (scheme, seed, num_perm), as numpy draws them.
_PERM_CACHE: dict[tuple[str, int, int], Any] = {}
_PERM_CACHE_MAX = 64
_PERM_LOCK = threading.Lock()


def _native_permutations(scheme: str, seed: int, num_perm: int) -> Any:
    key = (scheme, seed, num_perm)
    base = _PERM_CACHE.get(key)
    if base is None:
        base = _native.permutations(scheme, seed, num_perm)
        base.flags.writeable = False
        with _PERM_LOCK:
            while len(_PERM_CACHE) >= _PERM_CACHE_MAX:
                del _PERM_CACHE[next(iter(_PERM_CACHE))]
            _PERM_CACHE[key] = base
    if scheme == _SCHEME_LEGACY:
        # datasketch builds an (n, 2) array of (a, b) pairs and keeps its transpose.
        return base.copy().T
    return base.copy()


class MinHash:
    """MinHash: estimate the Jaccard similarity of sets (datasketch's ``MinHash``).

    See ``minhash.pyi`` for the arguments; they are datasketch 2.0.0's.
    """

    def __init__(
        self,
        num_perm: int = 128,
        seed: int = 1,
        gpu_mode: str = "disable",
        hashfunc: Any = None,
        hashobj: Any = None,  # Deprecated.
        hashvalues: Any = None,
        permutations: Any = None,
        scheme: str | None = None,
    ) -> None:
        if scheme is None:
            # Hash values and permutation parameters carry no trace of the
            # scheme that produced them (e.g. legacy values fit the affine32
            # range), so a default here would silently mislabel pre-2.0.0
            # state and defeat the cross-scheme comparison guards.
            if hashvalues is not None or permutations is not None:
                raise ValueError(
                    "scheme must be specified explicitly when initializing from existing "
                    "hash values or permutations: pass the scheme of the MinHash they came "
                    "from, or scheme='legacy' for values created by datasketch before 2.0.0."
                )
            scheme = _SCHEME_AFFINE32
        elif scheme not in _VALID_SCHEMES:
            raise ValueError(
                "scheme must be one of %s, got %r" % (", ".join(_VALID_SCHEMES), scheme)
            )
        self.scheme = scheme
        if hashvalues is not None:
            num_perm = len(hashvalues)
        if num_perm < 1:
            raise ValueError("num_perm must be positive")
        if num_perm > _hash_range:
            # Because 1) we don't want the size to be too large, and
            # 2) we are using 4 bytes to store the size value
            raise ValueError(
                "Cannot have more than %d number of permutation functions" % _hash_range
            )
        self.seed = seed
        self.num_perm = num_perm
        # Check the hash function.
        if hashfunc is None:
            hashfunc = sha1_hash64 if scheme == _SCHEME_AFFINE64 else sha1_hash32
        if not callable(hashfunc):
            raise ValueError("The hashfunc must be a callable.")
        self.hashfunc = hashfunc
        # Check for use of hashobj and issue warning.
        if hashobj is not None:
            warnings.warn(
                "hashobj is deprecated, use hashfunc instead.", DeprecationWarning, stacklevel=2
            )
        # Initialize hash values
        if hashvalues is not None:
            self.hashvalues = self._parse_hashvalues(hashvalues)
        else:
            self.hashvalues = self._init_hashvalues(num_perm)
        # Initalize permutation function parameters
        if permutations is not None:
            self.permutations = self._parse_permutations(permutations)
        else:
            self.permutations = self._init_permutations(num_perm)
        if len(self) != len(self.permutations[0]):
            raise ValueError("Numbers of hash values and permutations mismatch")

        # GPU state
        self._gpu_mode = gpu_mode
        self._a_gpu = None
        self._b_gpu = None

    def _init_hashvalues(self, num_perm: int) -> Any:
        if self.scheme == _SCHEME_LEGACY:
            return np.ones(num_perm, dtype=np.uint64) * _max_hash
        return np.full(num_perm, _SCHEME_MAX_HASHES[self.scheme], dtype=_SCHEME_DTYPES[self.scheme])

    def _init_permutations(self, num_perm: int) -> Any:
        seed = self.seed
        if type(num_perm) is int and isinstance(seed, (int, np.integer)):
            s = operator.index(seed)
            if 0 <= s <= 0xFFFFFFFF:
                return _native_permutations(self.scheme, s, num_perm)
        return self._init_permutations_numpy(num_perm)

    def _init_permutations_numpy(self, num_perm: int) -> Any:
        gen = np.random.RandomState(self.seed)
        if self.scheme == _SCHEME_LEGACY:
            # Create parameters for a random bijective permutation function
            # that maps a 32-bit hash value to another 32-bit hash value.
            # http://en.wikipedia.org/wiki/Universal_hashing
            return np.array(
                [
                    (
                        gen.randint(1, _mersenne_prime, dtype=np.uint64),
                        gen.randint(0, _mersenne_prime, dtype=np.uint64),
                    )
                    for _ in range(num_perm)
                ],
                dtype=np.uint64,
            ).T
        # Affine permutations h -> a * h + b mod 2^width. An odd `a` makes the
        # map bijective.
        width = _SCHEME_WIDTHS[self.scheme]
        dtype = _SCHEME_DTYPES[self.scheme]
        a = gen.randint(0, 1 << (width - 1), num_perm, dtype=dtype) * dtype(2) + dtype(1)
        b = gen.randint(0, 1 << width, num_perm, dtype=dtype)
        return np.array([a, b])

    def _to_scheme_array(self, values: Any, what: str) -> Any:
        """Convert values to an array of the scheme's dtype without loss."""
        if isinstance(values, np.ndarray):
            if (
                values.dtype.kind in "iu"
                and values.size
                and (
                    int(values.max()) > int(_SCHEME_MAX_HASHES[self.scheme])
                    or int(values.min()) < 0
                )
            ):
                raise ValueError("%s out of range for scheme %r" % (what, self.scheme))
            return values.astype(_SCHEME_DTYPES[self.scheme])
        try:
            return np.array(values, dtype=_SCHEME_DTYPES[self.scheme])
        except OverflowError as err:
            raise ValueError(
                "%s out of range for scheme %r: %s" % (what, self.scheme, err)
            ) from err

    def _parse_hashvalues(self, hashvalues: Any) -> Any:
        if self.scheme == _SCHEME_LEGACY:
            return np.array(hashvalues, dtype=np.uint64)
        return self._to_scheme_array(hashvalues, "hash values")

    def _parse_permutations(self, permutations: Any) -> Any:
        if self.scheme == _SCHEME_LEGACY:
            return permutations
        dtype = _SCHEME_DTYPES[self.scheme]
        if (
            isinstance(permutations, np.ndarray)
            and permutations.dtype == dtype
            and permutations.ndim == 2
            and permutations.shape[0] == 2
        ):
            # Already in canonical form (e.g. from copy()/union()/bulk()):
            # share the array across sketches like the legacy scheme does.
            parsed = permutations
        else:
            a = self._to_scheme_array(permutations[0], "permutation parameters")
            b = self._to_scheme_array(permutations[1], "permutation parameters")
            parsed = np.array([a, b])
        if np.any(parsed[0] & dtype(1) == 0):
            raise ValueError(
                "all `a` permutation parameters must be odd for scheme %r" % self.scheme
            )
        return parsed

    def _hash_input_array(self, hvs: Any) -> Any:
        """Convert hash function outputs to an array of the scheme's dtype."""
        try:
            return np.array(hvs, dtype=_SCHEME_DTYPES[self.scheme])
        except OverflowError as err:
            raise ValueError(
                "hashfunc returned a value out of range for scheme %r: %s. "
                "Use a hash function matching the scheme width, or scheme='affine64' "
                "for 64-bit hash values." % (self.scheme, err)
            ) from err

    def update(self, b: Any) -> None:
        """Update this MinHash with a new value (see ``minhash.pyi``)."""
        bits = _native_bits(self.scheme, self.hashfunc)
        if bits:
            new = _native.update_one(self.scheme, self.hashvalues, self.permutations, b, bits)
            if new is not None:
                self.hashvalues = new
                return
            hv = self.hashfunc(b)
        else:
            if not _sha1_bits(self.hashfunc):
                _warn_custom_hashfunc()
            hv = self.hashfunc(b)
            if self.scheme in _NATIVE_SCHEMES and type(hv) is int:
                new = _native.update_hashes(self.scheme, self.hashvalues, self.permutations, [hv])
                if new is not None:
                    self.hashvalues = new
                    return
        self._update_hash(hv)

    def _update_hash(self, hv: Any) -> None:
        # datasketch's MinHash.update after `hv = self.hashfunc(b)`.
        a, b = self.permutations
        if self.scheme == _SCHEME_LEGACY:
            phv = np.bitwise_and((a * hv + b) % _mersenne_prime, _max_hash)
        else:
            # A 1-element array rather than a scalar: numpy warns on scalar
            # overflow, while array arithmetic wraps silently as intended.
            hv = _fmix(self._hash_input_array([hv]), _SCHEME_WIDTHS[self.scheme])
            phv = a * hv + b
        self.hashvalues = np.minimum(phv, self.hashvalues)

    def _check_gpu(self) -> None:
        # pyroid computes on the CPU in every mode (the values are the same);
        # it keeps datasketch's error for a GPU that was required but is absent.
        if self._gpu_mode == "always" and not _gpu_available():
            raise RuntimeError("GPU mode 'always' requested but CuPy/CUDA device is unavailable.")

    def update_batch(self, b: Iterable[Any]) -> None:
        """Update this MinHash with new values (see ``minhash.pyi``)."""
        bits = _native_bits(self.scheme, self.hashfunc)
        if bits:
            tokens = b if type(b) is list or type(b) is tuple else list(b)
            # An empty batch is a no-op (the hash values stay the same object).
            if not tokens:
                return
            new = _native.update_tokens(
                self.scheme, self.hashvalues, self.permutations, tokens, bits
            )
            if new is not None:
                self._check_gpu()
                self.hashvalues = new
                return
            b = tokens
        elif not _sha1_bits(self.hashfunc):
            _warn_custom_hashfunc()
        # Hash on CPU to preserve hashfunc semantics
        hv_list = [self.hashfunc(_b) for _b in b]
        # Optimization: empty batch is a no-op (preserves original behavior)
        if not hv_list:
            return
        if self.scheme in _NATIVE_SCHEMES:
            new = _native.update_hashes(self.scheme, self.hashvalues, self.permutations, hv_list)
            if new is not None:
                self._check_gpu()
                self.hashvalues = new
                return
        self._update_hashes(hv_list)

    def _update_hashes(self, hv_list: list[Any]) -> None:
        # datasketch's MinHash.update_batch CPU path, after hashing.
        a, b = self.permutations
        self._check_gpu()
        if self.scheme == _SCHEME_LEGACY:
            hv = np.array(hv_list, dtype=np.uint64, ndmin=2).T
            phv = (hv * a + b) % _mersenne_prime
            phv = np.bitwise_and(phv, _max_hash)
        else:
            hv = self._hash_input_array(hv_list).reshape(-1, 1)
            hv = _fmix(hv, _SCHEME_WIDTHS[self.scheme])
            phv = hv * a + b
        self.hashvalues = np.minimum(self.hashvalues, phv.min(axis=0))

    def jaccard(self, other: Any) -> float:
        """Estimate the Jaccard similarity with ``other`` (see ``minhash.pyi``)."""
        if getattr(other, "scheme", None) != self.scheme:
            raise ValueError(
                "Cannot compute Jaccard given MinHash with different permutation schemes"
            )
        if other.seed != self.seed:
            raise ValueError("Cannot compute Jaccard given MinHash with different seeds")
        if len(self) != len(other):
            raise ValueError(
                "Cannot compute Jaccard given MinHash with different numbers of permutation "
                "functions"
            )
        return float(np.count_nonzero(self.hashvalues == other.hashvalues)) / float(len(self))

    def count(self) -> float:
        """Estimate the cardinality of the set (see ``minhash.pyi``)."""
        k = len(self)
        return (
            float(k)
            / np.sum(  # type: ignore[no-any-return]
                self.hashvalues / float(_SCHEME_MAX_HASHES[self.scheme])
            )
            - 1.0
        )

    def merge(self, other: Any) -> None:
        """Make this MinHash the union of itself and ``other`` (see ``minhash.pyi``)."""
        if getattr(other, "scheme", None) != self.scheme:
            raise ValueError("Cannot merge MinHash with different permutation schemes")
        if other.seed != self.seed:
            raise ValueError("Cannot merge MinHash with different seeds")
        if len(self) != len(other):
            raise ValueError("Cannot merge MinHash with different numbers of permutation functions")
        self.hashvalues = np.minimum(other.hashvalues, self.hashvalues)

    def digest(self) -> Any:
        """Return a copy of the hash values (see ``minhash.pyi``)."""
        return copy.copy(self.hashvalues)

    def is_empty(self) -> bool:
        """True if the MinHash is in its initial state (see ``minhash.pyi``)."""
        return not np.any(self.hashvalues != _SCHEME_MAX_HASHES[self.scheme])

    def clear(self) -> None:
        """Reset the hash values (see ``minhash.pyi``)."""
        self.hashvalues = self._init_hashvalues(len(self))

    def copy(self) -> MinHash:
        """Return a copy; preserves gpu_mode (rehydrates caches lazily)."""
        return MinHash(
            seed=self.seed,
            hashfunc=self.hashfunc,
            hashvalues=self.digest(),
            permutations=self.permutations,
            gpu_mode=self._gpu_mode,
            scheme=self.scheme,
        )

    def __len__(self) -> int:
        return len(self.hashvalues)

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is type(other)
            and self.scheme == other.scheme  # type: ignore[attr-defined]
            and self.seed == other.seed  # type: ignore[attr-defined]
            and np.array_equal(self.hashvalues, other.hashvalues)  # type: ignore[attr-defined]
        )

    @classmethod
    def union(cls, *mhs: Any) -> Any:
        """Return a new MinHash, the union of ``mhs`` (see ``minhash.pyi``)."""
        if len(mhs) < 2:
            raise ValueError("Cannot union less than 2 MinHash")
        num_perm = len(mhs[0])
        seed = mhs[0].seed
        scheme = mhs[0].scheme
        if any((seed != m.seed or num_perm != len(m) or scheme != m.scheme) for m in mhs):
            raise ValueError(
                "The unioning MinHash must have the same seed, number of permutation functions "
                "and scheme"
            )
        hashvalues = np.minimum.reduce([m.hashvalues for m in mhs])
        permutations = mhs[0].permutations
        return cls(
            num_perm=num_perm,
            seed=seed,
            hashfunc=mhs[0].hashfunc,
            hashvalues=hashvalues,
            permutations=permutations,
            gpu_mode=mhs[0]._gpu_mode,
            scheme=scheme,
        )

    @classmethod
    def bulk(cls, b: Iterable[Any], *, threads: int | None = None, **minhash_kwargs: Any) -> Any:
        """Compute MinHashes in bulk (see ``minhash.pyi``)."""
        return list(cls.generator(b, threads=threads, **minhash_kwargs))

    @classmethod
    def generator(
        cls, b: Iterable[Any], *, threads: int | None = None, **minhash_kwargs: Any
    ) -> Generator[Any, None, None]:
        """Compute MinHashes in a generator (see ``minhash.pyi``)."""
        m = cls(**minhash_kwargs)
        bits = _native_bits(m.scheme, m.hashfunc)
        if type(m) is not MinHash or not bits or m._gpu_mode == "always":
            for _b in b:
                _m = m.copy()
                _m.update_batch(_b)
                yield _m
            return
        it = iter(b)
        template: dict[str, Any] | None = None
        while True:
            docs = list(itertools.islice(it, _BULK_CHUNK))
            if not docs:
                return
            if template is None:
                # datasketch copies `m` for every document: the first copy
                # validates the state and gives the attributes every copy has.
                template = dict(m.copy().__dict__)
            res = _native.bulk(
                m.scheme, template["hashvalues"], m.permutations, docs, bits, threads=threads
            )
            if res is None:
                # Not the canonical state after all: datasketch's own loop.
                for _b in itertools.chain(docs, it):
                    _m = m.copy()
                    _m.update_batch(_b)
                    yield _m
                return
            rows, fallback = res
            slow = set(fallback)
            for i, doc in enumerate(docs):
                if i in slow:
                    _m = m.copy()
                    _m.update_batch(doc)
                    yield _m
                    continue
                _m = MinHash.__new__(MinHash)
                _m.__dict__.update(template)
                _m.hashvalues = rows[i].copy()
                yield _m

    # NOTE: datasketch drops CuPy device state on pickle, so round-tripped
    # MinHash objects remain portable; pyroid never creates any.
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Drop device arrays; keep gpu_mode as-is (portable across envs).
        state["_a_gpu"] = None
        state["_b_gpu"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # MinHash pickled before version 2.0.0 predates permutation schemes
        # and always used the legacy scheme.
        state.setdefault("scheme", _SCHEME_LEGACY)
        self.__dict__.update(state)


def minhash_batch(
    docs: Any,
    *,
    num_perm: int = 128,
    seed: int = 1,
    scheme: str | None = None,
    ngram: int = 1,
    threads: int | None = None,
) -> Any:
    """Compute MinHash signatures for many documents at once (see ``minhash.pyi``)."""
    m = MinHash(num_perm=num_perm, seed=seed, scheme=scheme)
    bits = 64 if m.scheme == _SCHEME_AFFINE64 else 32
    if (
        isinstance(docs, str)
        or hasattr(docs, "__arrow_c_stream__")
        or hasattr(docs, "__arrow_c_array__")
        or (isinstance(docs, (list, tuple)) and docs and isinstance(docs[0], str))
    ):
        n = operator.index(ngram)
        if n < 1:
            raise ValueError(f"ngram must be a positive int, got {n}")
        return _native.batch_texts(m.scheme, m.permutations, docs, n, threads=threads)
    if operator.index(ngram) != 1:
        raise ValueError("ngram applies to texts; token lists are hashed as given")
    if isinstance(docs, (list, tuple)):
        return _native.batch_tokens(
            m.scheme, m.hashvalues, m.permutations, list(docs), bits, threads=threads
        )
    from pyroid._pyroid import InputError

    raise InputError(
        "expected texts (str, list[str], tuple[str, ...], an Arrow string array) or a "
        f"list/tuple of token lists, got {type(docs).__name__}"
    )
