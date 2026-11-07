"""pyroid.datasketch's MinHash, LeanMinHash and minhash_batch against live datasketch 2.0.0.

Most tests run one scenario twice, with ``pyroid.datasketch`` and with ``datasketch``
(``both(fn)`` calls ``fn(lib)`` with each module), and require the same outcome: the same
attributes in the same order (``vars()``), hash values and permutations of the same dtype and
values, permutations with the same memory layout (contiguity, ``.base``), the same objects
shared between MinHash, and the same exception type and message. Hypothesis draws seeds,
sizes and tokens (``HYPOTHESIS_PROFILE=ci`` for more examples).

``minhash_batch`` (a pyroid extension) is checked against ``datasketch.MinHash.bulk`` of the
same shingles: the word n-grams of ``str.split()``, UTF-8 encoded.
"""

from __future__ import annotations

import array
import base64
import io
import pickle
import struct
import sys
import warnings
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any, Literal

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

import pyroid
import pyroid.datasketch as pds
import pyroid.datasketch.hashfunc
from pyroid._pyroid import _testing
from pyroid.datasketch import LeanMinHash, MinHash, _common, minhash_batch

ds: Any = pytest.importorskip("datasketch")
if ds.__version__ != "2.0.0":
    pytest.skip(f"needs datasketch 2.0.0, found {ds.__version__}", allow_module_level=True)

# Custom hash functions warn once per process; the one test that checks it resets the state.
pytestmark = pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning")
# numpy warns on some casts (floats, out-of-range values), in both libraries alike.
numpy_casts = pytest.mark.filterwarnings("ignore::RuntimeWarning")

Scheme = Literal["affine32", "affine64", "legacy"]
SCHEMES: tuple[Scheme, ...] = ("affine32", "affine64", "legacy")
DTYPES: dict[Scheme, type[np.unsignedinteger[Any]]] = {
    "affine32": np.uint32,
    "affine64": np.uint64,
    "legacy": np.uint64,
}
BYTEORDERS = "@=<>!"

# ---------------------------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------------------------


def outcome(f: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, ...]:
    """``("ok", result)``, or ``("raise", exception type, message)``."""
    try:
        return ("ok", f(*args, **kwargs))
    except Exception as e:
        return ("raise", type(e), str(e))


def drain(it: Iterator[Any]) -> tuple[list[Any], tuple[Any, ...] | None]:
    """The items an iterator yields, and the error that stops it (or None)."""
    out: list[Any] = []
    try:
        for x in it:
            out.append(x)  # noqa: PERF402  (the error must come after the items)
    except Exception as e:
        return out, (type(e), str(e))
    return out, None


def identity_pattern(objs: list[Any]) -> list[int]:
    """For each object, the index of the first object in `objs` that it *is*."""
    return [next(j for j, y in enumerate(objs) if y is x) for x in objs]


def _layout(a: NDArray[Any]) -> tuple[Any, ...]:
    flags = a.flags
    return (a.dtype, a.shape, flags.c_contiguous, flags.f_contiguous, flags.writeable)


def assert_same_array(p: Any, d: Any, *, structure: bool = False) -> None:
    assert type(p) is type(d), (type(p), type(d))
    assert p.dtype == d.dtype, (p.dtype, d.dtype)
    assert p.shape == d.shape, (p.shape, d.shape)
    assert np.array_equal(p, d), (p, d)
    if structure:
        assert _layout(p) == _layout(d)
        assert p.flags.owndata == d.flags.owndata
        assert type(p.base) is type(d.base), (p.base, d.base)
        if isinstance(d.base, np.ndarray):
            assert _layout(p.base) == _layout(d.base)
            assert type(p.base.base) is type(d.base.base)


def assert_same_hashfunc(p: Any, d: Any) -> None:
    if getattr(d, "__module__", None) == "datasketch.hashfunc" and p is not d:
        assert p is getattr(pyroid.datasketch.hashfunc, d.__name__), (p, d)
    else:
        assert p is d, (p, d)


def assert_same_minhash(p: Any, d: Any) -> None:
    assert type(p).__qualname__ == type(d).__qualname__, (type(p), type(d))
    assert type(p).__module__ == "pyroid." + type(d).__module__
    assert list(vars(p)) == list(vars(d))
    assert_same(p.scheme, d.scheme)
    assert_same(p.seed, d.seed)
    assert_same_array(p.hashvalues, d.hashvalues)
    for key, want in vars(d).items():
        got = vars(p)[key]
        if key == "hashfunc":
            assert_same_hashfunc(got, want)
        elif key == "permutations":
            assert_same_permutations(got, want)
        else:
            assert_same(got, want)


def assert_same_permutations(p: Any, d: Any) -> None:
    if isinstance(d, np.ndarray):
        assert_same_array(p, d, structure=True)
        assert_same_array(p[0], d[0], structure=True)
    else:
        assert_same(p, d)


def assert_same(p: Any, d: Any) -> None:
    """Assert that pyroid's `p` and datasketch's `d` are the same value (recursively)."""
    if isinstance(d, ds.MinHash):
        assert_same_minhash(p, d)
    elif isinstance(d, np.ndarray):
        assert_same_array(p, d)
    elif isinstance(d, (list, tuple)):
        assert type(p) is type(d), (p, d)
        assert len(p) == len(d), (p, d)
        for x, y in zip(p, d, strict=True):
            assert_same(x, y)
    elif isinstance(d, dict):
        assert type(p) is type(d)
        assert list(p) == list(d)
        for k in d:
            assert_same(p[k], d[k])
    elif callable(d) and not isinstance(d, type):
        assert_same_hashfunc(p, d)
    else:
        assert type(p) is type(d), (p, d)
        assert p == d or (p != p and d != d), (p, d)  # NaN == NaN here


def both(fn: Callable[..., Any], *args: Any) -> tuple[Any, Any]:
    """Run ``fn(lib, *args)`` with pyroid.datasketch and datasketch; assert the same outcome."""
    got = outcome(fn, pds, *args)
    want = outcome(fn, ds, *args)
    assert_same(got, want)
    return got, want


class _Remap(pickle.Unpickler):
    """Unpickle with the classes and functions of one library resolved in the other."""

    def __init__(self, data: bytes, to_pyroid: bool) -> None:
        super().__init__(io.BytesIO(data))
        self.to_pyroid = to_pyroid

    def find_class(self, module: str, name: str) -> Any:
        if self.to_pyroid and module.split(".")[0] == "datasketch":
            module = "pyroid." + module
        elif not self.to_pyroid and module.startswith("pyroid.datasketch"):
            module = module.removeprefix("pyroid.")
        return super().find_class(module, name)


def to_pyroid(data: bytes) -> Any:
    return _Remap(data, True).load()


def to_datasketch(data: bytes) -> Any:
    return _Remap(data, False).load()


# ---------------------------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------------------------

SCHEME = st.sampled_from(SCHEMES)
SEED = st.integers(0, 2**32 - 1)
NUM_PERM = st.integers(1, 96)
# SHA1 pads in 64-byte blocks: lengths around 55/56 and 64 change the number of blocks.
EDGE_BYTES = [b"", b"x" * 55, b"x" * 56, b"y" * 63, b"y" * 64, b"z" * 65, b"\x00", b"dup", b"dup"]
BYTES = st.one_of(
    st.binary(max_size=8),
    st.binary(min_size=50, max_size=140),
    st.sampled_from(EDGE_BYTES),
)
TOKENS = st.lists(BYTES, max_size=40)
OTHER_TOKENS = st.one_of(
    BYTES.map(bytearray),
    BYTES.map(lambda b: memoryview(b)),
    BYTES.map(np.bytes_),
    st.integers(0, 2**32 - 1).map(np.uint32),
    st.integers(-(2**63), 2**63 - 1).map(np.int64),
    st.floats(allow_nan=False).map(np.float64),
)
MIXED_TOKENS = st.lists(st.one_of(BYTES, OTHER_TOKENS), max_size=30)
DOCS = st.lists(TOKENS, max_size=12)
NUMPY_INT_TYPES: list[type[np.signedinteger[Any] | np.unsignedinteger[Any]]] = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]


@st.composite
def numpy_seeds(draw: st.DrawFn) -> Any:
    ty = draw(st.sampled_from(NUMPY_INT_TYPES))
    info = np.iinfo(ty)
    return ty(draw(st.integers(max(0, int(info.min)), min(2**32 - 1, int(info.max)))))


ANY_SEED = st.one_of(SEED, numpy_seeds(), st.booleans(), st.just(np.True_))

# ---------------------------------------------------------------------------------------------
# Permutations
# ---------------------------------------------------------------------------------------------


@given(seed=ANY_SEED, num_perm=st.integers(1, 600), scheme=SCHEME)
def test_permutations_match(seed: Any, num_perm: int, scheme: Scheme) -> None:
    both(lambda lib: lib.MinHash(num_perm=num_perm, seed=seed, scheme=scheme))


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize(
    "seed",
    [
        -1,
        2**32,
        2**64,
        np.int64(-1),
        np.uint64(2**32),
        np.int8(-5),
        1.0,
        1.5,
        np.float64(2.0),
        "1",
        b"1",
        1j,
        [1, -2],
        np.array([2**33]),
        object(),
    ],
    ids=repr,
)
def test_bad_seeds_raise_as_datasketch(seed: Any, scheme: Scheme) -> None:
    got, _ = both(lambda lib: lib.MinHash(num_perm=4, seed=seed, scheme=scheme))
    assert got[0] == "raise"


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize(
    "seed", [[1, 2, 3], (7,), np.array([5, 6], dtype=np.uint32), np.arange(10)], ids=repr
)
def test_sequence_seeds_match(seed: Any, scheme: Scheme) -> None:
    # numpy's RandomState takes sequences as seeds: the values are deterministic too.
    both(lambda lib: lib.MinHash(num_perm=7, seed=seed, scheme=scheme))


@pytest.mark.parametrize("scheme", SCHEMES)
def test_seed_none_draws_fresh_permutations(scheme: Scheme) -> None:
    p1, p2 = MinHash(16, seed=None, scheme=scheme), MinHash(16, seed=None, scheme=scheme)  # type: ignore[arg-type]
    d = ds.MinHash(16, seed=None, scheme=scheme)
    assert p1.seed is None
    assert list(vars(p1)) == list(vars(d))
    assert_same_array(p1.hashvalues, d.hashvalues)
    assert _layout(p1.permutations) == _layout(d.permutations)
    assert (p1.permutations.base is None) == (d.permutations.base is None)
    assert not np.array_equal(p1.permutations, p2.permutations)  # entropy, not a cached seed


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize(
    "num_perm", [0, -1, 1.0, 2.5, "3", None, np.int64(5), np.uint8(3), True, 2**32 + 1], ids=repr
)
def test_num_perm_values_match(num_perm: Any, scheme: Scheme) -> None:
    both(lambda lib: lib.MinHash(num_perm=num_perm, seed=3, scheme=scheme))


@pytest.mark.parametrize("scheme", SCHEMES)
def test_cached_permutations_are_not_shared(scheme: Scheme) -> None:
    # pyroid caches the permutation parameters per seed: each MinHash must still get its own.
    a = MinHash(8, 5, scheme=scheme)
    a.permutations[0][0] = 3
    a.permutations[1][:] = 7
    b = MinHash(8, 5, scheme=scheme)
    assert_same_permutations(b.permutations, ds.MinHash(8, 5, scheme=scheme).permutations)
    assert b.permutations is not a.permutations
    assert not np.shares_memory(a.permutations, b.permutations)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_bad_constructor_arguments_raise_as_datasketch(scheme: Scheme) -> None:
    perms = ds.MinHash(4, 1, scheme=scheme).permutations
    cases: list[dict[str, Any]] = [
        {"scheme": "affine16"},
        {"scheme": 32},
        {"hashfunc": 5},
        {"hashfunc": "sha1"},
        {"hashvalues": [1, 2, 3]},
        {"permutations": perms},
        {"hashvalues": [1, 2, 3], "permutations": perms, "scheme": scheme},
        {"num_perm": 8, "permutations": perms, "scheme": scheme},
        {"hashvalues": [], "scheme": scheme},
        {"hashvalues": [-1], "scheme": scheme},
        {"hashvalues": [2**64], "scheme": scheme},
        {"hashvalues": np.array([2**40], dtype=np.uint64), "scheme": scheme},
        {"permutations": ([2, 3], [0, 1]), "num_perm": 2, "scheme": scheme},
        {"permutations": ([-1, 3], [0, 1]), "num_perm": 2, "scheme": scheme},
    ]
    for kwargs in cases:
        both(lambda lib, kw: lib.MinHash(**kw), kwargs)


@pytest.mark.thread_unsafe(reason="pytest.warns")
def test_hashobj_warns_as_datasketch() -> None:
    for lib in (pds, ds):
        with pytest.warns(DeprecationWarning, match="hashobj is deprecated") as rec:
            lib.MinHash(4, hashobj=object())
        assert [w.filename for w in rec] == [__file__]


# ---------------------------------------------------------------------------------------------
# update / update_batch
# ---------------------------------------------------------------------------------------------


def _feed(lib: Any, kwargs: dict[str, Any], tokens: list[Any], how: str) -> Any:
    m = lib.MinHash(**kwargs)
    before = m.hashvalues
    if how == "update":
        res = [outcome(m.update, t) for t in tokens]
    elif how == "list":
        res = [outcome(m.update_batch, tokens)]
    elif how == "tuple":
        res = [outcome(m.update_batch, tuple(tokens))]
    elif how == "generator":
        res = [outcome(m.update_batch, (t for t in tokens))]
    else:  # two halves
        half = len(tokens) // 2
        res = [outcome(m.update_batch, tokens[:half]), outcome(m.update_batch, tokens[half:])]
    return res, m, m.hashvalues is before


HOW = st.sampled_from(["update", "list", "tuple", "generator", "halves"])


def _no_tokens() -> Iterator[bytes]:
    yield from ()


@given(tokens=TOKENS, scheme=SCHEME, num_perm=NUM_PERM, seed=SEED, how=HOW)
def test_update_bytes_match(
    tokens: list[bytes], scheme: Scheme, num_perm: int, seed: int, how: str
) -> None:
    kwargs = {"num_perm": num_perm, "seed": seed, "scheme": scheme}
    both(lambda lib: _feed(lib, kwargs, tokens, how))


@given(tokens=MIXED_TOKENS, scheme=SCHEME, how=HOW)
def test_update_other_buffers_match(tokens: list[Any], scheme: Scheme, how: str) -> None:
    # bytearray, memoryview and numpy scalars are hashed through the buffer protocol.
    both(lambda lib: _feed(lib, {"num_perm": 16, "scheme": scheme}, tokens, how))


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("how", ["update", "list", "tuple", "generator", "halves"])
@pytest.mark.parametrize(
    "bad", ["str", np.str_("np.str_"), 5, None, 1.5, [b"nested"], object()], ids=repr
)
def test_unhashable_token_raises_as_datasketch(bad: Any, how: str, scheme: Scheme) -> None:
    tokens = [b"a", b"b", bad, b"c"]
    got, _ = both(lambda lib: _feed(lib, {"num_perm": 8, "scheme": scheme}, tokens, how))
    assert any(r[0] == "raise" for r in got[1][0])


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize(
    "batch",
    [b"abc", "abc", {b"a": 1}, np.array([b"a", b"bc", b"a\x00"]), np.array([1, 2], np.uint32)],
    ids=repr,
)
def test_update_batch_other_iterables_match(batch: Any, scheme: Scheme) -> None:
    def run(lib: Any) -> Any:
        m = lib.MinHash(8, scheme=scheme)
        return outcome(m.update_batch, batch), m

    both(run)


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("gpu_mode", ["disable", "detect", "always"])
@pytest.mark.parametrize(
    "empty", [[], (), "iter", "generator", "array"], ids=["list", "tuple", "iter", "gen", "ndarray"]
)
def test_empty_batch_keeps_the_hash_values_object(
    empty: Any, gpu_mode: str, scheme: Scheme
) -> None:
    def run(lib: Any) -> Any:
        m = lib.MinHash(8, scheme=scheme, gpu_mode=gpu_mode)
        m.update(b"x")
        before = m.hashvalues
        batch: Any = (
            {
                "iter": iter([]),
                "generator": _no_tokens(),
                "array": np.array([], dtype="S1"),
            }.get(empty, empty)
            if isinstance(empty, str)
            else empty
        )
        m.update_batch(batch)
        return m, m.hashvalues is before

    got, _ = both(run)
    assert got[1][1] is True


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("hashfunc", ["sha1_hash32", "sha1_hash64"])
@pytest.mark.parametrize("how", ["update", "list", "generator"])
def test_sha1_hashfuncs_with_every_scheme(hashfunc: str, how: str, scheme: Scheme) -> None:
    # affine32 + sha1_hash64 raises datasketch's ValueError (a value above 2**32).
    tokens = [b"a", b"bb", b"c" * 70]

    def run(lib: Any) -> Any:
        kwargs = {
            "num_perm": 12,
            "seed": 9,
            "scheme": scheme,
            "hashfunc": getattr(lib.hashfunc, hashfunc),
        }
        return _feed(lib, kwargs, tokens, how)

    got, _ = both(run)
    if scheme == "affine32" and hashfunc == "sha1_hash64":
        assert got[1][0][0][:2] == ("raise", ValueError)
        assert "out of range for scheme 'affine32'" in got[1][0][0][2]


@pytest.mark.parametrize("scheme", SCHEMES)
@given(tokens=TOKENS)
def test_datasketch_hashfunc_objects_are_accepted(scheme: Scheme, tokens: list[bytes]) -> None:
    # datasketch's own sha1_hash32/64 (not pyroid's) given explicitly: same values, same object.
    for name in ("sha1_hash32", "sha1_hash64"):
        if scheme == "affine32" and name == "sha1_hash64":
            continue
        f = getattr(ds.hashfunc, name)
        p = MinHash(16, 2, scheme=scheme, hashfunc=f)
        d = ds.MinHash(16, 2, scheme=scheme, hashfunc=f)
        p.update_batch(tokens)
        d.update_batch(tokens)
        assert p.hashfunc is f
        assert_same_minhash(p, d)
        assert_same(
            MinHash.bulk([tokens], num_perm=16, hashfunc=f, scheme=scheme),
            ds.MinHash.bulk([tokens], num_perm=16, hashfunc=f, scheme=scheme),
        )


def _const(v: Any) -> Callable[[Any], Any]:
    return lambda _b: v


CUSTOM_VALUES = [
    0,
    5,
    2**32 - 1,
    2**32,
    2**40,
    2**63,
    2**64 - 1,
    2**64,
    2**70,
    -1,
    -(2**63),
    True,
    np.uint32(7),
    np.int64(-3),
    np.uint64(2**64 - 1),
    1.5,
    float("inf"),
    float("nan"),
    None,
    "s",
]


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("how", ["update", "list", "generator", "halves"])
@pytest.mark.parametrize("value", CUSTOM_VALUES, ids=repr)
@numpy_casts
def test_custom_hashfunc_values_match(value: Any, how: str, scheme: Scheme) -> None:
    f = _const(value)
    tokens = [b"a", b"b"]
    both(lambda lib: _feed(lib, {"num_perm": 6, "scheme": scheme, "hashfunc": f}, tokens, how))


@given(hashes=st.lists(st.integers(0, 2**64 - 1), max_size=30), scheme=SCHEME, how=HOW)
def test_custom_int_hashfunc_matches(hashes: list[int], scheme: Scheme, how: str) -> None:
    table = dict(enumerate(hashes))
    kwargs = {"num_perm": 24, "seed": 4, "scheme": scheme, "hashfunc": table.__getitem__}
    both(lambda lib: _feed(lib, kwargs, list(table), how))


def test_custom_hashfunc_exception_propagates() -> None:
    def boom(b: Any) -> int:
        raise KeyError(b)

    for how in ("update", "list", "generator"):
        both(lambda lib, how: _feed(lib, {"num_perm": 4, "hashfunc": boom}, [b"x"], how), how)


@pytest.mark.thread_unsafe(reason="resets the one-time-warning state")
def test_custom_hashfunc_warns_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_common, "_WARNED", set())
    m = MinHash(8, hashfunc=_const(5))
    with pytest.warns(pyroid.FallbackWarning, match="custom hashfunc") as rec:
        m.update(b"x")
    assert len(rec) == 1
    assert rec[0].filename == __file__  # attributed to the caller
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m.update(b"y")
        m.update_batch([b"z"])
        MinHash.bulk([[b"a"]], num_perm=4, hashfunc=_const(9))
    monkeypatch.setattr(_common, "_WARNED", set())
    with pytest.warns(pyroid.FallbackWarning, match="custom hashfunc"):
        MinHash(8, hashfunc=_const(5)).update_batch([b"x"])


@pytest.mark.thread_unsafe(reason="resets the one-time-warning state")
def test_sha1_hashfuncs_do_not_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_common, "_WARNED", set())
    funcs = [
        None,
        pds.sha1_hash32,
        pds.sha1_hash64,
        ds.hashfunc.sha1_hash32,
        ds.hashfunc.sha1_hash64,
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for scheme in SCHEMES:
            for f in funcs:
                m = MinHash(4, scheme=scheme, hashfunc=f)
                # affine32 + sha1_hash64 raises ValueError, but it is not a fallback.
                outcome(m.update, b"a")
                outcome(m.update_batch, [b"b"])
                outcome(MinHash.bulk, [[b"c"]], num_perm=4, scheme=scheme, hashfunc=f)
    assert [str(w.message) for w in caught] == []


# ---------------------------------------------------------------------------------------------
# User-provided hash values and permutations
# ---------------------------------------------------------------------------------------------


def _initial_values(scheme: Scheme, n: int) -> list[Any]:
    base = ds.MinHash(n, 3, scheme=scheme)
    base.update_batch([b"seed", b"values"])
    hv = base.hashvalues
    return [
        hv,
        hv.tolist(),
        tuple(hv.tolist()),
        hv.astype(np.int64),
        hv.astype(np.float64),
        hv.astype(">u8"),
        np.asfortranarray(hv),
        np.repeat(hv, 2)[::2],
    ]


def _permutation_forms(scheme: Scheme, n: int) -> list[Any]:
    perm = ds.MinHash(n, 3, scheme=scheme).permutations
    a, b = perm[0], perm[1]
    forms = [
        perm,
        (a, b),
        [a, b],
        np.ascontiguousarray(perm),
        np.asfortranarray(perm),
        perm.astype(np.uint64),
        perm.astype(np.float64),
        np.stack([a, b]).astype(">u8"),
    ]
    if scheme != "legacy":
        # The legacy scheme keeps lists as they are, and update() then computes
        # `list * hash value`: a list of billions of items, in datasketch too.
        forms.append((a.tolist(), b.tolist()))
    return forms


@pytest.mark.parametrize("scheme", SCHEMES)
@numpy_casts
def test_user_hashvalues_and_permutations_match(scheme: Scheme) -> None:
    n = 6
    tokens = [b"t1", b"t2", b"t3" * 30]
    for hv in _initial_values(scheme, n):
        for perm in _permutation_forms(scheme, n):

            def run(lib: Any, hv: Any = hv, perm: Any = perm) -> Any:
                kwargs = {"seed": 3, "scheme": scheme, "hashvalues": hv, "permutations": perm}
                m = lib.MinHash(**kwargs)
                shared = m.permutations is perm
                results = [_feed(lib, kwargs, tokens, how) for how in ("update", "list")]
                bulk = drain(lib.MinHash.generator([tokens, [], tokens[:1]], **kwargs))
                return m, shared, results, bulk, m.copy()

            both(run)


@pytest.mark.parametrize("scheme", SCHEMES)
@numpy_casts
def test_reassigned_attributes_run_datasketch_code(scheme: Scheme) -> None:
    # Hash values and permutations assigned after __init__, in non-canonical forms.
    n = 5
    tokens = [b"q", b"r"]
    hv_forms: list[Callable[[Any], Any]] = [
        lambda hv: hv.astype(np.int64),
        lambda hv: hv.astype(">u8" if hv.dtype == np.uint64 else ">u4"),
        lambda hv: np.repeat(hv, 2)[::2],
        lambda hv: hv[:3],
        lambda hv: hv.tolist(),
        lambda hv: np.array(hv, copy=True).reshape(1, -1),
    ]
    perm_forms: list[Callable[[Any], Any]] = [
        lambda pm: pm.astype(np.int64),
        lambda pm: (pm[0], pm[1]),
        lambda pm: pm[:, :3],
        lambda pm: pm.astype(np.uint32) if pm.dtype == np.uint64 else pm.astype(np.uint64),
        lambda pm: pm.T.copy(),
    ]
    for hv_form in hv_forms:
        for how in ("update", "list", "generator"):

            def run(lib: Any, f: Any = hv_form, how: str = how) -> Any:
                m = lib.MinHash(n, 2, scheme=scheme)
                m.hashvalues = f(m.hashvalues)
                if how == "update":
                    return [outcome(m.update, t) for t in tokens], m
                batch = tokens if how == "list" else iter(tokens)
                return outcome(m.update_batch, batch), m

            both(run)
    for perm_form in perm_forms:
        for how in ("update", "list"):

            def run(lib: Any, f: Any = perm_form, how: str = how) -> Any:
                m = lib.MinHash(n, 2, scheme=scheme)
                m.permutations = f(m.permutations)
                if how == "update":
                    return [outcome(m.update, t) for t in tokens], m
                return outcome(m.update_batch, tokens), m

            both(run)


class _SubArray(np.ndarray):
    pass


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize(
    "view",
    [lambda a: a.view(_SubArray), lambda a: np.ma.masked_array(a, mask=np.arange(a.size) == 1)],
    ids=["subclass", "masked"],
)
def test_ndarray_subclass_state_runs_datasketch_code(view: Any, scheme: Scheme) -> None:
    # numpy keeps the subclass (and a mask) through datasketch's arithmetic.
    def run(lib: Any) -> Any:
        out = []
        for attr in ("hashvalues", "permutations"):
            for how in ("update", "list"):
                m = lib.MinHash(4, scheme=scheme)
                setattr(m, attr, view(getattr(m, attr)))
                if how == "update":
                    m.update(b"x")
                else:
                    m.update_batch([b"x", b"y"])
                out.append((type(m.hashvalues).__name__, np.ma.getmaskarray(m.hashvalues), m))
        return out

    both(run)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_read_only_state_is_not_written(scheme: Scheme) -> None:
    def run(lib: Any) -> Any:
        m = lib.MinHash(8, scheme=scheme)
        m.update(b"a")
        m.hashvalues.flags.writeable = False
        m.permutations.flags.writeable = False
        frozen = m.hashvalues
        m.update(b"b")
        m.update_batch([b"c", b"d"])
        return m, frozen

    both(run)


# ---------------------------------------------------------------------------------------------
# gpu_mode
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("gpu_mode", ["always", "detect", "disable", "something-else"])
def test_gpu_modes_match(gpu_mode: str, scheme: Scheme) -> None:
    # No CuPy here: "always" raises RuntimeError in update_batch, after hashing (a hashing
    # error comes first) and before converting hash values (an out-of-range one comes after).
    cases: list[tuple[Any, list[Any]]] = [
        (None, [b"a", b"b"]),
        (None, []),
        (None, [b"a", "str"]),
        (_const(2**70), [b"a"]),
        (_const(-1), [b"a"]),
        (pds.sha1_hash64, [b"a"]),
    ]
    for hashfunc, tokens in cases:
        for how in ("update", "list", "generator"):

            def run(
                lib: Any, hashfunc: Any = hashfunc, tokens: Any = tokens, how: str = how
            ) -> Any:
                if hashfunc is pds.sha1_hash64:
                    hashfunc = lib.hashfunc.sha1_hash64
                kwargs = {
                    "num_perm": 4,
                    "scheme": scheme,
                    "gpu_mode": gpu_mode,
                    "hashfunc": hashfunc,
                }
                return _feed(lib, kwargs, tokens, how)

            both(run)
    docs = [[b"a"], [], [b"b"]]
    both(
        lambda lib: drain(lib.MinHash.generator(docs, num_perm=4, scheme=scheme, gpu_mode=gpu_mode))
    )


@pytest.mark.parametrize("gpu_mode", ["always", "detect"])
def test_gpu_mode_survives_copy_union_and_pickle(gpu_mode: str) -> None:
    def run(lib: Any) -> Any:
        m = lib.MinHash(8, gpu_mode=gpu_mode)
        m.update(b"a")
        u = lib.MinHash.union(m, m.copy())
        p = pickle.loads(pickle.dumps(m))
        return m.copy(), u, p, p.__getstate__()

    both(run)


# ---------------------------------------------------------------------------------------------
# bulk / generator
# ---------------------------------------------------------------------------------------------


def _bulk(lib: Any, docs: Any, **kwargs: Any) -> Any:
    ms = lib.MinHash.bulk(docs, **kwargs)
    return (
        ms,
        identity_pattern([m.permutations for m in ms]),
        identity_pattern([m.hashvalues for m in ms]),
        identity_pattern([m.hashfunc for m in ms]),
    )


@given(docs=DOCS, scheme=SCHEME, num_perm=NUM_PERM, seed=SEED)
def test_bulk_matches(docs: list[list[bytes]], scheme: Scheme, num_perm: int, seed: int) -> None:
    kwargs = {"num_perm": num_perm, "seed": seed, "scheme": scheme}
    got, want = both(lambda lib: _bulk(lib, docs, **kwargs))
    ms, ds_ms = got[1][0], want[1][0]
    assert all(m.permutations is ms[0].permutations for m in ms)
    for threads in (1, 2, None):
        assert_same(MinHash.bulk(docs, threads=threads, **kwargs), ds_ms)
        assert_same(list(MinHash.generator(iter(docs), threads=threads, **kwargs)), ds_ms)


@given(
    docs=st.lists(
        st.one_of(
            TOKENS,
            TOKENS.map(tuple),
            st.lists(st.one_of(BYTES, OTHER_TOKENS), max_size=6),
            TOKENS.map(lambda t: np.array(t, dtype=object)),
            st.lists(st.binary(min_size=1, max_size=5), max_size=5).map(np.array),
            TOKENS.map(lambda t: (x for x in t)),
            TOKENS.map(lambda t: dict.fromkeys(t)),
        ),
        max_size=10,
    ),
    scheme=SCHEME,
)
def test_bulk_document_kinds_match(docs: list[Any], scheme: Scheme) -> None:
    # Generators can be consumed once: each library gets its own copy of the documents.
    frozen = [list(d) if not isinstance(d, (list, tuple, np.ndarray, dict)) else d for d in docs]

    def fresh() -> list[Any]:
        return [
            iter(d) if not isinstance(d, (list, tuple, np.ndarray, dict)) else d for d in frozen
        ]

    both(lambda lib: _bulk(lib, fresh(), num_perm=8, scheme=scheme))


def _bad_docs(n: int, bad_at: int, bad: Any) -> list[Any]:
    docs: list[Any] = [[b"d%d" % i, b"x"] for i in range(n)]
    docs[bad_at] = [b"ok", bad, b"ok"]
    return docs


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize(
    ("n", "bad_at"), [(5, 0), (5, 2), (5, 4), (2100, 2049), (2100, 30), (4500, 4100)]
)
@pytest.mark.parametrize("bad", ["str", 7, None], ids=repr)
def test_generator_error_position_matches(n: int, bad_at: int, bad: Any, scheme: Scheme) -> None:
    docs = _bad_docs(n, bad_at, bad)

    def run(lib: Any) -> Any:
        ms, err = drain(lib.MinHash.generator(docs, num_perm=4, scheme=scheme))
        return len(ms), err, ms[-3:]

    got, _ = both(run)
    assert got[1][0] == bad_at


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize(
    ("n", "fail_at"), [(0, 0), (2, 2), (5, 0), (5, 3), (3000, 2500), (3000, 2048)]
)
def test_generator_source_error_position_matches(n: int, fail_at: int, scheme: Scheme) -> None:
    # The documents come from a generator that raises after `fail_at` documents: the
    # MinHash objects before it come out first, then the error (datasketch's order).
    def source() -> Iterator[list[bytes]]:
        for i in range(n):
            if i == fail_at:
                raise LookupError(f"source failed at {i}")
            yield [b"d%d" % i]

    def run(lib: Any) -> Any:
        ms, err = drain(lib.MinHash.generator(source(), num_perm=4, scheme=scheme))
        return len(ms), err

    both(run)


class _OtherList(list):  # type: ignore[type-arg]
    def __iter__(self) -> Iterator[Any]:
        return iter([b"from __iter__"])


class _OtherTuple(tuple):  # type: ignore[type-arg]
    def __iter__(self) -> Iterator[Any]:
        return iter([b"from __iter__"])


@pytest.mark.parametrize("scheme", SCHEMES)
def test_bulk_documents_are_iterated(scheme: Scheme) -> None:
    # datasketch hashes `for _b in doc`: a list or tuple subclass's own __iter__ counts.
    docs = [_OtherList([b"a", b"b"]), _OtherTuple((b"c",)), [b"d"], _OtherList()]
    both(lambda lib: _bulk(lib, docs, num_perm=8, scheme=scheme))
    both(lambda lib: drain(lib.MinHash.generator(iter(docs), num_perm=8, scheme=scheme)))


def test_bulk_bad_arguments_match() -> None:
    cases: list[dict[str, Any]] = [
        {"num_perm": 0},
        {"scheme": "nope"},
        {"hashvalues": [1, 2]},
        {"hashfunc": 3},
        {"unknown": 1},
    ]
    for kwargs in cases:
        both(lambda lib, kw: lib.MinHash.bulk([[b"a"]], **kw), kwargs)
        both(lambda lib, kw: drain(lib.MinHash.generator([[b"a"]], **kw)), kwargs)
    both(lambda lib: lib.MinHash.bulk(5))
    both(lambda lib: lib.MinHash.bulk([5]))
    both(lambda lib: lib.MinHash.bulk([]))
    both(lambda lib: lib.LeanMinHash.bulk([[b"a"]]))


@pytest.mark.parametrize("threads", [0, -1, "2", 1.5])
def test_bulk_bad_threads(threads: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        MinHash.bulk([[b"a"]], threads=threads)


def test_bulk_of_a_subclass_matches() -> None:
    class PSub(MinHash):
        pass

    class DSub(ds.MinHash):  # type: ignore[misc]
        pass

    docs = [[b"a", b"b"], [b"c"]]
    p, d = PSub.bulk(docs, num_perm=8), DSub.bulk(docs, num_perm=8)
    assert [type(m).__name__ for m in p] == [type(m).__name__ for m in d] == ["MinHash"] * 2
    assert_same(p, d)


@pytest.mark.timeout(60)
def test_bulk_10k_documents() -> None:
    # Any: the stub rejects list[list[bytes]] (list is invariant), see minhash.pyi.
    docs: Any = [[b"w%d" % (i * 7 + j) for j in range(i % 23)] for i in range(10_000)]
    for scheme in SCHEMES:
        ms = MinHash.bulk(docs, num_perm=64, scheme=scheme)
        assert len(ms) == len(docs)
        rows = minhash_batch(docs, num_perm=64, scheme=scheme)
        assert np.array_equal(np.stack([m.hashvalues for m in ms]), rows)
        sample = docs[::97]
        assert_same(ms[::97], ds.MinHash.bulk(sample, num_perm=64, scheme=scheme))
        one = MinHash.bulk(docs, num_perm=64, scheme=scheme, threads=1)
        assert np.array_equal(np.stack([m.hashvalues for m in one]), rows)


# ---------------------------------------------------------------------------------------------
# jaccard, count, merge, union, copy, digest, is_empty, clear, __eq__
# ---------------------------------------------------------------------------------------------


@given(a=TOKENS, b=TOKENS, scheme=SCHEME, num_perm=NUM_PERM)
def test_estimators_and_set_operations_match(
    a: list[bytes], b: list[bytes], scheme: Scheme, num_perm: int
) -> None:
    def run(lib: Any) -> Any:
        m1, m2 = lib.MinHash(num_perm, scheme=scheme), lib.MinHash(num_perm, scheme=scheme)
        m1.update_batch(a)
        m2.update_batch(b)
        out: list[Any] = [
            m1.jaccard(m2),
            m2.jaccard(m1),
            m1.count(),
            m2.count(),
            m1.is_empty(),
            m2.is_empty(),
            m1 == m2,
            m1 != m2,
            m1 == m1.copy(),
            len(m1),
        ]
        dig = m1.digest()
        out += [dig, dig is m1.hashvalues, m1.copy(), m1.copy().permutations is m1.permutations]
        u = lib.MinHash.union(m1, m2, m1)
        out += [u, u.permutations is m1.permutations, u.hashfunc is m1.hashfunc]
        m1.merge(m2)
        out += [m1, m1 == u, m1.jaccard(u)]
        m2.clear()
        out += [m2, m2.is_empty(), m2.count()]
        lean = lib.LeanMinHash(m1)
        out += [lean.jaccard(m1), m1.jaccard(lean), lean.count(), lean.is_empty()]
        return out

    got, _ = both(run)
    assert type(got[1][2]) is np.float64


@pytest.mark.parametrize("scheme", SCHEMES)
def test_comparison_errors_match(scheme: Scheme) -> None:
    other: Scheme = "legacy" if scheme != "legacy" else "affine32"

    def run(lib: Any) -> Any:
        m = lib.MinHash(8, 1, scheme=scheme)
        rivals = [
            lib.MinHash(8, 2, scheme=scheme),
            lib.MinHash(9, 1, scheme=scheme),
            lib.MinHash(8, 1, scheme=other),
            lib.LeanMinHash(seed=1, hashvalues=m.hashvalues, scheme=scheme),
            object(),
            5,
        ]
        out = []
        for r in rivals:
            out += [outcome(m.jaccard, r), outcome(m.merge, r), outcome(lib.MinHash.union, m, r)]
            out += [m == r, outcome(lambda r=r: r == m)]
        out += [outcome(lib.MinHash.union), outcome(lib.MinHash.union, m)]
        lean = lib.LeanMinHash(m)
        out += [outcome(lib.MinHash.union, lean, lean)]
        return out

    both(run)


# ---------------------------------------------------------------------------------------------
# Pickles
# ---------------------------------------------------------------------------------------------

# pickle.dumps of MinHash(num_perm=8, seed=42) updated with LEGACY_TOKENS under datasketch 1.10.0
# (from datasketch 2.0.0's test/test_minhash_schemes.py, MIT): no "scheme" in the state.
LEGACY_TOKENS = [b"datasketch", b"minhash", b"affine", b"legacy", b"fixture"]
LEGACY_MINHASH_PICKLE = base64.b64decode(
    "gASVHwIAAAAAAACMEmRhdGFza2V0Y2gubWluaGFzaJSMB01pbkhhc2iUk5QpgZR9lCiMBHNlZWSUSyqMCG51bV9wZXJtlEsIjAho"
    "YXNoZnVuY5SME2RhdGFza2V0Y2guaGFzaGZ1bmOUjAtzaGExX2hhc2gzMpSTlIwKaGFzaHZhbHVlc5SMFm51bXB5Ll9jb3JlLm11"
    "bHRpYXJyYXmUjAxfcmVjb25zdHJ1Y3SUk5SMBW51bXB5lIwHbmRhcnJheZSTlEsAhZRDAWKUh5RSlChLAUsIhZRoD4wFZHR5cGWU"
    "k5SMAnU4lImIh5RSlChLA4wBPJROTk5K/////0r/////SwB0lGKJQ0CTz7QkAAAAAEQcExgAAAAA2fUPgAAAAAAw7fUDAAAAAHvg"
    "Gm4AAAAALD0iQwAAAAAuEvcSAAAAACRoyTAAAAAAlHSUYowMcGVybXV0YXRpb25zlGgOaBFLAIWUaBOHlFKUKEsBSwJLCIaUaBuI"
    "Q4C0PerLZtzhHw6V9S5cA2ITSNSZx2r0YxsULMuYvK5BGXoYInJm1vAH1vSXGdJB7wfLVZF1SpHeDnTtbVVXuL0daOuSJGOs4hmC"
    "oKCmlz5EFTX9cA6VBkUFV+nUuAEDTBie/UTw6/UaFYEJMwAl21sWvJMB/r8UjA6gRBOeFJnzDpR0lGKMCV9ncHVfbW9kZZSMB2Rp"
    "c2FibGWUjAZfYV9ncHWUTowGX2JfZ3B1lE51Yi4="
)
# LeanMinHash(that MinHash).serialize(buf, "<") under 1.10.0.
LEGACY_LEAN_SERIALIZED_LE = bytes.fromhex(
    "2a000000000000000800000093cfb424441c1318d9f50f8030edf5037be01a6e2c3d22432e12f7122468c930"
)


@given(tokens=TOKENS, scheme=SCHEME, num_perm=NUM_PERM, seed=ANY_SEED)
def test_pickle_state_matches(
    tokens: list[bytes], scheme: Scheme, num_perm: int, seed: Any
) -> None:
    def run(lib: Any) -> Any:
        m = lib.MinHash(num_perm, seed, scheme=scheme)
        m.update_batch(tokens)
        state = m.__getstate__()
        rounds = [pickle.loads(pickle.dumps(m, protocol=proto)) for proto in (0, 2, 4, 5)]
        return m, state, rounds

    both(run)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_pickles_load_in_the_other_library(scheme: Scheme) -> None:
    p, d = MinHash(16, 7, scheme=scheme), ds.MinHash(16, 7, scheme=scheme)
    p.update_batch([b"a", b"b"])
    d.update_batch([b"a", b"b"])
    pl, dl = LeanMinHash(p), ds.LeanMinHash(d)
    for p_obj, d_obj in ((p, d), (pl, dl)):
        # A datasketch pickle read by pyroid, and a pyroid pickle read by datasketch.
        assert_same(to_pyroid(pickle.dumps(d_obj)), pickle.loads(pickle.dumps(d_obj)))
        assert_same(pickle.loads(pickle.dumps(p_obj)), to_datasketch(pickle.dumps(p_obj)))
        assert_same(to_pyroid(pickle.dumps(d_obj)), to_datasketch(pickle.dumps(p_obj)))


def test_pre_2_0_0_pickle_restores_as_legacy() -> None:
    p, d = to_pyroid(LEGACY_MINHASH_PICKLE), pickle.loads(LEGACY_MINHASH_PICKLE)
    assert type(p) is MinHash
    assert p.scheme == "legacy"
    assert_same(p, d)
    for m in (p, d):
        m.update(b"more")
        m.update_batch([b"and", b"more"])
    assert_same(p, d)
    fresh_p, fresh_d = MinHash(8, 42, scheme="legacy"), ds.MinHash(8, 42, scheme="legacy")
    fresh_p.update_batch([*LEGACY_TOKENS, b"more", b"and"])
    fresh_d.update_batch([*LEGACY_TOKENS, b"more", b"and"])
    assert p.jaccard(fresh_p) == d.jaccard(fresh_d) == 1.0

    # A state without "scheme" (any MinHash pickled before 2.0.0).
    def run(lib: Any) -> Any:
        m = lib.MinHash(8, 3, scheme="legacy")
        state = m.__getstate__()
        del state["scheme"]
        restored = lib.MinHash.__new__(lib.MinHash)
        restored.__setstate__(state)
        restored.update(b"x")
        return restored

    both(run)


# ---------------------------------------------------------------------------------------------
# LeanMinHash
# ---------------------------------------------------------------------------------------------


class _ByteArray(bytearray):
    pass


class _Bytes(bytes):
    pass


#: Buffers that serialize() writes into: pyroid has a native path for bytearray only.
WRITE_BUFFERS: dict[str, Callable[[int], Any]] = {
    "bytearray": bytearray,
    # struct writes alignment padding as zeros: so must the native path.
    "bytearray of 0xff": lambda n: bytearray(b"\xff" * n),
    "memoryview": lambda n: memoryview(bytearray(n)),
    "numpy": lambda n: np.zeros(n, dtype=np.uint8),
    "array": lambda n: array.array("B", bytes(n)),
    "bytearray subclass": _ByteArray,
    "bytes": bytes,
}
#: Buffers that deserialize() and __setstate__ read: native for bytes and bytearray.
READ_BUFFERS: dict[str, Callable[[bytes], Any]] = {
    "bytes": bytes,
    "bytearray": bytearray,
    "memoryview": memoryview,
    "memoryview at an offset": lambda raw: memoryview(b"pad" + raw)[3:],
    "numpy": lambda raw: np.frombuffer(raw, dtype=np.uint8),
    "array": lambda raw: array.array("B", raw),
    "bytes subclass": _Bytes,
    "str": lambda raw: raw.decode("latin-1"),
}


def _lean_bytes(lean: Any, byteorder: str, extra: int = 0) -> bytes:
    buf = bytearray(lean.bytesize(byteorder) + extra)
    lean.serialize(buf, byteorder)
    return bytes(buf)


def _serialize_everywhere(lean: Any, byteorder: Any, extra: int = 0) -> dict[str, Any]:
    """serialize() into every kind of buffer: (outcome, buffer bytes) per kind."""
    size = outcome(lean.bytesize, byteorder)
    out: dict[str, Any] = {"bytesize": size}
    if size[0] == "ok":
        for name, make in WRITE_BUFFERS.items():
            buf = make(size[1] + extra)
            out[name] = (outcome(lean.serialize, buf, byteorder), bytes(buf))
        out["short"] = outcome(lean.serialize, bytearray(size[1] - 1), byteorder)
    return out


def _read_everywhere(lib: Any, raw: bytes, byteorder: Any) -> dict[str, Any]:
    """deserialize() and __setstate__ from every kind of buffer."""
    out: dict[str, Any] = {}
    for name, make in READ_BUFFERS.items():
        out[name] = outcome(lib.LeanMinHash.deserialize, make(raw), byteorder)
        if byteorder == "@":
            empty = lib.LeanMinHash.__new__(lib.LeanMinHash)
            out[name + " setstate"] = (outcome(empty.__setstate__, make(raw)), vars(empty))
            if out[name + " setstate"][0][0] == "ok":
                out[name + " restored"] = empty
    return out


def _state(lean: Any) -> tuple[Any, ...]:
    """__getstate__ and a pickle round trip."""
    state = outcome(lean.__getstate__)
    if state[0] == "raise":
        return state, outcome(pickle.dumps, lean)
    return state, type(state[1]), pickle.loads(pickle.dumps(lean))


@given(
    tokens=TOKENS,
    scheme=SCHEME,
    num_perm=NUM_PERM,
    seed=st.one_of(SEED, st.integers(-(2**63), 2**63 - 1)),
    byteorder=st.sampled_from(BYTEORDERS),
    extra=st.integers(0, 9),
)
def test_lean_serialize_matches(
    tokens: list[bytes], scheme: Scheme, num_perm: int, seed: int, byteorder: str, extra: int
) -> None:
    def run(lib: Any) -> Any:
        m = lib.MinHash(num_perm, scheme=scheme)
        m.update_batch(tokens)
        lean = lib.LeanMinHash(seed=seed, hashvalues=m.hashvalues, scheme=scheme)
        written = _serialize_everywhere(lean, byteorder, extra)
        raw = written["bytearray"][1]
        return (
            lean,
            written,
            _read_everywhere(lib, raw, byteorder),
            _read_everywhere(lib, raw, "@"),
            _state(lean),
            hash(lean),
            lean == lib.LeanMinHash.deserialize(raw, byteorder),
            list(vars(lean)),
        )

    got, want = both(run)
    # Each library reads the bytes the other wrote (the same bytes, as checked above).
    raw_p, raw_d = got[1][1]["bytearray"][1], want[1][1]["bytearray"][1]
    assert_same(
        LeanMinHash.deserialize(raw_d, byteorder), ds.LeanMinHash.deserialize(raw_p, byteorder)
    )


@pytest.mark.parametrize("scheme", SCHEMES)
def test_lean_operations_match(scheme: Scheme) -> None:
    def run(lib: Any) -> Any:
        m1, m2 = lib.MinHash(16, 4, scheme=scheme), lib.MinHash(16, 4, scheme=scheme)
        m1.update_batch([b"a", b"b"])
        m2.update_batch([b"b", b"c"])
        l1, l2 = lib.LeanMinHash(m1), lib.LeanMinHash(m2)
        u = lib.LeanMinHash.union(l1, l2)
        c = l1.copy()
        l1.merge(l2)
        out: list[Any] = [u, c, l1, l1 == u, l1.digest(), len(l1), l1.count()]
        l2.clear()
        out += [l2, l2.is_empty(), outcome(l2.update, b"x"), outcome(l2.update_batch, [b"x"])]
        out += [_state(u), hash(u), hash(l1) == hash(u)]
        out += [
            lib.LeanMinHash(m1, scheme=scheme),
            lib.LeanMinHash(seed=4, hashvalues=m1.hashvalues.tolist(), scheme=scheme),
            lib.LeanMinHash(seed=4, hashvalues=tuple(m1.hashvalues.tolist()), scheme=scheme),
        ]
        return out

    both(run)


def test_lean_from_a_minhash_of_the_other_library() -> None:
    for scheme in SCHEMES:
        p, d = MinHash(8, scheme=scheme), ds.MinHash(8, scheme=scheme)
        p.update(b"x")
        d.update(b"x")
        assert_same(LeanMinHash(d), ds.LeanMinHash(d))
        assert_same(LeanMinHash(p), ds.LeanMinHash(p))


@pytest.mark.parametrize("byteorder", BYTEORDERS)
def test_pre_2_0_0_lean_buffers(byteorder: str) -> None:
    raws = [LEGACY_LEAN_SERIALIZED_LE] if byteorder == "<" else []
    values = [615829395, 403905604, 0, 2**32 - 1]
    raws.append(struct.pack(f"{byteorder}qi{len(values)}I", 42, len(values), *values))
    raws.append(struct.pack(f"{byteorder}qi", -7, 0))  # an empty legacy sketch (datasketch 1.x)

    def run(lib: Any, raw: bytes) -> Any:
        lean = lib.LeanMinHash.deserialize(raw, byteorder)
        return lean, _serialize_everywhere(lean, byteorder), _read_everywhere(lib, raw, byteorder)

    for raw in raws:
        got, _ = both(run, raw)
        assert got[1][0].scheme == "legacy"
        assert got[1][1]["bytearray"][1] == raw


#: Seeds struct cannot pack as "q" (pyroid's native path must leave them to struct), and
#: seeds it packs through __index__.
ODD_SEEDS = [
    2**63,
    -(2**63) - 1,
    2**64,
    1.5,
    "1",
    b"1",
    np.float64(1.0),
    np.uint64(2**63),
    np.uint64(5),
    np.int64(-5),
    np.int32(7),
    True,
    -(2**63),
    2**63 - 1,
]


@pytest.mark.parametrize("byteorder", [*BYTEORDERS, "", "x", "<>", None])
@pytest.mark.parametrize("scheme", SCHEMES)
def test_lean_seeds_struct_cannot_pack(scheme: Scheme, byteorder: Any) -> None:
    def run(lib: Any) -> Any:
        out = []
        for seed in ODD_SEEDS:
            lean = lib.LeanMinHash(seed=seed, hashvalues=[1, 2, 3], scheme=scheme)
            out += [_serialize_everywhere(lean, byteorder), _state(lean), outcome(hash, lean)]
        return out

    got, _ = both(run)
    if byteorder in tuple(BYTEORDERS):
        # ODD_SEEDS[0] is 2**63: struct.error for every buffer, and for pickling.
        written, state = got[1][0], got[1][1]
        assert written["bytearray"][0][:2] == ("raise", struct.error)
        assert written["memoryview"][0][:2] == ("raise", struct.error)
        assert state[0][:2] == ("raise", struct.error)


@pytest.mark.parametrize("byteorder", BYTEORDERS)
@pytest.mark.parametrize("scheme", SCHEMES)
@numpy_casts
def test_lean_values_struct_cannot_pack(scheme: Scheme, byteorder: str) -> None:
    # Legacy values are uint64 but serialized as "I": values >= 2**32 raise struct.error.
    # Hash values reassigned after __init__ (it is a slot) in other forms run struct too.
    width = 2**32 if scheme == "affine32" else 2**64
    forms: list[Callable[[Any], Any]] = [
        lambda hv: hv,
        lambda hv: hv.astype(np.int64),
        lambda hv: hv.astype(np.float64),
        lambda hv: hv.astype(">u8"),
        lambda hv: np.repeat(hv, 2)[::2],
        lambda hv: hv.tolist(),
        lambda hv: [-1, *hv.tolist()],
        lambda hv: [*hv.tolist(), 2**32],
        lambda hv: np.array([*hv.tolist(), 2**33], dtype=np.uint64),
        lambda hv: [*hv.tolist(), 2**64],
        lambda hv: hv[:0],
    ]

    def run(lib: Any) -> Any:
        out = []
        values = [1, 2**31, width - 1]
        if scheme == "legacy":
            values += [2**32, 2**40, 2**64 - 1]
        lean = lib.LeanMinHash(seed=3, hashvalues=values, scheme=scheme)
        out += [lean, _serialize_everywhere(lean, byteorder), _state(lean)]
        for form in forms:
            lean = lib.LeanMinHash(seed=3, hashvalues=[1, 2, width - 1], scheme=scheme)
            lean.hashvalues = form(lean.hashvalues)
            out += [_serialize_everywhere(lean, byteorder), _state(lean), outcome(hash, lean)]
        return out

    got, _ = both(run)
    written, state = got[1][1], got[1][2]
    if scheme == "legacy":  # values >= 2**32 in the uint64 array
        for kind in ("bytearray", "bytearray of 0xff", "memoryview", "numpy"):
            assert written[kind][0][:2] == ("raise", struct.error), kind
        assert state[0][:2] == ("raise", struct.error)
    else:
        assert written["bytearray"][0] == ("ok", None)


@pytest.mark.parametrize("byteorder", BYTEORDERS)
@pytest.mark.parametrize("scheme", SCHEMES)
def test_lean_truncated_buffers(scheme: Scheme, byteorder: str) -> None:
    m = ds.MinHash(3, 1, scheme=scheme)
    m.update(b"x")
    raw = _lean_bytes(ds.LeanMinHash(m), byteorder)
    for cut in range(len(raw)):
        both(lambda lib, cut: _read_everywhere(lib, raw[:cut], byteorder), cut)
    both(lambda lib: _read_everywhere(lib, raw[:-1], "@"))


@pytest.mark.parametrize("byteorder", BYTEORDERS)
def test_lean_unknown_scheme_codes_and_counts(byteorder: str) -> None:
    raws = []
    for code in (0, 3, 99, 255, 1, 2):
        for count in (-1, -2, -(2**31)):
            raw = bytearray(struct.pack(f"{byteorder}qi", 1, count) + bytes(40))
            raw[struct.calcsize("@qi")] = code
            raws.append(bytes(raw))
    raws.append(struct.pack(f"{byteorder}qi", 1, 2**31 - 1) + bytes(40))
    for buf in raws:
        both(lambda lib, buf: _read_everywhere(lib, buf, byteorder), buf)


def test_lean_errors_match() -> None:
    cases: list[Callable[[Any], Any]] = [
        lambda lib: lib.LeanMinHash(),
        lambda lib: lib.LeanMinHash(seed=1),
        lambda lib: lib.LeanMinHash(hashvalues=[1, 2]),
        lambda lib: lib.LeanMinHash(seed=1, hashvalues=[1, 2]),
        lambda lib: lib.LeanMinHash(seed=None, hashvalues=[1, 2], scheme="affine32"),
        lambda lib: lib.LeanMinHash(seed=1, hashvalues=[1, 2], scheme="nope"),
        lambda lib: lib.LeanMinHash(seed=1, hashvalues=[], scheme="affine32"),
        lambda lib: lib.LeanMinHash(seed=1, hashvalues=[], scheme="affine64"),
        lambda lib: lib.LeanMinHash(seed=1, hashvalues=np.array([], np.uint32), scheme="affine32"),
        lambda lib: lib.LeanMinHash(seed=1, hashvalues=[2**40], scheme="affine32"),
        lambda lib: lib.LeanMinHash(seed=1, hashvalues=[-1], scheme="legacy"),
        lambda lib: lib.LeanMinHash(lib.MinHash(4), scheme="legacy"),
        lambda lib: lib.LeanMinHash(lib.MinHash(4)).update(b"x"),
        lambda lib: lib.LeanMinHash(lib.MinHash(4)).serialize("text"),
        lambda lib: lib.LeanMinHash(lib.MinHash(4)).serialize(None),
        lambda lib: lib.LeanMinHash.deserialize(None),
        lambda lib: lib.LeanMinHash.deserialize(12),
        lambda lib: lib.LeanMinHash.__new__(lib.LeanMinHash).__setstate__(None),
        lambda lib: lib.LeanMinHash.union(lib.LeanMinHash(lib.MinHash(4))),
        lambda lib: lib.LeanMinHash.union(
            lib.LeanMinHash(lib.MinHash(4)), lib.LeanMinHash(lib.MinHash(4, 2))
        ),
        lambda lib: lib.LeanMinHash.union(
            lib.LeanMinHash(lib.MinHash(4)), lib.LeanMinHash(lib.MinHash(5))
        ),
    ]
    for case in cases:
        got, _ = both(case)
        assert got[0] == "raise", case


# ---------------------------------------------------------------------------------------------
# minhash_batch
# ---------------------------------------------------------------------------------------------

#: Every character str.split() splits on, and some that look like whitespace but are not.
WHITESPACE = [c for c in map(chr, range(sys.maxunicode + 1)) if c.isspace()]
NOT_WHITESPACE = [chr(c) for c in (0x200B, 0x180E, 0xFEFF, 0x2060, 0x00, 0x1B, 0x7F, 0x5F)]
assert {chr(c) for c in (0x1C, 0x1D, 0x1E, 0x1F, 0xA0, 0x85, 0x1680, 0x2028, 0x3000)} <= set(
    WHITESPACE
)
WORDS = [
    "a",
    "b",
    "ab",
    chr(0xE9),
    "e" + chr(0x301),
    chr(0x65E5) + chr(0x672C),
    chr(0x1F642),
    "x" * 70,
]
TEXT = st.lists(
    st.one_of(
        st.sampled_from(WHITESPACE),
        st.sampled_from(NOT_WHITESPACE),
        st.sampled_from(WORDS),
        st.characters(exclude_categories=["Cs"]),
    ),
    max_size=40,
).map("".join)
TEXTS = st.lists(st.one_of(TEXT, st.sampled_from(["", " ", chr(0x3000) + "\n\t", "a"])), max_size=8)
NGRAM = st.integers(1, 6)


def shingles(text: str, n: int) -> list[bytes]:
    words = text.split()
    return [" ".join(words[i : i + n]).encode("utf-8") for i in range(len(words) - n + 1)]


def reference(
    docs: list[list[bytes]], num_perm: int = 128, seed: int = 1, scheme: Scheme | None = None
) -> NDArray[Any]:
    """datasketch.MinHash.bulk(docs) as a (len(docs), num_perm) matrix."""
    ms = ds.MinHash.bulk(docs, num_perm=num_perm, seed=seed, scheme=scheme)
    dtype = DTYPES[scheme or "affine32"]
    out = np.empty((len(docs), num_perm), dtype=dtype)
    for i, m in enumerate(ms):
        out[i] = m.hashvalues
    return out


def assert_matrix(got: Any, want: NDArray[Any]) -> None:
    assert type(got) is np.ndarray
    assert got.dtype == want.dtype, (got.dtype, want.dtype)
    assert got.shape == want.shape, (got.shape, want.shape)
    assert np.array_equal(got, want)


@given(texts=TEXTS, n=NGRAM, scheme=st.one_of(st.none(), SCHEME), num_perm=NUM_PERM, seed=SEED)
def test_minhash_batch_texts_match_bulk(
    texts: list[str], n: int, scheme: Scheme | None, num_perm: int, seed: int
) -> None:
    want = reference([shingles(t, n) for t in texts], num_perm, seed, scheme)
    kw: dict[str, Any] = {"num_perm": num_perm, "seed": seed, "scheme": scheme, "ngram": n}
    if texts:  # an empty list is token lists or texts alike; see test_minhash_batch_empty
        assert_matrix(minhash_batch(texts, **kw), want)
        assert_matrix(minhash_batch(tuple(texts), **kw), want)
    for i, t in enumerate(texts):
        assert_matrix(minhash_batch(t, **kw), want[i])
    assert_matrix(minhash_batch(_testing.column(texts), **kw), want)
    assert_matrix(minhash_batch(_testing.column(texts, large=True, chunks=3), **kw), want)


@given(texts=st.lists(st.one_of(st.none(), TEXT), max_size=10), n=NGRAM, scheme=SCHEME)
def test_minhash_batch_arrow_nulls_are_empty_signatures(
    texts: list[str | None], n: int, scheme: Scheme
) -> None:
    want = reference([shingles(t or "", n) for t in texts], 32, 1, scheme)
    for chunks in (1, 2, 4):
        col = _testing.column(texts, large=chunks == 2, chunks=chunks)
        assert_matrix(minhash_batch(col, num_perm=32, scheme=scheme, ngram=n), want)


@pytest.mark.parametrize("arrow_type", ["string", "large_string", "string_view"])
@given(texts=st.lists(st.one_of(st.none(), TEXT), max_size=12), n=NGRAM, data=st.data())
def test_minhash_batch_pyarrow(
    pa: ModuleType, arrow_type: str, texts: list[str | None], n: int, data: st.DataObject
) -> None:
    ty = getattr(pa, arrow_type)()
    start = data.draw(st.integers(0, len(texts)))
    stop = data.draw(st.integers(start, len(texts)))
    want = reference([shingles(t or "", n) for t in texts[start:stop]], 16)
    arr = pa.array(texts, type=ty).slice(start, stop - start)
    assert_matrix(minhash_batch(arr, num_perm=16, ngram=n), want)
    chunked = pa.chunked_array([texts, texts], type=ty).slice(len(texts) + start, stop - start)
    assert_matrix(minhash_batch(chunked, num_perm=16, ngram=n), want)


TOKEN = st.one_of(BYTES, st.text(alphabet=st.characters(exclude_categories=["Cs"]), max_size=12))


@given(
    docs=st.lists(st.lists(TOKEN, max_size=20), min_size=1, max_size=10),
    scheme=st.one_of(st.none(), SCHEME),
    num_perm=NUM_PERM,
    seed=SEED,
)
def test_minhash_batch_token_lists_match_bulk(
    docs: list[list[bytes | str]], scheme: Scheme | None, num_perm: int, seed: int
) -> None:
    encoded = [[t.encode("utf-8") if isinstance(t, str) else t for t in d] for d in docs]
    want = reference(encoded, num_perm, seed, scheme)
    kw: dict[str, Any] = {"num_perm": num_perm, "seed": seed, "scheme": scheme}
    assert_matrix(minhash_batch(docs, **kw), want)
    assert_matrix(minhash_batch(tuple(docs), **kw), want)
    as_tuples: Any = [tuple(d) for d in docs]
    assert_matrix(minhash_batch(as_tuples, **kw), want)
    if all(isinstance(t, bytes) for d in docs for t in d):
        assert_matrix(np.stack([m.hashvalues for m in MinHash.bulk(docs, **kw)]), want)


@pytest.mark.parametrize("scheme", [None, *SCHEMES])
@pytest.mark.parametrize("ngram", [1, 5])
def test_minhash_batch_empty(ngram: int, scheme: Scheme | None) -> None:
    want = reference([], 12, 1, scheme)
    empties: list[Any] = [[], ()]
    for empty in empties:
        assert_matrix(minhash_batch(empty, num_perm=12, scheme=scheme, ngram=ngram), want)
    assert_matrix(minhash_batch(_testing.column([]), num_perm=12, scheme=scheme, ngram=ngram), want)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_minhash_batch_dtype_per_scheme(scheme: Scheme) -> None:
    cases: list[Any] = ["a b", ["a b"], [[b"a"]], [["a"]], _testing.column(["a b", None])]
    for docs in cases:
        got = minhash_batch(docs, num_perm=4, scheme=scheme)
        assert got.dtype == DTYPES[scheme]
    assert minhash_batch(["a"], num_perm=4).dtype == np.uint32


@given(
    text=TEXT,
    surrogate=st.characters(categories=["Cs"]),
    data=st.data(),
)
def test_minhash_batch_lone_surrogates_raise(
    text: str, surrogate: str, data: st.DataObject
) -> None:
    i = data.draw(st.integers(0, len(text)))
    bad = text[:i] + surrogate + text[i:]
    cases: list[Any] = [bad, [bad], ("ok", bad), [["ok"], ["fine", bad]], ([bad],)]
    for docs in cases:
        with pytest.raises(UnicodeEncodeError):
            minhash_batch(docs, num_perm=4)
        with pytest.raises(UnicodeEncodeError):
            minhash_batch(docs, num_perm=4, threads=1)


@pytest.mark.parametrize(
    "docs",
    [
        5,
        None,
        b"bytes",
        bytearray(b"x"),
        iter(["a"]),
        (t for t in ["a"]),
        {"a"},
        [1, 2],
        [b"a b"],
        ["a", 1],
        ["a", None],
        ["a", b"b"],
        [["a"], "b"],
        [[1]],
        [[b"a", None]],
        [[bytearray(b"a")]],
        [iter([b"a"])],
        [np.array([b"a"])],
        np.array(["a b"]),
    ],
    ids=repr,
)
def test_minhash_batch_wrong_types_raise_input_error(docs: Any) -> None:
    with pytest.raises(pyroid.InputError):
        minhash_batch(docs, num_perm=4)


def test_minhash_batch_parameter_errors_match_minhash() -> None:
    for kw in ({"num_perm": 0}, {"seed": -1}, {"seed": 1.5}, {"scheme": "nope"}, {"seed": 2**32}):
        want = outcome(ds.MinHash, **kw)
        assert want[0] == "raise"
        cases: list[Any] = [["a b"], [[b"a"]], "a b"]
        for docs in cases:
            assert outcome(minhash_batch, docs, **kw) == want
    for ngram in (0, -1):
        with pytest.raises(ValueError, match="ngram"):
            minhash_batch(["a b"], ngram=ngram)
    with pytest.raises(TypeError):
        minhash_batch(["a b"], ngram=1.5)  # type: ignore[arg-type]
    one_doc: Any = [[b"a"]]  # the stub rejects even this literal (list is invariant)
    with pytest.raises(ValueError, match="ngram applies to texts"):
        minhash_batch(one_doc, ngram=2)
    for threads in (0, -1):
        with pytest.raises(ValueError, match="threads"):
            minhash_batch(["a"], threads=threads)
        with pytest.raises(ValueError, match="threads"):
            minhash_batch(one_doc, threads=threads)


@pytest.mark.parametrize("seed", [np.int64(3), np.uint32(3), True, [1, 2]], ids=repr)
def test_minhash_batch_numpy_and_sequence_seeds(seed: Any) -> None:
    want = reference([shingles("a b c", 1)], 8, seed)
    assert_matrix(minhash_batch(["a b c"], num_perm=8, seed=seed), want)


@pytest.mark.timeout(60)
def test_minhash_batch_threads_identical() -> None:
    texts = [" ".join(f"w{(i * 31 + j) % 997}" for j in range(i % 40)) for i in range(3000)]
    tokens: Any = [[w.encode() for w in t.split()] for t in texts]
    for scheme in SCHEMES:
        for docs, n in ((texts, 3), (tokens, 1), (_testing.column(texts, chunks=5), 2)):
            base = minhash_batch(docs, num_perm=32, scheme=scheme, ngram=n, threads=1)
            for threads in (2, 3, None):
                got = minhash_batch(docs, num_perm=32, scheme=scheme, ngram=n, threads=threads)
                assert_matrix(got, base)
        want = reference([shingles(t, 3) for t in texts[::150]], 32, 1, scheme)
        assert_matrix(minhash_batch(texts, num_perm=32, scheme=scheme, ngram=3)[::150], want)


def test_minhash_batch_rows_are_the_minhash_hashvalues() -> None:
    # The documented equivalence, on the default arguments.
    texts = ["minhash is a probabilistic data structure", "", "minhash is a probability data"]
    rows = minhash_batch(texts, ngram=2)
    for text, row in zip(texts, rows, strict=True):
        m = ds.MinHash()
        m.update_batch(shingles(text, 2))
        assert_matrix(row, m.hashvalues)
