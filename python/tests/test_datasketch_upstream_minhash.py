# Ported from datasketch 2.0.0's test suite (MIT, © 2015 ekzhu).
# ruff: noqa: B905, N816, PIE808, PT009, PT027, UP031
"""datasketch 2.0.0's MinHash and LeanMinHash tests, run against pyroid.datasketch.

The bodies are datasketch's ``test/test_minhash.py``, ``test/test_lean_minhash.py``,
``test/test_minhash_schemes.py`` and ``test/test_minhash_gpu.py``, with the imports switched
to pyroid and type annotations added. Other changes, each marked in place:

- Pickles written by datasketch (the 1.10.0 fixtures) name ``datasketch.*`` classes; they are
  loaded with those names mapped to ``pyroid.datasketch.*`` (``_loads``), which is what a
  drop-in replacement must read.
- Tests of classes that pyroid delegates to datasketch (``bBitMinHash``) or has no code for
  (the CuPy shim, which needs datasketch's GPU branch) are kept and skipped. The LSH tests
  skip until ``pyroid.datasketch`` has ``MinHashLSH`` (and the forest and ensemble).

``test/test_integration.py`` has no MinHash or LeanMinHash test that runs on its own (every
test there needs a Redis or Cassandra server for the LSH storage layer), so none is ported.
"""

import base64
import io
import pickle
import struct
import types
import unittest
from typing import Any, Literal

import numpy as np
import pytest

import pyroid.datasketch
import pyroid.datasketch.minhash as _mh
from pyroid.datasketch import LeanMinHash, MinHash, minhash
from pyroid.datasketch.hashfunc import sha1_hash32, sha1_hash64
from pyroid.datasketch.minhash import _fmix

# A hashfunc other than sha1_hash32/64 warns once (pyroid.FallbackWarning); fake_hash_func is one.
pytestmark = pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning")

_Scheme = Literal["affine32", "affine64", "legacy"]

# bBitMinHash is datasketch's own class in pyroid.datasketch (delegated, not ported), so its
# tests are skipped; the name only has to resolve.
bBitMinHash: Any = None
SKIP_BBIT = "pyroid.datasketch delegates bBitMinHash to datasketch (not ported)"

# The LSH indexes are ported in pyroid.datasketch.lsh, .lshforest and .lshensemble; until they
# exist, the tests that need them skip.
MinHashLSH: Any = getattr(pyroid.datasketch, "MinHashLSH", None)
MinHashLSHForest: Any = getattr(pyroid.datasketch, "MinHashLSHForest", None)
MinHashLSHEnsemble: Any = getattr(pyroid.datasketch, "MinHashLSHEnsemble", None)
needs_lsh = pytest.mark.skipif(
    MinHashLSH is None or MinHashLSHForest is None or MinHashLSHEnsemble is None,
    reason="pyroid.datasketch has no MinHashLSH/MinHashLSHForest/MinHashLSHEnsemble yet",
)


# test/utils.py
def fake_hash_func(data: Any) -> Any:
    """Fake hash function that returns the input data unchanged."""
    return data


class _ToPyroid(pickle.Unpickler):
    """Unpickle with ``datasketch.*`` globals resolved in ``pyroid.datasketch.*``."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "datasketch" or module.startswith("datasketch."):
            module = "pyroid." + module
        return super().find_class(module, name)


def _loads(data: bytes) -> Any:
    """``pickle.loads`` of a pickle written by datasketch, as pyroid objects."""
    return _ToPyroid(io.BytesIO(data)).load()


# ---------------------------------------------------------------------------------------------
# test/test_minhash.py
# ---------------------------------------------------------------------------------------------


class TestMinHash(unittest.TestCase):
    def test_init(self) -> None:
        m1 = minhash.MinHash(4, 1, hashfunc=fake_hash_func)
        m2 = minhash.MinHash(4, 1, hashfunc=fake_hash_func)
        self.assertTrue(np.array_equal(m1.hashvalues, m2.hashvalues))
        self.assertTrue(np.array_equal(m1.permutations, m2.permutations))

    def test_is_empty(self) -> None:
        m = minhash.MinHash()
        self.assertTrue(m.is_empty())

    def test_update(self) -> None:
        m1 = minhash.MinHash(4, 1, hashfunc=fake_hash_func)
        m2 = minhash.MinHash(4, 1, hashfunc=fake_hash_func)
        m1.update(12)
        for i in range(4):
            self.assertTrue(m1.hashvalues[i] < m2.hashvalues[i])

    def test_update_batch(self) -> None:
        m1 = minhash.MinHash(4, 1, hashfunc=fake_hash_func)
        m2 = minhash.MinHash(4, 1, hashfunc=fake_hash_func)
        m1.update_batch([12, 24])
        for i in range(4):
            self.assertTrue(m1.hashvalues[i] < m2.hashvalues[i])

        m1 = minhash.MinHash(4, 1, hashfunc=fake_hash_func)
        m1.update(12)
        m1.update(24)
        m2 = minhash.MinHash(4, 1, hashfunc=fake_hash_func)
        m2.update_batch([12, 24])
        self.assertTrue(all(m1.hashvalues == m2.hashvalues))

    def test_jaccard(self) -> None:
        m1 = minhash.MinHash(4, 1, hashfunc=fake_hash_func)
        m2 = minhash.MinHash(4, 1, hashfunc=fake_hash_func)
        self.assertTrue(m1.jaccard(m2) == 1.0)
        m2.update(12)
        self.assertTrue(m1.jaccard(m2) == 0.0)
        m1.update(13)
        self.assertTrue(m1.jaccard(m2) < 1.0)

    def test_merge(self) -> None:
        m1 = minhash.MinHash(4, 1, hashfunc=fake_hash_func)
        m2 = minhash.MinHash(4, 1, hashfunc=fake_hash_func)
        m2.update(12)
        m1.merge(m2)
        self.assertTrue(m1.jaccard(m2) == 1.0)

    def test_union(self) -> None:
        m1 = minhash.MinHash(4, 1, gpu_mode="detect", hashfunc=fake_hash_func)
        m2 = minhash.MinHash(4, 1, gpu_mode="detect", hashfunc=fake_hash_func)
        m2.update(12)
        u = minhash.MinHash.union(m1, m2)
        self.assertTrue(u.jaccard(m2) == 1.0)
        self.assertIs(u.hashfunc, fake_hash_func)
        self.assertEqual(u._gpu_mode, "detect")
        u.update(13)

    def test_pickle(self) -> None:
        m = minhash.MinHash(4, 1, hashfunc=fake_hash_func)
        m.update(123)
        m.update(45)
        p = pickle.loads(pickle.dumps(m))
        self.assertEqual(p.seed, m.seed)
        self.assertTrue(np.array_equal(p.hashvalues, m.hashvalues))
        self.assertTrue(np.array_equal(p.permutations, m.permutations))

    def test_eq(self) -> None:
        m1 = minhash.MinHash(4, 1, hashfunc=fake_hash_func)
        m2 = minhash.MinHash(4, 1, hashfunc=fake_hash_func)
        m3 = minhash.MinHash(4, 2, hashfunc=fake_hash_func)
        m4 = minhash.MinHash(8, 1, hashfunc=fake_hash_func)
        m5 = minhash.MinHash(4, 1, hashfunc=fake_hash_func)
        m1.update(11)
        m2.update(12)
        m3.update(11)
        m4.update(11)
        m5.update(11)
        self.assertNotEqual(m1, m2)
        self.assertNotEqual(m1, m3)
        self.assertNotEqual(m1, m4)
        self.assertEqual(m1, m5)

        m1.update(12)
        m2.update(11)
        self.assertEqual(m1, m2)

    def test_count(self) -> None:
        m = minhash.MinHash(hashfunc=fake_hash_func)
        m.update(11)
        m.update(123)
        m.update(92)
        m.update(98)
        m.update(123218)
        m.update(32)
        c = m.count()
        self.assertGreaterEqual(c, 0)

    def test_byte_tokens(self) -> None:
        m = minhash.MinHash(4, 1)  # default scheme "affine32"
        m.update(b"Hello")
        self.assertListEqual(
            m.hashvalues.tolist(),
            [3231258861, 1388533999, 831950533, 1845606703],
        )
        m = minhash.MinHash(4, 1, scheme="affine64")
        m.update(b"Hello")
        self.assertListEqual(
            m.hashvalues.tolist(),
            [17263807126536662982, 6203611050164820875, 4141308390362884389, 4915270424612038026],
        )
        m = minhash.MinHash(4, 1, scheme="legacy")
        m.update(b"Hello")
        self.assertListEqual(
            m.hashvalues.tolist(),
            [734825475, 960773806, 359816889, 342714745],
        )

    def test_bulk(self) -> None:
        kwargs: dict[str, Any] = {"num_perm": 4, "seed": 1, "hashfunc": fake_hash_func}
        b = [[n * 4 for n in range(4)]] * 2
        m1 = minhash.MinHash(**kwargs)
        m1.update_batch(b[0])
        m2, m3 = minhash.MinHash.bulk(b, **kwargs)
        self.assertTrue(np.array_equal(m1.hashvalues, m2.hashvalues))
        self.assertTrue(np.array_equal(m1.hashvalues, m3.hashvalues))


@pytest.mark.skip(reason=SKIP_BBIT)
class TestbBitMinHash(unittest.TestCase):
    def setUp(self) -> None:
        self.m = minhash.MinHash(hashfunc=fake_hash_func)
        self.m.update(11)
        self.m.update(123)
        self.m.update(92)
        self.m.update(98)
        self.m.update(123218)
        self.m.update(32)

    def test_init(self) -> None:
        bBitMinHash(self.m, 1)
        bBitMinHash(self.m, 2)
        bBitMinHash(self.m, 3)
        bBitMinHash(self.m, 4)
        bBitMinHash(self.m, 5)
        bBitMinHash(self.m, 8)
        bBitMinHash(self.m, 12)
        bBitMinHash(self.m, 16)
        bBitMinHash(self.m, 27)
        bBitMinHash(self.m, 32)

    def test_eq(self) -> None:
        m1 = minhash.MinHash(4, 1, hashfunc=fake_hash_func)
        m2 = minhash.MinHash(4, 1, hashfunc=fake_hash_func)
        m3 = minhash.MinHash(4, 2, hashfunc=fake_hash_func)
        m4 = minhash.MinHash(8, 1, hashfunc=fake_hash_func)
        m5 = minhash.MinHash(4, 1, hashfunc=fake_hash_func)
        m1.update(11)
        m2.update(12)
        m3.update(11)
        m4.update(11)
        m5.update(11)
        m1 = bBitMinHash(m1)
        m2 = bBitMinHash(m2)
        m3 = bBitMinHash(m3)
        m4 = bBitMinHash(m4)
        m5 = bBitMinHash(m5)
        self.assertNotEqual(m1, m2)
        self.assertNotEqual(m1, m3)
        self.assertNotEqual(m1, m4)
        self.assertEqual(m1, m5)

    def test_jaccard(self) -> None:
        m1 = minhash.MinHash(4, 1, hashfunc=fake_hash_func)
        m2 = minhash.MinHash(4, 1, hashfunc=fake_hash_func)
        bm1 = bBitMinHash(m1)
        bm2 = bBitMinHash(m2)
        self.assertTrue(bm1.jaccard(bm2) == 1.0)

        m2.update(12)
        bm2 = bBitMinHash(m2)
        self.assertTrue(bm1.jaccard(bm2) < 1.0)

        m1.update(13)
        bm1 = bBitMinHash(m1)
        self.assertTrue(bm1.jaccard(bm2) < 1.0)

    def test_bytesize(self) -> None:
        s = bBitMinHash(self.m).bytesize()
        self.assertGreaterEqual(s, 8 * 2 + 4 + 1 + self.m.hashvalues.size / 64)

    def test_pickle(self) -> None:
        for num_perm in [1 << i for i in range(4, 10)]:
            m = minhash.MinHash(num_perm=num_perm, hashfunc=fake_hash_func)
            m.update(11)
            m.update(123)
            m.update(92)
            m.update(98)
            m.update(123218)
            m.update(32)
            for b in [1, 2, 3, 9, 27, 32]:
                bm = bBitMinHash(m, b)
                bm2 = pickle.loads(pickle.dumps(bm))
                self.assertEqual(bm, bm2)


# ---------------------------------------------------------------------------------------------
# test/test_lean_minhash.py
# ---------------------------------------------------------------------------------------------


class TestLeanMinHash(unittest.TestCase):
    def test_init(self) -> None:
        m1 = MinHash(4, 1, hashfunc=fake_hash_func)
        m2 = MinHash(4, 1, hashfunc=fake_hash_func)
        lm1 = LeanMinHash(m1)
        lm2 = LeanMinHash(m2)
        self.assertTrue(np.array_equal(lm1.hashvalues, lm2.hashvalues))
        self.assertTrue(np.array_equal(lm1.seed, lm2.seed))

        lm1 = LeanMinHash(seed=m1.seed, hashvalues=m1.hashvalues, scheme=m1.scheme)
        lm2 = LeanMinHash(seed=m2.seed, hashvalues=m2.hashvalues, scheme=m2.scheme)
        self.assertTrue(np.array_equal(lm1.hashvalues, lm2.hashvalues))
        self.assertTrue(np.array_equal(lm1.seed, lm2.seed))

    def test_is_empty(self) -> None:
        m = MinHash()
        lm = LeanMinHash(m)
        self.assertTrue(lm.is_empty())

    def test_update(self) -> None:
        m1 = MinHash(4, 1, hashfunc=fake_hash_func)
        try:
            lm1 = LeanMinHash(m1)
            lm1.update(12)
        except TypeError:
            pass
        else:
            raise Exception

    def test_jaccard(self) -> None:
        m1 = MinHash(4, 1, hashfunc=fake_hash_func)
        m2 = MinHash(4, 1, hashfunc=fake_hash_func)
        lm1 = LeanMinHash(m1)
        lm2 = LeanMinHash(m2)
        self.assertTrue(lm1.jaccard(lm2) == 1.0)
        m2.update(12)
        lm2 = LeanMinHash(m2)
        self.assertTrue(lm1.jaccard(lm2) == 0.0)
        m1.update(13)
        lm1 = LeanMinHash(m1)
        self.assertTrue(lm1.jaccard(lm2) < 1.0)

    def test_merge(self) -> None:
        m1 = MinHash(4, 1, hashfunc=fake_hash_func)
        m2 = MinHash(4, 1, hashfunc=fake_hash_func)
        m2.update(12)
        lm1 = LeanMinHash(m1)
        lm2 = LeanMinHash(m2)
        lm1.merge(lm2)
        self.assertTrue(lm1.jaccard(lm2) == 1.0)

    def test_union(self) -> None:
        m1 = MinHash(4, 1, hashfunc=fake_hash_func)
        m2 = MinHash(4, 1, hashfunc=fake_hash_func)
        m2.update(12)
        lm1 = LeanMinHash(m1)
        lm2 = LeanMinHash(m2)
        u = LeanMinHash.union(lm1, lm2)
        self.assertTrue(u.jaccard(lm2) == 1.0)

    def test_bytesize(self) -> None:
        m1 = MinHash(4, 1, hashfunc=fake_hash_func, scheme="legacy")
        lm1 = LeanMinHash(m1)
        self.assertTrue(lm1.bytesize() == (4 * 4) + 4 + 8)
        # The affine formats add one scheme code byte (shown here with "<",
        # which has no alignment padding).
        m2 = MinHash(4, 1, hashfunc=fake_hash_func, scheme="affine32")
        self.assertEqual(LeanMinHash(m2).bytesize("<"), (4 * 4) + 4 + 8 + 1)
        m3 = MinHash(4, 1, hashfunc=fake_hash_func, scheme="affine64")
        self.assertEqual(LeanMinHash(m3).bytesize("<"), (8 * 4) + 4 + 8 + 1)

    def test_serialize(self) -> None:
        m1 = MinHash(2, 1, hashfunc=fake_hash_func)
        lm1 = LeanMinHash(m1)
        buf = bytearray(lm1.bytesize())
        # Only test for syntax
        lm1.serialize(buf)

        m2 = MinHash(2, 1, hashfunc=fake_hash_func)
        lm2 = LeanMinHash(m2)
        size = lm1.bytesize()
        buf = bytearray(size * 2)
        lm1.serialize(buf)
        lm2.serialize(buf[size:])

    def test_deserialize(self) -> None:
        m1 = MinHash(10, 1, hashfunc=fake_hash_func)
        m1.update(123)
        lm1 = LeanMinHash(m1)
        buf = bytearray(lm1.bytesize())
        lm1.serialize(buf)

        # Test if we get back the exact same LeanMinHash objects after
        # deserializing from bytes
        lm1d = LeanMinHash.deserialize(buf)
        self.assertEqual(lm1d.seed, lm1.seed)
        self.assertEqual(len(lm1d.hashvalues), len(lm1.hashvalues))
        self.assertTrue(all(hvd == hv for hv, hvd in zip(lm1.hashvalues, lm1d.hashvalues)))

    def test_deserialize_byteorder(self) -> None:
        for byteorder in "@=<>!":
            m1 = MinHash(10, 1, hashfunc=fake_hash_func)
            m1.update(123)
            lm1 = LeanMinHash(m1)
            buf = bytearray(lm1.bytesize(byteorder))
            lm1.serialize(buf, byteorder)

            # Test if we get back the exact same LeanMinHash objects after
            # deserializing from bytes
            lm1d = LeanMinHash.deserialize(buf, byteorder)
            self.assertEqual(lm1d.seed, lm1.seed)
            self.assertEqual(len(lm1d.hashvalues), len(lm1.hashvalues))
            self.assertTrue(all(hvd == hv for hv, hvd in zip(lm1.hashvalues, lm1d.hashvalues)))

    def test_pickle(self) -> None:
        m = MinHash(4, 1, hashfunc=fake_hash_func)
        m.update(123)
        m.update(45)
        lm = LeanMinHash(m)

        p = pickle.loads(pickle.dumps(lm))
        self.assertEqual(p.seed, lm.seed)
        self.assertTrue(np.array_equal(p.hashvalues, lm.hashvalues))

    def test_eq(self) -> None:
        m1 = MinHash(4, 1, hashfunc=fake_hash_func)
        m2 = MinHash(4, 1, hashfunc=fake_hash_func)
        m3 = MinHash(4, 2, hashfunc=fake_hash_func)
        m4 = MinHash(8, 1, hashfunc=fake_hash_func)
        m5 = MinHash(4, 1, hashfunc=fake_hash_func)
        m1.update(11)
        m2.update(12)
        m3.update(11)
        m4.update(11)
        m5.update(11)
        lm1 = LeanMinHash(m1)
        lm2 = LeanMinHash(m2)
        lm3 = LeanMinHash(m3)
        lm4 = LeanMinHash(m4)
        lm5 = LeanMinHash(m5)
        self.assertNotEqual(lm1, lm2)
        self.assertNotEqual(lm1, lm3)
        self.assertNotEqual(lm1, lm4)
        self.assertEqual(lm1, lm5)

        m1.update(12)
        m2.update(11)
        lm1 = LeanMinHash(m1)
        lm2 = LeanMinHash(m2)
        self.assertEqual(lm1, lm2)

    def test_count(self) -> None:
        m = MinHash(hashfunc=fake_hash_func)
        m.update(11)
        m.update(123)
        m.update(92)
        m.update(98)
        m.update(123218)
        m.update(32)
        lm = LeanMinHash(m)
        c = lm.count()
        self.assertGreaterEqual(c, 0)

    def test_hash(self) -> None:
        m = MinHash(hashfunc=fake_hash_func)
        m.update(11)
        m.update(123)
        m.update(92)
        m.update(98)
        m.update(123218)
        m.update(32)
        lm1 = LeanMinHash(m)
        lm2 = LeanMinHash(m)
        self.assertEqual(hash(lm1), hash(lm2))
        m.update(444)
        lm3 = LeanMinHash(m)
        self.assertNotEqual(hash(lm1), hash(lm3))
        d: dict[LeanMinHash, bool] = {}  # upstream: d = dict()
        d[lm1] = True
        self.assertTrue(d[lm2])


# ---------------------------------------------------------------------------------------------
# test/test_minhash_schemes.py
# ---------------------------------------------------------------------------------------------

AFFINE_SCHEMES: tuple[_Scheme, ...] = ("affine32", "affine64")
ALL_SCHEMES: tuple[_Scheme, ...] = ("affine32", "affine64", "legacy")

# --------------------------------------------------------------------------
# Frozen format vectors. These freeze the affine schemes' permutation
# generation and hash computation: any change to these values is a breaking
# format change that invalidates persisted sketches.
# --------------------------------------------------------------------------
FROZEN_PERMUTATIONS: dict[_Scheme, tuple[list[int], list[int]]] = {
    # scheme -> (a, b) for seed=1, num_perm=4
    "affine32": (
        [3582191691, 4270784983, 1892572953, 3715639441],
        [491263, 550290313, 1298508491, 4290846341],
    ),
    "affine64": (
        [15385396165118722519, 8128538942144784529, 4219918138050323, 11154103013428313355],
        [2707168392203228096, 1703346441743126657, 3435894450290564679, 6374470260350024188],
    ),
}
FROZEN_HELLO_HASHVALUES: dict[_Scheme, list[int]] = {
    # scheme -> hashvalues of MinHash(4, 1, scheme=scheme).update(b"Hello")
    "affine32": [3231258861, 1388533999, 831950533, 1845606703],
    "affine64": [
        17263807126536662982,
        6203611050164820875,
        4141308390362884389,
        4915270424612038026,
    ],
}

# --------------------------------------------------------------------------
# Legacy fixtures generated with datasketch 1.10.0 (before the scheme
# parameter existed). They guarantee that the legacy scheme keeps producing
# bit-identical values and that old serialized payloads still load.
# --------------------------------------------------------------------------
LEGACY_TOKENS_A = [b"datasketch", b"minhash", b"affine", b"legacy", b"fixture"]
LEGACY_TOKENS_B = [b"datasketch", b"minhash", b"affine", b"probabilistic", b"structure"]
# MinHash(num_perm=8, seed=42) updated with LEGACY_TOKENS_A / LEGACY_TOKENS_B.
LEGACY_HASHVALUES_A = [
    615829395,
    403905604,
    2148529625,
    66448688,
    1847255163,
    1126317356,
    318181934,
    818505764,
]
LEGACY_HASHVALUES_B = [
    615829395,
    403905604,
    2148529625,
    23145198,
    1767462284,
    1126317356,
    1395086151,
    818505764,
]
LEGACY_JACCARD_AB = 0.625
# The permutation parameters of a legacy MinHash with num_perm=4, seed=1.
LEGACY_PERM_A = [775169054918279404, 2109959069025162, 401325382989534145, 1130051441076870728]
LEGACY_PERM_B = [1758426461858698312, 965365488286768773, 1703346441743126657, 1762784241922636284]
# pickle.dumps of the LEGACY_TOKENS_A MinHash under 1.10.0
LEGACY_MINHASH_PICKLE = base64.b64decode(
    "".join(
        [
            "gASVHwIAAAAAAACMEmRhdGFza2V0Y2gubWluaGFzaJSMB01pbkhhc2iUk5QpgZR9lCiMBHNlZWSUSyqMCG51bV9wZXJtlEsIjAho",
            "YXNoZnVuY5SME2RhdGFza2V0Y2guaGFzaGZ1bmOUjAtzaGExX2hhc2gzMpSTlIwKaGFzaHZhbHVlc5SMFm51bXB5Ll9jb3JlLm11",
            "bHRpYXJyYXmUjAxfcmVjb25zdHJ1Y3SUk5SMBW51bXB5lIwHbmRhcnJheZSTlEsAhZRDAWKUh5RSlChLAUsIhZRoD4wFZHR5cGWU",
            "k5SMAnU4lImIh5RSlChLA4wBPJROTk5K/////0r/////SwB0lGKJQ0CTz7QkAAAAAEQcExgAAAAA2fUPgAAAAAAw7fUDAAAAAHvg",
            "Gm4AAAAALD0iQwAAAAAuEvcSAAAAACRoyTAAAAAAlHSUYowMcGVybXV0YXRpb25zlGgOaBFLAIWUaBOHlFKUKEsBSwJLCIaUaBuI",
            "Q4C0PerLZtzhHw6V9S5cA2ITSNSZx2r0YxsULMuYvK5BGXoYInJm1vAH1vSXGdJB7wfLVZF1SpHeDnTtbVVXuL0daOuSJGOs4hmC",
            "oKCmlz5EFTX9cA6VBkUFV+nUuAEDTBie/UTw6/UaFYEJMwAl21sWvJMB/r8UjA6gRBOeFJnzDpR0lGKMCV9ncHVfbW9kZZSMB2Rp",
            "c2FibGWUjAZfYV9ncHWUTowGX2JfZ3B1lE51Yi4=",
        ]
    )
)
# LeanMinHash(minhash_a).serialize(buf, "<") under 1.10.0
LEGACY_LEAN_SERIALIZED_LE = bytes.fromhex(
    "2a000000000000000800000093cfb424441c1318d9f50f8030edf5037be01a6e2c3d22432e12f7122468c930"
)
# pickle.dumps(LeanMinHash(minhash_a)) under 1.10.0
LEGACY_LEAN_PICKLE = base64.b64decode(
    "".join(
        [
            "gASVewAAAAAAAACMF2RhdGFza2V0Y2gubGVhbl9taW5oYXNolIwLTGVhbk1pbkhhc2iUk5QpgZSMCGJ1aWx0aW5zlIwJYnl0ZWFy",
            "cmF5lJOUQywqAAAAAAAAAAgAAACTz7QkRBwTGNn1D4Aw7fUDe+Aabiw9IkMuEvcSJGjJMJSFlFKUYi4=",
        ]
    )
)
# pickle.dumps(bBitMinHash(minhash_a, b=4)) under 1.10.0
LEGACY_BBIT_PICKLE = base64.b64decode(
    "".join(
        [
            "gASVbQAAAAAAAACMGGRhdGFza2V0Y2guYl9iaXRfbWluaGFzaJSMC2JCaXRNaW5IYXNolJOUKYGUjAhidWlsdGluc5SMCWJ5dGVh",
            "cnJheZSTlEMdKgAAAAAAAAAEAAAAAAAAAAAIAAAAAAAAAOS8kDSUhZRSlGIu",
        ]
    )
)


class TestSchemeBasics(unittest.TestCase):
    def test_invalid_scheme_raises(self) -> None:
        with self.assertRaises(ValueError):
            MinHash(4, 1, scheme="affine128")  # type: ignore[arg-type]

    def test_default_scheme_is_affine32(self) -> None:
        m = MinHash(4, 1)
        self.assertEqual(m.scheme, "affine32")
        self.assertEqual(m.hashvalues.dtype, np.uint32)

    def test_default_hashfunc_matches_scheme_width(self) -> None:
        self.assertIs(MinHash(4, 1).hashfunc, sha1_hash32)
        self.assertIs(MinHash(4, 1, scheme="legacy").hashfunc, sha1_hash32)
        self.assertIs(MinHash(4, 1, scheme="affine64").hashfunc, sha1_hash64)
        self.assertIs(
            MinHash(4, 1, scheme="affine64", hashfunc=fake_hash_func).hashfunc, fake_hash_func
        )

    def test_dtype_and_empty_sentinel(self) -> None:
        expected: dict[_Scheme, tuple[Any, int]] = {
            "affine32": (np.uint32, (1 << 32) - 1),
            "affine64": (np.uint64, (1 << 64) - 1),
            "legacy": (np.uint64, (1 << 32) - 1),
        }
        for scheme, (dtype, sentinel) in expected.items():
            m = MinHash(4, 1, scheme=scheme)
            self.assertEqual(m.hashvalues.dtype, dtype, scheme)
            self.assertTrue((m.hashvalues == sentinel).all(), scheme)
            self.assertTrue(m.is_empty(), scheme)
            m.update(b"x")
            self.assertFalse(m.is_empty(), scheme)
            m.clear()
            self.assertTrue(m.is_empty(), scheme)

    def test_update_matches_update_batch(self) -> None:
        for scheme in ALL_SCHEMES:
            m1 = MinHash(16, 1, scheme=scheme)
            for t in (b"a", b"b", b"c"):
                m1.update(t)
            m2 = MinHash(16, 1, scheme=scheme)
            m2.update_batch([b"a", b"b", b"c"])
            self.assertTrue(np.array_equal(m1.hashvalues, m2.hashvalues), scheme)

    def test_jaccard_identical_and_disjoint(self) -> None:
        for scheme in ALL_SCHEMES:
            m1 = MinHash(32, 1, scheme=scheme, hashfunc=fake_hash_func)
            m2 = MinHash(32, 1, scheme=scheme, hashfunc=fake_hash_func)
            self.assertEqual(m1.jaccard(m2), 1.0, scheme)
            m2.update(12)
            self.assertEqual(m1.jaccard(m2), 0.0, scheme)

    def test_count(self) -> None:
        for scheme in ALL_SCHEMES:
            m = MinHash(scheme=scheme)
            m.update_batch([b"a", b"b", b"c", b"d", b"e", b"f"])
            self.assertGreaterEqual(m.count(), 0, scheme)

    def test_copy_merge_union_preserve_scheme(self) -> None:
        for scheme in ALL_SCHEMES:
            m1 = MinHash(16, 1, scheme=scheme)
            m1.update(b"a")
            m2 = MinHash(16, 1, scheme=scheme)
            m2.update(b"b")
            c = m1.copy()
            self.assertEqual(c.scheme, scheme)
            self.assertEqual(c, m1)
            u = MinHash.union(m1, m2)
            self.assertEqual(u.scheme, scheme)
            m1.merge(m2)
            self.assertEqual(u, m1)

    def test_pickle_roundtrip(self) -> None:
        for scheme in ALL_SCHEMES:
            m = MinHash(16, 1, scheme=scheme)
            m.update(b"a")
            p = pickle.loads(pickle.dumps(m))
            self.assertEqual(p.scheme, scheme)
            self.assertEqual(p, m)
            self.assertTrue(
                np.array_equal(np.asarray(p.permutations), np.asarray(m.permutations)), scheme
            )

    def test_bulk_and_generator_pass_scheme(self) -> None:
        data = [[b"a", b"b"], [b"c", b"d"]]
        for scheme in ALL_SCHEMES:
            for m in MinHash.bulk(data, num_perm=8, scheme=scheme):
                self.assertEqual(m.scheme, scheme)

    def test_eq_differs_across_schemes(self) -> None:
        m32 = MinHash(4, 1, scheme="affine32")
        m64 = MinHash(4, 1, scheme="affine64")
        ml = MinHash(4, 1, scheme="legacy")
        self.assertNotEqual(m32, m64)
        self.assertNotEqual(m32, ml)
        self.assertNotEqual(m64, ml)

    def test_cross_scheme_operations_refused(self) -> None:
        m32 = MinHash(8, 1, scheme="affine32")
        ml = MinHash(8, 1, scheme="legacy")
        with self.assertRaises(ValueError):
            m32.jaccard(ml)
        with self.assertRaises(ValueError):
            m32.merge(ml)
        with self.assertRaises(ValueError):
            MinHash.union(m32, ml)

    def test_hashfunc_out_of_range_raises(self) -> None:
        m = MinHash(4, 1, scheme="affine32", hashfunc=lambda x: 1 << 40)
        with self.assertRaises(ValueError):
            m.update(b"x")
        with self.assertRaises(ValueError):
            m.update_batch([b"x"])

    def test_affine64_uses_full_range(self) -> None:
        m = MinHash(64, 1, scheme="affine64")
        m.update_batch([b"a", b"b", b"c"])
        self.assertTrue((m.hashvalues >= (1 << 32)).any())

    def test_permutation_validation(self) -> None:
        # Even `a` breaks bijectivity and must be rejected.
        with self.assertRaises(ValueError):
            MinHash(num_perm=2, seed=1, scheme="affine32", permutations=([3, 2], [0, 1]))
        # Parameters beyond the scheme width must be rejected, not wrapped.
        with self.assertRaises(ValueError):
            MinHash(
                num_perm=2,
                seed=1,
                scheme="affine32",
                permutations=(
                    np.array([1 << 40, 3], dtype=np.uint64),
                    np.array([0, 1], dtype=np.uint64),
                ),
            )
        # Valid parameters round-trip through the constructor (e.g. copy()).
        m = MinHash(num_perm=8, seed=1, scheme="affine64")
        m2 = MinHash(num_perm=8, seed=1, scheme="affine64", permutations=m.permutations)
        self.assertTrue(np.array_equal(np.asarray(m.permutations), np.asarray(m2.permutations)))

    def test_hashvalues_out_of_range_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MinHash(seed=1, scheme="affine32", hashvalues=np.array([1 << 40, 1], dtype=np.uint64))

    def test_hashvalues_of_python_ints_convert_exactly(self) -> None:
        # dtype inference on mixed-magnitude Python ints would produce
        # float64 and silently round 64-bit values.
        values = [1, (1 << 64) - 59]
        m = MinHash(seed=1, scheme="affine64", hashvalues=values, permutations=([1, 3], [0, 1]))
        self.assertEqual(m.hashvalues.tolist(), values)


class TestExplicitSchemeRequirement(unittest.TestCase):
    """Constructing from raw hash values or permutations requires an explicit
    scheme: the values carry no trace of the scheme that produced them, so a
    default would silently mislabel pre-2.0.0 state and defeat the
    cross-scheme guards.
    """

    def test_minhash_raw_hashvalues_require_scheme(self) -> None:
        m = MinHash(8, 1, scheme="legacy")
        with self.assertRaises(ValueError) as ctx:
            MinHash(seed=m.seed, hashvalues=m.hashvalues)
        self.assertIn("legacy", str(ctx.exception))
        rebuilt = MinHash(seed=m.seed, hashvalues=m.hashvalues, scheme="legacy")
        self.assertEqual(rebuilt.jaccard(m), 1.0)

    def test_minhash_raw_permutations_require_scheme(self) -> None:
        m = MinHash(8, 1)
        with self.assertRaises(ValueError):
            MinHash(num_perm=8, seed=1, permutations=m.permutations)

    def test_lean_minhash_raw_hashvalues_require_scheme(self) -> None:
        m = MinHash(8, 1)
        with self.assertRaises(ValueError) as ctx:
            LeanMinHash(seed=m.seed, hashvalues=m.hashvalues)
        self.assertIn("legacy", str(ctx.exception))

    def test_lean_minhash_scheme_conflicting_with_minhash_rejected(self) -> None:
        m = MinHash(8, 1, scheme="affine32")
        with self.assertRaises(ValueError):
            LeanMinHash(m, scheme="legacy")
        self.assertEqual(LeanMinHash(m, scheme="affine32").scheme, "affine32")

    def test_num_perm_must_be_positive(self) -> None:
        for scheme in ALL_SCHEMES:
            with self.assertRaises(ValueError):
                MinHash(0, 1, scheme=scheme)
        with self.assertRaises(ValueError):
            MinHash(seed=1, hashvalues=[], scheme="affine32")

    def test_empty_affine_lean_minhash_rejected(self) -> None:
        # An empty affine sketch would serialize with a hash value count of
        # zero, indistinguishable from the legacy format on deserialization.
        with self.assertRaises(ValueError):
            LeanMinHash(seed=1, hashvalues=[], scheme="affine32")
        # Empty legacy payloads (possible under 1.x) still round-trip.
        lm = LeanMinHash(seed=1, hashvalues=[], scheme="legacy")
        buf = bytearray(lm.bytesize("<"))
        lm.serialize(buf, "<")
        self.assertEqual(LeanMinHash.deserialize(buf, "<").scheme, "legacy")

    def test_negative_hashvalues_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MinHash(seed=1, hashvalues=np.array([-1, 2], dtype=np.int64), scheme="affine32")
        with self.assertRaises(ValueError):
            MinHash(seed=1, hashvalues=[-1, 2], scheme="affine64")


@needs_lsh
class TestLSHSchemeGuards(unittest.TestCase):
    """LSH indexes must refuse MinHash of a scheme other than the one they
    were built with: mixed schemes silently return wrong results otherwise.
    """

    def _mh(self, scheme: _Scheme, seed: int = 1, num_perm: int = 128) -> MinHash:
        m = MinHash(num_perm, seed, scheme=scheme)
        m.update_batch([b"a", b"b", b"c"])
        return m

    def test_lsh_insert_and_query_mixed_scheme_raise(self) -> None:
        lsh = MinHashLSH(threshold=0.5, num_perm=128)
        lsh.insert("k32", self._mh("affine32"))
        with self.assertRaises(ValueError):
            lsh.insert("kleg", self._mh("legacy"))
        with self.assertRaises(ValueError):
            lsh.query(self._mh("affine64"))
        with self.assertRaises(ValueError):
            lsh.add_to_query_buffer(self._mh("affine64"))
        self.assertIn("k32", lsh.query(self._mh("affine32")))

    def test_lsh_insertion_session_guarded(self) -> None:
        lsh = MinHashLSH(threshold=0.5, num_perm=128)
        with lsh.insertion_session() as session:
            session.insert("k", self._mh("affine32"))
            with self.assertRaises(ValueError):
                session.insert("k2", self._mh("legacy"))

    def test_lsh_merge_mixed_scheme_raises(self) -> None:
        a = MinHashLSH(threshold=0.5, num_perm=128)
        a.insert("x", self._mh("affine32"))
        b = MinHashLSH(threshold=0.5, num_perm=128)
        b.insert("y", self._mh("legacy"))
        with self.assertRaises(ValueError):
            a.merge(b)
        # Merging propagates the learned scheme to an empty index.
        c = MinHashLSH(threshold=0.5, num_perm=128)
        c.merge(a)
        with self.assertRaises(ValueError):
            c.insert("z", self._mh("legacy"))

    def test_lsh_pickle_preserves_learned_scheme(self) -> None:
        lsh = MinHashLSH(threshold=0.5, num_perm=128)
        lsh.insert("k", self._mh("affine32"))
        restored = pickle.loads(pickle.dumps(lsh))
        with self.assertRaises(ValueError):
            restored.query(self._mh("legacy"))
        self.assertIn("k", restored.query(self._mh("affine32")))

    def test_lshforest_mixed_scheme_raises(self) -> None:
        forest = MinHashLSHForest(num_perm=128)
        forest.add("k", self._mh("affine32"))
        with self.assertRaises(ValueError):
            forest.add("k2", self._mh("legacy"))
        forest.index()
        with self.assertRaises(ValueError):
            forest.query(self._mh("legacy"), 1)
        self.assertIn("k", forest.query(self._mh("affine32"), 1))

    def test_lshensemble_mixed_scheme_raises(self) -> None:
        ensemble = MinHashLSHEnsemble(threshold=0.8, num_perm=128, num_part=2)
        ensemble.index([("k", self._mh("affine32"), 3)])
        self.assertIn("k", list(ensemble.query(self._mh("affine32"), 3)))
        with self.assertRaises(ValueError):
            list(ensemble.query(self._mh("legacy"), 3))

    def test_weighted_minhash_exempt(self) -> None:
        # pyroid delegates WeightedMinHashGenerator to datasketch (upstream imports it).
        WeightedMinHashGenerator = pyroid.datasketch.WeightedMinHashGenerator  # noqa: N806
        # WeightedMinHash has no permutation scheme and must keep working.
        gen = WeightedMinHashGenerator(10, sample_size=128, seed=1)
        v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        lsh = MinHashLSH(threshold=0.5, num_perm=128)
        lsh.insert("w1", gen.minhash(v))
        self.assertIn("w1", lsh.query(gen.minhash(v)))


class TestFrozenFormat(unittest.TestCase):
    """Freeze the affine formats: a change here invalidates persisted sketches."""

    def test_permutations_frozen(self) -> None:
        for scheme, (a, b) in FROZEN_PERMUTATIONS.items():
            m = MinHash(num_perm=4, seed=1, scheme=scheme)
            self.assertEqual(m.permutations[0].tolist(), a, scheme)
            self.assertEqual(m.permutations[1].tolist(), b, scheme)

    def test_hashvalues_frozen(self) -> None:
        for scheme, expected in FROZEN_HELLO_HASHVALUES.items():
            m = MinHash(4, 1, scheme=scheme)
            m.update(b"Hello")
            self.assertEqual(m.hashvalues.tolist(), expected, scheme)

    def test_permutations_deterministic_per_seed(self) -> None:
        for scheme in AFFINE_SCHEMES:
            p1 = np.asarray(MinHash(16, 7, scheme=scheme).permutations)
            p2 = np.asarray(MinHash(16, 7, scheme=scheme).permutations)
            p3 = np.asarray(MinHash(16, 8, scheme=scheme).permutations)
            self.assertTrue(np.array_equal(p1, p2), scheme)
            self.assertFalse(np.array_equal(p1, p3), scheme)


class TestBijectivity(unittest.TestCase):
    def test_a_parameters_are_odd(self) -> None:
        for scheme in AFFINE_SCHEMES:
            for seed in range(8):
                a = MinHash(num_perm=256, seed=seed, scheme=scheme).permutations[0]
                self.assertTrue((a % 2 == 1).all(), scheme)

    def test_fmix_is_injective_on_sample(self) -> None:
        rng = np.random.RandomState(5)
        x32 = np.unique(rng.randint(0, 1 << 32, size=200000, dtype=np.uint32))
        y32 = _fmix(x32, 32)
        self.assertEqual(y32.dtype, np.uint32)
        self.assertEqual(len(np.unique(y32)), len(x32))
        x64 = np.unique(rng.randint(0, 1 << 63, size=200000, dtype=np.uint64) * np.uint64(2))
        y64 = _fmix(x64, 64)
        self.assertEqual(y64.dtype, np.uint64)
        self.assertEqual(len(np.unique(y64)), len(x64))

    def test_fmix_fixes_zero(self) -> None:
        # fmix consists of xor-shift and multiplication steps, all of which
        # map zero to zero; a nonzero result would indicate broken constants.
        self.assertEqual(int(_fmix(np.array([0], dtype=np.uint32), 32)[0]), 0)
        self.assertEqual(int(_fmix(np.array([0], dtype=np.uint64), 64)[0]), 0)

    def test_permutation_is_injective_on_sample(self) -> None:
        rng = np.random.RandomState(6)
        cases: tuple[tuple[_Scheme, Any, int], ...] = (
            ("affine32", np.uint32, 32),
            ("affine64", np.uint64, 64),
        )
        for scheme, dtype, width in cases:
            m = MinHash(num_perm=2, seed=3, scheme=scheme)
            a, b = m.permutations[0][0], m.permutations[1][0]
            x = np.unique(rng.randint(0, 1 << min(width, 62), size=100000, dtype=dtype))
            phv = a * x + b
            self.assertEqual(len(np.unique(phv)), len(x), scheme)


class TestStatisticalBehavior(unittest.TestCase):
    """Seeded (deterministic) ports of the issue #212 bias experiments."""

    def _structured_bias(
        self, scheme: _Scheme, trials: int = 12, n: int = 8000, num_perm: int = 128
    ) -> float:
        # Sequential integers hashed with the identity function: true J = 1/3.
        # Without the fmix pre-mix the affine schemes would be biased by
        # about -0.05 here; the legacy scheme is biased regardless.
        errors = []
        for seed in range(trials):
            ma = MinHash(num_perm, seed=seed, hashfunc=fake_hash_func, scheme=scheme)
            mb = MinHash(num_perm, seed=seed, hashfunc=fake_hash_func, scheme=scheme)
            ma.update_batch(range(0, 2 * n))
            mb.update_batch(range(n, 3 * n))
            errors.append(ma.jaccard(mb) - 1.0 / 3.0)
        return sum(errors) / len(errors)

    def test_affine_unbiased_on_structured_input(self) -> None:
        for scheme in AFFINE_SCHEMES:
            self.assertLess(abs(self._structured_bias(scheme)), 0.03, scheme)

    def test_legacy_biased_on_structured_input(self) -> None:
        # Guards the test harness itself: the legacy scheme must reproduce
        # the known bias on this input (measured -0.061 for these seeds).
        self.assertLess(self._structured_bias("legacy"), -0.04)

    def _structured_count_error(
        self, scheme: _Scheme, trials: int = 12, n: int = 5000, num_perm: int = 256
    ) -> float:
        # Relative error of count() on sequential integers under an identity
        # hash -- the structured-input class the fmix pre-mix was built for.
        errors = []
        for seed in range(trials):
            m = MinHash(num_perm, seed=seed, hashfunc=fake_hash_func, scheme=scheme)
            m.update_batch(range(n))
            errors.append(m.count() / n - 1.0)
        return sum(errors) / len(errors)

    def test_affine_count_accurate_on_structured_input(self) -> None:
        # affine's fmix keeps cardinality estimates accurate (measured
        # about +0.013 to +0.027 for these seeds) well within tolerance.
        for scheme in AFFINE_SCHEMES:
            self.assertLess(abs(self._structured_count_error(scheme)), 0.10, scheme)

    def test_legacy_count_collapses_on_structured_input(self) -> None:
        # The legacy permutation collapses on structured input, so count()
        # is far off (measured about -0.40 for these seeds).
        self.assertLess(self._structured_count_error("legacy"), -0.15)

    def test_affine_accuracy_on_random_sets(self) -> None:
        rng = np.random.RandomState(99)
        for scheme in AFFINE_SCHEMES:
            errors = []
            for t in range(6):
                u = list(dict.fromkeys(rng.randint(0, 1 << 31, size=9000).tolist()))
                common, a_only, b_only = u[:2500], u[2500:5000], u[5000:7500]
                sa, sb = common + a_only, common + b_only
                true_j = len(set(sa) & set(sb)) / len(set(sa) | set(sb))
                ma = MinHash(128, seed=t, hashfunc=fake_hash_func, scheme=scheme)
                mb = MinHash(128, seed=t, hashfunc=fake_hash_func, scheme=scheme)
                ma.update_batch(sa)
                mb.update_batch(sb)
                errors.append(ma.jaccard(mb) - true_j)
            self.assertLess(abs(sum(errors) / len(errors)), 0.05, scheme)


class TestLegacyCompatibility(unittest.TestCase):
    """Bit-exact compatibility with sketches created by datasketch 1.10.0."""

    def _legacy_minhash(self, tokens: list[bytes]) -> MinHash:
        m = MinHash(num_perm=8, seed=42, scheme="legacy")
        for t in tokens:
            m.update(t)
        return m

    def test_legacy_update_bit_identical(self) -> None:
        self.assertEqual(
            self._legacy_minhash(LEGACY_TOKENS_A).hashvalues.tolist(), LEGACY_HASHVALUES_A
        )
        self.assertEqual(
            self._legacy_minhash(LEGACY_TOKENS_B).hashvalues.tolist(), LEGACY_HASHVALUES_B
        )

    def test_legacy_permutations_bit_identical(self) -> None:
        m = MinHash(num_perm=4, seed=1, scheme="legacy")
        self.assertEqual(m.permutations[0].tolist(), LEGACY_PERM_A)
        self.assertEqual(m.permutations[1].tolist(), LEGACY_PERM_B)

    def test_old_minhash_pickle_loads_as_legacy(self) -> None:
        old = _loads(LEGACY_MINHASH_PICKLE)  # upstream: pickle.loads
        self.assertIs(type(old), MinHash)  # pyroid: the pickle's classes resolve in pyroid
        self.assertEqual(old.scheme, "legacy")
        self.assertEqual(old.hashvalues.tolist(), LEGACY_HASHVALUES_A)
        self.assertAlmostEqual(
            old.jaccard(self._legacy_minhash(LEGACY_TOKENS_B)), LEGACY_JACCARD_AB
        )
        # Still updatable with bit-identical semantics.
        old.update(b"more")
        m = self._legacy_minhash([*LEGACY_TOKENS_A, b"more"])
        self.assertEqual(old.hashvalues.tolist(), m.hashvalues.tolist())

    def test_minhash_unpickles_without_scheme_as_legacy(self) -> None:
        m = MinHash(4, 1, hashfunc=fake_hash_func, scheme="legacy")
        state = m.__getstate__()
        state.pop("scheme")
        restored = MinHash.__new__(MinHash)
        restored.__setstate__(state)
        self.assertEqual(restored.scheme, "legacy")

    def test_old_lean_bytes_deserialize_and_reserialize_identically(self) -> None:
        lm = LeanMinHash.deserialize(LEGACY_LEAN_SERIALIZED_LE, "<")
        self.assertEqual(lm.scheme, "legacy")
        self.assertEqual(lm.seed, 42)
        self.assertEqual(lm.hashvalues.tolist(), LEGACY_HASHVALUES_A)
        buf = bytearray(lm.bytesize("<"))
        lm.serialize(buf, "<")
        self.assertEqual(bytes(buf), LEGACY_LEAN_SERIALIZED_LE)

    def test_old_lean_pickle_loads_and_state_unchanged(self) -> None:
        lm = _loads(LEGACY_LEAN_PICKLE)  # upstream: pickle.loads
        self.assertIs(type(lm), LeanMinHash)  # pyroid: the pickle's classes resolve in pyroid
        self.assertEqual(lm.scheme, "legacy")
        self.assertEqual(lm.hashvalues.tolist(), LEGACY_HASHVALUES_A)
        # A legacy LeanMinHash pickled by this version keeps the old byte
        # format, so it remains readable by versions before 2.0.0.
        self.assertEqual(pickle.loads(pickle.dumps(lm)), lm)
        self.assertEqual(bytes(lm.__getstate__()), bytes(_loads(LEGACY_LEAN_PICKLE).__getstate__()))

    @pytest.mark.skip(reason=SKIP_BBIT)
    def test_old_bbit_pickle_loads_as_legacy(self) -> None:
        bb = pickle.loads(LEGACY_BBIT_PICKLE)
        self.assertEqual(bb.scheme, "legacy")
        self.assertEqual(bb.seed, 42)
        self.assertEqual(bb.b, 4)
        rebuilt = bBitMinHash(self._legacy_minhash(LEGACY_TOKENS_A), b=4)
        self.assertEqual(bb, rebuilt)
        self.assertEqual(bytes(bb.__getstate__()), bytes(rebuilt.__getstate__()))


class TestLeanMinHashSchemes(unittest.TestCase):
    def _lean(self, scheme: _Scheme, num_perm: int = 16) -> LeanMinHash:
        m = MinHash(num_perm, 1, scheme=scheme)
        m.update_batch([b"a", b"b", b"c"])
        return LeanMinHash(m)

    def test_scheme_taken_from_minhash(self) -> None:
        for scheme in ALL_SCHEMES:
            self.assertEqual(self._lean(scheme).scheme, scheme)

    def test_init_from_seed_and_hashvalues(self) -> None:
        for scheme in ALL_SCHEMES:
            m = MinHash(16, 1, scheme=scheme)
            m.update(b"a")
            lm = LeanMinHash(seed=m.seed, hashvalues=m.hashvalues, scheme=scheme)
            self.assertEqual(lm.scheme, scheme)
            self.assertEqual(lm.jaccard(m), 1.0)

    def test_serialize_roundtrip_all_byteorders(self) -> None:
        for scheme in ALL_SCHEMES:
            lm = self._lean(scheme)
            for byteorder in "@=<>!":
                buf = bytearray(lm.bytesize(byteorder))
                lm.serialize(buf, byteorder)
                lm2 = LeanMinHash.deserialize(buf, byteorder)
                self.assertEqual(lm2, lm, (scheme, byteorder))
                self.assertEqual(lm2.scheme, scheme, (scheme, byteorder))

    def test_affine_payload_layout(self) -> None:
        # seed:int64, negated num_perm:int32, scheme code:uint8, values.
        cases: tuple[tuple[_Scheme, int, str], ...] = (("affine32", 1, "I"), ("affine64", 2, "Q"))
        for scheme, code, value_fmt in cases:
            lm = self._lean(scheme, num_perm=4)
            self.assertEqual(lm.bytesize("<"), struct.calcsize("<qiB4%s" % value_fmt), scheme)
            buf = bytearray(lm.bytesize("<"))
            lm.serialize(buf, "<")
            seed, num_perm_tag, scheme_code = struct.unpack_from("<qiB", buf, 0)
            self.assertEqual(seed, 1)
            self.assertEqual(num_perm_tag, -4)
            self.assertEqual(scheme_code, code)

    def test_unknown_scheme_code_raises(self) -> None:
        buf = bytearray(struct.calcsize("<qiB2I"))
        struct.pack_into("<qiB2I", buf, 0, 1, -2, 99, 5, 6)
        with self.assertRaises(ValueError):
            LeanMinHash.deserialize(buf, "<")

    def test_pickle_roundtrip(self) -> None:
        for scheme in ALL_SCHEMES:
            lm = self._lean(scheme)
            lm2 = pickle.loads(pickle.dumps(lm))
            self.assertEqual(lm2, lm)
            self.assertEqual(lm2.scheme, scheme)

    def test_copy(self) -> None:
        for scheme in ALL_SCHEMES:
            lm = self._lean(scheme)
            c = lm.copy()
            self.assertEqual(c, lm)
            self.assertEqual(c.scheme, scheme)

    def test_union_scheme_guard(self) -> None:
        lm32a, lm32b = self._lean("affine32"), self._lean("affine32")
        u = LeanMinHash.union(lm32a, lm32b)
        self.assertEqual(u.scheme, "affine32")
        with self.assertRaises(ValueError):
            LeanMinHash.union(lm32a, self._lean("legacy"))

    def test_cross_scheme_jaccard_refused(self) -> None:
        with self.assertRaises(ValueError):
            self._lean("affine32").jaccard(self._lean("legacy"))


@pytest.mark.skip(reason=SKIP_BBIT)
class TestBBitSchemes(unittest.TestCase):
    def test_scheme_mismatch_raises(self) -> None:
        m32 = MinHash(16, 1, scheme="affine32")
        ml = MinHash(16, 1, scheme="legacy")
        with self.assertRaises(ValueError):
            bBitMinHash(m32).jaccard(bBitMinHash(ml))

    def test_pickle_roundtrip_per_scheme(self) -> None:
        for scheme in ALL_SCHEMES:
            m = MinHash(64, 1, scheme=scheme)
            m.update_batch([b"a", b"b", b"c"])
            for b in (1, 4, 32):
                bb = bBitMinHash(m, b)
                bb2 = pickle.loads(pickle.dumps(bb))
                self.assertEqual(bb, bb2, (scheme, b))
                self.assertEqual(bb2.scheme, scheme, (scheme, b))

    def test_estimation_accuracy_per_scheme(self) -> None:
        # b-bit keeps the *low* bits of each min hash value, which are the
        # softest bits of a multiply-shift (affine) output, so verify the
        # estimate stays accurate under every scheme. Two sets of size 1000
        # sharing 500 elements have true Jaccard 500 / 1500 = 1/3.
        true_j = 1.0 / 3.0
        a = [b"w%d" % i for i in range(0, 1000)]
        b_set = [b"w%d" % i for i in range(500, 1500)]
        for scheme in ALL_SCHEMES:
            ma = MinHash(num_perm=512, seed=7, scheme=scheme)
            mb = MinHash(num_perm=512, seed=7, scheme=scheme)
            ma.update_batch(a)
            mb.update_batch(b_set)
            for nbits in (1, 4, 8, 16, 32):
                est = bBitMinHash(ma, nbits).jaccard(bBitMinHash(mb, nbits))
                self.assertLess(abs(est - true_j), 0.1, (scheme, nbits))
            # Keeping all 32 bits reproduces the full MinHash estimate.
            est32 = bBitMinHash(ma, 32).jaccard(bBitMinHash(mb, 32))
            self.assertLess(abs(est32 - ma.jaccard(mb)), 0.02, scheme)

    def test_setstate_unknown_scheme_code_raises(self) -> None:
        # The pickle path (__setstate__) must reject a corrupt scheme byte,
        # like deserialize() does. The byte sits right after the fixed header:
        # "qi" (seed, negated size) for LeanMinHash, "<qBdi" (seed, b, r,
        # negated size) for bBitMinHash. Valid codes are 1/2; 99 is unknown.
        m = MinHash(16, 1, scheme="affine32")
        m.update_batch([b"a", b"b", b"c"])

        lean_buf = bytearray(LeanMinHash(m).__getstate__())
        lean_buf[struct.calcsize("qi")] = 99
        with self.assertRaises(ValueError):
            LeanMinHash.__new__(LeanMinHash).__setstate__(lean_buf)

        bbit_buf = bytearray(bBitMinHash(m, 4).__getstate__())
        bbit_buf[struct.calcsize("<qBdi")] = 99
        with self.assertRaises(ValueError):
            bBitMinHash.__new__(bBitMinHash).__setstate__(bbit_buf)


class TestLeanMinHashSetstateScheme(unittest.TestCase):
    # pyroid: the LeanMinHash half of TestBBitSchemes.test_setstate_unknown_scheme_code_raises
    # (skipped above with the rest of that class), unchanged.
    def test_setstate_unknown_scheme_code_raises(self) -> None:
        m = MinHash(16, 1, scheme="affine32")
        m.update_batch([b"a", b"b", b"c"])

        lean_buf = bytearray(LeanMinHash(m).__getstate__())
        lean_buf[struct.calcsize("qi")] = 99
        with self.assertRaises(ValueError):
            LeanMinHash.__new__(LeanMinHash).__setstate__(lean_buf)


@needs_lsh
class TestLSHIntegration(unittest.TestCase):
    def _tokens(self, words: list[str]) -> list[bytes]:
        return [w.encode("utf8") for w in words]

    def test_minhash_lsh(self) -> None:
        base = [
            "minhash",
            "is",
            "a",
            "probabilistic",
            "data",
            "structure",
            "for",
            "estimating",
            "similarity",
        ]
        near = [*base[:-1], "resemblance"]
        far = [
            "completely",
            "different",
            "words",
            "with",
            "no",
            "overlap",
            "at",
            "all",
            "in",
            "this",
            "set",
        ]
        for scheme in ALL_SCHEMES:
            mq, mn, mf = (MinHash(128, scheme=scheme) for _ in range(3))
            mq.update_batch(self._tokens(base))
            mn.update_batch(self._tokens(near))
            mf.update_batch(self._tokens(far))
            lsh = MinHashLSH(threshold=0.5, num_perm=128)
            lsh.insert("near", mn)
            lsh.insert("far", mf)
            results = lsh.query(mq)
            self.assertIn("near", results, scheme)
            self.assertNotIn("far", results, scheme)

    def test_lshforest_roundtrip_hashvalues(self) -> None:
        tokens = self._tokens(
            ["minhash", "lsh", "forest", "top", "k", "query", "jaccard", "similarity"]
        )
        for scheme in ALL_SCHEMES:
            m = MinHash(128, scheme=scheme)
            m.update_batch(tokens)
            forest = MinHashLSHForest(num_perm=128)
            forest.add("key", m)
            forest.index()
            self.assertIn("key", forest.query(m, 1))
            hashvalues = forest.get_minhash_hashvalues("key")
            self.assertEqual(hashvalues.dtype, m.hashvalues.dtype, scheme)
            self.assertTrue(np.array_equal(hashvalues, m.hashvalues), scheme)


# ---------------------------------------------------------------------------------------------
# test/test_minhash_gpu.py
# ---------------------------------------------------------------------------------------------

# Robust GPU availability check
try:
    import cupy as cp  # type: ignore[import-not-found]

    try:
        GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        GPU_AVAILABLE = False
except Exception:
    GPU_AVAILABLE = False


def _make_data(n: int) -> list[bytes]:
    return [f"token-{i}".encode() for i in range(n)]


# The GPU path branches on the permutation scheme (legacy modular
# arithmetic vs affine multiply-shift), so parity must hold per scheme.
SCHEMES: list[_Scheme] = ["affine32", "affine64", "legacy"]


class TestMinHashGPU(unittest.TestCase):
    @unittest.skipUnless(GPU_AVAILABLE, "CuPy/CUDA not available")
    def test_update_batch_gpu_matches_cpu(self) -> None:
        data = _make_data(1000)

        for scheme in SCHEMES:
            with self.subTest(scheme=scheme):
                m_cpu = MinHash(num_perm=256, seed=7, gpu_mode="disable", scheme=scheme)
                m_cpu.update_batch(data)

                # Force GPU path
                m_gpu = MinHash(num_perm=256, seed=7, gpu_mode="always", scheme=scheme)
                m_gpu.update_batch(data)

                self.assertEqual(m_cpu.hashvalues.dtype, m_gpu.hashvalues.dtype)
                self.assertTrue(np.array_equal(m_cpu.hashvalues, m_gpu.hashvalues))

    @unittest.skipUnless(GPU_AVAILABLE, "CuPy/CUDA not available")
    def test_detect_mode_matches_cpu(self) -> None:
        """Auto-detect should produce identical results as pure CPU."""
        data1 = _make_data(500)
        data2 = _make_data(700)

        for scheme in SCHEMES:
            with self.subTest(scheme=scheme):
                m_cpu = MinHash(num_perm=128, seed=7, gpu_mode="disable", scheme=scheme)
                m_cpu.update_batch(data1)
                m_cpu.update_batch(data2)

                m_auto = MinHash(num_perm=128, seed=7, gpu_mode="detect", scheme=scheme)
                m_auto.update_batch(data1)
                m_auto.update_batch(data2)

                self.assertTrue(np.array_equal(m_cpu.hashvalues, m_auto.hashvalues))

    def test_pickle_roundtrip_is_portable(self) -> None:
        """Pickle should drop device state so round-tripped objects are portable.
        After unpickling, update_batch should still work and populate caches
        only if GPU is available and mode permits it.
        """
        m = MinHash(num_perm=128, seed=7, gpu_mode="detect")
        m2 = pickle.loads(pickle.dumps(m))

        # Should be able to update on any machine
        m2.update_batch(_make_data(64))

        # GPU caches presence should reflect availability & mode
        if "GPU_AVAILABLE" in globals() and GPU_AVAILABLE and m2._gpu_mode in ("detect", "always"):
            self.assertIsNotNone(m2._a_gpu)
            self.assertIsNotNone(m2._b_gpu)
        else:
            self.assertIsNone(m2._a_gpu)
            self.assertIsNone(m2._b_gpu)

    def test_always_mode_raises_when_no_device(self) -> None:
        """If GPU is unavailable, 'always' must raise at call-time."""
        if GPU_AVAILABLE:
            self.skipTest("GPU available; cannot force negative path.")
        m = MinHash(num_perm=64, seed=1, gpu_mode="always")
        with self.assertRaises(RuntimeError):
            m.update_batch(_make_data(32))


@pytest.mark.skip(
    reason="pyroid has no CuPy branch to inject a shim into: it always computes on the CPU"
)
class TestGPUPathOnCPU(unittest.TestCase):
    """Exercise the GPU branch of update_batch without a CUDA device by
    injecting a numpy-backed CuPy stand-in. This runs the same code the GPU
    path runs (per scheme) and asserts it matches the CPU path -- unlike the
    hardware-gated tests above, it always runs.
    """

    @staticmethod
    def _shim() -> Any:
        # Only the cupy surface update_batch / _ensure_gpu_caches touch.
        return types.SimpleNamespace(
            uint32=np.uint32,
            uint64=np.uint64,
            asarray=np.asarray,
            minimum=np.minimum,
            min=np.min,
            bitwise_and=np.bitwise_and,
            asnumpy=np.asarray,
        )

    def _run_with_shim(self, cp_module: Any, gpu_cache: bool) -> dict[_Scheme, Any]:
        saved = (_mh.cp, _mh._GPU_OK_CACHE)  # type: ignore[attr-defined]
        _mh.cp, _mh._GPU_OK_CACHE = cp_module, gpu_cache  # type: ignore[attr-defined]
        try:
            data = _make_data(500)
            results = {}
            for scheme in SCHEMES:
                m = MinHash(num_perm=128, seed=3, scheme=scheme, gpu_mode="always")
                m.update_batch(data)
                results[scheme] = m.hashvalues
            return results
        finally:
            _mh.cp, _mh._GPU_OK_CACHE = saved  # type: ignore[attr-defined]

    def test_gpu_branch_matches_cpu_all_schemes(self) -> None:
        gpu = self._run_with_shim(self._shim(), True)
        data = _make_data(500)
        for scheme in SCHEMES:
            m_cpu = MinHash(num_perm=128, seed=3, scheme=scheme, gpu_mode="disable")
            m_cpu.update_batch(data)
            self.assertEqual(m_cpu.hashvalues.dtype, gpu[scheme].dtype, scheme)
            self.assertTrue(np.array_equal(m_cpu.hashvalues, gpu[scheme]), scheme)

    def test_shim_catches_wrong_cache_dtype(self) -> None:
        # Mutation guard: forcing uint64 caches for affine32 changes the
        # pre-truncation minimum ordering, so the result must differ from CPU.
        bad = self._shim()
        bad.uint32 = np.uint64  # affine32 caches wrongly built as uint64
        gpu = self._run_with_shim(bad, True)
        m_cpu = MinHash(num_perm=128, seed=3, scheme="affine32", gpu_mode="disable")
        m_cpu.update_batch(_make_data(500))
        self.assertFalse(np.array_equal(m_cpu.hashvalues, gpu["affine32"]))
