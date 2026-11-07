"""pyroid.datasketch.MinHashLSHEnsemble vs live datasketch 2.0.0: (b, r), partitions, queries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import pyroid.datasketch as pd
from pyroid.datasketch import MinHash
from pyroid.datasketch.lshensemble import (
    MinHashLSHEnsemble,
    _false_negative_probability,
    _false_positive_probability,
    _optimal_param,
)

ds = pytest.importorskip("datasketch")
ds_ens = pytest.importorskip("datasketch.lshensemble")


def _entries(mod: Any, seed: int, count: int, scheme: str) -> list[tuple[int, Any, int]]:
    rng = np.random.RandomState(seed)
    out = []
    for key in range(count):
        lo = int(rng.randint(0, 50))
        size = int(rng.randint(1, 120))
        m = mod.MinHash(num_perm=128, scheme=scheme)
        m.update_batch([b"%d" % i for i in range(lo, lo + size)])
        out.append((key, m, size))
    return out


@pytest.mark.parametrize("threshold", [0.1, 0.5, 0.8, 0.9, 1.0])
@pytest.mark.parametrize("num_perm", [16, 128, 256])
def test_optimal_params_match(threshold: float, num_perm: int) -> None:
    xqs = np.exp(np.linspace(-5, 5, 10))
    for xq in xqs:
        for m in (2, 8):
            if m > num_perm:
                continue
            for w in ((0.5, 0.5), (0.2, 0.8)):
                want = ds_ens._optimal_param(threshold, num_perm, m, xq, *w)
                assert _optimal_param(threshold, num_perm, m, xq, *w) == want


@pytest.mark.parametrize(("b", "r"), [(1, 1), (4, 8), (32, 4), (128, 1)])
def test_probabilities_match(b: int, r: int) -> None:
    for t in (0.0, 0.3, 0.8, 1.0):
        for xq in (0.0067, 0.5, 1.0, 2.0, 148.4):
            assert _false_positive_probability(t, b, r, xq) == ds_ens._false_positive_probability(
                t, b, r, xq
            )
            assert _false_negative_probability(t, b, r, xq) == ds_ens._false_negative_probability(
                t, b, r, xq
            )


@pytest.mark.parametrize("scheme", ["affine32", "legacy"])
@pytest.mark.parametrize("threshold", [0.5, 0.8])
@pytest.mark.parametrize("num_part", [1, 4, 16])
def test_index_and_query_match(scheme: str, threshold: float, num_part: int) -> None:
    ref = ds.MinHashLSHEnsemble(threshold=threshold, num_perm=128, num_part=num_part)
    got = MinHashLSHEnsemble(threshold=threshold, num_perm=128, num_part=num_part)
    assert np.array_equal(got.params, ref.params)
    assert got.params.dtype == ref.params.dtype
    assert np.array_equal(got.xqs, ref.xqs)
    ref_e = _entries(ds, 7, 300, scheme)
    got_e = _entries(pd, 7, 300, scheme)
    ref.index(ref_e)
    got.index(got_e)
    assert got.lowers == ref.lowers
    assert got.uppers == ref.uppers
    for (key, rm, size), (_, gm, _) in zip(ref_e, got_e, strict=True):
        assert set(got.query(gm, size)) == set(ref.query(rm, size)), key
        assert (key in got) == (key in ref)
    assert got.is_empty() == ref.is_empty()


def test_errors_match() -> None:
    for kw in (
        {"threshold": 1.5},
        {"num_perm": 1},
        {"num_part": 0},
        {"m": 1},
        {"m": 200},
        {"weights": (0.6, 0.6)},
        {"weights": (-0.1, 1.1)},
    ):
        with pytest.raises(ValueError, match=r".") as want:
            ds.MinHashLSHEnsemble(**kw)
        with pytest.raises(ValueError, match=r".") as got:
            MinHashLSHEnsemble(**kw)
        assert str(got.value) == str(want.value)
    e = MinHashLSHEnsemble()
    with pytest.raises(ValueError, match="entries is empty"):
        e.index([])
    m = MinHash()
    with pytest.raises(ValueError, match="Set size must be positive"):
        e.index(iter([(1, m, 0)]))
