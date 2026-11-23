"""Indexes pickled by datasketch load into working pyroid indexes.

Under ``pyroid.patch("datasketch")``, unpickling a datasketch ``MinHashLSH``,
``MinHashLSHForest`` or ``MinHashLSHEnsemble`` builds pyroid's class from datasketch's
attributes. The unpickler below maps the classes as the patch does, in this process; one
test runs the real patch in a child interpreter.
"""

from __future__ import annotations

import io
import pickle
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import pyroid
import pyroid.datasketch as pds

ds = pytest.importorskip("datasketch")
np = pytest.importorskip("numpy")

RunPython = Callable[..., subprocess.CompletedProcess[str]]


class _AsPatched(pickle.Unpickler):
    """Resolve ``datasketch.*`` to ``pyroid.datasketch.*``, as ``pyroid.patch`` does."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "datasketch" or module.startswith("datasketch."):
            target = "pyroid.datasketch" + module[len("datasketch") :]
            mod = __import__(target, fromlist=[name])
            if hasattr(mod, name):
                return getattr(mod, name)
        return super().find_class(module, name)


def _load(obj: Any) -> Any:
    return _AsPatched(io.BytesIO(pickle.dumps(obj))).load()


def _minhashes(n: int, num_perm: int, seed: int = 1) -> list[Any]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        m = ds.MinHash(num_perm=num_perm)
        words = rng.integers(0, 60, size=12)
        m.update_batch([f"w{w}".encode() for w in words] + [f"d{i % 7}".encode()])
        out.append(m)
    return out


@pytest.mark.parametrize("prepickle", [False, True])
def test_minhashlsh(prepickle: bool) -> None:
    mhs = _minhashes(200, 64)
    lsh = ds.MinHashLSH(threshold=0.5, num_perm=64, prepickle=prepickle)
    for i, m in enumerate(mhs[:150]):
        lsh.insert(("doc", i) if i % 3 else f"k{i}", m)
    lsh.insert("k0", mhs[1], check_duplication=False)  # a key with two runs of band keys
    lsh.remove(("doc", 4))
    got = _load(lsh)
    assert type(got) is pds.MinHashLSH
    for m in mhs:
        assert sorted(map(repr, got.query(m))) == sorted(map(repr, lsh.query(m)))
    assert ("doc", 5) in got
    assert ("doc", 4) not in got
    assert "k0" in got
    assert got.get_counts() == lsh.get_counts()
    got.insert("new", mhs[180])
    assert "new" in got.query(mhs[180])
    got.remove("k0")
    assert "k0" not in got
    # pyroid's own pickles still round-trip.
    again = pickle.loads(pickle.dumps(got))
    assert sorted(map(repr, again.query(mhs[7]))) == sorted(map(repr, got.query(mhs[7])))


def test_minhashlsh_with_params_and_weights() -> None:
    mhs = _minhashes(60, 128, seed=3)
    lsh = ds.MinHashLSH(num_perm=128, params=(16, 4))
    for i, m in enumerate(mhs):
        lsh.insert(i, m)
    got = _load(lsh)
    assert (got.b, got.r, got.h) == (16, 4, 128)
    for m in _minhashes(60, 128, seed=4) + mhs:
        assert sorted(got.query(m)) == sorted(lsh.query(m))


@pytest.mark.parametrize("indexed", ["all", "none", "partly"])
def test_forest(indexed: str) -> None:
    mhs = _minhashes(120, 64, seed=5)
    forest = ds.MinHashLSHForest(num_perm=64, l=8)
    for i, m in enumerate(mhs[:80]):
        forest.add(i, m)
    if indexed != "none":
        forest.index()
    if indexed == "partly":
        for i, m in enumerate(mhs[80:], 80):
            forest.add(i, m)  # added after index(): found only through indexed band keys
    got = _load(forest)
    assert type(got) is pds.MinHashLSHForest
    assert [list(t) for t in got.sorted_hashtables] == forest.sorted_hashtables
    # Results come "in no particular order", as datasketch's; the sets are the same.
    for m in mhs:
        for k in (1, 5, 30):
            assert sorted(got.query(m, k), key=repr) == sorted(forest.query(m, k), key=repr)
    got.index()
    forest.index()
    for m in mhs[::7]:
        assert sorted(got.query(m, 10), key=repr) == sorted(forest.query(m, 10), key=repr)


def test_ensemble() -> None:
    mhs = _minhashes(100, 64, seed=7)
    ens = ds.MinHashLSHEnsemble(threshold=0.6, num_perm=64, num_part=4)
    ens.index([(i, m, 10 + i % 40) for i, m in enumerate(mhs)])
    got = _load(ens)
    assert type(got) is pds.MinHashLSHEnsemble
    for m in mhs:
        for size in (5, 20, 60):
            assert sorted(got.query(m, size), key=repr) == sorted(ens.query(m, size), key=repr)


def test_unported_storage_fails_at_load() -> None:
    lsh = ds.MinHashLSH(threshold=0.5, num_perm=16, hashfunc=lambda b: b[:4])
    state = lsh.__dict__.copy()
    obj = pds.MinHashLSH.__new__(pds.MinHashLSH)
    with pytest.raises(pyroid.PyroidError, match=r"without pyroid\.patch"):
        obj.__setstate__(state)


@pytest.mark.thread_unsafe(reason="runs a subprocess")
def test_under_the_real_patch(run_python: RunPython, tmp_path: Path) -> None:
    path = tmp_path / "lsh.pkl"
    mhs = _minhashes(30, 64, seed=9)
    lsh = ds.MinHashLSH(threshold=0.5, num_perm=64)
    forest = ds.MinHashLSHForest(num_perm=64)
    ens = ds.MinHashLSHEnsemble(threshold=0.8, num_perm=64, num_part=2)
    for i, m in enumerate(mhs):
        lsh.insert(i, m)
        forest.add(i, m)
    forest.index()
    ens.index([(i, m, 5 + i) for i, m in enumerate(mhs)])
    want = (lsh.query(mhs[3]), sorted(forest.query(mhs[3], 5)), sorted(ens.query(mhs[3], 8)))
    path.write_bytes(pickle.dumps((mhs[3], lsh, forest, ens)))
    proc = run_python(
        f"""
        import pickle, warnings
        warnings.simplefilter("ignore")
        import pyroid
        pyroid.patch("datasketch")
        m, lsh, forest, ens = pickle.loads(open({str(path)!r}, "rb").read())
        print(type(lsh).__module__, type(forest).__module__, type(ens).__module__)
        print(sorted(lsh.query(m)), sorted(forest.query(m, 5)), sorted(ens.query(m, 8)))
        """
    )
    assert proc.returncode == 0, proc.stderr
    mods, results = proc.stdout.splitlines()
    assert mods.split() == [
        "pyroid.datasketch.lsh",
        "pyroid.datasketch.lshforest",
        "pyroid.datasketch.lshensemble",
    ]
    assert results == f"{sorted(want[0])} {want[1]} {want[2]}"
