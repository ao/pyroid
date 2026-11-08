"""pyroid's on-disk MinHash kernels against live datatrove 0.10.1, file for file.

Each case writes the same ranks of documents through datatrove's ``MinhashDedupSignature``,
``MinhashDedupBuckets`` (one worker per bucket, then two, which splits each bucket's hash range)
and ``MinhashDedupCluster`` (with cluster ids and sizes), and through pyroid's ``_SigFileWriter``,
``_minhash_hash_range``, ``_minhash_buckets`` and ``_minhash_cluster``, then compares every file
byte for byte. The datatrove cases skip where datatrove or spaCy is missing; the rest of the file
needs only pyroid (and pyarrow for the Arrow inputs).

Run in the oracle env with the pyroid wheel installed, e.g.::

    python -m pytest python/tests/test_dedup_files.py
"""

from __future__ import annotations

import os
import random
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import pyroid
import pyroid.dedup as dedup
from pyroid._pyroid import dedup as native

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# 88 made-up words: two syllables each.
VOCAB = [
    a + b
    for a in ("ka", "lo", "mi", "ne", "ru", "sa", "to", "vi")
    for b in ("ber", "dan", "fel", "gor", "lin", "mor", "pex", "tus", "wen", "zor", "qua")
]


@dataclass(frozen=True)
class Case:
    """One configuration, and the knobs that force pyroid's out-of-core paths."""

    id: str
    hash_fc: str
    precision: int
    num_buckets: int
    hashes_per_bucket: int
    n_grams: int
    seed: int
    memory_budget: int  # stage 1; 1 spills every record
    max_open: int  # stage 2; 2 merges in passes
    batches: tuple[int, ...]  # sizes of the add() batches, cycled


CASES = [
    Case("fineweb-sha1-64", "sha1", 64, 14, 8, 5, 1, 256 * 2**20, 512, (3, 1, 8)),
    Case("xxhash-32-seed7", "xxhash", 32, 4, 3, 3, 7, 1, 2, (1, 7, 2)),
    Case("one-hash-per-bucket", "xxhash", 64, 9, 1, 2, 3, 4096, 2, (5,)),
    Case("sha1-32", "sha1", 32, 5, 2, 4, 11, 700, 3, (2, 9, 4)),
]


def corpus(seed: int, n_ranks: int = 5, empty_rank: int = 2) -> list[list[str | None]]:
    """Return ranks of near-duplicate families plus unique, short and empty documents.

    ``None`` is an Arrow null for pyroid and ``""`` for datatrove (neither has a signature).
    Rank ``empty_rank`` has no documents; the ranks are contiguous slices of one shuffled list.
    """
    rng = random.Random(seed)
    docs: list[str | None] = []
    for _ in range(8):
        base = [rng.choice(VOCAB) for _ in range(rng.randint(25, 60))]
        docs.append(" ".join(base))
        for _ in range(rng.randint(1, 5)):
            words = list(base)
            for _ in range(rng.randint(0, 2)):
                i = rng.randrange(len(words))
                op = rng.random()
                if op < 0.4:
                    words[i] = rng.choice(VOCAB)
                elif op < 0.7:
                    words[i] = words[i].upper()
                else:
                    words[i] += rng.choice([",", ".", "!", ";"])
            docs.append(" ".join(words))
    docs += [" ".join(rng.choice(VOCAB) for _ in range(rng.randint(6, 40))) for _ in range(15)]
    docs += ["one", "two words", "three short words", "", "four words in total", None, None]
    rng.shuffle(docs)
    # Rank 0 keeps at least a few documents with signatures: datatrove splits each bucket's
    # hash range at records of the first file.
    while sum(1 for d in docs[:6] if d and len(d.split()) >= 6) < 3:
        rng.shuffle(docs)
    cuts = sorted(rng.sample(range(6, len(docs)), n_ranks - 2))
    parts = [docs[a:b] for a, b in zip([0, *cuts], [*cuts, len(docs)], strict=True)]
    parts.insert(empty_rank, [])
    return parts


def tree(root: Path) -> dict[str, bytes]:
    """Return every file under ``root`` (hidden ones too) by relative path."""
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def pyroid_config(case: Case) -> dedup.MinhashConfig:
    return dedup.MinhashConfig(
        n_grams=case.n_grams,
        num_buckets=case.num_buckets,
        hashes_per_bucket=case.hashes_per_bucket,
        seed=case.seed,
        hash_config=dedup.HashConfig(case.precision, case.hash_fc),
    )


def batches(docs: list[str | None], sizes: tuple[int, ...]) -> list[list[str | None]]:
    out, i, k = [], 0, 0
    while i < len(docs):
        n = sizes[k % len(sizes)]
        out.append(docs[i : i + n])
        i, k = i + n, k + 1
    return out


def as_input(batch: list[str | None], k: int, pa: Any) -> Any:
    """Return batch ``k`` as pyroid input: a list of str, or Arrow (always if it has a null)."""
    if None not in batch and k % 3 == 0:
        return [t or "" for t in batch]
    if k % 3 == 1:
        return pa.chunked_array(
            [pa.array(batch[:1], pa.string()), pa.array(batch[1:], pa.string())]
        )
    return pa.array(batch, pa.large_string() if k % 2 else pa.string())


def write_ranks(
    ranks: list[list[str | None]],
    ncfg: Any,
    folder: Path,
    pa: Any,
    *,
    memory_budget: int = 256 * 2**20,
    sizes: tuple[int, ...] = (7,),
) -> None:
    """Run stage 1 for every rank through ``_SigFileWriter``."""
    for r, docs in enumerate(ranks):
        w = native._SigFileWriter(str(folder), r, ncfg, memory_budget=memory_budget)
        expected = 0
        for k, batch in enumerate(batches(docs, sizes)):
            assert w.add(as_input(batch, k, pa)) == (expected, expected + len(batch))
            expected += len(batch)
            assert w.documents == expected
        if memory_budget == 1 and any(d and len(d.split()) >= 6 for d in docs):
            # Every record was spilled as a run under .pyroid-tmp, nothing written yet.
            assert list((folder / ".pyroid-tmp").rglob("*.run"))
            assert not list(folder.glob(f"bucket_*/{r:05d}.minhash.sig"))
        assert w.finish() == len(docs)


def run_buckets(
    sig_dir: Path, out: Path, ncfg: Any, num_buckets: int, workers: int, max_open: int
) -> int:
    """Run stage 2 for every bucket worker; return the number of pairs."""
    pairs = 0
    for b in range(num_buckets):
        bucket = sig_dir / f"bucket_{b:03d}"
        files = [str(bucket / n) for n in sorted(os.listdir(bucket))]
        for w in range(workers):
            lo, hi = native._minhash_hash_range(files[0] if files else None, ncfg, w, workers)
            pairs += native._minhash_buckets(
                files,
                str(out / f"{b:05d}_{w:02d}.dups"),
                ncfg,
                hash_min=lo,
                hash_max=hi,
                memory_budget=0 if max_open < 4 else 64 * 2**20,
                max_open=max_open,
            )
    return pairs


@pytest.fixture(scope="module")
def dt() -> Any:
    """Return datatrove's minhash module; skip without it or spaCy."""
    pytest.importorskip("spacy")
    return pytest.importorskip("datatrove.pipeline.dedup.minhash")


def dt_run(dt: Any, case: Case, ranks: list[list[str | None]], root: Path) -> None:
    """Run datatrove's stages 1-3 into ``root/{sigs,dups1,clusters1,dups2,clusters2}``."""
    document: Any = pytest.importorskip("datatrove.data").Document
    hashing: Any = pytest.importorskip("datatrove.utils.hashing")
    cfg = dt.MinhashConfig(
        n_grams=case.n_grams,
        num_buckets=case.num_buckets,
        hashes_per_bucket=case.hashes_per_bucket,
        seed=case.seed,
        hash_config=hashing.HashConfig(hash_fc=case.hash_fc, precision=case.precision),
    )
    sig = dt.MinhashDedupSignature(output_folder=str(root / "sigs"), config=cfg)
    for r, docs in enumerate(ranks):
        data = (document(text=t or "", id=f"{r}/{i}") for i, t in enumerate(docs))
        sig.run(data, rank=r, world_size=len(ranks))
    for workers in (1, 2):
        world = case.num_buckets * workers
        for rank in range(world):
            dt.MinhashDedupBuckets(
                input_folder=str(root / "sigs"),
                output_folder=str(root / f"dups{workers}"),
                config=cfg,
            ).run(None, rank=rank, world_size=world)
        dt.MinhashDedupCluster(
            input_folder=str(root / f"dups{workers}"),
            output_folder=str(root / f"clusters{workers}"),
            config=cfg,
            save_cluster_id=True,
            save_cluster_size=True,
        ).run(None, 0, 1)


def per_doc(path: Path, ext: str, offsets: list[int], n: int, default: int) -> list[int]:
    """Return the per-document values of the ``.clusters`` or ``.sizes`` files, globally."""
    out = [default] * n
    for f in path.glob(f"*.{ext}"):
        base = offsets[int(f.name.split(".")[0])]
        for doc, value in struct.iter_unpack("<2I", f.read_bytes()):
            out[base + doc] = value
    return out


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_files_match_live_datatrove(dt: Any, case: Case, tmp_path: Path, pa: Any) -> None:
    ranks = corpus(case.seed)
    want, got = tmp_path / "datatrove", tmp_path / "pyroid"
    dt_run(dt, case, ranks, want)

    pcfg = pyroid_config(case)
    ncfg = dedup._native_config(pcfg)
    write_ranks(ranks, ncfg, got / "sigs", pa, memory_budget=case.memory_budget, sizes=case.batches)
    assert tree(got / "sigs") == tree(want / "sigs")
    assert len(tree(want / "sigs")) == case.num_buckets * len(ranks)
    assert tree(want / "sigs")["bucket_000/00002.minhash.sig"] == b""  # the empty rank

    flat = [t for docs in ranks for t in docs]
    offsets = [sum(len(d) for d in ranks[:r]) for r in range(len(ranks))]
    mem = dedup.MinHashDedup(pcfg)
    mem.add(pa.array(flat, pa.string()))
    for workers in (1, 2):
        dups, clusters = got / f"dups{workers}", got / f"clusters{workers}"
        pairs = run_buckets(got / "sigs", dups, ncfg, case.num_buckets, workers, case.max_open)
        assert tree(dups) == tree(want / f"dups{workers}"), workers
        assert pairs == sum(len(b) for b in tree(dups).values()) // 16
        assert pairs > 0
        names = sorted(str(p) for p in dups.glob("*.dups"))
        summary = native._minhash_cluster(
            names, str(clusters), save_cluster_id=True, save_cluster_size=True
        )
        assert tree(clusters) == tree(want / f"clusters{workers}"), workers
        removed = sorted(
            offsets[int(f.name.split(".")[0])] + d
            for f in clusters.glob("*.remove")
            for (d,) in struct.iter_unpack("<I", f.read_bytes())
        )
        assert summary["removed"] == len(removed) > 0
        assert summary["files"] == len(list(clusters.glob("*.remove")))
        assert summary["clusters"] == summary["nodes"] - summary["removed"]
        # The ranks are contiguous slices of one corpus: the in-memory index decides the same.
        assert removed == mem.duplicates()
        assert per_doc(clusters, "clusters", offsets, len(flat), -1) == mem.cluster_ids()
        assert per_doc(clusters, "sizes", offsets, len(flat), 1) == mem.cluster_sizes()

    # datatrove's stage 4 on pyroid's files keeps what the in-memory index keeps.
    document: Any = pytest.importorskip("datatrove.data").Document
    keep = mem.keep_mask()
    filt = dt.MinhashDedupFilter(
        input_folder=str(got / "clusters1"), load_cluster_ids=True, load_cluster_sizes=True
    )
    for r, docs in enumerate(ranks):
        data = (document(text=t or "", id=str(offsets[r] + i)) for i, t in enumerate(docs))
        kept = [int(d.id) for d in filt.run(data, rank=r, world_size=len(ranks))]
        assert kept == [offsets[r] + i for i in range(len(docs)) if keep[offsets[r] + i]]


def test_spills_and_hierarchical_merge_change_nothing(tmp_path: Path, pa: Any) -> None:
    """Tiny budgets and max_open=2 give the same bytes as one in-memory sort and a flat merge."""
    case = CASES[1]
    ranks = corpus(5, n_ranks=6, empty_rank=4)
    ncfg = dedup._native_config(pyroid_config(case))
    trees = []
    for k, (budget, max_open, sizes) in enumerate(
        [(256 * 2**20, 512, (1000,)), (1, 2, (1, 2, 3)), (300, 3, (4,)), (5000, 2, (13,))]
    ):
        root = tmp_path / str(k)
        write_ranks(ranks, ncfg, root / "sigs", pa, memory_budget=budget, sizes=sizes)
        for workers in (1, 3):
            run_buckets(
                root / "sigs", root / f"d{workers}", ncfg, case.num_buckets, workers, max_open
            )
        trees.append(tree(root))
    assert all(t == trees[0] for t in trees[1:])
    # No temporary file is left behind.
    assert not [n for n in trees[0] if any(part.startswith(".") for part in n.split("/"))]


def test_writers_on_threads(tmp_path: Path, pa: Any) -> None:
    """Several writers on several Python threads write what one thread writes."""
    ranks = corpus(9, n_ranks=6, empty_rank=1)
    ncfg = dedup._native_config(pyroid_config(CASES[3]))
    write_ranks(ranks, ncfg, tmp_path / "serial", pa, memory_budget=900, sizes=(3,))

    def one(r: int) -> None:
        w = native._SigFileWriter(str(tmp_path / "threads"), r, ncfg, memory_budget=900, threads=2)
        for k, batch in enumerate(batches(ranks[r], (3,))):
            w.add(as_input(batch, k, pa))
        w.finish()

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, range(len(ranks))))
    assert tree(tmp_path / "threads") == tree(tmp_path / "serial")


def test_writer_lifecycle(tmp_path: Path) -> None:
    ncfg = dedup._native_config(pyroid_config(CASES[2]))
    w = native._SigFileWriter(str(tmp_path), 3, ncfg, memory_budget=1)
    assert w.add("a single document is one text with enough words") == (0, 1)
    assert w.add(["x", "another document with several more words"]) == (1, 3)
    assert w.documents == 3
    with pytest.raises(pyroid.InputError):
        w.add([1, 2])
    assert w.documents == 3
    w.abort()
    w.abort()
    assert tree(tmp_path) == {}
    with pytest.raises(ValueError, match="the writer was aborted"):
        w.add(["more"])
    with pytest.raises(ValueError, match="the writer was aborted"):
        w.finish()

    w = native._SigFileWriter(str(tmp_path), 0, ncfg)
    w.add(["some words to hash here"])
    assert w.finish() == 1
    with pytest.raises(ValueError, match=r"finish\(\) was already called"):
        w.finish()
    w.abort()  # does nothing once finished
    files = tree(tmp_path)
    assert sorted(files) == [f"bucket_{b:03d}/00000.minhash.sig" for b in range(9)]
    assert all(len(v) == 12 for v in files.values())  # one <QI record each

    # Dropping an unfinished writer writes nothing either.
    w = native._SigFileWriter(str(tmp_path / "dropped"), 0, ncfg, memory_budget=1)
    w.add(["dropped before finish with many words"])
    del w
    assert tree(tmp_path / "dropped") == {}


def test_malformed_and_missing_files(tmp_path: Path) -> None:
    ncfg = dedup._native_config(pyroid_config(CASES[0]))  # 8 x u64 + u32: 68-byte records
    bad = tmp_path / "00001.minhash.sig"
    bad.write_bytes(bytes(70))
    out = str(tmp_path / "o" / "00000_00.dups")
    with pytest.raises(ValueError, match="70 bytes is not a whole number of 68-byte records"):
        native._minhash_buckets([str(bad)], out, ncfg)
    with pytest.raises(ValueError, match="70 bytes is not a whole number"):
        native._minhash_hash_range(str(bad), ncfg, 0, 2)
    rec = struct.pack("<8QI", *range(8, 0, -1), 0) + struct.pack("<8QI", *range(8), 1)
    bad.write_bytes(rec)
    with pytest.raises(ValueError, match="record 1 is out of order"):
        native._minhash_buckets([str(bad)], out, ncfg)
    with pytest.raises(ValueError, match="only 2 lines"):
        native._minhash_hash_range(str(bad), ncfg, 0, 3)
    with pytest.raises(ValueError, match="out of range"):
        native._minhash_hash_range(str(bad), ncfg, 2, 2)
    with pytest.raises(ValueError, match="at least 1"):
        native._minhash_hash_range(None, ncfg, 0, 0)
    assert native._minhash_hash_range(None, ncfg, 1, 2) == (0, 2**61 - 1)
    with pytest.raises(ValueError, match="max_open"):
        native._minhash_buckets([], out, ncfg, max_open=1)
    missing = tmp_path / "00009.minhash.sig"
    with pytest.raises(FileNotFoundError) as info:
        native._minhash_buckets([str(missing)], out, ncfg)
    assert info.value.filename == str(missing)
    assert native._minhash_buckets([], out, ncfg) == 0
    assert Path(out).read_bytes() == b""

    dups = tmp_path / "00000_00.dups"
    dups.write_bytes(bytes(20))
    with pytest.raises(ValueError, match="20 bytes is not a whole number of 16-byte records"):
        native._minhash_cluster([str(dups)], str(tmp_path / "c"))
    with pytest.raises(FileNotFoundError):
        native._minhash_cluster([str(tmp_path / "nope.dups")], str(tmp_path / "c"))
    dups.write_bytes(
        struct.pack("<4I", 0, 1, 2, 3) + struct.pack("<4I", 2**32 - 1, 2**32 - 1, 0, 1)
    )
    assert native._minhash_cluster([str(dups)], str(tmp_path / "c")) == {
        "nodes": 3,
        "removed": 2,
        "clusters": 1,
        "files": 2,
    }
    # The index match (SENTINEL, SENTINEL) became the root of both documents.
    assert (tmp_path / "c" / "000000.remove").read_bytes() == struct.pack("<I", 1)
    assert (tmp_path / "c" / "000002.remove").read_bytes() == struct.pack("<I", 3)
    assert native._minhash_cluster(
        [str(dups)], str(tmp_path / "c2"), ignore_index_matches=True
    ) == {"nodes": 2, "removed": 1, "clusters": 1, "files": 1}
    assert not list(tmp_path.rglob(".*"))
