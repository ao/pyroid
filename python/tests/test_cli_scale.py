"""``pyroid`` at dataset scale: shards, workers, resumable runs, the four MinHash stages."""

from __future__ import annotations

import gzip
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("numpy")

import pyroid.dedup  # noqa: E402
import pyroid.translit  # noqa: E402
from pyroid.cli import main  # noqa: E402

DATA = Path(os.environ.get("PYROID_DATA", "/data"))
DEV = DATA / "fineweb" / "fineweb-dev-10k.parquet"
LID_BIN = DATA / "models" / "lid.176.bin"
DIGEST_SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "cli" / "digest.py"

WORDS = [
    "river",
    "stone",
    "light",
    "winter",
    "harvest",
    "valley",
    "market",
    "garden",
    "window",
    "letter",
    "morning",
    "city",
    "train",
    "music",
    "paper",
    "story",
    "table",
    "forest",
    "summer",
    "bridge",
]


def _table(texts: list[Any], start: int = 0) -> Any:
    return pa.table({"id": [f"d{start + i}" for i in range(len(texts))], "text": texts})


def _write(t: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        pq.write_table(t, path)
    else:
        opener: Any = gzip.open if path.suffix == ".gz" else open
        with opener(path, "wt", encoding="utf-8") as f:
            for r in t.to_pylist():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _rows(path: Path) -> list[dict[str, Any]]:
    """Every row under ``path`` (a file or a directory of outputs), in name order."""
    from pyroid.integrations import _io

    rows: list[dict[str, Any]] = []
    for src in _io.resolve_inputs([path]):
        for t in _io.iter_tables(src, 1 << 16):
            rows += t.to_pylist()
    return rows


def _corpus(n: int, seed: int = 0) -> list[str]:
    """Documents with near-duplicates (one word changed) spread far apart."""
    rng = random.Random(seed)
    docs = [" ".join(rng.choice(WORDS) for _ in range(40)) for _ in range(n)]
    for i in range(0, n, 7):
        j = rng.randrange(n)
        words = docs[i].split()
        words[rng.randrange(len(words))] = "changed"
        docs[j] = " ".join(words)
    docs[3] = "too short"
    return docs


def _shards(tmp: Path, docs: list[str], sizes: list[int], ext: str = ".parquet") -> Path:
    d = tmp / "in"
    pos = 0
    for i, n in enumerate(sizes):
        _write(_table(docs[pos : pos + n], pos), d / f"part-{i:03d}{ext}")
        pos += n
    assert pos == len(docs)
    return d


# ------------------------------------------------------------------------------------- layouts


def test_shard_spec() -> None:
    from pyroid.integrations._shards import Shard

    assert Shard.parse("2/5").take(list(range(12))) == [2, 7]
    assert Shard.parse("0/1").tag == "00000-of-00001"
    for bad in ("5/5", "-1/3", "1", "a/b", "1/0"):
        with pytest.raises(ValueError, match="--shard"):
            Shard.parse(bad)
    with pytest.raises(SystemExit):
        main(["translit", "in", "out", "--shard", "3/3"])


def test_directory_out_mirrors_inputs(tmp_path: Path) -> None:
    src = tmp_path / "in"
    _write(_table(["café", "naïve"]), src / "a" / "x.parquet")
    _write(_table(["Ελληνικά"]), src / "b" / "y.jsonl.gz")
    (src / "_SUCCESS").write_text("")
    out = tmp_path / "out"
    assert main(["translit", str(src), str(out)]) == 0
    names = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert [n for n in names if not n.startswith(".pyroid/")] == ["a/x.parquet", "b/y.jsonl.gz"]
    assert pq.read_table(out / "a" / "x.parquet").column("text").to_pylist() == ["cafe", "naive"]
    assert _rows(out / "b" / "y.jsonl.gz")[0]["text"] == pyroid.translit.anyascii("Ελληνικά")
    # --format converts every output
    out2 = tmp_path / "out2"
    assert main(["translit", str(src), str(out2), "--format", "jsonl.zst"]) == 0
    assert (out2 / "a" / "x.jsonl.zst").exists()
    assert (out2 / "b" / "y.jsonl.zst").exists()


def test_merge_and_shard_names(tmp_path: Path) -> None:
    docs = [f"doc {i} é" for i in range(10)]
    src = _shards(tmp_path, docs, [3, 3, 4])
    # a file OUT with --shard gets the shard in its name
    for i in range(2):
        assert main(["translit", str(src), str(tmp_path / "o.parquet"), "--shard", f"{i}/2"]) == 0
    a = pq.read_table(tmp_path / "o-00000-of-00002.parquet").column("id").to_pylist()
    b = pq.read_table(tmp_path / "o-00001-of-00002.parquet").column("id").to_pylist()
    assert a == [f"d{i}" for i in (0, 1, 2, 6, 7, 8, 9)]  # files 0 and 2
    assert b == ["d3", "d4", "d5"]
    assert not list(tmp_path.glob(".o*.pyroid"))  # a file OUT's state goes when it is done
    # --merge into a directory: one part per shard
    out = tmp_path / "merged"
    for i in range(2):
        assert main(["translit", str(src), str(out), "--merge", "--shard", f"{i}/2"]) == 0
    parts = sorted(p.name for p in out.glob("part-*"))
    assert parts == ["part-00000-of-00002.parquet", "part-00001-of-00002.parquet"]
    # an empty shard writes nothing
    assert main(["translit", str(src), str(tmp_path / "e.parquet"), "--shard", "4/5"]) == 0
    assert not list(tmp_path.glob("e*.parquet"))


def test_refuses_foreign_directory(tmp_path: Path) -> None:
    src = _write(_table(["x"]), tmp_path / "in.parquet")
    out = tmp_path / "out"
    out.mkdir()
    (out / "notes.txt").write_text("mine")
    with pytest.raises(SystemExit, match="not written by pyroid"):
        main(["translit", str(src), str(out)])
    assert main(["translit", str(src), str(out), "--overwrite"]) == 0
    assert (out / "notes.txt").read_text() == "mine"
    excl = out / "dropped"
    with pytest.raises(SystemExit, match="inside OUT"):
        main(["langid", str(src), str(out), "--excluded", str(excl)])


# ------------------------------------------------------------------------------------ resuming


@pytest.mark.thread_unsafe  # patches the translit step
def test_resume_after_a_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pyroid.cli as cli

    src = _shards(tmp_path, [f"doc {i}" for i in range(12)], [3, 3, 3, 3])
    out = tmp_path / "out"
    seen: list[str] = []
    fixed: list[bool] = []  # set below; not an env var: --iterations reuses the fixtures
    real = cli._STEPS["translit"]

    def flaky(t: Any, o: Any) -> Any:
        first = t.column("id")[0].as_py()
        seen.append(first)
        if first == "d6" and not fixed:
            raise ValueError("boom")
        return real(t, o)

    monkeypatch.setitem(cli._STEPS, "translit", flaky)
    assert main(["translit", str(src), str(out)]) == 1  # file 2 of 4 fails
    assert sorted(p.name for p in out.glob("*.parquet")) == ["part-000.parquet", "part-001.parquet"]
    assert not list(out.rglob("*.partial"))
    fixed.append(True)
    seen.clear()
    assert main(["translit", str(src), str(out), "--stats-json", str(tmp_path / "s.json")]) == 0
    assert seen == ["d6", "d9"]  # the finished files are skipped
    s = json.loads((tmp_path / "s.json").read_text())
    assert (s["docs_in"], s["docs_out"], s["files"]) == (12, 12, 4)
    seen.clear()
    assert main(["translit", str(src), str(out)]) == 0
    assert seen == []  # nothing left to do
    assert main(["translit", str(src), str(out), "--overwrite"]) == 0
    assert seen == ["d0", "d3", "d6", "d9"]
    # other options make another run: refuse to mix the two
    with pytest.raises(SystemExit, match="different run"):
        main(["translit", str(src), str(out), "--mode", "unidecode"])


def test_stats_merges_shards(tmp_path: Path) -> None:
    src = _shards(tmp_path, [f"doc {i}" for i in range(10)], [2, 3, 5])
    out = tmp_path / "out"
    for i in range(3):
        assert main(["translit", str(src), str(out), "--shard", f"{i}/3"]) == 0
    assert sorted(p.name for p in (out / ".pyroid").glob("stats-*.json")) == [
        f"stats-{i:05d}-of-00003.json" for i in range(3)
    ]
    j = tmp_path / "all.json"
    assert main(["stats", str(out), "--json", str(j)]) == 0
    s = json.loads(j.read_text())
    assert (s["docs_in"], s["files"], s["files_total"]) == (10, 3, 3)


@pytest.mark.thread_unsafe  # reads stderr
def test_progress_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = _shards(tmp_path, [f"doc {i}" for i in range(6)], [3, 3])
    assert main(["translit", str(src), str(tmp_path / "o.parquet"), "--progress"]) == 0
    err = capsys.readouterr().err
    assert "pyroid translit: 2/2 files, 6 docs" in err
    assert main(["translit", str(src), str(tmp_path / "q.parquet")]) == 0
    assert capsys.readouterr().err == ""  # quiet by default


# ----------------------------------------------------------------------------------- workers


@pytest.mark.thread_unsafe  # spawns processes
def test_workers_match_one_process(tmp_path: Path) -> None:
    docs = _corpus(300)
    src = _shards(tmp_path, docs, [50, 100, 30, 120])
    one, many = tmp_path / "one", tmp_path / "many"
    assert main(["translit", str(src), str(one)]) == 0
    assert main(["translit", str(src), str(many), "--workers", "3", "--threads", "1"]) == 0
    assert _rows(one) == _rows(many)
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    assert main(["dedup", str(src), str(a), "--batch-size", "17"]) == 0
    assert main(["dedup", str(src), str(b), "--workers", "2"]) == 0
    assert _rows(a) == _rows(b)


# ---------------------------------------------------------------------- MinHash over many files


def _expected(docs: list[str], **cfg: Any) -> list[str]:
    mask = pyroid.dedup.minhash_dedup(docs, **cfg)
    return [f"d{i}" for i, k in enumerate(mask) if k]


@pytest.mark.parametrize(
    "cfg",
    [
        {},
        {"hash_fc": "sha1", "num_buckets": 14, "hashes_per_bucket": 8},
        {
            "hash_fc": "xxhash",
            "precision": 32,
            "num_buckets": 4,
            "hashes_per_bucket": 3,
            "n_grams": 3,
            "seed": 7,
        },
    ],
)
def test_dedup_one_pass_matches_in_memory(tmp_path: Path, cfg: dict[str, Any]) -> None:
    docs = _corpus(400, seed=1)
    src = _shards(tmp_path, docs, [100, 0, 150, 150], ext=".jsonl.gz")
    out, excl = tmp_path / "out.parquet", tmp_path / "dups.parquet"
    flags = [f"--{k.replace('_', '-')}={v}" for k, v in cfg.items()]
    assert main(["dedup", str(src), str(out), "--excluded", str(excl), *flags]) == 0
    kept = [r["id"] for r in _rows(out)]
    assert kept == _expected(docs, **cfg)
    assert len(kept) < len(docs)
    assert {r["filter_reason"] for r in _rows(excl)} == {"minhash"}
    assert len(_rows(excl)) == len(docs) - len(kept)


def test_four_stages_match_in_memory(tmp_path: Path) -> None:
    docs = _corpus(500, seed=2)
    src = _shards(tmp_path, docs, [120, 80, 0, 200, 100])
    sigs, dups, rem, out = (tmp_path / n for n in ("sigs", "dups", "removed", "out"))
    cfg = ["--hash-fc", "sha1", "--num-buckets", "6", "--hashes-per-bucket", "4"]
    for i in range(2):
        assert main(["dedup", "signatures", str(src), str(sigs), "--shard", f"{i}/2", *cfg]) == 0
    assert sorted(p.name for p in (sigs / "bucket_000").iterdir()) == [
        f"{r:05d}.minhash.sig" for r in range(5)
    ]
    for i in range(3):  # 3 shards of 6 buckets: 2 buckets each
        assert main(["dedup", "buckets", str(sigs), str(dups), "--shard", f"{i}/3"]) == 0
    assert len(list(dups.glob("*.dups"))) == 6
    assert main(["dedup", "cluster", str(dups), str(rem)]) == 0
    for i in range(2):
        assert (
            main(
                ["dedup", "filter", str(src), str(out), "--removed", str(rem), "--shard", f"{i}/2"]
            )
            == 0
        )
    kept = [r["id"] for r in _rows(out)]
    assert kept == _expected(docs, hash_fc="sha1", num_buckets=6, hashes_per_bucket=4)
    # bucket hash ranges (12 shards over 6 buckets) write the same pairs, split in two
    dups2 = tmp_path / "dups2"
    for i in range(12):
        assert main(["dedup", "buckets", str(sigs), str(dups2), "--shard", f"{i}/12"]) == 0
    rem2 = tmp_path / "removed2"
    assert main(["dedup", "cluster", str(dups2), str(rem2)]) == 0
    assert sorted(p.name for p in rem.glob("*.remove")) == sorted(
        p.name for p in rem2.glob("*.remove")
    )
    for p in rem.glob("*.remove"):
        assert p.read_bytes() == (rem2 / p.name).read_bytes()


def test_stages_refuse_incomplete_or_mismatched_inputs(tmp_path: Path) -> None:
    docs = _corpus(60, seed=3)
    src = _shards(tmp_path, docs, [20, 20, 20])
    sigs, dups, rem = tmp_path / "sigs", tmp_path / "dups", tmp_path / "rem"
    assert main(["dedup", "signatures", str(src), str(sigs), "--shard", "0/2"]) == 0
    with pytest.raises(SystemExit, match="incomplete: 2 of 3"):
        main(["dedup", "buckets", str(sigs), str(dups)])
    assert main(["dedup", "signatures", str(src), str(sigs), "--shard", "1/2"]) == 0
    with pytest.raises(SystemExit, match="differs"):
        main(["dedup", "buckets", str(sigs), str(dups), "--num-buckets", "5"])
    assert main(["dedup", "buckets", str(sigs), str(dups), "--shard", "0/2"]) == 0
    with pytest.raises(SystemExit, match="incomplete"):
        main(["dedup", "cluster", str(dups), str(rem)])
    assert main(["dedup", "buckets", str(sigs), str(dups), "--shard", "1/2"]) == 0
    assert main(["dedup", "cluster", str(dups), str(rem)]) == 0
    _write(_table(["extra file"]), src / "part-999.parquet")
    with pytest.raises(SystemExit, match="not the ones"):
        main(["dedup", "filter", str(src), str(tmp_path / "o"), "--removed", str(rem)])
    with pytest.raises(SystemExit, match="staged flow"):
        main(["dedup", str(src), str(tmp_path / "o2"), "--shard", "0/2"])


def test_filter_cluster_columns(tmp_path: Path) -> None:
    docs = _corpus(80, seed=4)
    src = _shards(tmp_path, docs, [40, 40])
    sigs, dups, rem, out = (tmp_path / n for n in ("sigs", "dups", "removed", "out"))
    assert main(["dedup", "signatures", str(src), str(sigs)]) == 0
    assert main(["dedup", "buckets", str(sigs), str(dups)]) == 0
    assert (
        main(["dedup", "cluster", str(dups), str(rem), "--save-cluster-id", "--save-cluster-size"])
        == 0
    )
    args = ["dedup", "filter", str(src), str(out), "--removed", str(rem)]
    assert main([*args, "--load-cluster-ids", "--load-cluster-sizes"]) == 0
    index = pyroid.dedup.MinHashDedup()
    index.add(docs)
    keep, ids, sizes = index.keep_mask(), index.cluster_ids(), index.cluster_sizes()
    rows = _rows(out)
    assert [r["id"] for r in rows] == [f"d{i}" for i, k in enumerate(keep) if k]
    got_sizes = [r["minhash_cluster_size"] for r in rows]
    assert got_sizes == [s for s, k in zip(sizes, keep, strict=True) if k]
    # cluster ids are numbered in (file, doc) order, as datatrove does; the clusters are the same
    by_id: dict[int, set[str]] = {}
    for r in rows:
        if r["minhash_cluster_id"] >= 0:
            by_id.setdefault(r["minhash_cluster_id"], set()).add(r["id"])
    assert len(by_id) == sum(1 for i, k in zip(ids, keep, strict=True) if k and i >= 0)


# ------------------------------------------------------------------------------ remote inputs


@pytest.mark.thread_unsafe  # fsspec's memory filesystem is one per process
def test_fsspec_memory_inputs(tmp_path: Path) -> None:
    fsspec = pytest.importorskip("fsspec")
    fs = fsspec.filesystem("memory")
    docs = _corpus(90, seed=5)
    for i in range(3):
        with fs.open(f"/scale-test/data/{i:02d}.parquet", "wb") as f:
            pq.write_table(_table(docs[30 * i : 30 * (i + 1)], 30 * i), f)
    try:
        out = tmp_path / "out.parquet"
        assert main(["dedup", "memory://scale-test/data/*.parquet", str(out)]) == 0
        assert [r["id"] for r in _rows(out)] == _expected(docs)
        out2 = tmp_path / "t"
        assert main(["translit", "memory://scale-test/data", str(out2), "--workers", "1"]) == 0
        assert sorted(p.name for p in out2.glob("*.parquet")) == [
            "00.parquet",
            "01.parquet",
            "02.parquet",
        ]
    finally:
        fs.rm("/scale-test", recursive=True)


# --------------------------------------------------------------------- the FineWeb chain, sharded

ADV = DATA / "parity" / "adversarial.parquet"
#: datatrove's FineWeb chain on the parity corpus (dev + adversarial), as docs/compatibility.md
PARITY_DIGEST_PREFIX = "c0c6b746"


def _digest(input_path: Path, kept: Path, excluded: list[Path]) -> str:
    import importlib.util

    spec = importlib.util.spec_from_file_location("pyroid_cli_digest", DIGEST_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return str(mod.digest(str(input_path), str(kept), [str(p) for p in excluded]))


@pytest.mark.timeout(1800)
@pytest.mark.thread_unsafe
def test_fineweb_sharded_flows_are_digest_identical(tmp_path: Path) -> None:
    """The parity corpus (13,232 docs, 335 MinHash removals) in 5 files: worker processes, and
    shards plus the four MinHash stages, give the single-process digest, which is datatrove's."""
    if not (DEV.exists() and ADV.exists() and LID_BIN.exists()):
        pytest.skip("needs the dev slice, the adversarial set and lid.176.bin under $PYROID_DATA")
    t = pa.concat_tables([pq.read_table(p, columns=["id", "text"]) for p in (DEV, ADV)])
    whole = tmp_path / "parity.parquet"
    pq.write_table(t, whole)
    src = tmp_path / "parts"
    src.mkdir()
    step = -(-t.num_rows // 5)
    for i in range(5):
        pq.write_table(t.slice(i * step, step), src / f"parity-{i:02d}.parquet")
    lid = ["--recipe", "fineweb", "--lid-model", str(LID_BIN)]
    # one process, one file
    k1, x1 = tmp_path / "kept.parquet", tmp_path / "excl.parquet"
    assert main(["clean", str(whole), str(k1), *lid, "--excluded", str(x1)]) == 0
    d1 = _digest(whole, k1, [x1])
    assert d1.startswith(PARITY_DIGEST_PREFIX)
    assert sum(r["filter_reason"] == "minhash" for r in _rows(x1)) == 335
    # one machine, two worker processes, MinHash inside the run
    out, excl = tmp_path / "w-out", tmp_path / "w-excl"
    assert main(["clean", str(src), str(out), *lid, "--excluded", str(excl), "--workers", "2"]) == 0
    assert _digest(whole, out, [excl]) == d1
    # three "machines": clean without MinHash, then datatrove's four stages
    clean, cx = tmp_path / "clean", tmp_path / "clean-excl"
    for i in range(3):
        args = ["clean", str(src), str(clean), *lid, "--excluded", str(cx), "--no-dedup"]
        assert main([*args, "--shard", f"{i}/3"]) == 0
    cfg = ["--hash-fc", "sha1", "--precision", "64"]  # FineWeb's MinHash settings
    sigs, dups, rem = tmp_path / "sigs", tmp_path / "dups", tmp_path / "rem"
    for i in range(3):
        assert main(["dedup", "signatures", str(clean), str(sigs), "--shard", f"{i}/3", *cfg]) == 0
    for i in range(2):
        assert main(["dedup", "buckets", str(sigs), str(dups), "--shard", f"{i}/2"]) == 0
    assert main(["dedup", "cluster", str(dups), str(rem)]) == 0
    final, fx = tmp_path / "final", tmp_path / "final-excl"
    for i in range(3):
        args = ["dedup", "filter", str(clean), str(final), "--removed", str(rem)]
        assert main([*args, "--excluded", str(fx), "--shard", f"{i}/3"]) == 0
    assert _digest(whole, final, [cx, fx]) == d1


@pytest.mark.thread_unsafe  # spawns processes
def test_python_dash_m_with_workers(tmp_path: Path) -> None:
    """Spawned workers import ``pyroid.__main__`` again; it must not rerun the command."""
    import subprocess

    src = _shards(tmp_path, [f"café {i}" for i in range(8)], [2, 2, 2, 2])
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, "-m", "pyroid", "translit", str(src), str(out), "--workers", "2"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert [x["text"] for x in _rows(out)] == [f"cafe {i}" for i in range(8)]


# ------------------------------------------------------------------------- Windows file handling
# Windows refuses to rename or delete a file that is still open or mapped, where Linux and macOS
# allow it. `strict_fs` makes os.replace/rename/unlink/remove behave the Windows way on Linux
# (from /proc/self/fd and /proc/self/maps); on Windows itself the OS is the check.


def _held(path: str) -> list[str]:
    real = os.path.realpath(path)
    held = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.readlink(f"/proc/self/fd/{fd}") == real:
                held.append(f"fd {fd}")
        except OSError:
            continue
    with open("/proc/self/maps", encoding="utf-8", errors="replace") as f:
        held += [f"mapped at {ln.split()[0]}" for ln in f if ln.rstrip("\n").endswith(" " + real)]
    return held


@pytest.fixture
def strict_fs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The paths renamed or removed; raises PermissionError if one was still open or mapped."""
    if not Path("/proc/self/maps").exists():
        pytest.skip("needs /proc (Linux) to see open files and maps")
    touched: list[str] = []

    def strict(name: str) -> None:
        real = getattr(os, name)

        def wrapped(*args: Any, **kw: Any) -> Any:
            paths = [os.fspath(p) for p in args[:2] if isinstance(p, (str, os.PathLike))]
            if name in ("unlink", "remove"):
                paths = paths[:1]
            base = None
            if kw.get("dir_fd") is not None:
                base = os.readlink(f"/proc/self/fd/{kw['dir_fd']}")
            for p in paths:
                full = os.path.join(base, p) if base else p
                if os.path.exists(full):
                    held = _held(full)
                    if held:
                        raise PermissionError(f"{name}({full}): still {', '.join(held)}")
                    touched.append(full)
            return real(*args, **kw)

        monkeypatch.setattr(os, name, wrapped)

    for name in ("replace", "rename", "unlink", "remove"):
        strict(name)
    return touched


@pytest.mark.thread_unsafe  # patches os
def test_no_file_is_renamed_or_removed_while_open(tmp_path: Path, strict_fs: list[str]) -> None:
    docs = _corpus(200, seed=6)
    src = _shards(tmp_path, docs, [50, 50, 100], ext=".jsonl.gz")
    out, excl = tmp_path / "o.parquet", tmp_path / "x.parquet"
    assert main(["dedup", str(src), str(out), "--excluded", str(excl)]) == 0
    assert main(["translit", str(src), str(tmp_path / "t"), "--format", "parquet"]) == 0
    assert main(["translit", str(src), str(tmp_path / "m.jsonl.gz")]) == 0
    assert any(p.endswith(".partial") for p in strict_fs)  # the checks ran
    assert [r["id"] for r in _rows(out)] == _expected(docs)
    leftovers = [p for p in tmp_path.rglob("*") if p.name.endswith((".partial", ".tmp"))]
    assert leftovers == []


@pytest.mark.thread_unsafe  # patches os and the writer
def test_clean_resumes_after_a_failure_and_cleans_up(
    tmp_path: Path, strict_fs: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``clean`` with MinHash fails while writing OUT; the rerun finishes from the spools."""
    if not (DEV.exists() and LID_BIN.exists()):
        pytest.skip("needs the pinned dev slice and lid.176.bin under $PYROID_DATA")
    from pyroid.integrations import _io
    from pyroid.integrations.recipes import FineWeb

    sample = pq.read_table(DEV, columns=["text"]).slice(0, 40).column("text")
    recipe = FineWeb(lid_model=str(LID_BIN))
    good = sample.filter(recipe.annotate(sample).keep).to_pylist()[:6]
    cleaned = recipe.annotate(good).text.to_pylist()
    texts = [*good, good[0], "too short"]  # an exact copy for MinHash, a drop for the filters
    src = _shards(tmp_path, texts, [3, 5])
    out, excl = tmp_path / "out" / "kept.parquet", tmp_path / "out" / "dropped.parquet"
    args = ["clean", str(src), str(out), "--recipe", "fineweb", "--lid-model", str(LID_BIN)]
    args += ["--excluded", str(excl), "--batch-size", "3"]
    real = _io.Writer.write

    def write(self: Any, table: Any) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(_io.Writer, "write", write)
    with pytest.raises(RuntimeError, match="disk full"):
        main(args)
    state = out.parent / ".kept.parquet.pyroid"
    assert (state / "done" / "p1-000000.json").exists()  # the filter stage of both files is kept
    assert not out.exists()
    monkeypatch.setattr(_io.Writer, "write", real)
    assert main(args) == 0
    assert [r["text"] for r in _rows(out)] == cleaned
    reasons = {r["id"]: r["filter_reason"] for r in _rows(excl)}
    assert reasons["d6"] == "minhash"
    assert reasons["d7"].startswith("gopher")
    assert sorted(p.name for p in out.parent.iterdir()) == ["dropped.parquet", "kept.parquet"]
    assert any(p.endswith(".arrow") for p in strict_fs)  # the spools were removed after closing


# --------------------------------------------------------------------- resume edge cases (review)


def test_rerun_after_losing_an_output_still_dedups(tmp_path: Path) -> None:
    """The work directory is gone after a finished run; redoing one file must redo the MinHash
    decision, not skip it."""
    docs = _corpus(300, seed=7)
    src = _shards(tmp_path, docs, [100, 100, 100])
    out = tmp_path / "out"
    assert main(["dedup", str(src), str(out)]) == 0
    want = _expected(docs)
    assert [r["id"] for r in _rows(out)] == want
    (out / "part-001.parquet").unlink()
    assert main(["dedup", str(src), str(out)]) == 0
    assert [r["id"] for r in _rows(out)] == want


def test_rerun_with_other_outputs_redoes_them(tmp_path: Path) -> None:
    src = _shards(tmp_path, [f"doc {i}" for i in range(6)], [3, 3])
    out = tmp_path / "out"
    assert main(["translit", str(src), str(out)]) == 0
    assert main(["translit", str(src), str(out), "--format", "jsonl"]) == 0
    assert sorted(p.name for p in out.glob("*.jsonl")) == ["part-000.jsonl", "part-001.jsonl"]
    x1, x2 = tmp_path / "x1", tmp_path / "x2"
    args = ["langid", str(src), str(tmp_path / "l"), "--keep", "zz"]
    assert main([*args, "--excluded", str(x1)]) == 0
    assert main([*args, "--excluded", str(x2)]) == 0
    assert len(_rows(x2)) == 6


@pytest.mark.thread_unsafe  # patches the translit step
def test_records_survive_a_move(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Records hold paths relative to OUT and inputs by name: another mount point resumes."""
    import pyroid.cli as cli

    a = tmp_path / "a"
    src = _shards(a, [f"doc {i}" for i in range(6)], [3, 3])
    assert main(["translit", str(src), str(a / "out")]) == 0
    b = tmp_path / "b"
    a.rename(b)
    calls: list[int] = []
    real = cli._STEPS["translit"]

    def counted(t: Any, o: Any) -> Any:
        calls.append(1)
        return real(t, o)

    monkeypatch.setitem(cli._STEPS, "translit", counted)
    assert main(["translit", str(b / "in"), str(b / "out")]) == 0
    assert calls == []  # nothing redone


@pytest.mark.thread_unsafe  # patches the translit step
def test_finished_merge_is_not_redone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pyroid.cli as cli

    src = _shards(tmp_path, [f"doc {i}" for i in range(6)], [3, 3])
    out = tmp_path / "out"
    assert main(["translit", str(src), str(out), "--merge"]) == 0
    calls: list[int] = []
    real = cli._STEPS["translit"]

    def counted(t: Any, o: Any) -> Any:
        calls.append(1)
        return real(t, o)

    monkeypatch.setitem(cli._STEPS, "translit", counted)
    assert main(["translit", str(src), str(out), "--merge"]) == 0
    assert calls == []
    assert len(_rows(out / "part-00000-of-00001.parquet")) == 6


def test_empty_spool_and_starting_shards(tmp_path: Path) -> None:
    from pyroid.cli import _check_out, _Ipc, _read_ipc

    p = str(tmp_path / "s.arrow")
    _Ipc(p).close(None)  # an empty JSONL input: no rows, no schema
    assert os.path.getsize(p) == 0
    assert list(_read_ipc(p)) == []
    # another shard is creating OUT/.pyroid right now: OUT is still ours
    out = tmp_path / "out"
    (out / ".pyroid").mkdir(parents=True)
    a = pytest.importorskip("argparse").Namespace(output=str(out), excluded=None, merge=False)
    from pyroid.cli import _layout
    from pyroid.integrations._shards import Shard

    _check_out(_layout(a, [], Shard(), excluded=False), False, [])


@pytest.mark.thread_unsafe  # reads stderr
def test_stages_refuse_lost_or_unfinished_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    docs = _corpus(60, seed=8)
    src = _shards(tmp_path, docs, [30, 30])
    sigs, dups, rem = tmp_path / "sigs", tmp_path / "dups", tmp_path / "rem"
    assert main(["dedup", "signatures", str(src), str(sigs)]) == 0
    (sigs / "bucket_003" / "00001.minhash.sig").unlink()
    assert main(["dedup", "buckets", str(sigs), str(dups)]) == 1
    assert "1 signature files for 2 input files" in capsys.readouterr().err
    # a cluster stage that did not finish: REMOVED has .pyroid but no run.json
    (rem / ".pyroid").mkdir(parents=True)
    with pytest.raises(SystemExit, match="incomplete"):
        main(["dedup", "filter", str(src), str(tmp_path / "o"), "--removed", str(rem)])
    # a folder with no lists at all (datatrove's, when nothing is a duplicate) warns
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["dedup", "filter", str(src), str(tmp_path / "o2"), "--removed", str(empty)]) == 0
    assert "no .remove files" in capsys.readouterr().err
