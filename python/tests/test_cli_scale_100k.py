"""The FineWeb chain over the 100k headline slice split into 16 files, run as on many machines.

Two checks (``PYROID_TEST_SCALE=1``; a few minutes on a release build):

1. ``pyroid clean --no-dedup --shard I/4`` + ``pyroid dedup signatures|buckets|cluster|filter``
   give the per-document digest of datatrove's own FineWeb chain on this slice (the benchmark's
   pinned digest, ``bench/results/aws-c7i.4xlarge/2026-11-06/headline/e2e/equivalence.json``).
2. datatrove 0.10.1's ``MinhashDedupSignature`` (16 tasks, one per file),
   ``MinhashDedupBuckets`` (14 tasks) and ``MinhashDedupCluster`` on the same cleaned files write
   byte-identical ``.minhash.sig``, ``.dups`` and ``.remove`` files.
"""

from __future__ import annotations

import concurrent.futures as cf
import importlib.util
import multiprocessing
import os
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("numpy")

from pyroid.cli import main  # noqa: E402

pytestmark = [
    pytest.mark.thread_unsafe,
    pytest.mark.timeout(3600),
    pytest.mark.skipif(not os.environ.get("PYROID_TEST_SCALE"), reason="set PYROID_TEST_SCALE=1"),
]

DATA = Path(os.environ.get("PYROID_DATA", "/data"))
HEADLINE = DATA / "fineweb" / "fineweb-100k.parquet"
LID_BIN = DATA / "models" / "lid.176.bin"
DIGEST_SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "cli" / "digest.py"
#: datatrove 0.10.1's FineWeb chain on the 100k slice (bench/pipeline.py, 8 workers).
DATATROVE_DIGEST = "da8979997cd283aa235e0df2f58853f1f34b5f31ff225f6076d5645f1087bed1"
FILES = 16


def _digest(kept: Path, excluded: list[Path]) -> str:
    spec = importlib.util.spec_from_file_location("pyroid_cli_digest", DIGEST_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return str(mod.digest(str(HEADLINE), str(kept), [str(p) for p in excluded]))


def _dt(name: str) -> Any:
    import importlib

    return importlib.import_module(name)


def _dt_config() -> Any:
    cfg, hashing = _dt("datatrove.pipeline.dedup.minhash"), _dt("datatrove.utils.hashing")
    return cfg.MinhashConfig(hash_config=hashing.HashConfig(precision=64, hash_fc="sha1"))


def _dt_signatures(args: tuple[str, str, int]) -> None:
    """datatrove's stage 1 for one rank (in a worker process)."""
    document = _dt("datatrove.data").Document
    signature = _dt("datatrove.pipeline.dedup.minhash").MinhashDedupSignature

    path, out, rank = args
    t = pq.read_table(path, columns=["id", "text"]).to_pydict()
    docs = (document(text=x, id=i) for i, x in zip(t["id"], t["text"], strict=True))
    signature(output_folder=out, config=_dt_config()).run(docs, rank, FILES)


def _same_tree(a: Path, b: Path, pattern: str) -> int:
    names_a = sorted(p.relative_to(a).as_posix() for p in a.glob(pattern))
    names_b = sorted(p.relative_to(b).as_posix() for p in b.glob(pattern))
    assert names_a == names_b, (names_a[:5], names_b[:5])
    for n in names_a:
        assert (a / n).read_bytes() == (b / n).read_bytes(), n
    return len(names_a)


def test_100k_in_16_files(tmp_path: Path) -> None:
    pytest.importorskip("datatrove")
    pytest.importorskip("spacy")
    if not (HEADLINE.exists() and LID_BIN.exists()):
        pytest.skip("needs the 100k headline slice and lid.176.bin under $PYROID_DATA")
    t = pq.read_table(HEADLINE)
    src = tmp_path / "raw"
    src.mkdir()
    step = -(-t.num_rows // FILES)
    for i in range(FILES):
        pq.write_table(t.slice(i * step, step), src / f"fineweb-{i:03d}.parquet")
    del t
    clean, cx = tmp_path / "clean", tmp_path / "clean-excl"
    sigs, dups, rem = tmp_path / "sigs", tmp_path / "dups", tmp_path / "removed"
    final, fx = tmp_path / "final", tmp_path / "final-excl"
    recipe = ["--recipe", "fineweb", "--lid-model", str(LID_BIN), "--no-dedup"]
    for i in range(4):  # four "machines", two worker processes each
        args = ["clean", str(src), str(clean), *recipe, "--excluded", str(cx)]
        assert main([*args, "--shard", f"{i}/4", "--workers", "2"]) == 0
    for i in range(4):
        args = ["dedup", "signatures", str(clean), str(sigs), "--hash-fc", "sha1"]
        assert main([*args, "--shard", f"{i}/4"]) == 0
    for i in range(2):
        assert main(["dedup", "buckets", str(sigs), str(dups), "--shard", f"{i}/2"]) == 0
    assert main(["dedup", "cluster", str(dups), str(rem)]) == 0
    for i in range(4):
        args = ["dedup", "filter", str(clean), str(final), "--removed", str(rem)]
        assert main([*args, "--excluded", str(fx), "--shard", f"{i}/4"]) == 0
    assert _digest(final, [cx, fx]) == DATATROVE_DIGEST

    # datatrove's own stages 1-3 on the same cleaned files
    dt = _dt("datatrove.pipeline.dedup.minhash")

    dt_sigs, dt_dups, dt_rem = tmp_path / "dt-sigs", tmp_path / "dt-dups", tmp_path / "dt-rem"
    files = sorted(clean.glob("*.parquet"))
    assert len(files) == FILES
    ctx = multiprocessing.get_context("spawn")
    with cf.ProcessPoolExecutor(min(8, os.cpu_count() or 1), mp_context=ctx) as pool:
        list(pool.map(_dt_signatures, [(str(f), str(dt_sigs), r) for r, f in enumerate(files)]))
    cfg = _dt_config()
    for b in range(cfg.num_buckets):
        dt.MinhashDedupBuckets(
            input_folder=str(dt_sigs), output_folder=str(dt_dups), config=cfg
        ).run(None, b, cfg.num_buckets)
    dt.MinhashDedupCluster(input_folder=str(dt_dups), output_folder=str(dt_rem), config=cfg).run()
    assert _same_tree(sigs, dt_sigs, "bucket_*/*.minhash.sig") == FILES * cfg.num_buckets
    assert _same_tree(dups, dt_dups, "*.dups") == cfg.num_buckets
    assert _same_tree(rem, dt_rem, "*.remove") > 0
