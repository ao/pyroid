"""``pyroid clean --workers N`` stops on SIGINT and SIGTERM within seconds, resumably.

The signal goes to the main process only (as ``kill PID`` or a job scheduler sends it): the
main process stops its workers, every process of the run exits, no temporary file is left,
and running the command again finishes the run with the output of an uninterrupted one.
"""

from __future__ import annotations

import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("numpy")

pytestmark = [
    pytest.mark.thread_unsafe(reason="runs the CLI in subprocesses"),
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and process groups"),
]

PARTS = 12
DOCS_PER_PART = 2_500
WORDS = (
    "the",
    "of",
    "and",
    "to",
    "in",
    "a",
    "is",
    "that",
    "for",
    "it",
    "as",
    "was",
    "with",
    "be",
    "by",
    "on",
    "not",
    "he",
    "this",
    "are",
    "or",
    "his",
    "from",
    "at",
    "which",
    "but",
    "have",
    "an",
    "they",
    "you",
    "were",
    "her",
    "she",
    "there",
    "one",
    "all",
    "we",
    "can",
    "their",
    "has",
    "been",
    "if",
    "more",
    "when",
    "will",
    "would",
    "who",
    "so",
    "no",
    "time",
    "people",
    "city",
    "river",
    "music",
    "water",
    "history",
)


def _doc(rng: random.Random) -> str:
    paras = []
    for _ in range(rng.randint(4, 8)):
        sents = [
            " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 20))).capitalize() + "."
            for _ in range(rng.randint(3, 6))
        ]
        paras.append(" ".join(sents))
    return "\n".join(paras)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Input parts, a one-label ("en") synthetic fastText model, and the uninterrupted output."""
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from test_fasttext import SPECS, build
    finally:
        sys.path.pop(0)
    import dataclasses

    root = tmp_path_factory.mktemp("signals")
    spec = dataclasses.replace(SPECS["softmax-dense"], labels=("__label__en",))
    (root / "lid.bin").write_bytes(build(spec).data)
    (root / "parts").mkdir()
    rng = random.Random(7)
    for i in range(PARTS):
        docs = [_doc(rng) for _ in range(DOCS_PER_PART)]
        table = pa.table({"id": [f"{i}-{j}" for j in range(len(docs))], "text": docs})
        pq.write_table(table, root / "parts" / f"part-{i:02d}.parquet", row_group_size=500)
    t0 = time.perf_counter()
    proc = _clean(root, "ref")
    _, err = proc.communicate(timeout=600)
    assert proc.returncode == 0, err.decode()
    (root / "ref_seconds").write_text(str(time.perf_counter() - t0))
    return root


def _clean(root: Path, out: str) -> subprocess.Popen[bytes]:
    cmd = [sys.executable, "-I", "-m", "pyroid", "clean", "parts", out, "--recipe", "fineweb"]
    cmd += ["--lid-model", str(root / "lid.bin"), "--workers", "2", "--threads", "1"]
    env = {k: v for k, v in os.environ.items() if k not in ("PYROID_NUM_THREADS",)}
    return subprocess.Popen(
        cmd,
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - a zombie we may not signal
        return True
    return True


def _rows(folder: Path) -> list[dict[str, object]]:
    files = sorted(p for p in folder.rglob("*.parquet") if not p.name.startswith("."))
    return [row for f in files for row in pq.read_table(f).to_pylist()]


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"])
def test_clean_stops_on_a_signal_and_resumes(run_dir: Path, sig: signal.Signals) -> None:
    ref_s = float((run_dir / "ref_seconds").read_text())
    out = f"out-{sig.name}"
    proc = _clean(run_dir, out)
    time.sleep(max(0.8, 0.35 * ref_s))
    if proc.poll() is not None:
        proc.communicate()
        pytest.skip(f"the run finished in under {0.35 * ref_s:.1f} s, before the signal")
    proc.send_signal(sig)
    t0 = time.perf_counter()
    _, err = proc.communicate(timeout=60)
    stopped = time.perf_counter() - t0
    assert proc.returncode == 128 + sig, err.decode()
    assert stopped < 10, stopped
    assert (
        f"pyroid clean: {'terminated' if sig == signal.SIGTERM else 'interrupted'}" in err.decode()
    )
    assert "Traceback" not in err.decode(), err.decode()
    deadline = time.monotonic() + 10
    while _group_alive(proc.pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _group_alive(proc.pid), "a worker process outlived the run"
    leftovers = [
        p.name
        for p in (run_dir / out).rglob(".*")
        if p.is_file() and p.suffix in (".partial", ".tmp")
    ]
    assert leftovers == []
    again = _clean(run_dir, out)
    _, err = again.communicate(timeout=600)
    assert again.returncode == 0, err.decode()
    assert _rows(run_dir / out) == _rows(run_dir / "ref")
