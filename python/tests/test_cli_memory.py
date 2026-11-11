"""Peak memory of the ``pyroid`` command on a 1,000,000-document synthetic corpus.

Each command runs in a child process; its peak RSS (its own ``VmHWM`` on Linux) must stay under
a fixed bound that does not grow with the corpus. For scale: pyroid's in-memory MinHash index
alone would hold 896 bytes of signatures per document (854 MiB here).

The FineWeb-recipe case needs ``lid.176.bin`` and about a minute of CPU on a release build, so it
runs only with ``PYROID_TEST_SCALE=1``.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("numpy")

if sys.platform == "win32":
    pytest.skip("reads peak RSS through /proc or resource (Linux, macOS)", allow_module_level=True)

pytestmark = [pytest.mark.thread_unsafe, pytest.mark.timeout(1200)]

N = 1_000_000
FILES = 4
DATA = Path(os.environ.get("PYROID_DATA", "/data"))
LID_BIN = DATA / "models" / "lid.176.bin"

#: Peak RSS bounds, MiB. The interpreter with pyarrow and pyroid loaded is about 100 MiB.
BOUND_STREAM = 300
BOUND_DEDUP = 400  # plus the 64 MiB sort buffer the tests pass
BOUND_CLEAN = 900  # plus the LID model (lid.176.bin is 125 MiB on disk)
if sys.platform == "darwin" and platform.machine() == "arm64":
    # The dedup commands' peak is higher on macOS arm64 (16 KiB pages, and its malloc keeps freed
    # memory resident longer, probably): GitHub's macos-15 runners measured 407 and 427 MiB on two
    # of eight legs, where Linux aarch64 gives 306 (one pass) and 260 (signatures), and macOS
    # x86_64 stays under 400. The bound is still fixed, so a peak that grows with N still fails.
    BOUND_DEDUP = 512

VOCAB = [
    "the",
    "of",
    "and",
    "to",
    "in",
    "is",
    "was",
    "that",
    "for",
    "on",
    "with",
    "as",
    "by",
    "at",
    "from",
    "his",
    "her",
    "they",
    "this",
    "have",
    "had",
    "are",
    "not",
    "but",
    "be",
    "or",
    "which",
    "one",
    "you",
    "all",
    "were",
    "we",
    "when",
    "there",
    "can",
    "been",
    "has",
    "more",
    "if",
    "will",
    "would",
    "who",
    "so",
    "no",
    "about",
    "out",
    "many",
    "then",
    "them",
    "some",
    "these",
    "time",
    "two",
    "may",
    "first",
    "like",
    "over",
    "into",
    "only",
    "other",
    "new",
    "years",
    "after",
    "also",
    "most",
    "made",
    "world",
    "city",
    "river",
    "north",
    "south",
    "state",
    "family",
    "house",
    "water",
    "school",
    "music",
    "church",
    "market",
    "garden",
    "station",
    "village",
    "island",
    "bridge",
    "forest",
    "mountain",
    "valley",
    "summer",
    "winter",
    "morning",
    "evening",
    "history",
    "people",
    "country",
    "government",
    "company",
    "century",
    "children",
    "letter",
    "window",
    "paper",
    "story",
    "table",
    "voice",
    "field",
    "road",
    "light",
    "stone",
    "train",
    "street",
]


def _doc(rng: random.Random, sentences: int, words: int) -> str:
    lines = []
    for _ in range(sentences):
        s = " ".join(rng.choices(VOCAB, k=words))
        lines.append(s[0].upper() + s[1:] + ".")
    return "\n".join(lines)


def _corpus(root: Path, sentences: int, words: int) -> Path:
    """N documents in FILES Parquet files; 2% are near-copies of an earlier document."""
    rng = random.Random(1234)
    d = root / "in"
    d.mkdir()
    per = N // FILES
    docs: list[str] = []
    for f in range(FILES):
        texts = []
        for i in range(per):
            if docs and rng.random() < 0.02:
                base = docs[rng.randrange(len(docs))].split(" ")
                base[rng.randrange(len(base))] = "changed"
                texts.append(" ".join(base))
            else:
                texts.append(_doc(rng, sentences, words))
            if len(docs) < 5000 and i % 100 == 0:
                docs.append(texts[-1])
        ids = [f"f{f}-{i}" for i in range(per)]
        pq.write_table(pa.table({"id": ids, "text": texts}), d / f"part-{f}.parquet")
    return d


#: Runs the command in a fresh interpreter, then reports that process's own peak RSS: on Linux
#: VmHWM, which (unlike the parent's view through wait4) does not include the RSS this test
#: process had when it forked the child.
_CHILD = """
import os, resource, sys
from pyroid.cli import main
rc = main(sys.argv[1:])
peak = None
try:
    with open("/proc/self/status") as f:
        peak = next(int(ln.split()[1]) / 1024 for ln in f if ln.startswith("VmHWM:"))
except OSError:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = r / 2**20 if sys.platform == "darwin" else r / 1024
with open(os.environ["PYROID_PEAK_OUT"], "w") as f:
    f.write(repr(peak))
sys.exit(rc)
"""


def _peak_mib(*args: str) -> float:
    """Run ``pyroid ARGS`` in a child interpreter; return its peak RSS in MiB."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "peak")
        r = subprocess.run(
            [sys.executable, "-c", _CHILD, *args],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env={**os.environ, "PYROID_PEAK_OUT": out},
            check=False,
            timeout=1200,
        )
        assert r.returncode == 0, r.stderr.decode(errors="replace")
        with open(out, encoding="utf-8") as f:
            mib = float(f.read())
    if log := os.environ.get("PYROID_TEST_PEAKS"):  # a file to log each command's peak in
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"{mib:.0f}\t{' '.join(args)}\n")
    return mib


@pytest.fixture(scope="module")
def short_docs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _corpus(tmp_path_factory.mktemp("short"), sentences=1, words=12)


def _count(path: Path) -> int:
    files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
    return sum(pq.read_metadata(p).num_rows for p in files)


def test_dedup_one_pass(short_docs: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    peak = _peak_mib("dedup", str(short_docs), str(out), "--sort-buffer-mb", "64")
    kept = _count(out)
    assert 0.9 * N < kept < N  # the near-copies go
    assert peak < BOUND_DEDUP, f"peak RSS {peak:.0f} MiB"


def test_four_stages(short_docs: Path, tmp_path: Path) -> None:
    sigs, dups, rem, out = (tmp_path / n for n in ("sigs", "dups", "rem", "out"))
    peaks: dict[str, float] = {}
    peaks["signatures"] = _peak_mib(
        "dedup", "signatures", str(short_docs), str(sigs), "--sort-buffer-mb", "64"
    )
    peaks["buckets"] = _peak_mib("dedup", "buckets", str(sigs), str(dups))
    peaks["cluster"] = _peak_mib("dedup", "cluster", str(dups), str(rem))
    peaks["filter"] = _peak_mib("dedup", "filter", str(short_docs), str(out), "--removed", str(rem))
    assert 0.9 * N < _count(out) < N
    over = {k: round(v) for k, v in peaks.items() if v >= BOUND_DEDUP}
    assert not over, f"peak RSS over {BOUND_DEDUP} MiB: {over}"


def test_streaming_commands(short_docs: Path, tmp_path: Path) -> None:
    peak = _peak_mib("translit", str(short_docs), str(tmp_path / "t.parquet"))
    assert peak < BOUND_STREAM, f"peak RSS {peak:.0f} MiB"
    assert _count(tmp_path / "t.parquet") == N


@pytest.mark.skipif(not os.environ.get("PYROID_TEST_SCALE"), reason="set PYROID_TEST_SCALE=1")
def test_fineweb_recipe(tmp_path: Path) -> None:
    if not LID_BIN.exists():
        pytest.skip("needs lid.176.bin under $PYROID_DATA")
    src = _corpus(tmp_path, sentences=6, words=10)
    out, excl = tmp_path / "out", tmp_path / "excl"
    args: list[Any] = ["clean", str(src), str(out), "--recipe", "fineweb"]
    args += ["--lid-model", str(LID_BIN), "--excluded", str(excl), "--sort-buffer-mb", "64"]
    peak = _peak_mib(*args)
    kept = _count(out)
    assert kept > N // 2  # most documents pass the filters; the near-copies go
    assert peak < BOUND_CLEAN, f"peak RSS {peak:.0f} MiB"
