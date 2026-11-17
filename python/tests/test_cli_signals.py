"""``pyroid clean --workers N`` stops on SIGINT and SIGTERM within seconds, resumably.

The signal goes to the main process only (as ``kill PID`` or a job scheduler sends it): the
main process stops its workers, every process of the run exits, no temporary file is left,
and running the command again finishes the run with the output of an uninterrupted one.
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.context
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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


def _clean(
    root: Path, out: str, workers: int = 2, *, ignore_sigint: bool = False
) -> subprocess.Popen[bytes]:
    cmd = [sys.executable, "-I", "-m", "pyroid", "clean", "parts", out, "--recipe", "fineweb"]
    if ignore_sigint:  # as a shell starts a background job: exec keeps an ignored signal ignored
        code = "import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_IGN); "
        code += "os.execv(sys.executable, sys.argv[1:])"
        cmd = [sys.executable, "-I", "-c", code, *cmd]
    cmd += ["--lid-model", str(root / "lid.bin"), "--workers", str(workers), "--threads", "1"]
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
def test_clean_stops_on_a_signal_and_resumes(
    run_dir: Path, sig: signal.Signals, tmp_path: Path
) -> None:
    ref_s = float((run_dir / "ref_seconds").read_text())
    out = tmp_path / "out"  # tmp_path: each of pytest-run-parallel's --iterations starts afresh
    proc = _clean(run_dir, str(out))
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
    assert _leftovers(out) == []
    again = _clean(run_dir, str(out))
    _, err = again.communicate(timeout=600)
    assert again.returncode == 0, err.decode()
    assert _rows(out) == _rows(run_dir / "ref")


def _leftovers(out: Path) -> list[str]:
    """Temporary files a stopped run left (in OUT or its work directory)."""
    return [
        str(p.relative_to(out))
        for p in out.rglob("*")
        if p.is_file() and (".partial" in p.name or p.name.endswith(".tmp"))
    ]


TRIES = 12


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM], ids=["SIGINT", "SIGTERM"])
def test_a_signal_to_the_whole_group_at_any_point_stops_the_run(
    run_dir: Path, sig: signal.Signals, tmp_path: Path
) -> None:
    """Ctrl-C in a terminal reaches every process of the group, the workers' too.

    It hung forever in about one try in four: a worker that took both its own signal and the
    main process's SIGTERM while it dropped its file then waited for a next file that never came,
    and the pool joined it. Tries land at random points of the run: starting workers, reading,
    writing, the MinHash stages, finishing.

    A signal that lands as the run ends, after its command has finished, ends the process by its
    default action (or comes too late to matter): the output is then complete.
    """
    ref_s = float((run_dir / "ref_seconds").read_text())
    rng = random.Random(f"group-{sig.name}")
    stopped = []
    for k in range(TRIES):
        out = tmp_path / f"g-{k}"
        proc = _clean(run_dir, str(out), workers=4)
        # From the moment the command has set up its run (run.json), just before it starts its
        # workers: earlier, Python is still importing and the signal's default action applies.
        deadline = time.monotonic() + 60
        while not (out / ".pyroid" / "run.json").exists() and proc.poll() is None:
            assert time.monotonic() < deadline, "the run did not start"
            time.sleep(0.005)
        time.sleep(rng.uniform(0.0, 0.9) * ref_s)
        if proc.poll() is not None:
            proc.communicate()
            continue
        os.killpg(proc.pid, sig)
        try:
            _, raw = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            pytest.fail(f"try {k}: still running 30 s after {sig.name} to the group")
        err = raw.decode()
        deadline = time.monotonic() + 10
        while _group_alive(proc.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _group_alive(proc.pid), f"try {k}: a process of the run outlived it"
        assert _leftovers(out) == [], k
        if proc.returncode in (0, -sig):  # the run had finished
            assert err == "", (k, proc.returncode, err)
            assert _rows(out) == _rows(run_dir / "ref"), k
            continue
        assert proc.returncode == 128 + sig, (k, proc.returncode, err)
        word = "terminated" if sig == signal.SIGTERM else "interrupted"
        assert err == f"pyroid clean: {word}\n", (k, err)
        stopped.append(out)
    if not stopped:
        pytest.skip("every run finished before its signal")
    again = _clean(run_dir, str(stopped[-1]), workers=4)
    _, raw = again.communicate(timeout=600)
    assert again.returncode == 0, raw.decode()
    assert _rows(stopped[-1]) == _rows(run_dir / "ref")


def test_a_run_that_ignores_sigint_goes_on_through_it(run_dir: Path, tmp_path: Path) -> None:
    """A shell script's background job starts with SIGINT ignored, and Ctrl-C in its terminal
    reaches it all the same. The run goes on, its workers too: they stopped, and the run failed
    with "a worker process exited (exit code 0) while processing ..."."""
    out = tmp_path / "out"
    proc = _clean(run_dir, str(out), workers=4, ignore_sigint=True)
    deadline = time.monotonic() + 60
    while not (out / ".pyroid" / "run.json").exists() and proc.poll() is None:
        assert time.monotonic() < deadline, "the run did not start"
        time.sleep(0.005)
    for delay in (0.0, 0.1, 0.3, 0.6):  # the workers start, then run their first files
        time.sleep(delay)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
    _, err = proc.communicate(timeout=600)
    assert (proc.returncode, err.decode()) == (0, "")
    assert _rows(out) == _rows(run_dir / "ref")


_SPAWN_START = multiprocessing.context.SpawnProcess.start


def test_a_signal_while_the_workers_start_is_acted_on_once_all_have(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The main thread blocks SIGINT and SIGTERM while it starts the workers, but the kernel
    gives a group's signal to another thread (pyarrow's) and Python then raised it in the main
    thread anyway, inside ``Process.start``: the child got no start-up data and printed an
    EOFError. Now the signal is acted on once every worker has started."""
    import threading

    from pyroid.integrations import _shards

    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signal handlers run in the main thread")
    idle = threading.Event()
    other = threading.Thread(target=idle.wait, daemon=True)
    other.start()
    started: list[Any] = []

    def start(self: Any) -> None:
        _SPAWN_START(self)
        started.append(self)
        if len(started) == 1:
            assert other.ident is not None
            signal.pthread_kill(other.ident, signal.SIGINT)  # as the kernel can
            time.sleep(0.2)  # the main thread's next check is in here

    try:
        with monkeypatch.context() as m:
            m.setattr(multiprocessing.context.SpawnProcess, "start", start)
            with pytest.raises(KeyboardInterrupt):
                _shards._spawn(multiprocessing.get_context("spawn"), 2, None, False)
    finally:
        idle.set()
    assert len(started) == 2
    assert all(p.exitcode is not None for p in started)  # stopped and joined
