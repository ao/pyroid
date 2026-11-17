"""Sharded, resumable, multi-process runs of the ``pyroid`` CLI's per-file work.

A run's state lives in a directory next to its output (``OUT/.pyroid`` for a directory OUT,
``.NAME.pyroid`` beside a file OUT):

- ``done/KEY.json``: one record per finished unit (usually one input file): the run's
  fingerprint, the input's size, the outputs and the unit's stats. A rerun with the same
  fingerprint skips these units.
- ``run.json``: what the run is (command, options, inputs), for people and for later stages.

Outputs are written to temporary names and renamed into place, and a unit's record is written
only after its outputs, so a crash at any point leaves no half-written file under a final name
and a rerun redoes only unfinished units.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import multiprocessing
import os
import pickle
import signal
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from multiprocessing import connection as mp_connection
from pathlib import Path
from typing import Any

import pyroid

from . import _io

# --------------------------------------------------------------------------------------- sharding


@dataclass(frozen=True)
class Shard:
    """``--shard I/N``: this process does items ``I, I + N, I + 2N, ...`` (datatrove's split)."""

    index: int = 0
    count: int = 1

    @classmethod
    def parse(cls, spec: str) -> Shard:
        """Parse ``"I/N"`` with ``0 <= I < N``."""
        try:
            i, n = (int(x) for x in spec.split("/"))
        except ValueError:
            raise ValueError(f"--shard {spec!r}: expected I/N, e.g. 0/8") from None
        if n < 1 or not 0 <= i < n:
            raise ValueError(f"--shard {spec!r}: need 0 <= I < N")
        return cls(i, n)

    def take(self, items: Sequence[Any]) -> list[Any]:
        """This shard's items."""
        return list(items[self.index :: self.count])

    @property
    def tag(self) -> str:
        """``00003-of-00008``."""
        return f"{self.index:05d}-of-{self.count:05d}"

    def __str__(self) -> str:
        return f"{self.index}/{self.count}"


# ------------------------------------------------------------------------------------ fingerprint


def fingerprint(command: str, options: Mapping[str, Any], sources: Sequence[_io.Source]) -> str:
    """A short hash of what decides a run's outputs (not how it is split or scheduled).

    Inputs count by name (relative to their IN) and size, not by absolute path or URI, so a
    machine that mounts the same data elsewhere resumes the same run.
    """
    blob = json.dumps(
        {
            "command": command,
            "options": dict(sorted(options.items())),
            "inputs": [[s.name, s.size] for s in sources],
            "pyroid": pyroid.__version__,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def folder_digest(folder: str | os.PathLike[str], *patterns: str) -> str:
    """A short hash of the names and sizes of the files matching ``patterns`` under ``folder``."""
    root = Path(folder)
    h = hashlib.sha256()
    names = sorted({p for pat in patterns for p in root.glob(pat) if p.is_file()})
    for p in names:
        h.update(f"{p.relative_to(root).as_posix()}\t{p.stat().st_size}\n".encode())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------------------ state


class StateError(Exception):
    """The output already holds results the current run must not silently replace."""


@dataclass
class State:
    """The ``done`` records and ``run.json`` of one output (see the module docstring).

    Records store output paths relative to the directory that holds the state (OUT, or the
    directory of a file OUT) when they are inside it, so they stay valid on another mount.
    """

    root: Path
    fingerprint: str

    @property
    def done_dir(self) -> Path:
        return self.root / "done"

    def _store(self, path: str) -> str:
        base = os.path.abspath(self.root.parent)
        p = os.path.abspath(path)
        if os.path.commonpath([base, p]) == base:
            return Path(os.path.relpath(p, base)).as_posix()
        return p

    def _load(self, path: str) -> str:
        return os.path.normpath(os.path.join(os.path.abspath(self.root.parent), path))

    def record(self, key: str) -> dict[str, Any] | None:
        """The record of unit ``key``, or None (missing or unreadable)."""
        try:
            data = json.loads((self.done_dir / f"{key}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def is_done(
        self, key: str, size: int | None = None, outputs: Sequence[str] | None = None
    ) -> bool:
        """True if ``key`` finished in a run with this fingerprint and its outputs still exist.

        ``size``: the input's size now (a different recorded size means the input changed).
        ``outputs``: the paths the unit writes now; a record of other paths (another
        ``--format`` or ``--excluded``) is not done.
        """
        r = self.record(key)
        if r is None or r.get("fingerprint") != self.fingerprint:
            return False
        if size is not None and r.get("size") not in (None, size):
            return False
        recorded = [self._load(p) for p in r.get("outputs", [])]
        # The current paths are as the command names them (relative to the working directory),
        # the recorded ones relative to the state's directory.
        now = [os.path.normpath(os.path.abspath(p)) for p in outputs or ()]
        if outputs is not None and sorted(recorded) != sorted(now):
            return False
        return all(os.path.exists(p) for p in recorded)

    def stale(self, keys: Iterable[str]) -> list[str]:
        """The keys whose record comes from a run with another fingerprint."""
        out = []
        for k in keys:
            r = self.record(k)
            if r is not None and r.get("fingerprint") != self.fingerprint:
                out.append(k)
        return out

    def mark(self, key: str, record: Mapping[str, Any]) -> None:
        """Record ``key`` as finished (after its outputs are in place)."""
        data = {"fingerprint": self.fingerprint, **record}
        if "outputs" in data:
            data["outputs"] = [self._store(p) for p in data["outputs"]]
        _io.atomic_write_text(self.done_dir / f"{key}.json", json.dumps(data, indent=1) + "\n")

    def forget(self, keys: Iterable[str]) -> None:
        """Drop the records of ``keys`` (their units will be redone)."""
        for k in keys:
            _io.remove(self.done_dir / f"{k}.json")

    def records(self) -> list[dict[str, Any]]:
        """Every record under ``done/`` (any fingerprint), in key order."""
        if not self.done_dir.is_dir():
            return []
        out = []
        for p in sorted(self.done_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict):
                out.append({"key": p.stem, **data})
        return out

    def write_run(self, info: Mapping[str, Any]) -> None:
        """Write ``run.json``."""
        data = {"fingerprint": self.fingerprint, "pyroid": pyroid.__version__, **info}
        _io.atomic_write_text(self.root / "run.json", json.dumps(data, indent=1) + "\n")

    def read_run(self) -> dict[str, Any] | None:
        """``run.json``, or None."""
        try:
            data = json.loads((self.root / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None


def pending(
    state: State, units: Sequence[Unit], *, overwrite: bool, what: str = "files"
) -> list[Unit]:
    """The units still to do. Raises StateError for records of another run (unless overwrite)."""
    keys = [u.key for u in units]
    stale = state.stale(keys)
    if stale and not overwrite:
        raise StateError(
            f"{state.root.parent}: {len(stale)} finished {what} come from a different run "
            "(other options, inputs or pyroid version); pass --overwrite to redo them, "
            "or use another output path"
        )
    if overwrite:
        state.forget(keys)
        return list(units)
    return [u for u in units if not state.is_done(u.key, u.size, u.outputs)]


# ------------------------------------------------------------------------------------------ stats


@dataclass
class Stats:
    """Document counts of one unit or of a whole run."""

    docs_in: int = 0
    docs_out: int = 0
    removed: Counter[str] = field(default_factory=Counter)

    def add(self, other: Stats | Mapping[str, Any]) -> None:
        if isinstance(other, Stats):
            self.docs_in += other.docs_in
            self.docs_out += other.docs_out
            self.removed.update(other.removed)
            return
        self.docs_in += int(other.get("docs_in", 0))
        self.docs_out += int(other.get("docs_out", 0))
        self.removed.update({k: int(v) for k, v in other.get("removed", {}).items()})

    def as_dict(self) -> dict[str, Any]:
        removed = {k: v for k, v in self.removed.items() if v}
        return {
            "docs_in": self.docs_in,
            "docs_out": self.docs_out,
            "removed": dict(sorted(removed.items(), key=lambda kv: (-kv[1], kv[0]))),
        }


def summary(
    command: str, stats: Stats, *, wall: float, extra: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The ``--stats-json`` document (the 1.0 keys first, then the run's layout)."""
    by_stage: Counter[str] = Counter()
    for reason, n in stats.removed.items():
        if n:
            by_stage[reason.split(":", 1)[0]] += n
    d = stats.as_dict()
    return {
        "command": command,
        **d,
        "removed_by_stage": dict(sorted(by_stage.items(), key=lambda kv: (-kv[1], kv[0]))),
        "wall_s": round(wall, 3),
        "pyroid": pyroid.__version__,
        **(extra or {}),
    }


def print_summary(d: Mapping[str, Any]) -> None:
    """``--stats``: counts and drops by reason, on stderr."""
    lines = [f"pyroid {d['command']}: {d['docs_in']:,} in, {d['docs_out']:,} out, {d['wall_s']} s"]
    width = max((len(k) for k in d["removed"]), default=0)
    for k, n in d["removed"].items():
        pct = 100 * n / d["docs_in"] if d["docs_in"] else 0.0
        lines.append(f"  dropped {k:<{width}}  {n:>9,}  {pct:5.1f}%")
    say("".join(f"{ln}\n" for ln in lines))


_muted = False  # stderr was a closed pipe: messages are dropped
_WINDOWS = sys.platform == "win32"


def closed_pipe(e: BaseException) -> bool:
    """True for the error of a write to a pipe that nobody reads any more.

    POSIX raises BrokenPipeError. On Windows the C runtime reports the same write as EINVAL
    (from ERROR_NO_DATA), and Python raises a plain OSError.
    """
    if isinstance(e, BrokenPipeError):
        return True
    return _WINDOWS and isinstance(e, OSError) and e.errno == errno.EINVAL


def say(text: str) -> None:
    """Write ``text`` to stderr, for people (progress, stats, errors).

    When stderr is a pipe that closed (``2>&1 | head -1``), this and later messages are dropped
    and the run goes on: its outputs and exit status do not depend on them.
    """
    global _muted
    if _muted:
        return
    try:
        sys.stderr.write(text)
        sys.stderr.flush()
    except (OSError, ValueError) as e:  # ValueError: stderr was closed
        if isinstance(e, OSError) and not closed_pipe(e):
            raise
        _muted = True
        mute(sys.stderr)


def mute(stream: Any) -> None:
    """Point ``stream``'s descriptor at the null device: nothing more fails on it, nor at exit."""
    with contextlib.suppress(Exception):
        null = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(null, stream.fileno())
        finally:
            os.close(null)


# --------------------------------------------------------------------------------------- progress


class Progress:
    """``--progress``: one status line on stderr (rewritten in place on a terminal)."""

    def __init__(self, label: str, units: int, total_bytes: int | None, *, enabled: bool) -> None:
        self.label = label
        self.units = units
        self.total_bytes = total_bytes
        self.enabled = enabled
        self.done_units = 0
        self.done_bytes = 0
        self.docs = 0
        self.t0 = time.monotonic()
        self._last = 0.0
        self._tty = enabled and sys.stderr.isatty()
        self._width = 0

    def docs_seen(self, n: int) -> None:
        self.docs += n
        self.show()

    def unit_done(self, size: int | None) -> None:
        self.done_units += 1
        self.done_bytes += size or 0
        self.show()

    def line(self) -> str:
        el = time.monotonic() - self.t0
        rate = self.docs / el if el > 0 else 0.0
        parts = [
            f"{self.label}: {self.done_units:,}/{self.units:,} files",
            f"{self.docs:,} docs",
            f"{_si(rate)} docs/s",
            f"elapsed {_hms(el)}",
        ]
        if self.total_bytes and self.done_bytes and self.done_units < self.units:
            eta = el * (self.total_bytes - self.done_bytes) / self.done_bytes
            parts.append(f"eta {_hms(eta)}")
        return ", ".join(parts)

    def show(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self._last < (0.5 if self._tty else 10.0):
            return
        self._last = now
        text = self.line()
        if self._tty:
            pad = " " * max(0, self._width - len(text))
            say(f"\r{text}{pad}")
            self._width = len(text)
        else:
            say(text + "\n")

    def close(self) -> None:
        if not self.enabled:
            return
        self.show(force=True)
        if self._tty:
            say("\n")


def _si(x: float) -> str:
    for unit, div in (("M", 1e6), ("k", 1e3)):
        if x >= div:
            return f"{x / div:.1f}{unit}"
    return f"{x:.0f}"


def _hms(s: float) -> str:
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


# ------------------------------------------------------------------------------------------ units


@dataclass(frozen=True)
class Unit:
    """One piece of work: usually one input file, with where its outputs go."""

    key: str
    label: str
    size: int | None
    payload: Any
    outputs: tuple[str, ...] | None = None  # what the unit writes (checked against its record)


Report = Callable[[int], None]
UnitFn = Callable[[Any, Report], dict[str, Any]]

#: Seconds the main process gives its workers to stop (each drops the unit it is on and removes
#: its temporary files, usually within a second) before it kills them.
STOP_GRACE = 10.0
_SIGNALS = (signal.SIGINT, signal.SIGTERM)

_interrupted = False  # set in a worker by its first Ctrl-C or SIGTERM


class Terminated(KeyboardInterrupt):
    """SIGTERM, raised in the main thread (as ``KeyboardInterrupt`` is for SIGINT)."""


class _WorkerError(Exception):
    """The traceback of an exception a worker raised: its ``__cause__`` in the main process."""

    def __init__(self, tb: str) -> None:
        super().__init__(tb)
        self.tb = tb

    def __str__(self) -> str:
        return self.tb


def _on_signal(signum: int, frame: Any) -> None:
    """A worker's SIGINT and SIGTERM handler: the first signal raises KeyboardInterrupt.

    The unit the worker is on (or its wait for the next one) stops there, and the worker exits.
    Later signals do nothing: Ctrl-C reaches every process of the group and the main process
    then sends SIGTERM, and a second KeyboardInterrupt would cut short the unit's clean-up, which
    removes its temporary files.
    """
    global _interrupted
    if _interrupted:
        return
    _interrupted = True
    raise KeyboardInterrupt


def _watch_parent() -> None:
    """Stop this worker when its parent dies (killed, crashed): nothing else would.

    The running unit is interrupted as by Ctrl-C (its temporary outputs are removed);
    a worker waiting for its next unit sees its connection close.
    """
    import _thread
    import threading

    parent = multiprocessing.parent_process()
    if parent is None:  # pragma: no cover - always set in a spawned worker
        return

    def watch() -> None:
        parent.join()
        _thread.interrupt_main()
        time.sleep(5.0)
        os._exit(1)

    threading.Thread(target=watch, name="pyroid-parent-watch", daemon=True).start()


def _setup_threads(threads: int | None) -> None:
    if threads is not None:
        # Before pyroid's pool is first used in this process (it is built lazily).
        os.environ["PYROID_NUM_THREADS"] = str(threads)
        try:
            import pyarrow as pa

            pa.set_cpu_count(threads)
            pa.set_io_thread_count(max(threads, 4))
        except ImportError:  # pragma: no cover - the CLI checks for pyarrow first
            pass


def _portable(e: BaseException) -> tuple[BaseException, str]:
    """``e`` if it pickles (else a RuntimeError with its text), and its formatted traceback."""
    import traceback

    tb = "".join(traceback.format_exception(e))
    try:
        pickle.loads(pickle.dumps(e))
    except Exception:
        e = RuntimeError(f"{type(e).__name__}: {e}")
    return e, tb


def _worker(conn: Any, threads: int | None, progress: bool) -> None:
    """A worker process: run the units the main process sends, one at a time.

    Messages to the main process: ``("progress", rows)``, then ``("done", result)`` or
    ``("error", exception, traceback)`` per unit. The worker stops when the main process closes
    the connection or dies, and at its first SIGINT or SIGTERM (after reporting the interrupted
    unit). It shares no lock with another process, so it can stop at any point.
    """
    try:
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
        if hasattr(signal, "pthread_sigmask"):
            # The main process started this one with both blocked; one that came in the meantime
            # is delivered now, to _on_signal.
            signal.pthread_sigmask(signal.SIG_UNBLOCK, _SIGNALS)
        _watch_parent()
        _setup_threads(threads)

        def report(n: int) -> None:
            if _interrupted:  # the unit's code swallowed the KeyboardInterrupt (a finalizer)
                raise KeyboardInterrupt
            if progress and n:
                conn.send(("progress", n))

        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                return
            except Exception as e:  # the unit does not unpickle here
                conn.send(("error", *_portable(e)))
                continue
            fn, payload = msg
            try:
                result = fn(payload, report)
            except BaseException as e:
                conn.send(("error", *_portable(e)))
                if isinstance(e, KeyboardInterrupt):
                    return
                continue
            try:
                conn.send(("done", result))
            except (pickle.PicklingError, TypeError, AttributeError) as e:  # does not pickle
                conn.send(("error", *_portable(e)))
    except (KeyboardInterrupt, OSError):
        return  # interrupted, or the main process stopped or lost this worker


@dataclass
class _Worker:
    proc: Any
    conn: Any
    unit: Unit | None = None


def _spawn(ctx: Any, count: int, threads: int | None, progress: bool) -> list[_Worker]:
    """Start ``count`` workers, each with SIGINT and SIGTERM blocked until its handlers are set.

    Python's own handlers would otherwise raise KeyboardInterrupt (and print a traceback) in a
    worker that is still importing.
    """
    pool: list[_Worker] = []
    mask = hasattr(signal, "pthread_sigmask")
    old = signal.pthread_sigmask(signal.SIG_BLOCK, _SIGNALS) if mask else None
    try:
        for _ in range(count):
            mine, theirs = ctx.Pipe()
            proc = ctx.Process(
                target=_worker, args=(theirs, threads, progress), name="pyroid-worker"
            )
            pool.append(_Worker(proc, mine))
            proc.start()
            theirs.close()
    except BaseException:
        _stop(pool, gently=False)
        raise
    finally:
        if old is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, old)
    return pool


@contextlib.contextmanager
def _held_signals(*, redeliver: bool) -> Iterator[list[int]]:
    """Note SIGINT and SIGTERM instead of acting on them, in the main thread, during the block.

    Yields the list of the signals that came. With ``redeliver``, the first is raised again after
    the block, for the handler it would have reached.
    """
    import threading

    got: list[int] = []
    if threading.current_thread() is not threading.main_thread():
        yield got
        return

    def note(signum: int, frame: Any) -> None:
        got.append(signum)

    old = {s: signal.signal(s, note) for s in _SIGNALS}
    try:
        yield got
    finally:
        for s, h in old.items():
            signal.signal(s, h if h is not None else signal.SIG_DFL)  # None: set outside Python
    if got and redeliver:
        signal.raise_signal(got[0])


def _how(code: int | None) -> str:
    """How a worker process ended, for a message."""
    if code is not None and code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = f"signal {-code}"
        return f"killed by {name}" + (", out of memory?" if -code == 9 else "")
    return f"exit code {code}"


def _stop(pool: Sequence[_Worker], *, gently: bool) -> None:
    """End every worker and wait for it.

    ``gently``: the run is complete; the workers are idle and exit when their connection closes.
    Otherwise each gets SIGTERM and drops the unit it is on, removing its temporary files.
    Workers still alive after :data:`STOP_GRACE` seconds, or after another Ctrl-C or SIGTERM
    meanwhile, are killed.
    """
    started = [w for w in pool if w.proc.pid is not None]
    with _held_signals(redeliver=gently) as held:
        for w in started:
            if not gently and w.proc.exitcode is None:
                with contextlib.suppress(OSError):
                    w.proc.terminate()
            with contextlib.suppress(OSError):
                w.conn.close()
        deadline = time.monotonic() + STOP_GRACE
        while not held:
            alive = [w.proc.sentinel for w in started if w.proc.exitcode is None]
            left = deadline - time.monotonic()
            if not alive or left <= 0:
                break
            mp_connection.wait(alive, timeout=min(left, 0.1))
        killed = [w for w in started if w.proc.exitcode is None]
        for w in killed:
            with contextlib.suppress(OSError):
                w.proc.kill()
        for w in started:
            w.proc.join()
        if killed and not held:
            say(
                f"pyroid: killed {len(killed)} worker process(es) that did not stop within "
                f"{STOP_GRACE:.0f} s\n"
            )


def _collect(
    w: _Worker, progress: Progress | None, out: list[tuple[Unit, dict[str, Any]]]
) -> BaseException | None:
    """Read what worker ``w`` sent; return the failure of its unit, if it failed or died."""
    while w.unit is not None:
        try:
            if not w.conn.poll():
                break
            msg = w.conn.recv()
        except (EOFError, OSError):
            break
        if msg[0] == "progress":
            if progress is not None:
                progress.docs_seen(msg[1])
            continue
        u, w.unit = w.unit, None
        if msg[0] == "done":
            if progress is not None:
                progress.unit_done(u.size)
            out.append((u, msg[1]))
            return None
        exc, tb = msg[1], msg[2]
        exc.__cause__ = _WorkerError(tb)
        return _noted(exc, u.label)
    if w.unit is not None and w.proc.exitcode is not None:
        u, w.unit = w.unit, None
        return ChildProcessError(
            f"a worker process exited ({_how(w.proc.exitcode)}) while processing {u.label}"
        )
    return None


def _run_workers(
    units: Sequence[Unit], fn: UnitFn, workers: int, threads: int | None, progress: Progress | None
) -> list[tuple[Unit, dict[str, Any]]]:
    ctx = multiprocessing.get_context("spawn")
    show = progress is not None and progress.enabled
    out: list[tuple[Unit, dict[str, Any]]] = []
    todo = list(reversed(units))  # pop() takes them in order
    failure: BaseException | None = None
    pool: list[_Worker] = []
    try:
        pool = _spawn(ctx, min(workers, len(units)), worker_threads(workers, threads), show)
        while True:
            for w in pool:
                if w.unit is not None or not todo or failure is not None:
                    continue
                u = todo.pop()
                try:
                    w.conn.send((fn, u.payload))
                except OSError:  # the worker died while idle (killed?)
                    w.proc.join(5.0)
                    failure = ChildProcessError(
                        f"a worker process exited ({_how(w.proc.exitcode)}) before processing "
                        f"{u.label}"
                    )
                    continue
                w.unit = u
            busy = [w for w in pool if w.unit is not None]
            if not busy:
                break
            waits = [w.conn for w in busy] + [w.proc.sentinel for w in busy]
            ready = set(mp_connection.wait(waits, timeout=0.5))
            for w in busy:
                if w.conn in ready or w.proc.sentinel in ready:
                    err = _collect(w, progress, out)
                    if failure is None:
                        failure = err
    except BaseException:
        # Ctrl-C, SIGTERM or an error here: stop the workers now. A worker that got no signal
        # itself (the main process alone was signalled) would finish its unit first.
        _stop(pool, gently=False)
        raise
    _stop(pool, gently=True)
    if failure is not None:
        raise failure
    return out


def worker_threads(workers: int, threads: int | None) -> int | None:
    """Pool size for each of ``workers`` processes: ``threads`` if given, else an equal share."""
    if threads is not None:
        return threads
    if workers <= 1:
        return None
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return max(1, (cpus or 1) // workers)


def _noted(exc: BaseException, label: str) -> BaseException:
    """``exc`` with a note naming the unit it came from (shown in tracebacks and CLI errors)."""
    note = f"while processing {label}"
    if note not in getattr(exc, "__notes__", ()):
        exc.add_note(note)
    return exc


def run_units(
    units: Sequence[Unit],
    fn: UnitFn,
    *,
    workers: int = 1,
    threads: int | None = None,
    progress: Progress | None = None,
) -> list[tuple[Unit, dict[str, Any]]]:
    """Run ``fn(unit.payload, report)`` for every unit; return ``(unit, result)`` as they finished.

    With ``workers > 1`` the units run in that many spawned processes (``fn`` and the payloads
    must pickle), each with ``worker_threads(workers, threads)`` pyroid threads. On the first
    failure no new unit starts, the running ones finish (their outputs and records are kept),
    and the failure is raised with a note naming its unit. On Ctrl-C, SIGTERM or an error in
    this process, every worker stops at once, dropping the unit it is on.
    """
    if workers <= 1 or len(units) <= 1:
        out = []
        for u in units:
            report = progress.docs_seen if progress is not None else _noop
            try:
                result = fn(u.payload, report)
            except Exception as e:
                _noted(e, u.label)
                raise
            if progress is not None:
                progress.unit_done(u.size)
            out.append((u, result))
        return out
    return _run_workers(units, fn, workers, threads, progress)


def _noop(n: int) -> None:
    return None


def rmtree(path: str | os.PathLike[str]) -> None:
    """Remove a directory tree, retrying files Windows reports as in use; best effort."""
    import shutil

    def retry(func: Callable[..., Any], p: str, _exc: BaseException) -> None:
        for delay in (0.05, 0.1, 0.2, 0.4, 0.8):
            time.sleep(delay)
            try:
                func(p)
                return
            except FileNotFoundError:
                return
            except PermissionError:
                continue
            except OSError:
                return

    if not os.path.exists(path):
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=retry)
    else:  # pragma: no cover - 3.11
        shutil.rmtree(path, onerror=lambda f, p, ei: retry(f, p, ei[1]))
