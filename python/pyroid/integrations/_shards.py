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

import concurrent.futures as cf
import hashlib
import json
import multiprocessing
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
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
    """A short hash of what decides a run's outputs (not how it is split or scheduled)."""
    blob = json.dumps(
        {
            "command": command,
            "options": dict(sorted(options.items())),
            "inputs": [[s.uri, s.size] for s in sources],
            "pyroid": pyroid.__version__,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------ state


class StateError(Exception):
    """The output already holds results the current run must not silently replace."""


@dataclass
class State:
    """The ``done`` records and ``run.json`` of one output (see the module docstring)."""

    root: Path
    fingerprint: str

    @property
    def done_dir(self) -> Path:
        return self.root / "done"

    def record(self, key: str) -> dict[str, Any] | None:
        """The record of unit ``key``, or None (missing or unreadable)."""
        try:
            data = json.loads((self.done_dir / f"{key}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def is_done(self, key: str, size: int | None = None) -> bool:
        """True if ``key`` finished in a run with this fingerprint (and the input size matches)."""
        r = self.record(key)
        if r is None or r.get("fingerprint") != self.fingerprint:
            return False
        if size is not None and r.get("size") not in (None, size):
            return False
        return all(Path(p).exists() for p in r.get("outputs", []))

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
    return [u for u in units if not state.is_done(u.key, u.size)]


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
    w = sys.stderr.write
    w(f"pyroid {d['command']}: {d['docs_in']:,} in, {d['docs_out']:,} out, {d['wall_s']} s\n")
    width = max((len(k) for k in d["removed"]), default=0)
    for k, n in d["removed"].items():
        pct = 100 * n / d["docs_in"] if d["docs_in"] else 0.0
        w(f"  dropped {k:<{width}}  {n:>9,}  {pct:5.1f}%\n")


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
            sys.stderr.write(f"\r{text}{pad}")
            self._width = len(text)
        else:
            sys.stderr.write(text + "\n")
        sys.stderr.flush()

    def close(self) -> None:
        if not self.enabled:
            return
        self.show(force=True)
        if self._tty:
            sys.stderr.write("\n")
            sys.stderr.flush()


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


Report = Callable[[int], None]
UnitFn = Callable[[Any, Report], dict[str, Any]]

_queue: Any = None  # the worker processes' progress queue (set by _init_worker)


def _init_worker(threads: int | None, queue: Any) -> None:
    global _queue
    _queue = queue
    if threads is not None:
        # Before pyroid's pool is first used in this process (it is built lazily).
        os.environ["PYROID_NUM_THREADS"] = str(threads)
        try:
            import pyarrow as pa

            pa.set_cpu_count(threads)
            pa.set_io_thread_count(max(threads, 4))
        except ImportError:  # pragma: no cover - the CLI checks for pyarrow first
            pass


def _report_to_queue(n: int) -> None:
    if _queue is not None and n:
        _queue.put(n)


def _call(fn: UnitFn, payload: Any) -> dict[str, Any]:
    return fn(payload, _report_to_queue)


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
) -> Iterator[tuple[Unit, dict[str, Any]]]:
    """Run ``fn(unit.payload, report)`` for every unit; yield ``(unit, result)`` as they finish.

    With ``workers > 1`` the units run in that many spawned processes (``fn`` and the payloads
    must pickle), each with ``worker_threads(workers, threads)`` pyroid threads. On the first
    failure no new unit starts, the running ones finish (their outputs and records are kept),
    and the failure is raised with a note naming its unit.
    """
    if workers <= 1 or len(units) <= 1:
        for u in units:
            report = progress.docs_seen if progress is not None else _noop
            try:
                result = fn(u.payload, report)
            except Exception as e:
                _noted(e, u.label)
                raise
            if progress is not None:
                progress.unit_done(u.size)
            yield u, result
        return

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.SimpleQueue() if progress is not None and progress.enabled else None
    per = worker_threads(workers, threads)
    with cf.ProcessPoolExecutor(
        max_workers=min(workers, len(units)),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(per, queue),
    ) as pool:
        futures = {pool.submit(_call, fn, u.payload): u for u in units}
        waiting = set(futures)
        failure: tuple[Unit, BaseException] | None = None
        try:
            while waiting:
                done, waiting = cf.wait(waiting, timeout=0.5, return_when=cf.FIRST_COMPLETED)
                if queue is not None and progress is not None:
                    while not queue.empty():
                        progress.docs_seen(queue.get())
                for f in done:
                    if f.cancelled():
                        continue
                    u = futures[f]
                    exc = f.exception()
                    if exc is not None:
                        if failure is None:
                            failure = (u, exc)
                            for g in waiting:
                                g.cancel()
                        continue
                    if progress is not None:
                        progress.unit_done(u.size)
                    yield u, f.result()
                if failure is not None:
                    waiting = {g for g in waiting if not g.cancelled()}
        except BaseException:
            for g in futures:
                g.cancel()
            raise
        if failure is not None:
            u, exc = failure
            raise _noted(exc, u.label)


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
