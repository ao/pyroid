"""Drive a module with warcio's API through seeded read plans and record everything a caller can observe.

A *trace* is the list of events one run produces: the iterator's construction, every ``next()``, every record's
fields (format, type, WARC and HTTP headers as ``repr``, content type, lengths, digest checker), every read from
``raw_stream`` / ``content_stream()`` (length, sha1, first bytes), offsets and lengths from the iterator, the
iterator's ``offset`` / ``err_count``, whatever was written to ``sys.stderr`` or logged during each call, and
every exception (type, message, where) together with what the next ``next()`` does. Two implementations agree
when their traces are equal, event for event; ``first_diff`` says where they part.

The run is set by a ``Config``:

- ``opts``: ``ArchiveIterator`` keyword arguments (``no_record_parse``, ``verify_http``, ``arc2warc``,
  ``ensure_http_headers``, ``block_size``, ``check_digests`` in False / True / "log" / "raise").
- ``plan``: what is done with each record. ``iterate`` (nothing: the iterator reads to the end itself),
  ``content`` (``content_stream().read()``, datatrove's call), ``blocks`` (``content_stream().read(16384)`` to
  the end), ``raw`` (``raw_stream.read()``), ``offsets`` (``get_record_offset`` / ``get_record_length``, the
  indexer's calls), ``random`` (a seeded mix per record of ``read(n)`` / ``readline(n)`` on both streams, a fresh
  ``content_stream()``, ``tell()``, ``read_to_end()``, offsets, header lookups and the digest checker, for
  n in 0, 1, 2, 3, 7, 64, 500, 4096, 16384, 70000, None, -1), ``stop`` (two records, then ``close()``).
- ``opener``: what the iterator reads. ``file`` (``open(path, "rb")``), ``fileio`` (unbuffered), ``bytesio``,
  ``unseekable`` (no ``tell``: warcio wraps it in ``UnseekableYetTellable``), ``short`` (``read(n)`` returns at
  most a seeded few bytes), ``logged`` (records every ``read(n)`` call and its length, in the ``io`` channel),
  ``gzipfile`` (``gzip.GzipFile(fileobj=open(path, "rb"))``) and ``datatrove`` (what datatrove's ``WarcReader``
  opens: ``DataFolder.open(name, "rb", compression="infer")``, a GzipFile over fsspec's file for ``.gz``).
- ``iterator``: ``ArchiveIterator``, ``WARCIterator`` or ``ARCIterator``.

Events are lists of str / int / bool / None, so a trace is JSON and its sha256 can be pinned (``digest``). Events
in the ``io`` channel (``fh.tell()`` after each record, the read calls of the ``logged`` opener) are compared
separately: they show how an implementation consumes its source, which users see only through ``tell()``.

``arc2warc`` gives converted ARC records a fresh ``uuid4`` record ID; those IDs are checked for shape and
replaced by ``<urn:uuid:*>`` before comparing.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import logging
import random
import re
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Event = list[Any]
UUID4 = re.compile(r"<urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}>")
MASK = "<urn:uuid:*>"
PREFIX = 40  # bytes of each read shown in the trace (all of them are hashed)
SIZES = (0, 1, 2, 3, 7, 64, 500, 4096, 16384, 70000, None, -1)
LOOKUPS = (
    "WARC-Type",
    "warc-type",
    "CONTENT-LENGTH",
    "WARC-Target-URI",
    "WARC-Record-ID",
    "WARC-Block-Digest",
    "WARC-Identified-Payload-Type",
    "Content-Type",
    "uri",
    "X-Missing",
)
HTTP_LOOKUPS = ("Content-Type", "content-encoding", "Transfer-Encoding", "Content-Length", "X-Missing")


@dataclass(frozen=True)
class Config:
    name: str
    opts: tuple[tuple[str, Any], ...] = ()
    plan: str = "content"
    opener: str = "file"
    iterator: str = "ArchiveIterator"
    seed: int = 0

    @property
    def kwargs(self) -> dict[str, Any]:
        return dict(self.opts)

    @property
    def needs_gzip(self) -> bool:
        return self.opener == "gzipfile"


def cfg(name: str, plan: str = "content", opener: str = "file", **opts: Any) -> Config:
    iterator = opts.pop("iterator", "ArchiveIterator")
    seed = opts.pop("seed", 0)
    return Config(name, tuple(sorted(opts.items())), plan, opener, iterator, seed)


# Configurations for the pinned Common Crawl slices (a few MB each): the read paths datatrove, the indexer and
# warcio's own tools use, plus a seeded random plan over each opener family.
CC_CONFIGS: tuple[Config, ...] = (
    cfg("content/file"),
    cfg("iterate/file", "iterate"),
    cfg("offsets/bytesio", "offsets", "bytesio"),
    cfg("random/file", "random", seed=1),
    cfg("raw/unseekable", "raw", "unseekable"),
    cfg("blocks/logged", "blocks", "logged", block_size=4096),
    cfg("content/gzipfile", "content", "gzipfile"),
    cfg("random/datatrove", "random", "datatrove", seed=2),
    cfg("digests-log/file", "content", check_digests="log"),
    cfg("no-record-parse/file", "raw", no_record_parse=True),
    cfg("arc2warc/fileio", "content", "fileio", arc2warc=True),
    cfg("verify-ensure/short", "random", "short", verify_http=True, ensure_http_headers=True, seed=3),
)
# Configurations for synthetic and warcio's test files (small): every option and opener.
SYNTH_CONFIGS: tuple[Config, ...] = (
    *CC_CONFIGS,
    cfg("content/datatrove", "content", "datatrove"),
    cfg("random2/bytesio", "random", "bytesio", seed=4),
    cfg("random3/logged", "random", "logged", seed=5, block_size=100),
    cfg("stop/file", "stop"),
    cfg("digests-true/file", "random", check_digests=True, seed=6),
    cfg("digests-raise/file", "content", check_digests="raise"),
    cfg("digests-raise-random/gzipfile", "random", "gzipfile", check_digests="raise", seed=7),
    cfg("verify-http/file", "content", verify_http=True),
    cfg("ensure-http/file", "offsets", ensure_http_headers=True),
    cfg("arc2warc-random/file", "random", arc2warc=True, seed=8),
    cfg("block-1/bytesio", "content", "bytesio", block_size=1),
    cfg("block-7/short", "blocks", "short", block_size=7),
    cfg("block-65536/file", "random", block_size=65536, seed=9),
    cfg("no-record-parse-random/file", "random", no_record_parse=True, seed=10),
    cfg("warc-iterator/file", "content", iterator="WARCIterator"),
    cfg("arc-iterator/file", "content", iterator="ARCIterator"),
    cfg("arc-iterator-arc2warc/file", "content", iterator="ARCIterator", arc2warc=True),
)
# Configurations the "big" synthetic files (up to ~10 MB decompressed) skip: one Python read per byte.
SLOW = {"block-1/bytesio", "block-7/short"}


# ------------------------------------------------------------------------------------------------ sources


class Logged:
    """A file wrapper that records every ``read(n)`` (n, bytes returned) into ``log``; seekable."""

    def __init__(self, f: Any, log: list[Event]) -> None:
        self.f = f
        self.log = log

    def read(self, n: int = -1) -> bytes:
        b = self.f.read(n)
        self.log.append(["io", "read", n, len(b)])
        return b

    def tell(self) -> int:
        return int(self.f.tell())

    def close(self) -> None:
        self.f.close()


class Unseekable:
    """``read`` only: ``tell`` raises, as on a pipe."""

    def __init__(self, f: Any) -> None:
        self.f = f

    def read(self, n: int = -1) -> bytes:
        return bytes(self.f.read(n))

    def tell(self) -> int:
        raise io.UnsupportedOperation("not seekable")

    def close(self) -> None:
        self.f.close()


class Short:
    """``read(n)`` returns at most a seeded number of bytes (1..k), like a socket; ``tell`` works."""

    def __init__(self, f: Any, seed: int) -> None:
        self.f = f
        self.rng = random.Random(seed)

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            return bytes(self.f.read(n))
        return bytes(self.f.read(min(n, self.rng.choice((1, 3, 100, 1000, 5000, 16384)))))

    def tell(self) -> int:
        return int(self.f.tell())

    def close(self) -> None:
        self.f.close()


def datatrove_open(path: Path, compression: str | None = "infer") -> Any:
    """What datatrove 0.10.1's WarcReader opens: its DataFolder over the file's directory."""
    from datatrove.io import get_datafolder

    return get_datafolder(str(path.parent)).open(path.name, "rb", compression=compression)


@contextlib.contextmanager
def opened(path: Path, cfg: Config, log: list[Event]) -> Iterator[Any]:
    o = cfg.opener
    if o == "datatrove":
        with datatrove_open(path) as f:
            yield f
        return
    if o == "bytesio":
        yield io.BytesIO(path.read_bytes())
        return
    if o == "fileio":
        with open(path, "rb", buffering=0) as f:
            yield f
        return
    with open(path, "rb") as f:
        if o == "file":
            yield f
        elif o == "unseekable":
            yield Unseekable(f)
        elif o == "short":
            yield Short(f, cfg.seed)
        elif o == "logged":
            yield Logged(f, log)
        elif o == "gzipfile":
            with gzip.GzipFile(fileobj=f) as g:
                yield g
        else:
            raise ValueError(f"unknown opener {o!r}")


# ------------------------------------------------------------------------------------------------ capture


class _Handler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.got: list[tuple[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.got.append((record.levelname, record.getMessage()))


class Step:
    """Run one call with ``sys.stderr`` and logging captured; ``events`` gets the outcome and the messages."""

    def __init__(self, events: list[Event], exc_name: Callable[[BaseException], str]) -> None:
        self.events = events
        self.exc_name = exc_name
        self.handler = _Handler()
        self.last = -1

    def __call__(
        self, label: str, fn: Callable[[], Any], show: Callable[[Any], Any] | None = None
    ) -> tuple[bool, Any]:
        """(ok, value); on an exception, (False, exception). StopIteration is an outcome like any other."""
        buf = io.StringIO()
        old = sys.stderr
        root = logging.getLogger()
        root.addHandler(self.handler)
        self.handler.got = []
        sys.stderr = buf
        try:
            value = fn()
            ok = True
        except BaseException as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            value, ok = e, False
        finally:
            sys.stderr = old
            root.removeHandler(self.handler)
        self.last = len(self.events)
        if ok:
            self.events.append([label, "ok", None if show is None else show(value)])
        else:
            self.events.append([label, "raise", *describe_exc(value, self.exc_name)])
        if text := buf.getvalue():
            self.events.append(["stderr", text])
        for level, msg in self.handler.got:
            self.events.append(["log", level, msg])
        return ok, value


def describe_exc(e: BaseException, exc_name: Callable[[BaseException], str]) -> list[Any]:
    return [
        exc_name(e),
        str(e),
        repr(getattr(e, "msg", None)) if exc_name(e) == "ArchiveLoadFailed" else None,
    ]


def exc_namer(*own: str) -> Callable[[BaseException], str]:
    """Exception names that do not depend on which package defines warcio's classes.

    Classes from warcio or from ``own`` packages compare by class name (``ArchiveLoadFailed``,
    ``StatusAndHeadersParserException``); everything else by module and qualified name (``zlib.error``,
    ``gzip.BadGzipFile``, ``builtins.EOFError``).
    """
    tops = {"warcio", *own}

    def name(e: BaseException) -> str:
        t = type(e)
        if t.__module__.split(".")[0] in tops:
            return t.__qualname__
        return f"{t.__module__}.{t.__qualname__}"

    return name


def chunk(b: Any) -> list[Any]:
    if not isinstance(b, bytes | bytearray):
        return ["not-bytes", type(b).__name__, repr(b)[:80]]
    return [len(b), hashlib.sha1(b).hexdigest()[:16], repr(bytes(b[:PREFIX]))]


# ------------------------------------------------------------------------------------------------ records


def _masked_headers(h: Any, mask: bool) -> list[tuple[Any, Any]]:
    hs = list(h.headers)
    if (
        mask
        and len(hs) >= 3
        and hs[0][0] == "WARC-Type"
        and hs[1][0] == "WARC-Record-ID"
        and hs[2][0] in ("WARC-Target-URI", "WARC-Filename")
    ):
        rid = hs[1][1]
        hs[1] = (hs[1][0], MASK if isinstance(rid, str) and UUID4.fullmatch(rid) else rid)
    return hs


def headers_view(h: Any, mask: bool = False) -> list[Any] | None:
    """Everything about a StatusAndHeaders: type name, protocol, statusline, headers, total_len, bool, to_str."""
    if h is None:
        return None
    hs = _masked_headers(h, mask)
    s = h.to_str()
    if mask and len(hs) > 1 and hs[1][1] == MASK and isinstance(s, str):
        s = s.replace(str(h.headers[1][1]), MASK, 1)
    return [
        type(h).__name__,
        repr(h.protocol),
        repr(h.statusline),
        repr(type(h.headers).__name__),
        repr(hs),
        repr(h.total_len),
        bool(h),
        repr(s),
    ]


def record_view(rec: Any, mask: bool) -> list[Any]:
    dc = getattr(rec, "digest_checker", None)
    return [
        repr(rec.format),
        repr(rec.rec_type),
        headers_view(rec.rec_headers, mask),
        headers_view(rec.http_headers),
        repr(rec.content_type),
        repr(rec.length),
        repr(rec.payload_length),
        None if dc is None else [repr(dc.passed), repr(list(dc.problems))],
        type(rec).__name__,
    ]


def _lookup(rec: Any, name: str, mask: bool) -> Any:
    v = rec.rec_headers.get_header(name)
    if mask and isinstance(v, str) and UUID4.fullmatch(v) and name.lower() == "warc-record-id":
        v = MASK
    return [
        repr(v),
        name in rec.rec_headers,
        repr(rec.rec_headers.get(name, "dflt") if v is None else "found"),
    ]


def _http_lookup(rec: Any, name: str) -> Any:
    h = rec.http_headers
    if h is None:
        return None
    return [repr(h.get_header(name)), repr(h.get_statuscode())]


# ------------------------------------------------------------------------------------------------ the run


def iter_events(
    mod: Any,
    path: Path,
    cfg: Config,
    limit: int = 0,
    exc_name: Callable[[BaseException], str] | None = None,
    sink: Callable[[list[Event]], None] | None = None,
) -> list[Event]:
    """The trace of one run of ``mod`` (a namespace with warcio's ``ArchiveIterator`` & co.) over ``path``.

    With ``sink``, the events are handed over record by record (one call per record, then one for the end) and
    the returned list is empty: memory stays flat on whole Common Crawl files.
    """
    events: list[Event] = []
    step = Step(events, exc_name or exc_namer())
    mask = bool(cfg.kwargs.get("arc2warc"))
    io_log: list[Event] = []
    with opened(path, cfg, io_log) as src:
        cls = getattr(mod, cfg.iterator)
        ok, it = step("init", lambda: cls(src, **cfg.kwargs))
        if not ok:
            return flush(events, sink)
        ok, gen = step("iter", lambda: iter(it))
        if not ok:
            return flush(events, sink)

        def tell() -> None:
            if hasattr(src, "tell"):
                try:
                    events.append(["io", "tell", src.tell()])
                except (OSError, ValueError):
                    events.append(["io", "tell", "unsupported"])
            events.extend(io_log)
            io_log.clear()

        i = 0
        while True:
            ok, rec = step("next", lambda: next(gen), lambda r: None)
            if not ok:
                if isinstance(rec, StopIteration):
                    events[step.last] = ["next", "stop"]
                else:
                    step("next-after-raise", lambda: next(gen), lambda r: "record")
                    if events[step.last][:3] == ["next-after-raise", "raise", "builtins.StopIteration"]:
                        events[step.last] = ["next-after-raise", "stop"]
                break
            events[step.last] = ["record", i, *record_view(rec, mask)]
            events.append(["state", repr(getattr(it, "offset", None)), repr(getattr(it, "err_count", None))])
            tell()
            run_plan(step, it, rec, cfg, i, mask)
            i += 1
            if sink is not None:
                sink(events)
                events.clear()
            if (limit and i >= limit) or (cfg.plan == "stop" and i >= 2):
                break
        events.append(["end", repr(getattr(it, "offset", None)), repr(getattr(it, "err_count", None))])
        tell()
        step("close", lambda: it.close())
    return flush(events, sink)


def flush(events: list[Event], sink: Callable[[list[Event]], None] | None) -> list[Event]:
    if sink is not None:
        sink(events)
        events.clear()
    return events


def run_plan(step: Step, it: Any, rec: Any, cfg: Config, i: int, mask: bool) -> None:
    plan = cfg.plan
    if plan == "iterate":
        return
    if plan in ("content", "stop"):
        step("cs.read", lambda: rec.content_stream().read(), chunk)
        return
    if plan == "blocks":
        ok, cs = step("cs.new", rec.content_stream, lambda s: type(s).__name__)
        n = 0
        while ok:
            ok, b = step("cs.read", lambda: cs.read(16384), chunk)
            n += 1
            if not ok or not b or n > 100_000:
                break
        return
    if plan == "raw":
        step("raw.read", lambda: rec.raw_stream.read(), chunk)
        return
    if plan == "offsets":
        step("offset", it.get_record_offset, repr)
        step("length", it.get_record_length, repr)
        return
    if plan != "random":
        raise ValueError(f"unknown plan {plan!r}")
    rng = random.Random(cfg.seed * 1_000_003 + i)
    cs: list[Any] = []
    for _ in range(rng.randint(0, 6)):
        op = rng.choice(
            (
                "raw.read",
                "raw.read",
                "raw.readline",
                "cs.read",
                "cs.read",
                "cs.readline",
                "cs.new",
                "raw.tell",
                "read_to_end",
                "offset",
                "length",
                "digest",
                "lookup",
                "http-lookup",
            )
        )
        n = rng.choice(SIZES)
        if op in ("cs.read", "cs.readline", "cs.new") and (op == "cs.new" or not cs):
            ok, s = step("cs.new", rec.content_stream, lambda s: type(s).__name__)
            cs[:] = [s] if ok else []
            if op == "cs.new" or not ok:
                continue
        if op == "raw.read":
            step(f"raw.read({n})", lambda n=n: rec.raw_stream.read(n), chunk)
        elif op == "raw.readline":
            step(f"raw.readline({n})", lambda n=n: rec.raw_stream.readline(n), chunk)
        elif op == "cs.read":
            step(f"cs.read({n})", lambda n=n: cs[0].read(n), chunk)
        elif op == "cs.readline":
            step(f"cs.readline({n})", lambda n=n: cs[0].readline(n), chunk)
        elif op == "raw.tell":
            step("raw.tell", lambda: rec.raw_stream.tell(), repr)
        elif op == "read_to_end":
            step("read_to_end", it.read_to_end, repr)
        elif op == "offset":
            step("offset", it.get_record_offset, repr)
        elif op == "length":
            step("length", it.get_record_length, repr)
        elif op == "digest":
            dc = rec.digest_checker
            step("digest", lambda dc=dc: [repr(dc.passed), repr(list(dc.problems))])
        elif op == "lookup":
            name = rng.choice(LOOKUPS)
            step(f"lookup({name})", lambda name=name: _lookup(rec, name, mask))
        else:
            name = rng.choice(HTTP_LOOKUPS)
            step(f"http-lookup({name})", lambda name=name: _http_lookup(rec, name))


# ------------------------------------------------------------------------------------------------ compare


def split(trace: Sequence[Event]) -> tuple[list[Event], list[Event]]:
    """(main events, io events)."""
    main, io_ = [], []
    for e in trace:
        (io_ if e and e[0] == "io" else main).append(e)
    return main, io_


def canon(e: Event) -> str:
    return json.dumps(e, ensure_ascii=False, separators=(",", ":"))


def digest(events: Sequence[Event]) -> str:
    h = hashlib.sha256()
    for e in events:
        h.update(canon(e).encode("utf-8", "surrogatepass"))
        h.update(b"\n")
    return h.hexdigest()[:20]


@dataclass(frozen=True)
class Diff:
    index: int
    got: Event | None
    want: Event | None
    context: list[Event]

    def short(self) -> dict[str, Any]:
        return {"at": self.index, "got": self.got, "want": self.want, "before": self.context[-3:]}


def first_diff(got: Sequence[Event], want: Sequence[Event]) -> Diff | None:
    for i in range(max(len(got), len(want))):
        g = got[i] if i < len(got) else None
        w = want[i] if i < len(want) else None
        if g != w:
            return Diff(i, g, w, list(want[max(0, i - 5) : i]))
    return None


def outline(events: Sequence[Event]) -> dict[str, int]:
    """Counts for a report line: records, reads, exceptions, messages."""
    out = {"records": 0, "raise": 0, "stderr": 0, "log": 0}
    for e in events:
        if e[0] == "record":
            out["records"] += 1
        elif len(e) > 1 and e[1] == "raise":
            out["raise"] += 1
        elif e[0] in ("stderr", "log"):
            out[e[0]] += 1
    return out
