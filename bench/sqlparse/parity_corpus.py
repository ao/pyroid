#!/usr/bin/env python3
"""pyroid.sqlparse against sqlparse 0.6.1 on a real-SQL corpus (bench/sqlparse/corpus.py's JSONL).

    python bench/sqlparse/parity_corpus.py /target/sqlcorpus/corpus.jsonl --out /target/sqlparity
    python bench/sqlparse/parity_corpus.py CORPUS --out DIR --offset 50000 --limit 50000
    python bench/sqlparse/parity_corpus.py --summarize DIR

Every document goes through both libraries: ``lexer.tokenize`` (token type identity and value),
``parse`` (each tree field by field: class, ttype, value, normalized, the is_* flags, parent
links by position, children; and ``str`` of each statement), ``split``, ``split`` with
``strip_semicolon=True``, and ``format`` under the option sets in ``FORMAT_SETS``. Exceptions
compare by class, message and ``__cause__``. Each call has a time limit (``--timeout``,
SIGALRM): a document sqlparse cannot finish in time is skipped for that check, one pyroid
cannot finish counts as a mismatch. Documents run in ``--workers`` processes (each imports
both libraries); with ``--max-seconds``, a task still running well past it is reported as
unfinished (its documents are listed for a rerun).

The run prints, per check, the documents compared and the mismatches, then the mismatch
classes (check and first-difference path) with the first documents of each, and minimises the
first ``--minimize`` documents of each class (lines, then characters, while the same class
persists) into reproducers. ``DIR/results-OFFSET.jsonl`` keeps one line per document and
``DIR/repro-OFFSET.json`` the reproducers; ``--summarize DIR`` merges the slices of several
runs. Timings are a debug or release build's, as installed, on whatever machine runs this.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import multiprocessing as mp
import os
import re
import signal
import sys
import time
import warnings
from typing import Any

FORMAT_SETS: dict[str, dict[str, Any]] = {
    "default": {},
    "reindent": {"reindent": True},
    "reindent_aligned": {"reindent_aligned": True},
    "case": {"keyword_case": "upper", "identifier_case": "lower"},
    "capitalize_strip_ws": {"keyword_case": "capitalize", "strip_whitespace": True},
    "strip_comments": {"strip_comments": True},
    "spaces_around_ops": {"use_space_around_operators": True},
    "comma_first_wrap": {"reindent": True, "comma_first": True, "wrap_after": 40},
    "tabs_columns_after_first": {
        "reindent": True,
        "indent_tabs": True,
        "indent_columns": True,
        "indent_after_first": True,
    },
    "compact_width4": {"reindent": True, "compact": True, "indent_width": 4},
    "python": {"output_format": "python", "reindent": True},
    "php": {"output_format": "php"},
    "truncate": {"truncate_strings": 10, "truncate_char": "..."},
}
CHECKS = ["tokenize", "parse", "split", "split_strip", *(f"format:{k}" for k in FORMAT_SETS)]

PS: Any = None
SP: Any = None
SHARED = False


class Timeout(BaseException):
    """The per-call time limit (a BaseException, so library code cannot swallow it)."""


def _alarm(signum: int, frame: Any) -> None:
    raise Timeout


def init(timeout: float) -> None:
    """Import both libraries (in each worker)."""
    global PS, SP, SHARED, TIMEOUT
    import pyroid.sqlparse as ps

    import sqlparse as sp

    if sp.__version__ != "0.6.1":
        raise SystemExit(f"needs sqlparse 0.6.1, not {sp.__version__}")
    PS, SP = ps, sp
    SHARED = ps.tokens.Token is sp.tokens.Token
    TIMEOUT = timeout
    warnings.simplefilter("ignore")
    signal.signal(signal.SIGALRM, _alarm)


TIMEOUT = 20.0


# ---- outcomes (as in python/tests/test_sqlparse_parity.py) --------------------------------------


def tt_key(tt: Any) -> Any:
    if tt is None:
        return None
    return (repr(tt), id(tt)) if SHARED else repr(tt)


def cls_key(cls: type) -> Any:
    if SHARED or cls.__module__ == "builtins":
        return (cls.__module__, cls.__qualname__, id(cls))
    return cls.__qualname__


def exc_key(e: BaseException) -> tuple[Any, ...]:
    c = e.__cause__
    return ("raise", cls_key(type(e)), str(e), None if c is None else (type(c).__qualname__, str(c)))


def tree(stmt: Any) -> list[tuple[Any, ...]]:
    order, where, stack = [], {}, [((), stmt)]
    while stack:
        path, tok = stack.pop()
        where[id(tok)] = path
        order.append((path, tok))
        kids = getattr(tok, "tokens", None)
        if isinstance(kids, list):
            stack.extend(((*path, i), kids[i]) for i in reversed(range(len(kids))))
    out = []
    for path, tok in order:
        p = tok.parent
        pp = None if p is None else where.get(id(p), ("outside", type(p).__name__, str(p)))
        kids = getattr(tok, "tokens", None)
        out.append(
            (
                path,
                cls_key(type(tok)),
                tt_key(tok.ttype),
                tok.value,
                tok.normalized,
                tok.is_keyword,
                tok.is_group,
                tok.is_whitespace,
                tok.is_newline,
                pp,
                len(kids) if isinstance(kids, list) else None,
            )
        )
    return out


def outcome(lib: Any, check: str, text: str) -> Any:
    try:
        if check == "tokenize":
            items = []
            try:
                items.extend((tt_key(t), v) for t, v in lib.lexer.tokenize(text))
            except Exception as e:  # noqa: BLE001 (compared)
                return ("items", items, exc_key(e))
            return ("ok", items)
        if check == "parse":
            return ("ok", [(str(s), tree(s)) for s in lib.parse(text)])
        if check == "split":
            return ("ok", lib.split(text))
        if check == "split_strip":
            return ("ok", lib.split(text, strip_semicolon=True))
        return ("ok", lib.format(text, **FORMAT_SETS[check.split(":", 1)[1]]))
    except Exception as e:  # noqa: BLE001 (compared)
        return exc_key(e)


def timed(lib: Any, check: str, text: str, timeout: float) -> tuple[Any, float]:
    t0 = time.perf_counter()
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        r = outcome(lib, check, text)
    except Timeout:
        r = ("timeout",)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return r, time.perf_counter() - t0


def first_diff(a: Any, b: Any, path: str = "") -> str:
    if isinstance(a, tuple | list) and isinstance(b, tuple | list) and type(a) is type(b):
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            if x != y:
                return first_diff(x, y, f"{path}[{i}]")
        if len(a) != len(b):
            return f"{path}: length {len(a)} (pyroid) != {len(b)} (sqlparse)"
    return f"{path or '<top>'}: pyroid {a!r:.300} | sqlparse {b!r:.300}"


def diff_class(check: str, diff: str) -> str:
    """A mismatch class: the check and the first-difference path without indices."""
    head = diff.split(":", 1)[0]
    if "timeout" in diff.split("|")[0] and "pyroid ('timeout',)" in diff:
        return f"{check} | pyroid timeout"
    return f"{check} | {re.sub(r'[0-9]+', '#', head)}"


def compare(check: str, text: str, timeout: float) -> tuple[str, str | None, float, float]:
    """(status, first difference, sqlparse seconds, pyroid seconds) of one check."""
    want, ts = timed(SP, check, text, timeout)
    if want == ("timeout",):
        return "sqlparse-timeout", None, ts, 0.0
    got, tp = timed(PS, check, text, timeout)
    if got == want:
        return "same", None, ts, tp
    if got == ("timeout",):
        return "pyroid-timeout", "pyroid ('timeout',)", ts, tp
    return "diff", first_diff(got, want)[:2000], ts, tp


def run_doc(doc: dict[str, Any]) -> dict[str, Any]:
    text = doc["text"]
    res: dict[str, Any] = {"id": doc["id"], "source": doc["source"], "kind": doc["kind"]}
    res["len"] = len(text)
    checks: dict[str, str] = {}
    diffs: dict[str, str] = {}
    ts = tp = 0.0
    slow = 0
    for check in CHECKS:
        if slow >= 2:
            checks[check] = "skipped"
            continue
        status, diff, a, b = compare(check, text, TIMEOUT)
        checks[check] = status
        ts += a
        tp += b
        if status == "sqlparse-timeout":
            slow += 1
        if diff is not None:
            diffs[check] = diff
    res.update(checks=checks, diffs=diffs, t_sqlparse=round(ts, 4), t_pyroid=round(tp, 4))
    return res


def run_chunk(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [run_doc(d) for d in docs]


# ---- minimising a mismatch ---------------------------------------------------------------------


def still(check: str, cls: str, text: str, timeout: float) -> bool:
    status, diff, _, _ = compare(check, text, timeout)
    return status in ("diff", "pyroid-timeout") and diff is not None and diff_class(check, diff) == cls


def ddmin(items: list[str], keep: Any, deadline: float) -> list[str]:
    """Delta debugging: a 1-minimal sublist of `items` for which `keep` holds (time-boxed)."""
    n = 2
    while len(items) >= 2 and time.monotonic() < deadline:
        size = max(1, len(items) // n)
        reduced = False
        for i in range(0, len(items), size):
            cand = items[:i] + items[i + size :]
            if cand and keep(cand):
                items, n, reduced = cand, max(n - 1, 2), True
                break
            if time.monotonic() > deadline:
                return items
        if not reduced:
            if size == 1:
                break
            n = min(len(items), n * 2)
    return items


def minimise(check: str, cls: str, text: str, budget: float, timeout: float) -> str:
    deadline = time.monotonic() + budget
    lines = text.splitlines(keepends=True)
    lines = ddmin(lines, lambda c: still(check, cls, "".join(c), timeout), deadline)
    chars = ddmin(list("".join(lines)), lambda c: still(check, cls, "".join(c), timeout), deadline)
    return "".join(chars)


def repro_call(check: str, text: str) -> str:
    """The call that shows a mismatch of `check` on `text`."""
    if check == "tokenize":
        return f"list(lexer.tokenize({text!r}))"
    if check == "split_strip":
        return f"split({text!r}, strip_semicolon=True)"
    if check.startswith("format:"):
        return f"format({text!r}, **{FORMAT_SETS[check.split(':', 1)[1]]!r})"
    return f"{check}({text!r})"


# ---- the run -----------------------------------------------------------------------------------


def load(path: str, offset: int, limit: int, sources: set[str] | None) -> list[dict[str, Any]]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if limit and len(docs) >= limit:
                break
            d = json.loads(line)
            if sources is None or d["source"] in sources:
                d["line"] = i
                docs.append(d)
    return docs


def report(results: list[dict[str, Any]], repros: list[dict[str, Any]], out: Any) -> None:
    counts = {c: {} for c in CHECKS}
    classes: dict[str, list[dict[str, Any]]] = {}
    per_source: dict[str, int] = {}
    ts = tp = 0.0
    for r in results:
        per_source[r["source"]] = per_source.get(r["source"], 0) + 1
        ts += r.get("t_sqlparse", 0.0)
        tp += r.get("t_pyroid", 0.0)
        for c, s in r["checks"].items():
            counts[c][s] = counts[c].get(s, 0) + 1
        for c, d in r["diffs"].items():
            classes.setdefault(diff_class(c, d), []).append({"id": r["id"], "len": r["len"], "diff": d})
    print(f"documents: {len(results)}", file=out)
    print("per source: " + ", ".join(f"{k} {v}" for k, v in sorted(per_source.items())), file=out)
    print(f"seconds in calls: sqlparse {ts:.1f}, pyroid {tp:.1f} (as installed)", file=out)
    print(
        f"{'check':32} {'compared':>9} {'same':>9} {'diff':>6} {'pyroid-timeout':>15} "
        f"{'sqlparse-timeout':>17} {'skipped':>8}",
        file=out,
    )
    for c in CHECKS:
        k = counts[c]
        compared = k.get("same", 0) + k.get("diff", 0) + k.get("pyroid-timeout", 0)
        print(
            f"{c:32} {compared:9} {k.get('same', 0):9} {k.get('diff', 0):6} "
            f"{k.get('pyroid-timeout', 0):15} {k.get('sqlparse-timeout', 0):17} "
            f"{k.get('skipped', 0):8}",
            file=out,
        )
    print(f"\nmismatch classes: {len(classes)}", file=out)
    for cls, items in sorted(classes.items(), key=lambda kv: -len(kv[1])):
        items.sort(key=lambda e: e["len"])
        print(
            f"\n== {cls}: {len(items)} documents; shortest: "
            + ", ".join(f"{e['id']} ({e['len']} chars)" for e in items[:5]),
            file=out,
        )
        print(f"   {items[0]['diff'][:600]}", file=out)
        for rp in repros:
            if rp["class"] == cls:
                print(
                    f"   reproducer ({rp['doc']}, {rp['original_len']} -> "
                    f"{len(rp['input'])} chars): {rp['call']}",
                    file=out,
                )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("corpus", nargs="?", help="corpus.jsonl from bench/sqlparse/corpus.py")
    ap.add_argument("--out", required=True, help="results directory")
    ap.add_argument("--offset", type=int, default=0, help="first line of the corpus to run")
    ap.add_argument("--limit", type=int, default=0, help="lines to run (0: all)")
    ap.add_argument("--sources", help="comma-separated sources to run (default: all)")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--timeout", type=float, default=20.0, help="seconds per call (20)")
    ap.add_argument("--chunk", type=int, default=8, help="documents per worker task (8)")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop submitting after this")
    ap.add_argument("--minimize", type=int, default=2, help="reproducers per class (2)")
    ap.add_argument("--minimize-seconds", type=float, default=30.0, help="budget per reproducer")
    ap.add_argument("--summarize", action="store_true", help="merge DIR's slices and report")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    if args.summarize:
        results, repros = [], []
        for p in sorted(glob.glob(os.path.join(args.out, "results-*.jsonl"))):
            with open(p, encoding="utf-8") as f:
                results += [json.loads(line) for line in f]
        for p in sorted(glob.glob(os.path.join(args.out, "repro-*.json"))):
            with open(p, encoding="utf-8") as f:
                repros += json.load(f)
        report(results, repros, sys.stdout)
        return 0
    if not args.corpus:
        ap.error("the corpus path is required")
    docs = load(args.corpus, args.offset, args.limit, set(args.sources.split(",")) if args.sources else None)
    t0 = time.monotonic()
    chunks: list[list[dict[str, Any]]] = [[]]
    size = 0
    for d in docs:  # at most --chunk documents or about 20,000 characters per task
        if chunks[-1] and (len(chunks[-1]) >= args.chunk or size + len(d["text"]) > 20000):
            chunks.append([])
            size = 0
        chunks[-1].append(d)
        size += len(d["text"])
    chunks = [c for c in chunks if c]
    results: list[dict[str, Any]] = []
    ctx = mp.get_context("fork")
    pool = ctx.Pool(args.workers, initializer=init, initargs=(args.timeout,))
    pending = []
    stopped = False
    for ch in chunks:
        if args.max_seconds and time.monotonic() - t0 > args.max_seconds:
            stopped = True
            break
        pending.append((ch, pool.apply_async(run_chunk, (ch,)), None))
        while len([p for p in pending if not p[1].ready()]) >= args.workers + 2:
            time.sleep(0.05)
    # Past the deadline a task still running counts as unfinished (a slow document or a hang;
    # rerun its documents alone to tell).
    deadline = t0 + args.max_seconds + 4 * args.timeout + 30 if args.max_seconds else None
    unfinished: list[dict[str, Any]] = []
    for ch, res, _ in pending:
        try:
            if deadline is None:
                results += res.get()
            else:
                results += res.get(timeout=max(0.1, deadline - time.monotonic()))
        except mp.TimeoutError:
            unfinished += ch
    pool.terminate()
    elapsed = time.monotonic() - t0
    tag = f"{args.offset:08d}"
    with open(os.path.join(args.out, f"results-{tag}.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=True) + "\n" for r in results)
    # Reproducers: the shortest documents of each class, minimised in this process.
    init(min(args.timeout, 10.0))
    by_id = {d["id"]: d for d in docs}
    classes: dict[str, list[tuple[int, str, str]]] = {}
    for r in results:
        for c, d in r["diffs"].items():
            if c in CHECKS:
                classes.setdefault(diff_class(c, d), []).append((r["len"], r["id"], c))
    repros = []
    for cls, items in classes.items():
        for _, doc_id, check in sorted(items)[: args.minimize]:
            text = by_id[doc_id]["text"]
            small = minimise(check, cls, text, args.minimize_seconds, TIMEOUT)
            call = repro_call(check, small)
            got, want = (outcome(PS, check, small), outcome(SP, check, small))
            repros.append(
                {
                    "class": cls,
                    "doc": doc_id,
                    "check": check,
                    "original_len": len(text),
                    "input": small,
                    "call": call,
                    "diff": first_diff(got, want)[:1500],
                }
            )
    with open(os.path.join(args.out, f"repro-{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(repros, f, indent=1, ensure_ascii=True)
    buf = io.StringIO()
    report(results, repros, buf)
    print(buf.getvalue())
    nxt = pending[-1][0][-1]["line"] + 1 if pending else args.offset
    if unfinished:
        with open(os.path.join(args.out, f"unfinished-{tag}.json"), "w", encoding="utf-8") as f:
            json.dump([{"id": d["id"], "line": d["line"], "len": len(d["text"])} for d in unfinished], f)
        print(
            f"unfinished: {len(unfinished)} documents (see unfinished-{tag}.json), first ids: "
            + ", ".join(d["id"] for d in unfinished[:10])
        )
    print(
        f"run: {len(results)} of {len(docs)} documents in {elapsed:.0f} s, {args.workers} workers"
        + (f" (stopped at --max-seconds; continue with --offset {nxt})" if stopped else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
