"""A start-up probe for the consumer runs: which sqlparse each process used, and how much.

On ``PYTHONPATH`` (after pyroid's hook directory, whose ``sitecustomize`` runs this one next),
it appends one JSON line per process to ``$PROBE_LOG`` at exit: the pid, ``sys.argv``, the
module ``sqlparse.format`` came from (``None`` if sqlparse was never imported) and, when
``$PYROID_PATCH`` is set, the number of calls that reached each entry point of pyroid.sqlparse's
native core. Forked workers (``multiprocessing`` fork pools, Django's parallel runner) write
their own line too.

With ``$PROBE_CAPTURE`` set (and no patch), it also wraps ``sqlparse.lexer.tokenize`` and
``sqlparse.format`` to append the distinct inputs they get (and format's options) to that file
as JSON lines, for ``replay.py``.
"""

import atexit
import json
import os
import sys

_LOG = os.environ.get("PROBE_LOG")
_COUNTS = {}


def _count(name, fn):
    def call(*args, **kwargs):
        _COUNTS[name] = _COUNTS.get(name, 0) + 1
        return fn(*args, **kwargs)

    return call


def _wrap():
    try:
        from pyroid._pyroid import sqlparse as native
    except Exception:  # noqa: BLE001 - another interpreter: nothing to count
        return
    for name in dir(native):
        fn = getattr(native, name)
        if name.startswith("_") and not name.startswith("__") and name != "_configure" and callable(fn):
            setattr(native, name, _count(name, fn))


def _write():
    if _LOG is None or _COUNTS.get("_written"):
        return
    _COUNTS["_written"] = 1
    sp = sys.modules.get("sqlparse")
    fmt = getattr(getattr(sp, "format", None), "__module__", None)
    line = {
        "pid": os.getpid(),
        "argv": list(sys.argv[:4]),
        "format": fmt,
        "counts": {k: v for k, v in _COUNTS.items() if k != "_written"},
    }
    with open(_LOG, "a") as f:
        f.write(json.dumps(line) + "\n")


def _after_fork(_):
    import multiprocessing.util as mu

    _COUNTS.clear()
    mu.Finalize(None, _write, exitpriority=100)  # forked workers leave through os._exit


def _capture(path):
    """Record the text every sqlparse.lexer.tokenize call gets, and format's options."""
    import functools

    import sqlparse.lexer

    import sqlparse

    seen = set()

    def record(fn, sql, options=None):
        if not isinstance(sql, str):
            sql = f"<{type(sql).__name__}>"
        key = (fn, sql, json.dumps(options, sort_keys=True, default=repr))
        if key in seen:
            return
        seen.add(key)
        with open(path, "a") as f:
            f.write(json.dumps({"fn": fn, "sql": sql, "options": options}, default=repr) + "\n")

    tokenize, fmt = sqlparse.lexer.tokenize, sqlparse.format

    @functools.wraps(tokenize)
    def tokenize_(sql, encoding=None):
        record("tokenize", sql)
        return tokenize(sql, encoding)

    @functools.wraps(fmt)
    def format_(sql, encoding=None, **options):
        record("format", sql, options)
        return fmt(sql, encoding, **options)

    sqlparse.lexer.tokenize, sqlparse.format = tokenize_, format_


class _Anchor:
    pass


_ANCHOR = _Anchor()
if os.environ.get("PROBE_CAPTURE") and not os.environ.get("PYROID_PATCH"):
    _capture(os.environ["PROBE_CAPTURE"])
if _LOG:
    if os.environ.get("PYROID_PATCH"):
        _wrap()
    atexit.register(_write)
    try:
        import multiprocessing.util as _mu

        _mu.register_after_fork(_ANCHOR, _after_fork)
    except Exception:  # noqa: BLE001, S110 - the probe must never break the process it watches
        pass
