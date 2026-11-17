"""pyroid.patch("sqlparse"): pyroid's functions on sqlparse's own modules, or shims without it.

With sqlparse installed, ``patch("sqlparse")`` sets ``parse``, ``parsestream``, ``format`` and
``split`` on ``sqlparse``, ``tokenize`` on ``sqlparse.lexer`` and ``group`` on
``sqlparse.engine.grouping``, so code calling ``sqlparse.format(...)`` (Django, the debug
toolbar's FilterStack) runs natively, and ``unpatch`` puts sqlparse's functions back. Without
sqlparse, ``sqlparse`` and its submodules are pyroid.sqlparse's modules. Patching is
process-wide state, so every test runs in a fresh interpreter.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pytest

from pyroid.integrations import _patch

pytestmark = pytest.mark.thread_unsafe(reason="subprocesses and process-wide import state")


def _sqlparse_version() -> str | None:
    try:
        return version("sqlparse")
    except PackageNotFoundError:
        return None


# The patched functions give sqlparse 0.6.1's results: compare with that release only.
needs_sqlparse = pytest.mark.skipif(
    _sqlparse_version() != "0.6.1", reason="needs sqlparse 0.6.1 installed"
)

# `python -m pyroid run`: the pyroid command line imports pyarrow before it gets to `run`.
needs_cli = pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None,
    reason="the pyroid command line needs pyarrow (pyroid[cli])",
)

# Environment that would leak a patch (or a warning filter) into the children.
_CLEAN = ("PYTHONPATH", "PYROID_PATCH", "_PYROID_PATCH_WARNED", "PYTHONWARNINGS", "TEST_HIDE")

# Prepended to every child. TEST_HIDE=1 makes an installed sqlparse look uninstalled: only the
# top-level name is hidden, so submodules of pyroid's stand-in (sqlparse.__main__) still import.
PRELUDE = """\
import importlib, importlib.abc, importlib.util, json, os, pickle, sys, warnings
class _Hide(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "sqlparse":
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None
if os.environ.get("TEST_HIDE"):
    sys.meta_path.insert(0, _Hide())
def patch_quietly(*names):
    import pyroid.integrations as pi
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pi.patch(*names)
SWAPS = {
    "sqlparse": ("parse", "parsestream", "format", "split"),
    "sqlparse.lexer": ("tokenize",),
    "sqlparse.engine.grouping": ("group",),
}
def current():
    return {m: {n: getattr(sys.modules[m], n) for n in ns} for m, ns in SWAPS.items()}
def all_pyroid():
    from pyroid.sqlparse import _patched
    return all(f is getattr(_patched, n) for fs in current().values() for n, f in fs.items())
SQLS = [
    "select a, b as c, count(*) from t1 join t2 on t1.id = t2.id where x > 1 group by a;",
    "insert into t (a, b) values (1, 'x'), (2, 'y'); update t set a = a + 1 where b like 'x%'",
    "-- head\\nCREATE TABLE t (id integer PRIMARY KEY, name varchar(10) NOT NULL) /* tail */;",
    "with q as (select * from t where c in (1, 2, 3)) select case when a then 1 else 2 end from q",
    "SELECT \\"x\\".\\"y\\" FROM \\"x\\" WHERE \\"x\\".\\"z\\" = %s ORDER BY 1 DESC LIMIT 21",
    "select sum(a) over (partition by b order by c) from t having count(*) > 2",
]
def shape(tok):
    # The whole tree: classes, token types, values, and the parent links.
    if tok.is_group:
        assert all(t.parent is tok for t in tok.tokens), "parent links"
        return [type(tok).__name__, [shape(t) for t in tok.tokens]]
    return [type(tok).__name__, str(tok.ttype), tok.value]
"""


def child(
    code: str,
    *,
    args: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
    hide: bool = False,
    cwd: Path | None = None,
    timeout: float = 180.0,
) -> Any:
    """Run `code` in a fresh interpreter (not ``-I``: ``PYTHONPATH`` reaches it).

    Returns the JSON it printed last; fails the test if it did not exit with 0.
    """
    child_env = {k: v for k, v in os.environ.items() if k not in _CLEAN}
    child_env.update(env or {})
    if hide:
        child_env["TEST_HIDE"] = "1"
    cmd = [sys.executable, "-c", PRELUDE + textwrap.dedent(code), *args]
    try:
        proc = subprocess.run(
            cmd, env=child_env, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"child did not finish in {timeout} s\n{exc.stdout!r}\n{exc.stderr!r}")
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    lines = proc.stdout.strip().splitlines()
    return json.loads(lines[-1]) if lines else None


def run_script(
    argv: list[str], cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    child_env = {k: v for k, v in os.environ.items() if k not in _CLEAN}
    child_env.update(env or {})
    try:
        return subprocess.run(
            [sys.executable, *argv],
            env=child_env,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"{argv} did not finish in 300 s\n{exc.stdout!r}\n{exc.stderr!r}")


def last_json(proc: subprocess.CompletedProcess[str]) -> Any:
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# With sqlparse installed: pyroid's functions on sqlparse's modules
# ---------------------------------------------------------------------------


def test_importing_pyroid_patches_nothing() -> None:
    out = child(
        """
        import pyroid, pyroid.sqlparse, pyroid.integrations as pi
        try:
            import sqlparse
        except ImportError:
            print(json.dumps([pi.patched(), None]))
        else:
            print(json.dumps([pi.patched(), all_pyroid()]))
        """
    )
    assert out[0] == []
    assert out[1] in (False, None)


@needs_sqlparse
def test_patch_swaps_and_unpatch_restores() -> None:
    out = child(
        """
        import sqlparse, sqlparse.lexer, sqlparse.engine.grouping
        from sqlparse import format as early_format
        import pyroid.integrations as pi
        before = current()
        modules = {k: v for k, v in sys.modules.items() if k.startswith("sqlparse")}
        want = [early_format(s, reindent=True, keyword_case="upper") for s in SQLS]
        trees = [[shape(st) for st in sqlparse.parse(s)] for s in SQLS]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pi.patch("sqlparse")
            pi.patch("sqlparse")  # idempotent, warns once
        swapped = all_pyroid()
        same_modules = all(sys.modules.get(k) is v for k, v in modules.items())
        shim = getattr(sys.modules["sqlparse"], "__pyroid__", False)
        got = [sqlparse.format(s, reindent=True, keyword_case="upper") for s in SQLS]
        got_trees = [[shape(st) for st in sqlparse.parse(s)] for s in SQLS]
        stmt = sqlparse.parse(SQLS[0])[0]
        classes = [type(stmt) is sqlparse.sql.Statement,
                   isinstance(stmt.tokens[2], sqlparse.sql.IdentifierList),
                   stmt.tokens[0].ttype is sqlparse.tokens.Keyword.DML,
                   stmt.get_type()]
        early_kept = early_format is before["sqlparse"]["format"]
        patched = pi.patched()
        pi.unpatch("sqlparse")
        restored = all(
            getattr(sys.modules[m], n) is f for m, fs in before.items() for n, f in fs.items()
        )
        print(json.dumps({
            "warnings": [(w.category.__name__, str(w.message), w.filename) for w in caught],
            "swapped": swapped, "same_modules": same_modules, "shim": shim,
            "same": got == want, "trees": got_trees == trees, "classes": classes,
            "early_kept": early_kept, "patched": patched, "restored": restored,
            "after": pi.patched(),
        }))
        """
    )
    assert out["swapped"]
    assert out["same_modules"]  # no shims: sqlparse's own module objects stay in sys.modules
    assert out["shim"] is False
    assert out["same"]
    assert out["trees"]
    assert out["classes"] == [True, True, True, "SELECT"]
    assert out["early_kept"]
    assert out["patched"] == ["sqlparse"]
    assert out["restored"]
    assert out["after"] == []
    assert len(out["warnings"]) == 1, out["warnings"]
    category, message, filename = out["warnings"][0]
    assert category == "PatchWarning"
    assert filename == "<string>"  # the caller's line, not pyroid's
    assert message.startswith(
        "pyroid.integrations.patch: the installed sqlparse now uses pyroid's drop-ins "
        "(pyroid.sqlparse, which reproduces sqlparse 0.6.1's parse, parsestream, format, split"
    )
    assert "Names imported from sqlparse before this call (from sqlparse import format)" in message
    assert "pyroid.integrations.unpatch('sqlparse')" in message


@needs_sqlparse
def test_patching_context_manager() -> None:
    out = child(
        """
        import sqlparse
        import pyroid.integrations as pi
        before = current()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pi.patching("sqlparse"):
                inside = all_pyroid()
            after = current() == before
            pi.patch("sqlparse")
            with pi.patching("sqlparse"):  # already patched: the block does not own it
                pass
            kept = all_pyroid()
            pi.unpatch()
        print(json.dumps([inside, after, kept, current() == before]))
        """
    )
    assert out == [True, True, True, True]


@needs_sqlparse
def test_version_notice() -> None:
    out = child(
        """
        import sqlparse
        sqlparse.__version__ = "0.5.3"  # as if another release were installed
        import pyroid.integrations as pi
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pi.patch("sqlparse")
        print(json.dumps([(w.category.__name__, str(w.message)) for w in caught]))
        """
    )
    # pyroid.sqlparse says so when it is imported (by the patch), then the patch does.
    assert [c for c, _ in out] == ["RuntimeWarning", "PatchWarning", "PatchWarning"]
    assert out[0][1].startswith(
        "pyroid.sqlparse reproduces sqlparse 0.6.1, but sqlparse 0.5.3 is installed: pyroid "
        "shares its token types and classes"
    )
    assert out[2][1] == (
        "pyroid.integrations.patch: the installed sqlparse is 0.5.3, but pyroid.sqlparse "
        "reproduces sqlparse 0.6.1: the patched functions give sqlparse 0.6.1's results, the "
        "rest is sqlparse 0.5.3."
    )


@needs_sqlparse
def test_limits_are_read_from_sqlparse_grouping() -> None:
    """MAX_GROUPING_TOKENS / MAX_GROUPING_DEPTH set on sqlparse apply to the patched functions."""
    out = child(
        """
        import sqlparse, sqlparse.engine.grouping as grouping
        from sqlparse.exceptions import SQLParseError
        import pyroid.integrations as pi
        WIDE = "select " + ", ".join(f"a{i}" for i in range(40)) + " from t"
        DEEP = "select " + "(" * 30 + "1" + ")" * 30
        def outcome(call):
            try:
                r = call()
            except SQLParseError as e:
                return ["SQLParseError", str(e)]
            return ["ok", str(r[0]) if isinstance(r, tuple) else r]
        def stack(sql):
            s = sqlparse.engine.FilterStack()
            s.enable_grouping()
            return tuple(s.run(sql))
        def observe():
            res = []
            for tokens, depth in ((None, None), (20, None), (None, 10)):
                grouping.MAX_GROUPING_TOKENS = 10000 if tokens is None else tokens
                grouping.MAX_GROUPING_DEPTH = 100 if depth is None else depth
                for sql in (WIDE, DEEP):
                    res.append(outcome(lambda: sqlparse.parse(sql)))
                    res.append(outcome(lambda: stack(sql)))
                    res.append(outcome(lambda: sqlparse.format(sql, reindent=True)))
            grouping.MAX_GROUPING_TOKENS, grouping.MAX_GROUPING_DEPTH = 10000, 100
            return res
        want = observe()
        patch_quietly("sqlparse")
        got = observe()
        print(json.dumps([want, got]))
        """
    )
    want, got = out
    assert got == want
    assert any(o[0] == "SQLParseError" for o in want)  # the limits did bite


@needs_sqlparse
def test_trees_mix_with_sqlparses_filters_and_filterstack() -> None:
    """Trees from the patched parse go through sqlparse's own filters, and the other way round."""
    out = child(
        """
        import copy
        import sqlparse
        from sqlparse import engine, filters
        from sqlparse.engine.statement_splitter import StatementSplitter
        import pyroid.sqlparse.filters as pfilters
        orig_parse, orig_group = sqlparse.parse, sqlparse.engine.grouping.group
        orig_tokenize = sqlparse.lexer.tokenize

        def through(mod, stmts):
            out = []
            for st in stmts:
                mod.StripCommentsFilter().process(st)
                mod.StripWhitespaceFilter().process(st)
                mod.ReindentFilter(width=4, wrap_after=30).process(st)
                out.append(mod.SerializerUnicode.process(st))
            return out

        def aligned(mod, stmts):
            return [str(mod.AlignedIndentFilter().process(st)) for st in stmts]

        def toolbar_like(sql):
            # A FilterStack with Python filters, as django-debug-toolbar builds one.
            Other = sqlparse.tokens.Other
            class Bold:
                def process(self, stmt):
                    i = 0
                    while i < len(stmt.tokens):
                        tok = stmt.tokens[i]
                        if tok.is_keyword:
                            stmt.insert_before(i, sqlparse.sql.Token(Other, "<b>"))
                            end = sqlparse.sql.Token(Other, "</b>")
                            stmt.insert_after(i + 1, end, skip_ws=False)
                            i += 2
                        elif tok.is_group:
                            self.process(tok)
                        i += 1
            stack = engine.FilterStack()
            stack.enable_grouping()
            stack.stmtprocess.append(filters.AlignedIndentFilter(char="&nbsp;", n="<br/>"))
            stack.stmtprocess.append(Bold())
            stack.postprocess.append(filters.SerializerUnicode())
            return "".join(stack.run(sql))

        def regroup(group_fn, sql):
            # Ungrouped statements of sqlparse's tokens, grouped by `group_fn`.
            return [shape(group_fn(st)) for st in StatementSplitter().process(orig_tokenize(sql))]

        res = {}
        res["want_filters"] = [through(filters, orig_parse(s)) for s in SQLS]
        res["want_aligned"] = [aligned(filters, orig_parse(s)) for s in SQLS]
        res["want_toolbar"] = [toolbar_like(s) for s in SQLS]
        res["want_group"] = [regroup(orig_group, s) for s in SQLS]
        patch_quietly("sqlparse")
        assert all_pyroid()
        # pyroid's trees through sqlparse's filters.
        res["got_filters"] = [through(filters, sqlparse.parse(s)) for s in SQLS]
        res["got_aligned"] = [aligned(filters, sqlparse.parse(s)) for s in SQLS]
        # sqlparse's FilterStack on pyroid's tokenize and group.
        res["got_toolbar"] = [toolbar_like(s) for s in SQLS]
        # sqlparse's ungrouped statements grouped by pyroid.
        res["got_group"] = [regroup(sqlparse.engine.grouping.group, s) for s in SQLS]
        # sqlparse's trees through pyroid's filters, and deep copies of pyroid's trees.
        res["mixed_filters"] = [through(pfilters, orig_parse(s)) for s in SQLS]
        res["copies"] = [[shape(copy.deepcopy(st)) for st in sqlparse.parse(s)] for s in SQLS]
        res["want_copies"] = [[shape(st) for st in orig_parse(s)] for s in SQLS]
        print(json.dumps(res))
        """
    )
    assert out["got_filters"] == out["want_filters"]
    assert out["got_aligned"] == out["want_aligned"]
    assert out["got_toolbar"] == out["want_toolbar"]
    assert out["got_group"] == out["want_group"]
    assert out["mixed_filters"] == out["want_filters"]
    assert out["copies"] == out["want_copies"]


@needs_sqlparse
def test_pickling_the_patched_functions_and_their_trees() -> None:
    out = child(
        """
        import functools
        import sqlparse
        from sqlparse import format as early_format
        from pyroid.sqlparse import _patched
        patch_quietly("sqlparse")
        res = {}
        for name, fn in current()["sqlparse"].items():
            for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                assert pickle.loads(pickle.dumps(fn, proto)) is fn, (name, proto)
        res["ref"] = pickle.dumps(sqlparse.format, 4).count(b"pyroid.sqlparse._patched")
        part = pickle.loads(pickle.dumps(functools.partial(sqlparse.format, reindent=True)))
        res["partial"] = part(SQLS[0]) == early_format(SQLS[0], reindent=True)
        stmt = sqlparse.parse(SQLS[3])[0]
        back = pickle.loads(pickle.dumps(stmt))
        res["tree"] = shape(back) == shape(stmt) and type(back) is sqlparse.sql.Statement
        # A function taken from sqlparse before the patch no longer is sqlparse.format.
        try:
            pickle.dumps(early_format)
            res["early"] = "pickled"
        except pickle.PicklingError as e:
            res["early"] = "PicklingError"
        print(json.dumps(res))
        """
    )
    assert out == {"ref": 1, "partial": True, "tree": True, "early": "PicklingError"}


POOL = """\
import functools, json, multiprocessing as mp, sys, warnings
import sqlparse

SQLS = ["select a, b from t where c = 1", "insert into t values (1, 2); delete from t"]

def probe(sql):
    import sqlparse
    from pyroid.sqlparse import _patched
    return [sqlparse.format is _patched.format, sqlparse.format(sql, reindent=True)]

if __name__ == "__main__":
    want = [sqlparse.format(s, reindent=True) for s in SQLS]
    if "--patch" in sys.argv:
        import pyroid.integrations as pi
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pi.patch("sqlparse")
    out = {}
    for method in (m for m in ("spawn", "fork", "forkserver") if m in mp.get_all_start_methods()):
        with mp.get_context(method).Pool(2) as pool:
            direct = pool.map(functools.partial(sqlparse.format, reindent=True), SQLS)
            probed = pool.map(probe, SQLS)
        out[method] = {
            "direct": direct == want,
            "same": [p[1] for p in probed] == want,
            "patched": [p[0] for p in probed],
        }
    print(json.dumps(out))
"""


@needs_sqlparse
def test_pool_map_with_the_patched_functions(tmp_path: Path) -> None:
    (tmp_path / "pool.py").write_text(POOL)
    out = last_json(run_script(["pool.py", "--patch"], cwd=tmp_path))
    assert out["spawn"] == {"direct": True, "same": True, "patched": [False, False]}
    if "fork" in out:  # forked workers inherit the patch
        assert out["fork"] == {"direct": True, "same": True, "patched": [True, True]}


@needs_sqlparse
@needs_cli
def test_pool_map_under_pyroid_run(tmp_path: Path) -> None:
    # Spawned and forkserver workers are patched by the start-up hook too.
    (tmp_path / "pool.py").write_text(POOL)
    proc = run_script(["-m", "pyroid", "run", "-p", "sqlparse", "pool.py"], cwd=tmp_path)
    out = last_json(proc)
    for method, res in out.items():
        assert res == {"direct": True, "same": True, "patched": [True, True]}, method
    assert proc.stderr.count("PatchWarning") == 1, proc.stderr


CHILDREN = """\
import json, multiprocessing as mp, subprocess, sys

def probe(_=None):
    import sqlparse, sqlparse.lexer, sqlparse.engine.grouping as g
    from pyroid.sqlparse import _patched
    return [sqlparse.format is _patched.format, sqlparse.lexer.tokenize is _patched.tokenize,
            g.group is _patched.group, sqlparse.split("select 1; select 2")]

if __name__ == "__main__":
    results = {"main": probe(), "name": __name__, "argv": sys.argv[1:]}
    for method in (m for m in ("spawn", "forkserver") if m in mp.get_all_start_methods()):
        with mp.get_context(method).Pool(2) as pool:
            results[method] = pool.map(probe, range(2))
    code = "import json, sqlparse; print(json.dumps(sqlparse.format.__module__))"
    sub = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    results["subprocess"] = json.loads(sub.stdout)
    results["subprocess_stderr"] = sub.stderr
    print(json.dumps(results))
"""


@needs_sqlparse
@needs_cli
@pytest.mark.parametrize("how", ["script", "module"])
def test_pyroid_run_patches_the_target_and_its_children(tmp_path: Path, how: str) -> None:
    (tmp_path / "children.py").write_text(CHILDREN)
    target = ["children.py"] if how == "script" else ["-m", "children"]
    proc = run_script(["-m", "pyroid", "run", "-p", "sqlparse", *target, "x"], cwd=tmp_path)
    out = last_json(proc)
    probe = [True, True, True, ["select 1;", "select 2"]]
    assert out["main"] == probe
    assert out["name"] == "__main__"
    assert out["argv"] == ["x"]
    assert out["spawn"] == [probe, probe]
    if sys.platform != "win32":
        assert out["forkserver"] == [probe, probe]
    assert out["subprocess"] == "pyroid.sqlparse._patched"
    assert out["subprocess_stderr"] == ""
    assert proc.stderr.count("PatchWarning") == 1, proc.stderr


@needs_sqlparse
@needs_cli
def test_pyroid_run_sqlparses_command_line(tmp_path: Path) -> None:
    sql = tmp_path / "in.sql"
    sql.write_text("select a,b from t where c=1; -- note\ninsert into t values (1,2)\n")
    args = ["-r", "-k", "upper", "-s", str(sql)]
    plain = run_script(["-m", "sqlparse", *args], cwd=tmp_path)
    assert plain.returncode == 0, plain.stderr
    proc = run_script(
        ["-m", "pyroid", "run", "-p", "sqlparse", "-m", "sqlparse", *args], cwd=tmp_path
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == plain.stdout
    assert "SELECT a," in proc.stdout


@needs_sqlparse
def test_start_up_hook_patches_from_the_environment(tmp_path: Path) -> None:
    code = (
        "import json, sqlparse\n"
        "from pyroid.sqlparse import _patched\n"
        "print(json.dumps([sqlparse.format is _patched.format,\n"
        "                  sqlparse.format('select 1', keyword_case='upper')]))\n"
    )
    env = {"PYTHONPATH": _patch.HOOK_DIR, "PYROID_PATCH": "sqlparse"}
    proc = run_script(["-c", code], cwd=tmp_path, env=env)
    assert last_json(proc) == [True, "SELECT 1"]
    assert proc.stderr.count("PatchWarning") == 1, proc.stderr
    # Already warned higher up the process tree: quiet.
    proc = run_script(["-c", code], cwd=tmp_path, env={**env, "_PYROID_PATCH_WARNED": "sqlparse"})
    assert last_json(proc) == [True, "SELECT 1"]
    assert proc.stderr == ""


@needs_sqlparse
def test_patch_from_many_threads_swaps_once() -> None:
    out = child(
        """
        import threading
        import sqlparse
        import pyroid.integrations as pi
        before = current()
        barrier = threading.Barrier(8)
        def work():
            barrier.wait()
            pi.patch("sqlparse")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            threads = [threading.Thread(target=work) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        swapped = all_pyroid()
        pi.unpatch("sqlparse")
        print(json.dumps([swapped, len([w for w in caught if w.category is pi.PatchWarning]),
                          current() == before]))
        """
    )
    assert out == [True, 1, True]


@needs_sqlparse
def test_toggling_the_patch_while_other_threads_call_sqlparse() -> None:
    """patch/unpatch in a loop while 4 threads format, split, parse and run FilterStacks."""
    out = child(
        """
        import threading, time
        import sqlparse
        import pyroid.integrations as pi
        BUDGET = 3.0  # seconds of toggling
        def run_stack(sql):
            stack = sqlparse.engine.FilterStack()
            stack.enable_grouping()
            return [shape(st) for st in stack.run(sql)]
        def observe(sql):
            return [
                sqlparse.format(sql, reindent=True, keyword_case="upper"),
                sqlparse.format(sql, strip_comments=True),
                sqlparse.split(sql),
                [shape(st) for st in sqlparse.parse(sql)],
                run_stack(sql),
                [list(map(str, p)) for p in sqlparse.lexer.tokenize(sql)],
            ]
        want = [observe(s) for s in SQLS]
        patch_quietly("sqlparse")  # import and warm pyroid.sqlparse before the threads start
        assert [observe(s) for s in SQLS] == want
        pi.unpatch("sqlparse")
        stop = threading.Event()
        errors, calls = [], [0] * 4
        def work(i):
            k = 0
            while not stop.is_set():
                j = (k + i) % len(SQLS)
                try:
                    got = observe(SQLS[j])
                except BaseException as e:  # report, don't hang
                    errors.append(f"{type(e).__name__}: {e}")
                    break
                if got != want[j]:
                    errors.append(f"different result for {SQLS[j]!r}")
                k += 1
            calls[i] = k
        threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        toggles = 0
        deadline = time.monotonic() + BUDGET
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            while time.monotonic() < deadline:
                pi.patch("sqlparse")
                pi.unpatch("sqlparse")
                toggles += 1
        stop.set()
        for t in threads:
            t.join(60)
        print(json.dumps({"errors": errors[:5], "n_errors": len(errors), "toggles": toggles,
                          "calls": calls, "alive": sum(t.is_alive() for t in threads),
                          "patched": pi.patched()}))
        """,
        timeout=240,
    )
    assert out["errors"] == [], out
    assert out["alive"] == 0
    assert out["patched"] == []
    assert out["toggles"] > 10, out
    assert min(out["calls"]) > 0, out


# ---------------------------------------------------------------------------
# Without sqlparse: pyroid.sqlparse's modules stand in for sqlparse's
# ---------------------------------------------------------------------------


def test_without_sqlparse_the_modules_are_pyroids(tmp_path: Path) -> None:
    out = child(
        """
        import io, runpy
        import pyroid.integrations as pi
        try:
            import sqlparse
            hidden = False
        except ImportError:
            hidden = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pi.patch("sqlparse")
        import sqlparse
        from sqlparse import sql, tokens as T
        from sqlparse.exceptions import SQLParseError
        import sqlparse.engine.grouping as grouping
        import pyroid.sqlparse
        warned = [(w.category.__name__, str(w.message)) for w in caught]
        res = {"hidden": hidden, "warnings": warned}
        res["module"] = sqlparse is pyroid.sqlparse
        res["version"] = sqlparse.__version__
        res["format"] = sqlparse.format("select a,b from t where c=1", reindent=True,
                                        keyword_case="upper")
        res["split"] = sqlparse.split("select 1; select 2")
        stmt = sqlparse.parse(SQLS[0])[0]
        res["isinstance"] = [isinstance(stmt, sql.Statement),
                             isinstance(stmt.tokens[2], sql.IdentifierList),
                             stmt.tokens[0].ttype is T.Keyword.DML, T.Keyword.DML in T.Keyword,
                             stmt.get_type()]
        res["find_spec"] = [importlib.util.find_spec(n) is not None
                            for n in ("sqlparse", "sqlparse.sql", "sqlparse.engine.grouping")]
        stack = sqlparse.engine.FilterStack()
        stack.enable_grouping()
        stack.stmtprocess.append(sqlparse.filters.StripCommentsFilter())
        res["stack"] = [str(s) for s in stack.run("select 1 -- x\\n; select 2")]
        grouping.MAX_GROUPING_TOKENS = 5
        try:
            sqlparse.parse("select a, b, c, d, e, f from t")
            res["limit"] = "no error"
        except SQLParseError as e:
            res["limit"] = str(e)
        grouping.MAX_GROUPING_TOKENS = 10000
        # python -m sqlparse
        path = os.path.join(os.getcwd(), "in.sql")
        with open(path, "w") as f:
            f.write("select a,b from t")
        sys.argv, stdout, code = ["sqlparse", "-r", path], sys.stdout, None
        sys.stdout = io.StringIO()
        try:
            runpy.run_module("sqlparse", run_name="__main__", alter_sys=True)
        except SystemExit as e:
            code = e.code
        res["cli"], sys.stdout = [code, sys.stdout.getvalue()], stdout
        pi.unpatch("sqlparse")
        res["left"] = sorted(k for k in sys.modules if k == "sqlparse" or k.startswith("sqlparse."))
        try:
            importlib.import_module("sqlparse")
            res["after"] = "imported"
        except ImportError:
            res["after"] = "ImportError"
        print(json.dumps(res))
        """,
        hide=True,
        cwd=tmp_path,
    )
    assert out["hidden"]
    assert len(out["warnings"]) == 1, out["warnings"]
    category, message = out["warnings"][0]
    assert category == "PatchWarning"
    assert message.startswith(
        "pyroid.integrations.patch: `import sqlparse` now gives pyroid's drop-in (pyroid.sqlparse"
    )
    assert "sqlparse is not installed (pip install sqlparse==0.6.1)" in message
    assert out["module"]
    assert out["version"] == "0.6.1"
    assert out["format"] == "SELECT a,\n       b\nFROM t\nWHERE c=1"
    assert out["split"] == ["select 1;", "select 2"]
    assert out["isinstance"] == [True, True, True, True, "SELECT"]
    assert out["find_spec"] == [True, True, True]
    assert out["stack"] == ["select 1 \n; ", "select 2"]
    assert out["limit"] == "Maximum number of tokens exceeded (5)."
    assert out["cli"] == [0, "select a,\n       b\nfrom t"]
    assert out["left"] == []
    assert out["after"] == "ImportError"


def test_without_sqlparse_blocked_in_sys_modules() -> None:
    out = child(
        """
        sys.modules["sqlparse"] = None  # as if it were not installed
        import pyroid.integrations as pi
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pi.patch("sqlparse")
        import sqlparse
        from sqlparse import tokens as T
        first = next(sqlparse.parse("x")[0].flatten())
        res = [sqlparse.split("select 1; select 2"), first.ttype is T.Name, str(caught[0].message)]
        pi.unpatch("sqlparse")
        res.append(sys.modules["sqlparse"] is None)
        print(json.dumps(res))
        """
    )
    assert out[:2] == [["select 1;", "select 2"], True]
    assert "importing sqlparse is blocked (sys.modules['sqlparse'] is None)" in out[2]
    assert out[3] is True
