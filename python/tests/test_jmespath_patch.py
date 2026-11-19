"""pyroid.patch("jmespath"): pyroid's search and compile on jmespath's own module.

With jmespath installed, ``patch("jmespath")`` sets ``search`` and ``compile`` on ``jmespath``,
so code that calls ``jmespath.search(...)`` (boto3, botocore, Ansible's ``json_query``) runs
natively; exceptions stay jmespath's classes and ``unpatch`` restores the originals. Without
jmespath, ``jmespath`` and its submodules are shims. Patching is process-wide state, so every
test runs in a fresh interpreter.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import pytest

pytestmark = pytest.mark.thread_unsafe(reason="subprocesses and process-wide import state")


def _jmespath_ok() -> bool:
    try:
        return version("jmespath") == "1.1.0"
    except PackageNotFoundError:
        return False


needs_jmespath = pytest.mark.skipif(not _jmespath_ok(), reason="needs jmespath 1.1.0")

_CLEAN = ("PYTHONPATH", "PYROID_PATCH", "_PYROID_PATCH_WARNED", "PYTHONWARNINGS", "TEST_HIDE")

# TEST_HIDE=jmespath makes an installed jmespath look uninstalled.
PRELUDE = """\
import importlib, importlib.abc, json, os, sys, warnings
HIDE = set(filter(None, os.environ.get("TEST_HIDE", "").split(",")))
class _Hide(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in HIDE:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None
if HIDE:
    sys.meta_path.insert(0, _Hide())
DATA = {"Reservations": [{"Instances": [{"InstanceId": "i-1", "State": {"Name": "running"}},
                                         {"InstanceId": "i-2", "State": {"Name": "stopped"}}]}]}
QUERY = "Reservations[].Instances[?State.Name=='running'].InstanceId[]"
"""


def child(code: str, *, env: dict[str, str] | None = None, hide: tuple[str, ...] = ()) -> Any:
    """Run `code` in a fresh interpreter; return the JSON it printed last."""
    child_env = {k: v for k, v in os.environ.items() if k not in _CLEAN}
    child_env.update(env or {})
    if hide:
        child_env["TEST_HIDE"] = ",".join(hide)
    cmd = [sys.executable, "-I", "-c", PRELUDE + textwrap.dedent(code)]
    proc = subprocess.run(
        cmd, env=child_env, capture_output=True, text=True, timeout=300, check=False
    )
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    lines = proc.stdout.strip().splitlines()
    return json.loads(lines[-1]) if lines else None


@needs_jmespath
def test_patch_swaps_and_unpatch_restores() -> None:
    out = child(
        """
        import jmespath, jmespath.exceptions
        import pyroid, pyroid.jmespath as pj
        originals = (jmespath.search, jmespath.compile)
        want = jmespath.search(QUERY, DATA)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pyroid.patch("jmespath")
            pyroid.patch("jmespath")
        swapped = jmespath.search is pj.search and jmespath.compile is pj.compile
        got = jmespath.search(QUERY, DATA)
        compiled = type(jmespath.compile(QUERY)).__module__
        try:
            jmespath.search("a[", {})
        except jmespath.exceptions.ParseError as e:
            caught_error = type(e).__module__
        class F(jmespath.functions.Functions):
            @jmespath.functions.signature({"types": ["string"]})
            def _func_shout(self, s):
                return s.upper()
        custom = jmespath.search("shout(a)", {"a": "x"}, jmespath.Options(custom_functions=F()))
        pyroid.unpatch("jmespath")
        restored = (jmespath.search, jmespath.compile) == originals
        print(json.dumps({
            "warnings": [(w.category.__name__, str(w.message)) for w in caught],
            "swapped": swapped, "same": got == want, "compiled": compiled,
            "error": caught_error, "custom": custom, "restored": restored,
        }))
        """
    )
    assert out["swapped"]
    assert out["same"]
    assert out["compiled"] == "pyroid.jmespath.parser"
    assert out["error"] == "jmespath.exceptions"
    assert out["custom"] == "X"
    assert out["restored"]
    assert len(out["warnings"]) == 1, out["warnings"]
    assert out["warnings"][0][0] == "PatchWarning"


def test_patch_without_jmespath_installs_shims() -> None:
    out = child(
        """
        import pyroid
        pyroid.patch("jmespath")
        import jmespath
        from jmespath import exceptions, functions
        from jmespath.visitor import Options
        from jmespath.parser import Parser
        try:
            jmespath.compile("a[")
        except exceptions.ParseError:
            error = "ParseError"
        print(json.dumps({
            "shim": getattr(jmespath, "__pyroid__", False), "version": jmespath.__version__,
            "result": jmespath.search(QUERY, DATA, Options()),
            "parser": type(Parser().parse("a")).__name__, "error": error,
            "functions": hasattr(functions, "Functions"),
        }))
        """,
        hide=("jmespath",),
        env={"PYTHONWARNINGS": "ignore::UserWarning"},
    )
    assert out == {
        "shim": True,
        "version": "1.1.0",
        "result": ["i-1"],
        "parser": "ParsedResult",
        "error": "ParseError",
        "functions": True,
    }


_TABLE_ADD = """
    import pyroid
    pyroid.patch("jmespath")
    import jmespath, jmespath.exceptions, jmespath.functions
    jmespath.functions.Functions.FUNCTION_TABLE["shout"] = {
        "function": lambda self, x: x.upper(), "signature": [{"types": ["string"]}]}
    out = [jmespath.search("shout('a')", {}), jmespath.compile("[*].shout(@)").search(["b"])]
    try:
        jmespath.search("shout(`1`)", {})
    except jmespath.exceptions.JMESPathTypeError as e:
        out.append(type(e).__name__)
    print(json.dumps(out))
"""


@pytest.mark.parametrize("installed", [True, False])
def test_functions_added_to_the_function_table_under_the_patch(installed: bool) -> None:
    # P67's jm_function_table_add: jmespath's default Functions() reads its FUNCTION_TABLE on
    # every call, so boto3-style code that registers a function there finds it.
    if installed and not _jmespath_ok():
        pytest.skip("needs jmespath 1.1.0")
    out = child(
        _TABLE_ADD,
        hide=() if installed else ("jmespath",),
        env={"PYTHONWARNINGS": "ignore::UserWarning"},
    )
    assert out == ["A", ["B"], "JMESPathTypeError"]


@needs_jmespath
def test_pyroid_run_patches_jmespath(tmp_path: Any) -> None:
    pytest.importorskip("pyarrow", reason="the pyroid command line needs pyarrow (pyroid[cli])")
    script = tmp_path / "query.py"
    script.write_text(
        "import json, jmespath, pyroid.jmespath as pj\n"
        "print(json.dumps([jmespath.search is pj.search,\n"
        "                  jmespath.search('a.b', {'a': {'b': 1}})]))\n"
    )
    child_env = {k: v for k, v in os.environ.items() if k not in _CLEAN}
    child_env["PYTHONWARNINGS"] = "ignore::UserWarning"
    proc = subprocess.run(
        [sys.executable, "-m", "pyroid", "run", "-p", "jmespath", str(script)],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == [True, 1]
