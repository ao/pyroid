"""A module whose optional dependency is missing says which pyroid extra to install."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

RunPython = Callable[..., subprocess.CompletedProcess[str]]

# (module, modules hidden from it, the missing name, what the message must tell you to install)
CASES = [
    ("pyroid.integrations.recipes", ["pyarrow"], "pyarrow", "pip install 'pyroid[cli]'"),
    ("pyroid.integrations.recipes", ["numpy"], "numpy", "pip install 'pyroid[cli]'"),
    ("pyroid.integrations.hf", ["pyarrow"], "pyarrow", "pip install 'pyroid[hf]'"),
    ("pyroid.integrations.nemo_curator", ["numpy"], "numpy", "pip install 'pyroid[nemo]'"),
    (
        "pyroid.integrations.nemo_curator",
        ["nemo_curator"],
        "nemo_curator",
        "pip install nemo-curator==1.4.0",
    ),
    (
        "pyroid.integrations.datatrove",
        ["datatrove"],
        "datatrove",
        "pip install 'pyroid[datatrove]'",
    ),
    ("pyroid.fasttext", ["numpy"], "numpy", "pip install 'pyroid[fasttext]'"),
    ("pyroid.datasketch", ["numpy"], "numpy", "pip install 'pyroid[datasketch]'"),
]

# Importable with none of the optional dependencies.
BARE = [
    "pyroid.cli",
    "pyroid.dedup",
    "pyroid.filters",
    "pyroid.ftfy",
    "pyroid.integrations",
    "pyroid.integrations._io",  # pyarrow is imported where it reads or writes a file
    "pyroid.integrations._minhash",
    "pyroid.integrations._shards",
    "pyroid.jieba",
    "pyroid.json_repair",
    "pyroid.metrics.rouge_scorer",
    "pyroid.metrics.sacrebleu",
    "pyroid.nemo",
    "pyroid.nltk",
    "pyroid.text",
    "pyroid.trafilatura",
    "pyroid.translit",
    "pyroid.warc",
]
OPTIONAL = ["numpy", "pyarrow", "pandas", "polars", "datasets", "datatrove", "nemo_curator"]


@pytest.mark.thread_unsafe(reason="runs a subprocess")
@pytest.mark.parametrize(("module", "hidden", "name", "fix"), CASES)
def test_missing_dependency_names_the_extra(
    run_python: RunPython, module: str, hidden: list[str], name: str, fix: str
) -> None:
    proc = run_python(
        f"""
        import importlib, sys
        for m in {hidden!r}:
            sys.modules[m] = None  # import raises ModuleNotFoundError
        try:
            importlib.import_module({module!r})
        except ModuleNotFoundError as e:
            assert e.name == {name!r}, (e.name, str(e))
            assert type(e.__cause__) is ModuleNotFoundError, repr(e.__cause__)
            print(e)
        else:
            print("imported")
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout.strip()
    assert out.startswith(f"{module} needs {name}"), out
    assert fix in out, out


@pytest.mark.thread_unsafe(reason="runs a subprocess")
def test_bare_install_modules_import(run_python: RunPython) -> None:
    proc = run_python(
        f"""
        import importlib, sys
        for m in {OPTIONAL!r}:
            sys.modules[m] = None
        for m in {BARE!r}:
            importlib.import_module(m)
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.thread_unsafe(reason="runs a subprocess")
def test_pyroid_run_needs_no_optional_dependency(run_python: RunPython, tmp_path: Path) -> None:
    # `pyroid run` only patches imports; the other commands need pyarrow and say so.
    script = tmp_path / "uses_ftfy.py"
    script.write_text(
        "import ftfy\nprint(ftfy.fix_text('âœ” done'), ftfy.__pyroid__)\n", encoding="utf-8"
    )
    proc = run_python(
        f"""
        import sys
        for m in {OPTIONAL!r}:
            sys.modules[m] = None
        from pyroid.cli import main
        assert main(["run", "-p", "ftfy", {str(script)!r}]) == 0
        assert "pyarrow" not in sys.modules or sys.modules["pyarrow"] is None
        try:
            main(["clean", "--recipe", "fineweb", "in.jsonl", "out.parquet"])
        except SystemExit as e:
            print(e)
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.splitlines() == [
        "✔ done True",
        "pyroid: this command needs pyarrow (and numpy): pip install 'pyroid[cli]'",
    ], proc.stdout


@pytest.mark.thread_unsafe(reason="runs a subprocess")
def test_io_needs_pyarrow_where_it_reads_or_writes(run_python: RunPython) -> None:
    proc = run_python(
        """
        import sys
        sys.modules["pyarrow"] = None
        from pyroid.integrations import _io
        assert _io.file_format("a.jsonl.gz") == "jsonl"
        try:
            _io.pa.schema([])
        except ModuleNotFoundError as e:
            assert e.name == "pyarrow", e.name
            print(e)
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == (
        "pyroid.integrations._io needs pyarrow: pip install 'pyroid[cli]' (or pip install pyarrow)"
    )


@pytest.mark.thread_unsafe(reason="runs a subprocess")
def test_commands_without_arrow_need_no_optional_dependency(
    run_python: RunPython, tmp_path: Path
) -> None:
    """--help, stats and the MinHash stages that read and write only signature and duplicate
    files run with neither pyarrow nor numpy. The signatures are written natively here (what
    ``pyroid dedup signatures`` writes from Arrow batches), so this also runs where pyarrow is
    not installed."""
    proc = run_python(
        f"""
        import contextlib, io, json, sys
        from pathlib import Path
        for m in {OPTIONAL!r}:
            sys.modules[m] = None
        from pyroid._pyroid import dedup as _native
        from pyroid.cli import main
        from pyroid.integrations import _minhash

        root = Path({str(tmp_path)!r})
        sigs, dups, removed = root / "sigs", root / "dups", root / "removed"
        words = "the quick brown fox jumps over the lazy dog".split()
        # Letters, not digits: datatrove's text simplification replaces digits.
        docs = [" ".join(w + chr(97 + i) for w in words * 3) for i in range(20)]
        cfg = _minhash.config_dict()
        for rank, texts in enumerate([docs[:10], [*docs[10:], docs[3]]]):
            w = _native._SigFileWriter(str(sigs), rank, _minhash.native_config(cfg))
            w.add(texts)
            w.finish()
        for argv in (["--help"], ["dedup", "--help"], ["dedup", "cluster", "--help"]):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                try:
                    main(argv)
                except SystemExit as e:
                    assert e.code == 0, (argv, e.code)
            assert out.getvalue().startswith("usage: pyroid"), (argv, out.getvalue()[:200])
        assert main(["dedup", "buckets", str(sigs), str(dups)]) == 0
        assert main(["dedup", "cluster", str(dups), str(removed)]) == 0
        with contextlib.redirect_stdout(io.StringIO()) as out:
            assert main(["stats", str(dups)]) == 0
        stats = json.loads(out.getvalue())
        assert all(sys.modules.get(m) is None for m in {OPTIONAL!r})
        found = {{p.name: p.read_bytes().hex() for p in sorted(removed.glob("*.remove"))}}
        print(json.dumps({{"stats": sorted(stats), "removed": found}}))
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    got = json.loads(proc.stdout.splitlines()[-1])
    assert "files" in got["stats"], got
    # The copy of document 3 is the 11th document of file 1 (index 10, a little-endian uint32).
    assert got["removed"] == {"000001.remove": (10).to_bytes(4, "little").hex()}, got
