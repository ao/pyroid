"""A module whose optional dependency is missing says which pyroid extra to install."""

from __future__ import annotations

import subprocess
from collections.abc import Callable

import pytest

RunPython = Callable[..., subprocess.CompletedProcess[str]]

# (module, modules hidden from it, the missing name, what the message must tell you to install)
CASES = [
    ("pyroid.integrations._io", ["pyarrow"], "pyarrow", "pip install 'pyroid[cli]'"),
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
