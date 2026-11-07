"""Self-tests of the harness itself: --impl parsing, adapter resolution, report schema, manifest integrity."""

from __future__ import annotations

import json
import warnings

import pytest

from pyroid_parity import data as D
from pyroid_parity import impl as I
from pyroid_parity.report import Collector, main, render, validate, write


@pytest.mark.parametrize(
    ("specs", "want"),
    [
        ([], I.REFERENCE),
        (["pyroid.ftfy"], {**I.REFERENCE, "ftfy": "pyroid.ftfy"}),
        (["ftfy"], I.REFERENCE),
        (
            ["pyroid.fasttext", "pyroid.filters"],
            {**I.REFERENCE, "fasttext": "pyroid.fasttext", "filters": "pyroid.filters"},
        ),
        (["datatrove"], I.REFERENCE),
        (["dedup=my.minhash"], {**I.REFERENCE, "dedup": "my.minhash"}),
    ],
)
def test_parse_specs(specs, want):
    assert I.parse_specs(specs) == want


@pytest.mark.parametrize("bad", ["pyroid", "pyroid.translit", "nosuchsuite=x"])
def test_parse_specs_rejects_ambiguous(bad):
    with pytest.raises(I.ImplError):
        I.parse_specs([bad])


def test_resolve_reference_ftfy_is_generic():
    impl = I.resolve("ftfy", "ftfy")
    assert impl.adapter == "generic" and impl.is_reference and impl.version == "6.3.1"


def test_resolve_datatrove_uses_adapter():
    pytest.importorskip("datatrove")
    assert I.resolve("filters", "datatrove").adapter == "pyroid_parity.adapters.filters_datatrove"


def test_resolve_missing_module_raises():
    with pytest.raises(ImportError):
        I.resolve("ftfy", "pyroid_does_not_exist.ftfy")


def test_generic_filters_contract_shapes():
    class Mod:
        @staticmethod
        def gopher_quality(texts):
            return [len(t) > 3 for t in texts], [None if len(t) > 3 else "gopher_short_doc" for t in texts]

        @staticmethod
        def c4_quality(texts):
            class R:
                keep = [True for _ in texts]
                reason = [None for _ in texts]
                text = [t.strip() for t in texts]

            return R()

    g = I.GenericFilters(Mod)
    assert g.run("gopher_quality", ["abcd", "ab"], {}) == [
        ("ok", True, None, None),
        ("ok", False, "gopher_short_doc", None),
    ]
    assert g.run("c4_quality", [" x ", "y"], {}) == [("ok", True, None, "x"), ("ok", True, None, None)]


def test_generic_filters_localizes_exceptions():
    class Mod:
        @staticmethod
        def fineweb_quality(texts):
            if any(not t for t in texts):
                raise ZeroDivisionError
            return [True] * len(texts), [None] * len(texts)

    assert I.GenericFilters(Mod).run("fineweb_quality", ["a", ""], {}) == [
        ("ok", True, None, None),
        ("raise", "ZeroDivisionError"),
    ]


def test_report_validates_and_renders():
    pytest.importorskip("jsonschema")
    c = Collector()
    c.impls["ftfy"] = I.resolve("ftfy", "ftfy").describe()
    c.references["ftfy"] = {"name": "ftfy", "version": "6.3.1", "tolerance": None}
    c.record("ftfy", "golden", True)
    c.record("ftfy", "hypothesis", False, case="x", input="\udc00", got="\udc00", want="�")
    c.skip("dedup", "removed", "needs full corpus")
    rep = c.build()
    assert validate(rep) == []
    assert rep["status"] == "fail" and rep["suites"]["ftfy"]["checks"]["hypothesis"]["diff"] == 1
    assert rep["suites"]["dedup"]["status"] == "skip"
    json.dumps(rep, ensure_ascii=False)  # lone surrogates are repr()'d, so the report is valid JSON
    assert "❌ fail" in render(rep)


def test_manifest_entries_are_well_formed():
    files = D.load_manifest()["files"]
    assert files, "bench/manifest.json is empty"
    for rel, e in files.items():
        assert len(e["sha256"]) == 64 and e["bytes"] > 0, rel
    assert sum(e["bytes"] for e in files.values()) <= 1_500_000_000  # data + models budget


def test_manifest_shows_only_this_platforms_references(tmp_path, monkeypatch):
    here = D.platform_key()
    manifest = tmp_path / "manifest.json"
    entry = {"sha256": "0" * 64, "bytes": 1}
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "files": {
                    "any.parquet": entry,
                    "here.parquet": {**entry, "platform": here},
                    "other.parquet": {**entry, "platform": "plan9-mips"},
                },
            }
        ),
        "utf-8",
    )
    monkeypatch.setattr(D, "MANIFEST", manifest)
    monkeypatch.setattr(D, "DATA", tmp_path)
    assert sorted(D.load_manifest()["files"]) == ["any.parquet", "here.parquet"]
    # Pinning one platform's file keeps the other platforms' entries.
    (tmp_path / "here.parquet").write_bytes(b"x")
    D.pin("here.parquet", platform=here)
    assert sorted(json.loads(manifest.read_text("utf-8"))["files"]) == [
        "any.parquet",
        "here.parquet",
        "other.parquet",
    ]


def test_every_platform_specific_reference_is_named_for_its_platform():
    from pyroid_parity import specs

    files = D._load_all()["files"]
    lid = {rel: e["platform"] for rel, e in files.items() if rel.startswith("parity/fasttext-ref")}
    assert "linux-aarch64" in lid.values() and "linux-x86_64" in lid.values()
    assert all(specs.lid_ref(p) == rel for rel, p in lid.items())
    assert all("platform" not in e for rel, e in files.items() if rel not in lid)


def test_content_hash_is_format_independent():
    assert D.content_sha256(["a"], ["b"]) == D.content_sha256(iter(["a"]), iter(["b"]))
    assert D.content_sha256(["ab"], ["c"]) != D.content_sha256(
        ["a"], ["bc"]
    )  # \0 separators prevent collisions


def _warns(*categories: type[Warning]) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for c in categories:
            warnings.warn("x", c, stacklevel=1)
    return I.warning_categories(caught)


def test_warning_categories_leave_out_pyroid_fallback_warning_only():
    fallback = type("FallbackWarning", (RuntimeWarning,), {"__module__": "pyroid"})
    sub = type("SlowPath", (fallback,), {"__module__": "pyroid.ftfy"})
    namesake = type("FallbackWarning", (RuntimeWarning,), {"__module__": "elsewhere"})
    assert I.is_fallback_warning(fallback) and I.is_fallback_warning(sub)
    assert not I.is_fallback_warning(namesake) and not I.is_fallback_warning(RuntimeWarning)
    assert _warns(fallback, sub) == []
    assert _warns(fallback, namesake, RuntimeWarning, DeprecationWarning) == [
        "DeprecationWarning",
        "FallbackWarning",
        "RuntimeWarning",
    ]


def test_warning_comparison_does_not_depend_on_call_order():
    """A candidate that warns once per process compares the same on its first and later calls."""
    fallback = type("FallbackWarning", (RuntimeWarning,), {"__module__": "pyroid"})
    warned: set[str] = set()

    def candidate() -> None:
        if "slow path" not in warned:
            warned.add("slow path")
            warnings.warn("delegating", fallback, stacklevel=2)
        warnings.warn("renamed", DeprecationWarning, stacklevel=2)

    def compared() -> list[str]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            candidate()
        return I.warning_categories(caught)

    assert compared() == compared() == ["DeprecationWarning"]


def test_only_render_writes_the_job_summary(tmp_path, monkeypatch, capsys):
    summary = tmp_path / "summary.md"
    summary.write_text("", "utf-8")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    c = Collector()
    c.record("ftfy", "golden", True)
    report = tmp_path / "parity-report.json"
    write(c.build(), report)  # what the pytest plugin calls at session end
    assert summary.read_text("utf-8") == ""
    assert main(["render", str(report)]) == 0
    assert summary.read_text("utf-8").count("### Parity:") == 1
    assert "### Parity:" in capsys.readouterr().out  # and it still prints the table


# Stands in for `python -m pyroid_parity.refs metrics`: runpy runs refs.py as __main__, the module copy the bug
# lived in, while metrics_cases.ref_metrics writes through its own import of pyroid_parity.refs. A stub table
# replaces the oracle run (minutes) so only the write and the exit code are under test.
_RUN_REFS_METRICS = """
import runpy, sys
import pyarrow as pa
from pyroid_parity import metrics_cases
metrics_cases.build_reference = lambda *_: pa.table({"case": ["c"], "key": ["k"], "out": ["o"]})
sys.argv = ["refs", "metrics", *sys.argv[1:]]
runpy.run_module("pyroid_parity.refs", run_name="__main__", alter_sys=True)
"""


def test_refs_exit_code_reports_a_mismatch_written_through_another_import(tmp_path):
    import os
    import subprocess
    import sys

    from pyroid_parity.metrics_cases import REF

    manifest = tmp_path / "manifest.json"
    env = {**os.environ, "PYROID_DATA": str(tmp_path / "data"), "PYROID_MANIFEST": str(manifest)}

    def refs_metrics(*args: str) -> subprocess.CompletedProcess[str]:
        cmd = [sys.executable, "-c", _RUN_REFS_METRICS, *args]
        return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120, check=False)

    pinned = refs_metrics("--pin")
    assert pinned.returncode == 0, pinned.stdout + pinned.stderr
    same = refs_metrics()
    assert same.returncode == 0, same.stdout + same.stderr
    assert f"verified {REF}" in same.stdout

    files = json.loads(manifest.read_text("utf-8"))
    files["files"][REF]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(files), "utf-8")
    differs = refs_metrics()
    assert differs.returncode == 1, differs.stdout + differs.stderr
    assert f"not reproduced (differ from or missing in {manifest}): {REF}" in differs.stderr
