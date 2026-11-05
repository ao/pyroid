"""Self-tests of the harness itself: --impl parsing, adapter resolution, report schema, manifest integrity."""

from __future__ import annotations

import json

import pytest

from pyroid_parity import data as D
from pyroid_parity import impl as I
from pyroid_parity.report import Collector, render, validate


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


def test_content_hash_is_format_independent():
    assert D.content_sha256(["a"], ["b"]) == D.content_sha256(iter(["a"]), iter(["b"]))
    assert D.content_sha256(["ab"], ["c"]) != D.content_sha256(
        ["a"], ["bc"]
    )  # \0 separators prevent collisions
