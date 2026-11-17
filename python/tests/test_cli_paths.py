"""``pyroid`` refuses a run whose outputs overlap each other, its inputs or its work directory.

``pyroid clean IN OUT --excluded OUT`` wrote the dropped rows over the kept ones and exited 0.
Every combination of OUT, ``--excluded``, ``--tmp-dir``, ``--stats-json`` and the inputs is
checked before anything is read or written: the same path, relative or absolute, through a
symlink, and directories inside each other.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from pyroid.cli import main  # noqa: E402

pytestmark = pytest.mark.thread_unsafe(reason="changes the working directory")


def _enter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``tmp_path`` as the working directory, with ``in.parquet`` (texts with duplicates).

    Not a fixture: pytest-run-parallel gives each of a test's ``--iterations`` its own
    ``tmp_path`` only when the test asks for it itself.
    """
    monkeypatch.chdir(tmp_path)
    texts = ["a", "b", "a", "c", "b"]
    pq.write_table(pa.table({"id": [f"d{i}" for i in range(5)], "text": texts}), "in.parquet")
    return tmp_path


def _exact(*args: str) -> list[str]:
    return ["dedup", "--method", "exact", "in.parquet", *args]


def _refused(argv: list[str], match: str) -> None:
    with pytest.raises(SystemExit, match=match) as e:
        main(argv)
    assert e.value.code not in (0, None)


def test_excluded_same_file_as_out_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    here = _enter(tmp_path, monkeypatch)
    assert main(_exact("same.parquet")) == 0
    kept = pq.read_table("same.parquet").to_pylist()
    assert [r["id"] for r in kept] == ["d0", "d1", "d3"]
    same = "OUT and --excluded are the same file"
    for excl in ("same.parquet", str(here / "same.parquet"), "sub/../same.parquet"):
        _refused(_exact("same.parquet", "--excluded", excl, "--overwrite"), same)
    os.symlink("same.parquet", "link.parquet")
    _refused(_exact("same.parquet", "--excluded", "link.parquet", "--overwrite"), same)
    _refused(_exact("link.parquet", "--excluded", "same.parquet", "--overwrite"), same)
    if sys.platform == "darwin":  # case-insensitive by default
        _refused(_exact("same.parquet", "--excluded", "SAME.parquet", "--overwrite"), same)
    assert pq.read_table("same.parquet").to_pylist() == kept  # the kept rows are still there


def test_the_clean_command_of_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, en_lid: Path
) -> None:
    here = _enter(tmp_path, monkeypatch)
    argv = ["clean", "in.parquet", "same.parquet", "--recipe", "fineweb", "--lid-model"]
    _refused([*argv, str(en_lid), "--excluded", "same.parquet", "--stats"], "the same file")
    assert not Path("same.parquet").exists()
    assert not list(here.glob(".same*"))


def test_directory_out_and_excluded_must_be_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    here = _enter(tmp_path, monkeypatch)
    argv = ["dedup", "in.parquet"]
    inside = "--excluded must not be OUT or inside it"
    _refused([*argv, "od/", "--excluded", "od/"], inside)
    _refused([*argv, "od/", "--excluded", "od/dropped/"], inside)
    _refused([*argv, "od/", "--excluded", str(here / "od")], inside)
    _refused([*argv, "od/kept/", "--excluded", "od/"], "OUT must not be inside --excluded")
    Path("real").mkdir()
    os.symlink("real", "alias")
    _refused([*argv, "real/", "--excluded", "alias/"], inside)
    assert not Path("od").exists()
    assert main([*argv, "od/", "--excluded", "dropped/"]) == 0


def test_tmp_dir_and_stats_json_must_not_be_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enter(tmp_path, monkeypatch)
    _refused(_exact("o.parquet", "--tmp-dir", "o.parquet"), "--tmp-dir o.parquet is an output")
    Path("afile").write_text("x")
    _refused(_exact("o.parquet", "--tmp-dir", "afile"), "--tmp-dir afile is not a directory")
    _refused(_exact("o.parquet", "--stats-json", "o.parquet"), "--stats-json o.parquet is also")
    _refused(_exact("o.parquet", "--stats-json", "in.parquet"), "is also an input")
    assert main(["translit", "in.parquet", "t/"]) == 0
    _refused(["translit", "in.parquet", "t/", "--stats-json", "t/in.parquet"], "is also an output")
    assert not Path("o.parquet").exists()


def test_inputs_never_overlap_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enter(tmp_path, monkeypatch)
    _refused(_exact("in.parquet", "--overwrite"), "both an input and an output")
    _refused(
        ["dedup", "--method", "exact", "in.parquet", "o.parquet", "--excluded", "in.parquet"],
        "both an input and an output",
    )
    os.symlink("in.parquet", "alias.parquet")
    _refused(["dedup", "--method", "exact", "alias.parquet", "in.parquet"], "both an input")
    # an input inside a directory OUT is refused by name, not silently skipped
    Path("out/raw").mkdir(parents=True)
    Path("in.parquet").rename("out/raw/in.parquet")
    _refused(["translit", "out/raw/", "out/", "--overwrite"], "is inside OUT")
    _refused(["translit", "out/raw/in.parquet", "out/", "--overwrite"], "is inside OUT")


def test_an_input_below_tmp_dir_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enter(tmp_path, monkeypatch)
    # Only the run's own work directory there is skipped (it said "no .parquet or .jsonl files").
    Path("scratch").mkdir()
    Path("in.parquet").rename("scratch/in.parquet")
    assert (
        main(
            [
                "dedup",
                "--method",
                "exact",
                "scratch/in.parquet",
                "o.parquet",
                "--tmp-dir",
                "scratch",
            ]
        )
        == 0
    )
    assert [r["id"] for r in pq.read_table("o.parquet").to_pylist()] == ["d0", "d1", "d3"]
    assert (
        main(["dedup", "--method", "exact", "scratch/", "o2.parquet", "--tmp-dir", "scratch"]) == 0
    )
    assert sorted(p.name for p in Path("scratch").iterdir()) == ["in.parquet"]


def test_stage_folders_must_be_apart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enter(tmp_path, monkeypatch)
    assert main(["dedup", "signatures", "in.parquet", "sigs/"]) == 0
    _refused(["dedup", "buckets", "sigs", "sigs"], "DUPS and SIGS must be separate folders")
    _refused(["dedup", "buckets", "sigs", "sigs/dups"], "separate folders")
    assert main(["dedup", "buckets", "sigs", "dups"]) == 0
    _refused(["dedup", "cluster", "dups", "dups"], "REMOVED and DUPS must be separate folders")
    assert main(["dedup", "cluster", "dups", "removed"]) == 0
    filt = ["dedup", "filter", "in.parquet"]
    _refused([*filt, "removed/", "--removed", "removed"], "OUT and --removed must be separate")
    _refused([*filt, "f/", "--removed", "removed", "--excluded", "removed/x/"], "--excluded")
    assert main([*filt, "f/", "--removed", "removed"]) == 0


@pytest.fixture(scope="module")
def en_lid(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic fastText model whose only label is ``__label__en``."""
    pytest.importorskip("numpy")
    import dataclasses

    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from test_fasttext import SPECS, build
    finally:
        sys.path.pop(0)
    path = tmp_path_factory.mktemp("lid") / "en.bin"
    spec: Any = dataclasses.replace(SPECS["softmax-dense"], labels=("__label__en",))
    path.write_bytes(build(spec).data)
    return path
