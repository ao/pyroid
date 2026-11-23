"""``pyroid.cli.main([...])`` called from several threads of one process at once."""

from __future__ import annotations

import dataclasses
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("numpy")

from pyroid.cli import main  # noqa: E402


def _write(texts: list[str], path: Path) -> Path:
    pq.write_table(pa.table({"id": [f"d{i}" for i in range(len(texts))], "text": texts}), path)
    return path


def _lid(path: Path, label: str) -> Path:
    """A synthetic fastText model whose only label is ``label``."""
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from test_fasttext import SPECS, build
    finally:
        sys.path.pop(0)
    path.write_bytes(build(dataclasses.replace(SPECS["softmax-dense"], labels=(label,))).data)
    return path


@pytest.mark.thread_unsafe(reason="patches the translit step")
def test_a_failed_call_leaves_the_runs_of_other_calls_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call that fails removes what its own runs left, never what a call in another thread is
    writing. All the calls of a process shared one list of runs: the first to fail emptied the
    work and state directories of the others ("No such file or directory: .run.json...tmp")."""
    import pyroid.cli as cli

    ok_src = _write(["Ünïcödé", "b"], tmp_path / "ok.parquet")
    bad_src = _write(["fail"], tmp_path / "bad.parquet")
    entered, go = threading.Event(), threading.Event()
    real = cli._STEPS["translit"]

    def step(t: Any, o: Any) -> Any:
        if t.column("text")[0].as_py() == "fail":
            raise ValueError("boom")
        entered.set()  # the other call is under way: its state directory has no record yet
        assert go.wait(60)
        return real(t, o)

    monkeypatch.setitem(cli._STEPS, "translit", step)
    out = tmp_path / "out.parquet"
    codes: list[int] = []
    other = threading.Thread(target=lambda: codes.append(main(["translit", str(ok_src), str(out)])))
    other.start()
    try:
        assert entered.wait(60)
        assert main(["translit", str(bad_src), str(tmp_path / "bad-out.parquet")]) == 1
    finally:
        go.set()
        other.join(60)
    assert codes == [0]
    assert pq.read_table(out).column("text").to_pylist() == ["Unicode", "b"]
    assert (tmp_path / ".out.parquet.pyroid" / "run.json").is_file()  # a rerun is a no-op
    # The failed call tidied up after itself, and only after itself.
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        ".out.parquet.pyroid",
        "bad.parquet",
        "ok.parquet",
        "out.parquet",
    ]


def test_calls_in_threads_at_once(tmp_path: Path) -> None:
    """Refused, failed and finished calls in eight threads at once, each with its own OUT."""
    src = _write(["Ünïcödé", "b"], tmp_path / "in.parquet")
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "d0", "text": "x"}\n{"id": "d1", "text": "unterminated\n')
    threads, rounds = 8, 5
    start = threading.Barrier(threads)
    errors: list[str] = []

    def calls(n: int) -> None:
        start.wait()
        for r in range(rounds):
            try:
                one(n, r)
            except Exception as e:
                errors.append(f"t{n}-{r}: {e!r}")

    def one(n: int, r: int) -> None:
        here = tmp_path / f"t{n}-{r}"
        out = here / "out"
        out.mkdir(parents=True)
        (out / "notes.txt").write_text("mine")
        try:
            main(["translit", str(src), str(out)])
            errors.append(f"{here}: a foreign OUT was not refused")
        except SystemExit:
            pass
        if main(["translit", str(bad), str(here / "bad.parquet")]) != 1:
            errors.append(f"{here}: an unreadable input did not fail")
        finished = (
            ["translit", str(src), str(out), "--overwrite"],
            ["translit", str(src), str(here / "o.parquet"), "--mode", "unidecode"],
            ["dedup", str(src), str(here / "d.parquet"), "--method", "exact"],
        )
        errors.extend(f"{here}: {args} failed" for args in finished if main(args) != 0)
        texts = pq.read_table(here / "o.parquet").column("text").to_pylist()
        if texts != ["Unicode", "b"] or (out / "notes.txt").read_text() != "mine":
            errors.append(f"{here}: wrong output")
        if (here / ".bad.parquet.pyroid").exists():
            errors.append(f"{here}: the failed run left its state")

    pool = [threading.Thread(target=calls, args=(n,)) for n in range(threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join(120)
    assert errors == []


def test_each_call_loads_its_own_model(tmp_path: Path) -> None:
    """A model file changed between two calls is read again: recipes were cached per process,
    for good, by path."""
    src = _write(["The council of the city met on Tuesday."], tmp_path / "in.parquet")
    model = tmp_path / "lid.bin"
    labels = []
    for i, label in enumerate(("__label__en", "__label__fr")):
        _lid(model, label)
        out = tmp_path / f"o{i}.parquet"
        assert main(["langid", str(src), str(out), "--model", str(model)]) == 0
        labels.append(pq.read_table(out).column("language").to_pylist())
    assert labels == [["en"], ["fr"]]
