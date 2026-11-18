"""The ``pyroid`` command line: formats, streaming, subcommands, digest identity with the chain."""

from __future__ import annotations

import errno
import gzip
import importlib.util
import io
import json
import os
import re
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import pyroid  # noqa: E402
import pyroid.dedup  # noqa: E402
import pyroid.translit  # noqa: E402
from pyroid.cli import main  # noqa: E402

DATA = Path(os.environ.get("PYROID_DATA", "/data"))
DEV = DATA / "fineweb" / "fineweb-dev-10k.parquet"
LID_BIN = DATA / "models" / "lid.176.bin"
LID_FTZ = DATA / "models" / "lid.176.ftz"
DIGEST_SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "cli" / "digest.py"
DEV_DIGEST = "ef314b8d6ae95cc63c6cfa8b1a3a76ee5bc101eb88a66729beb10ecb7226473b"

BASE = (
    "The quick brown fox jumps over the lazy dog while the farmer watches from the porch and "
    "thinks about the weather, the harvest and the long winter that is surely coming soon to "
    "the valley where his family has lived for many generations. "
)


def _table(texts: list[Any], **extra: list[Any]) -> Any:
    cols = {"id": [f"d{i}" for i in range(len(texts))], "text": texts, **extra}
    return pa.table(cols)


def _write(t: Any, path: Path) -> Path:
    if path.suffix == ".parquet":
        pq.write_table(t, path)
    else:
        opener: Any = gzip.open if path.suffix == ".gz" else open
        with opener(path, "wt", encoding="utf-8") as f:
            for r in t.to_pylist():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _read(path: Path) -> Any:
    from pyroid.integrations import _io

    tables = list(_io.read_tables(_io.input_files(path), 1 << 16))
    return pa.concat_tables(tables) if tables else None


# --------------------------------------------------------------------------------------- basics


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert pyroid.__version__ in capsys.readouterr().out


def test_python_dash_m(tmp_path: Path) -> None:
    src = _write(_table(["Ünïcödé", "ascii"]), tmp_path / "in.parquet")
    out = tmp_path / "out.jsonl"
    r = subprocess.run(
        [sys.executable, "-m", "pyroid", "translit", str(src), str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert [json.loads(x)["text"] for x in out.read_text().splitlines()] == ["Unicode", "ascii"]


#: The leading bytes of each compressed output (the suffixes `pyroid --help` lists).
MAGIC = {".gz": b"\x1f\x8b", ".bz2": b"BZh", ".zst": b"\x28\xb5\x2f\xfd", ".lz4": b'\x04"M\x18'}


@pytest.mark.parametrize("fmt_in", [".parquet", ".jsonl", ".jsonl.gz"])
@pytest.mark.parametrize(
    "fmt_out", [".parquet", ".jsonl", ".ndjson.gz", ".json.zst", ".jsonl.bz2", ".jsonl.lz4"]
)
def test_translit_formats(tmp_path: Path, fmt_in: str, fmt_out: str) -> None:
    texts = ["Ålborg — “quoted” ½", "日本語", "plain", "Ελληνικά"] * 5
    src = _write(_table(texts, n=list(range(len(texts)))), tmp_path / f"in{fmt_in}")
    out = tmp_path / f"out{fmt_out}"
    assert main(["translit", str(src), str(out), "--batch-size", "3"]) == 0
    assert out.read_bytes().startswith(MAGIC.get(out.suffix, b""))
    t = _read(out)
    assert t.column("text").to_pylist() == pyroid.translit.anyascii(texts)
    assert t.column("n").to_pylist() == list(range(len(texts)))
    assert t.column_names == ["id", "text", "n"]
    assert not list(tmp_path.glob("*.partial"))


def test_translit_output_column_and_unidecode(tmp_path: Path) -> None:
    texts = ["café", "\U0001f600 smile"]
    src = _write(_table(texts), tmp_path / "in.parquet")
    out = tmp_path / "out.parquet"
    args = ["translit", str(src), str(out), "--mode", "unidecode", "--errors", "replace"]
    assert main([*args, "--output-column", "ascii"]) == 0
    t = pq.read_table(out)
    assert t.column("text").to_pylist() == texts
    assert t.column("ascii").to_pylist() == [pyroid.translit.unidecode(x, "replace") for x in texts]


def test_directory_input_in_name_order(tmp_path: Path) -> None:
    d = tmp_path / "shards"
    d.mkdir()
    _write(_table(["b1", "b2"]), d / "b.parquet")
    _write(_table(["a1"]), d / "a.parquet")
    (d / "notes.txt").write_text("ignored")
    out = tmp_path / "out.parquet"
    assert main(["translit", str(d), str(out), "--batch-size", "2"]) == 0
    assert pq.read_table(out).column("text").to_pylist() == ["a1", "b1", "b2"]


def test_empty_input(tmp_path: Path) -> None:
    src = _write(
        _table([]).cast(pa.schema([("id", pa.string()), ("text", pa.string())])),
        tmp_path / "in.parquet",
    )
    out = tmp_path / "out.parquet"
    assert main(["dedup", str(src), str(out), "--method", "exact"]) == 0
    assert pq.read_table(out).num_rows == 0


def test_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = _write(_table(["x"]), tmp_path / "in.parquet")
    out = tmp_path / "out.parquet"
    with pytest.raises(SystemExit, match="no column 'body'"):
        main(["translit", str(src), str(out), "--text-column", "body"])
    assert not out.exists()
    assert not list(tmp_path.glob("*.partial"))
    out.write_text("keep me")
    with pytest.raises(SystemExit, match="--overwrite"):
        main(["translit", str(src), str(out)])
    assert out.read_text() == "keep me"
    assert main(["translit", str(src), str(out), "--overwrite"]) == 0
    assert main(["translit", str(src), str(tmp_path / "out.csv")]) == 1
    err = capsys.readouterr().err
    assert "unknown format" in err
    assert "end OUT with / to make it a directory" in err
    assert main(["translit", str(tmp_path / "missing.parquet"), str(tmp_path / "o.parquet")]) == 1
    with pytest.raises(SystemExit):
        main(["clean", str(src), str(tmp_path / "o.parquet")])  # --recipe is required
    with pytest.raises(SystemExit, match=r"in\.parquet is both an input and an output"):
        main(["translit", str(src), str(src), "--overwrite"])


@pytest.mark.thread_unsafe(reason="monkeypatches pyroid.cli and the environment")
def test_file_and_input_errors_are_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyroid.cli as cli

    monkeypatch.delenv("PYROID_DEBUG", raising=False)
    src = _write(_table(["x"]), tmp_path / "in.parquet")
    args = ["translit", str(src), str(tmp_path / "o.parquet")]
    cases = [
        IsADirectoryError(21, "Is a directory", "models/"),
        PermissionError(13, "Permission denied", "/out/x.parquet"),
        OSError("could not download lid.176.bin: https://example.invalid/lid.176.bin"),
        EOFError("Compressed file ended before the end-of-stream marker was reached"),
        pyroid.InputError("expected a string column, got int64"),
    ]
    for err in cases:

        def body(a: Any, err: BaseException = err) -> int:
            raise err

        monkeypatch.setattr(cli, "_cmd_translit", body)
        assert main(args) == 1
        assert capsys.readouterr().err == f"pyroid translit: {err}\n"
    # A real one: the model path is a directory.
    out = tmp_path / "c.parquet"
    rc = main(["clean", str(src), str(out), "--recipe", "fineweb", "--lid-model", str(tmp_path)])
    assert rc == 1
    msg = capsys.readouterr().err
    assert msg.startswith("pyroid clean: ")
    assert "Traceback" not in msg


@pytest.mark.thread_unsafe(reason="monkeypatches pyroid.cli and the environment")
def test_missing_optional_dependency_is_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ImportError from a command exits 1 with one line that says what to install."""
    import pyroid.cli as cli

    monkeypatch.delenv("PYROID_DEBUG", raising=False)
    src = _write(_table(["x"]), tmp_path / "in.parquet")
    args = ["translit", str(src), str(tmp_path / "o.parquet")]
    cases = [
        (
            ModuleNotFoundError("No module named 'cchardet'", name="cchardet"),
            "No module named 'cchardet' (pip install 'pyroid[warc]')",
        ),
        (
            ImportError("failed to find libmagic.  Check your installation"),
            "failed to find libmagic.  Check your installation (python-magic needs the libmagic "
            "library, e.g. apt install libmagic1)",
        ),
        (
            ModuleNotFoundError("No module named 'tldextract.cache'", name="tldextract.cache"),
            "No module named 'tldextract.cache' (pip install tldextract)",
        ),
        (ModuleNotFoundError("x needs y (pip install y)", name="y"), "x needs y (pip install y)"),
        (ModuleNotFoundError("No module named 'zzz'", name="zzz"), "No module named 'zzz'"),
    ]
    for err, line in cases:

        def body(a: Any, err: ImportError = err) -> int:
            raise err

        monkeypatch.setattr(cli, "_cmd_translit", body)
        assert main(args) == 1
        assert capsys.readouterr().err == f"pyroid translit: {line}\n"
    with monkeypatch.context() as m:  # undone at once: pytest-run-parallel repeats the test
        m.setenv("PYROID_DEBUG", "1")
        with pytest.raises(ModuleNotFoundError, match="zzz"):
            main(args)


@pytest.mark.thread_unsafe(reason="reads stderr")
def test_a_text_column_of_another_type_is_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], en_lid: Path
) -> None:
    """langid printed a traceback (pyarrow's replace_substring on int64); now every command
    says what is wrong, before it starts, for the first file and for any later one."""
    ints = tmp_path / "ints.parquet"
    pq.write_table(pa.table({"id": ["a", "b"], "text": [1, 2]}), ints)
    out = tmp_path / "o.parquet"
    msg = "column 'text' holds int64, not strings (--text-column names the text column)"
    for argv in (
        ["langid", str(ints), str(out), "--model", str(en_lid)],
        ["clean", str(ints), str(out), "--recipe", "fineweb", "--lid-model", str(en_lid)],
        ["translit", str(ints), str(out)],
    ):
        with pytest.raises(SystemExit, match=re.escape(msg)):
            main(argv)
    # a later file of another type: one line, naming the file
    d = tmp_path / "in"
    d.mkdir()
    _write(_table([_english(1)]), d / "a.parquet")
    pq.write_table(pa.table({"id": ["b"], "text": [3]}), d / "b.parquet")
    assert main(["langid", str(d), str(tmp_path / "l"), "--model", str(en_lid)]) == 1
    err = capsys.readouterr().err
    assert err == f"pyroid langid: {msg}\n  (while processing {d / 'b.parquet'})\n"


@pytest.mark.thread_unsafe(reason="monkeypatches pyroid.cli and reads stderr")
def test_warc_recipe_on_a_table_file_is_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyroid.cli as cli

    monkeypatch.setattr(cli, "_check_warc", lambda a: None)  # datatrove need not be installed
    monkeypatch.delenv("PYROID_DEBUG", raising=False)
    src = _write(_table(["x"]), tmp_path / "x.parquet")
    argv = ["clean", str(src), str(tmp_path / "y.parquet"), "--recipe", "fineweb-warc"]
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "reads WARC, ARC or WET files, not Parquet or JSON Lines" in err
    assert err.count("\n") == 1
    assert not list(tmp_path.glob(".y.parquet.pyroid"))
    # a file that does not read as a WARC (warcio's message): one line too
    from pyroid.warc import ArchiveLoadFailed

    def body(a: Any) -> int:
        raise ArchiveLoadFailed("Invalid ARC record, first line: PAR1")

    with monkeypatch.context() as m:  # undone here: --iterations reruns the test with patches on
        m.setattr(cli, "_cmd_clean", body)
        assert main(argv) == 1
    assert capsys.readouterr().err.startswith("pyroid clean: Invalid ARC record")


def test_translit_errors_takes_unidecodes_values(tmp_path: Path) -> None:
    src = _write(_table(["Ünïcödé"]), tmp_path / "in.parquet")
    out = str(tmp_path / "o.parquet")
    with pytest.raises(SystemExit) as e:
        main(["translit", str(src), out, "--mode", "unidecode", "--errors", "bogus"])
    assert e.value.code == 2  # a usage error, as for any bad choice
    with pytest.raises(SystemExit, match="apply to --mode unidecode"):
        main(["translit", str(src), out, "--errors", "replace"])
    with pytest.raises(SystemExit, match="apply to --mode unidecode"):
        main(["translit", str(src), out, "--replace-str", "_"])
    assert not Path(out).exists()
    assert main(["translit", str(src), out, "--mode", "unidecode", "--errors", "strict"]) == 0


@pytest.mark.thread_unsafe(reason="reads stderr and patches the JSON chunk size")
def test_a_jsonl_parse_error_names_its_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """pyarrow said "Invalid encoding in string. in row 1" for an unterminated string."""
    lines = ['{"text": "ok"}', '{"text": "fine"}', '{"text": "unterminated}']
    # LF, and CRLF (Windows' text mode writes it): the CR is not named as the problem
    for end in ("\n", "\r\n"):
        bad = tmp_path / "bad.jsonl"
        bad.write_bytes("".join(f"{ln}{end}" for ln in lines).encode())
        assert main(["translit", str(bad), str(tmp_path / "o.jsonl")]) == 1
        err = capsys.readouterr().err
        assert err == f"pyroid translit: {bad}: line 3: Unterminated string starting at column 10\n"
    good = tmp_path / "good.jsonl"
    good.write_bytes(b'{"text": "ok"}\r\n{"text": "fine"}\r\n')
    assert main(["translit", str(good), str(tmp_path / "g.jsonl")]) == 0
    assert [r["text"] for r in _read(tmp_path / "g.jsonl").to_pylist()] == ["ok", "fine"]
    # past the first chunk, the line number counts from the start of the file
    from pyroid.integrations import _io

    monkeypatch.setattr(_io, "JSON_CHUNK_SIZE", 4096)
    many = tmp_path / "many.jsonl"
    with many.open("w", encoding="utf-8") as f:
        f.writelines('{"text": "%s"}\n' % ("x" * 50) for _ in range(2_000))
        f.write('{"text": "x}\n')
    with pytest.raises(ValueError, match=r"line 2001: Unterminated string starting at column 10"):
        for _ in _io.iter_tables(_io.resolve_inputs([many])[0], 1000):
            pass


@pytest.mark.thread_unsafe(reason="reads stderr")
def test_a_failed_run_leaves_nothing_behind(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run that fails before finishing any file leaves no hidden state, work directory or
    directory OUT (each failed attempt left a .OUT.pyroid/)."""
    src = _write(_table([_english(1), _english(2)]), tmp_path / "in.parquet")
    missing = str(tmp_path / "missing" / "lid.bin")
    common = ["--recipe", "fineweb", "--lid-model", missing]
    tmp = tmp_path / "scratch"
    tmp.mkdir()
    for out in ("o.parquet", "d"):
        for extra in ([], ["--no-dedup"], ["--tmp-dir", str(tmp)]):
            argv = ["clean", str(src), str(tmp_path / out), *common, *extra]
            assert main(argv) == 1
            # One line naming the model (the OS's words differ: "[Errno 2] No such file or
            # directory" here, "[WinError 3] The system cannot find the path specified" on
            # Windows), and the note naming the input.
            first, note = capsys.readouterr().err.splitlines()
            assert first.startswith("pyroid clean: ")
            assert "lid.bin" in first
            assert note == f"  (while processing {src})"
            assert sorted(p.name for p in tmp_path.iterdir()) == ["in.parquet", "scratch"], argv
            assert list(tmp.iterdir()) == []
    # a directory OUT of another run keeps all it had, run.json included
    d = tmp_path / "d"
    assert main(["translit", str(src), str(d)]) == 0
    before = {p: p.read_bytes() for p in d.rglob("*") if p.is_file()}
    with pytest.raises(SystemExit, match="from a different run"):
        main(["translit", str(src), str(d), "--mode", "unidecode"])
    assert {p: p.read_bytes() for p in d.rglob("*") if p.is_file()} == before


def test_stats_into_a_closed_pipe(tmp_path: Path) -> None:
    """``--stats 2>&1 | head -1``: the run completes, tidies up and exits 0 (it exited 120 and
    left OUT/.pyroid/work-* behind)."""
    src = _write(_table(["Ünïcödé", "ascii"]), tmp_path / "in.parquet")
    out = tmp_path / "out"
    cmd = [sys.executable, "-m", "pyroid", "translit", str(src), str(out), "--stats", "--progress"]
    r, w = os.pipe()
    os.close(r)  # nobody reads stderr
    try:
        proc = subprocess.run(cmd, stderr=w, stdout=subprocess.DEVNULL, check=False)
    finally:
        os.close(w)
    assert proc.returncode == 0
    assert _read(out / "in.parquet").column("text").to_pylist() == ["Unicode", "ascii"]
    assert not list((out / ".pyroid").glob("work-*"))
    # pyroid stats into a closed stdout: exit 1 quietly, as Python's tools do
    r, w = os.pipe()
    os.close(r)
    try:
        stats = subprocess.run(
            [sys.executable, "-m", "pyroid", "stats", str(out)],
            stdout=w,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    finally:
        os.close(w)
    assert stats.returncode == 1
    assert stats.stderr == ""


class _ClosedPipe:
    """A stream whose writes fail as a write to a pipe nobody reads does on Windows."""

    def write(self, text: str) -> int:
        raise OSError(errno.EINVAL, "Invalid argument")

    def flush(self) -> None:
        pass

    def fileno(self) -> int:
        raise OSError("no descriptor")  # mute() gives up quietly


@pytest.mark.thread_unsafe(reason="replaces sys.stderr and sys.stdout")
def test_a_closed_pipe_on_windows_is_einval_not_epipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows' C runtime reports a write to a closed pipe as EINVAL; the run exited 120 there."""
    from pyroid.integrations import _shards

    src = _write(_table(["a", "b"]), tmp_path / "in.parquet")
    with monkeypatch.context() as m:
        m.setattr(_shards, "_muted", None)
        m.setattr(_shards, "_WINDOWS", True)
        m.setattr(sys, "stderr", _ClosedPipe())
        _shards.say("lost\n")  # no error: stderr is muted from here
        assert _shards._muted is sys.stderr
        assert main(["translit", str(src), str(tmp_path / "o"), "--stats"]) == 0
        m.setattr(sys, "stdout", _ClosedPipe())
        assert main(["stats", str(tmp_path / "o")]) == 1  # quietly, as for EPIPE
    with monkeypatch.context() as m:
        # elsewhere EINVAL is another error, which is not silenced
        m.setattr(_shards, "_muted", None)
        m.setattr(_shards, "_WINDOWS", False)
        m.setattr(sys, "stderr", _ClosedPipe())
        with pytest.raises(OSError, match="Invalid argument"):
            _shards.say("x\n")


@pytest.mark.thread_unsafe(reason="replaces sys.stderr")
def test_messages_go_to_a_new_stderr_after_a_closed_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A closed stderr silenced every later message of the process, to any stderr."""
    from pyroid.integrations import _shards

    with monkeypatch.context() as m:
        m.setattr(_shards, "_muted", None)
        m.setattr(_shards, "_WINDOWS", True)
        m.setattr(sys, "stderr", _ClosedPipe())
        _shards.say("lost\n")
        new = io.StringIO()
        m.setattr(sys, "stderr", new)
        _shards.say("seen\n")
        assert new.getvalue() == "seen\n"


def test_the_program_leaves_late_signals_their_default_action(tmp_path: Path) -> None:
    """Once its command has finished, ``pyroid`` lets SIGINT and SIGTERM end the process.

    A signal during interpreter shutdown was a KeyboardInterrupt in an atexit handler (and its
    traceback), a lost signal (exit 0), or, after CPython reset the handlers, death by the signal:
    now it is always the last. ``sys.unraisablehook``, which the program sets for the run, is put
    back. main() called with arguments, from Python code, leaves the handlers as they were.
    """
    src = _write(_table(["a"]), tmp_path / "in.parquet")
    code = (
        "import signal, sys\n"
        "from pyroid.cli import main\n"
        f"sys.argv = ['pyroid', 'translit', {str(src)!r}, {str(tmp_path / 'o')!r}]\n"
        "rc = main()\n"
        "dfl = [signal.getsignal(s) is signal.SIG_DFL for s in (signal.SIGINT, signal.SIGTERM)]\n"
        "print(rc, *dfl, sys.unraisablehook is sys.__unraisablehook__)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert (r.returncode, r.stdout, r.stderr) == (0, "0 True True True\n", "")
    before = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    assert main(["translit", str(src), str(tmp_path / "o2")]) == 0
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="POSIX signals")
def test_a_signal_raised_where_python_cannot_raise_it_comes_again(
    run_python: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    """Ctrl-C handled inside a weakref callback: Python printed "Exception ignored" and dropped
    the KeyboardInterrupt (a worker's, at the end of a run). The main process gets the signal
    again just after; a worker, which has noted it, drops it quietly."""
    r = run_python(
        """
        import signal, sys, time, weakref
        from pyroid.integrations import _shards

        def term(signum, frame):
            raise _shards.Terminated

        def swallowed(exc):
            class Obj:
                pass

            def callback(ref):
                raise exc  # as a signal handler that runs here would

            obj = Obj()
            ref = weakref.ref(obj, callback)
            del obj  # the callback runs; its exception cannot propagate

        signal.signal(signal.SIGTERM, term)
        sys.unraisablehook = _shards.redelivering(sys.unraisablehook)
        for exc in (KeyboardInterrupt, _shards.Terminated):
            try:
                swallowed(exc)
                time.sleep(10)
                print("not delivered")
            except KeyboardInterrupt as e:
                print(type(e).__name__)
        _shards._interrupted = True  # as in a worker that took its signal
        swallowed(KeyboardInterrupt)
        time.sleep(0.2)
        print("dropped")
        """
    )
    assert (r.returncode, r.stdout, r.stderr) == (0, "KeyboardInterrupt\nTerminated\ndropped\n", "")


# ----------------------------------------------------------------------------------------- dedup


def test_dedup_exact(tmp_path: Path) -> None:
    texts = ["a", "b", "a", "c", "b", "A "]
    src = _write(_table(texts), tmp_path / "in.jsonl")
    out, excl = tmp_path / "out.parquet", tmp_path / "dup.parquet"
    stats = tmp_path / "stats.json"
    args = ["dedup", str(src), str(out), "--method", "exact", "--excluded", str(excl)]
    assert main([*args, "--batch-size", "2", "--stats-json", str(stats)]) == 0
    assert pq.read_table(out).column("text").to_pylist() == ["a", "b", "c", "A "]
    x = pq.read_table(excl)
    assert x.column("id").to_pylist() == ["d2", "d4"]
    assert x.column("filter_reason").to_pylist() == ["exact", "exact"]
    s = json.loads(stats.read_text())
    assert (s["docs_in"], s["docs_out"], s["removed"]) == (6, 4, {"exact": 2})
    out2 = tmp_path / "out2.parquet"
    assert main(["dedup", str(src), str(out2), "--method", "exact", "--normalize"]) == 0
    assert pq.read_table(out2).column("text").to_pylist() == ["a", "b", "c"]


def test_dedup_minhash_matches_library(tmp_path: Path) -> None:
    texts = []
    for i in range(40):
        doc = f"Document number {i} says: " + BASE * 2
        texts.append(doc)
        if i % 3 == 0:
            texts.append(doc.replace("lazy", "sleepy"))  # a near-duplicate
    d = tmp_path / "in"
    d.mkdir()
    _write(_table(texts[:30]), d / "0.parquet")
    _write(_table(texts[30:]), d / "1.parquet")
    out, excl = tmp_path / "out.parquet", tmp_path / "dup.parquet"
    cfg = ["--hash-fc", "sha1", "--num-buckets", "14", "--hashes-per-bucket", "8"]
    assert (
        main(["dedup", str(d), str(out), "--excluded", str(excl), "--batch-size", "7", *cfg]) == 0
    )
    mask = pyroid.dedup.minhash_dedup(texts, hash_fc="sha1", num_buckets=14, hashes_per_bucket=8)
    assert pq.read_table(out).column("text").to_pylist() == [
        t for t, k in zip(texts, mask, strict=True) if k
    ]
    assert pq.read_table(excl).num_rows == mask.count(False) > 0


# ---------------------------------------------------------------------------------------- langid


def _ftz() -> Path:
    for m in (LID_FTZ, LID_BIN):
        if m.exists():
            return m
    pytest.skip("no pinned lid.176 model")


def test_langid(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    import pyroid.fasttext

    model = _ftz()
    texts = [
        "This is an English sentence about the weather.",
        "Ceci est une phrase en français.",
        "Guten Morgen, wie geht es dir?\nMit zwei Zeilen.",
        "これは日本語の文です。",
    ]
    src = _write(_table(texts, language=["xx"] * 4), tmp_path / "in.parquet")
    out = tmp_path / "out.parquet"
    assert main(["langid", str(src), str(out), "--model", str(model), "--top-k", "3"]) == 0
    t = pq.read_table(out)
    m = pyroid.fasttext.load_model(model)
    want = [m.predict(x.replace("\n", " "), k=3) for x in texts]
    assert t.column_names == [
        "id",
        "text",
        "language",
        "language_score",
        "top_languages",
        "top_language_scores",
    ]
    assert t.column("language").to_pylist() == [lb[0].removeprefix("__label__") for lb, _ in want]
    assert t.column("language_score").to_pylist() == [float(p[0]) for _, p in want]
    assert t.column("top_languages").to_pylist() == [
        [x.removeprefix("__label__") for x in lb] for lb, _ in want
    ]
    kept, excl = tmp_path / "en.parquet", tmp_path / "other.jsonl"
    args = [
        "langid",
        str(src),
        str(kept),
        "--model",
        str(model),
        "--keep",
        "en,fr",
        "--min-score",
        "0.5",
    ]
    assert main([*args, "--excluded", str(excl)]) == 0
    assert pq.read_table(kept).column("language").to_pylist() == ["en", "fr"]
    rows = [json.loads(x) for x in excl.read_text().splitlines()]
    assert [r["filter_reason"] for r in rows] == ["lid", "lid"]


# ----------------------------------------------------------------------------------------- clean


def _digest_module() -> Any:
    if not DIGEST_SCRIPT.exists():
        pytest.skip("examples/cli/digest.py is not in this checkout")
    spec = importlib.util.spec_from_file_location("pyroid_cli_digest", DIGEST_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    ("batch_size", "threads", "out_ext"), [(10_000, "0", ".parquet"), (3001, "1", ".jsonl.gz")]
)
def test_clean_fineweb_is_digest_identical_to_the_chain(
    tmp_path: Path, batch_size: int, threads: str, out_ext: str
) -> None:
    pytest.importorskip("numpy")
    if not (DEV.exists() and LID_BIN.exists()):
        pytest.skip("needs the pinned dev slice and lid.176.bin under $PYROID_DATA")
    digest = _digest_module()
    out, excl = tmp_path / f"kept{out_ext}", tmp_path / "dropped.parquet"
    stats = tmp_path / "stats.json"
    args = ["clean", str(DEV), str(out), "--recipe", "fineweb", "--lid-model", str(LID_BIN)]
    args += ["--excluded", str(excl), "--batch-size", str(batch_size), "--threads", threads]
    assert main([*args, "--stats-json", str(stats)]) == 0
    assert digest.digest(str(DEV), str(out), str(excl)) == DEV_DIGEST
    s = json.loads(stats.read_text())
    assert (s["docs_in"], s["docs_out"]) == (10_000, 8_407)
    assert s["removed_by_stage"] == {
        "gopher_quality": 1524,
        "fineweb_quality": 27,
        "gopher_repetition": 24,
        "c4_quality": 14,
        "lid": 4,
    }
    t = _read(out)
    assert t.column_names == pq.read_schema(DEV).names  # language/_score replaced in place
    assert not list(tmp_path.glob(".pyroid-*"))  # the dedup spool is gone


def test_clean_with_no_survivor_keeps_the_columns(tmp_path: Path) -> None:
    """When the filters drop every row, the output still has the input's columns."""
    pytest.importorskip("numpy")
    if not LID_BIN.exists():
        pytest.skip("needs lid.176.bin")
    src = _write(_table(["too short"], url=["https://example.org/"]), tmp_path / "in.parquet")
    for out, extra in [("o.parquet", []), ("o2.parquet", ["--no-dedup"])]:
        args = ["clean", str(src), str(tmp_path / out), "--recipe", "fineweb", "--lid-model"]
        assert main([*args, str(LID_BIN), *extra]) == 0
        t = pq.read_table(tmp_path / out)
        assert t.num_rows == 0
        assert t.column_names == ["id", "text", "url", "language", "language_score"], extra


def test_clean_small_no_dedup_and_stats(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("numpy")
    if not (DEV.exists() and LID_BIN.exists()):
        pytest.skip("needs the pinned dev slice and lid.176.bin")
    from pyroid.integrations.recipes import FineWeb

    sample = pq.read_table(DEV, columns=["text"]).slice(0, 20).column("text")
    good = sample.filter(FineWeb(lid_model=str(LID_BIN)).annotate(sample).keep)[0].as_py()
    texts = [good, "too short", None, "Ceci est un texte en français, " * 20, good]
    src = _write(_table(texts), tmp_path / "in.parquet")
    out, excl = tmp_path / "out.parquet", tmp_path / "x.parquet"
    args = ["clean", str(src), str(out), "--recipe", "fineweb", "--lid-model", str(LID_BIN)]
    assert main([*args, "--excluded", str(excl), "--stats"]) == 0
    err = capsys.readouterr().err
    assert "pyroid clean: 5 in" in err
    assert "dropped lid" in err
    reasons = dict(
        zip(
            pq.read_table(excl).column("id").to_pylist(),
            pq.read_table(excl).column("filter_reason").to_pylist(),
            strict=True,
        )
    )
    assert reasons["d2"] == "null"
    assert reasons["d3"] == "lid"
    assert reasons["d1"].startswith(("gopher_", "lid"))
    assert reasons.get("d4") == "minhash"  # an exact copy of d0
    out2 = tmp_path / "out2.parquet"
    assert main(["clean", str(src), str(out2), *args[3:], "--no-dedup"]) == 0
    assert pq.read_table(out2).column("id").to_pylist() == ["d0", "d4"]


def test_clean_ftfy_config(tmp_path: Path) -> None:
    """--ftfy picks ftfy's defaults, datatrove's FTFYFormatter() config, or no ftfy step."""
    pytest.importorskip("numpy")
    if not (DEV.exists() and LID_BIN.exists()):
        pytest.skip("needs the pinned dev slice and lid.176.bin")
    from pyroid.integrations.recipes import FTFY_FORMATTER, FineWeb

    sample = pq.read_table(DEV, columns=["text"]).slice(0, 200).column("text").to_pylist()
    src = _write(_table(sample), tmp_path / "in.parquet")
    args = ["--recipe", "fineweb", "--lid-model", str(LID_BIN), "--no-dedup"]
    texts = {}
    for mode in ["default", "datatrove", "off"]:
        out = tmp_path / f"{mode}.parquet"
        assert main(["clean", str(src), str(out), *args, "--ftfy", mode]) == 0
        texts[mode] = dict(
            zip(
                pq.read_table(out).column("id").to_pylist(),
                pq.read_table(out).column("text").to_pylist(),
                strict=True,
            )
        )
    for mode, config in [("default", None), ("datatrove", FTFY_FORMATTER), ("off", False)]:
        ann = FineWeb(lid_model=str(LID_BIN), ftfy_config=config, minhash=None).annotate(
            pa.chunked_array([sample])
        )
        want = {
            f"d{i}": t
            for i, (t, k) in enumerate(zip(ann.text.to_pylist(), ann.keep.to_pylist(), strict=True))
            if k
        }
        assert texts[mode] == want, mode
    # FineWeb's curly quotes stay with datatrove's config and are straightened by ftfy's.
    curly = [i for i, t in texts["datatrove"].items() if "\u2019" in t]
    assert curly
    assert all("\u2019" not in texts["default"].get(i, "") for i in curly)
    with pytest.raises(SystemExit, match="--ftfy applies to --recipe fineweb"):
        main(
            [
                "clean",
                str(src),
                str(tmp_path / "w.parquet"),
                "--recipe",
                "fineweb-warc",
                "--ftfy",
                "off",
            ]
        )


def test_ftfy_formatter_config_is_datatroves() -> None:
    ftfy_fmt = pytest.importorskip("datatrove.pipeline.formatters.ftfy")
    pytest.importorskip("ftfy")
    from pyroid.integrations.recipes import FTFY_FORMATTER

    assert tuple(ftfy_fmt.FTFYFormatter().config) == tuple(FTFY_FORMATTER)


# ---------------------------------------------------------------------- the dedup spool's cleanup
# Windows can't delete a file this process still has open or memory-mapped (PermissionError),
# where Linux and macOS can. `strict_unlink` makes os.unlink of pyroid's spools behave the Windows
# way on Linux (from /proc/self/fd and /proc/self/maps); on Windows the OS itself is the check.
# A whole `clean` run under the same check is in test_cli_scale.py.


def _held(path: str) -> list[str]:
    """How this process still holds ``path``: open descriptors and memory maps (Linux)."""
    real = os.path.realpath(path)
    held = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.readlink(f"/proc/self/fd/{fd}") == real:
                held.append(f"fd {fd}")
        except OSError:  # the descriptor listdir itself used, already closed
            continue
    with open("/proc/self/maps", encoding="utf-8", errors="replace") as f:
        held += [f"mapped at {ln.split()[0]}" for ln in f if ln.rstrip("\n").endswith(" " + real)]
    return held


@pytest.fixture
def strict_unlink(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The spool paths os.unlink removed; it raises if one was still open or mapped."""
    linux = Path("/proc/self/maps").exists()
    if not linux and sys.platform != "win32":
        pytest.skip("needs /proc (Linux) to see open files and maps, or Windows itself")
    real_unlink = os.unlink
    removed: list[str] = []

    def unlink(path: Any, *args: Any, **kwargs: Any) -> None:
        p = os.fspath(path)
        if p.endswith((".arrow", ".arrow.partial")) and os.path.exists(p):
            held = _held(p) if linux else []
            if held:
                raise PermissionError(f"{p} is still {', '.join(held)} (Windows refuses this)")
            removed.append(p)
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    return removed


def _removed_under(removed: list[str], where: Path) -> list[str]:
    """The spools in ``removed`` that were under ``where``.

    pytest-run-parallel's ``--iterations`` reruns a test with the same fixture values but a fresh
    ``tmp_path``, so ``strict_unlink`` also holds the earlier runs' spools.
    """
    return [p for p in removed if Path(p).is_relative_to(where)]


def _spool_tables() -> list[Any]:
    texts = [f"{BASE} {i}" for i in range(9)]
    return [_table(texts[:4]), _table(texts[4:5]), _table(texts[5:])]


@pytest.mark.thread_unsafe  # patches os.unlink
def test_spool_is_closed_before_it_is_removed(tmp_path: Path, strict_unlink: list[str]) -> None:
    from pyroid.cli import _Ipc, _read_ipc

    tables = _spool_tables()
    path = str(tmp_path / "kept.arrow")
    spool = _Ipc(path)
    for t in tables:
        spool.write(t)
    spool.close()
    back = list(_read_ipc(path))  # the tables are still referenced, and must not pin the file
    os.unlink(path)
    assert _removed_under(strict_unlink, tmp_path) == [path]
    assert not list(tmp_path.iterdir())
    assert pa.concat_tables(back).equals(pa.concat_tables(tables))
    # stopping early closes the file too
    spool = _Ipc(path)
    for t in tables:
        spool.write(t)
    spool.close()
    for _ in _read_ipc(path):
        break
    os.unlink(path)


@pytest.mark.thread_unsafe
def test_spool_abort_removes_the_partial_file(tmp_path: Path, strict_unlink: list[str]) -> None:
    from pyroid.cli import _Ipc

    path = str(tmp_path / "kept.arrow")
    spool = _Ipc(path)
    spool.write(_spool_tables()[0])  # the IPC writer and its file are still open
    spool.abort()
    spool.abort()  # idempotent
    assert _removed_under(strict_unlink, tmp_path) == [path + ".partial"]
    assert not list(tmp_path.iterdir())


@pytest.fixture(scope="module")
def en_lid(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic fastText model whose only label is ``__label__en`` (every text is English)."""
    pytest.importorskip("numpy")
    import dataclasses

    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from test_fasttext import SPECS, build
    finally:
        sys.path.pop(0)
    path = tmp_path_factory.mktemp("lid") / "en.bin"
    path.write_bytes(
        build(dataclasses.replace(SPECS["softmax-dense"], labels=("__label__en",))).data
    )
    return path


def _english(i: int) -> str:
    lines = [
        f"The council of the city met on Tuesday {i} and talked about the plans for the new park.",
        "Many people from the north side came to the meeting, and they asked about the costs.",
        "The mayor said that the work would start in the spring if the weather is good enough.",
        f"Residents who live near the river were worried about traffic during the {i} months.",
        "After a long discussion, the members agreed to hold another meeting before the vote.",
    ]
    return "\n".join(lines)


def test_clean_into_one_file_from_inputs_with_other_schemas(tmp_path: Path, en_lid: Path) -> None:
    # Each file is read with its own schema; one output file takes them all, matched by name
    # (it failed at the very end with "Target schema's field names are not matching").
    src = tmp_path / "in"
    src.mkdir()
    a = pa.table({"text": [_english(i) for i in range(30)], "id": [f"a{i}" for i in range(30)]})
    b = pa.table(
        {"id": [f"b{i}" for i in range(30)], "text": [_english(100 + i) for i in range(30)]}
    )
    c = b.append_column("source", pa.array(["x"] * 30)).rename_columns(["id", "text", "source"])
    for name, t in (("a", a), ("b", b), ("c", c)):
        pq.write_table(t.replace_schema_metadata(None), src / f"{name}.parquet")
    common = ["--recipe", "fineweb", "--lid-model", str(en_lid), "--no-dedup"]
    assert main(["clean", str(src), str(tmp_path / "dir"), *common]) == 0
    assert main(["clean", str(src), str(tmp_path / "one.parquet"), *common]) == 0
    one = pq.read_table(tmp_path / "one.parquet")
    assert one.column_names[:2] == ["text", "id"]
    assert "source" in one.column_names
    want = [
        {**dict.fromkeys(one.column_names), **row}
        for n in ("a", "b", "c")
        for row in pq.read_table(tmp_path / "dir" / f"{n}.parquet").to_pylist()
    ]
    assert len(want) > 0
    assert one.to_pylist() == want
    assert [p.name for p in tmp_path.glob(".one.parquet.pyroid/done/*")] == [
        "merged-00000-of-00001.json"
    ]


# ------------------------------------------------- options and inputs checked before any work


@pytest.mark.thread_unsafe(reason="reads stderr")
def test_a_bad_compression_is_refused_before_any_work(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], en_lid: Path
) -> None:
    # It failed at the first write, after the whole MinHash phase: the work it left behind made
    # the corrected command refuse to run ("comes from a different run") without --overwrite.
    src = _write(_table([_english(i) for i in range(20)]), tmp_path / "in.parquet")
    out = tmp_path / "out.parquet"
    args = ["clean", str(src), str(out), "--recipe", "fineweb", "--lid-model", str(en_lid)]
    with pytest.raises(SystemExit) as e:
        main([*args, "--compression", "bogus"])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "pyroid clean: error: argument --compression: Unsupported compression: bogus" in err
    assert not (tmp_path / ".out.parquet.pyroid").exists()
    assert main([*args, "--compression", "zstd"]) == 0
    assert pq.ParquetFile(out).metadata.row_group(0).column(0).compression == "ZSTD"
    for name in ("NONE", "snappy", "gzip"):
        assert main([*args, "--compression", name, "--overwrite"]) == 0


@pytest.mark.thread_unsafe(reason="monkeypatches pyroid.integrations._io and reads stderr")
def test_a_refused_run_names_the_unfinished_work_when_out_is_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    en_lid: Path,
) -> None:
    from pyroid.integrations import _io

    src = _write(_table([_english(i) for i in range(20)]), tmp_path / "in.parquet")
    out = tmp_path / "out.parquet"
    args = ["clean", str(src), str(out), "--recipe", "fineweb", "--lid-model", str(en_lid)]

    def full(self: Any, table: Any) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    with monkeypatch.context() as m:
        m.setattr(_io.Writer, "write", full)
        assert main(args) == 1  # after the MinHash phase: its records stay, for a resume
    assert not out.exists()
    capsys.readouterr()
    with pytest.raises(
        SystemExit, match=r"the unfinished work in .*out\.parquet\.pyroid comes from"
    ):
        main([*args, "--compression", "zstd"])
    assert main(args) == 0  # the same command resumes
