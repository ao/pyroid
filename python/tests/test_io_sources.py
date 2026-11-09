"""pyroid.integrations._io: inputs (local, fsspec, pyarrow), streaming reads, atomic outputs."""

from __future__ import annotations

import contextlib
import functools
import gzip
import json
import os
import pickle
import re
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pafs = pytest.importorskip("pyarrow.fs")

from pyroid.integrations import _io  # noqa: E402

LINUX = Path("/proc/self/fd").is_dir()
# The originals, for tests that wrap them (pytest-run-parallel reruns a test with its patches on).
REAL_REPLACE = os.replace
REAL_UNLINK = os.unlink


def _table(n: int, start: int = 0) -> Any:
    ids = list(range(start, start + n))
    return pa.table({"id": ids, "text": [f"doc {i}" for i in ids], "n": [i * 0.5 for i in ids]})


def _lines(table: Any) -> bytes:
    return b"".join(json.dumps(r).encode() + b"\n" for r in table.to_pylist())


def _write(path: Path, table: Any, row_group_size: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".parquet"):
        pq.write_table(table, path, row_group_size=row_group_size)
    elif path.name.endswith(".gz"):
        path.write_bytes(gzip.compress(_lines(table)))
    else:
        path.write_bytes(_lines(table))
    return path


def _names(sources: list[Any]) -> list[str]:
    return [s.name for s in sources]


def _rows(tables: Any) -> list[int]:
    return [t.num_rows for t in tables]


def _held(path: str | Path) -> list[str]:
    """This process's open descriptors and memory maps of ``path`` (Linux)."""
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
def tree(tmp_path: Path) -> Path:
    """A dataset directory with data files, a nested one, and files that must be skipped."""
    d = tmp_path / "d"
    _write(d / "a.parquet", _table(3))
    _write(d / "b.jsonl", _table(2, 3))
    _write(d / "sub" / "c.parquet", _table(4, 5))
    _write(d / "sub" / "deeper" / "e.jsonl.gz", _table(1, 9))
    (d / "notes.txt").write_text("not data")
    _write(d / ".hidden" / "x.parquet", _table(1))
    _write(d / "_tmp" / "y.parquet", _table(1))
    _write(d / ".a.parquet", _table(1))
    _write(d / "sub" / "_delta_log" / "0.json", _table(1))
    (d / "_SUCCESS").write_bytes(b"")
    return d


ALL = ["a.parquet", "b.jsonl", "sub/c.parquet", "sub/deeper/e.jsonl.gz"]

# ------------------------------------------------------------------------------- local inputs


def test_directory_is_listed_recursively(tree: Path) -> None:
    found = _io.resolve_inputs([tree])
    assert _names(found) == ALL
    assert [s.format for s in found] == ["parquet", "jsonl", "parquet", "jsonl"]
    assert [s.uri for s in found] == [str(tree.joinpath(*n.split("/"))) for n in ALL]
    assert [s.size for s in found] == [os.path.getsize(s.uri) for s in found]
    # A file URI is the same local directory.
    assert _io.resolve_inputs([tree.as_uri()]) == found


def test_single_file(tree: Path) -> None:
    (src,) = _io.resolve_inputs([str(tree / "sub" / "c.parquet")])
    assert (src.uri, src.name, src.format) == (
        str(tree / "sub" / "c.parquet"),
        "c.parquet",
        "parquet",
    )
    # Named explicitly, a hidden file is read.
    assert _names(_io.resolve_inputs([tree / ".a.parquet"])) == [".a.parquet"]


@pytest.mark.parametrize(
    ("pattern", "names"),
    [
        ("*.parquet", ["a.parquet"]),
        ("?.jsonl", ["b.jsonl"]),
        ("[ab].*", ["a.parquet", "b.jsonl"]),
        ("[!a].*", ["b.jsonl"]),
        ("**/*.parquet", ["a.parquet", "sub/c.parquet"]),
        ("**/*.jsonl*", ["b.jsonl", "sub/deeper/e.jsonl.gz"]),
        ("**", ALL),
        # A matched directory stands for everything below it, as when the shell expands it.
        ("*", ALL),
        ("su?/*", ["sub/c.parquet", "sub/deeper/e.jsonl.gz"]),
        ("sub/**/e.*", ["deeper/e.jsonl.gz"]),  # the root is "sub", the literal prefix
        ("**/deeper/*", ["sub/deeper/e.jsonl.gz"]),
        ("*.PARQUET", None),  # case-sensitive everywhere
    ],
)
def test_globs(tree: Path, pattern: str, names: list[str] | None) -> None:
    spec = f"{tree}/{pattern}"
    if names is None:
        with pytest.raises(FileNotFoundError, match=r"no \.parquet or \.jsonl files"):
            _io.resolve_inputs([spec])
        return
    found = _io.resolve_inputs([spec])
    assert _names(found) == names
    root = tree / "sub" if pattern.startswith("sub/") else tree
    assert [s.uri for s in found] == [str(root.joinpath(*n.split("/"))) for n in names]


@pytest.mark.thread_unsafe  # changes the working directory
def test_relative_glob(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tree)
    found = _io.resolve_inputs(["*.jsonl", "sub"])
    assert [s.uri for s in found] == [
        "b.jsonl",
        os.path.join("sub", "c.parquet"),
        os.path.join("sub", "deeper", "e.jsonl.gz"),
    ]
    assert _io.resolve_inputs([Path("a.parquet")])[0].uri == "a.parquet"


def test_name_order_and_spec_order(tmp_path: Path, tree: Path) -> None:
    d = tmp_path / "order"
    for name in ["a0.parquet", "B.parquet", "a/x.parquet", "a.parquet", "a-1.parquet"]:
        _write(d / name, _table(1))
    # Plain string order of the names ("/" < "0", "B" < "a").
    assert _names(_io.resolve_inputs([d])) == [
        "B.parquet",
        "a-1.parquet",
        "a.parquet",
        "a/x.parquet",
        "a0.parquet",
    ]
    # Across specs, the order given.
    found = _io.resolve_inputs([tree / "sub", tree / "a.parquet", f"{tree}/*.jsonl"])
    assert _names(found) == ["c.parquet", "deeper/e.jsonl.gz", "a.parquet", "b.jsonl"]


@pytest.mark.parametrize(
    "specs",
    [
        ["{d}", "{d}/a.parquet"],
        ["{d}/a.parquet", "{d}/sub/../a.parquet"],
        ["{d}/*.parquet", "{d}/**/a.parquet"],
        ["{d}/sub", "{d}/sub/deeper"],
    ],
    ids=["dir-file", "same-file", "two-globs", "nested-dirs"],
)
def test_listed_twice(tree: Path, specs: list[str]) -> None:
    with pytest.raises(ValueError, match="listed twice"):
        _io.resolve_inputs([s.format(d=tree) for s in specs])


def test_no_match_and_bad_specs(tmp_path: Path, tree: Path) -> None:
    (tmp_path / "empty").mkdir()
    (tmp_path / "text").mkdir()
    (tmp_path / "text" / "a.txt").write_text("x")
    specs: list[str | Path] = [tmp_path / "empty", tmp_path / "text", f"{tree}/*.csv"]
    for spec in [*specs, f"{tmp_path}/nope/*"]:
        with pytest.raises(FileNotFoundError, match=r": no \.parquet or \.jsonl files"):
            _io.resolve_inputs([spec])
    with pytest.raises(FileNotFoundError, match=r"missing\.parquet: no such file or directory"):
        _io.resolve_inputs([tmp_path / "missing.parquet"])
    with pytest.raises(ValueError, match=r"notes\.txt: unknown format"):
        _io.resolve_inputs([tree / "notes.txt"])
    assert _io.resolve_inputs([]) == []


@pytest.mark.thread_unsafe  # changes the working directory
def test_exclude(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tree / "out" / "a.parquet", _table(1))
    _write(tree / "work" / "x.parquet", _table(1))
    found = _io.resolve_inputs([tree], exclude=[tree / "out", tree / "work", tree / "missing"])
    assert _names(found) == ALL
    found = _io.resolve_inputs([f"{tree}/**/a.parquet"], exclude=[tree / "out"])
    assert _names(found) == ["a.parquet"]
    assert _names(_io.resolve_inputs([tree], exclude=[tree / "sub" / "c.parquet"])) == [
        "a.parquet",
        "b.jsonl",
        "out/a.parquet",
        "sub/deeper/e.jsonl.gz",
        "work/x.parquet",
    ]
    # Spelled differently (relative, with ".."), the same directory.
    with monkeypatch.context() as m:
        m.chdir(tree / "sub")
        found = _io.resolve_inputs([tree / "sub"], exclude=["deeper", "../out"])
        assert _names(found) == ["c.parquet"]
    with pytest.raises(FileNotFoundError, match=r"no \.parquet or \.jsonl files"):
        _io.resolve_inputs([tree / "sub"], exclude=[tree])


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_symlink_cycle_is_listed_once(tree: Path) -> None:
    with contextlib.suppress(FileExistsError):  # pytest-run-parallel: threads share the tree
        (tree / "sub" / "loop").symlink_to(tree, target_is_directory=True)
    assert _names(_io.resolve_inputs([tree])) == ALL


def test_input_files_and_read_tables(tmp_path: Path, tree: Path) -> None:
    files = _io.input_files(tree)
    assert files == [tree.joinpath(*n.split("/")) for n in ALL]
    # Later JSONL files are parsed with the first file's schema (n is a double).
    d = tmp_path / "j"
    _write(d / "1.jsonl", pa.table({"n": [0.5, 1.5]}))
    _write(d / "2.jsonl", pa.table({"n": [2, 3, 4]}))
    _write(d / "3.jsonl.gz", pa.table({"n": [5, 6, 7, 8]}))
    tables = list(_io.read_tables(_io.input_files(d), 4))
    assert _rows(tables) == [4, 4, 1]
    assert pa.concat_tables(tables).column("n").to_pylist() == [0.5, 1.5, 2, 3, 4, 5, 6, 7, 8]
    assert {t.schema.field("n").type for t in tables} == {pa.float64()}
    empty = _write(tmp_path / "e.parquet", _table(0))
    (t,) = _io.read_tables([empty], 4)
    assert t.num_rows == 0
    assert t.schema.names == ["id", "text", "n"]
    with pytest.raises(ValueError, match="reads local paths only"):
        _io.input_files("s3://bucket/x.parquet")


# ------------------------------------------------------------------------------- local reads


def test_iter_tables_exact_batches(tmp_path: Path) -> None:
    t = _table(25)
    src = _io.resolve_inputs([_write(tmp_path / "x.parquet", t, row_group_size=7)])[0]
    tables = list(_io.iter_tables(src, 10))
    assert _rows(tables) == [10, 10, 5]
    assert pa.concat_tables(tables).equals(t)
    assert _rows(_io.iter_tables(src, 7)) == [7, 7, 7, 4]
    assert _rows(_io.iter_tables(src, 25)) == [25]
    assert _rows(_io.iter_tables(src, 100)) == [25]
    (proj,) = _io.iter_tables(src, 100, columns=["n", "id"])
    assert proj.column_names == ["n", "id"]
    assert proj.column("id").to_pylist() == list(range(25))
    with pytest.raises(ValueError, match=r"x\.parquet: no column 'nope' \(columns: id, text, n\)"):
        list(_io.iter_tables(src, 10, columns=["nope"]))
    with pytest.raises(ValueError, match="batch_size"):
        list(_io.iter_tables(src, 0))
    assert _io.read_schema(src) == t.schema


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.gz", ".ndjson.zst"])
def test_iter_tables_jsonl(tmp_path: Path, suffix: str) -> None:
    t = _table(10)
    path = tmp_path / f"x{suffix}"
    if suffix.endswith(".zst"):
        with pa.output_stream(str(path), compression="zstd") as f:
            f.write(_lines(t))
    else:
        _write(path, t)
    (src,) = _io.resolve_inputs([path])
    tables = list(_io.iter_tables(src, 3))
    assert _rows(tables) == [3, 3, 3, 1]
    assert pa.concat_tables(tables).equals(t)
    (proj,) = _io.iter_tables(src, 100, columns=["text"])
    assert proj.column_names == ["text"]
    assert _io.read_schema(src) == t.schema


@pytest.mark.thread_unsafe  # patches the module's chunk sizes
def test_jsonl_chunks_cut_at_line_ends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Small chunks and blocks: lines cross chunk ends, and one line is longer than a block."""
    monkeypatch.setattr(_io, "JSON_CHUNK_SIZE", 50)
    monkeypatch.setattr(_io, "JSON_BLOCK_SIZE", 64)
    texts = [f"doc {i}" for i in range(40)]
    texts[17] = "x" * 1000
    t = pa.table({"id": list(range(40)), "text": texts})
    path = _write(tmp_path / "x.jsonl", t)
    path.write_bytes(path.read_bytes().replace(b"\n", b"\n\n", 3))  # blank lines are skipped
    (src,) = _io.resolve_inputs([path])
    tables = list(_io.iter_tables(src, 16))
    assert _rows(tables) == [16, 16, 8]
    assert pa.concat_tables(tables).equals(t)
    # A field the first chunk did not have is an error, as in pyarrow's streaming reader.
    path.write_bytes(_lines(t) + b'{"id": 40, "text": "x", "extra": 1}\n')
    with pytest.raises(ValueError, match=r"x\.jsonl: .*unexpected field"):
        list(_io.iter_tables(src, 16))


def test_empty_files(tmp_path: Path) -> None:
    _write(tmp_path / "e.parquet", _table(0))
    j = tmp_path / "e.jsonl"
    j.write_bytes(b"")
    g = tmp_path / "e.jsonl.gz"
    g.write_bytes(gzip.compress(b"\n"))
    z = tmp_path / "z.jsonl.gz"
    z.write_bytes(b"")
    sj, sg, sp, sz = _io.resolve_inputs([tmp_path])
    (t,) = _io.iter_tables(sp, 10)
    assert t.num_rows == 0
    assert t.schema == _table(0).schema
    (t,) = _io.iter_tables(sp, 10, columns=["text"])
    assert t.schema.names == ["text"]
    for s in (sj, sg, sz):
        assert list(_io.iter_tables(s, 10)) == []
        assert _io.read_schema(s) is None


def test_parse_errors_name_the_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"not parquet at all")
    (src,) = _io.resolve_inputs([bad])
    with pytest.raises(ValueError, match=r"bad\.parquet: "):
        list(_io.iter_tables(src, 10))
    with pytest.raises(ValueError, match=r"bad\.parquet: "):
        _io.read_schema(src)
    j = tmp_path / "bad.jsonl"
    j.write_bytes(b'{"a": 1}\n{"a": \n')
    (src,) = _io.resolve_inputs([j])
    with pytest.raises(ValueError, match=r"bad\.jsonl: "):
        list(_io.iter_tables(src, 10))


@pytest.mark.skipif(not LINUX, reason="needs /proc/self/fd to see open files")
@pytest.mark.parametrize("name", ["x.parquet", "x.jsonl.gz", "x.jsonl"])
def test_iter_tables_closes_its_file(tmp_path: Path, name: str) -> None:
    path = _write(tmp_path / name, _table(20), row_group_size=5)
    (src,) = _io.resolve_inputs([path])
    g = _io.iter_tables(src, 2)
    assert next(g).num_rows == 2
    assert _held(path)  # open while the consumer iterates
    g.close()  # the consumer stops early
    assert _held(path) == []
    assert sum(_rows(_io.iter_tables(src, 3))) == 20
    assert _held(path) == []
    assert _io.read_schema(src) is not None
    assert _held(path) == []
    old = _io.read_tables([path], 2)  # the same for the older, multi-file reader
    assert next(old).num_rows == 2
    assert _held(path)
    old.close()
    assert _held(path) == []


# ---------------------------------------------------------------------------- fsspec memory://


# The memory filesystem hands every reader of a file the same file object (one position), so
# the tests that read it run on one thread (thread_unsafe) under pytest-run-parallel.
Mem = tuple[Any, str, dict[str, Any]]


@pytest.fixture
def mem() -> Iterator[Mem]:
    """fsspec's in-memory filesystem, a fresh root in it (the store is process-global), the data."""
    fsspec = pytest.importorskip("fsspec")
    fs = fsspec.filesystem("memory")
    root = f"/pyroid-test-{uuid.uuid4().hex}"
    yield fs, root, _mem_dataset(fs, root)
    fs.rm(root, recursive=True)


def _mem_dataset(fs: Any, root: str) -> dict[str, Any]:
    tables = {
        "a.parquet": _table(25),
        "sub/b.jsonl.gz": _table(10, 25),
        "sub/c.jsonl": _table(8, 35),
    }
    for name, t in tables.items():
        with fs.open(f"{root}/data/{name}", "wb") as f:
            if name.endswith(".parquet"):
                pq.write_table(t, f, row_group_size=7)
            else:
                f.write(gzip.compress(_lines(t)) if name.endswith(".gz") else _lines(t))
    with fs.open(f"{root}/data/empty.parquet", "wb") as f:
        pq.write_table(_table(0), f)
    fs.pipe(f"{root}/data/empty.jsonl", b"")
    fs.pipe(f"{root}/data/.cache/x.parquet", b"skipped")
    fs.pipe(f"{root}/data/_SUCCESS", b"")
    fs.pipe(f"{root}/data/README.md", b"# skipped")
    return tables


MEM_ALL = ["a.parquet", "empty.jsonl", "empty.parquet", "sub/b.jsonl.gz", "sub/c.jsonl"]


@pytest.mark.thread_unsafe  # the memory filesystem (see Mem)
def test_memory_directory_glob_and_file(mem: Mem) -> None:
    fs, root, _ = mem
    base = f"memory://{root}/data"
    found = _io.resolve_inputs([base])
    assert _names(found) == MEM_ALL
    assert [s.uri for s in found] == [f"{base}/{n}" for n in MEM_ALL]
    assert [s.size for s in found] == [fs.size(f"{root}/data/{n}") for n in MEM_ALL]
    assert [s.format for s in found] == ["parquet", "jsonl", "parquet", "jsonl", "jsonl"]
    assert isinstance(found[0]._fs, _io._Fsspec)
    assert _names(_io.resolve_inputs([f"{base}/**/*.jsonl*"])) == [
        "empty.jsonl",
        "sub/b.jsonl.gz",
        "sub/c.jsonl",
    ]
    assert _names(_io.resolve_inputs([f"{base}/sub/?.jsonl"])) == ["c.jsonl"]
    assert _names(_io.resolve_inputs([f"{base}/*"])) == MEM_ALL
    (one,) = _io.resolve_inputs([f"{base}/sub/b.jsonl.gz"])
    assert (one.uri, one.name, one.format) == (f"{base}/sub/b.jsonl.gz", "b.jsonl.gz", "jsonl")
    assert one.size == fs.size(f"{root}/data/sub/b.jsonl.gz")
    with pytest.raises(ValueError, match="listed twice"):
        _io.resolve_inputs([base, f"{base}/a.parquet"])
    with pytest.raises(FileNotFoundError, match="nope: no such file or directory"):
        _io.resolve_inputs([f"memory://{root}/nope"])
    for spec in [f"{base}/*.csv", f"memory://{root}/nope/*.parquet", f"{base}/README.md/*"]:
        with pytest.raises(FileNotFoundError, match=r"no \.parquet or \.jsonl files"):
            _io.resolve_inputs([spec])
    with pytest.raises(ValueError, match=r"README\.md: unknown format"):
        _io.resolve_inputs([f"{base}/README.md"])


@pytest.mark.thread_unsafe  # the memory filesystem (see Mem)
def test_memory_reads(mem: Mem) -> None:
    _, root, tables = mem
    a, ej, ep, b, c = _io.resolve_inputs([f"memory://{root}/data"])
    got = list(_io.iter_tables(a, 10))  # row groups of 7
    assert _rows(got) == [10, 10, 5]
    assert pa.concat_tables(got).equals(tables["a.parquet"])
    assert _rows(_io.iter_tables(b, 4)) == [4, 4, 2]
    assert pa.concat_tables(_io.iter_tables(b, 4)).equals(tables["sub/b.jsonl.gz"])
    assert pa.concat_tables(_io.iter_tables(c, 100)).equals(tables["sub/c.jsonl"])
    (proj,) = _io.iter_tables(a, 100, columns=["text"])
    assert proj.column("text").to_pylist() == tables["a.parquet"].column("text").to_pylist()
    (proj,) = _io.iter_tables(c, 100, columns=["id"])
    assert proj.column_names == ["id"]
    (empty,) = _io.iter_tables(ep, 10)
    assert empty.num_rows == 0
    assert empty.schema == _table(0).schema
    assert list(_io.iter_tables(ej, 10)) == []
    assert _io.read_schema(a) == tables["a.parquet"].schema
    assert _io.read_schema(b) == tables["sub/b.jsonl.gz"].schema
    assert _io.read_schema(ej) is None


@pytest.mark.thread_unsafe  # the memory filesystem (see Mem); patches its open()
def test_memory_files_are_closed(mem: Mem, monkeypatch: pytest.MonkeyPatch) -> None:
    _, root, _ = mem
    a, _, _, b, _ = _io.resolve_inputs([f"memory://{root}/data"])
    fs = a._fs.fs  # fsspec caches one instance per thread: patch the one the sources use
    # A MemoryFile's close() keeps its buffer and leaves .closed False: count the calls instead.
    opened: list[Any] = []
    closed: list[Any] = []
    real_open = functools.partial(type(fs).open, fs)  # not a patch left by an earlier iteration

    def tracking_open(*args: Any, **kwargs: Any) -> Any:
        f = real_open(*args, **kwargs)
        real_close = functools.partial(type(f).close, f)

        def close() -> None:
            closed.append(f)
            real_close()

        f.close = close
        opened.append(f)
        return f

    monkeypatch.setattr(fs, "open", tracking_open)
    for src in (a, b):
        g = _io.iter_tables(src, 2)
        next(g)
        assert opened[-1] not in closed
        g.close()
        assert opened[-1] in closed
        _io.read_schema(src)
        assert opened[-1] in closed
    assert len(opened) == 4
    assert all(f in closed for f in opened)


@pytest.mark.thread_unsafe  # the memory filesystem (see Mem)
def test_sources_pickle(mem: Mem, tree: Path) -> None:
    _, root, _ = mem
    for src in [*_io.resolve_inputs([f"memory://{root}/data"]), *_io.resolve_inputs([tree])]:
        back = pickle.loads(pickle.dumps(src))
        assert back == src
        assert back._path == src._path
        assert _rows(_io.iter_tables(back, 100)) == _rows(_io.iter_tables(src, 100))
    back = pickle.loads(pickle.dumps(_io.resolve_inputs([f"memory://{root}/data"])[0]))
    # fsspec's instance cache (one per thread): the filesystem this thread already has.
    assert back._fs.fs is pytest.importorskip("fsspec").filesystem("memory")


# ------------------------------------------------------------------------------ pyarrow fallback


@pytest.mark.thread_unsafe  # hides fsspec in sys.modules
def test_pyarrow_filesystem_without_fsspec(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "fsspec", None)  # import fsspec raises ImportError
    found = [s for _, s in _io._uri_sources(tree.as_uri())]
    assert _names(found) == ALL
    assert all(isinstance(s._fs, pafs.LocalFileSystem) for s in found)
    assert [s.uri for s in found] == [f"{tree.as_uri()}/{n}" for n in ALL]
    assert [s.size for s in found] == [os.path.getsize(tree.joinpath(*n.split("/"))) for n in ALL]
    a, b, c, e = found
    assert _rows(_io.iter_tables(c, 3)) == [3, 1]
    assert pa.concat_tables(_io.iter_tables(e, 3)).equals(_table(1, 9))
    assert pa.concat_tables(_io.iter_tables(b, 3)).equals(_table(2, 3))
    assert _io.read_schema(a) == _table(3).schema
    assert _rows(_io.iter_tables(pickle.loads(pickle.dumps(a)), 2)) == [2, 1]
    glob = [s for _, s in _io._uri_sources(f"{tree.as_uri()}/**/*.parquet")]
    assert _names(glob) == ["a.parquet", "sub/c.parquet"]
    (one,) = (s for _, s in _io._uri_sources(f"{tree.as_uri()}/b.jsonl"))
    assert (one.name, one.size) == ("b.jsonl", os.path.getsize(tree / "b.jsonl"))
    with pytest.raises(ValueError, match=r"^nosuch://b/x: reading nosuch:// needs fsspec"):
        _io.resolve_inputs(["nosuch://b/x"])


def test_fsspec_local_filesystem(tree: Path) -> None:
    pytest.importorskip("fsspec")
    found = [s for _, s in _io._uri_sources(tree.as_uri())]
    assert _names(found) == ALL
    assert all(isinstance(s._fs, _io._Fsspec) for s in found)
    assert pa.concat_tables(_io.iter_tables(found[3], 1)).equals(_table(1, 9))


@pytest.mark.thread_unsafe  # hides fsspec, patches the module
def test_missing_filesystem_says_what_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "fsspec", None)

    def no_s3(uri: str) -> Any:
        raise pa.ArrowNotImplementedError("Got S3 URI but Arrow compiled without S3 support")

    monkeypatch.setattr(_io, "_arrow_from_uri", no_s3)
    msg = (
        "s3://b/x: reading s3:// needs fsspec with s3fs (pip install s3fs), "
        "or a pyarrow built with S3"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        _io.resolve_inputs(["s3://b/x"])
    with pytest.raises(
        ValueError, match=r"^hf://datasets/a/b: .* \(pip install huggingface_hub\)$"
    ):
        _io.resolve_inputs(["hf://datasets/a/b"])


# ------------------------------------------------------------------------------- output names


@pytest.mark.parametrize(
    ("name", "safe"),
    [
        ("a/b.parquet", "a/b.parquet"),
        ('a<b>:c"d|e?f*g\\h.parquet', "a_b__c_d_e_f_g_h.parquet"),
        ("x\x01y\x1f\x7f.jsonl", "x_y__.jsonl"),
        ("dir. /file. .", "dir/file"),
        ("...", "_"),
        ("CON.parquet", "_CON.parquet"),
        ("nul", "_nul"),
        ("sub/Com1.json", "sub/_Com1.json"),
        ("lpt9.tar.gz", "_lpt9.tar.gz"),
        ("aux .txt", "_aux .txt"),
        ("COM\u00b9.x", "_COM\u00b9.x"),
        ("console.parquet", "console.parquet"),
        ("COM10.parquet", "COM10.parquet"),
        ("c:x.json", "c_x.json"),
        (".hidden/_x", ".hidden/_x"),
    ],
)
def test_safe_relpath(name: str, safe: str) -> None:
    assert _io.safe_relpath(name) == safe
    assert _io.safe_relpath(safe) == safe


@pytest.mark.parametrize(
    "name", ["", "/abs/x", "\\\\server\\x", "C:/x", "a//b", "a/", "a/./b", "../x"]
)
def test_safe_relpath_rejects(name: str) -> None:
    with pytest.raises(ValueError, match="file name"):
        _io.safe_relpath(name)


def test_output_path(tmp_path: Path) -> None:
    def src(name: str) -> Any:
        return _io.Source(uri=name, name=name, size=None, format=_io.file_format(name))

    assert (
        _io.output_path(tmp_path, src("a/x.jsonl.gz"), ".parquet") == tmp_path / "a" / "x.parquet"
    )
    assert _io.output_path(tmp_path, src("x.PARQUET"), ".jsonl.zst") == tmp_path / "x.jsonl.zst"
    assert _io.output_path(tmp_path, src("x.ndjson"), ".json") == tmp_path / "x.json"
    assert _io.output_path(tmp_path, src("a.b.pq"), ".jsonl") == tmp_path / "a.b.jsonl"
    assert _io.output_path(str(tmp_path), src("a/CON.parquet")) == tmp_path / "a" / "_CON.parquet"
    with pytest.raises(ValueError, match=r"starts with '\.'"):
        _io.output_path(tmp_path, src("x.parquet"), "parquet")


# ---------------------------------------------------------------------- replace, remove, atomic


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The backoff waits, recorded instead of slept."""
    waits: list[float] = []
    monkeypatch.setattr(_io, "_sleep", waits.append)
    return waits


@pytest.mark.thread_unsafe  # patches os.replace
def test_replace_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    sleeps.clear()  # pytest-run-parallel --iterations reruns with the same fixture values
    calls: list[int] = []

    def flaky(src: Any, dst: Any) -> None:
        calls.append(1)
        if len(calls) <= 2:
            raise PermissionError(13, "in use by another process")
        REAL_REPLACE(src, dst)

    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("new")
    b.write_text("old")
    monkeypatch.setattr(os, "replace", flaky)
    _io.replace(a, b)
    assert b.read_text() == "new"
    assert not a.exists()
    assert len(calls) == 3
    assert sleeps == [0.02, 0.04]

    def locked(src: Any, dst: Any) -> None:
        calls.append(1)
        raise PermissionError(13, "in use by another process")

    calls.clear()
    sleeps.clear()
    monkeypatch.setattr(os, "replace", locked)
    with pytest.raises(PermissionError):
        _io.replace(b, a)
    assert len(calls) == 10
    assert 2.0 <= sum(sleeps) <= 3.0
    monkeypatch.setattr(os, "replace", REAL_REPLACE)
    sleeps.clear()
    with pytest.raises(FileNotFoundError):  # other errors are not retried
        _io.replace(tmp_path / "missing", a)
    assert sleeps == []


@pytest.mark.thread_unsafe  # patches os.unlink
def test_remove(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
    sleeps.clear()  # pytest-run-parallel --iterations reruns with the same fixture values
    _io.remove(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        _io.remove(tmp_path / "missing", missing_ok=False)
    p = tmp_path / "x"
    p.write_text("x")
    calls: list[int] = []

    def flaky(path: Any) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError(13, "in use by another process")
        REAL_UNLINK(path)

    monkeypatch.setattr(os, "unlink", flaky)
    _io.remove(p)
    assert not p.exists()
    assert sleeps == [0.02]


def test_atomic_write(tmp_path: Path) -> None:
    out = tmp_path / "new" / "dir" / "stats.json"
    _io.atomic_write_text(out, '{"a": "\u00e9"}\n')
    assert out.read_bytes() == '{"a": "\u00e9"}\n'.encode()
    _io.atomic_write_bytes(out, b"\x00\r\n")  # replaces; bytes as given (no newline translation)
    assert out.read_bytes() == b"\x00\r\n"
    assert os.listdir(out.parent) == ["stats.json"]
    if sys.platform != "win32":  # the umask applies, as for a file made with open()
        ref = tmp_path / "ref"
        ref.write_text("")
        assert out.stat().st_mode == ref.stat().st_mode


@pytest.mark.thread_unsafe  # patches os.replace
def test_atomic_write_uses_a_hidden_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "stats.json"
    out.write_text("old")
    seen: list[str] = []

    def failing(src: Any, dst: Any) -> None:
        seen.append(os.fspath(src))
        raise OSError("disk on fire")

    monkeypatch.setattr(os, "replace", failing)
    with pytest.raises(OSError, match="disk on fire"):
        _io.atomic_write_text(out, "new")
    (tmp,) = seen
    assert Path(tmp).parent == tmp_path
    assert re.fullmatch(r"\.stats\.json\.[0-9a-f]{8}\.tmp", Path(tmp).name)
    assert out.read_text() == "old"
    assert os.listdir(tmp_path) == ["stats.json"]


# ------------------------------------------------------------------------------------- Writer


def _partials(d: Path, name: str) -> list[str]:
    pat = re.compile(rf"\.{re.escape(name)}\.[0-9a-f]{{8}}\.partial")
    return [p for p in os.listdir(d) if pat.fullmatch(p)]


@pytest.mark.parametrize("name", ["out.parquet", "out.jsonl", "out.jsonl.gz", "out.json.zst"])
def test_writer_is_atomic(tmp_path: Path, name: str) -> None:
    out = tmp_path / "sub" / name
    w = _io.Writer(out)
    w.write(_table(3))
    w.write(_table(0))
    w.write(_table(4, 3))
    assert not out.exists()
    assert len(_partials(out.parent, name)) == 1
    assert w.tmp is not None
    assert w.tmp.name == _partials(out.parent, name)[0]
    w.close()
    assert os.listdir(out.parent) == [name]
    assert w.rows == 7
    (src,) = _io.resolve_inputs([out])
    assert pa.concat_tables(_io.iter_tables(src, 100)).equals(_table(7))
    w.close()  # again: nothing to do
    w.abort()  # after a successful close: nothing to do
    assert out.exists()
    # Replaces an existing file.
    w = _io.Writer(out)
    w.write(_table(1))
    w.close()
    assert sum(_rows(_io.iter_tables(src, 100))) == 1


def test_writer_abort(tmp_path: Path) -> None:
    out = tmp_path / "out.parquet"
    w = _io.Writer(out)
    w.write(_table(3))
    w.abort()
    assert os.listdir(tmp_path) == []
    w.abort()  # idempotent
    with pytest.raises(ValueError, match="aborted"):
        w.close()
    with pytest.raises(ValueError, match="aborted"):
        w.write(_table(1))
    _io.Writer(tmp_path / "never.jsonl").abort()  # nothing written: nothing to drop
    assert os.listdir(tmp_path) == []


@pytest.mark.thread_unsafe  # patches the module's replace()
def test_writer_failed_close_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(src: Any, dst: Any) -> None:
        raise OSError("disk on fire")

    monkeypatch.setattr(_io, "replace", failing)
    for name in ("out.parquet", "out.jsonl.gz"):
        w = _io.Writer(tmp_path / name)
        w.write(_table(3))
        with pytest.raises(OSError, match="disk on fire"):
            w.close()
        assert os.listdir(tmp_path) == []
        w.abort()  # the caller's cleanup after a failed close is safe
        assert os.listdir(tmp_path) == []


def test_writer_empty_outputs(tmp_path: Path) -> None:
    for name in ("a.parquet", "b.jsonl", "c.jsonl.gz", "d.parquet"):
        w = _io.Writer(tmp_path / name)
        w.close(_table(0).schema if name == "d.parquet" else None)
    _, _, c, d = _io.resolve_inputs([tmp_path])
    assert pq.read_table(tmp_path / "a.parquet").num_columns == 0  # a valid file, no columns
    assert (tmp_path / "b.jsonl").read_bytes() == b""
    assert list(_io.iter_tables(c, 10)) == []  # a valid, empty gzip stream
    assert _io.read_schema(d) == _table(0).schema
    assert sorted(os.listdir(tmp_path)) == ["a.parquet", "b.jsonl", "c.jsonl.gz", "d.parquet"]


@pytest.mark.skipif(not LINUX, reason="needs /proc/self/fd to see open files")
@pytest.mark.thread_unsafe  # patches os.replace
@pytest.mark.parametrize("name", ["out.parquet", "out.jsonl.gz"])
def test_writer_closes_before_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    checked: list[str] = []

    def checking(src: Any, dst: Any) -> None:
        held = _held(src)
        assert not held, f"{src} is still {', '.join(held)} at the rename"
        checked.append(os.fspath(src))
        REAL_REPLACE(src, dst)

    monkeypatch.setattr(os, "replace", checking)
    w = _io.Writer(tmp_path / name)
    w.write(_table(5))
    assert w.tmp is not None
    assert _held(w.tmp)  # the partial file is open while writing
    w.close()
    assert len(checked) == 1
    assert _held(tmp_path / name) == []
