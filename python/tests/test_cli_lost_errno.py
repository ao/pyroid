"""The CLI where a failed os call reports EAGAIN instead of its own errno.

On free-threaded CPython linked with musl (Alpine), a thread whose os call fails, or whose
directory listing ends, while another thread stops the world (a garbage collection) can see
errno EAGAIN instead of the call's own (CPython gh-161769). ``pyroid translit`` in eight threads
then failed with "[Errno 11] Resource temporarily unavailable" on paths every run creates or
removes: ``Path.mkdir(parents=True)`` creates the parents only on FileNotFoundError, and a
missing file was known by FileNotFoundError. Here the os functions are made to do that on
every platform: each error they raise becomes ``BlockingIOError(EAGAIN)``, and the first listing
of each directory fails at its end.
"""

from __future__ import annotations

import errno
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Self

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("numpy")

from pyroid.cli import main  # noqa: E402

pytestmark = pytest.mark.thread_unsafe(reason="replaces functions of the os module")
_real_scandir = os.scandir


def _ls(path: Path) -> list[str]:
    """The names in ``path``, listed with the real ``os.scandir``."""
    with _real_scandir(path) as it:
        return sorted(e.name for e in it)


def _eagain(filename: Any = None, filename2: Any = None) -> BlockingIOError:
    return BlockingIOError(errno.EAGAIN, os.strerror(errno.EAGAIN), filename, None, filename2)


def _losing(fn: Callable[..., Any]) -> Callable[..., Any]:
    def call(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except OSError as e:
            raise _eagain(e.filename, e.filename2) from None

    return call


class _Listing:
    """An ``os.scandir`` iterator whose end raises EAGAIN (once, if ``fail``)."""

    def __init__(self, it: Any, fail: bool) -> None:
        self._it, self._fail = it, fail

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[os.DirEntry[str]]:
        return self

    def __next__(self) -> os.DirEntry[str]:
        try:
            return next(self._it)
        except StopIteration:
            if self._fail:
                self._fail = False
                raise _eagain() from None
            raise

    def close(self) -> None:
        self._it.close()


@pytest.fixture
def lost_errno(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the os functions; the list collects the calls whose errno was lost."""
    lost: list[str] = []
    listed: set[tuple[int, int]] = set()
    real_scandir, real_listdir = os.scandir, os.listdir

    def first(target: Any) -> bool:
        st = os.stat(target)
        key = (st.st_dev, st.st_ino)
        new = key not in listed
        listed.add(key)
        return new

    def scandir(path: Any = ".") -> _Listing:
        fail = first(path)
        if fail:
            lost.append(f"scandir {path}")
        return _Listing(real_scandir(path), fail)

    def listdir(path: Any = ".") -> list[str]:
        names = real_listdir(path)
        if first(path):
            lost.append(f"listdir {path}")
            raise _eagain(path)
        return names

    def losing(name: str) -> Callable[..., Any]:
        fn = _losing(getattr(os, name))

        def call(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except BlockingIOError:
                lost.append(f"{name} {args[0] if args else ''}")
                raise

        return call

    for name in ("mkdir", "rmdir", "unlink", "remove", "open", "replace", "rename"):
        monkeypatch.setattr(os, name, losing(name))
    monkeypatch.setattr(os, "scandir", scandir)
    monkeypatch.setattr(os, "listdir", listdir)
    return lost


def _write(texts: list[str], path: Path) -> Path:
    pq.write_table(pa.table({"id": [f"d{i}" for i in range(len(texts))], "text": texts}), path)
    return path


def _texts(path: Path) -> list[str]:
    texts: list[str] = pq.read_table(path).column("text").to_pylist()
    return texts


def test_runs_decide_by_the_disk_not_the_errno(tmp_path: Path, lost_errno: list[str]) -> None:
    src = _write(["Ünïcödé", "b"], tmp_path / "in.parquet")
    docs = tmp_path / "docs"
    os.makedirs(docs / "sub")
    _write(["Ünïcödé"], docs / "a.parquet")
    _write(["b"], docs / "sub" / "b.parquet")
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "d0", "text": "x"}\n{"id": "d1", "text": "unterminated\n')
    lost_errno.clear()

    out = tmp_path / "out"
    assert main(["translit", str(src), str(out)]) == 0  # OUT/.pyroid/done: parents made
    assert main(["translit", str(src), str(out), "--overwrite"]) == 0  # its old records removed
    assert _texts(out / "in.parquet") == ["Unicode", "b"]
    # A merged file OUT: its parts wait in .o.parquet.pyroid/work-*/parts.
    assert main(["translit", str(src), str(tmp_path / "o.parquet"), "--mode", "unidecode"]) == 0
    assert _texts(tmp_path / "o.parquet") == ["Unicode", "b"]
    assert _ls(tmp_path / ".o.parquet.pyroid") == ["done", "run.json"]
    # A directory input is walked; a failed run leaves no state.
    assert main(["translit", str(docs), str(tmp_path / "w")]) == 0
    assert _texts(tmp_path / "w" / "sub" / "b.parquet") == ["b"]
    assert main(["translit", str(bad), str(tmp_path / "bad.parquet")]) == 1
    assert not (tmp_path / ".bad.parquet.pyroid").exists()
    assert main(["dedup", str(src), str(tmp_path / "d.parquet"), "--method", "exact"]) == 0
    # The calls the CLI made did lose their errno (else this tests nothing).
    assert any(x.startswith("mkdir ") for x in lost_errno), lost_errno
    assert any(x.startswith("unlink ") for x in lost_errno), lost_errno
    assert any(x.startswith(("scandir ", "listdir ")) for x in lost_errno), lost_errno


def test_a_foreign_out_is_still_refused(tmp_path: Path, lost_errno: list[str]) -> None:
    src = _write(["b"], tmp_path / "in.parquet")
    out = tmp_path / "out"
    out.mkdir()
    (out / "notes.txt").write_text("mine")
    lost_errno.clear()
    with pytest.raises(SystemExit, match="not empty and was not written by pyroid"):
        main(["translit", str(src), str(out)])
    assert any(x.startswith(("scandir ", "listdir ")) for x in lost_errno), lost_errno


def test_minhash_stages(tmp_path: Path, lost_errno: list[str]) -> None:
    lines = [  # three texts with no word in common, four copies of each
        " ".join(["ant bee cat dog eel fox gnu hen"] * 4),
        " ".join(["ibis jay kiwi lynx mole newt owl puma"] * 4),
        " ".join(["quail rat seal tern urial vole wolf yak"] * 4),
    ]
    texts = [lines[i % 3] for i in range(12)]
    src = _write(texts, tmp_path / "in.parquet")
    sigs, dups, rem = tmp_path / "sigs", tmp_path / "dups", tmp_path / "rem"
    out = tmp_path / "out.parquet"
    lost_errno.clear()
    assert main(["dedup", "signatures", str(src), str(sigs)]) == 0
    assert main(["dedup", "buckets", str(sigs), str(dups)]) == 0
    assert main(["dedup", "cluster", str(dups), str(rem)]) == 0
    assert main(["dedup", "filter", str(src), str(out), "--removed", str(rem)]) == 0
    assert len(_texts(out)) == 3
    # In one command, with the signatures under --tmp-dir (a directory made with mode 0700).
    scratch, one = tmp_path / "scratch", tmp_path / "one.parquet"
    assert (
        main(["dedup", str(src), str(one), "--method", "minhash", "--tmp-dir", str(scratch)]) == 0
    )
    assert _texts(one) == _texts(out)
    assert _ls(scratch) == []
    assert any(x.startswith(("scandir ", "listdir ")) for x in lost_errno), lost_errno
