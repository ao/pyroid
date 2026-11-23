"""The ``OSError`` for a file pyroid's Rust code fails to open: subclass, errno, message, filename.

On Windows the code of an ``io::Error`` is a Windows error, not an errno: ERROR_PATH_NOT_FOUND
is 3, which is ESRCH elsewhere (``ProcessLookupError``). The mapping is checked here for both
platforms; the real errors only for the running one.
"""

from __future__ import annotations

import errno
import os
import sys
from pathlib import Path

import pytest

from pyroid._pyroid import _testing

WINDOWS = sys.platform == "win32"


@pytest.mark.parametrize(
    ("code", "text", "windows", "want"),
    [
        # Windows: the code is the winerror (which gives the errno), the message has no period.
        (
            3,
            "The system cannot find the path specified. (os error 3)",
            True,
            (3, "The system cannot find the path specified", 3),
        ),
        (5, "Access is denied. (os error 5)", True, (5, "Access is denied", 5)),
        (
            2,
            "No such file or directory (os error 2)",
            False,
            (2, "No such file or directory", None),
        ),
        (13, "Permission denied (os error 13)", False, (13, "Permission denied", None)),
        # A message without io::Error's suffix stays as it is.
        (2, "gone", False, (2, "gone", None)),
    ],
    ids=["win-path-not-found", "win-access-denied", "posix-enoent", "posix-eacces", "no-suffix"],
)
def test_os_error_args(code: int, text: str, windows: bool, want: tuple[object, ...]) -> None:
    assert _testing.os_error_args(code, text, windows=windows) == want


def test_os_error_args_give_the_subclass_on_this_platform() -> None:
    # ERROR_PATH_NOT_FOUND on Windows, ENOENT elsewhere: FileNotFoundError either way.
    code = 3 if WINDOWS else errno.ENOENT
    errno_, strerror, winerror = _testing.os_error_args(code, "gone", windows=WINDOWS)
    e = OSError(errno_, strerror, "x", *(() if winerror is None else (winerror,)))
    assert type(e) is FileNotFoundError
    assert e.errno == errno.ENOENT


def summary(e: BaseException) -> tuple[object, ...]:
    assert isinstance(e, OSError)
    filename = None if e.filename is None else os.fspath(e.filename)
    return type(e), e.errno, e.strerror, filename, getattr(e, "winerror", None)


@pytest.fixture(params=["file", "dir"], ids=["missing-file", "missing-dir"])
def missing(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    """A path that does not exist; for "dir", its folder does not exist either."""
    return tmp_path / "nope.bin" if request.param == "file" else tmp_path / "no-dir" / "nope.bin"


def stat_error(path: Path) -> OSError:
    """What ``os.stat(path)`` raises for a missing `path`: with the winerror on Windows."""
    with pytest.raises(FileNotFoundError) as info:
        os.stat(path)
    return info.value


def test_warc_raises_what_open_raises(missing: Path) -> None:
    import pyroid.warc as pw

    with pytest.raises(FileNotFoundError) as ref:
        open(missing, "rb")  # noqa: SIM115
    want = summary(ref.value)
    assert want[0] is FileNotFoundError
    for source in (missing, str(missing)):
        with pytest.raises(FileNotFoundError) as info:
            pw.ArchiveIterator(source)
        assert summary(info.value) == want
        with pytest.raises(FileNotFoundError) as info:
            next(pw.iter_documents(source))
        assert summary(info.value) == want


def test_fasttext_load_model_raises_like_stat(missing: Path) -> None:
    pytest.importorskip("numpy")
    import pyroid.fasttext as ft

    want = summary(stat_error(missing))
    assert want[0] is FileNotFoundError
    with pytest.raises(FileNotFoundError) as info:
        ft.load_model(str(missing))
    assert summary(info.value) == want


def test_minhash_stage_raises_like_stat(missing: Path, tmp_path: Path) -> None:
    from pyroid._pyroid import dedup as native

    want = summary(stat_error(missing))
    assert want[0] is FileNotFoundError
    with pytest.raises(FileNotFoundError) as info:
        native._minhash_cluster([str(missing)], str(tmp_path / "out"))
    assert summary(info.value) == want
