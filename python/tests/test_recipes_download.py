"""The fastText LID download says what it does before it starts (it is 125 MiB)."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyarrow")

from pyroid.integrations import recipes


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def _fake_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    src = tmp_path / "src.bin"
    src.write_bytes(b"x" * (3 << 20))
    monkeypatch.setattr(recipes, "LID_176_URL", src.as_uri())
    monkeypatch.setattr(recipes, "LID_176_MIRRORS", ())
    monkeypatch.setattr(recipes, "LID_176_SHA256", hashlib.sha256(src.read_bytes()).hexdigest())
    monkeypatch.setenv("PYROID_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("PYROID_LID_MODEL", raising=False)
    return tmp_path / "cache" / "lid.176.bin"


@pytest.mark.thread_unsafe(reason="patches module globals and the environment")
def test_download_prints_one_line_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    target = _fake_model(tmp_path, monkeypatch)
    assert recipes.lid_model_path() == str(target)
    err = capsys.readouterr().err
    lines = err.splitlines()
    assert len(lines) == 1, err
    assert "downloading fastText's lid.176.bin (125 MiB" in lines[0]
    assert str(target) in lines[0]
    assert "PYROID_LID_MODEL" in lines[0]
    # The cached copy is used from then on, silently.
    assert recipes.lid_model_path() == str(target)
    assert capsys.readouterr().err == ""


@pytest.mark.thread_unsafe(reason="patches module globals, the environment and sys.stderr")
def test_download_shows_progress_on_a_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_model(tmp_path, monkeypatch)
    tty = _Tty()
    monkeypatch.setattr("sys.stderr", tty)
    recipes.lid_model_path()
    out = tty.getvalue()
    assert "\rpyroid: lid.176.bin   3.0 of 3.0 MiB" in out
    assert out.endswith("\n")
