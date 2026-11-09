"""pyroid.patch("trafilatura"): the shims for trafilatura, trafilatura.core, .deduplication, .meta.

Patching is process-wide, so each case runs in a fresh interpreter. The cases with the real
trafilatura need 2.3.1 installed (GIL builds: importing it loads lxml); the others hide it, or put
a fake trafilatura 1.11.0 first on ``sys.path``.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from pyroid.integrations import _patch

pytestmark = pytest.mark.thread_unsafe(reason="subprocesses")

FREE_THREADED = not getattr(sys, "_is_gil_enabled", lambda: True)()
HAVE_REAL = (
    not FREE_THREADED
    and importlib.util.find_spec("trafilatura") is not None
    and importlib.metadata.version("trafilatura") == "2.3.1"
)
needs_real = pytest.mark.skipif(not HAVE_REAL, reason="needs trafilatura 2.3.1 (GIL builds)")

PRELUDE = """\
import importlib.util, inspect, json, sys, warnings
warnings.simplefilter("ignore", DeprecationWarning)
PAGE = "<html><body><article><p>" + "Words of the article. " * 20 + "</p></article></body></html>"
def patch(*names):
    import pyroid
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pyroid.patch(*names)
    return [(w.category.__name__, str(w.message)) for w in caught]
def shape(fn):
    return [(p.name, p.kind.name, repr(p.default) if p.name != "config" else id(p.default))
            for p in inspect.signature(fn).parameters.values()]
"""


def child(run_python: Callable[..., Any], code: str) -> Any:
    proc = run_python(PRELUDE + textwrap.dedent(code))
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_supported() -> None:
    assert "trafilatura" in _patch.SUPPORTED
    assert "trafilatura" not in _patch.DEFAULT_RUN_NAMES


@needs_real
def test_shim_with_real_trafilatura(run_python: Callable[..., Any]) -> None:
    out = child(
        run_python,
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import trafilatura as early  # imported before patch(): the real one
        caught = patch("trafilatura")
        import trafilatura, trafilatura.core, trafilatura.deduplication, trafilatura.meta
        from trafilatura import extract, bare_extraction, extract_with_metadata
        from trafilatura.core import extract as core_extract
        from trafilatura.deduplication import LRUCache, LRU_TEST
        from trafilatura.meta import reset_caches
        import pyroid.trafilatura as pt
        real = trafilatura.__wrapped__
        assert real is early and trafilatura.__pyroid__ and trafilatura.__version__ == "2.3.1"
        assert core_extract is extract and trafilatura.core.extract is extract
        assert extract.__module__ == "pyroid.trafilatura"
        assert LRUCache is pt.LRUCache and LRU_TEST is pt.LRU_TEST
        # exactly trafilatura's signatures (no compat=), config defaulting to its DEFAULT_CONFIG
        for name in ("extract", "bare_extraction", "extract_with_metadata"):
            assert shape(getattr(trafilatura, name)) == shape(getattr(real, name)), name
        default = inspect.signature(extract).parameters["config"].default
        assert default is real.settings.DEFAULT_CONFIG
        # what pyroid does not provide is trafilatura's own
        assert trafilatura.html2txt is real.html2txt and trafilatura.fetch_url is real.fetch_url
        assert trafilatura.settings is real.settings
        assert extract(PAGE) == real.extract(PAGE)
        assert bare_extraction(PAGE).text == real.bare_extraction(PAGE).text
        assert extract(PAGE, deduplicate=True) and pt.LRU_TEST.cache
        reset_caches()
        assert not pt.LRU_TEST.cache
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            xml = [extract(PAGE, output_format="xml") for _ in range(2)]
        assert xml[0] == real.extract(PAGE, output_format="xml")
        fallbacks = [x for x in w if x.category.__name__ == "FallbackWarning"]
        assert len(fallbacks) == 1, w
        import pyroid
        pyroid.unpatch("trafilatura")
        assert sys.modules["trafilatura"] is real
        print(json.dumps(caught))
        """,
    )
    assert len(out) == 1
    category, message = out[0]
    assert category == "PatchWarning"
    assert "`import trafilatura` now gives pyroid's drop-in (pyroid.trafilatura" in message
    assert "still comes from the installed trafilatura" in message


def test_shim_without_trafilatura(run_python: Callable[..., Any]) -> None:
    out = child(
        run_python,
        """
        class Hide:
            def find_spec(self, name, path=None, target=None):
                if name.partition(".")[0] == "trafilatura":
                    raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        sys.meta_path.insert(0, Hide())
        caught = patch("trafilatura")
        import trafilatura
        from trafilatura.core import extract
        from trafilatura.deduplication import LRUCache
        import pyroid, pyroid.trafilatura as pt
        assert trafilatura.__wrapped__ is None and trafilatura.__version__ == "2.3.1"
        assert extract(PAGE) == pt.extract(PAGE) and extract(PAGE).startswith("Words of")
        assert LRUCache is pt.LRUCache
        try:
            trafilatura.html2txt
        except AttributeError as e:
            assert "trafilatura is not installed" in str(e), e
        else:
            raise AssertionError("html2txt")
        for call in (lambda: extract(PAGE, output_format="xml"),
                     lambda: trafilatura.extract_with_metadata(PAGE)):
            try:
                call()
            except pyroid.PyroidError as e:
                assert "pip install trafilatura==2.3.1" in str(e), e
            else:
                raise AssertionError("no PyroidError")
        trafilatura.meta.reset_caches()
        assert importlib.util.find_spec("trafilatura") is not None
        print(json.dumps(caught))
        """,
    )
    assert [c for c, _ in out] == ["PatchWarning"]
    assert (
        "is unavailable: trafilatura is not installed (pip install trafilatura==2.3.1)" in out[0][1]
    )


def test_shim_with_another_trafilatura_version(
    run_python: Callable[..., Any], tmp_path: Path
) -> None:
    fake = tmp_path / "site" / "trafilatura"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text(
        '__version__ = "1.11.0"\nfrom . import core, meta, deduplication, settings\n'
        "def html2txt(s):\n    return 'fake'\n"
    )
    (fake / "core.py").write_text("def extract(*a, **k):\n    return 'fake 1.11'\n")
    (fake / "meta.py").write_text("CALLS = []\ndef reset_caches():\n    CALLS.append(1)\n")
    (fake / "deduplication.py").write_text("class LRUCache:\n    pass\n")
    (fake / "settings.py").write_text("DEFAULT_CONFIG = None\n")
    out = child(
        run_python,
        f"""
        sys.path.insert(0, {str(fake.parent)!r})
        caught = patch("trafilatura")
        import trafilatura, pyroid, pyroid.trafilatura as pt
        real = trafilatura.__wrapped__
        assert real.__version__ == "1.11.0" and trafilatura.__version__ == "2.3.1"
        assert trafilatura.extract(PAGE) == pt.extract(PAGE)  # trafilatura 2.3.1's result
        assert trafilatura.html2txt("x") == "fake"
        trafilatura.meta.reset_caches()
        assert real.meta.CALLS == [1]
        try:
            trafilatura.extract(PAGE, output_format="xml")
        except pyroid.PyroidError as e:
            assert "installed trafilatura is 1.11.0, not 2.3.1" in str(e), e
        else:
            raise AssertionError("no PyroidError")
        print(json.dumps(caught))
        """,
    )
    assert [c for c, _ in out] == ["PatchWarning", "PatchWarning"]
    assert (
        "the installed trafilatura is 1.11.0, but pyroid.trafilatura reproduces trafilatura "
        "2.3.1: the patched functions give trafilatura 2.3.1's results"
    ) in out[1][1]
