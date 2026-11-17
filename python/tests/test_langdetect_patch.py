"""``pyroid.patch("langdetect")``: langdetect's own tests, objects made before the patch,
results equal to unpatched langdetect, and threads racing patch()/unpatch()."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parent.parent / "parity" / "data" / "langdetect-1.0.9" / "tests"

pytestmark = pytest.mark.thread_unsafe


@pytest.mark.parametrize("block", ["0", "1"])
def test_langdetects_own_tests(
    run_python: Callable[..., Any], block: str, tmp_path: Path
) -> None:
    if not UPSTREAM.is_dir():
        pytest.skip("needs the repository's parity/data")
    pytest.importorskip("six")  # langdetect's tests import it
    if block == "0":
        pytest.importorskip("langdetect")
    ini = tmp_path / "pytest.ini"
    ini.write_text("[pytest]\n", encoding="utf-8")  # not parity/'s configuration
    code = f"""
        import sys
        sys.path.insert(0, {str(HERE)!r})
        import pytest
        sys.exit(pytest.main(["-q", "-c", {str(ini)!r}, "-p", "_langdetect_upstream",
                              "-p", "no:cacheprovider", "--rootdir", {str(UPSTREAM.parent)!r},
                              {str(UPSTREAM)!r}]))
    """
    out = run_python(code, env={"PYROID_BLOCK_LANGDETECT": block}, timeout=240)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "8 passed" in out.stdout, out.stdout


def test_patched_results_equal_langdetect(run_python: Callable[..., Any]) -> None:
    pytest.importorskip("langdetect")
    code = f"""
        import sys, warnings
        sys.path.insert(0, {str(HERE)!r})
        from _langdetect_samples import SAMPLES
        import langdetect
        from langdetect import DetectorFactory, detect, detect_langs
        DetectorFactory.seed = 0
        def run():
            out = []
            for t in SAMPLES.values():
                out.append((detect(t), [(x.lang, x.prob.hex()) for x in detect_langs(t)]))
            for t in ("", "123"):
                try:
                    detect(t)
                except langdetect.LangDetectException as e:
                    out.append(("err", e.code, str(e)))
            return out
        expected = run()
        early = DetectorFactory().create  # a bound method of an object made before the patch
        d_before = langdetect.detector_factory._factory.create()
        d_before.append(SAMPLES["fr"])
        import pyroid
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pyroid.patch("langdetect")
        assert any("langdetect" in str(x.message) for x in w), [str(x.message) for x in w]
        from langdetect.detector import Detector
        import pyroid.langdetect._patch as lp
        assert Detector.__dict__["_detect_block"] is lp._detect_block
        assert langdetect.detect is lp.detect
        assert run() == expected
        assert detect is not langdetect.detect and run() == expected  # names imported before
        assert d_before.detect() == "fr"
        pyroid.unpatch("langdetect")
        assert Detector.__dict__["_detect_block"] is not lp._detect_block
        assert langdetect.detect is detect
        assert run() == expected
        print("ok")
    """
    out = run_python(code, timeout=240)
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.strip() == "ok"


def test_patch_without_langdetect(run_python: Callable[..., Any]) -> None:
    code = """
        import sys, warnings
        sys.modules["langdetect"] = None
        import pyroid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pyroid.patch("langdetect")
        import langdetect
        from langdetect import DetectorFactory, detect, LangDetectException
        from langdetect.detector_factory import init_factory
        from langdetect.utils.lang_profile import LangProfile
        from langdetect.lang_detect_exception import ErrorCode
        import pyroid.langdetect as pl
        assert DetectorFactory is pl.DetectorFactory and LangProfile is pl.LangProfile
        DetectorFactory.seed = 0
        print(detect("Ceci est une phrase en français."), ErrorCode.CantDetectError)
    """
    out = run_python(code)
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.split() == ["fr", "5"]


def test_threads_while_patching(run_python: Callable[..., Any]) -> None:
    pytest.importorskip("langdetect")
    code = f"""
        import sys, threading, time, warnings
        sys.path.insert(0, {str(HERE)!r})
        from _langdetect_samples import SAMPLES
        import langdetect
        from langdetect import DetectorFactory
        DetectorFactory.seed = 0
        texts = list(SAMPLES.values())[:12]
        expected = [[(x.lang, x.prob.hex()) for x in langdetect.detect_langs(t)] for t in texts]
        import pyroid
        warnings.simplefilter("ignore")
        stop = threading.Event()
        bad = []
        def work():
            while not stop.is_set():
                for t, e in zip(texts, expected):
                    got = [(x.lang, x.prob.hex()) for x in langdetect.detect_langs(t)]
                    if got != e:
                        bad.append(t)
                    d = langdetect.detector_factory._factory.create()
                    d.append(t)
                    if [(x.lang, x.prob.hex()) for x in d.get_probabilities()] != e:
                        bad.append(t)
        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        deadline = time.monotonic() + 3
        n = 0
        while time.monotonic() < deadline:
            pyroid.patch("langdetect")
            pyroid.unpatch("langdetect")
            n += 1
        stop.set()
        for t in threads:
            t.join()
        print(n, len(bad))
    """
    out = run_python(code, timeout=240)
    assert out.returncode == 0, out.stdout + out.stderr
    n, bad = map(int, out.stdout.split())
    assert n > 0 and bad == 0
