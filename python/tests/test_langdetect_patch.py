"""``pyroid.patch("langdetect")``: langdetect's own tests, objects made before the patch,
results equal to unpatched langdetect, and threads racing patch()/unpatch()."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parent.parent / "parity" / "data" / "langdetect-1.0.9" / "tests"

pytestmark = pytest.mark.thread_unsafe


@pytest.mark.parametrize("block", ["0", "1"])
def test_langdetects_own_tests(run_python: Callable[..., Any], block: str, tmp_path: Path) -> None:
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
        block = Detector.__dict__["_detect_block"]
        assert block.__code__ is lp._CODES["_detect_block"]  # langdetect's function, pyroid's code
        assert langdetect.detect is detect and detect.__code__ is lp._CODES["detect"]
        assert run() == expected
        assert d_before.detect() == "fr"
        pyroid.unpatch("langdetect")
        assert Detector.__dict__["_detect_block"] is block
        assert block.__code__ is not lp._CODES["_detect_block"]
        assert langdetect.detect is detect and detect.__code__ is not lp._CODES["detect"]
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
    assert n > 0
    assert bad == 0


# The outside evaluation's (P61) reproducers for the patch.


def test_patched_functions_pickle_for_an_unpatched_process(run_python: Callable[..., Any]) -> None:
    pytest.importorskip("langdetect")
    # Pickles made and loaded by this test's own child processes only.
    code = """
        import pickle, subprocess, sys, warnings
        from langdetect import detect as early, detect_langs as early_langs
        from langdetect.detector import Detector
        warnings.simplefilter("ignore")
        import pyroid
        pyroid.patch("langdetect")
        import langdetect
        objs = [langdetect.detect, langdetect.detect_langs, early, early_langs, Detector.append,
                Detector._detect_block, langdetect.detector_factory.detect]
        blob = pickle.dumps(objs)
        assert b"pyroid" not in blob, blob
        child = (
            "import pickle, sys; from langdetect import DetectorFactory; DetectorFactory.seed = 0;"
            "f = pickle.loads(sys.stdin.buffer.read());"
            "import langdetect; assert f[0] is langdetect.detect and f[2] is f[0];"
            "print(f[0]('Hello world, this is English text.'),"
            " f[1]('Bonjour tout le monde')[0].lang)"
        )
        out = subprocess.run([sys.executable, "-I", "-c", child], input=blob, capture_output=True)
        print(out.stdout.decode().strip(), out.stderr.decode().strip()[-300:])
        assert pickle.loads(blob)[0] is langdetect.detect
    """
    out = run_python(code, timeout=240)
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.split()[:2] == ["en", "fr"], out.stdout


def test_spawn_workers_without_the_patch(run_python: Callable[..., Any], tmp_path: Path) -> None:
    pytest.importorskip("langdetect")
    script = tmp_path / "job.py"
    script.write_text(
        "import multiprocessing as mp, warnings\n"
        "import langdetect\n"
        "from langdetect import DetectorFactory\n"
        "DetectorFactory.seed = 0\n"
        "TEXTS = ['Hello world, this is English.', 'Das ist ein Haus.', 'Ceci est une phrase.']\n"
        "if __name__ == '__main__':\n"
        "    warnings.simplefilter('ignore')\n"
        "    expected = [langdetect.detect(t) for t in TEXTS]\n"
        "    import pyroid\n"
        "    pyroid.patch('langdetect')\n"
        "    with mp.get_context('spawn').Pool(2) as pool:\n"
        "        got = pool.map(langdetect.detect, TEXTS)\n"
        "    print(got == expected, got)\n",
        encoding="utf-8",
    )
    path = str(script)
    code = (
        f"import runpy, sys\nsys.argv = [{path!r}]\nrunpy.run_path({path!r}, run_name='__main__')"
    )
    out = run_python(code, timeout=240)
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.startswith("True"), out.stdout


def test_customised_langdetect_under_the_patch(run_python: Callable[..., Any]) -> None:
    pytest.importorskip("langdetect")
    code = r"""
        import copy, pickle, re, warnings
        import langdetect
        from langdetect import DetectorFactory, detector_factory
        from langdetect.detector import Detector
        from langdetect.utils.ngram import NGram
        DetectorFactory.seed = 0
        detector_factory.init_factory()
        f = detector_factory._factory
        WWW = "Lies das: www.spiegel.de/politik/deutschland www.zeit.de/wirtschaft ok?"
        RU = "Привет мир hello world ok fine"
        def langs(r):
            return [(x.lang, x.prob.hex()) for x in r]
        def call(fn):
            try:
                return fn()
            except Exception as e:
                return (type(e).__name__, str(e))
        class Lower(Detector):
            def cleaning_text(self):
                self.text = self.text.lower()
        class Ngrams(Detector):
            def _extract_ngrams(self):
                return ["a", "b"]
        def det(cls, text, **attrs):
            d = cls(f)
            for k, v in attrs.items():
                setattr(d, k, v)
            d.append(text)
            return call(lambda: langs(d.get_probabilities())), d.text
        def run():
            out = [det(Lower, "Otec matka syn."), det(Ngrams, "Bonjour")]
            out.append(det(Detector, "Hello 123", URL_RE=re.compile("[a-z]+")))
            old = Detector.URL_RE
            Detector.URL_RE = re.compile(r"(?:https?://|www\.)[-_.?&~;+=/#0-9A-Za-z]{1,2076}")
            out.append(langs(langdetect.detect_langs(WWW)))
            Detector.URL_RE = old
            Detector.cleaning_text, old = (lambda self: None), Detector.cleaning_text
            out.append((langs(langdetect.detect_langs(RU)), det(Detector, RU)))
            Detector.cleaning_text = old
            NGram.N_GRAM = 2
            out.append(langs(langdetect.detect_langs("Otec matka syn.")))
            NGram.N_GRAM = 3
            g = copy.deepcopy(f)
            d = g.create(); d.append("Das ist ein Haus.")
            out.append(langs(pickle.loads(pickle.dumps(d)).get_probabilities()))
            out.append(langs(langdetect.detect_langs("Otec matka syn.")))
            return out
        expected = run()
        import pyroid
        warnings.simplefilter("ignore")
        pyroid.patch("langdetect")
        got = run()
        for e, g in zip(expected, got):
            assert e == g, (e, g)
        pyroid.unpatch("langdetect")
        assert run() == expected
        print("ok", len(expected))
    """
    out = run_python(code, timeout=240)
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.startswith("ok"), out.stdout


def test_shim_default_factory(run_python: Callable[..., Any]) -> None:
    code = """
        import json, sys, warnings
        sys.modules["langdetect"] = None
        import pyroid
        warnings.simplefilter("ignore")
        pyroid.patch("langdetect")
        import langdetect
        from langdetect import DetectorFactory, detector_factory
        import pyroid.langdetect as pl
        assert detector_factory._factory is None
        detector_factory.init_factory()
        assert detector_factory._factory is pl.detector_factory._factory is not None
        ours = pl.detector_factory._factory
        print(detector_factory._factory.get_lang_list()[:2] == ours.get_lang_list()[:2])
        custom = DetectorFactory()
        custom.load_json_profile([
            json.dumps({"name": "aa", "freq": {"a": 10, "b": 5, "ab": 4}, "n_words": [15, 4, 0]}),
            json.dumps({"name": "bb", "freq": {"b": 10, "c": 5, "bc": 4}, "n_words": [15, 4, 0]}),
        ])
        DetectorFactory.seed = 0
        detector_factory._factory = custom
        assert pl.detector_factory._factory is custom
        print([(x.lang, x.prob.hex()) for x in langdetect.detect_langs("ab ab")])
        same_dir = detector_factory.PROFILES_DIRECTORY == pl.PROFILES_DIRECTORY
        print("_factory" in dir(detector_factory), same_dir)
    """
    out = run_python(code)
    assert out.returncode == 0, out.stdout + out.stderr
    lines = out.stdout.splitlines()
    assert lines[0] == "True"
    assert lines[1] == "[('aa', '0x1.ffffffffff2d6p-1')]"  # langdetect 1.0.9's result
    assert lines[2] == "True True"


def test_langdetect_pickles_load_under_the_shim(
    run_python: Callable[..., Any], tmp_path: Path
) -> None:
    pytest.importorskip("langdetect")
    blob = tmp_path / "ld.pickle"
    make = f"""
        import pickle, langdetect
        from langdetect import DetectorFactory
        from langdetect.detector_factory import PROFILES_DIRECTORY
        f = DetectorFactory()
        f.load_profile(PROFILES_DIRECTORY)
        f.seed = 0
        d = f.create()
        d.append("Ceci est une phrase en français.")
        blob = pickle.dumps((f, d))  # before d detects: the loaded one detects
        def langs(text):
            e = f.create()
            e.append(text)
            return [(x.lang, x.prob.hex()) for x in e.get_probabilities()]
        with open({str(blob)!r}, "wb") as fh:
            want = (langs("Das ist ein Haus."), langs("Ceci est une phrase en français."))
            pickle.dump((blob, *want), fh)
    """
    out = run_python(make, timeout=240)
    assert out.returncode == 0, out.stdout + out.stderr
    # The pickle was written by this test's own child above.
    load = f"""
        import pickle, sys, warnings
        sys.modules["langdetect"] = None
        import pyroid
        warnings.simplefilter("ignore")
        pyroid.patch("langdetect")
        with open({str(blob)!r}, "rb") as fh:
            blob, want, want_d = pickle.load(fh)
        f, d = pickle.loads(blob)
        import pyroid.langdetect as pl
        assert type(f) is pl.DetectorFactory and type(d) is pl.Detector
        e = f.create(); e.append("Das ist ein Haus.")
        print([(x.lang, x.prob.hex()) for x in e.get_probabilities()] == want)
        print([(x.lang, x.prob.hex()) for x in d.get_probabilities()] == want_d)
    """
    out = run_python(load, timeout=240)
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.split() == ["True", "True"]


def test_ctrl_c_during_install_or_undo_restores_the_code(run_python: Callable[..., Any]) -> None:
    pytest.importorskip("langdetect")
    code = """
        import builtins, warnings
        import langdetect, langdetect.detector as d, langdetect.detector_factory as f
        import pyroid.langdetect._patch as lp
        fns = [d.Detector.append, d.Detector._detect_block, f.detect, f.detect_langs]
        codes = [fn.__code__ for fn in fns]
        def pristine():
            same = [fn.__code__ for fn in fns] == codes
            return same and not any(lp._HOOK in m.__dict__ for m in (d, f))
        fire = [0, None]
        def step(*args):
            fire[0] += 1
            if fire[0] == fire[1]:
                raise KeyboardInterrupt
            builtins.setattr(*args)
        lp.setattr = step
        installs = undos = 0
        for k in range(1, 20):
            fire[:] = [0, k]
            try:
                undo = lp.install()
            except KeyboardInterrupt:
                installs += 1
                assert pristine(), ("install", k)
                continue
            fire[:] = [0, None]
            assert not pristine()
            fire[:] = [0, k - 6 if k > 6 else None]
            try:
                undo()
            except KeyboardInterrupt:
                undos += 1
                fire[:] = [0, None]
                undo()  # an undo step runs again
            assert pristine(), ("undo", k)
        print(installs, undos)
    """
    out = run_python(code)
    assert out.returncode == 0, out.stdout + out.stderr
    installs, undos = map(int, out.stdout.split())
    # 6 swaps in install (2 hook globals, 4 codes); 4 setattr calls in undo (the codes).
    assert (installs, undos) == (6, 4), out.stdout
