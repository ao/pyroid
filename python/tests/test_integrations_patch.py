"""pyroid.integrations.patch: explicit, reversible ``sys.modules`` shims, ``pyroid run``, the hook.

Patching is process-wide state, so almost everything runs in a fresh interpreter. Tests that
need the real ftfy / fasttext / anyascii skip without them; the "without" cases hide a package
that is installed (a meta path finder that says it is not), so they run everywhere.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

import pyroid.integrations as integrations
from pyroid.integrations import _patch

pytestmark = pytest.mark.thread_unsafe(reason="subprocesses and process-wide import state")


def installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def real_ftfy_ok() -> bool:
    if not installed("ftfy"):
        return False
    from importlib.metadata import version

    return version("ftfy") == "6.3.1"


needs_ftfy = pytest.mark.skipif(not real_ftfy_ok(), reason="needs ftfy 6.3.1")
needs_anyascii = pytest.mark.skipif(not installed("anyascii"), reason="needs anyascii")
needs_numpy = pytest.mark.skipif(not installed("numpy"), reason="pyroid.fasttext needs numpy")
LID = Path(os.environ.get("PYROID_DATA", "/data")) / "models/lid.176.ftz"
needs_lid = pytest.mark.skipif(not LID.is_file(), reason=f"no {LID} (set PYROID_DATA)")

# Environment that would leak a patch (or a warning filter) into the children.
_CLEAN = ("PYTHONPATH", "PYROID_PATCH", "_PYROID_PATCH_WARNED", "PYTHONWARNINGS", "TEST_HIDE")

# Prepended to every child: TEST_HIDE=ftfy,... makes installed packages look uninstalled.
PRELUDE = """\
import importlib, importlib.abc, importlib.util, json, os, sys, warnings
HIDE = set(filter(None, os.environ.get("TEST_HIDE", "").split(",")))
class _Hide(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in HIDE:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None
if HIDE:
    sys.meta_path.insert(0, _Hide())
def patch_quietly(*names):
    import pyroid.integrations as pi
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pi.patch(*names)
"""


def child(
    code: str,
    *,
    args: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
    hide: tuple[str, ...] = (),
    cwd: Path | None = None,
    flags: tuple[str, ...] = (),
    prelude: bool = True,
    timeout: float = 180.0,
) -> subprocess.CompletedProcess[str]:
    """Run `code` (``-c``) in a fresh interpreter; not ``-I``, so ``PYTHONPATH`` reaches it."""
    child_env = {k: v for k, v in os.environ.items() if k not in _CLEAN}
    child_env.update(env or {})
    if hide:
        child_env["TEST_HIDE"] = ",".join(hide)
    source = (PRELUDE if prelude else "") + textwrap.dedent(code)
    cmd = [sys.executable, *flags, "-c", source, *args]
    try:
        return subprocess.run(
            cmd,
            env=child_env,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"child did not finish in {timeout} s\n{exc.stdout!r}\n{exc.stderr!r}")


def ok(proc: subprocess.CompletedProcess[str]) -> Any:
    """Assert the child succeeded; return the JSON it printed last (or None)."""
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    lines = proc.stdout.strip().splitlines()
    return json.loads(lines[-1]) if lines else None


# ---------------------------------------------------------------------------
# API checks (in this process: they must not change anything)
# ---------------------------------------------------------------------------


def test_pyroid_patch_alias_is_lazy() -> None:
    # `import pyroid; pyroid.patch(...)` works, and loads pyroid.integrations only when called.
    warned = ok(
        child(
            """
            import json, sys, warnings
            import pyroid
            assert "patch" in pyroid.__all__ and "unpatch" in pyroid.__all__
            assert pyroid.patch.__module__ == "pyroid"
            assert not any(m.startswith("pyroid.integrations") for m in sys.modules)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pyroid.patch("anyascii")
            import anyascii, pyroid.translit
            assert anyascii.anyascii is pyroid.translit.anyascii
            import pyroid.integrations as pi
            assert pi.patched() == ("anyascii",)
            pyroid.unpatch()
            assert pi.patched() == ()
            assert sys.modules.get("anyascii") is not anyascii  # the shim is gone
            print(json.dumps([(w.category.__name__, w.filename) for w in caught]))
            """
        )
    )
    assert warned, "no PatchWarning"
    assert all(cat == "PatchWarning" for cat, _ in warned), warned
    assert all(filename == "<string>" for _, filename in warned), warned  # the caller's line


def test_names_are_checked_before_anything_is_patched() -> None:
    try:
        with pytest.raises(ValueError, match=r"'ftfy', 'fasttext', 'anyascii'"):
            integrations.patch("ftfy", "nope")
        assert "ftfy" not in integrations.patched()
        with pytest.raises(ValueError, match="differs from Unidecode's"):
            integrations.patch("unidecode")
        with pytest.raises(TypeError, match="at least one name"):
            integrations.patch()
        with pytest.raises(TypeError, match="at least one name"):
            _patch.patching()
        with pytest.raises(ValueError, match="cannot patch 'nope'"):
            integrations.unpatch("nope")
        integrations.unpatch("anyascii")  # not patched: nothing to do
    finally:
        integrations.unpatch()
    assert integrations.patched() == ()
    assert issubclass(integrations.PatchWarning, UserWarning)
    assert _patch.SUPPORTED == ("ftfy", "fasttext", "anyascii")
    assert (Path(_patch.HOOK_DIR) / "sitecustomize.py").is_file()


def test_nothing_is_automatic() -> None:
    # Installing and importing pyroid changes no import: no .pth file, no sitecustomize, and
    # pyroid.integrations does not even load the patch machinery.
    out = ok(
        child(
            """
            import glob, site
            import pyroid, pyroid.ftfy, pyroid.translit, pyroid.integrations
            if importlib.util.find_spec("numpy"):
                import pyroid.fasttext
            assert "pyroid.integrations._patch" not in sys.modules
            real = {}
            for name in ("ftfy", "fasttext", "anyascii"):
                try:
                    real[name] = not getattr(importlib.import_module(name), "__pyroid__", False)
                except ImportError:
                    pass
            dirs = site.getsitepackages() + [site.getusersitepackages()]
            pth = [p for d in dirs for p in glob.glob(os.path.join(d, "*.pth"))]
            mentions = [p for p in pth if "pyroid" in open(p, encoding="utf-8").read()]
            sc = getattr(sys.modules.get("sitecustomize"), "__file__", "") or ""
            print(json.dumps({"real": real, "pth": mentions, "sitecustomize": sc}))
            """,
            flags=("-I",),
        )
    )
    assert all(out["real"].values()), out
    assert out["pth"] == []
    assert "pyroid" not in out["sitecustomize"]


# ---------------------------------------------------------------------------
# ftfy
# ---------------------------------------------------------------------------


@needs_ftfy
def test_ftfy_shim_with_real_ftfy() -> None:
    ok(
        child(
            """
            import types
            import pyroid.ftfy as impl
            patch_quietly("ftfy")
            import ftfy
            from ftfy import fix_text
            real = ftfy.__wrapped__
            assert ftfy is sys.modules["ftfy"] is importlib.import_module("ftfy")
            assert isinstance(ftfy, types.ModuleType) and type(ftfy) is not types.ModuleType
            assert ftfy.__name__ == "ftfy" and ftfy.__pyroid__ is True
            assert ftfy.__version__ == impl.FTFY_VERSION == "6.3.1"
            assert real is not ftfy and real.__file__.endswith("ftfy/__init__.py")
            # pyroid's objects themselves (no wrapper), except what pyroid only delegates.
            for name in impl.__all__:
                want = getattr(real if name in impl.DELEGATED else impl, name)
                assert getattr(ftfy, name) is want, name
            assert fix_text is impl.fix_text and ftfy.ftfy is impl.fix_text
            assert ftfy.fix_text("âœ” No problems") == "✔ No problems"
            # Everything else is the real ftfy's, submodules included.
            import ftfy.fixes, ftfy.badness
            import ftfy.cli  # not imported by ftfy itself: loaded through the shim's __path__
            from ftfy.chardata import possible_encoding
            assert ftfy.fixes is real.fixes and ftfy.fixes is sys.modules["ftfy.fixes"]
            assert ftfy.cli is sys.modules["ftfy.cli"] and ftfy.cli.__name__ == "ftfy.cli"
            assert ftfy.guess_bytes is real.guess_bytes and ftfy.fix_file is real.fix_file
            assert ftfy.__file__ == real.__file__ and ftfy.__path__ == real.__path__
            assert {"fixes", "fix_text", "guess_bytes"} <= set(dir(ftfy))
            assert {"fixes", "fix_text", "guess_bytes"} <= set(ftfy.__all__)
            try:
                ftfy.no_such_thing
            except AttributeError as e:
                assert "no_such_thing" in str(e)
            else:
                raise AssertionError("no AttributeError")
            # find_spec / resources see a package with the real loader and location.
            spec = importlib.util.find_spec("ftfy")
            assert spec is ftfy.__spec__ and spec.name == "ftfy"
            assert spec.origin == real.__spec__.origin and spec.loader is real.__spec__.loader
            import importlib.resources
            assert importlib.resources.files("ftfy").joinpath("fixes.py").is_file()
            assert "pyroid shim" in repr(ftfy)
            """
        )
    )


def test_ftfy_shim_without_real_ftfy() -> None:
    ok(
        child(
            """
            import pyroid, pyroid.ftfy as impl
            patch_quietly("ftfy")
            import ftfy
            from ftfy import fix_text, TextFixerConfig, guess_bytes
            assert ftfy.__pyroid__ and ftfy.__wrapped__ is None
            assert fix_text is impl.fix_text and guess_bytes is impl.guess_bytes
            assert ftfy.fix_text("Ã³") == "ó"
            assert ftfy.fix_text("Ã³", TextFixerConfig(fix_encoding=False)) == "Ã³"
            assert importlib.util.find_spec("ftfy") is ftfy.__spec__
            try:
                ftfy.fixes
            except AttributeError as e:
                assert "pyroid's ftfy shim" in str(e) and "not installed" in str(e), e
            else:
                raise AssertionError("no AttributeError")
            try:
                import ftfy.fixes
            except ModuleNotFoundError as e:
                assert e.name == "ftfy.fixes", e
            else:
                raise AssertionError("ftfy.fixes imported")
            # What pyroid only delegates raises PyroidError, as without the patch.
            try:
                ftfy.guess_bytes(b"abc")
            except pyroid.PyroidError as e:
                assert "not installed" in str(e)
            else:
                raise AssertionError("no PyroidError")
            """,
            hide=("ftfy",),
        )
    )


@needs_ftfy
def test_ftfy_fallback_reaches_the_real_ftfy_not_the_shim() -> None:
    # pyroid.ftfy's slow path does `import ftfy` at call time; under the patch that is the shim,
    # whose fix_text is pyroid's again: pyroid.ftfy must use its __wrapped__ (or it recurses).
    ok(
        child(
            """
            import pyroid, pyroid.ftfy as impl
            fallback = impl._ftfy
            patch_quietly("ftfy")
            import ftfy
            real = ftfy.__wrapped__
            assert impl._ftfy is fallback  # pyroid.ftfy sees through the shim itself
            def calls(fn):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        result = ("ok", fn())
                    except Exception as e:
                        result = ("error", type(e).__name__, str(e))
                return result, [(w.category.__name__, w.lineno) for w in caught]
            cases = [
                lambda: ftfy.fix_text("abc", max_decode_length=5.0),
                lambda: ftfy.fix_text("âœ” ok", max_decode_length=1e6),
                lambda: ftfy.fix_text("x", normalization="nfc"),
                lambda: ftfy.fix_text_batch(["x", "Ã³"], normalization="nfc"),
                lambda: ftfy.fix_and_explain("Ã³", max_decode_length=7.0),
                lambda: ftfy.fix_encoding("Ã³", max_decode_length=2.0),
                lambda: impl.guess_bytes(b"abc"),
                lambda: impl.apply_plan("x", []),
            ]
            for case in cases:
                first, warned = calls(case)
                line = case.__code__.co_firstlineno
                assert warned == [("FallbackWarning", line)], (warned, line)  # once, at the caller
                again, warned = calls(case)
                assert again == first and warned == [], warned
            # The same results and errors as the real ftfy.
            assert calls(cases[0])[0] == ("ok", real.fix_text("abc", max_decode_length=5.0))
            assert calls(cases[1])[0] == ("ok", "✔ ok")
            assert calls(cases[2])[0][:2] == ("error", "ValueError")
            assert calls(cases[6])[0] == ("ok", real.guess_bytes(b"abc"))
            import pyroid.integrations as pi
            pi.unpatch("ftfy")
            assert impl._ftfy is fallback
            """
        )
    )


def test_ftfy_fallback_without_real_ftfy_raises_pyroid_error() -> None:
    ok(
        child(
            """
            import pyroid, pyroid.ftfy as impl
            patch_quietly("ftfy")
            import ftfy
            for call in (
                lambda: ftfy.fix_text("abc", normalization="XYZ"),
                lambda: ftfy.fix_text_batch(["x"], normalization="XYZ"),
                lambda: ftfy.apply_plan("x", []),
                lambda: impl.explain_unicode("x"),
            ):
                try:
                    call()
                except pyroid.PyroidError as e:
                    assert "not installed" in str(e), e
                else:
                    raise AssertionError("no PyroidError")
            assert ftfy.fix_text("Ã³") == "ó"
            """,
            hide=("ftfy",),
        )
    )


@needs_ftfy
def test_other_ftfy_version_warns_and_falls_back_to_it() -> None:
    ok(
        child(
            """
            import ftfy as real
            real.__version__ = "6.2.0"  # what pyroid sees if another ftfy is installed
            import pyroid.ftfy, pyroid.integrations as pi
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pi.patch("ftfy")
            messages = [str(w.message) for w in caught]
            assert [w.category for w in caught] == [pi.PatchWarning] * 2, messages
            assert "the installed ftfy is 6.2.0" in messages[1] and "6.3.1" in messages[1]
            import ftfy
            assert ftfy.__version__ == "6.3.1" and ftfy.__wrapped__ is real
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ftfy.fix_text("abc", max_decode_length=5.0)
            # pyroid.ftfy's own warning about the version it falls back to: it got the real ftfy.
            got = [(w.category.__name__, str(w.message)) for w in caught]
            assert ("RuntimeWarning", "pyroid.ftfy is falling back to ftfy 6.2.0, not 6.3.1") in got
            """
        )
    )


# ---------------------------------------------------------------------------
# fasttext
# ---------------------------------------------------------------------------

TEXTS = [
    "hello world, this is English",
    "bonjour tout le monde",
    "guten Morgen, wie geht es dir",
    "日本語",
]


@needs_numpy
@needs_lid
def test_fasttext_shim() -> None:
    out = ok(
        child(
            """
            import pyroid, pyroid.fasttext as impl
            path, texts = sys.argv[1], json.loads(sys.argv[2])
            before = None
            if importlib.util.find_spec("fasttext"):
                import fasttext
                m = fasttext.load_model(path)
                before = [[list(lab), [float(p) for p in pr]] for lab, pr in
                          (m.predict(t, k=3) for t in texts)]
            patch_quietly("fasttext")
            import fasttext
            import fasttext.FastText
            from fasttext.FastText import _FastText, load_model
            assert fasttext.__pyroid__ and fasttext.FastText.__pyroid__
            assert fasttext.FastText is sys.modules["fasttext.FastText"]
            assert fasttext.load_model is impl.load_model is load_model
            assert fasttext.tokenize is impl.tokenize is fasttext.FastText.tokenize
            assert importlib.util.find_spec("fasttext") is fasttext.__spec__
            assert importlib.util.find_spec("fasttext.FastText") is fasttext.FastText.__spec__
            assert fasttext.tokenize("a b\\nc") == ["a", "b", "</s>", "c"]
            # datatrove's pattern: _FastText(path), then score.item() on the probabilities.
            model = _FastText(path)
            assert type(model) is impl.Model and isinstance(model, _FastText)
            assert type(_FastText(model_path=path)) is impl.Model
            assert type(fasttext.load_model(path)) is impl.Model
            langs, scores = model.predict(texts[1].replace("\\n", " "), k=1)
            pairs = {lang.split("__")[2]: score.item() for lang, score in zip(langs, scores)}
            assert list(pairs) == ["fr"], pairs
            after = [[list(lab), [float(p) for p in pr]] for lab, pr in
                     (model.predict(t, k=3) for t in texts)]
            real = fasttext.FastText.__wrapped__
            if real is None:
                try:
                    _FastText()
                except pyroid.PyroidError as e:
                    assert "not installed" in str(e), e
                else:
                    raise AssertionError("no PyroidError")
                try:
                    fasttext.train_supervised
                except AttributeError as e:
                    assert "pyroid's fasttext shim" in str(e)
                else:
                    raise AssertionError("no AttributeError")
            else:
                assert fasttext.__wrapped__.FastText is real
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    empty = _FastText()  # no model: only the real fastText can make that
                assert type(empty) is real._FastText and isinstance(empty, _FastText)
                assert [w.category for w in caught] == [pyroid.FallbackWarning]
            print(json.dumps({"before": before, "after": after, "real": real is not None}))
            """,
            args=(str(LID), json.dumps(TEXTS)),
        )
    )
    if out["before"] is not None:  # the same labels and probabilities as fastText
        for (lab0, p0), (lab1, p1) in zip(out["before"], out["after"], strict=True):
            assert lab0 == lab1
            assert p1 == pytest.approx(p0, abs=1e-6)


@needs_numpy
@needs_lid
def test_fasttext_shim_without_fasttext() -> None:
    # As on CPython 3.14+ / free-threaded builds, where no fastText wheel exists.
    ok(
        child(
            """
            import pyroid, pyroid.fasttext as impl
            patch_quietly("fasttext")
            import fasttext
            from fasttext.FastText import _FastText
            assert fasttext.__wrapped__ is None and fasttext.FastText.__wrapped__ is None
            assert importlib.util.find_spec("fasttext") is not None
            assert not hasattr(fasttext, "train_supervised")  # fast-langdetect checks this
            model = _FastText(sys.argv[1])
            assert model.predict("bonjour tout le monde")[0] == ("__label__fr",)
            try:
                import fasttext.util
            except ModuleNotFoundError as e:
                assert e.name == "fasttext.util"
            else:
                raise AssertionError("fasttext.util imported")
            try:
                _FastText(None, {"lr": 0.1})
            except pyroid.PyroidError as e:
                assert "_FastText(args=...)" in str(e), e
            else:
                raise AssertionError("no PyroidError")
            """,
            args=(str(LID),),
            hide=("fasttext", "fasttext_pybind"),
        )
    )


# ---------------------------------------------------------------------------
# anyascii
# ---------------------------------------------------------------------------


@needs_anyascii
def test_anyascii_shim_and_code_imported_before_keeps_working() -> None:
    ok(
        child(
            """
            from anyascii import anyascii as early  # imported before patch(): the real one
            import pyroid.translit as impl
            patch_quietly("anyascii")
            import anyascii
            assert anyascii.anyascii is impl.anyascii and early is not impl.anyascii
            assert anyascii.__version__ == impl.ANYASCII_VERSION == "0.3.3"
            assert anyascii.__wrapped__._blocks is anyascii._blocks  # forwarded
            # The real function loads its tables with importlib.resources ("anyascii._data"),
            # which resolves the package through sys.modules: the shim. It must still work.
            assert not anyascii._blocks
            text = "Björk Guðmundsdóttir, 東京, Ελλάδα"
            want = "Bjork Gudmundsdottir, DongJing, Ellada"
            assert early(text) == anyascii.anyascii(text) == want
            assert len(anyascii._blocks) >= 3
            """
        )
    )


def test_anyascii_shim_without_anyascii() -> None:
    ok(
        child(
            """
            patch_quietly("anyascii")
            from anyascii import anyascii
            import pyroid.translit as impl
            assert anyascii is impl.anyascii and anyascii("Björk") == "Bjork"
            assert sys.modules["anyascii"].__wrapped__ is None
            """,
            hide=("anyascii",),
        )
    )


# ---------------------------------------------------------------------------
# unpatch, idempotence, warnings
# ---------------------------------------------------------------------------

SCENARIOS = [
    (name, scenario)
    for name in ("ftfy", "fasttext", "anyascii")
    for scenario in ("not imported", "imported", "blocked", "not installed")
]


@pytest.mark.parametrize(("name", "scenario"), SCENARIOS, ids=[f"{n}-{s}" for n, s in SCENARIOS])
def test_unpatch_restores_sys_modules_exactly(name: str, scenario: str) -> None:
    if name == "fasttext" and not installed("numpy"):
        pytest.skip("pyroid.fasttext needs numpy")
    if scenario in ("not imported", "imported") and not installed(name):
        pytest.skip(f"needs {name}")
    ok(
        child(
            """
            import pyroid.ftfy, pyroid.integrations as pi
            name, scenario = sys.argv[1], sys.argv[2]
            def owned():
                return {k: v for k, v in sys.modules.items()
                        if k == name or k.startswith(name + ".")}
            if scenario == "imported":
                importlib.import_module(name)
            elif scenario == "blocked":
                sys.modules[name] = None
            before = owned()
            fallback = pyroid.ftfy._ftfy
            patch_quietly(name)
            shim = sys.modules[name]
            assert shim.__pyroid__ and pi.patched() == (name,)
            subs = {"ftfy": ["ftfy.fixes", "ftfy.cli"], "fasttext": ["fasttext.FastText"]}
            for sub in subs.get(name, []):
                try:
                    importlib.import_module(sub)
                except ImportError:
                    assert shim.__wrapped__ is None and sub != "fasttext.FastText"
            pi.unpatch()
            after = owned()
            assert sorted(after) == sorted(before), (sorted(before), sorted(after))
            assert all(after[k] is before[k] for k in before)
            assert pyroid.ftfy._ftfy is fallback and pi.patched() == ()
            # A module that imported the shim keeps a working one.
            fn = {"ftfy": "fix_text", "fasttext": "tokenize", "anyascii": "anyascii"}[name]
            assert callable(getattr(shim, fn))
            """,
            args=(name, scenario),
            hide=(name,) if scenario == "not installed" else (),
        )
    )


def test_patch_is_idempotent_and_warns_once_per_name() -> None:
    ok(
        child(
            """
            import pyroid.ftfy, pyroid.translit  # (their import-time warnings, if any, go first)
            import pyroid.integrations as pi
            import pyroid.integrations._patch as p
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                line = sys._getframe().f_lineno; pi.patch("ftfy", "anyascii", "ftfy")
                ftfy, anyascii = sys.modules["ftfy"], sys.modules["anyascii"]
                pi.patch("ftfy")
                assert sys.modules["ftfy"] is ftfy
                assert pi.patched() == ("ftfy", "anyascii")
                pi.unpatch("ftfy")
                assert pi.patched() == ("anyascii",) and sys.modules.get("ftfy") is not ftfy
                pi.patch("ftfy")  # a new shim, but no second warning
                assert sys.modules["ftfy"] is not ftfy and sys.modules["ftfy"].__pyroid__
                with p.patching("ftfy", "anyascii"):  # both already patched: left patched
                    pass
                assert pi.patched() == ("ftfy", "anyascii")
                pi.unpatch()
                with p.patching("anyascii") as nothing:
                    assert nothing is None and pi.patched() == ("anyascii",)
                assert pi.patched() == ()
            warned = [w for w in caught if w.category is pi.PatchWarning]
            assert len(warned) == len(caught) == 2, [str(w.message) for w in caught]
            assert all((w.filename, w.lineno) == ("<string>", line) for w in warned)
            text = [str(w.message) for w in warned]
            assert "`import ftfy` now gives pyroid's drop-in (pyroid.ftfy" in text[0]
            assert "6.3.1" in text[0] and "unpatch('ftfy')" in text[0]
            assert "before this call keep the real one" in text[0] or "unavailable" in text[0]
            assert "`import anyascii`" in text[1] and "unpatch('anyascii')" in text[1]
            """
        )
    )


def test_patch_from_many_threads_installs_one_shim() -> None:
    ok(
        child(
            """
            import threading
            import pyroid.ftfy, pyroid.integrations as pi
            barrier, shims = threading.Barrier(8), []
            def work():
                barrier.wait()
                pi.patch("ftfy")
                shims.append(sys.modules["ftfy"])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                threads = [threading.Thread(target=work) for _ in range(8)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
            assert len(shims) == 8 and len({id(s) for s in shims}) == 1
            assert len([w for w in caught if w.category is pi.PatchWarning]) == 1
            """
        )
    )


@needs_ftfy
def test_modules_that_imported_before_keep_the_real_module() -> None:
    ok(
        child(
            """
            import ftfy as early
            from ftfy import fix_text as early_fix
            patch_quietly("ftfy")
            import ftfy
            assert early is ftfy.__wrapped__ and early is not ftfy
            assert early.fix_text is early_fix and early_fix is not ftfy.fix_text
            assert early.fix_text("Ã³") == ftfy.fix_text("Ã³") == "ó"
            """
        )
    )


# ---------------------------------------------------------------------------
# pyroid run
# ---------------------------------------------------------------------------

RUN = "import sys; from pyroid.integrations._patch import run; sys.exit(run(sys.argv[1:]))"

PROBE = """\
import json, os, sys
try:
    import ftfy
except ImportError:  # not patched, not installed
    ftfy = None
print(json.dumps({
    "argv": sys.argv, "name": __name__, "path0": sys.path[0],
    "spec": getattr(__spec__, "name", None), "patched": getattr(ftfy, "__pyroid__", False),
    "env": os.environ.get("PYROID_PATCH"),
    "hook": os.environ.get("PYTHONPATH", "").split(os.pathsep)[0],
}))
sys.exit(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
"""


def run_cli(
    argv: list[str], cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return child(RUN, args=tuple(argv), cwd=cwd, env=env, prelude=False)


def test_run_script(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/probe.py").write_text(PROBE)
    proc = run_cli(["-p", "ftfy", "sub/probe.py", "3", "--flag"], cwd=tmp_path)
    assert proc.returncode == 3, proc.stderr  # sys.exit in the script is the exit code
    out = json.loads(proc.stdout)
    assert out["argv"] == ["sub/probe.py", "3", "--flag"]
    assert out["name"] == "__main__"
    assert out["spec"] is None
    assert out["path0"] == str((tmp_path / "sub").resolve())  # as `python sub/probe.py`
    assert out["patched"] is True
    assert out["env"] == "ftfy"
    assert out["hook"] == _patch.HOOK_DIR
    assert proc.stderr.count("PatchWarning") == 1, proc.stderr


def test_run_module(tmp_path: Path) -> None:
    (tmp_path / "probe.py").write_text(PROBE)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/__init__.py").write_text("")
    (tmp_path / "pkg/__main__.py").write_text(PROBE)
    proc = run_cli(["-p", "ftfy,anyascii", "-m", "probe", "0", "x"], cwd=tmp_path)
    out = ok(proc)
    assert out["argv"] == [str(tmp_path / "probe.py"), "0", "x"]  # as `python -m probe 0 x`
    assert out["name"] == "__main__"
    assert out["spec"] == "probe"
    assert out["path0"] == str(tmp_path)
    assert out["patched"] is True
    assert out["env"] == "ftfy,anyascii"
    out = ok(run_cli(["-p", "ftfy", "-mpkg"], cwd=tmp_path))
    assert out["spec"] == "pkg.__main__"
    assert out["patched"] is True


def test_run_target_exceptions_propagate(tmp_path: Path) -> None:
    (tmp_path / "boom.py").write_text("raise ValueError('boom')\n")
    proc = run_cli(["-p", "ftfy", "boom.py"], cwd=tmp_path)
    assert proc.returncode == 1
    assert "ValueError: boom" in proc.stderr
    assert 'boom.py", line 1' in proc.stderr


@pytest.mark.parametrize(
    ("argv", "code", "message"),
    [
        ([], 2, "give a SCRIPT or -m MODULE"),
        (["-p"], 2, "expected a comma-separated list"),
        (["-m"], 2, "expected a module name"),
        (["--bogus", "x.py"], 2, "unrecognized option '--bogus'"),
        (["-p", "ftfy,nope", "x.py"], 2, "cannot patch 'nope'"),
        (["-p", "unidecode", "x.py"], 2, "differs from Unidecode's"),
        (["-p", " , ", "x.py"], 2, "no module names given"),
        (["missing.py"], 2, "can't open file"),
        (["-p", "ftfy", "-m", "no_such_module_xyz"], 1, "No module named no_such_module_xyz"),
        (["-h"], 0, ""),
    ],
)
def test_run_usage(tmp_path: Path, argv: list[str], code: int, message: str) -> None:
    proc = run_cli(argv, cwd=tmp_path)
    assert proc.returncode == code, proc.stderr
    assert message in proc.stderr
    if code == 0:
        assert proc.stdout.startswith("usage: pyroid run")
        assert "-p, --patch NAMES" in proc.stdout


def test_run_names_default_to_the_environment_then_ftfy_fasttext(tmp_path: Path) -> None:
    (tmp_path / "probe.py").write_text(PROBE)
    out = ok(run_cli(["probe.py"], cwd=tmp_path, env={"PYROID_PATCH": " anyascii "}))
    assert out["env"] == "anyascii"
    assert out["patched"] is False
    out = ok(run_cli(["probe.py"], cwd=tmp_path))
    expected = "ftfy,fasttext" if installed("numpy") else "ftfy"
    assert out["env"] == expected
    assert out["patched"] is True
    proc = run_cli(["probe.py"], cwd=tmp_path, env={"PYROID_PATCH": "nope"})
    assert proc.returncode == 2
    assert "$PYROID_PATCH: cannot patch 'nope'" in proc.stderr


CHILDREN = """\
import json, multiprocessing as mp, subprocess, sys

def probe(_=None):
    import anyascii, ftfy
    return [getattr(ftfy, "__pyroid__", False), getattr(anyascii, "__pyroid__", False)]

if __name__ == "__main__":
    results = {"main": probe()}
    for method in ("spawn", "forkserver"):
        with mp.get_context(method).Pool(2) as pool:
            results[method] = pool.map(probe, range(2))
    code = "import anyascii, ftfy; print(getattr(ftfy, '__pyroid__', 0), anyascii.__pyroid__)"
    sub = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    results["subprocess"] = sub.stdout.split()
    results["subprocess_stderr"] = sub.stderr
    print(json.dumps(results))
"""


def test_run_patches_child_processes_and_warns_once(tmp_path: Path) -> None:
    (tmp_path / "children.py").write_text(CHILDREN)
    proc = run_cli(["-p", "ftfy,anyascii", "children.py"], cwd=tmp_path)
    out = ok(proc)
    assert out["main"] == [True, True]
    assert out["spawn"] == out["forkserver"] == [[True, True], [True, True]]
    assert out["subprocess"] == ["True", "True"]
    # Shown by the parent only, not again by each worker (stderr is shared with them).
    assert out["subprocess_stderr"] == ""
    assert proc.stderr.count("PatchWarning") == 2, proc.stderr


# ---------------------------------------------------------------------------
# The sitecustomize hook
# ---------------------------------------------------------------------------

HOOK_PROBE = """\
import json, os, sys
try:
    import ftfy
except ImportError:  # not patched, not installed
    ftfy = None
sc = sys.modules.get("sitecustomize")
print(json.dumps({
    "ftfy": getattr(ftfy, "__pyroid__", False),
    "loaded": "pyroid.integrations._patch" in sys.modules,
    "sitecustomize": getattr(sc, "__file__", None), "mark": getattr(sc, "MARK", None),
    "other": os.environ.get("OTHER_SITECUSTOMIZE"),
    "warned": os.environ.get("_PYROID_PATCH_WARNED"),
}))
"""


def hook_env(*extra: Path, **env: str) -> dict[str, str]:
    path = os.pathsep.join([_patch.HOOK_DIR, *map(str, extra)])
    return {"PYTHONPATH": path, **env}


def test_hook_patches_from_the_environment(tmp_path: Path) -> None:
    env = hook_env(PYROID_PATCH="ftfy,anyascii")
    proc = child(HOOK_PROBE, env=env, prelude=False, cwd=tmp_path)
    out = ok(proc)
    assert out["ftfy"] is True
    assert out["warned"] == "ftfy,anyascii"
    assert proc.stderr.count("PatchWarning") == 2, proc.stderr
    # Already warned higher up the process tree: quiet.
    env = hook_env(PYROID_PATCH="ftfy", _PYROID_PATCH_WARNED="ftfy")
    proc = child(HOOK_PROBE, env=env, prelude=False, cwd=tmp_path)
    assert ok(proc)["ftfy"] is True
    assert proc.stderr == ""


def test_hook_is_inert_without_the_variable(tmp_path: Path) -> None:
    out = ok(child(HOOK_PROBE, env=hook_env(), prelude=False, cwd=tmp_path))
    assert out["ftfy"] is False
    assert out["loaded"] is False


def test_hook_runs_the_sitecustomize_it_shadows(tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / "sitecustomize.py").write_text(
        "import os\nos.environ['OTHER_SITECUSTOMIZE'] = 'ran'\nMARK = 42\n"
    )
    for env in (hook_env(other, PYROID_PATCH="ftfy"), hook_env(other)):
        proc = child(HOOK_PROBE, env=env, prelude=False, cwd=tmp_path)
        out = ok(proc)
        assert out["other"] == "ran"
        assert out["mark"] == 42
        assert out["sitecustomize"] == str(other / "sitecustomize.py")
        assert out["ftfy"] is ("PYROID_PATCH" in env)
        assert "Error in sitecustomize" not in proc.stderr


def test_hook_warns_about_unknown_names(tmp_path: Path) -> None:
    proc = child(HOOK_PROBE, env=hook_env(PYROID_PATCH="ftfy,nope"), prelude=False, cwd=tmp_path)
    assert ok(proc)["ftfy"] is True
    assert "ignoring 'nope' in $PYROID_PATCH" in proc.stderr


def test_hook_is_inert_when_imported_as_part_of_the_package() -> None:
    # pkgutil.walk_packages, stubtest, docs tools... may import it as a (namespace) submodule.
    ok(
        child(
            """
            before = sys.modules.get("sitecustomize")
            importlib.import_module("pyroid.integrations._autopatch.sitecustomize")
            assert sys.modules.get("sitecustomize") is before
            assert "pyroid.integrations._patch" not in sys.modules
            """,
            env={"PYROID_PATCH": "ftfy"},
        )
    )


def test_hook_skips_an_interpreter_without_pyroid(tmp_path: Path) -> None:
    # A subprocess may run another Python: the hook must neither fail nor complain there.
    fake = tmp_path / "fake"
    (fake / "pyroid").mkdir(parents=True)
    (fake / "pyroid/__init__.py").write_text("raise ImportError('not this pyroid')\n")
    code = "import sys; print(sys.modules['sitecustomize'].__name__)"
    proc = child(code, env=hook_env(fake, PYROID_PATCH="ftfy"), prelude=False, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "sitecustomize"
    assert proc.stderr == ""
