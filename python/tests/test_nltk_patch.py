"""pyroid.patch("nltk"): pyroid.nltk's tokenizers on NLTK's own modules, and NLTK's back after.

With NLTK installed, ``patch("nltk")`` sets pyroid's ``sent_tokenize``, ``word_tokenize``,
``punkt_model_available`` and tokenizer classes on ``nltk``, ``nltk.tokenize`` and the
``punkt``, ``treebank`` and ``destructive`` submodules; ``unpatch`` puts the exact original
objects back, and pyroid's fallbacks call NLTK's originals meanwhile (never pyroid again).
Without NLTK the same names come from shims. Patching is process-wide state, so every test runs
in a fresh interpreter.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from typing import Any

import pytest

from pyroid.integrations import _patch

pytestmark = pytest.mark.thread_unsafe(reason="subprocesses and process-wide import state")

NLTK_VERSION = "3.10.4"


def _nltk_ok() -> bool:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("nltk") == NLTK_VERSION
    except PackageNotFoundError:
        return False


needs_nltk = pytest.mark.skipif(not _nltk_ok(), reason=f"needs nltk {NLTK_VERSION}")

# Environment that would leak a patch (or a warning filter) into the children.
_CLEAN = ("PYTHONPATH", "PYROID_PATCH", "_PYROID_PATCH_WARNED", "PYTHONWARNINGS", "TEST_HIDE")

# Prepended to every child: TEST_HIDE=nltk makes an installed NLTK look uninstalled.
PRELUDE = """\
import importlib, importlib.abc, json, os, sys, warnings
HIDE = set(filter(None, os.environ.get("TEST_HIDE", "").split(",")))
class _Hide(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in HIDE:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None
if HIDE:
    sys.meta_path.insert(0, _Hide())
# What patch("nltk") swaps, by module (pyroid.integrations._patch._NLTK_SWAPS, spelled out).
_TOP = ["sent_tokenize", "word_tokenize", "punkt_model_available", "PunktSentenceTokenizer",
        "PunktTokenizer", "TreebankWordTokenizer", "TreebankWordDetokenizer", "NLTKWordTokenizer",
        "wordpunct_tokenize", "WordPunctTokenizer", "RegexpTokenizer", "WhitespaceTokenizer",
        "BlanklineTokenizer", "regexp_tokenize", "blankline_tokenize"]
SWAPS = {
    "nltk": _TOP,
    "nltk.tokenize": _TOP,
    "nltk.tokenize.punkt": ["PunktSentenceTokenizer", "PunktTokenizer"],
    "nltk.tokenize.regexp": ["RegexpTokenizer", "WhitespaceTokenizer", "BlanklineTokenizer",
                             "WordPunctTokenizer", "regexp_tokenize", "blankline_tokenize",
                             "wordpunct_tokenize"],
    "nltk.tokenize.treebank": ["TreebankWordTokenizer", "TreebankWordDetokenizer"],
    "nltk.tokenize.destructive": ["NLTKWordTokenizer"],
}
TEXTS = [
    "Mr. Smith went to Washington. He liked it!",
    "They'll pay $3.88 (roughly). \\"Yes,\\" he said. Didn't they?",
    "",
    "Ich muss usw. fur einen Kuchen einzukaufen. Ich muss.",
]
def results(mod):
    # Everything patch("nltk") swaps, called through `mod` (nltk itself).
    import nltk.tokenize.punkt as P, nltk.tokenize.treebank as T, nltk.tokenize.destructive as D
    import nltk.tokenize.regexp as R
    out = {}
    for t in TEXTS:
        out.setdefault("sent", []).append(mod.sent_tokenize(t))
        out.setdefault("sent_de", []).append(mod.tokenize.sent_tokenize(t, "german"))
        out.setdefault("word", []).append(mod.word_tokenize(t))
        out.setdefault("word_line", []).append(mod.tokenize.word_tokenize(t, preserve_line=True))
        out.setdefault("punkt", []).append(P.PunktTokenizer("german").tokenize(t, False))
        out.setdefault("spans", []).append(list(mod.PunktTokenizer().span_tokenize(t)))
        out.setdefault("pst", []).append(P.PunktSentenceTokenizer().tokenize(t))
        out.setdefault("tb", []).append(T.TreebankWordTokenizer().tokenize(t, True))
        out.setdefault("tb_spans", []).append(list(mod.TreebankWordTokenizer().span_tokenize(t)))
        out.setdefault("nw", []).append(D.NLTKWordTokenizer().tokenize(t))
        toks = mod.word_tokenize(t)
        out.setdefault("detok", []).append(T.TreebankWordDetokenizer().detokenize(toks))
        out.setdefault("wordpunct", []).append(mod.wordpunct_tokenize(t))
        out.setdefault("blankline", []).append(mod.tokenize.blankline_tokenize(t + "\\n\\n" + t))
        out.setdefault("regexp", []).append(R.regexp_tokenize(t, r"[\\w']+"))
        out.setdefault("rx", []).append(R.RegexpTokenizer(r"\\s+", True, False).tokenize(t))
        out.setdefault("rx_spans", []).append(list(mod.WhitespaceTokenizer().span_tokenize(t)))
        out.setdefault("rx_fb", []).append(R.RegexpTokenizer(r"[A-Z]\\w*").tokenize(t))
    out["available"] = [mod.tokenize.punkt_model_available(), mod.punkt_model_available("xx")]
    return out
"""


def child(
    code: str,
    *,
    env: dict[str, str] | None = None,
    hide: tuple[str, ...] = (),
    isolated: bool = True,
) -> Any:
    """Run `code` in a fresh interpreter; assert it succeeded; return the JSON it printed last."""
    child_env = {k: v for k, v in os.environ.items() if k not in _CLEAN}
    child_env.update(env or {})
    if hide:
        child_env["TEST_HIDE"] = ",".join(hide)
    flags = ["-I"] if isolated else []
    cmd = [sys.executable, *flags, "-c", PRELUDE + textwrap.dedent(code)]
    try:
        proc = subprocess.run(
            cmd, env=child_env, capture_output=True, text=True, timeout=300, check=False
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"child did not finish in 300 s\n{exc.stdout!r}\n{exc.stderr!r}")
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    lines = proc.stdout.strip().splitlines()
    return json.loads(lines[-1]) if lines else None


def test_swaps_list_is_the_documented_one() -> None:
    swaps = {mod: list(names) for mod, names in _patch._NLTK_SWAPS.items()}
    assert swaps["nltk.tokenize.punkt"] == ["PunktSentenceTokenizer", "PunktTokenizer"]
    assert swaps["nltk.tokenize.treebank"] == ["TreebankWordTokenizer", "TreebankWordDetokenizer"]
    assert swaps["nltk.tokenize.destructive"] == ["NLTKWordTokenizer"]
    assert "punkt_model_available" in swaps["nltk.tokenize"]
    assert "nltk" in _patch.SUPPORTED


@needs_nltk
def test_patch_swaps_and_unpatch_restores(punkt_tab: Any) -> None:
    out = child(
        """
        import nltk, nltk.tokenize, nltk.tokenize.punkt, nltk.tokenize.treebank
        import nltk.tokenize.destructive
        from nltk.tokenize.api import TokenizerI
        import pyroid, pyroid.nltk as pn
        mods = {m: importlib.import_module(m) for m in SWAPS}
        originals = {(m, n): getattr(mods[m], n) for m in SWAPS for n in SWAPS[m]}
        others = {n: getattr(nltk.tokenize.punkt, n) for n in ("PunktTrainer", "PunktParameters",
                  "PunktLanguageVars", "PunktToken", "load_punkt_params")}
        others["TweetTokenizer"] = nltk.tokenize.TweetTokenizer
        others["_get_punkt_tokenizer"] = nltk.tokenize._get_punkt_tokenizer
        want = results(nltk)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pyroid.patch("nltk")
            pyroid.patch("nltk")  # idempotent, warns once
        import pyroid.integrations as pi
        assert pi.patched() == ("nltk",)
        not_swapped = [f"{m}.{n}" for m in SWAPS for n in SWAPS[m]
                       if getattr(mods[m], n) is not getattr(pn, n)]
        assert sys.modules["nltk"] is nltk  # no shim: NLTK's own modules
        got = results(nltk)
        kinds = {
            "punkt": isinstance(nltk.tokenize.punkt.PunktTokenizer(), TokenizerI),
            "pst": isinstance(nltk.PunktSentenceTokenizer(), TokenizerI),
            "tb": isinstance(nltk.TreebankWordTokenizer(), TokenizerI),
            "nw": isinstance(nltk.NLTKWordTokenizer(), TokenizerI),
            "detok": isinstance(nltk.TreebankWordDetokenizer(), TokenizerI),
            "regexp": isinstance(nltk.RegexpTokenizer(r"\\w+"), TokenizerI),
            "wordpunct": isinstance(nltk.tokenize.regexp.WordPunctTokenizer(), TokenizerI),
        }
        kept = [n for n, o in others.items() if (getattr(nltk.tokenize.punkt, n, None)
                or getattr(nltk.tokenize, n)) is not o]
        batch = nltk.sent_tokenize(TEXTS)  # pyroid's extension works through nltk's name
        pyroid.unpatch("nltk")
        assert pi.patched() == ()
        restored = [f"{m}.{n}" for (m, n), o in originals.items() if getattr(mods[m], n) is not o]
        after = results(nltk)
        with warnings.catch_warnings(record=True) as again:
            warnings.simplefilter("always")
            pyroid.patch("nltk")  # once per process
            pyroid.unpatch()
        print(json.dumps({
            "warnings": [(w.category.__name__, str(w.message), w.filename) for w in caught],
            "again": [str(w.message) for w in again],
            "not_swapped": not_swapped, "same": got == want, "after": after == want,
            "restored_wrong": restored, "kinds": kinds, "kept_wrong": kept,
            "batch": batch == want["sent"], "originals_left": len(pn._delegate.ORIGINALS),
        }))
        """
    )
    assert out["not_swapped"] == []
    assert out["same"], "results differ while patched"
    assert out["after"], "results differ after unpatch"
    assert out["restored_wrong"] == []
    assert out["kept_wrong"] == []
    assert all(out["kinds"].values()), out["kinds"]
    assert out["batch"]
    assert out["originals_left"] == 0
    assert len(out["warnings"]) == 1, out["warnings"]
    category, message, filename = out["warnings"][0]
    assert category == "PatchWarning"
    assert "pyroid.unpatch" in message or "unpatch('nltk')" in message
    assert filename == "<string>"  # the caller's line
    assert out["again"] == []


@needs_nltk
def test_names_imported_before_the_patch_keep_working(punkt_tab: Any) -> None:
    out = child(
        """
        from nltk import sent_tokenize, word_tokenize
        from nltk.tokenize.punkt import PunktTokenizer, PunktSentenceTokenizer
        from nltk.tokenize.treebank import TreebankWordTokenizer
        import nltk.tokenize
        want = [PunktTokenizer("german").tokenize(t) for t in TEXTS]
        want_words = [word_tokenize(t, "german") for t in TEXTS]
        want_pst = PunktSentenceTokenizer().tokenize("A b. C d.")
        warnings.simplefilter("ignore")
        import pyroid
        pyroid.patch("nltk")
        res = {}
        # NLTK's own objects, held from before: still NLTK's, and still working.
        res["is_nltk"] = (PunktTokenizer.__module__, sent_tokenize.__module__)
        try:
            res["orig_punkt"] = [PunktTokenizer("german").tokenize(t) for t in TEXTS] == want
        except Exception as e:
            res["orig_punkt"] = f"{type(e).__name__}: {e}"
        try:
            res["orig_pst"] = PunktSentenceTokenizer().tokenize("A b. C d.") == want_pst
        except Exception as e:
            res["orig_pst"] = f"{type(e).__name__}: {e}"
        res["orig_sent"] = [sent_tokenize(t, "german") for t in TEXTS] == want
        res["orig_word"] = [word_tokenize(t, "german") for t in TEXTS] == want_words
        res["orig_tb"] = TreebankWordTokenizer().tokenize("Don't.") == ["Do", "n't", "."]
        pyroid.unpatch()
        # NLTK's cache of loaded tokenizers holds NLTK's own class again.
        cached = nltk.tokenize._get_punkt_tokenizer("german")
        res["cache"] = f"{type(cached).__module__}.{type(cached).__qualname__}"
        print(json.dumps(res))
        """
    )
    assert out["is_nltk"] == ["nltk.tokenize.punkt", "nltk.tokenize"]
    assert out["orig_punkt"] is True, out["orig_punkt"]
    assert out["orig_pst"] is True, out["orig_pst"]
    assert out["orig_sent"] is True
    assert out["orig_word"] is True
    assert out["orig_tb"] is True
    assert out["cache"] == "nltk.tokenize.punkt.PunktTokenizer", out["cache"]


@needs_nltk
def test_fallbacks_while_patched_call_nltk(punkt_tab: Any) -> None:
    out = child(
        """
        import nltk, nltk.tokenize.punkt as P, nltk.tokenize.treebank as T
        import nltk.tokenize.destructive as D
        import regex
        ORIG = {"pst": P.PunktSentenceTokenizer, "punkt": P.PunktTokenizer,
                "tb": T.TreebankWordTokenizer, "detok": T.TreebankWordDetokenizer,
                "nw": D.NLTKWordTokenizer}
        train = " ".join(TEXTS) * 5 + " The Corp. ltd. is big. Inc. is a word. Dr. Who."
        class Token(P.PunktToken):
            @property
            def first_upper(self):
                return self.tok[0].isupper() or self.tok[0].isdigit()
        class Vars(P.PunktLanguageVars):
            _re_word_start = r"[^\\(\\"\\`{\\[:;&\\#\\*@\\)}\\]\\-,!]"
        class Ends(P.PunktLanguageVars):
            sent_end_chars = (".", "?", "!", "^")
        extra = (regex.compile(r"!"), r" EXCLAIM ")
        text = "He left at 3 p.m. 4 people came. Wow! Mr. X. 5 more^ Ok."
        params = ORIG["punkt"]()._params
        def run(cls, tb, detok, nw):
            t = tb(); t.PUNCTUATION = [*ORIG["tb"].PUNCTUATION, extra]
            d = detok(); d.STARTING_QUOTES = [(regex.compile(r"``"), "<<")]
            n = nw(); n.ENDING_QUOTES = [*ORIG["nw"].ENDING_QUOTES, extra]
            return {
                "trained": cls(train).tokenize(text),
                "token_cls": cls(params, token_cls=Token).tokenize(text),
                "lang_vars": cls(params, lang_vars=Vars()).tokenize(text),
                "ends": cls(params, lang_vars=Ends()).tokenize(text),
                "tb": t.tokenize(text), "detok": d.detokenize(["``", "Hi", "''"]),
                "nw": n.tokenize(text),
            }
        want = run(ORIG["pst"], ORIG["tb"], ORIG["detok"], ORIG["nw"])
        import pyroid, pyroid.nltk as pn
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pyroid.patch("nltk")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = run(P.PunktSentenceTokenizer, T.TreebankWordTokenizer,
                      T.TreebankWordDetokenizer, D.NLTKWordTokenizer)
            trainer = pn.PunktTrainer(train)
        native_ends = P.PunktSentenceTokenizer(params, lang_vars=Ends())._impl() is not None
        print(json.dumps({
            "same": {k: got[k] == want[k] for k in want}, "got": got, "want": want,
            "fallback": sorted({str(w.message) for w in caught}),
            "categories": sorted({w.category.__name__ for w in caught}),
            "trainer": type(trainer).__module__, "native_ends": native_ends,
        }))
        """
    )
    assert all(out["same"].values()), (out["got"], out["want"])
    assert out["categories"] == ["FallbackWarning"]
    assert out["trainer"] == "nltk.tokenize.punkt"
    assert out["native_ends"], "a subclass of NLTK's PunktLanguageVars runs natively while patched"


@needs_nltk
def test_patching_context_and_hook(punkt_tab: Any) -> None:
    out = child(
        """
        import nltk
        orig = nltk.word_tokenize
        import pyroid.integrations as pi, pyroid.nltk as pn
        warnings.simplefilter("ignore")
        with pi.patching("nltk"):
            inside = nltk.word_tokenize is pn.word_tokenize
        print(json.dumps({"inside": inside, "after": nltk.word_tokenize is orig}))
        """
    )
    assert out == {"inside": True, "after": True}
    hooked = child(
        """
        import nltk, pyroid.nltk as pn
        print(json.dumps({
            "word": nltk.word_tokenize is pn.word_tokenize,
            "punkt": nltk.tokenize.punkt.PunktTokenizer is pn.PunktTokenizer,
            "result": nltk.word_tokenize("Don't stop."),
        }))
        """,
        env={"PYROID_PATCH": "nltk", "PYTHONPATH": _patch.HOOK_DIR, "PYTHONWARNINGS": "ignore"},
        isolated=False,
    )
    assert hooked == {"word": True, "punkt": True, "result": ["Do", "n't", "stop", "."]}


def test_patch_without_nltk(punkt_tab: Any) -> None:
    out = child(
        """
        import pyroid, pyroid.nltk as pn
        try:
            import nltk
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError("nltk is not hidden")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pyroid.patch("nltk")
        import nltk, nltk.tokenize, nltk.tokenize.punkt, nltk.tokenize.treebank
        import nltk.tokenize.destructive
        from nltk.tokenize import word_tokenize, PunktTokenizer
        from nltk.tokenize.treebank import TreebankWordDetokenizer
        mods = {m: sys.modules[m] for m in SWAPS}
        missing = [f"{m}.{n}" for m in SWAPS for n in SWAPS[m]
                   if getattr(mods[m], n, None) is not getattr(pn, n)]
        res = {
            "shim": all(getattr(m, "__pyroid__", False) for m in mods.values()),
            "missing": missing,
            "word": word_tokenize("Don't stop. Ok."),
            "sent": nltk.sent_tokenize("Mr. Smith went. He did."),
            "punkt": PunktTokenizer("english").tokenize("A b. C d."),
            "detok": TreebankWordDetokenizer().detokenize(["I", "ca", "n't", "."]),
            "warnings": [str(w.message) for w in caught],
        }
        try:
            nltk.download("punkt_tab")
        except AttributeError as e:
            res["download"] = str(e)
        try:
            pn.PunktTrainer("A b. C d.")
        except Exception as e:
            res["trainer"] = type(e).__name__
        tok = nltk.tokenize.treebank.TreebankWordTokenizer()
        tok.PUNCTUATION = []
        try:
            tok.tokenize("a, b")
        except Exception as e:
            res["changed"] = type(e).__name__
        pyroid.unpatch()
        try:
            import nltk.tokenize
        except ModuleNotFoundError:
            res["gone"] = True
        print(json.dumps(res))
        """,
        hide=("nltk",),
    )
    assert out["shim"]
    assert out["missing"] == []
    assert out["word"] == ["Do", "n't", "stop", ".", "Ok", "."]
    assert out["sent"] == ["Mr. Smith went.", "He did."]
    assert out["punkt"] == ["A b.", "C d."]
    assert out["detok"] == "I can't."
    assert len(out["warnings"]) == 1
    assert "unavailable" in out["warnings"][0]
    assert "has no attribute 'download'" in out["download"]
    assert out["trainer"] == "PyroidError", out["trainer"]
    assert out["changed"] == "PyroidError", out["changed"]
    assert out.get("gone") is True
