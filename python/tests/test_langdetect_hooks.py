"""pyroid.langdetect where langdetect's classes are customised, pickled or patched.

Each check runs the same steps on langdetect 1.0.9 and on pyroid (both loaded from langdetect's
own profile directory, so the language order is the same) and compares the results, with
probabilities as ``float.hex`` and their type. The cases are the outside evaluation's (P61)
reproducers: patterns and methods overridden, NGram's tables changed, numpy counts, a seed
error, pickles and copies, the patch's pickles and shims.
"""

from __future__ import annotations

import copy
import functools
import importlib
import json
import os
import pickle
import random
import re
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest

ld = pytest.importorskip("langdetect")
import langdetect.detector as ld_detector  # noqa: E402
import langdetect.utils.ngram as ld_ngram  # noqa: E402

import pyroid.langdetect as pl  # noqa: E402
import pyroid.langdetect.detector as pl_detector  # noqa: E402
import pyroid.langdetect.utils.ngram as pl_ngram  # noqa: E402

pytestmark = pytest.mark.thread_unsafe  # the cases change class attributes of both libraries

PROFILES = ld.PROFILES_DIRECTORY
# (module, Detector, NGram) of each library.
LIBS = {
    "langdetect": (ld, ld_detector.Detector, ld_ngram.NGram),
    "pyroid": (pl, pl_detector.Detector, pl_ngram.NGram),
}


def _langs(results: list[Any]) -> list[tuple[Any, str, str]]:
    return [(x.lang, float(x.prob).hex(), type(x.prob).__name__) for x in results]


def _call(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as exc:  # the same exception from both is part of the result
        return ("raised", type(exc).__name__, str(exc))


def _same(case: Callable[..., Any]) -> Any:
    """`case(module, Detector, NGram)` on both libraries; the results must be equal."""
    got = {name: _call(functools.partial(case, *lib)) for name, lib in LIBS.items()}
    assert got["pyroid"] == got["langdetect"]
    return got["pyroid"]


@pytest.fixture(autouse=True)
def _restore() -> Iterator[None]:
    """Put back every class attribute and seed a case changes."""
    saved = [(cls, dict(vars(cls))) for _, d, n in LIBS.values() for cls in (d, n)]
    seeds = [(m.DetectorFactory, m.DetectorFactory.seed) for m, _, _ in LIBS.values()]
    try:
        yield
    finally:
        for cls, attrs in saved:
            for name in set(vars(cls)) - set(attrs):
                delattr(cls, name)
            for name, value in attrs.items():
                if vars(cls).get(name) is not value and name not in ("__dict__", "__weakref__"):
                    setattr(cls, name, value)
        for cls, seed in seeds:
            cls.seed = seed


def _factory(module: Any, seed: Any = 0) -> Any:
    f = module.DetectorFactory()
    f.load_profile(PROFILES)
    f.seed = seed
    return f


_FACTORIES: dict[str, Any] = {}


def _loaded(module: Any) -> Any:
    """One factory per library with the default profiles (seed 0), shared by the cases."""
    if module.__name__ not in _FACTORIES:
        _FACTORIES[module.__name__] = _factory(module)
    return _FACTORIES[module.__name__]


def _probs(module: Any, text: str, cls: Any = None, **attrs: Any) -> Any:
    f = _loaded(module)
    d = (cls or module.detector_factory.Detector)(f)
    for name, value in attrs.items():
        setattr(d, name, value)
    d.append(text)
    return _langs(d.get_probabilities()), d.text


# ---------------------------------------------------------------------------
# W1: URL_RE and MAIL_RE
# ---------------------------------------------------------------------------

WWW = "Lies das: www.spiegel.de/politik/deutschland www.zeit.de/wirtschaft www.faz.net/aktuell ok?"
MENTIONS = "@maria_garcia @jose_luis @ana_lopez @carlos_ruiz thanks a lot"
URL_WWW = re.compile(r"(?:https?://|www\.)[-_.?&~;+=/#0-9A-Za-z]{1,2076}")
MAIL_AT = re.compile(r"[-_.0-9A-Za-z]{0,64}@[-_.0-9A-Za-z]{1,255}")


@pytest.mark.parametrize(
    ("attr", "pattern", "text"), [("URL_RE", URL_WWW, WWW), ("MAIL_RE", MAIL_AT, MENTIONS)]
)
def test_patterns_set_on_the_class(attr: str, pattern: re.Pattern[str], text: str) -> None:
    def case(m: Any, d: Any, n: Any) -> Any:
        m.DetectorFactory.seed = 0
        stock = (_langs(m.detect_langs(text)), _probs(m, text))
        setattr(d, attr, pattern)
        return stock, _langs(m.detect_langs(text)), m.detect(text), _probs(m, text)

    stock, custom, _, _ = _same(case)
    assert stock[0] != custom  # the pattern changes the result


def test_patterns_set_on_an_instance() -> None:
    def case(m: Any, d: Any, n: Any) -> Any:
        return (
            _probs(m, WWW, URL_RE=URL_WWW),
            _probs(m, MENTIONS, MAIL_RE=MAIL_AT),
            _probs(m, "hello 123 world 456", URL_RE=re.compile(r"[a-z]+")),
        )

    _same(case)


def test_equal_patterns_keep_the_native_path() -> None:
    def case(m: Any, d: Any, n: Any) -> Any:
        d.URL_RE = re.compile(d.URL_RE.pattern)
        return _probs(m, "Ceci est une phrase en français, voir https://example.org/x?a=1 merci")

    _same(case)


# ---------------------------------------------------------------------------
# W2: methods overridden or replaced
# ---------------------------------------------------------------------------


def test_cleaning_text_overridden_in_a_subclass() -> None:
    def case(m: Any, d: Any, n: Any) -> Any:
        class Lower(d):  # type: ignore[misc]
            def cleaning_text(self: Any) -> None:
                self.text = self.text.lower()

        class Keep(d):  # type: ignore[misc]
            def cleaning_text(self: Any) -> None:
                pass

        return [
            _probs(m, t, cls=c)
            for c in (Lower, Keep)
            for t in ("Otec matka syn.", "Привет мир hello world ok fine", "日本語 abc def ghi jkl")
        ]

    _same(case)


def test_cleaning_text_replaced_on_the_class_and_instance() -> None:
    text = "Привет мир hello world ok fine"

    def case(m: Any, d: Any, n: Any) -> Any:
        original = d.cleaning_text
        d.cleaning_text = lambda self: None
        m.DetectorFactory.seed = 0
        on_class = (_probs(m, text), _langs(m.detect_langs(text)), m.detect(text))
        d.cleaning_text = original
        det = _loaded(m).create()
        det.cleaning_text = lambda: None
        det.append(text)
        return on_class, _langs(det.get_probabilities())

    _same(case)


@pytest.mark.parametrize(
    "method",
    [
        "_extract_ngrams",
        "_init_probability",
        "_update_lang_prob",
        "_normalize_prob",
        "get_probabilities",
        "_sort_probability",
        "append",
        "detect",
    ],
)
def test_other_methods_overridden(method: str) -> None:
    def case(m: Any, d: Any, n: Any) -> Any:
        language = m.detector.Language
        bodies: dict[str, Any] = {
            "_extract_ngrams": lambda self: ["a", "b"],
            "_init_probability": lambda self: [1.0] * len(self.langlist),
            "_update_lang_prob": lambda self, prob, word, alpha: d._update_lang_prob(
                self, prob, word, alpha * 2
            ),
            "_normalize_prob": lambda self, prob: d._normalize_prob(self, prob) * 0.999999,
            "get_probabilities": lambda self: [language("xx", 1.0)],
            "_sort_probability": lambda self, prob: [language("yy", 0.5)],
            "append": lambda self, text: d.append(self, text.upper()),
            "detect": lambda self: "zz",
        }
        sub = type("Sub", (d,), {method: bodies[method]})
        out = [
            _call(functools.partial(_probs, m, t, cls=sub))
            for t in ("Bonjour le monde", "Das ist ein Haus.")
        ]
        setattr(d, method, bodies[method])
        m.DetectorFactory.seed = 0
        out.append(_call(lambda: _langs(m.detect_langs("Bonjour le monde"))))
        out.append(_call(lambda: m.detect("Das ist ein Haus.")))
        return out

    _same(case)


def test_custom_random() -> None:
    class MyRandom(random.Random):
        def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
            return 0.0

    def case(m: Any, d: Any, n: Any) -> Any:
        det = _loaded(m).create()
        det.random = MyRandom()
        det.append("Otec matka syn.")
        return _langs(det.get_probabilities())

    _same(case)


def test_word_lang_prob_map_assigned() -> None:
    def case(m: Any, d: Any, n: Any) -> Any:
        f = m.DetectorFactory()
        f.load_json_profile(
            [
                json.dumps(
                    {"name": "aa", "freq": {"a": 10, "b": 5, "ab": 4}, "n_words": [15, 4, 0]}
                ),
                json.dumps(
                    {"name": "bb", "freq": {"b": 10, "c": 5, "bc": 4}, "n_words": [15, 4, 0]}
                ),
            ]
        )
        g = m.DetectorFactory()
        g.langlist = list(f.langlist)
        g.word_lang_prob_map = {k: list(v) for k, v in f.word_lang_prob_map.items()}
        g.seed = 0
        det = g.create()
        det.append("ab bc ab")
        first = _langs(det.get_probabilities())
        det = g.create()
        det.word_lang_prob_map = {"c": [0.0, 1.0]}
        det.append("ab bc ab")
        return first, _langs(det.get_probabilities())

    _same(case)


# ---------------------------------------------------------------------------
# W5 and F: NGram
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "change",
    ["n_gram", "latin1", "cjk_dict", "vi", "normalize", "add_char"],
)
def test_ngram_changed(change: str) -> None:
    texts = (
        "Otec matka syn.",
        "Ça coûte 10 € « cher » ?",
        "中文是中国的语言\uff0c日本語も",
        "Tiếng Việt có dấu",
    )

    def case(m: Any, d: Any, n: Any) -> Any:
        m.DetectorFactory.seed = 0
        if change == "n_gram":
            n.N_GRAM = 2
        elif change == "latin1":
            n.LATIN1_EXCLUDED = "çô"
        elif change == "cjk_dict":
            n.CJK_MAP = {**n.CJK_MAP, "语": "一"}
        elif change == "vi":
            n.NORMALIZED_VI_CHARS = [s[::-1] for s in n.NORMALIZED_VI_CHARS]
        elif change == "normalize":
            original = n.normalize.__func__

            def lower(cls: Any, ch: str) -> Any:
                return original(cls, ch).lower()

            n.normalize = classmethod(lower)
        elif change == "add_char":
            original_add = n.add_char
            n.add_char = lambda self, ch: original_add(self, ch.upper() if ch == "a" else ch)

        def module_level(t: str) -> Any:
            return _langs(m.detect_langs(t))

        return [
            (_call(functools.partial(module_level, t)), _call(functools.partial(_probs, m, t)))
            for t in texts
        ]

    _same(case)


def test_cjk_map_edited_in_place() -> None:
    text = "中文是中国的语言"

    def case(m: Any, d: Any, n: Any) -> Any:
        m.DetectorFactory.seed = 0
        key = next(ch for ch in text if ch not in n.CJK_MAP)
        n.CJK_MAP[key] = "一"
        try:
            return _langs(m.detect_langs(text)), _probs(m, text)
        finally:
            del n.CJK_MAP[key]

    _same(case)


def test_ngram_tables() -> None:
    for name in (
        "LATIN1_EXCLUDED",
        "TO_NORMALIZE_VI_CHARS",
        "DMARK_CLASS",
        "NORMALIZED_VI_CHARS",
        "CJK_CLASS",
    ):
        assert getattr(pl_ngram.NGram, name) == getattr(ld_ngram.NGram, name), name
    assert pl_ngram.NGram.CJK_MAP == ld_ngram.NGram.CJK_MAP
    assert len(pl_ngram.NGram.CJK_MAP) == 6667
    assert list(pl_ngram.NGram.CJK_MAP) == list(ld_ngram.NGram.CJK_MAP)


# ---------------------------------------------------------------------------
# W4: profiles with numpy counts
# ---------------------------------------------------------------------------


def test_numpy_counts() -> None:
    np = pytest.importorskip("numpy")
    files = sorted(os.listdir(PROFILES))
    texts = (
        "War doesn't show who's right, just who's left.",
        "Otec matka syn.",
        "Das ist ein Haus",
    )

    def case(m: Any, d: Any, n: Any) -> Any:
        lp = importlib.import_module(m.__name__ + ".utils.lang_profile").LangProfile
        f = m.DetectorFactory()
        for i, name in enumerate(files):
            with open(os.path.join(PROFILES, name), encoding="utf-8") as fh:
                j = json.load(fh)
            freq = {k: np.int64(v) for k, v in j["freq"].items()}
            f.add_profile(lp(j["name"], freq, [np.int64(x) for x in j["n_words"]]), i, len(files))
        f.seed = 0
        out = []
        for t in texts:
            det = f.create()
            det.append(t)
            out.append(_langs(det.get_probabilities()))
        row = f.word_lang_prob_map["a"]
        return out, [type(x).__name__ for x in row[:3]], len(f.word_lang_prob_map)

    _same(case)


def test_numpy_counts_mixed_with_json_profiles() -> None:
    np = pytest.importorskip("numpy")

    def case(m: Any, d: Any, n: Any) -> Any:
        lp = importlib.import_module(m.__name__ + ".utils.lang_profile").LangProfile
        f = m.DetectorFactory()
        f.add_profile(lp("aa", {"a": 10, "ab": 4, "b": 1}, [15, 4, 0]), 0, 3)
        f.add_profile(lp("bb", {"b": np.int32(10), "bc": np.int32(4)}, [np.int32(15), 4, 0]), 1, 3)
        f.add_profile(lp("cc", {"c": 3.5, "a": 1}, [7, 1, 1]), 2, 3)
        f.seed = 0
        det = f.create()
        det.append("ab bc ca")
        return _langs(det.get_probabilities()), {
            k: [type(x).__name__ for x in v] for k, v in f.word_lang_prob_map.items()
        }

    _same(case)


# ---------------------------------------------------------------------------
# W6: a seed that random.seed() rejects
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", ["\ud800", (1, 2), [3]])
def test_second_call_after_a_seed_error(seed: Any) -> None:
    def case(m: Any, d: Any, n: Any) -> Any:
        f = _factory(m, seed)
        det = f.create()
        det.append("Hello world")
        first = _call(det.get_probabilities)
        featureless = f.create()
        featureless.append("123")
        m.DetectorFactory.seed = seed
        return (
            first,
            det.detect(),
            det.text,
            _call(featureless.detect),
            _call(lambda: m.detect("123")),
            _call(lambda: m.detect("Hello world")),
        )

    _same(case)


# ---------------------------------------------------------------------------
# C2: pickles and copies
# ---------------------------------------------------------------------------


def _round_trip(obj: Any) -> Any:
    return pickle.loads(pickle.dumps(obj))  # an object this test made


def test_factories_and_detectors_pickle_and_copy() -> None:
    def case(m: Any, d: Any, n: Any) -> Any:
        f = _factory(m)
        det = f.create()
        det.append("Ceci est une phrase en français.")
        out: list[Any] = []
        for make in (_round_trip, copy.deepcopy, copy.copy):
            g = make(f)
            e = g.create()
            e.append("Das ist ein Haus.")
            out.append((g.get_lang_list() == f.get_lang_list(), _langs(e.get_probabilities())))
            out.append(_langs(make(det).get_probabilities()))
        done = f.create()
        done.append("Hola, ¿cómo estás?")
        done.get_probabilities()
        again = pickle.loads(pickle.dumps(done))
        out.append((_langs(again.get_probabilities()), again.text))
        p = importlib.import_module(m.__name__ + ".utils.lang_profile").LangProfile("xx")
        p.update("hello world")
        out.append(dict(pickle.loads(pickle.dumps(p)).freq))
        return out

    _same(case)


def test_a_pickled_factory_keeps_its_custom_rows() -> None:
    f = pl.DetectorFactory()
    f.add_profile(pl.LangProfile("aa", {"a": 1, "abcd": 2, "": 3}, [1, 1, 1]), 0, 2)
    f.add_profile(pl.LangProfile("bb", {"b": 1, ("x",): 1}, [1, 1, 1]), 1, 3)
    g = pickle.loads(pickle.dumps(f))
    assert list(g.word_lang_prob_map) == list(f.word_lang_prob_map)
    assert [g.word_lang_prob_map[k] for k in g.word_lang_prob_map] == [
        f.word_lang_prob_map[k] for k in f.word_lang_prob_map
    ]


# ---------------------------------------------------------------------------
# S1: lone surrogates in long texts
# ---------------------------------------------------------------------------


def test_lone_surrogates_do_not_read_the_whole_text() -> None:
    pl.DetectorFactory.seed = 0
    doc = "Das ist ein langer deutscher Text über viele Dinge. " * 200_000  # 10 MB
    variants = {
        "plain": doc,
        "start": "\ud800" + doc,
        "end": doc + "\ud800",
        "ascii": "x" + doc.encode("ascii", "replace").decode(),
    }

    def best(text: str) -> float:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            pl.detect(text)
            times.append(time.perf_counter() - t0)
        return min(times)

    plain = best(variants["plain"])
    for name in ("start", "end"):
        # Reading 10 MB of code points takes tens of ms; the 10,000-character cut, well under one.
        assert best(variants[name]) < 5 * plain + 0.01, name
    got = pl.detect_batch([variants["end"], variants["start"], "\ud800" + "é" * 30_000 + doc])
    assert got == [
        pl.detect(t) for t in (variants["end"], variants["start"], "\ud800" + "é" * 30_000 + doc)
    ]


def test_long_texts_with_surrogates_match_langdetect() -> None:
    url = "https://example.org/" + "a" * 2000 + " "
    texts = [
        "\ud800" + "Ceci est une phrase en français. " * 600,
        "Ceci est une phrase en français. " * 600 + "\udfff",
        "é" + url * 30 + "Das ist ein Haus. " * 700 + "\ud83d",
        "\ud83d " * 2000 + "Hello world, this is English. " * 500,
    ]

    def case(m: Any, d: Any, n: Any) -> Any:
        m.DetectorFactory.seed = 0
        return [(_langs(m.detect_langs(t)), _probs(m, t)) for t in texts]

    _same(case)
    pl.DetectorFactory.seed = 0
    batch = [_langs(x) for x in pl.detect_langs_batch(texts)]
    assert batch == [_langs(pl.detect_langs(t)) for t in texts]
