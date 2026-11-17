"""pyroid.langdetect against langdetect 1.0.9: the same languages and probabilities, bit for bit.

Each check runs a langdetect ``DetectorFactory`` and a pyroid one, both loaded from
langdetect's own profile directory (so the language order is the same), on the same text
with the same seed, and compares ``(lang, prob.hex())`` lists, or the exception (type,
message, ``code``), plus ``Detector.text`` and ``Detector.langprob`` afterwards.
"""

from __future__ import annotations

import importlib
import random
from typing import Any

import pytest
from _langdetect_samples import SAMPLES
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

ld = pytest.importorskip("langdetect")

import pyroid.langdetect as pl  # noqa: E402
from pyroid._pyroid import langdetect as native  # noqa: E402


@pytest.fixture(scope="module")
def factories() -> tuple[Any, Any]:
    ours = pl.DetectorFactory()
    ours.load_profile(ld.PROFILES_DIRECTORY)
    theirs = ld.DetectorFactory()
    theirs.load_profile(ld.PROFILES_DIRECTORY)
    assert ours.get_lang_list() == theirs.get_lang_list()
    return ours, theirs


def _run(factory: Any, text: Any, seed: Any, **attrs: Any) -> Any:
    detector = factory.create()
    detector.seed = seed
    try:
        for name, value in attrs.items():
            if name == "prior":
                detector.set_prior_map(value)
            else:
                setattr(detector, name, value)
        detector.append(text)
        langs = [(lang.lang, lang.prob.hex()) for lang in detector.get_probabilities()]
        out: Any = ("ok", langs)
    except Exception as exc:
        out = ("err", type(exc).__name__, str(exc), getattr(exc, "code", None))
    langprob = detector.langprob
    return out, detector.text, None if langprob is None else [p.hex() for p in langprob]


def _same(factories: tuple[Any, Any], text: Any, seed: Any = 0, **attrs: Any) -> None:
    ours, theirs = factories
    assert _run(ours, text, seed, **attrs) == _run(theirs, text, seed, **attrs)


@pytest.mark.parametrize("name", sorted(SAMPLES))
@pytest.mark.parametrize("seed", [0, 1, 42, 2**40 + 7, -3])
def test_samples(factories: tuple[Any, Any], name: str, seed: int) -> None:
    _same(factories, SAMPLES[name], seed)


@pytest.mark.parametrize("seed", ["seed", b"bytes", bytearray(b"xyz"), 1.5, -0.0, True, 2**200])
def test_seed_types(factories: tuple[Any, Any], seed: Any) -> None:
    _same(factories, SAMPLES["de"], seed)


@pytest.mark.parametrize("seed", [[1], (1, 2), {"a": 1}])
def test_unsupported_seed_types_raise_the_same(factories: tuple[Any, Any], seed: Any) -> None:
    _same(factories, SAMPLES["fr"], seed)


@pytest.mark.parametrize(
    "text",
    ["", " ", "   \t\n", "1234", "http://example.com", "a@b.cd", "!!!", "　", "a", "A"],
)
def test_edge_texts(factories: tuple[Any, Any], text: str) -> None:
    _same(factories, text)


@pytest.mark.parametrize("text", [b"bytes", None, 12, ["a"]])
def test_non_str_raises_the_same(factories: tuple[Any, Any], text: Any) -> None:
    _same(factories, text)


def test_lone_surrogates(factories: tuple[Any, Any]) -> None:
    for text in ["\ud800", "abc\udfffdef ghi", "😀 test" + "\ud800" * 3, "é\udc80" * 50]:
        _same(factories, text)


@pytest.mark.parametrize("limit", [0, 1, 5, 17, 100, -1, 10**12, 3.0, 5.5, True])
def test_max_text_length(factories: tuple[Any, Any], limit: Any) -> None:
    _same(factories, SAMPLES["it"] * 3, max_text_length=limit)


def test_long_texts_are_cut_the_same(factories: tuple[Any, Any]) -> None:
    long = " ".join(SAMPLES[k] for k in ("en", "de", "ru", "ja")) * 200
    for limit in (10, 1000, 10000, 30000):
        _same(factories, long, max_text_length=limit)
    # Many URLs and mail addresses (each shrinks to one space) before the cut.
    noisy = " ".join(f"http://e{i}.com/x?y=1 m{i}@x{i}.org word{i}" for i in range(3000))
    _same(factories, noisy)
    _same(factories, "x" * 5000 + "@" + "y" * 600 + " " + SAMPLES["en"])


@pytest.mark.parametrize("alpha", [0.0, 0.1, 2.5, 10, -0.5, 1e-300])
def test_alpha(factories: tuple[Any, Any], alpha: Any) -> None:
    _same(factories, SAMPLES["pt"], alpha=alpha)


@pytest.mark.parametrize("n_trial", [0, 1, 3, 20])
def test_n_trial(factories: tuple[Any, Any], n_trial: int) -> None:
    _same(factories, SAMPLES["nl"], n_trial=n_trial)


@pytest.mark.parametrize(
    "prior",
    [{"en": 0.5, "de": 0.5}, {"fr": 1}, {"ja": 0.2, "zh-cn": 0.3, "ko": 0.5}, {"xx": 1.0}],
)
def test_prior_maps(factories: tuple[Any, Any], prior: dict[str, float]) -> None:
    _same(factories, SAMPLES["de"], prior=prior)
    _same(factories, SAMPLES["ja"], prior=prior)


def test_prior_map_errors(factories: tuple[Any, Any]) -> None:
    ours, theirs = factories
    for prior in ({"en": -1.0}, {}, {"en": 0.0}):
        a, b = ours.create(), theirs.create()
        for d in (a, b):
            with pytest.raises(ld.LangDetectException) as info:
                d.set_prior_map(prior)
            assert info.value.code == ld.lang_detect_exception.ErrorCode.InitParamError
        assert a.prior_map == b.prior_map


def test_verbose_prints_the_same(
    factories: tuple[Any, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    ours, theirs = factories
    out = []
    for factory in (ours, theirs):
        d = factory.create()
        d.seed = 3
        d.set_verbose()
        d.append(SAMPLES["sv"][:30])
        d.detect()
        out.append(capsys.readouterr().out)
    assert out[0] == out[1]
    assert out[0].count("==>") == 7


def test_appends_accumulate(factories: tuple[Any, Any]) -> None:
    ours, theirs = factories
    results = []
    for factory in (ours, theirs):
        d = factory.create()
        d.seed = 0
        d.append("hello  ")
        d.append("  world http://x.y ")
        d.append(SAMPLES["fi"])
        results.append((d.text, [(x.lang, x.prob.hex()) for x in d.get_probabilities()], d.text))
    assert results[0] == results[1]


def test_module_level_functions(factories: tuple[Any, Any]) -> None:
    saved = ld.DetectorFactory.seed, pl.DetectorFactory.seed
    try:
        for seed in (0, 5):
            ld.DetectorFactory.seed = pl.DetectorFactory.seed = seed
            for name in ("en", "ar", "zh-tw", "caps"):
                text = SAMPLES[name]
                assert pl.detect(text) == ld.detect(text)
                ours = [(x.lang, x.prob.hex()) for x in pl.detect_langs(text)]
                assert ours == [(x.lang, x.prob.hex()) for x in ld.detect_langs(text)]
    finally:
        ld.DetectorFactory.seed, pl.DetectorFactory.seed = saved


# ---------------------------------------------------------------------------
# Hypothesis: random text and seeds
# ---------------------------------------------------------------------------

_WORDS = [w for s in SAMPLES.values() for w in s.split()]
_any_char = st.one_of(
    st.characters(),
    st.characters(categories=["Cs"]),
    st.sampled_from("aAzZ @.-_:/hHtTpPsṢ̀́̃̉あ一가"),
)
_texts = st.one_of(
    st.text(_any_char, max_size=60),
    st.lists(st.sampled_from(_WORDS), max_size=25).map(" ".join),
    st.lists(
        st.one_of(
            st.sampled_from(_WORDS),
            st.from_regex(r"https?://[-_.?&~;+=/#0-9A-Za-z]{1,30}", fullmatch=True),
            st.from_regex(r"[-_.0-9A-Za-z]{1,8}@[-_.0-9A-Za-z]{1,8}", fullmatch=True),
            st.text(_any_char, max_size=6),
        ),
        max_size=20,
    ).map(" ".join),
)
_seeds = st.one_of(st.integers(-(2**70), 2**70), st.text(max_size=5), st.floats(allow_nan=False))


@given(text=_texts, seed=_seeds)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_random_texts(factories: tuple[Any, Any], text: str, seed: Any) -> None:
    _same(factories, text, seed)


@given(text=_texts, limit=st.integers(-2, 80))
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_random_texts_cut(factories: tuple[Any, Any], text: str, limit: int) -> None:
    _same(factories, text, 1, max_text_length=limit)


# ---------------------------------------------------------------------------
# Custom profiles
# ---------------------------------------------------------------------------

_keys = st.text(st.sampled_from("abcAB あ\ud800é"), min_size=0, max_size=4)
_profile = st.fixed_dictionaries(
    {
        "name": st.text("xyz", min_size=1, max_size=3),
        "freq": st.dictionaries(_keys, st.one_of(st.integers(0, 50), st.floats(0, 50)), max_size=8),
        "n_words": st.lists(st.integers(0, 60), min_size=2, max_size=4),
    }
)


def _load(factory: Any, how: str, profiles: list[dict[str, Any]], index: int, size: int) -> Any:
    import json

    try:
        if how == "json":
            factory.load_json_profile([json.dumps(p) for p in profiles])
        else:
            for i, p in enumerate(profiles):
                profile = (
                    pl.LangProfile
                    if factory.__module__.startswith("pyroid")
                    else ld.utils.lang_profile.LangProfile
                )(**p)
                factory.add_profile(profile, index + i, size)
        return "ok"
    except Exception as exc:
        return (type(exc).__name__, str(exc), getattr(exc, "code", None))


@given(
    profiles=st.lists(_profile, min_size=2, max_size=4),
    how=st.sampled_from(["json", "add"]),
    index=st.integers(-3, 3),
    size=st.integers(-1, 5),
    texts=st.lists(st.text(st.sampled_from("abcAB あ\ud800é"), max_size=12), max_size=3),
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_custom_profiles(
    profiles: list[dict[str, Any]], how: str, index: int, size: int, texts: list[str]
) -> None:
    importlib.import_module("langdetect.utils.lang_profile")  # ld.utils.lang_profile

    ours, theirs = pl.DetectorFactory(), ld.DetectorFactory()
    assert _load(ours, how, profiles, index, size) == _load(theirs, how, profiles, index, size)
    assert ours.get_lang_list() == theirs.get_lang_list()
    assert dict(ours.word_lang_prob_map) == theirs.word_lang_prob_map
    if not theirs.get_lang_list():
        return
    for text in texts:
        for seed in (0, 7):
            assert _run(ours, text, seed) == _run(theirs, text, seed)


# ---------------------------------------------------------------------------
# The random draws
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 12345, 2**32 - 1, 2**32, 2**64 + 3, -9, "a", b"b", 0.5])
def test_draws_match_random(seed: Any) -> None:
    _seed_arg = importlib.import_module("pyroid.langdetect.detector")._seed_arg

    ops = [("gauss", 3), ("random", 5)] + [
        ("below", n) for n in (1, 2, 3, 7, 1000, 2**31, 2**40, 2**63)
    ]
    ours = native._draws(_seed_arg(seed), ops * 30)
    r = random.Random(seed)
    theirs: list[Any] = []
    for op, n in ops * 30:
        if op == "gauss":
            theirs.extend(r.gauss(0.0, 1.0) for _ in range(n))
        elif op == "random":
            theirs.extend(r.random() for _ in range(n))
        else:
            theirs.append(getattr(r, "_randbelow")(n))  # noqa: B009
    assert [x.hex() if isinstance(x, float) else x for x in ours] == [
        x.hex() if isinstance(x, float) else x for x in theirs
    ]


@given(seed=st.integers(-(2**126), 2**126), n=st.integers(1, 2**64 - 1))
def test_draws_match_random_hypothesis(seed: int, n: int) -> None:
    ours = native._draws(seed, [("gauss", 4), ("below", n), ("random", 3)])
    r = random.Random(seed)
    below = getattr(r, "_randbelow")  # noqa: B009 (not in typeshed)
    theirs = [r.gauss(0.0, 1.0) for _ in range(4)] + [below(n)] + [r.random() for _ in range(3)]
    assert ours == theirs
