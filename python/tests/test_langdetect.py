"""pyroid.langdetect: langdetect's API, the batch functions, threads and Ctrl-C."""

from __future__ import annotations

import os
import pickle
import sys
import threading
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from _langdetect_samples import SAMPLES

import pyroid.langdetect as pl
from pyroid._pyroid import _testing
from pyroid.langdetect import detector_factory


def _langs(results: list[Any]) -> list[tuple[str, str]]:
    return [(x.lang, x.prob.hex()) for x in results]


@pytest.fixture
def seeded() -> Iterator[None]:
    saved = pl.DetectorFactory.seed
    pl.DetectorFactory.seed = 0
    try:
        yield
    finally:
        pl.DetectorFactory.seed = saved


def _factory(seed: Any = 0) -> pl.DetectorFactory:
    pl.init_factory()
    factory = detector_factory._factory
    assert factory is not None
    copy = pl.DetectorFactory()
    copy._profiles = factory._profiles  # share the loaded table
    copy.langlist = factory.langlist
    copy.set_seed(seed)
    return copy


# ---------------------------------------------------------------------------
# The API
# ---------------------------------------------------------------------------


@pytest.mark.thread_unsafe
def test_detect_and_detect_langs(seeded: None) -> None:
    assert pl.detect(SAMPLES["en"]) == "en"
    assert pl.detect(SAMPLES["ja"]) == "ja"
    langs = pl.detect_langs(SAMPLES["de"])
    assert langs[0].lang == "de"
    assert langs == sorted(langs, reverse=True)
    assert repr(langs[0]) == f"de:{langs[0].prob}"


def test_no_features_raises() -> None:
    for text in ("", "   ", "1234 !!", "http://example.com/abc"):
        with pytest.raises(pl.LangDetectException, match="No features in text") as info:
            pl.detect(text)
        assert info.value.code == pl.ErrorCode.CantDetectError == 5
        assert info.value.get_code() == 5


def test_error_codes() -> None:
    assert {k: v for k, v in vars(pl.ErrorCode).items() if not k.startswith("_")} == {
        "NoTextError": 0,
        "FormatError": 1,
        "FileLoadError": 2,
        "DuplicateLangError": 3,
        "NeedLoadProfileError": 4,
        "CantDetectError": 5,
        "CantOpenTrainData": 6,
        "TrainDataFormatError": 7,
        "InitParamError": 8,
    }


def test_language() -> None:
    a, b = pl.Language("a", 0.1), pl.Language("b", 0.5)
    assert a < b and not b < a
    assert str(pl.Language(None, 0)) == ""
    assert str(pl.Language("en", 1.0)) == "en:1.0"
    assert a != pl.Language("a", 0.1)  # no __eq__, as in langdetect
    # Our own object, round-tripped in memory.
    assert pickle.loads(pickle.dumps(a)).prob == 0.1


def test_default_profiles() -> None:
    pl.init_factory()
    factory = detector_factory._factory
    assert factory is not None
    langs = factory.get_lang_list()
    assert len(langs) == 55 and sorted(langs) == sorted(os.listdir(pl.PROFILES_DIRECTORY))
    m = factory.word_lang_prob_map
    assert len(m) == 87598
    assert "the" in m and "\ud800" not in m and 5 not in m
    assert len(m["the"]) == 55 and m["the"][langs.index("en")] > 0
    with pytest.raises(KeyError):
        m["no such n-gram"]


def test_factory_api() -> None:
    f = pl.DetectorFactory()
    with pytest.raises(pl.LangDetectException, match="Need to load profiles") as info:
        f.create()
    assert info.value.code == pl.ErrorCode.NeedLoadProfileError
    f.load_profile(pl.PROFILES_DIRECTORY)
    assert len(f.get_lang_list()) == 55
    with pytest.raises(pl.LangDetectException, match="Profile format error") as info:
        f.load_profile(pl.PROFILES_DIRECTORY)  # duplicates, wrapped as in langdetect
    assert info.value.code == pl.ErrorCode.FormatError
    f.clear()
    assert f.get_lang_list() == [] and len(f.word_lang_prob_map) == 0
    with pytest.raises(pl.LangDetectException, match="Need more than 2 profiles"):
        f.load_json_profile(['{"name": "x", "freq": {}, "n_words": [1, 1, 1]}'])
    with pytest.raises(pl.LangDetectException, match="Profile format error.") as info:
        f.load_json_profile(["{", "}"])
    assert info.value.code == pl.ErrorCode.FormatError


def test_load_profile_errors(tmp_path: Any) -> None:
    f = pl.DetectorFactory()
    with pytest.raises(pl.LangDetectException, match="Not found profile") as info:
        f.load_profile(str(tmp_path))
    assert info.value.code == pl.ErrorCode.NeedLoadProfileError
    (tmp_path / "bad").write_text("not json", encoding="utf-8")
    with pytest.raises(pl.LangDetectException, match="Profile format error in") as info:
        f.load_profile(str(tmp_path))
    assert isinstance(info.value.__context__, ValueError)


def test_detector_attributes() -> None:
    d = _factory(3).create(alpha=0.7)
    assert (d.alpha, d.n_trial, d.max_text_length, d.prior_map, d.verbose, d.seed) == (
        0.7,
        7,
        10000,
        None,
        False,
        3,
    )
    d.append("Hello   world http://example.com mail@example.org ")
    assert d.text == "Hello world "
    assert d.detect() in d.langlist
    assert d.langprob is not None and len(d.langprob) == 55
    import random

    assert isinstance(d.random, random.Random)


def test_custom_profiles_detect() -> None:
    f = pl.DetectorFactory()
    f.load_json_profile(
        [
            '{"freq":{"A":3,"B":6,"C":3,"AB":2,"BC":1,"ABC":2,"BBC":1,"CBA":1},"n_words":[12,3,4],"name":"lang1"}',
            '{"freq":{"A":6,"B":3,"C":3,"AA":3,"AB":2,"ABC":1,"ABA":1,"CAA":1},"n_words":[12,5,3],"name":"lang2"}',
        ]
    )
    f.set_seed(0)
    d = f.create()
    d.append("AAA")
    assert d.detect() == "lang2"


def test_ngram_and_profile() -> None:
    from pyroid.langdetect.utils.ngram import NGram

    assert NGram.normalize("1") == " " and NGram.normalize("ア") == "ア"
    assert NGram.normalize_vi("à") == "à"
    with pytest.raises(TypeError):
        NGram.normalize("ab")
    g = NGram()
    for ch in "Ab":
        g.add_char(ch)
    assert (g.get(1), g.get(2), g.get(3)) == ("b", "Ab", " Ab")
    p = pl.LangProfile("xx")
    p.update("abc abc")
    assert p.n_words == [6, 7, 5]


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------

_TEXTS = [SAMPLES[k] for k in sorted(SAMPLES)] * 5


@pytest.mark.thread_unsafe
def test_batch_equals_calls(seeded: None) -> None:
    expected = [pl.detect(t) for t in _TEXTS]
    assert pl.detect_batch(_TEXTS) == expected
    assert pl.detect_batch(tuple(_TEXTS)) == tuple(expected)
    langs = pl.detect_langs_batch(_TEXTS)
    assert [_langs(r) for r in langs] == [_langs(pl.detect_langs(t)) for t in _TEXTS]
    for threads in (1, 2, 8):
        assert pl.detect_batch(_TEXTS, threads=threads) == expected


@pytest.mark.thread_unsafe
def test_batch_errors(seeded: None) -> None:
    texts = ["Das ist ein Satz.", "", "Ceci est une phrase."]
    with pytest.raises(pl.LangDetectException) as info:
        pl.detect_batch(texts)
    assert info.value.__notes__ == ["pyroid.langdetect: raised for the text at index 1"]
    assert pl.detect_batch(texts, errors="none") == ["de", None, "fr"]
    langs = pl.detect_langs_batch(texts, errors="none")
    assert langs[1] is None and langs[0][0].lang == "de"
    with pytest.raises(TypeError, match="expected string or bytes-like object") as tinfo:
        pl.detect_batch(["ok text", 5])
    assert "index 1" in tinfo.value.__notes__[0]
    with pytest.raises(TypeError):
        pl.detect_batch("a str")
    with pytest.raises(ValueError, match="errors"):
        pl.detect_batch(texts, errors="ignore")


@pytest.mark.thread_unsafe
def test_batch_arrow(seeded: None) -> None:
    texts = [SAMPLES["en"], None, SAMPLES["ru"], "", SAMPLES["ko"]]
    col = _testing.column(texts)
    out = pl.detect_batch(col, errors="none")
    assert out.to_pylist() == ["en", None, "ru", None, "ko"]
    langs = pl.detect_langs_batch(_testing.column(texts, large=True, chunks=2), errors="none")
    assert langs[1] is None and langs[3] is None and langs[2][0].lang == "ru"
    with pytest.raises(pl.LangDetectException) as info:
        pl.detect_batch(col)
    assert "index 3" in info.value.__notes__[0]


@pytest.mark.thread_unsafe
def test_batch_pyarrow(seeded: None, pa: Any) -> None:
    arr = pa.chunked_array([[SAMPLES["fr"], None], [SAMPLES["th"]]], type=pa.large_string())
    assert pa.array(pl.detect_batch(arr)).to_pylist() == ["fr", None, "th"]


@pytest.mark.thread_unsafe
def test_batch_without_a_seed_is_random_but_valid() -> None:
    saved = pl.DetectorFactory.seed
    pl.DetectorFactory.seed = None
    try:
        out = pl.detect_langs_batch([SAMPLES["no"]] * 200)
    finally:
        pl.DetectorFactory.seed = saved
    probs = {r[0].prob for r in out}
    assert len(probs) > 1  # independent generators
    assert all(r[0].lang in ("no", "da") for r in out)


def test_batch_large(seeded: None) -> None:
    texts = [SAMPLES[k] for k in ("en", "es", "zh-cn", "ar")] * 5000
    out = pl.detect_batch(texts)
    assert out[:4] == ["en", "es", "zh-cn", "ar"] and out[-4:] == out[:4]


# ---------------------------------------------------------------------------
# Threads, free-threading, Ctrl-C
# ---------------------------------------------------------------------------


def test_threads_give_the_same_results() -> None:
    f = _factory(11)

    def one(text: str) -> list[tuple[str, str]]:
        d = f.create()
        d.append(text)
        return _langs(d.get_probabilities())

    texts = list(SAMPLES.values())
    expected = [one(t) for t in texts]
    results: list[Any] = [None] * 8
    barrier = threading.Barrier(8)

    def work(k: int) -> None:
        barrier.wait()
        results[k] = [one(t) for t in texts]

    threads = [threading.Thread(target=work, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(r == expected for r in results)


def test_profiles_change_while_detecting() -> None:
    # add_profile from another thread never changes a detection half-way.
    f = pl.DetectorFactory()
    f.load_profile(pl.PROFILES_DIRECTORY)
    f.set_seed(0)
    stop = threading.Event()
    errors: list[BaseException] = []

    def mutate() -> None:
        i = 0
        while not stop.is_set():
            try:
                f.add_profile(pl.LangProfile(f"x{i}", {"zz": 1}, [1, 1, 1]), 0, 55)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)
            i += 1

    t = threading.Thread(target=mutate)
    t.start()
    try:
        for _ in range(50):
            d = f.create()
            d.append(SAMPLES["en"])
            try:
                d.detect()
            except IndexError:
                pass  # the language list grew past the rows' length: langdetect raises too
    finally:
        stop.set()
        t.join()
    assert not errors


@pytest.mark.thread_unsafe
def test_ctrl_c_stops_a_batch(run_python: Callable[..., Any]) -> None:
    code = """
        import _thread, threading, time
        import pyroid.langdetect as pl
        pl.init_factory()
        texts = ["Le renard brun rapide saute par-dessus le chien paresseux."] * 400_000
        threading.Timer(0.3, _thread.interrupt_main).start()
        t = time.perf_counter()
        try:
            pl.detect_batch(texts, threads=2)
        except KeyboardInterrupt:
            print("interrupted", round(time.perf_counter() - t, 2))
    """
    out = run_python(code, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("interrupted"), out.stdout + out.stderr
    assert float(out.stdout.split()[1]) < 5


@pytest.mark.thread_unsafe
def test_without_langdetect(run_python: Callable[..., Any]) -> None:
    code = """
        import sys
        sys.modules["langdetect"] = None
        import pyroid.langdetect as pl
        assert pl.LangDetectException.__module__ == "pyroid.langdetect.lang_detect_exception"
        assert pl.Language.__module__ == "pyroid.langdetect.language"
        pl.DetectorFactory.seed = 0
        print(pl.detect("Das ist ein kleiner deutscher Satz."), len(pl.detect_langs("Hola, ¿cómo estás?")))
        try:
            pl.detect("")
        except pl.LangDetectException as e:
            print(e.code)
    """
    out = run_python(code)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["de", "1", "5"]


def test_interop_with_langdetect_classes() -> None:
    ld = pytest.importorskip("langdetect")
    from langdetect.language import Language

    assert pl.LangDetectException is ld.LangDetectException
    assert pl.Language is Language
    with pytest.raises(ld.LangDetectException):
        pl.detect("")
    assert isinstance(pl.detect_langs(SAMPLES["pl"])[0], Language)
    f = pl.DetectorFactory()
    f.add_profile(ld.utils.lang_profile.LangProfile("aa", {"a": 1}, [1, 1, 1]), 0, 2)
    f.add_profile(ld.utils.lang_profile.LangProfile("bb", {"b": 1}, [1, 1, 1]), 1, 2)
    d = f.create()
    d.append("b")
    assert d.detect() == "bb"


def test_default_order_follows_langdetect() -> None:
    ld = pytest.importorskip("langdetect")
    pl.init_factory()
    ld.detector_factory.init_factory()
    ours, theirs = detector_factory._factory, ld.detector_factory._factory
    assert ours is not None
    assert ours.get_lang_list() == theirs.get_lang_list()
    assert sys.modules["pyroid.langdetect"] is pl
