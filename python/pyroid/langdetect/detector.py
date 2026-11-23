# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""langdetect's ``Detector``: ``append`` and ``_detect_block`` run in Rust.

The attributes are langdetect's (``text``, ``langprob``, ``alpha``, ``n_trial``,
``max_text_length``, ``prior_map``, ``verbose``, ``seed``, ``langlist``,
``word_lang_prob_map``), and so are the results, to the last bit of every probability, for
the same seed. Whatever the Rust code cannot reproduce runs langdetect's own algorithm in
Python, with the same results: ``set_verbose()``, attribute values of unusual types, and
customised classes (``URL_RE``, a subclass's ``cleaning_text``, ``NGram.N_GRAM``, ...).
"""

from __future__ import annotations

import operator
import random as _random
import re
import sys
from collections.abc import Iterator, Mapping
from typing import Any

from pyroid._pyroid import langdetect as _native
from pyroid.langdetect import _compat
from pyroid.langdetect._compat import ErrorCode, LangDetectException, Language
from pyroid.langdetect.utils import ngram as _ngram
from pyroid.langdetect.utils.ngram import NGram

__all__ = ["Detector"]

_NUMBER = (int, float)
_MAX_INT_SEED = 1 << 127
# Ints the native side takes as they are (n_trial as an i64; the others as floats, exactly).
_MAX_INT_PARAM = 1 << 53
_Random = _random.Random


class _SeedCache:
    """Seeded states for seeds other than ints, by type and value (a few at most)."""

    def __init__(self) -> None:
        self._states: dict[tuple[type, Any], Any] = {}

    def get(self, seed: Any) -> Any:
        key = (type(seed), seed)
        try:
            return self._states[key]
        except (KeyError, TypeError):
            pass
        state = _native._SeedState(_Random(seed).getstate())
        try:
            if len(self._states) >= 16:
                self._states.clear()
            self._states[key] = state
        except TypeError:  # unhashable (bytearray): not cached
            pass
        return state


_SEEDS = _SeedCache()


class _Unsupported(Exception):  # noqa: N818
    """This call takes langdetect's Python algorithm."""


def _seed_arg(seed: Any) -> Any:
    """The native form of a detector's seed (raises _Unsupported for other types)."""
    if seed is None:
        return None
    if type(seed) is int or type(seed) is bool:
        if -_MAX_INT_SEED < seed < _MAX_INT_SEED:
            return int(seed)
        return _SEEDS.get(seed)
    if type(seed) in (str, bytes, bytearray, float):
        try:
            return _SEEDS.get(seed)
        except Exception:  # random.seed() raises (a lone surrogate): after the n-grams
            raise _Unsupported from None
    raise _Unsupported


def _params(alpha: Any, n_trial: Any, owner: Any) -> tuple[Any, ...]:
    """``_detect_block``'s numbers, if they are all ints and floats (else _Unsupported)."""
    values = (
        alpha,
        n_trial,
        owner.ALPHA_WIDTH,
        owner.ITERATION_LIMIT,
        owner.CONV_THRESHOLD,
        owner.BASE_FREQ,
    )
    # langdetect's own types first (the common case), then any int or float.
    if (
        type(values[0]) is float
        and type(n_trial) is int
        and type(values[2]) is float
        and type(values[3]) is int
        and type(values[4]) is float
        and type(values[5]) is int
        and -_MAX_INT_PARAM < values[3] < _MAX_INT_PARAM
        and -_MAX_INT_PARAM < values[5] < _MAX_INT_PARAM
    ):
        return values
    if type(n_trial) is not int or not all(type(v) in _NUMBER for v in values):
        raise _Unsupported
    if not all(-_MAX_INT_PARAM < v < _MAX_INT_PARAM for v in values if type(v) is int):
        raise _Unsupported
    return values


def _zero_division() -> Any:
    """Raise the interpreter's own ZeroDivisionError for a float division by zero."""
    zero = 0.0
    return 1.0 / zero


class _WordMap(Mapping[Any, list[float]]):
    """``word_lang_prob_map``: n-gram -> one probability per language (read-only view)."""

    __slots__ = ("_profiles",)

    def __init__(self, profiles: Any) -> None:
        self._profiles = profiles

    def __getitem__(self, word: Any) -> list[float]:
        return self._profiles._row(word)  # type: ignore[no-any-return]

    def __contains__(self, word: object) -> bool:
        return bool(self._profiles._contains(word))

    def __len__(self) -> int:
        return len(self._profiles)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._profiles._keys())

    def __repr__(self) -> str:
        return f"<word_lang_prob_map: {len(self)} n-grams>"


# langdetect 1.0.9's patterns, which the Rust `append` matches.
_URL_RE = re.compile(r"https?://[-_.?&~;+=/#0-9A-Za-z]{1,2076}")
_MAIL_RE = re.compile(r"[-_.0-9A-Za-z]{1,64}@[-_0-9A-Za-z]{1,255}[-_.0-9A-Za-z]{1,255}")


def _pattern_stock(p: Any, stock: re.Pattern[str]) -> bool:
    """`p` is `stock`, or a compiled `re` pattern equal to it."""
    return p is stock or (
        type(p) is re.Pattern and p.pattern == stock.pattern and p.flags == stock.flags
    )


def _append_stock(self: Any) -> bool:
    """``append`` can run in Rust: the patterns and ``NGram.normalize_vi`` are langdetect's."""
    try:
        url, mail = self.URL_RE, self.MAIL_RE
        return (
            (url is _URL_RE or _pattern_stock(url, _URL_RE))
            and (mail is _MAIL_RE or _pattern_stock(mail, _MAIL_RE))
            and _APPEND_CHECKS.ok()
        )
    except Exception:  # a missing attribute: langdetect's code raises it
        return False


def _append_native(self: Any, text: Any) -> bool:
    """``append`` in Rust, for langdetect's patterns (checked by the caller).

    Returns False, having done nothing, unless `text` is a str and ``max_text_length`` an int.
    """
    limit = self.max_text_length
    if type(limit) is not int or not isinstance(text, str):
        return False
    added = _native.append_text(text, 0 if limit < 0 else min(limit, sys.maxsize))
    if added:
        self.text += added
    return True


def _append(self: Any, text: Any) -> None:
    """``Detector.append``, on any object with langdetect's ``text`` and ``max_text_length``."""
    if _append_stock(self) and _append_native(self, text):
        return
    # langdetect's code, on this detector's patterns and NGram.
    text = self.URL_RE.sub(" ", text)
    text = self.MAIL_RE.sub(" ", text)
    text = NGram.normalize_vi(text)
    n = operator.index(min(len(text), self.max_text_length))
    if type(text) is not str:
        pre: Any = 0
        for i in range(n):
            ch = text[i]
            if ch != " " or pre != " ":
                self.text += ch
            pre = ch
        return
    added = _native.collapse(text, max(n, 0))
    if added:
        self.text += added


# The attributes of a detector that langdetect's `_detect_block` reads through `self` and the
# Rust code stands for (methods; `random`).
_BLOCK_METHODS = ("_extract_ngrams", "_init_probability", "_update_lang_prob", "_normalize_prob")


def _block_mode(self: Any) -> int:
    """How ``_detect_block`` runs (see `_mode`); an attribute that is missing gives 2."""
    try:
        if type(self) is _DETECTOR and _BLOCK_CHECKS.ok():  # the class-level checks, natively
            d = self.__dict__
            word_map = d["word_lang_prob_map"]
            random = d.get("_random")
            if (
                type(word_map) is _WordMap
                and word_map._profiles is d["_profiles"]
                and (random is None or type(random) is _Random)
                and "_extract_ngrams" not in d
                and "_init_probability" not in d
                and "_update_lang_prob" not in d
                and "_normalize_prob" not in d
            ):
                return 0 if _DETECTOR.cleaning_text is _CLEAN and "cleaning_text" not in d else 1
        return _mode(self)
    except Exception:
        return 2  # langdetect's code raises it where langdetect does


def _mode(self: Any) -> int:
    """How ``_detect_block`` runs.

    0: in Rust; 1: in Rust after the detector's own ``cleaning_text()``; 2: langdetect's
    Python algorithm (customised methods, ``random``, ``word_lang_prob_map`` or ``NGram``).
    """
    cls = type(self)
    d = self.__dict__
    word_map = d.get("word_lang_prob_map")
    if (
        cls._extract_ngrams is not _EXTRACT
        or cls._init_probability is not _INIT_PROB
        or cls._update_lang_prob is not _UPDATE
        or cls._normalize_prob is not _NORMALIZE_PROB
        or cls.random is not _RANDOM_PROPERTY
        or type(word_map) is not _WordMap
        or word_map._profiles is not d.get("_profiles")
        or NGram is not _NGRAM
        or not _ngram._stock()
    ):
        return 2
    random = d.get("_random")
    if (
        (random is not None and type(random) is not _Random)
        or "_extract_ngrams" in d
        or "_init_probability" in d
        or "_update_lang_prob" in d
        or "_normalize_prob" in d
    ):
        return 2
    return 0 if cls.cleaning_text is _CLEAN and "cleaning_text" not in d else 1


def _native_args(self: Any) -> tuple[Any, ...]:
    """Read the native ``_detect_block``'s arguments.

    Raises _Unsupported for langdetect's Python path, also for any error reading them gives
    (langdetect's code raises it where langdetect does).
    """
    try:
        if self.verbose or type(self.text) is not str:
            raise _Unsupported
        prior = self.prior_map
        if prior is not None and not (type(prior) is list and all(type(p) is float for p in prior)):
            raise _Unsupported
        return (
            self.text,
            len(self.langlist),
            _seed_arg(self.seed),
            _params(self.alpha, self.n_trial, self),
            prior,
        )
    except _Unsupported:
        raise
    except Exception:
        raise _Unsupported from None


def _native_block(self: Any, profiles: Any) -> None:
    """Run ``Detector._detect_block`` in Rust.

    Raises _Unsupported, having changed nothing, for langdetect's Python path.
    """
    text, n_langs, seed, params, prior = _native_args(self)
    _finish_block(self, _native.detect_block(profiles, text, n_langs, seed, params, prior))


def _block_cleaned(self: Any, profiles: Any) -> None:
    """Run ``Detector._detect_block`` after its first step, ``self.cleaning_text()``.

    The detector's own ``cleaning_text`` has run; the rest runs in Rust if it can, else as
    langdetect's Python code.
    """
    try:
        text, n_langs, seed, params, prior = _native_args(self)
    except _Unsupported:
        _block_python_rest(self)
        return
    args = (profiles, text, n_langs, seed, params, prior, False)
    _finish_block(self, _native.detect_block(*args))


def _finish_block(self: Any, result: tuple[Any, int, Any]) -> None:
    cleaned, code, langprob = result
    if cleaned is not None:
        self.text = cleaned
    if code == 1:
        raise LangDetectException(ErrorCode.CantDetectError, "No features in text.")
    self.langprob = langprob
    if code == 2:
        raise IndexError("list index out of range")
    if code == 3:
        _zero_division()


class Detector:
    """Detect the language of a text: langdetect's ``Detector``.

    Made by :meth:`DetectorFactory.create`. :meth:`append` the text, then call :meth:`detect`
    or :meth:`get_probabilities`. Detections with :meth:`set_verbose`, with attributes of
    other types than langdetect's (a ``Decimal`` alpha, a seed of an unusual type), or with
    customised classes (``URL_RE`` or ``MAIL_RE`` set, methods overridden in a subclass,
    ``NGram``'s tables changed) run langdetect's algorithm in Python, with the same results.
    A subclass that overrides only :meth:`cleaning_text` keeps the Rust trials.
    """

    ALPHA_DEFAULT = 0.5
    ALPHA_WIDTH = 0.05

    ITERATION_LIMIT = 1000
    PROB_THRESHOLD = 0.1
    CONV_THRESHOLD = 0.99999
    BASE_FREQ = 10000
    UNKNOWN_LANG = "unknown"

    URL_RE = _URL_RE
    MAIL_RE = _MAIL_RE

    def __init__(self, factory: Any) -> None:
        self.word_lang_prob_map = factory.word_lang_prob_map
        self.langlist = factory.langlist
        self.seed = factory.seed
        self._profiles = factory._profiles
        self.text = ""
        self.langprob: list[float] | None = None

        self.alpha = self.ALPHA_DEFAULT
        self.n_trial = 7
        self.max_text_length = 10000
        self.prior_map: list[float] | None = None
        self.verbose = False

    def __setstate__(self, state: dict[str, Any]) -> None:
        state = dict(state)
        if "_profiles" not in state:
            # A langdetect Detector's pickle (pyroid.patch("langdetect") without langdetect):
            # its map is a dict, its generator the attribute `random`.
            word_map = state.pop("word_lang_prob_map", {})
            generator = state.pop("random", None)
            profiles = _profiles_from(word_map)
            state["_profiles"] = profiles if profiles is not None else _native._Profiles()
            state["word_lang_prob_map"] = _WordMap(profiles) if profiles is not None else word_map
            if generator is not None:
                state["_random"] = generator
        self.__dict__.update(state)

    @property
    def random(self) -> _random.Random:
        """A ``random.Random``, as langdetect makes one per detector.

        pyroid's detections draw from their own generator, seeded the same way; a generator of
        another class set here (a subclass) is used, by langdetect's Python algorithm.
        """
        r = self.__dict__.get("_random")
        if r is None:
            r = self.__dict__["_random"] = _Random()
        return r  # type: ignore[no-any-return]

    @random.setter
    def random(self, value: Any) -> None:
        self.__dict__["_random"] = value

    def set_verbose(self) -> None:
        """Print each step of the detection, as langdetect does (slow)."""
        self.verbose = True

    def set_alpha(self, alpha: float) -> None:
        """Set the smoothing parameter (default 0.5)."""
        self.alpha = alpha

    def set_prior_map(self, prior_map: Any) -> None:
        """Set prior probabilities per language.

        Raises:
            LangDetectException: A negative probability, or none above zero
                (``InitParamError``).
        """
        self.prior_map = [0.0] * len(self.langlist)
        sump = 0.0
        for i in range(len(self.prior_map)):
            lang = self.langlist[i]
            if lang in prior_map:
                p = prior_map[lang]
                if p < 0:
                    raise LangDetectException(
                        ErrorCode.InitParamError, "Prior probability must be non-negative."
                    )
                self.prior_map[i] = p
                sump += p
        if sump <= 0.0:
            raise LangDetectException(
                ErrorCode.InitParamError, "More one of prior probability must be non-zero."
            )
        for i in range(len(self.prior_map)):
            self.prior_map[i] /= sump

    def set_max_text_length(self, max_text_length: int) -> None:
        """Use at most this many characters of the text (default 10,000)."""
        self.max_text_length = max_text_length

    def append(self, text: str) -> None:
        """Append text: URLs and mail addresses become spaces, runs of spaces one space."""
        _append(self, text)

    def cleaning_text(self) -> None:
        """Remove Latin letters from a text that is mostly in another script."""
        cleaned = _native.clean_text(self.text)
        if cleaned is not None:
            self.text = cleaned

    def detect(self) -> Any:
        """The language with the highest probability, or ``'unknown'``.

        Raises:
            LangDetectException: The text has no features (``CantDetectError``).
        """
        probabilities = self.get_probabilities()
        if probabilities:
            return probabilities[0].lang
        return self.UNKNOWN_LANG

    def get_probabilities(self) -> list[Any]:
        """The languages above ``PROB_THRESHOLD``, most probable first.

        Raises:
            LangDetectException: The text has no features (``CantDetectError``).
        """
        if self.langprob is None:
            self._detect_block()
        return self._sort_probability(self.langprob)

    def _detect_block(self) -> None:
        mode = _block_mode(self)
        if mode == 0:
            try:
                _native_block(self, self._profiles)
                return
            except _Unsupported:
                pass
        # langdetect 1.0.9's algorithm, line for line (the trials in Rust if they can be).
        self.cleaning_text()
        if mode == 1:
            _block_cleaned(self, self._profiles)
        else:
            _block_python_rest(self)

    def _detect_block_python(self) -> None:
        # langdetect 1.0.9's algorithm, line for line.
        self.cleaning_text()
        _block_python_rest(self)

    def _init_probability(self) -> list[Any]:
        if self.prior_map is not None:
            return list(self.prior_map)
        return [1.0 / len(self.langlist)] * len(self.langlist)

    def _extract_ngrams(self) -> list[Any]:
        word_map = self.word_lang_prob_map
        text = self.text
        if type(word_map) is _WordMap and type(text) is str and NGram is _NGRAM and _ngram._stock():
            return list(_native.extract_ngrams(word_map._profiles, text))
        # langdetect's code, on this detector's map and NGram.
        ngram_range = list(range(1, NGram.N_GRAM + 1))
        result = []
        ngram = NGram()
        for ch in text:
            ngram.add_char(ch)
            if ngram.capitalword:
                continue
            for n in ngram_range:
                # optimized w = ngram.get(n)
                if len(ngram.grams) < n:
                    break
                w = ngram.grams[-n:]
                if w and w != " " and w in self.word_lang_prob_map:
                    result.append(w)
        return result

    def _update_lang_prob(self, prob: list[Any], word: Any, alpha: Any) -> bool:
        if word is None or word not in self.word_lang_prob_map:
            return False

        lang_prob_map = self.word_lang_prob_map[word]
        if self.verbose:
            print(  # noqa: T201
                "%s(%s): %s"  # noqa: UP031 (langdetect's formatting)
                % (word, self._unicode_encode(word), self._word_prob_to_string(lang_prob_map))
            )

        weight = alpha / self.BASE_FREQ
        for i in range(len(prob)):
            prob[i] *= weight + lang_prob_map[i]
        return True

    def _word_prob_to_string(self, prob: list[Any]) -> str:
        result = ""
        for j in range(len(prob)):
            p = prob[j]
            if p >= 0.00001:
                result += " %s:%.5f" % (self.langlist[j], p)  # noqa: UP031
        return result

    def _normalize_prob(self, prob: list[Any]) -> Any:
        maxp, sump = 0.0, sum(prob)
        for i in range(len(prob)):
            p = prob[i] / sump
            if maxp < p:
                maxp = p
            prob[i] = p
        return maxp

    def _sort_probability(self, prob: list[Any]) -> list[Any]:
        result = [
            Language(lang, p)
            for (lang, p) in zip(self.langlist, prob, strict=False)
            if p > self.PROB_THRESHOLD
        ]
        result.sort(reverse=True)
        return result

    def _unicode_encode(self, word: str) -> str:
        buf = ""
        for ch in word:
            if ch >= "\u0080":
                st = hex(0x10000 + ord(ch))[2:]
                while len(st) < 4:
                    st = "0" + st
                buf += r"\u" + st[1:5]
            else:
                buf += ch
        return buf


_NGRAM = NGram
_DETECTOR = Detector
_D = Detector.__dict__
_INIT = _D["__init__"]
_APPEND = _D["append"]
_CLEAN = _D["cleaning_text"]
_DETECT = _D["detect"]
_GET_PROBABILITIES = _D["get_probabilities"]
_DETECT_BLOCK = _D["_detect_block"]
_SORT = _D["_sort_probability"]
_EXTRACT = _D["_extract_ngrams"]
_INIT_PROB = _D["_init_probability"]
_UPDATE = _D["_update_lang_prob"]
_NORMALIZE_PROB = _D["_normalize_prob"]
_RANDOM_PROPERTY = _D["random"]
del _D
# What langdetect's detect/detect_langs reach through the class: read in one call.
_HOOK_NAMES = (
    "__init__",
    "append",
    "detect",
    "get_probabilities",
    "_detect_block",
    "_sort_probability",
    "cleaning_text",
    "_extract_ngrams",
    "_init_probability",
    "_update_lang_prob",
    "_normalize_prob",
    "random",
    "URL_RE",
    "MAIL_RE",
)
_HOOKS = operator.attrgetter(*_HOOK_NAMES)
_STOCK_HOOKS = _HOOKS(Detector)
# The class constants of a native call.
_CONSTANTS = operator.attrgetter(
    "ALPHA_DEFAULT", "ALPHA_WIDTH", "ITERATION_LIMIT", "CONV_THRESHOLD", "BASE_FREQ"
)


def _class_params(cls: Any) -> tuple[Any, ...]:
    """``_params(cls.ALPHA_DEFAULT, 7, cls)`` (a fresh detector's), langdetect's types first."""
    alpha, width, limit, conv, base = _CONSTANTS(cls)
    if (
        type(alpha) is float
        and type(width) is float
        and type(limit) is int
        and type(conv) is float
        and type(base) is int
        and -_MAX_INT_PARAM < limit < _MAX_INT_PARAM
        and -_MAX_INT_PARAM < base < _MAX_INT_PARAM
    ):
        return (alpha, 7, width, limit, conv, base)
    return _params(alpha, 7, cls)


def _class_stock(cls: Any) -> bool:
    """A fresh ``cls(factory)`` detects exactly as one native call does.

    Every method langdetect's ``detect``/``detect_langs`` reach is pyroid's, and so are the
    patterns, ``NGram`` and ``Language``; the class constants are read and passed on.
    """
    try:
        return (
            _HOOKS(cls) == _STOCK_HOOKS  # functions by identity, patterns by value
            and NGram is _NGRAM
            and Language is _compat.Language
            and _ngram._stock()
        )
    except Exception:  # a missing attribute: langdetect's code raises it
        return False


def _block_python_rest(self: Any) -> None:
    """Run langdetect 1.0.9's ``_detect_block`` after ``self.cleaning_text()``, line for line."""
    ngrams = self._extract_ngrams()
    if not ngrams:
        raise LangDetectException(ErrorCode.CantDetectError, "No features in text.")

    self.langprob = [0.0] * len(self.langlist)

    self.random.seed(self.seed)
    for _ in range(self.n_trial):
        prob = self._init_probability()
        alpha = self.alpha + self.random.gauss(0.0, 1.0) * self.ALPHA_WIDTH

        i = 0
        while True:
            self._update_lang_prob(prob, self.random.choice(ngrams), alpha)
            if i % 5 == 0:
                if self._normalize_prob(prob) > self.CONV_THRESHOLD or i >= self.ITERATION_LIMIT:
                    break
                if self.verbose:
                    print(">", self._sort_probability(prob))  # noqa: T201
            i += 1
        for j in range(len(self.langprob)):
            self.langprob[j] += prob[j] / self.n_trial
        if self.verbose:
            print("==>", self._sort_probability(prob))  # noqa: T201


def _profiles_from(word_map: Any) -> Any:
    """A native table for langdetect's own map (a dict of str to lists of floats), else None."""
    if type(word_map) is not dict:
        return None
    return _native._Profiles._from_map(word_map, keep_all=True)


def _class_items() -> list[tuple[Any, str, Any, int]]:
    """`_class_stock(Detector)`'s checks for a native check list (`_ngram._stock_items`)."""
    module = sys.modules[__name__]
    hooks = [
        (_DETECTOR, n, v, _ngram._EQ if n in ("URL_RE", "MAIL_RE") else _ngram._IS)
        for n, v in zip(_HOOK_NAMES, _STOCK_HOOKS, strict=True)
    ]
    return [
        *hooks,
        (module, "NGram", _NGRAM, _ngram._IS),
        (module, "Language", _compat.Language, _ngram._IS),
        *_ngram._stock_items(),
    ]


_IS = _ngram._IS
_MODULE = sys.modules[__name__]
# `_block_mode`'s class-level checks (for a detector whose class is pyroid's Detector).
_BLOCK_CHECKS = _native._Checks(
    _ngram._items(
        [
            (_DETECTOR, "_extract_ngrams", _EXTRACT, _IS),
            (_DETECTOR, "_init_probability", _INIT_PROB, _IS),
            (_DETECTOR, "_update_lang_prob", _UPDATE, _IS),
            (_DETECTOR, "_normalize_prob", _NORMALIZE_PROB, _IS),
            (_DETECTOR, "random", _RANDOM_PROPERTY, _IS),
            (_MODULE, "NGram", _NGRAM, _IS),
            *_ngram._stock_items(),
        ]
    )
)
# `_append_stock`'s checks besides the patterns (which may be a detector's own).
_APPEND_CHECKS = _native._Checks(
    _ngram._items(
        [
            (_MODULE, "NGram", _NGRAM, _IS),
            (_NGRAM, "normalize_vi", _ngram._VI_FUNC.__func__, _ngram._FUNC),
            (_NGRAM, "TO_NORMALIZE_VI_CHARS", _ngram._VI_BASE, _ngram._EQ),
            (_NGRAM, "DMARK_CLASS", _ngram._VI_MARKS, _ngram._EQ),
            (_NGRAM, "NORMALIZED_VI_CHARS", _ngram._VI_TABLE, _ngram._EQ),
            (_NGRAM, "ALPHABET_WITH_DMARK", _ngram._VI_RE, _ngram._EQ),
        ]
    )
)
