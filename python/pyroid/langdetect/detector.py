# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""langdetect's ``Detector``: ``append`` and ``_detect_block`` run in Rust.

The attributes are langdetect's (``text``, ``langprob``, ``alpha``, ``n_trial``,
``max_text_length``, ``prior_map``, ``verbose``, ``seed``, ``langlist``,
``word_lang_prob_map``), and so are the results, to the last bit of every probability, for
the same seed. Detections that langdetect would print about (``set_verbose()``), or with
attribute values of unusual types, run langdetect's own algorithm in Python, with the same
results.
"""

from __future__ import annotations

import operator
import random as _random
import re
import sys
from typing import Any

from pyroid._pyroid import langdetect as _native
from pyroid.langdetect._compat import ErrorCode, LangDetectException, Language

__all__ = ["Detector"]

_NUMBER = (int, float)
_MAX_INT_SEED = 1 << 127


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
        state = _native._SeedState(_random.Random(seed).getstate())
        try:
            if len(self._states) >= 16:
                self._states.clear()
            self._states[key] = state
        except TypeError:  # unhashable (bytearray): not cached
            pass
        return state


_SEEDS = _SeedCache()


class _Unsupported(Exception):
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
        return _SEEDS.get(seed)
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
    if type(n_trial) is not int or not all(type(v) in _NUMBER for v in values):
        raise _Unsupported
    if not all(abs(v) < 1e308 for v in values if type(v) is int):
        raise _Unsupported
    return values


def _zero_division() -> Any:
    """Raise the interpreter's own ZeroDivisionError for a float division by zero."""
    zero = 0.0
    return 1.0 / zero


def _append(self: Any, text: Any) -> None:
    """``Detector.append``, on any object with langdetect's ``text`` and ``max_text_length``."""
    if not isinstance(text, str):
        self.URL_RE.sub(" ", text)  # raises what langdetect raises for a non-str
        raise TypeError(f"expected str, got {type(text).__name__}")
    limit = self.max_text_length
    if type(limit) is int:
        self.text += _native.append_text(text, 0 if limit < 0 else min(limit, sys.maxsize))
        return
    # Other types: `range(min(len(text), max_text_length))` decides, as in langdetect.
    substituted = _native.substitute(text)
    n = operator.index(min(len(substituted), limit))
    self.text += _native.collapse(substituted, max(n, 0))


def _native_block(self: Any, profiles: Any) -> None:
    """``Detector._detect_block`` in Rust; raises _Unsupported for langdetect's Python path."""
    if self.verbose or type(self.text) is not str:
        raise _Unsupported
    prior = self.prior_map
    if prior is not None and not (type(prior) is list and all(type(p) is float for p in prior)):
        raise _Unsupported
    seed = _seed_arg(self.seed)
    params = _params(self.alpha, self.n_trial, self)
    cleaned, code, langprob = _native.detect_block(
        profiles, self.text, len(self.langlist), seed, params, prior
    )
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

    Made by :meth:`DetectorFactory.create`. ``append`` the text, then call :meth:`detect` or
    :meth:`get_probabilities`.
    """

    ALPHA_DEFAULT = 0.5
    ALPHA_WIDTH = 0.05

    ITERATION_LIMIT = 1000
    PROB_THRESHOLD = 0.1
    CONV_THRESHOLD = 0.99999
    BASE_FREQ = 10000
    UNKNOWN_LANG = "unknown"

    URL_RE = re.compile(r"https?://[-_.?&~;+=/#0-9A-Za-z]{1,2076}")
    MAIL_RE = re.compile(r"[-_.0-9A-Za-z]{1,64}@[-_0-9A-Za-z]{1,255}[-_.0-9A-Za-z]{1,255}")

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

    def __getattr__(self, name: str) -> Any:
        # langdetect makes a `random.Random()` per detector; pyroid makes it on first use.
        if name == "random":
            r = self.__dict__["random"] = _random.Random()
            return r
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def set_verbose(self) -> None:
        """Print what the detection does (slow: runs langdetect's Python algorithm)."""
        self.verbose = True

    def set_alpha(self, alpha: float) -> None:
        """Set the smoothing parameter."""
        self.alpha = alpha

    def set_prior_map(self, prior_map: Any) -> None:
        """Set prior information about language probabilities."""
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
        """Specify max size of target text to use for language detection (default 10000)."""
        self.max_text_length = max_text_length

    def append(self, text: str) -> None:
        """Append the target text; the part past ``max_text_length`` is cut off."""
        _append(self, text)

    def cleaning_text(self) -> None:
        """Remove Latin letters from a text that is mostly not Latin, as langdetect does."""
        cleaned = _native.clean_text(self.text)
        if cleaned is not None:
            self.text = cleaned

    def detect(self) -> Any:
        """The language with the highest probability, or ``'unknown'``."""
        probabilities = self.get_probabilities()
        if probabilities:
            return probabilities[0].lang
        return self.UNKNOWN_LANG

    def get_probabilities(self) -> list[Any]:
        """The languages above ``PROB_THRESHOLD``, most probable first."""
        if self.langprob is None:
            self._detect_block()
        return self._sort_probability(self.langprob)

    def _detect_block(self) -> None:
        try:
            _native_block(self, self._profiles)
        except _Unsupported:
            self._detect_block_python()

    def _detect_block_python(self) -> None:
        # langdetect 1.0.9's algorithm, line for line.
        self.cleaning_text()
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
                    if (
                        self._normalize_prob(prob) > self.CONV_THRESHOLD
                        or i >= self.ITERATION_LIMIT
                    ):
                        break
                    if self.verbose:
                        print(">", self._sort_probability(prob))  # noqa: T201
                i += 1
            for j in range(len(self.langprob)):
                self.langprob[j] += prob[j] / self.n_trial
            if self.verbose:
                print("==>", self._sort_probability(prob))  # noqa: T201

    def _init_probability(self) -> list[Any]:
        if self.prior_map is not None:
            return list(self.prior_map)
        return [1.0 / len(self.langlist)] * len(self.langlist)

    def _extract_ngrams(self) -> list[str]:
        return list(_native.extract_ngrams(self._profiles, self.text))

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
