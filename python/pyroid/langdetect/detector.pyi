# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""langdetect's ``Detector``; ``append`` and the detection run in Rust."""

import random
from collections.abc import Mapping
from re import Pattern
from typing import Any, ClassVar

from pyroid.langdetect.language import Language

__all__ = ["Detector"]

class Detector:
    """Detect the language of a text: langdetect's ``Detector``.

    Made by :meth:`DetectorFactory.create`. :meth:`append` the text, then call :meth:`detect`
    or :meth:`get_probabilities`. Detections with :meth:`set_verbose`, or with attributes of
    other types than langdetect's (a ``Decimal`` alpha, a seed of an unusual type), run
    langdetect's algorithm in Python, with the same results.
    """

    ALPHA_DEFAULT: ClassVar[float]
    ALPHA_WIDTH: ClassVar[float]
    ITERATION_LIMIT: ClassVar[int]
    PROB_THRESHOLD: ClassVar[float]
    CONV_THRESHOLD: ClassVar[float]
    BASE_FREQ: ClassVar[int]
    UNKNOWN_LANG: ClassVar[str]
    URL_RE: ClassVar[Pattern[str]]
    MAIL_RE: ClassVar[Pattern[str]]

    word_lang_prob_map: Mapping[Any, list[float]]
    langlist: list[Any]
    seed: Any
    text: str
    langprob: list[float] | None
    alpha: float
    n_trial: int
    max_text_length: int
    prior_map: list[float] | None
    verbose: bool
    def __init__(self, factory: Any) -> None: ...
    @property
    def random(self) -> random.Random:
        """A ``random.Random``, as langdetect makes one per detector (unused by pyroid)."""
    @random.setter
    def random(self, value: Any) -> None: ...
    def set_verbose(self) -> None:
        """Print each step of the detection, as langdetect does (slow)."""
    def set_alpha(self, alpha: float) -> None:
        """Set the smoothing parameter (default 0.5)."""
    def set_prior_map(self, prior_map: Mapping[Any, float]) -> None:
        """Set prior probabilities per language.

        Raises:
            LangDetectException: A negative probability, or none above zero
                (``InitParamError``).
        """
    def set_max_text_length(self, max_text_length: int) -> None:
        """Use at most this many characters of the text (default 10,000)."""
    def append(self, text: str) -> None:
        """Append text: URLs and mail addresses become spaces, runs of spaces one space."""
    def cleaning_text(self) -> None:
        """Remove Latin letters from a text that is mostly in another script."""
    def detect(self) -> str:
        """The language with the highest probability, or ``'unknown'``.

        Raises:
            LangDetectException: The text has no features (``CantDetectError``).
        """
    def get_probabilities(self) -> list[Language]:
        """The languages above ``PROB_THRESHOLD``, most probable first.

        Raises:
            LangDetectException: The text has no features (``CantDetectError``).
        """
