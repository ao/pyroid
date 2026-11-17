# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""langdetect's ``DetectorFactory``, ``init_factory``, ``detect`` and ``detect_langs``.

The profiles load as in langdetect (``os.listdir`` order, ``json``, :class:`LangProfile`),
into a table in Rust. The default factory reads pyroid's copy of langdetect 1.0.9's
profiles (:data:`PROFILES_DIRECTORY`), in the order ``os.listdir`` gives for langdetect's own
profile directory when langdetect is installed, so the languages' order (and with it every
probability's last bit) is the one langdetect has on this machine.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator, Mapping
from os import path
from typing import Any

from pyroid._pyroid import langdetect as _native
from pyroid.langdetect import _compat
from pyroid.langdetect._compat import ErrorCode, LangDetectException, Language
from pyroid.langdetect.detector import Detector, _params, _seed_arg, _Unsupported
from pyroid.langdetect.utils.lang_profile import LangProfile

__all__ = [
    "PROFILES_DIRECTORY",
    "DetectorFactory",
    "detect",
    "detect_langs",
    "init_factory",
]


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


def _add_slow(profiles: Any, profile: Any, index: Any, langsize: Any) -> None:
    """``add_profile``'s loop, in Python, for profiles of other shapes than JSON's."""
    for word in profile.freq:
        if not profiles._contains(word):
            row = [0.0] * langsize
            profiles._create(word, len(row))
        length = len(word)
        if 1 <= length <= 3:
            prob = 1.0 * profile.freq.get(word) / profile.n_words[length - 1]
            # `row[index] = prob`: Python's own checks of the index.
            row = [None] * profiles._row_len(word)
            row[index] = True
            profiles._set(word, row.index(True), float(prob))


class DetectorFactory:
    """Load language profiles and make :class:`Detector` objects (langdetect's class).

    ``DetectorFactory.seed = 0`` makes every detection reproducible, as in langdetect.
    """

    seed: Any = None

    def __init__(self) -> None:
        self._profiles = _native._Profiles()
        self.langlist: list[Any] = []

    @property
    def word_lang_prob_map(self) -> Mapping[Any, list[float]]:
        """n-gram -> probability per language, a read-only view of the loaded profiles."""
        return _WordMap(self._profiles)

    def load_profile(self, profile_directory: Any) -> None:
        """Load every profile file in `profile_directory` (in ``os.listdir`` order)."""
        self._load_files(profile_directory, os.listdir(profile_directory))

    def _load_files(self, profile_directory: Any, list_files: list[str]) -> None:
        if not list_files:
            raise LangDetectException(
                ErrorCode.NeedLoadProfileError, "Not found profile: " + profile_directory
            )

        langsize, index = len(list_files), 0
        for filename in list_files:
            if filename.startswith("."):
                continue
            filename = path.join(profile_directory, filename)
            if not path.isfile(filename):
                continue

            f = None
            try:
                f = open(filename, encoding="utf-8")  # noqa: SIM115 (langdetect's structure)
                json_data = json.load(f)
                profile = LangProfile(**json_data)
                self.add_profile(profile, index, langsize)
                index += 1
            # As langdetect, the original error is the new one's __context__. Unlike langdetect,
            # KeyboardInterrupt and SystemExit pass through.
            except OSError:
                raise LangDetectException(  # noqa: B904
                    ErrorCode.FileLoadError, 'Cannot open "%s"' % filename  # noqa: UP031
                )
            except Exception:
                raise LangDetectException(  # noqa: B904
                    ErrorCode.FormatError,
                    'Profile format error in "%s"' % filename,  # noqa: UP031
                )
            finally:
                if f:
                    f.close()

    def load_json_profile(self, json_profiles: Any) -> None:
        """Load profiles from JSON strings (at least two)."""
        langsize, index = len(json_profiles), 0
        if langsize < 2:
            raise LangDetectException(ErrorCode.NeedLoadProfileError, "Need more than 2 profiles.")

        for json_profile in json_profiles:
            try:
                json_data = json.loads(json_profile)
                profile = LangProfile(**json_data)
                self.add_profile(profile, index, langsize)
                index += 1
            except Exception:
                raise LangDetectException(ErrorCode.FormatError, "Profile format error.")  # noqa: B904

    def add_profile(self, profile: Any, index: Any, langsize: Any) -> None:
        """Add one :class:`LangProfile` as language number `index` of `langsize`."""
        lang = profile.name
        if lang in self.langlist:
            raise LangDetectException(
                ErrorCode.DuplicateLangError, "Duplicate the same language profile."
            )
        self.langlist.append(lang)
        if not self._profiles._add(profile.freq, profile.n_words, index, langsize):
            _add_slow(self._profiles, profile, index, langsize)

    def clear(self) -> None:
        """Forget every profile."""
        self.langlist = []
        self._profiles = _native._Profiles()

    def create(self, alpha: float | None = None) -> Detector:
        """Construct a :class:`Detector`, with smoothing parameter `alpha` if given."""
        detector = self._create_detector()
        if alpha is not None:
            detector.set_alpha(alpha)
        return detector

    def _create_detector(self) -> Detector:
        if not self.langlist:
            raise LangDetectException(ErrorCode.NeedLoadProfileError, "Need to load profiles.")
        return Detector(self)

    def set_seed(self, seed: Any) -> None:
        """Set this factory's seed (``None``: a fresh random seed per detection)."""
        self.seed = seed

    def get_lang_list(self) -> list[Any]:
        """The loaded languages, in load order."""
        return list(self.langlist)


#: pyroid's copy of langdetect 1.0.9's 55 language profiles.
PROFILES_DIRECTORY = path.join(path.dirname(__file__), "profiles")
_factory: DetectorFactory | None = None
_LOCK = threading.Lock()


def _default_files() -> list[str]:
    """The profile file names, in the order langdetect's ``init_factory`` would read them."""
    ours = os.listdir(PROFILES_DIRECTORY)
    theirs_dir = _compat._profiles_dir()
    if theirs_dir is not None:
        try:
            theirs = os.listdir(theirs_dir)
        except OSError:
            return ours
        if sorted(theirs) == sorted(ours):
            return theirs
    return ours


def init_factory() -> None:
    """Load the default profiles into the module's factory (once per process)."""
    global _factory
    if _factory is None:
        with _LOCK:
            if _factory is None:
                factory = DetectorFactory()
                factory._load_files(PROFILES_DIRECTORY, _default_files())
                _factory = factory


def _fast(factory: DetectorFactory, text: Any, probs: bool) -> Any:
    """`detect`/`detect_langs` in one native call; raises _Unsupported for the general path."""
    if not isinstance(text, str):
        raise _Unsupported
    seed = _seed_arg(factory.seed)
    params = _params(Detector.ALPHA_DEFAULT, 7, Detector)
    threshold = Detector.PROB_THRESHOLD
    if type(threshold) not in (int, float):
        raise _Unsupported
    code, value = _native.detect_one(
        factory._profiles,
        factory.langlist,
        text,
        seed,
        params,
        threshold,
        probs,
        Detector.UNKNOWN_LANG,
    )
    if code == 0:
        return value
    if code == 1:
        raise LangDetectException(ErrorCode.CantDetectError, "No features in text.")
    raise _Unsupported  # an error with a detector's state: raise it the general way


def detect(text: str) -> Any:
    """The language of `text` (``'unknown'`` if no language is above the threshold).

    Raises:
        LangDetectException: The text has no features (no letters, only URLs, ...).
    """
    init_factory()
    factory = _factory
    assert factory is not None
    try:
        return _fast(factory, text, False)
    except _Unsupported:
        pass
    detector = factory.create()
    detector.append(text)
    return detector.detect()


def detect_langs(text: str) -> list[Any]:
    """The languages of `text` with their probabilities, most probable first.

    Raises:
        LangDetectException: The text has no features (no letters, only URLs, ...).
    """
    init_factory()
    factory = _factory
    assert factory is not None
    try:
        return [Language(lang, p) for lang, p in _fast(factory, text, True)]
    except _Unsupported:
        pass
    detector = factory.create()
    detector.append(text)
    return detector.get_probabilities()
