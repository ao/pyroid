# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""langdetect's ``DetectorFactory``, ``init_factory``, ``detect`` and ``detect_langs``.

The profiles load as in langdetect (``os.listdir`` order, ``json``, :class:`LangProfile`),
into a table in Rust. The default factory reads pyroid's copy of langdetect 1.0.9's
profiles (:data:`PROFILES_DIRECTORY`), in the order ``os.listdir`` gives for langdetect's own
profile directory when langdetect is installed, so the languages' order (and with it every
probability's last bit) is the one langdetect has on this machine.
"""

from __future__ import annotations

import copyreg
import json
import os
import sys
import threading
from collections.abc import Mapping
from os import path
from typing import Any

from pyroid._pyroid import langdetect as _native
from pyroid.langdetect import _compat
from pyroid.langdetect import detector as _detector
from pyroid.langdetect._compat import ErrorCode, LangDetectException, Language
from pyroid.langdetect.detector import Detector, _seed_arg, _Unsupported, _WordMap
from pyroid.langdetect.utils import ngram as _ngram
from pyroid.langdetect.utils.lang_profile import LangProfile

__all__ = [
    "PROFILES_DIRECTORY",
    "DetectorFactory",
    "detect",
    "detect_langs",
    "init_factory",
]


def _load_profiles(data: bytes, foreign: list[Any]) -> Any:
    """A native profile table from its pickle (``copyreg`` reduces ``_Profiles`` to this)."""
    return _native._Profiles._load(data, foreign)


copyreg.pickle(_native._Profiles, lambda p: (_load_profiles, p._dump()))


class DetectorFactory:
    """Load language profiles and make :class:`Detector` objects (langdetect's class).

    ``DetectorFactory.seed = 0`` makes every detection reproducible, as in langdetect.
    """

    seed: Any = None

    def __init__(self) -> None:
        self._profiles = _native._Profiles()
        # langdetect's dict, kept instead of the native table once a profile probability is
        # not a float (numpy counts) or a map of another shape was assigned: then detections
        # run langdetect's Python algorithm on it.
        self._exotic: Any = None
        self.langlist: list[Any] = []

    def __setstate__(self, state: dict[str, Any]) -> None:
        state = dict(state)
        if "_profiles" not in state:
            # A langdetect DetectorFactory's pickle (pyroid.patch("langdetect") without
            # langdetect): its map is the attribute `word_lang_prob_map`.
            word_map = state.pop("word_lang_prob_map", {})
            state.setdefault("langlist", [])
            self.__dict__.update(state)
            self.word_lang_prob_map = word_map
            return
        state.setdefault("_exotic", None)
        self.__dict__.update(state)

    @property
    def word_lang_prob_map(self) -> Mapping[Any, list[Any]]:
        """n-gram -> probability per language, a read-only view of the loaded profiles.

        It is langdetect's dict itself when a profile probability is not a float (numpy
        counts) or a map of another shape was assigned; detections then run langdetect's
        Python algorithm on it.
        """
        if self._exotic is not None:
            return self._exotic  # type: ignore[no-any-return]
        return _WordMap(self._profiles)

    @word_lang_prob_map.setter
    def word_lang_prob_map(self, word_map: Any) -> None:
        profiles = _detector._profiles_from(word_map)
        if profiles is not None:
            self._profiles, self._exotic = profiles, None
        else:
            self._profiles, self._exotic = _native._Profiles(), word_map

    def load_profile(self, profile_directory: Any) -> None:
        """Load every profile file in `profile_directory`, in ``os.listdir`` order.

        Raises:
            LangDetectException: No files (``NeedLoadProfileError``), a file that cannot be
                read (``FileLoadError``) or parsed, or a language loaded twice
                (``FormatError``).
        """
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
                    ErrorCode.FileLoadError,
                    'Cannot open "%s"' % filename,  # noqa: UP031
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
        """Load profiles from JSON strings (at least two).

        Raises:
            LangDetectException: Fewer than two (``NeedLoadProfileError``), or one that is not
                a profile (``FormatError``).
        """
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
        """Add one :class:`LangProfile` as language number `index` of `langsize`.

        Raises:
            LangDetectException: The language is loaded already (``DuplicateLangError``).
        """
        lang = profile.name
        if lang in self.langlist:
            raise LangDetectException(
                ErrorCode.DuplicateLangError, "Duplicate the same language profile."
            )
        self.langlist.append(lang)
        word_map = self._exotic
        if word_map is None:
            if self._profiles._add(profile.freq, profile.n_words, index, langsize):
                return
            word_map = self._add_slow(profile, index, langsize)
            if word_map is None:
                return
        # langdetect's own loop, on its dict.
        for word in profile.freq:
            if word not in word_map:
                word_map[word] = [0.0] * langsize
            length = len(word)
            if 1 <= length <= 3:
                prob = 1.0 * profile.freq.get(word) / profile.n_words[length - 1]
                word_map[word][index] = prob

    def _add_slow(self, profile: Any, index: Any, langsize: Any) -> Any:
        """Run ``add_profile``'s loop in Python into the native table.

        For profiles of other shapes than JSON's. At the first probability that is not a float
        (numpy counts give ``numpy.float64``) the table becomes langdetect's dict, and this
        returns it: the caller runs langdetect's loop on it again from the start, which sets
        the words already done to the same values.
        """
        profiles = self._profiles
        for word in profile.freq:
            if not profiles._contains(word):
                row = [0.0] * langsize
                profiles._create(word, len(row))
            length = len(word)
            if 1 <= length <= 3:
                prob = 1.0 * profile.freq.get(word) / profile.n_words[length - 1]
                if type(prob) is not float:
                    word_map = {w: profiles._row(w) for w in profiles._keys()}
                    self._profiles, self._exotic = _native._Profiles(), word_map
                    return word_map
                # `row[index] = prob`: Python's own checks of the index.
                row = [None] * profiles._row_len(word)
                row[index] = True
                profiles._set(word, row.index(True), prob)
        return None

    def clear(self) -> None:
        """Forget every profile."""
        self.langlist = []
        self._profiles = _native._Profiles()
        self._exotic = None

    def create(self, alpha: float | None = None) -> Detector:
        """A new :class:`Detector`, with smoothing parameter `alpha` if given.

        Raises:
            LangDetectException: No profiles are loaded (``NeedLoadProfileError``).
        """
        detector = self._create_detector()
        if alpha is not None:
            detector.set_alpha(alpha)
        return detector

    def _create_detector(self) -> Detector:
        if not self.langlist:
            raise LangDetectException(ErrorCode.NeedLoadProfileError, "Need to load profiles.")
        return Detector(self)

    def set_seed(self, seed: Any) -> None:
        """Set this factory's seed."""
        self.seed = seed

    def get_lang_list(self) -> list[Any]:
        """The loaded languages, in load order."""
        return list(self.langlist)


#: pyroid's copy of langdetect 1.0.9's 55 language profiles.
PROFILES_DIRECTORY = path.join(path.dirname(__file__), "profiles")
_DEFAULT_DIRECTORY = PROFILES_DIRECTORY
_factory: DetectorFactory | None = None
_LOCK = threading.Lock()


def _default_files() -> list[str]:
    """The profile file names, in the order langdetect's ``init_factory`` would read them."""
    ours = os.listdir(_DEFAULT_DIRECTORY)
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
                directory = PROFILES_DIRECTORY
                if (
                    type(factory) is _FACTORY
                    and _FACTORY.load_profile is _LOAD_PROFILE
                    and directory == _DEFAULT_DIRECTORY
                ):
                    factory._load_files(directory, _default_files())
                else:  # langdetect's call (a replaced class, method or directory)
                    factory.load_profile(directory)
                _factory = factory


_FACTORY = DetectorFactory
_F = DetectorFactory.__dict__
_LOAD_PROFILE = _F["load_profile"]
_CREATE = _F["create"]
_CREATE_DETECTOR = _F["_create_detector"]
_MAP_PROPERTY = _F["word_lang_prob_map"]
_MAX_SEED = _detector._MAX_INT_SEED
del _F
_DETECTOR = _detector._DETECTOR
_CUSTOMISED = 6  # detect_checked's code: a check failed


def _plain(factory: Any) -> bool:
    """`factory.create()`, ``append`` and ``detect()`` give what one native call gives."""
    try:
        cls = type(factory)
        d = factory.__dict__
        return (
            cls is _FACTORY
            and cls.create is _CREATE
            and cls._create_detector is _CREATE_DETECTOR
            and cls.word_lang_prob_map is _MAP_PROPERTY
            and Detector is _detector._DETECTOR
            and d["_exotic"] is None
            and "create" not in d
            and "_create_detector" not in d
            and _detector._class_stock(Detector)
        )
    except Exception:  # a missing attribute: langdetect's code raises it
        return False


def _fast(factory: Any, text: Any, probs: bool) -> Any:
    """`detect`/`detect_langs` in one native call; raises _Unsupported for the general path.

    The native call first checks the factory, the classes (``_CHECKS``) and the seed and reads
    ``Detector``'s constants; when a check fails (code 6) the checks run again in Python, which
    also accept constants and seeds of other types and a ``CJK_MAP`` changed back.
    """
    if not isinstance(text, str):
        raise _Unsupported
    code, value = _native.detect_fast(_CHECKS, _FACTORY, _DETECTOR_ATTRS, factory, text, probs)
    if code == _CUSTOMISED:
        if not _plain(factory):
            raise _Unsupported
        seed = _seed_arg(factory.seed)
        profiles, langlist = factory._profiles, factory.langlist
        params = _detector._class_params(Detector)
        threshold = Detector.PROB_THRESHOLD
        if type(threshold) not in (int, float):
            raise _Unsupported
        code, value = _native.detect_one(
            profiles, langlist, text, seed, params, threshold, probs, Detector.UNKNOWN_LANG
        )
    if code == 0:
        return value
    if code == 1:
        raise LangDetectException(ErrorCode.CantDetectError, "No features in text.")
    raise _Unsupported  # an error with a detector's state: raise it the general way


def detect(text: str) -> Any:
    """The language of `text` (``'unknown'`` if none is above the threshold).

    Raises:
        LangDetectException: `text` has no features (no letters, only URLs, ...).
    """
    init_factory()
    factory: Any = _factory
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
        LangDetectException: `text` has no features (no letters, only URLs, ...).
    """
    init_factory()
    factory: Any = _factory
    try:
        return [Language(lang, p) for lang, p in _fast(factory, text, True)]
    except _Unsupported:
        pass
    detector = factory.create()
    detector.append(text)
    return detector.get_probabilities()


# What one native detect() stands for, checked by the native call itself: DetectorFactory's
# methods, this module's Detector, and Detector's and NGram's (see `_plain`).
_CHECKS = _native._Checks(
    _ngram._items(
        [
            (sys.modules[__name__], "Detector", _DETECTOR, _ngram._IS),
            (DetectorFactory, "create", _CREATE, _ngram._IS),
            (DetectorFactory, "_create_detector", _CREATE_DETECTOR, _ngram._IS),
            (DetectorFactory, "word_lang_prob_map", _MAP_PROPERTY, _ngram._IS),
            *_detector._class_items(),
        ]
    )
)
_DETECTOR_ATTRS = _ngram._own(_DETECTOR) or _DETECTOR  # where detect_fast reads the constants
