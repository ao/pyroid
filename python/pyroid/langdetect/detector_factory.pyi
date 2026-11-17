# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""langdetect's ``DetectorFactory``, ``init_factory``, ``detect`` and ``detect_langs``."""

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Final

from pyroid.langdetect.detector import Detector
from pyroid.langdetect.language import Language

__all__ = ["PROFILES_DIRECTORY", "DetectorFactory", "detect", "detect_langs", "init_factory"]

#: pyroid's copy of langdetect 1.0.9's 55 language profiles.
PROFILES_DIRECTORY: Final[str]

class DetectorFactory:
    """Load language profiles and make :class:`Detector` objects (langdetect's class).

    ``DetectorFactory.seed = 0`` makes every detection reproducible, as in langdetect.
    """

    #: The seed of every detection (``None``: a fresh random seed each time).
    seed: ClassVar[Any]
    #: The loaded languages, in load order.
    langlist: list[Any]
    def __init__(self) -> None: ...
    @property
    def word_lang_prob_map(self) -> Mapping[Any, list[float]]:
        """n-gram -> probability per language, a read-only view of the loaded profiles.

        It is langdetect's dict itself when a profile probability is not a float (numpy
        counts) or a map of another shape was assigned; detections then run langdetect's
        Python algorithm on it.
        """
    @word_lang_prob_map.setter
    def word_lang_prob_map(self, word_map: Mapping[Any, list[Any]]) -> None: ...
    def load_profile(self, profile_directory: str) -> None:
        """Load every profile file in `profile_directory`, in ``os.listdir`` order.

        Raises:
            LangDetectException: No files (``NeedLoadProfileError``), a file that cannot be
                read (``FileLoadError``) or parsed, or a language loaded twice
                (``FormatError``).
        """
    def load_json_profile(self, json_profiles: Sequence[str]) -> None:
        """Load profiles from JSON strings (at least two).

        Raises:
            LangDetectException: Fewer than two (``NeedLoadProfileError``), or one that is not
                a profile (``FormatError``).
        """
    def add_profile(self, profile: Any, index: int, langsize: int) -> None:
        """Add one :class:`LangProfile` as language number `index` of `langsize`.

        Raises:
            LangDetectException: The language is loaded already (``DuplicateLangError``).
        """
    def clear(self) -> None:
        """Forget every profile."""
    def create(self, alpha: float | None = None) -> Detector:
        """A new :class:`Detector`, with smoothing parameter `alpha` if given.

        Raises:
            LangDetectException: No profiles are loaded (``NeedLoadProfileError``).
        """
    def set_seed(self, seed: Any) -> None:
        """Set this factory's seed."""
    def get_lang_list(self) -> list[Any]:
        """The loaded languages, in load order."""

def init_factory() -> None:
    """Load the default profiles into the module's factory (once per process)."""

def detect(text: str) -> str:
    """The language of `text` (``'unknown'`` if none is above the threshold).

    Raises:
        LangDetectException: `text` has no features (no letters, only URLs, ...).
    """

def detect_langs(text: str) -> list[Language]:
    """The languages of `text` with their probabilities, most probable first.

    Raises:
        LangDetectException: `text` has no features (no letters, only URLs, ...).
    """
