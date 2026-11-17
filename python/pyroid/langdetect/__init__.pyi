# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""A drop-in for langdetect 1.0.9, in Rust.

```python
from pyroid.langdetect import DetectorFactory, detect, detect_langs

DetectorFactory.seed = 0
detect("Ein kurzer deutscher Satz.")  # 'de'
detect_langs("Bonjour tout le monde")  # [fr:0.9999...]
pyroid.langdetect.detect_batch(texts)  # one language per text, on pyroid's thread pool
```

``detect``, ``detect_langs``, :class:`DetectorFactory`, :class:`Detector` (``append``,
``detect``, ``get_probabilities``, ``set_alpha``, ``set_max_text_length``, ``set_prior_map``,
``set_verbose``), custom profiles (``load_profile``, ``load_json_profile``, ``add_profile``,
:class:`LangProfile`) and :class:`LangDetectException` with its :data:`ErrorCode` behave as
langdetect 1.0.9's. With the same ``DetectorFactory.seed`` they give the same languages and
the same probabilities, to the last bit: pyroid reproduces langdetect's text clean-up, its
n-grams and its random draws (CPython's Mersenne Twister, ``gauss`` and ``choice``, in
langdetect's order) and every floating-point operation, including ``sum()``, which CPython 3.12
changed to compensated summation. Without a seed, each detection draws from a generator seeded
by the operating system, as langdetect's own ``random.Random`` per detector is; the program's
global ``random`` state is not used in either library.

With langdetect installed, :class:`Language`, :class:`LangDetectException` and
:data:`ErrorCode` are langdetect's own classes, so ``except langdetect.LangDetectException``
and ``isinstance(x, langdetect.language.Language)`` work with pyroid's results.

The default profiles are pyroid's copy of langdetect 1.0.9's 55 (:data:`PROFILES_DIRECTORY`).
langdetect loads them in ``os.listdir`` order, which depends on the file system and decides the
order of :meth:`DetectorFactory.get_lang_list`, of ties and of the floating-point sums; pyroid
uses the order of langdetect's own profile directory when langdetect is installed, and of its
copy otherwise.

Differences from langdetect:

- ``DetectorFactory.word_lang_prob_map`` is a read-only view; changing the lists it returns
  does not change detections (use ``add_profile``).
- Profiles are parsed with ``json``, never ``simplejson``.
- ``Detector`` methods other than the public ones (``_update_lang_prob``, ...) are not hooks:
  overriding them in a subclass does not change ``detect()``.
- Ctrl-C while ``load_profile`` reads a file raises ``KeyboardInterrupt``; langdetect turns it
  into ``LangDetectException``.
- A frequency in a custom profile that is not an int or a float is converted with ``float()``
  after langdetect's arithmetic.
"""

from typing import Final, Literal, overload

from pyroid._pyroid import Column
from pyroid._typing import ArrowStrings
from pyroid.langdetect.detector import Detector as Detector
from pyroid.langdetect.detector_factory import PROFILES_DIRECTORY as PROFILES_DIRECTORY
from pyroid.langdetect.detector_factory import DetectorFactory as DetectorFactory
from pyroid.langdetect.detector_factory import detect as detect
from pyroid.langdetect.detector_factory import detect_langs as detect_langs
from pyroid.langdetect.detector_factory import init_factory as init_factory
from pyroid.langdetect.lang_detect_exception import ErrorCode as ErrorCode
from pyroid.langdetect.lang_detect_exception import LangDetectException as LangDetectException
from pyroid.langdetect.language import Language as Language
from pyroid.langdetect.utils.lang_profile import LangProfile as LangProfile

__all__ = [
    "LANGDETECT_VERSION",
    "PROFILES_DIRECTORY",
    "Detector",
    "DetectorFactory",
    "ErrorCode",
    "LangDetectException",
    "LangProfile",
    "Language",
    "detect",
    "detect_batch",
    "detect_langs",
    "detect_langs_batch",
    "init_factory",
]

#: The langdetect release whose behaviour this module reproduces.
LANGDETECT_VERSION: Final[str]
__version__: Final[str]

@overload
def detect_batch(
    texts: list[str], *, errors: Literal["raise"] = "raise", threads: int | None = None
) -> list[str]:
    """The language of every text, as ``[detect(t) for t in texts]`` gives them.

    Each text is detected as a separate :func:`detect` call would (with the default factory,
    seeded with ``DetectorFactory.seed``), so the results equal the loop's; the texts are
    processed on pyroid's shared thread pool, and Ctrl-C stops the call within a fraction of
    a second.

    Args:
        texts: A list or tuple of str, or an Arrow string array (pyarrow, polars, ...).
        errors: ``"raise"``: the first text langdetect cannot detect (no letters, only a URL)
            raises :class:`LangDetectException`, with a note naming its index; ``"none"``: its
            result is ``None``.
        threads: At most this many tasks at once on pyroid's pool (``None``: all of it; ``1``:
            the calling thread).

    Returns:
        A list (a tuple for a tuple) of language codes, or a :class:`pyroid.Column` of strings
        for Arrow input, where an Arrow null gives a null.

    Raises:
        LangDetectException: With ``errors="raise"``, for a text with no features.
        TypeError: An item is not a str (langdetect's own message, with a note).
        ValueError: `errors` is not ``"raise"`` or ``"none"``.
    """

@overload
def detect_batch(
    texts: list[str], *, errors: Literal["none"], threads: int | None = None
) -> list[str | None]: ...
@overload
def detect_batch(
    texts: tuple[str, ...],
    *,
    errors: Literal["raise", "none"] = "raise",
    threads: int | None = None,
) -> tuple[str | None, ...]: ...
@overload
def detect_batch(
    texts: ArrowStrings, *, errors: Literal["raise", "none"] = "raise", threads: int | None = None
) -> Column[str]: ...
@overload
def detect_langs_batch(
    texts: list[str], *, errors: Literal["raise"] = "raise", threads: int | None = None
) -> list[list[Language]]:
    """The languages and probabilities of every text, as ``[detect_langs(t) for t in texts]``.

    Args:
        texts: A list or tuple of str, or an Arrow string array.
        errors: As for :func:`detect_batch`.
        threads: As for :func:`detect_batch`.

    Returns:
        One list of :class:`Language` per text, in a list (a tuple for a tuple); ``None`` for an
        Arrow null, and with ``errors="none"`` for a text with no features.

    Raises:
        As for :func:`detect_batch`.
    """

@overload
def detect_langs_batch(
    texts: list[str] | ArrowStrings,
    *,
    errors: Literal["raise", "none"] = "raise",
    threads: int | None = None,
) -> list[list[Language] | None]: ...
@overload
def detect_langs_batch(
    texts: tuple[str, ...],
    *,
    errors: Literal["raise", "none"] = "raise",
    threads: int | None = None,
) -> tuple[list[Language] | None, ...]: ...
