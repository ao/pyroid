# Type stubs for pyroid.translit. The module docstring is kept here on purpose:
# the API docs are built from this stub.
"""Unicode to ASCII transliteration, permissively licensed (``import pyroid.translit``).

The ``unidecode()`` call you know, without the GPL, batched and fast. Every
function takes a str, a list or tuple of str, or an Arrow string array/column,
returns the same kind, and runs batches on pyroid's thread pool without the
GIL.

- :func:`anyascii` is **byte-identical to the ``anyascii`` package 0.3.3**
  (ISC) on every code point.
- :func:`unidecode` has **Unidecode 1.4.0's signature and ``errors=``
  semantics** (``ignore``, ``replace``, ``strict`` with
  :class:`UnidecodeError` and its ``index``, ``preserve``; the lone-surrogate
  ``RuntimeWarning``) on top of the same anyascii data. Its output therefore
  is **not** Unidecode's: see "Differences from Unidecode".
- :func:`is_ascii` is a mask, ``str.isascii()`` per document.

The data comes from anyascii 0.3.3 (Copyright (c) 2020-2025 Hunter WB, ISC
License), via the ``any_ascii`` Rust crate. Nothing in pyroid is derived from
Unidecode's GPL tables.

Which characters have no transliteration
----------------------------------------
:func:`unidecode`'s ``errors=`` applies to characters anyascii has no mapping
for: unassigned and private-use code points, lone surrogates, and the
characters of scripts or blocks anyascii does not cover (for example most
Tangut, Khitan and some CJK extension ideographs). anyascii's own table lists
the 123,799 non-ASCII code points it maps, 1,192 of them to the empty string
on purpose (combining marks, zero-width joiners, variation selectors, emoji
skin-tone modifiers, Arabic tatweel, silent vowel carriers): those are removed
in every mode and never raise.

Differences from Unidecode
--------------------------
- **Transliterations differ** (anyascii's are not Unidecode's). For example,
  CJK ideographs are joined without spaces (``"北京"`` → ``"BeiJing"``, where
  Unidecode gives ``"Bei Jing "``), emoji become ``:shortcodes:`` (Unidecode
  drops them), and many scripts anyascii covers have no Unidecode mapping.
- **Which characters are missing differs**, so ``errors="strict"`` raises,
  and ``errors="replace"`` substitutes, on different characters. Unidecode
  has no mapping for 80,814 of the 123,799 non-ASCII code points anyascii
  maps (most of the CJK extensions, emoji and the supplementary planes). No
  assigned character has a Unidecode mapping but no anyascii one.
- Lists, tuples and Arrow columns are accepted (Unidecode raises
  ``AttributeError``); other non-str inputs raise :class:`pyroid.InputError`
  (a ``TypeError``) instead of ``AttributeError``.
- For a batch, the first failing document raises, with a note naming it.
- :class:`UnidecodeError` is pyroid's own class (a ``ValueError`` and a
  :class:`pyroid.PyroidError`), not an instance of Unidecode's.
- The surrogate ``RuntimeWarning`` is attributed to the caller's line rather
  than to a line inside the library.

How different the output is (measured against Unidecode 1.4.0, default
``errors="ignore"``, CPython 3.15 / Unicode 17.0):

- Of the 42,985 non-ASCII code points both libraries map, 6,705 (15.6%)
  give identical output, 58.8% after stripping spaces (Unidecode often adds
  a trailing space, e.g. after CJK ideographs).
- Identical documents: FineWeb (English web text) 79.7% (68% of the docs
  that are not pure ASCII); FineWeb-2 French and Spanish 90.4%, German
  75.2%; Arabic, Chinese, Hindi, Japanese and Russian 0%, where the share of
  non-ASCII characters mapped identically is 96% (Russian), 89% (Arabic),
  77% (Hindi), 60% (Japanese) and 6% (Chinese, mostly the trailing spaces).
- The most frequent differences on that text: EM DASH ``"--"`` → ``"-"``,
  Devanagari vowel signs (``"aa"`` → ``"a"``), Cyrillic ``я`` ``"ia"`` →
  ``"ya"``, ``£`` ``"PS"`` → ``"GBP"``, ``®`` ``"(r)"`` → ``"(R)"``.
"""

from typing import Final, overload

from pyroid._pyroid import Column, PyroidError
from pyroid._typing import ArrowStrings

__all__ = [
    "ANYASCII_VERSION",
    "UnidecodeError",
    "anyascii",
    "is_ascii",
    "unidecode",
    "unidecode_expect_ascii",
    "unidecode_expect_nonascii",
]

#: The anyascii data version (the ``anyascii`` package this matches).
ANYASCII_VERSION: Final[str]

class UnidecodeError(PyroidError, ValueError):
    """``unidecode(..., errors="strict")`` met a character without a transliteration.

    Also raised for an invalid ``errors`` value, once a character needs it (as
    Unidecode does).

    Attributes:
        index: Position of the character in the str (``None`` for an invalid
            ``errors`` value).
    """

    index: int | None
    def __init__(self, message: str, index: int | None = None) -> None: ...

@overload
def anyascii(string: str, *, threads: int | None = None) -> str:
    """Transliterate to ASCII, exactly as ``anyascii.anyascii`` 0.3.3 does.

    Each character is replaced by anyascii's mapping (ASCII stays as it is;
    characters it has no mapping for, and lone surrogates, are removed).
    Pure-ASCII documents are returned as the original objects; Arrow nulls stay
    null.

    Args:
        string: A str, a list or tuple of str, or an Arrow string array/column
            (``string``, ``large_string`` or ``string_view``).
        threads: ``None`` uses pyroid's shared thread pool; ``n`` runs at most
            ``n`` tasks at once; ``1`` runs on the calling thread.

    Returns:
        The same kind as ``string``: str, list, tuple, or a
        :class:`pyroid.Column` of strings (``large_string`` for
        ``large_string`` input, else ``string``).

    Raises:
        pyroid.InputError: ``string`` is not one of the accepted kinds, or
            holds a non-str item or a non-string Arrow type.
        UnicodeDecodeError: An Arrow row is not valid UTF-8.
    """

@overload
def anyascii(string: list[str], *, threads: int | None = None) -> list[str]: ...
@overload
def anyascii(string: tuple[str, ...], *, threads: int | None = None) -> tuple[str, ...]: ...
@overload
def anyascii(string: ArrowStrings, *, threads: int | None = None) -> Column[str]: ...
@overload
def unidecode(
    string: str, errors: str = "ignore", replace_str: str = "?", *, threads: int | None = None
) -> str:
    """Unidecode's ``unidecode()``, with anyascii's transliterations.

    ASCII stays as it is; every other character is replaced by anyascii's
    mapping, as in :func:`anyascii`. ``errors`` says what to do with a
    character that has no mapping (see the module docs), as in Unidecode:

    - ``"ignore"`` (default): leave it out.
    - ``"replace"``: put ``replace_str`` in its place.
    - ``"strict"``: raise :class:`UnidecodeError`, whose ``index`` is the
      character's position.
    - ``"preserve"``: keep the character (the result is then not pure ASCII).

    As in Unidecode, ``errors`` and ``replace_str`` are only checked when such
    a character is met, and each lone surrogate gives a ``RuntimeWarning``
    ("Surrogate character ... will be ignored. ...") and then counts as a
    character without a mapping. Pure-ASCII documents are returned as the
    original objects; Arrow nulls stay null.

    Args:
        string: A str, a list or tuple of str, or an Arrow string array/column.
        errors: ``"ignore"``, ``"replace"``, ``"strict"`` or ``"preserve"``.
        replace_str: The replacement for ``errors="replace"``.
        threads: As for :func:`anyascii`.

    Returns:
        The same kind as ``string`` (see :func:`anyascii`).

    Raises:
        UnidecodeError: ``errors="strict"`` and a character has no mapping, or
            ``errors`` is not one of the four values and a character needed
            it. For a batch, the first failing document raises, with a note
            naming it.
        TypeError: ``errors="replace"``, ``replace_str`` is not a str, and a
            character needed it (Unidecode's message).
        pyroid.InputError: As for :func:`anyascii`.
        UnicodeDecodeError: An Arrow row is not valid UTF-8.
        UnicodeEncodeError: Arrow input with a ``replace_str`` holding a lone
            surrogate.
    """

@overload
def unidecode(
    string: list[str],
    errors: str = "ignore",
    replace_str: str = "?",
    *,
    threads: int | None = None,
) -> list[str]: ...
@overload
def unidecode(
    string: tuple[str, ...],
    errors: str = "ignore",
    replace_str: str = "?",
    *,
    threads: int | None = None,
) -> tuple[str, ...]: ...
@overload
def unidecode(
    string: ArrowStrings,
    errors: str = "ignore",
    replace_str: str = "?",
    *,
    threads: int | None = None,
) -> Column[str]: ...

unidecode_expect_ascii = unidecode
unidecode_expect_nonascii = unidecode

@overload
def is_ascii(string: str, *, threads: int | None = None) -> bool:
    """True for each document that is pure ASCII (``str.isascii()``).

    A predicate: it never builds new strings, it returns a mask. A Python str
    knows whether it is ASCII, so str, list and tuple input is answered without
    scanning the text (``threads`` matters only for Arrow input).

    Args:
        string: A str, a list or tuple of str, or an Arrow string array/column.
        threads: As for :func:`anyascii`.

    Returns:
        bool, ``list[bool]``, ``tuple[bool, ...]``, or a :class:`pyroid.Column`
        of bools (nulls stay null), matching ``string``.

    Raises:
        pyroid.InputError: ``string`` has the wrong type.
        UnicodeDecodeError: An Arrow row is not valid UTF-8.
    """

@overload
def is_ascii(string: list[str], *, threads: int | None = None) -> list[bool]: ...
@overload
def is_ascii(string: tuple[str, ...], *, threads: int | None = None) -> tuple[bool, ...]: ...
@overload
def is_ascii(string: ArrowStrings, *, threads: int | None = None) -> Column[bool]: ...
