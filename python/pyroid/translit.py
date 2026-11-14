"""Unicode to ASCII transliteration, permissively licensed (``pyroid.translit``).

Batches run in Rust without the GIL, on the anyascii 0.3.3 data (ISC licence);
nothing comes from Unidecode's GPL tables. :func:`anyascii` is byte-identical
to the ``anyascii`` package; :func:`unidecode` has Unidecode's signature and
``errors=`` semantics with anyascii's transliterations. Types and docs are in
``translit.pyi``. anyascii is Copyright (c) 2020-2025, Hunter WB (ISC License).
"""

from typing import Any

from pyroid import _versions
from pyroid._pyroid import PyroidError
from pyroid._pyroid import translit as _native

ANYASCII_VERSION = "0.3.3"

_versions.check(__name__, "anyascii", f"anyascii=={ANYASCII_VERSION}")


class UnidecodeError(PyroidError, ValueError):
    """``unidecode(..., errors="strict")`` met a character without a transliteration.

    Also raised for an invalid ``errors`` value, once a character needs it (as
    Unidecode does).

    Attributes:
        index: Position of the character in the str (``None`` for an invalid
            ``errors`` value).
    """

    def __init__(self, message: str, index: int | None = None) -> None:
        super().__init__(message)
        self.index = index


anyascii = _native.anyascii


def _anyascii_package(string: Any) -> Any:
    """``anyascii.anyascii`` for anything but an exact ``str``, as the package treats it.

    ``pyroid.patch("anyascii")`` uses it through ``_native.anyascii_package``: the package
    returns an object whose ``isascii()`` is true as it is (an ASCII ``bytes``, a ``str``
    subclass), and otherwise joins the transliterations of the characters it iterates over
    (a list or a generator of characters), with its errors (``ord()`` of an int or of a longer
    ``str`` raises ``TypeError``).
    """
    try:
        if string.isascii():
            return string
    except AttributeError:
        pass
    result = []
    for char in string:
        codepoint = ord(char)
        result.append(char if codepoint <= 0x7F else _native.anyascii(chr(codepoint)))
    return "".join(result)

unidecode = _native.unidecode
# Unidecode's two spellings of the same function.
unidecode_expect_ascii = _native.unidecode
unidecode_expect_nonascii = _native.unidecode
is_ascii = _native.is_ascii

__all__ = [
    "ANYASCII_VERSION",
    "UnidecodeError",
    "anyascii",
    "is_ascii",
    "unidecode",
    "unidecode_expect_ascii",
    "unidecode_expect_nonascii",
]
