# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""langdetect's ``NGram``: the n-gram buffer and character normalisation (in Rust).

``normalize`` and ``normalize_vi`` run in Rust while ``NGram``'s tables are langdetect's;
with ``LATIN1_EXCLUDED``, ``CJK_MAP`` or the Vietnamese tables changed they run langdetect's
Python code on the changed tables.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

from pyroid._pyroid import langdetect as _native
from pyroid.langdetect.utils import _tables

__all__ = ["NGram"]


class _CjkMap(dict[str, str]):
    """``NGram.CJK_MAP``: a dict that records that it was changed (then normalisation reads it)."""

    __slots__ = ("changed",)

    def __init__(self) -> None:
        super().__init__()
        self.changed = False

    # Each sets the flag after the change, so a detection that starts after it sees it.

    def __setitem__(self, key: Any, value: Any) -> None:
        try:
            super().__setitem__(key, value)
        finally:
            self.changed = True

    def __delitem__(self, key: Any) -> None:
        try:
            super().__delitem__(key)
        finally:
            self.changed = True

    def __ior__(self, other: Any) -> Any:
        try:
            return super().__ior__(other)
        finally:
            self.changed = True

    def clear(self) -> None:
        try:
            super().clear()
        finally:
            self.changed = True

    def pop(self, *args: Any) -> Any:
        try:
            return super().pop(*args)
        finally:
            self.changed = True

    def popitem(self) -> Any:
        try:
            return super().popitem()
        finally:
            self.changed = True

    def setdefault(self, *args: Any) -> Any:
        try:
            return super().setdefault(*args)
        finally:
            self.changed = True

    def update(self, *args: Any, **kwargs: Any) -> None:
        try:
            super().update(*args, **kwargs)
        finally:
            self.changed = True

    def __reduce__(self) -> Any:
        return dict, (dict(self),)  # pickles and copies as a plain dict


class NGram:
    """The last three normalised characters of a text, as langdetect's ``NGram``."""

    LATIN1_EXCLUDED = _tables.LATIN1_EXCLUDED
    N_GRAM = 3

    def __init__(self) -> None:
        self.grams = " "
        self.capitalword = False

    def add_char(self, ch: str) -> None:
        """Append a character into the n-gram buffer."""
        ch = self.normalize(ch)
        last_char = self.grams[-1]
        if last_char == " ":
            self.grams = " "
            self.capitalword = False
            if ch == " ":
                return
        elif len(self.grams) >= self.N_GRAM:
            self.grams = self.grams[1:]
        self.grams += ch

        if ch.isupper():
            if last_char.isupper():
                self.capitalword = True
        else:
            self.capitalword = False

    def get(self, n: int) -> str | None:
        """The last `n` characters (`n` = 1, 2 or 3), or ``None``."""
        if self.capitalword:
            return None
        if n < 1 or n > self.N_GRAM or len(self.grams) < n:
            return None
        if n == 1:
            ch = self.grams[-1]
            if ch == " ":
                return None
            return ch
        return self.grams[-n:]

    @classmethod
    def normalize(cls, ch: Any) -> str:
        """One character as langdetect normalises it."""
        if not isinstance(ch, str):
            ord(ch)  # raises what langdetect raises
        try:
            stock = cls.LATIN1_EXCLUDED == _LATIN1 and _cjk_stock(cls.CJK_MAP)
        except Exception:  # a missing table: langdetect's code raises it
            stock = False
        if stock:
            return _native.normalize(ch)  # type: ignore[no-any-return]
        # langdetect's code for the two blocks whose tables are class attributes.
        cp = ord(ch)
        if 0x80 <= cp <= 0xFF:  # Latin-1 Supplement
            if cls.LATIN1_EXCLUDED.find(ch) >= 0:
                ch = " "
            return ch  # type: ignore[no-any-return]
        if 0x4E00 <= cp <= 0x9FFF:  # CJK Unified Ideographs
            return cls.CJK_MAP.get(ch, ch)
        return _native.normalize(ch)  # type: ignore[no-any-return]

    @classmethod
    def normalize_vi(cls, text: Any) -> str:
        """Fold a Vietnamese letter followed by a combining mark into one character."""
        try:
            stock = isinstance(text, str) and _vi_stock(cls)
        except Exception:
            stock = False
        if not stock:

            def repl(m: Any) -> Any:
                alphabet = cls.TO_NORMALIZE_VI_CHARS.find(m.group(1))
                dmark = cls.DMARK_CLASS.find(m.group(2))  # Diacritical Mark
                return cls.NORMALIZED_VI_CHARS[dmark][alphabet]

            return cls.ALPHABET_WITH_DMARK.sub(repl, text)  # type: ignore[no-any-return]
        return _native.normalize_vi(text)  # type: ignore[no-any-return]

    NORMALIZED_VI_CHARS: ClassVar[list[str]] = list(_tables.NORMALIZED_VI_CHARS)
    TO_NORMALIZE_VI_CHARS = _tables.TO_NORMALIZE_VI_CHARS
    DMARK_CLASS = _tables.DMARK_CLASS
    ALPHABET_WITH_DMARK = re.compile(
        "([" + TO_NORMALIZE_VI_CHARS + "])([" + DMARK_CLASS + "])", re.UNICODE
    )

    # CJK Kanji Normalization Mapping
    CJK_CLASS: ClassVar[list[str]] = list(_tables.CJK_CLASS)
    CJK_MAP: ClassVar[dict[str, str]] = _CjkMap()

    @classmethod
    def _init_cjk_map(cls) -> None:
        for cjk_list in cls.CJK_CLASS:
            representative = cjk_list[0]
            for ch in cjk_list:
                cls.CJK_MAP[ch] = representative


NGram._init_cjk_map()
_CJK_MAP: Any = NGram.CJK_MAP
_CJK_MAP.changed = False
_CJK_PRISTINE = dict(_CJK_MAP)
_LATIN1 = NGram.LATIN1_EXCLUDED
_VI = (
    NGram.TO_NORMALIZE_VI_CHARS,
    NGram.DMARK_CLASS,
    list(NGram.NORMALIZED_VI_CHARS),
    NGram.ALPHABET_WITH_DMARK,
)
_VI_FUNC = NGram.__dict__["normalize_vi"]


def _cjk_stock(cjk_map: Any) -> bool:
    """`cjk_map` is pyroid's ``CJK_MAP``, unchanged (the native normalisation's table)."""
    if cjk_map is not _CJK_MAP:
        return False
    if not _CJK_MAP.changed:
        return True
    if dict.__eq__(_CJK_MAP, _CJK_PRISTINE) is True:  # changed back
        _CJK_MAP.changed = False
        return True
    return False


def _vi_stock(cls: Any) -> bool:
    """``cls``'s Vietnamese tables are langdetect's (the native ``normalize_vi``'s)."""
    return (
        cls.TO_NORMALIZE_VI_CHARS,
        cls.DMARK_CLASS,
        cls.NORMALIZED_VI_CHARS,
        cls.ALPHABET_WITH_DMARK,
    ) == _VI


def _stock() -> bool:
    """``NGram`` is as langdetect defines it: what the native n-gram extraction reproduces."""
    try:
        return _as_defined()
    except Exception:  # a missing attribute: langdetect's code raises it
        return False


def _as_defined() -> bool:
    n = NGram.N_GRAM
    d = NGram.__dict__
    return (
        type(n) is int
        and n == 3
        and NGram.LATIN1_EXCLUDED == _LATIN1
        and _cjk_stock(NGram.CJK_MAP)
        and d.get("__init__") is _INIT
        and d.get("add_char") is _ADD_CHAR
        and d.get("normalize") is _NORMALIZE
        and _vi_stock(NGram)
        and d.get("normalize_vi") is _VI_FUNC
    )


_INIT = NGram.__dict__["__init__"]
_ADD_CHAR = NGram.__dict__["add_char"]
_NORMALIZE = NGram.__dict__["normalize"]
