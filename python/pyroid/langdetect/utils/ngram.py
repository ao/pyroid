# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""langdetect's ``NGram``: the n-gram buffer and character normalisation (in Rust).

``normalize`` and ``normalize_vi`` run in Rust while ``NGram``'s tables are langdetect's;
with ``LATIN1_EXCLUDED``, ``CJK_MAP`` or the Vietnamese tables changed they run langdetect's
Python code on the changed tables.
"""

from __future__ import annotations

import gc
import operator
import re
import types
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
    """The last three normalised characters of a text, as langdetect's ``NGram``.

    The class attributes are langdetect's tables. Changing one (``N_GRAM``,
    ``LATIN1_EXCLUDED``, an entry of ``CJK_MAP``, ...) changes the detections as in
    langdetect, which then run langdetect's Python algorithm (langdetect's speed).
    """

    LATIN1_EXCLUDED = _tables.LATIN1_EXCLUDED
    N_GRAM = 3

    def __init__(self) -> None:
        self.grams = " "
        self.capitalword = False

    def add_char(self, ch: str) -> None:
        """Append one character."""
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
        """Fold a Vietnamese letter and a following combining mark into one character."""
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
_VI_BASE = NGram.TO_NORMALIZE_VI_CHARS
_VI_MARKS = NGram.DMARK_CLASS
_VI_TABLE = list(NGram.NORMALIZED_VI_CHARS)
_VI_RE = NGram.ALPHABET_WITH_DMARK
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
    pattern = cls.ALPHABET_WITH_DMARK
    return (
        cls.TO_NORMALIZE_VI_CHARS == _VI_BASE
        and cls.DMARK_CLASS == _VI_MARKS
        and cls.NORMALIZED_VI_CHARS == _VI_TABLE  # a copy: a change in place is seen
        and (pattern is _VI_RE or pattern == _VI_RE)
    )


def _stock() -> bool:
    """``NGram`` is as langdetect defines it: what the native n-gram extraction reproduces.

    It runs before every native detection: one flat expression, no calls in the usual case.
    """
    try:
        n = NGram.N_GRAM
        d = NGram.__dict__
        return (
            type(n) is int
            and n == 3
            and _TABLES(NGram) == _STOCK_TABLES  # strs and the pattern by value
            and NGram.CJK_MAP is _CJK_MAP
            and (not _CJK_MAP.changed or _cjk_stock(_CJK_MAP))
            and NGram.NORMALIZED_VI_CHARS == _VI_TABLE  # a copy: a change in place is seen
            and d["normalize"] is _NORMALIZE
            and d["normalize_vi"] is _VI_FUNC
        )
    except Exception:  # a missing attribute: langdetect's code raises it
        return False


_NORMALIZE = NGram.__dict__["normalize"]
_TABLES = operator.attrgetter(
    "LATIN1_EXCLUDED",
    "TO_NORMALIZE_VI_CHARS",
    "DMARK_CLASS",
    "ALPHABET_WITH_DMARK",
    "__init__",
    "add_char",
)
_STOCK_TABLES = _TABLES(NGram)
_INIT = NGram.__dict__["__init__"]
_ADD_CHAR = NGram.__dict__["add_char"]

# The modes of a native check (pyroid._pyroid.langdetect._Checks), and the flag of an owner read
# as a dict.
_IS, _EQ, _INT, _FUNC, _LEN = range(5)
_DICT = 8


def _own(owner: Any) -> Any:
    """The dict a native check reads `owner`'s attributes from: a class's own, a module's.

    A class's own ``__dict__`` (a mapping proxy's mapping) holds what ``getattr`` finds for the
    classes checked here (they derive from ``object`` only), and one lookup in it is cheaper.
    """
    if isinstance(owner, types.ModuleType):
        return vars(owner)
    if isinstance(owner, type):
        refs = gc.get_referents(owner.__dict__)
        if len(refs) == 1 and type(refs[0]) is dict and refs[0] == dict(owner.__dict__):
            return refs[0]
    return None


def _items(items: list[tuple[Any, str, Any, int]]) -> list[tuple[Any, str, Any, int]]:
    """`items` for a native check list, each class or module owner read from its dict."""
    out = []
    for owner, name, expected, mode in items:
        d = _own(owner)
        out.append(
            (owner, name, expected, mode) if d is None else (d, name, expected, mode | _DICT)
        )
    return out


def _stock_items() -> list[tuple[Any, str, Any, int]]:
    """`_stock`'s checks as ``(owner, attribute, expected, mode)``, for a native check list."""
    return [
        (NGram, "N_GRAM", 3, _INT),
        (NGram, "LATIN1_EXCLUDED", _LATIN1, _EQ),
        (NGram, "CJK_MAP", _CJK_MAP, _IS),
        (_CJK_MAP, "changed", False, _IS),
        (NGram, "NORMALIZED_VI_CHARS", _VI_TABLE, _EQ),
        (NGram, "TO_NORMALIZE_VI_CHARS", _VI_BASE, _EQ),
        (NGram, "DMARK_CLASS", _VI_MARKS, _EQ),
        (NGram, "ALPHABET_WITH_DMARK", _VI_RE, _EQ),
        (NGram, "__init__", _INIT, _IS),
        (NGram, "add_char", _ADD_CHAR, _IS),
        (NGram, "normalize", _NORMALIZE.__func__, _FUNC),
        (NGram, "normalize_vi", _VI_FUNC.__func__, _FUNC),
    ]
