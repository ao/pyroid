# Derived from NLTK 3.10.4 (Apache-2.0), © 2001-2026 NLTK Project; modified.
"""The regular-expression tokenizers (``nltk.tokenize.regexp``)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from pyroid._pyroid import nltk as _native
from pyroid.nltk import _delegate

_DEFAULT_FLAGS = re.UNICODE | re.MULTILINE | re.DOTALL


class RegexpTokenizer:
    r"""NLTK's ``RegexpTokenizer``: the matches of a pattern, or the text between them.

    Common patterns run in Rust: ``\\w+``, ``\\w+|[^\\w\\s]+``, ``\\w+|\\$[\\d\\.]+|\\S+``,
    ``[\\w']+``, ``\\S+``, ``\\s+``, ``[a-zA-Z]+`` and the blank-line pattern, spelled exactly so,
    with no flags beyond the default ``re.UNICODE | re.MULTILINE | re.DOTALL``. Any other
    pattern runs in NLTK (one :class:`pyroid.FallbackWarning`).

    Args:
        pattern: The pattern (a str, or a compiled pattern whose ``.pattern`` is used).
        gaps: The pattern matches the separators, not the tokens.
        discard_empty: With ``gaps``, drop empty tokens.
        flags: The flags the pattern is compiled with.
    """

    def __init__(
        self,
        pattern: Any,
        gaps: bool = False,
        discard_empty: bool = True,
        flags: int = _DEFAULT_FLAGS,
    ) -> None:
        pattern = getattr(pattern, "pattern", pattern)
        self._pattern = pattern
        self._gaps = gaps
        self._discard_empty = discard_empty
        self._flags = flags
        self._regexp = None

    def _native(self) -> bool:
        p, f = self._pattern, self._flags
        return (
            type(p) is str
            and isinstance(f, int)
            and 0 <= f < 1 << 32
            and _native.regexp_native(p, int(f))
        )

    def _nltk(self) -> Any:
        cls = _delegate.attr(
            "nltk.tokenize.regexp:RegexpTokenizer", "RegexpTokenizer with this pattern or flags"
        )
        return cls(self._pattern, self._gaps, self._discard_empty, self._flags)

    def tokenize(self, text: Any, *, threads: int | None = None) -> Any:
        """The tokens of `text` (one list per document for a batch)."""
        if not self._native():
            twin = self._nltk()
            if isinstance(text, str):
                return twin.tokenize(text)
            return [None if t is None else twin.tokenize(t) for t in _native.texts(text)]
        return _native.regexp_tokenize(
            text,
            self._pattern,
            bool(self._gaps),
            bool(self._discard_empty),
            int(self._flags),
            threads=threads,
        )

    def span_tokenize(self, text: str) -> Iterator[tuple[int, int]]:
        """The ``(start, end)`` offsets of the tokens (a generator, as NLTK's)."""
        if not self._native():
            yield from self._nltk().span_tokenize(text)
            return
        yield from _native.regexp_spans(
            text, self._pattern, bool(self._gaps), bool(self._discard_empty), int(self._flags)
        )

    def tokenize_sents(self, strings: list[str], *, threads: int | None = None) -> list[list[str]]:
        """:meth:`tokenize` of each string."""
        if not isinstance(strings, (list, tuple)):
            return [self.tokenize(s) for s in strings]
        return self.tokenize(list(strings), threads=threads)  # type: ignore[no-any-return]

    def span_tokenize_sents(self, strings: list[str]) -> Iterator[list[tuple[int, int]]]:
        """:meth:`span_tokenize` of each string, as lists."""
        for s in strings:
            yield list(self.span_tokenize(s))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(pattern={self._pattern!r}, gaps={self._gaps!r}, "
            f"discard_empty={self._discard_empty!r}, flags={self._flags!r})"
        )


class WhitespaceTokenizer(RegexpTokenizer):
    r"""Split on whitespace runs (NLTK's ``WhitespaceTokenizer``, ``\\s+`` with ``gaps=True``)."""

    def __init__(self) -> None:
        RegexpTokenizer.__init__(self, r"\s+", gaps=True)


class BlanklineTokenizer(RegexpTokenizer):
    """Split at blank lines (NLTK's ``BlanklineTokenizer``)."""

    def __init__(self) -> None:
        RegexpTokenizer.__init__(self, r"(?:(?<!\s)\s*)?\n\s*\n\s*", gaps=True)


class WordPunctTokenizer(RegexpTokenizer):
    r"""Split into runs of word characters and of other non-space characters (``\w+|[^\w\s]+``)."""

    def __init__(self) -> None:
        RegexpTokenizer.__init__(self, r"\w+|[^\w\s]+")


def regexp_tokenize(
    text: Any,
    pattern: Any,
    gaps: bool = False,
    discard_empty: bool = True,
    flags: int = _DEFAULT_FLAGS,
) -> Any:
    """:meth:`RegexpTokenizer.tokenize` with a tokenizer made from the arguments."""
    return RegexpTokenizer(pattern, gaps, discard_empty, flags).tokenize(text)


blankline_tokenize = BlanklineTokenizer().tokenize
wordpunct_tokenize = WordPunctTokenizer().tokenize
