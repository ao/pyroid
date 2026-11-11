# Derived from NLTK 3.10.4 (Apache-2.0), © 2001-2026 NLTK Project; modified.
"""The Treebank word tokenizers and detokenizer (``nltk.tokenize.treebank`` / ``destructive``)."""

from __future__ import annotations

import importlib.util
import re
import threading
import warnings
from collections.abc import Iterator
from typing import Any, ClassVar

from pyroid._pyroid import nltk as _native
from pyroid.nltk import _delegate
from pyroid.nltk._punkt import UNICODE_VERSION

# NLTK's pattern tables, as sources (pattern, flags, replacement). NLTK compiles them with the
# `regex` module; the class attributes below do too when it is installed (else with `re`).
_U = re.UNICODE
_SOURCES: dict[str, dict[str, Any]] = {
    "treebank": {
        "STARTING_QUOTES": [
            (r"^\"", 0, r"``"),
            (r"(``)", 0, r" \1 "),
            (r"([ \(\[{<])(\"|\'{2})", 0, r"\1 `` "),
        ],
        "PUNCTUATION": [
            (r"([:,])([^\d])", 0, r" \1 \2"),
            (r"([:,])$", 0, r" \1 "),
            (r"\.\.\.", 0, r" ... "),
            (r"[;@#$%&]", 0, r" \g<0> "),
            (r'([^\.])(\.)([\]\)}>"\']*)\s*$', 0, r"\1 \2\3 "),
            (r"[?!]", 0, r" \g<0> "),
            (r"([^'])' ", 0, r"\1 ' "),
        ],
        "PARENS_BRACKETS": (r"[\]\[\(\)\{\}\<\>]", 0, r" \g<0> "),
        "CONVERT_PARENTHESES": [
            (r"\(", 0, "-LRB-"),
            (r"\)", 0, "-RRB-"),
            (r"\[", 0, "-LSB-"),
            (r"\]", 0, "-RSB-"),
            (r"\{", 0, "-LCB-"),
            (r"\}", 0, "-RCB-"),
        ],
        "DOUBLE_DASHES": (r"--", 0, r" -- "),
        "ENDING_QUOTES": [
            (r"''", 0, " '' "),
            (r'"', 0, " '' "),
            (r"([^' ])('[sS]|'[mM]|'[dD]|') ", 0, r"\1 \2 "),
            (r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) ", 0, r"\1 \2 "),
        ],
    },
    "destructive": {
        "STARTING_QUOTES": [
            ("([\u00ab\u201c\u2018\u201e]|[`]+)", _U, r" \1 "),
            (r"^\"", 0, r"``"),
            (r"(``)", 0, r" \1 "),
            (r"([ \(\[{<])(\"|\'{2})", 0, r"\1 `` "),
            (r"(?i)(?<!\w)(\')(?!(?:re|ve|ll|m|t|s|d|n)\b)(?=\w)", _U, r"\1 "),
        ],
        "ENDING_QUOTES": [
            ("([\u00bb\u201d\u2019])", _U, r" \1 "),
            (r"''", 0, " '' "),
            (r'"', 0, " '' "),
            (r"\s+", 0, " "),
            (r"([^' ])('[sS]|'[mM]|'[dD]|') ", 0, r"\1 \2 "),
            (r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) ", 0, r"\1 \2 "),
        ],
        "PUNCTUATION": [
            (r'([^\.])(\.)([\]\)}>"\'' "\u00bb\u201d\u2019 " r"]*+)\s*$", _U, r"\1 \2 \3 "),
            (r"([:,])([^\d])", 0, r" \1 \2"),
            (r"([:,])$", 0, r" \1 "),
            (r"\.{2,}", _U, r" \g<0> "),
            (r"[;@#$%&]", 0, r" \g<0> "),
            (r"[\u2012-\u2015]", _U, r" \g<0> "),
            (r'([^\.])(\.)([\]\)}>"\']*)\s*$', 0, r"\1 \2\3 "),
            (r"[?!]", 0, r" \g<0> "),
            (r"([^'])' ", 0, r"\1 ' "),
            (r"[*]", _U, r" \g<0> "),
        ],
        "PARENS_BRACKETS": (r"[\]\[\(\)\{\}\<\>]", 0, r" \g<0> "),
        "CONVERT_PARENTHESES": [
            (r"\(", 0, "-LRB-"),
            (r"\)", 0, "-RRB-"),
            (r"\[", 0, "-LSB-"),
            (r"\]", 0, "-RSB-"),
            (r"\{", 0, "-LCB-"),
            (r"\}", 0, "-RCB-"),
        ],
        "DOUBLE_DASHES": (r"--", 0, r" -- "),
    },
}
_CONTRACTIONS2 = [
    r"(?i)\b(can)(?#X)(not)\b",
    r"(?i)\b(d)(?#X)('ye)\b",
    r"(?i)\b(gim)(?#X)(me)\b",
    r"(?i)\b(gon)(?#X)(na)\b",
    r"(?i)\b(got)(?#X)(ta)\b",
    r"(?i)\b(lem)(?#X)(me)\b",
    r"(?i)\b(more)(?#X)('n)\b",
    r"(?i)\b(wan)(?#X)(na)(?=\s)",
]
_CONTRACTIONS3 = [r"(?i) ('t)(?#X)(is)\b", r"(?i) ('t)(?#X)(was)\b"]
_CONTRACTIONS4 = [r"(?i)\b(whad)(dd)(ya)\b", r"(?i)\b(wha)(t)(cha)\b"]


class MacIntyreContractions:
    """List of contractions adapted from Robert MacIntyre's tokenizer (NLTK's)."""

    CONTRACTIONS2: ClassVar[list[str]] = list(_CONTRACTIONS2)
    CONTRACTIONS3: ClassVar[list[str]] = list(_CONTRACTIONS3)
    CONTRACTIONS4: ClassVar[list[str]] = list(_CONTRACTIONS4)


def _engine() -> Any:
    return __import__("regex") if importlib.util.find_spec("regex") is not None else re


class _Patterns:
    """A class attribute holding one of NLTK's pattern tables, compiled on first access."""

    def __init__(self, table: str, name: str) -> None:
        self.table, self.name = table, name
        self.value: Any = None
        self.snapshot: Any = None
        self._lock = threading.Lock()

    def mutated(self) -> bool:
        """Whether the list was changed in place (`.append`, item assignment, ...)."""
        value = self.value  # published after the snapshot (see __get__)
        if not isinstance(value, list):
            return False
        return len(value) != len(self.snapshot) or any(
            a is not b for a, b in zip(value, self.snapshot, strict=True)
        )

    def __get__(self, obj: Any, owner: Any = None) -> Any:
        if self.value is None:
            # One thread compiles. Without the lock, two first accesses could each store a list and
            # leave the snapshot of one beside the other, and mutated() would then stay True.
            with self._lock:
                if self.value is None:
                    value = self._compile()
                    self.snapshot = tuple(value) if isinstance(value, list) else None
                    self.value = value
        return self.value

    def _compile(self) -> Any:
        if self.name.startswith("CONTRACTIONS"):
            srcs = _CONTRACTIONS2 if self.name == "CONTRACTIONS2" else _CONTRACTIONS3
            if self.table == "detok":
                eng = _engine()
                return [eng.compile(p.replace("(?#X)", r"\s")) for p in srcs]
            return [re.compile(p) for p in srcs]  # NLTK's tokenizers compile these with re
        eng = _engine()
        src = _SOURCES[self.table][self.name] if self.table != "detok" else _DETOK[self.name]
        if isinstance(src, tuple):
            return (eng.compile(src[0], src[1]), src[2])
        return [(eng.compile(p, f), r) for p, f, r in src]


_DETOK: dict[str, Any] = {
    "ENDING_QUOTES": [
        (r"([^' ])\s('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) ", 0, r"\1\2 "),
        (r"([^' ])\s('[sS]|'[mM]|'[dD]|') ", 0, r"\1\2 "),
        (r"([^'\s])\s(\'\')", 0, r"\1\2"),
        (r"([,.;:!?'])\s+(\"|\'\')", 0, r"\1\2"),
        (r"(\'\')\s([.,:)\]>};%])", 0, r"\1\2"),
        (r"''", 0, '"'),
        (r'([,.;:!?])"(\')', 0, r"\1\2" '"'),
    ],
    "DOUBLE_DASHES": (r" -- ", 0, r"--"),
    "CONVERT_PARENTHESES": [
        ("-LRB-", 0, "("),
        ("-RRB-", 0, ")"),
        ("-LSB-", 0, "["),
        ("-RSB-", 0, "]"),
        ("-LCB-", 0, "{"),
        ("-RCB-", 0, "}"),
    ],
    "PARENS_BRACKETS": [
        (r"([\[\(\{\<])\s", 0, r"\g<1>"),
        (r"\s([\]\)\}\>])", 0, r"\g<1>"),
        (r"([\]\)\}\>])\s([:;,.])", 0, r"\1\2"),
    ],
    "PUNCTUATION": [
        (r"([^'])\s'\s", 0, r"\1' "),
        (r"\s([?!])", 0, r"\g<1>"),
        (r'([^\.])\s(\.)(?!\.)([\]\)}>"\']*)', 0, r"\1\2\3"),
        (r"([#$])\s", 0, r"\g<1>"),
        (r"\s([;%])", 0, r"\g<1>"),
        (r"\s\.\.\.\s", 0, r"..."),
        (r"\s([:,])", 0, r"\1"),
    ],
    "STARTING_QUOTES": [
        (r"([ (\[{<])\s``", 0, r"\1``"),
        (r"(``)\s", 0, r"\1"),
        (r"``", 0, r'"'),
    ],
}

_TOKENIZER_ATTRS = (
    "STARTING_QUOTES",
    "PUNCTUATION",
    "PARENS_BRACKETS",
    "CONVERT_PARENTHESES",
    "DOUBLE_DASHES",
    "ENDING_QUOTES",
    "CONTRACTIONS2",
    "CONTRACTIONS3",
)


def _overridden(obj: Any, base: type, names: tuple[str, ...]) -> list[str]:
    """The pattern attributes `obj`'s class or instance replaces, or that were edited in place."""
    out = [n for n in names if n in getattr(obj, "__dict__", {})]
    for n in names:
        desc = vars(base).get(n)
        if isinstance(desc, _Patterns) and desc.mutated() and n not in out:
            out.append(n)
    for cls in type(obj).__mro__:
        if cls is base:
            break
        out.extend(n for n in names if n in vars(cls) and n not in out)
    return out


def _warn_return_str(stacklevel: int = 3) -> None:
    warnings.warn(
        "Parameter 'return_str' has been deprecated and should no longer be used.",
        category=DeprecationWarning,
        stacklevel=stacklevel,
    )


class _WordTokenizer:
    _flavor_nltk = False
    _nltk_path = ""

    def _twin(self) -> Any:
        """None, or an NLTK tokenizer with this instance's changed patterns."""
        changed = _overridden(self, self._base(), (*_TOKENIZER_ATTRS, "_contractions"))
        if not changed:
            return None
        cls = _delegate.attr(self._nltk_path, f"{type(self).__name__} with changed patterns")
        twin = cls()
        for n in changed:
            setattr(twin, n, getattr(self, n))
        return twin

    def _base(self) -> type:
        raise NotImplementedError

    def tokenize(
        self,
        text: Any,
        convert_parentheses: bool = False,
        return_str: bool = False,
        *,
        threads: int | None = None,
    ) -> Any:
        """Return the tokens of `text` (one list per document for a batch)."""
        if self._flavor_nltk:
            if return_str:
                _warn_return_str()
        elif return_str is not False:
            _warn_return_str()
        twin = self._twin()
        if twin is not None:
            if isinstance(text, str):
                return twin.tokenize(text, convert_parentheses)
            return [
                None if t is None else twin.tokenize(t, convert_parentheses)
                for t in _native.texts(text)
            ]
        return _native.treebank(
            text, self._flavor_nltk, bool(convert_parentheses), UNICODE_VERSION, threads=threads
        )

    def span_tokenize(self, text: str) -> Iterator[tuple[int, int]]:
        """The ``(start, end)`` offsets of the tokens in `text` (a generator, as NLTK's)."""
        twin = self._twin()
        if twin is not None:
            yield from twin.span_tokenize(text)
            return
        yield from _native.treebank_spans(text, self._flavor_nltk, UNICODE_VERSION)

    def tokenize_sents(self, strings: list[str], *, threads: int | None = None) -> list[list[str]]:
        """:meth:`tokenize` of each string."""
        if self._twin() is not None or not isinstance(strings, list):
            return [self.tokenize(s) for s in strings]
        return _native.treebank(  # type: ignore[no-any-return]
            strings, self._flavor_nltk, False, UNICODE_VERSION, threads=threads
        )

    def span_tokenize_sents(self, strings: list[str]) -> Iterator[list[tuple[int, int]]]:
        """:meth:`span_tokenize` of each string, as lists."""
        for s in strings:
            yield list(self.span_tokenize(s))


class TreebankWordTokenizer(_WordTokenizer):
    """NLTK's ``TreebankWordTokenizer`` (the Penn Treebank sed script's rules), in Rust."""

    _nltk_path = "nltk.tokenize.treebank:TreebankWordTokenizer"
    STARTING_QUOTES: Any = _Patterns("treebank", "STARTING_QUOTES")
    PUNCTUATION: Any = _Patterns("treebank", "PUNCTUATION")
    PARENS_BRACKETS: Any = _Patterns("treebank", "PARENS_BRACKETS")
    CONVERT_PARENTHESES: Any = _Patterns("treebank", "CONVERT_PARENTHESES")
    DOUBLE_DASHES: Any = _Patterns("treebank", "DOUBLE_DASHES")
    ENDING_QUOTES: Any = _Patterns("treebank", "ENDING_QUOTES")
    _contractions = MacIntyreContractions()
    CONTRACTIONS2: Any = _Patterns("treebank", "CONTRACTIONS2")
    CONTRACTIONS3: Any = _Patterns("treebank", "CONTRACTIONS3")

    def _base(self) -> type:
        return TreebankWordTokenizer


class NLTKWordTokenizer(_WordTokenizer):
    """NLTK's ``NLTKWordTokenizer``, the improved Treebank rules ``word_tokenize`` uses, in Rust."""

    _flavor_nltk = True
    _nltk_path = "nltk.tokenize.destructive:NLTKWordTokenizer"
    STARTING_QUOTES: Any = _Patterns("destructive", "STARTING_QUOTES")
    ENDING_QUOTES: Any = _Patterns("destructive", "ENDING_QUOTES")
    PUNCTUATION: Any = _Patterns("destructive", "PUNCTUATION")
    PARENS_BRACKETS: Any = _Patterns("destructive", "PARENS_BRACKETS")
    CONVERT_PARENTHESES: Any = _Patterns("destructive", "CONVERT_PARENTHESES")
    DOUBLE_DASHES: Any = _Patterns("destructive", "DOUBLE_DASHES")
    _contractions = MacIntyreContractions()
    CONTRACTIONS2: Any = _Patterns("destructive", "CONTRACTIONS2")
    CONTRACTIONS3: Any = _Patterns("destructive", "CONTRACTIONS3")

    def _base(self) -> type:
        return NLTKWordTokenizer


_DETOK_ATTRS = (
    "CONTRACTIONS2",
    "CONTRACTIONS3",
    "ENDING_QUOTES",
    "DOUBLE_DASHES",
    "CONVERT_PARENTHESES",
    "PARENS_BRACKETS",
    "PUNCTUATION",
    "STARTING_QUOTES",
    "_contractions",
)


class TreebankWordDetokenizer:
    """NLTK's ``TreebankWordDetokenizer``: undo the Treebank tokenizer's padding, in Rust."""

    _contractions = MacIntyreContractions()
    CONTRACTIONS2: Any = _Patterns("detok", "CONTRACTIONS2")
    CONTRACTIONS3: Any = _Patterns("detok", "CONTRACTIONS3")
    ENDING_QUOTES: Any = _Patterns("detok", "ENDING_QUOTES")
    DOUBLE_DASHES: Any = _Patterns("detok", "DOUBLE_DASHES")
    CONVERT_PARENTHESES: Any = _Patterns("detok", "CONVERT_PARENTHESES")
    PARENS_BRACKETS: Any = _Patterns("detok", "PARENS_BRACKETS")
    PUNCTUATION: Any = _Patterns("detok", "PUNCTUATION")
    STARTING_QUOTES: Any = _Patterns("detok", "STARTING_QUOTES")

    def tokenize(self, tokens: list[str], convert_parentheses: bool = False) -> str:
        """Join `tokens` into text, undoing the tokenizer's padding."""
        changed = _overridden(self, TreebankWordDetokenizer, _DETOK_ATTRS)
        if changed:
            cls = _delegate.attr(
                "nltk.tokenize.treebank:TreebankWordDetokenizer",
                "TreebankWordDetokenizer with changed patterns",
            )
            twin = cls()
            for n in changed:
                setattr(twin, n, getattr(self, n))
            return twin.tokenize(tokens, convert_parentheses)  # type: ignore[no-any-return]
        return _native.detokenize(list(tokens), bool(convert_parentheses))  # type: ignore[no-any-return]

    def detokenize(self, tokens: list[str], convert_parentheses: bool = False) -> str:
        """Duck-typing the abstract *tokenize()*."""
        return self.tokenize(tokens, convert_parentheses)


def align_tokens(tokens: list[str], sentence: str) -> list[tuple[int, int]]:
    """The ``(start, end)`` offsets of `tokens`, found in order in `sentence` (NLTK's).

    Raises:
        ValueError: A token is not in `sentence` after the previous one.
    """
    return _native.align_tokens(list(tokens), sentence)  # type: ignore[no-any-return]
