# Derived from NLTK 3.10.4 (Apache-2.0), © 2001-2026 NLTK Project; modified.
"""Punkt (``nltk.tokenize.punkt``): parameters, language variables and the tokenizers."""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import Any

from pyroid._pyroid import nltk as _native
from pyroid.nltk import _data, _delegate

UNICODE_VERSION = unicodedata.unidata_version

# Orthographic context flags (BEG/MID/UNK position x UC/LC first letter).
_ORTHO_BEG_UC = 1 << 1
_ORTHO_MID_UC = 1 << 2
_ORTHO_UNK_UC = 1 << 3
_ORTHO_BEG_LC = 1 << 4
_ORTHO_MID_LC = 1 << 5
_ORTHO_UNK_LC = 1 << 6
_ORTHO_UC = _ORTHO_BEG_UC + _ORTHO_MID_UC + _ORTHO_UNK_UC
_ORTHO_LC = _ORTHO_BEG_LC + _ORTHO_MID_LC + _ORTHO_UNK_LC
_ORTHO_MAP = {
    ("initial", "upper"): _ORTHO_BEG_UC,
    ("internal", "upper"): _ORTHO_MID_UC,
    ("unknown", "upper"): _ORTHO_UNK_UC,
    ("initial", "lower"): _ORTHO_BEG_LC,
    ("internal", "lower"): _ORTHO_MID_LC,
    ("unknown", "lower"): _ORTHO_UNK_LC,
}

REASON_DEFAULT_DECISION = "default decision"
REASON_KNOWN_COLLOCATION = "known collocation (both words)"
REASON_ABBR_WITH_ORTHOGRAPHIC_HEURISTIC = "abbreviation + orthographic heuristic"
REASON_ABBR_WITH_SENTENCE_STARTER = "abbreviation + frequent sentence starter"
REASON_INITIAL_WITH_ORTHOGRAPHIC_HEURISTIC = "initial + orthographic heuristic"
REASON_NUMBER_WITH_ORTHOGRAPHIC_HEURISTIC = "initial + orthographic heuristic"
REASON_INITIAL_WITH_SPECIAL_ORTHOGRAPHIC_HEURISTIC = "initial + special orthographic heuristic"


# ---------------------------------------------------------------------------
# Parameters: containers that count their changes
# ---------------------------------------------------------------------------


#: Bumped by every change to a pyroid parameter container or PunktParameters attribute, so a
#: tokenizer can tell in O(1) that nothing changed since its last call.
_GEN = [0]


class _Set(set[Any]):
    """A set that counts its changes, so a tokenizer sees in-place edits of its parameters."""

    __slots__ = ("_gen",)

    def __init__(self, items: Iterable[Any] = ()) -> None:
        super().__init__(items)
        self._gen = 0

    def __reduce__(self) -> Any:  # pickles and copies as a plain set
        return (set, (list(self),))


def _counting_set_method(name: str) -> Any:
    base = getattr(set, name)

    def method(self: _Set, *args: Any) -> Any:
        self._gen += 1
        _GEN[0] += 1
        return base(self, *args)

    method.__name__ = name
    return method


for _name in (
    "add",
    "clear",
    "discard",
    "pop",
    "remove",
    "update",
    "difference_update",
    "intersection_update",
    "symmetric_difference_update",
    "__ior__",
    "__iand__",
    "__isub__",
    "__ixor__",
):
    setattr(_Set, _name, _counting_set_method(_name))


class _Ortho(defaultdict[str, int]):
    """``defaultdict(int)`` that counts its changes (a missing key reads as 0, as NLTK's)."""

    __slots__ = ("_gen",)

    def __init__(self, items: Any = ()) -> None:
        super().__init__(int, items)
        self._gen = 0

    def __reduce__(self) -> Any:
        return (defaultdict, (int, dict(self)))


def _counting_dict_method(name: str) -> Any:
    base = getattr(defaultdict, name)

    def method(self: _Ortho, *args: Any, **kwargs: Any) -> Any:
        self._gen += 1
        _GEN[0] += 1
        return base(self, *args, **kwargs)

    method.__name__ = name
    return method


for _name in ("__setitem__", "__delitem__", "clear", "pop", "popitem", "setdefault", "update", "__ior__"):
    setattr(_Ortho, _name, _counting_dict_method(_name))


class PunktParameters:
    """Stores data used to perform sentence boundary detection with Punkt.

    The same attributes as NLTK's: ``abbrev_types``, ``collocations``, ``sent_starters`` (sets)
    and ``ortho_context`` (a ``defaultdict(int)``). Change them in place or assign new ones;
    tokenizers using these parameters see the change at their next call.
    """

    def __init__(self) -> None:
        self.abbrev_types: set[str] = _Set()
        self.collocations: set[tuple[str, str]] = _Set()
        self.sent_starters: set[str] = _Set()
        self.ortho_context: defaultdict[str, int] = _Ortho()

    def __setattr__(self, name: str, value: Any) -> None:
        # A plain set or defaultdict(int) assigned here is stored as a copy that counts its
        # changes, so tokenizers see later in-place edits made through this object.
        if name in _SET_ATTRS and type(value) is set:
            value = _Set(value)
        elif name == "ortho_context" and type(value) is defaultdict and value.default_factory is int:
            value = _Ortho(value)
        object.__setattr__(self, name, value)
        _GEN[0] += 1

    def __setstate__(self, state: dict[str, Any]) -> None:
        for name, value in state.items():
            setattr(self, name, value)

    def clear_abbrevs(self) -> None:
        self.abbrev_types = _Set()

    def clear_collocations(self) -> None:
        self.collocations = _Set()

    def clear_sent_starters(self) -> None:
        self.sent_starters = _Set()

    def clear_ortho_context(self) -> None:
        self.ortho_context = _Ortho()

    def add_ortho_context(self, typ: str, flag: int) -> None:
        self.ortho_context[typ] |= flag

    def _debug_ortho_context(self, typ: str) -> Iterator[str]:
        context = self.ortho_context[typ]
        yield from _ortho_names(context)


def _ortho_names(context: int) -> Iterator[str]:
    for flag, name in (
        (_ORTHO_BEG_UC, "BEG-UC"),
        (_ORTHO_MID_UC, "MID-UC"),
        (_ORTHO_UNK_UC, "UNK-UC"),
        (_ORTHO_BEG_LC, "BEG-LC"),
        (_ORTHO_MID_LC, "MID-LC"),
        (_ORTHO_UNK_LC, "UNK-LC"),
    ):
        if context & flag:
            yield name


_SET_ATTRS = frozenset({"abbrev_types", "collocations", "sent_starters"})


def _adopt(params: Any) -> Any:
    """`params` as a pyroid PunktParameters (a copy of NLTK's or another duck-typed object whose
    tables are sets and a ``defaultdict(int)``), else `params` itself (it then runs in NLTK)."""
    if isinstance(params, PunktParameters):
        return params
    try:
        tables = (params.abbrev_types, params.collocations, params.sent_starters, params.ortho_context)
    except AttributeError:
        return params
    ab, co, ss, oc = tables
    if not all(type(c) is set for c in (ab, co, ss)):
        return params
    if type(oc) is not defaultdict or oc.default_factory is not int:
        return params
    out = PunktParameters()
    out.abbrev_types, out.collocations, out.sent_starters, out.ortho_context = ab, co, ss, oc
    return out


def _from_raw(raw: _data.RawParams) -> PunktParameters:
    params = PunktParameters()
    params.collocations = _Set(raw.collocations)
    params.sent_starters = _Set(raw.sent_starters)
    params.abbrev_types = _Set(raw.abbrev_types)
    params.ortho_context = _Ortho(raw.ortho_context)
    return params


# ---------------------------------------------------------------------------
# Language variables
# ---------------------------------------------------------------------------


class PunktLanguageVars:
    """NLTK's language-dependent variables of the Punkt algorithm.

    Subclass it and change ``sent_end_chars`` (a tuple of one-character strings) to split
    on other characters; that runs in Rust. Overriding anything else (the patterns, or
    ``word_tokenize``) is supported by handing the tokenizer to NLTK, with a
    :class:`pyroid.FallbackWarning`.
    """

    __slots__ = ("_re_period_context", "_re_word_tokenizer")

    def __getstate__(self) -> int:
        return 1

    def __setstate__(self, state: Any) -> int:
        return 1

    sent_end_chars: tuple[str, ...] = (".", "?", "!")
    internal_punctuation = ",:;"

    _re_word_start = r"[^\(\"\`{\[:;&\#\*@\)}\]\-,]"
    _re_multi_char_punct = r"(?:\-{2,}|\.{2,}|(?:\.\s){2,}\.)"
    _word_tokenize_fmt = r"""(
        %(MultiChar)s
        |
        (?=%(WordStart)s)\S+?  # Accept word characters until end is found
        (?= # Sequences marking a word's end
            \s|                                 # White-space
            $|                                  # End-of-string
            %(NonWord)s|%(MultiChar)s|          # Punctuation
            ,(?=$|\s|%(NonWord)s|%(MultiChar)s) # Comma if at end of word
        )
        |
        \S
    )"""
    _period_context_fmt = r"""
        %(SentEndChars)s             # a potential sentence ending
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            \s+(?P<next_tok>\S+)     # or whitespace and some other token
        ))"""

    def word_tokenize(self, s: str) -> list[str]:
        """Tokenize a string to split off punctuation other than periods (NLTK's)."""
        return _nltk_lang_vars(self).word_tokenize(s)  # type: ignore[no-any-return]

    def period_context_re(self) -> Any:
        """NLTK's compiled period-context pattern (from NLTK)."""
        return _nltk_lang_vars(self).period_context_re()


#: Names a PunktLanguageVars subclass may override and still run in Rust.
_NATIVE_LANG_ATTRS = frozenset({"sent_end_chars", "internal_punctuation"})
_PLAIN_CLASS_ATTRS = frozenset(
    {"__module__", "__qualname__", "__doc__", "__slots__", "__dict__", "__weakref__", "__firstlineno__",
     "__static_attributes__", "__annotations__", "__annotate__", "__annotate_func__", "__annotations_cache__",
     "__orig_bases__", "__parameters__"}
)


def _lang_overrides(lang_vars: Any) -> set[str]:
    """The attributes `lang_vars` changes relative to NLTK's default PunktLanguageVars."""
    names: set[str] = set()
    bases = _base_lang_classes()
    for cls in type(lang_vars).__mro__:
        if cls in bases or cls is object:
            break
        names.update(n for n in vars(cls) if n not in _PLAIN_CLASS_ATTRS)
    names.update(getattr(lang_vars, "__dict__", {}))
    return names


def _base_lang_classes() -> tuple[type, ...]:
    import sys

    real = sys.modules.get("nltk.tokenize.punkt")
    real = getattr(real, "__wrapped__", real)
    nltk_cls = _delegate.ORIGINALS.get("nltk.tokenize.punkt:PunktLanguageVars") or getattr(
        real, "PunktLanguageVars", None
    )
    return (PunktLanguageVars, nltk_cls) if isinstance(nltk_cls, type) else (PunktLanguageVars,)


def _native_end_chars(lang_vars: Any) -> str | None:
    """``lang_vars.sent_end_chars`` as a str of distinct chars, or None if Rust cannot run it."""
    if lang_vars is None:
        return ".?!"
    if not isinstance(lang_vars, _base_lang_classes()) or _lang_overrides(lang_vars) - _NATIVE_LANG_ATTRS:
        return None
    chars = lang_vars.sent_end_chars
    if not isinstance(chars, (tuple, list)) or not all(type(c) is str and len(c) == 1 for c in chars):
        return None
    if not chars:  # NLTK's pattern "[]" does not compile: let NLTK raise its error
        return None
    return "".join(chars)


def _nltk_lang_vars(lang_vars: Any) -> Any:
    """An NLTK PunktLanguageVars equal to `lang_vars` (for the NLTK fallback)."""
    base = _delegate.attr("nltk.tokenize.punkt:PunktLanguageVars", "PunktLanguageVars", warn=False)
    if isinstance(lang_vars, base):
        return lang_vars
    overrides = {
        n: getattr(type(lang_vars), n)
        for n in _lang_overrides(lang_vars) - set(getattr(lang_vars, "__dict__", {}))
    }
    obj = type(type(lang_vars).__name__, (base,), overrides)()
    for n, v in getattr(lang_vars, "__dict__", {}).items():
        setattr(obj, n, v)
    return obj


# ---------------------------------------------------------------------------
# The tokenizers
# ---------------------------------------------------------------------------

_DEFAULT_TOKEN = object()


def _fingerprint(params: Any) -> tuple[Any, ...] | None:
    """What identifies `params`' contents cheaply, or None if Rust cannot use them."""
    try:
        ab, co, ss, oc = params.abbrev_types, params.collocations, params.sent_starters, params.ortho_context
    except AttributeError:
        return None
    if not all(isinstance(c, (set, frozenset)) for c in (ab, co, ss)):
        return None
    if not isinstance(oc, defaultdict) or oc.default_factory is not int:
        return None
    return tuple(x for c in (ab, co, ss, oc) for x in (id(c), len(c), getattr(c, "_gen", 0)))


def _native_params(params: Any, end_chars: str) -> Any:
    """A native Punkt for `params`, or None if they hold values Rust cannot represent."""
    oc = []
    for k, v in params.ortho_context.items():
        if type(k) is not str or not isinstance(v, int):
            return None
        oc.append((k, v & 0x7F))
    return _native._Punkt(
        [s for s in params.abbrev_types if type(s) is str],
        [
            c
            for c in params.collocations
            if type(c) is tuple and len(c) == 2 and type(c[0]) is str and type(c[1]) is str
        ],
        [s for s in params.sent_starters if type(s) is str],
        oc,
        end_chars,
        UNICODE_VERSION,
    )


def _per_doc(text: Any, fn: Any) -> Any:
    """`fn(text)` for a str; for a batch, `fn` per document (``None`` for an Arrow null)."""
    if isinstance(text, str):
        return fn(text)
    return [None if t is None else fn(t) for t in _native.texts(text)]


def _raise(err: tuple[str, str] | None) -> None:
    """Raise the exception NLTK's generator raises at this point (``(type name, message)``)."""
    if err is not None:
        kind, msg = err
        raise (IndexError if kind == "IndexError" else AssertionError)(*((msg,) if msg else ()))


class PunktSentenceTokenizer:
    """NLTK's ``PunktSentenceTokenizer``, running in Rust.

    Args:
        train_text: Training text (trained with NLTK's ``PunktTrainer``, a
            :class:`pyroid.FallbackWarning`), or a :class:`PunktParameters` (pyroid's or
            NLTK's) to use. ``None`` or empty: no parameters.
        verbose: Passed to the trainer.
        lang_vars: A :class:`PunktLanguageVars` (pyroid's or NLTK's).
        token_cls: NLTK's ``PunktToken`` or a subclass; anything but the default runs in NLTK.
    """

    def __init__(
        self,
        train_text: Any = None,
        verbose: bool = False,
        lang_vars: Any = None,
        token_cls: Any = _DEFAULT_TOKEN,
    ) -> None:
        if lang_vars is None:
            lang_vars = PunktLanguageVars()
        if not isinstance(self, PunktSentenceTokenizer):
            # NLTK's own PunktTokenizer.__init__ calls the module's PunktSentenceTokenizer,
            # which is this class while pyroid.patch("nltk") is active: run NLTK's.
            original = _delegate.ORIGINALS.get("nltk.tokenize.punkt:PunktSentenceTokenizer")
            if original is None:
                raise TypeError(f"{type(self).__name__} is not a PunktSentenceTokenizer")
            kwargs = {} if token_cls is _DEFAULT_TOKEN else {"token_cls": token_cls}
            original.__init__(self, train_text, verbose, lang_vars, **kwargs)
            return
        self._lang_vars = lang_vars
        self._Token = token_cls
        self._params = PunktParameters()
        self._cache: tuple[Any, Any] | None = None  # (key, native Punkt or None)
        self._fast: tuple[Any, ...] | None = None  # what _impl checks first
        if train_text:
            self._params = self.train(train_text, verbose)

    @property
    def _params(self) -> Any:
        return self.__dict__["_params_"]

    @_params.setter
    def _params(self, value: Any) -> None:
        self.__dict__["_params_"] = _adopt(value)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache"] = None  # the native tokenizer is rebuilt on first use
        state["_fast"] = None
        return state

    def train(self, train_text: Any, verbose: bool = False) -> Any:
        """Derive parameters from `train_text` (with NLTK's trainer), or return it unchanged."""
        if not isinstance(train_text, str):
            return train_text
        trainer = _delegate.attr("nltk.tokenize.punkt:PunktTrainer", "PunktTrainer (training)")
        return trainer(
            train_text, lang_vars=_nltk_lang_vars(self._lang_vars), token_cls=self._nltk_token()
        ).get_params()

    def _nltk_token(self) -> Any:
        if self._Token is _DEFAULT_TOKEN:
            return _delegate.attr("nltk.tokenize.punkt:PunktToken", "PunktToken", warn=False)
        return self._Token

    # -- native or NLTK -------------------------------------------------------------------

    def _impl(self) -> Any:
        """The native Punkt for the current parameters, or None (then NLTK runs)."""
        params = self.__dict__["_params_"]
        lang_vars = self._lang_vars
        fast = self._fast
        if (
            fast is not None
            and fast[0] is params
            and fast[1] is lang_vars
            and fast[2] == _GEN[0]
            and fast[3] is getattr(lang_vars, "sent_end_chars", None)
            and fast[4] is self._Token
        ):
            return fast[5]
        native = self._impl_slow(params)
        tracked = isinstance(params, PunktParameters) and all(
            isinstance(c, (_Set, _Ortho, frozenset))
            for c in (params.abbrev_types, params.collocations, params.sent_starters, params.ortho_context)
        )
        self._fast = (
            (params, lang_vars, _GEN[0], getattr(lang_vars, "sent_end_chars", None), self._Token, native)
            if tracked and not getattr(lang_vars, "__dict__", None)
            else None
        )
        return native

    def _impl_slow(self, params: Any) -> Any:
        end_chars = _native_end_chars(self._lang_vars) if self._Token is _DEFAULT_TOKEN else None
        fp = _fingerprint(params) if end_chars is not None else None
        key = (id(params), fp, end_chars)
        cache = self._cache
        if cache is not None and cache[0] == key:
            return cache[1]
        native = None if fp is None or end_chars is None else _native_params(params, end_chars)
        self._cache = (key, native)
        return native

    def _nltk(self, what: str) -> Any:
        """An NLTK PunktSentenceTokenizer with these parameters (one FallbackWarning per reason)."""
        cls = _delegate.attr("nltk.tokenize.punkt:PunktSentenceTokenizer", what)
        twin = cls(lang_vars=_nltk_lang_vars(self._lang_vars), token_cls=self._nltk_token())
        twin._params = self._params
        return twin

    def _why(self) -> str:
        if self._Token is not _DEFAULT_TOKEN:
            return "PunktSentenceTokenizer with a custom token_cls"
        if _native_end_chars(self._lang_vars) is None:
            return "PunktSentenceTokenizer with custom PunktLanguageVars patterns"
        return "PunktSentenceTokenizer with parameters that are not sets of str"

    # -- tokenization ---------------------------------------------------------------------

    def tokenize(
        self, text: Any, realign_boundaries: bool = True, *, threads: int | None = None
    ) -> Any:
        """Split `text` into sentences (a list of lists for a batch)."""
        native = self._impl()
        if native is None:
            twin = self._nltk(self._why())
            return _per_doc(text, lambda t: twin.tokenize(t, realign_boundaries))
        return native.sentences(text, realign_boundaries, threads=threads)

    def sentences_from_text(
        self, text: Any, realign_boundaries: bool = True, *, threads: int | None = None
    ) -> Any:
        """:meth:`tokenize`."""
        return self.tokenize(text, realign_boundaries, threads=threads)

    def span_tokenize(
        self, text: Any, realign_boundaries: bool = True, *, threads: int | None = None
    ) -> Any:
        """The ``(start, end)`` offsets of the sentences: a generator for a str, as NLTK's
        (a list of lists of tuples for a batch)."""
        if isinstance(text, str):
            return self._span_gen(text, realign_boundaries)
        native = self._impl()
        if native is None:
            twin = self._nltk(self._why())
            return _per_doc(text, lambda t: list(twin.span_tokenize(t, realign_boundaries)))
        return native.spans(text, realign_boundaries, threads=threads)

    def _span_gen(self, text: str, realign_boundaries: bool) -> Iterator[tuple[int, int]]:
        native = self._impl()
        if native is None:
            yield from self._nltk(self._why()).span_tokenize(text, realign_boundaries)
            return
        yield from native.spans(text, realign_boundaries)

    def tokenize_sents(self, strings: list[str], *, threads: int | None = None) -> list[list[str]]:
        """:meth:`tokenize` of each string."""
        native = self._impl()
        if native is None or not isinstance(strings, list):
            return [self.tokenize(s) for s in strings]
        return native.sentences(strings, True, threads=threads)  # type: ignore[no-any-return]

    def span_tokenize_sents(
        self, strings: list[str], *, threads: int | None = None
    ) -> Iterator[list[tuple[int, int]]]:
        """:meth:`span_tokenize` of each string, as lists."""
        for s in strings:
            yield list(self.span_tokenize(s))

    def text_contains_sentbreak(self, text: str) -> bool:
        """Whether `text` contains a sentence break (one that is not at its last token)."""
        native = self._impl()
        if native is None:
            return bool(self._nltk(self._why()).text_contains_sentbreak(text))
        return native.contains_sentbreak(text)  # type: ignore[no-any-return]

    def sentences_from_text_legacy(self, text: str) -> Iterator[str]:
        """The sentences, found by annotating every token (NLTK's legacy method)."""
        native = self._impl()
        if native is None:
            yield from self._nltk(self._why()).sentences_from_text_legacy(text)
            return
        items, err = native.legacy(text)
        yield from items
        _raise(err)

    def sentences_from_tokens(self, tokens: Iterable[str]) -> Iterator[list[str]]:
        """Group a sequence of word tokens into sentences (lists of tokens)."""
        native = self._impl()
        toks = list(tokens)
        if native is None or not all(type(t) is str for t in toks):
            yield from self._nltk(self._why()).sentences_from_tokens(toks)
            return
        lengths, err = native.sentences_from_tokens(toks)
        i = 0
        for n in lengths:
            yield toks[i : i + n]
            i += n
        _raise(err)

    def debug_decisions(self, text: str) -> Iterator[dict[str, Any]]:
        """One dict per candidate sentence break, saying why it was decided as it was."""
        native = self._impl()
        if native is None:
            yield from self._nltk(self._why()).debug_decisions(text)
            return
        decisions, err = native.debug_decisions(text)
        for (
            period_index,
            ctx,
            type1,
            type2,
            in_abbrs,
            is_initial,
            is_starter,
            ortho,
            contexts,
            colloc,
            reason,
            decision,
        ) in decisions:
            yield {
                "period_index": period_index,
                "text": ctx,
                "type1": type1,
                "type2": type2,
                "type1_in_abbrs": in_abbrs,
                "type1_is_initial": is_initial,
                "type2_is_sent_starter": is_starter,
                "type2_ortho_heuristic": ortho,
                "type2_ortho_contexts": set(_ortho_names(contexts)),
                "collocation": colloc,
                "reason": reason,
                "break_decision": decision,
            }
        _raise(err)

    def dump(self, tokens: Any) -> str:
        """NLTK's debugging dump (runs in NLTK)."""
        return self._nltk("PunktSentenceTokenizer.dump").dump(tokens)  # type: ignore[no-any-return]

    PUNCTUATION = tuple(";:,.!?")


class PunktTokenizer(PunktSentenceTokenizer):
    """A :class:`PunktSentenceTokenizer` with the parameters of an installed ``punkt_tab`` model.

    Args:
        lang: The model's name (``"english"``, ``"german"``, ...).

    Raises:
        TypeError: `lang` is not a str.
        ValueError: `lang` cannot name one model.
        LookupError: The model is not installed (``nltk.download('punkt_tab')``).
    """

    def __init__(self, lang: str = "english") -> None:
        PunktSentenceTokenizer.__init__(self)
        self._save_dir: str | None = None
        self.load_lang(lang)

    def load_lang(self, lang: str = "english") -> None:
        """Load the ``punkt_tab`` model for `lang`."""
        lang = _data.punkt_language(lang)
        self._params = _from_raw(_data.load(lang))
        self._lang = lang
        self._save_dir = None

    @property
    def save_dir(self) -> str:
        """NLTK's private directory for saved parameters (created by NLTK)."""
        if self._save_dir is None:
            make = _delegate.attr("nltk.data:make_staging_dir", "PunktTokenizer.save_dir")
            self._save_dir = make(prefix=f"nltk_punkt_{self._lang}_")
        return self._save_dir

    def save_params(self) -> str:
        """Write the parameters as tab files under :attr:`save_dir` (with NLTK); return it."""
        save = _delegate.attr("nltk.tokenize.punkt:save_punkt_params", "save_punkt_params")
        return save(self._params, dir=self.save_dir)  # type: ignore[no-any-return]


def load_punkt_params(lang_dir: Any) -> PunktParameters:
    """Read the parameters of a ``punkt_tab`` model directory (a path or an NLTK path pointer)."""
    import os

    if isinstance(lang_dir, (str, os.PathLike)):
        root = os.fspath(lang_dir)

        def open_file(name: str) -> Any:
            return open(os.path.join(root, name), encoding="utf-8")

        return _from_raw(_data._parse(open_file))
    load = _delegate.attr("nltk.tokenize.punkt:load_punkt_params", "load_punkt_params")
    raw = load(lang_dir)
    return _from_raw(_data.RawParams(raw.collocations, raw.sent_starters, raw.abbrev_types, raw.ortho_context))


DEBUG_DECISION_FMT = """Text: {text!r} (at offset {period_index})
Sentence break? {break_decision} ({reason})
Collocation? {collocation}
{type1!r}:
    known abbreviation: {type1_in_abbrs}
    is initial: {type1_is_initial}
{type2!r}:
    known sentence starter: {type2_is_sent_starter}
    orthographic heuristic suggests is a sentence starter? {type2_ortho_heuristic}
    orthographic contexts in training: {type2_ortho_contexts}
"""


def format_debug_decision(d: dict[str, Any]) -> str:
    """Format one :meth:`PunktSentenceTokenizer.debug_decisions` dict as NLTK does."""
    return DEBUG_DECISION_FMT.format(**d)


def PunktTrainer(*args: Any, **kwargs: Any) -> Any:  # noqa: N802  NLTK's class name
    """NLTK's ``PunktTrainer`` (training is not ported; :class:`pyroid.FallbackWarning`)."""
    return _delegate.attr("nltk.tokenize.punkt:PunktTrainer", "PunktTrainer (training)")(*args, **kwargs)
