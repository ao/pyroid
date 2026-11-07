# Derived from jieba 0.42.1 (MIT), © 2013 Sun Junyi; modified.
"""A drop-in, Rust-backed replacement for jieba 0.42.1 (``import pyroid.jieba as jieba``).

``cut``, ``lcut``, ``cut_for_search``, ``lcut_for_search``, ``tokenize``,
``load_userdict``, ``add_word``, ``del_word``, ``suggest_freq`` and
:class:`Tokenizer` (default or custom dictionary) give the same words as jieba
0.42.1 for every input. The segmentation runs in Rust; jieba's default
dictionary is built in, so loading it takes no time and jieba itself is not
needed. :mod:`pyroid.jieba.posseg`, :mod:`pyroid.jieba.analyse` and paddle mode
are forwarded to the real jieba, with a one-time :class:`pyroid.FallbackWarning`
(:class:`pyroid.PyroidError` if jieba is not installed).

Beyond jieba, the segmentation functions also take a list or tuple of str, or
an Arrow string column, and return a list with one list of words per document,
segmented in parallel on pyroid's shared thread pool (``threads=``).
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Iterator, Mapping
from typing import Any

from pyroid._pyroid import jieba as _native

__all__ = [
    "DEFAULT_DICT",
    "DEFAULT_DICT_NAME",
    "JIEBA_VERSION",
    "Tokenizer",
    "add_word",
    "calc",
    "cut",
    "cut_for_search",
    "del_word",
    "disable_parallel",
    "dt",
    "enable_parallel",
    "get_DAG",
    "get_FREQ",
    "get_dict_file",
    "initialize",
    "lcut",
    "lcut_for_search",
    "load_userdict",
    "setLogLevel",
    "set_dictionary",
    "suggest_freq",
    "tokenize",
    "user_word_tag_tab",
]

#: The jieba release whose behaviour this module reproduces.
JIEBA_VERSION = "0.42.1"
__version__ = JIEBA_VERSION
__license__ = "MIT"

DEFAULT_DICT = None
DEFAULT_DICT_NAME = "dict.txt"

default_logger = logging.getLogger(__name__)

# jieba's module-level patterns, for code that reads them (the Rust port has its own copies).
re_userdict = re.compile(r"^(.+?)( [0-9]+)?( [a-z]+)?$", re.U)
re_eng = re.compile(r"[a-zA-Z0-9]", re.U)
re_han_default = re.compile("([一-鿕a-zA-Z0-9+#&\\._%\\-]+)", re.U)
re_skip_default = re.compile("(\r\n|\\s)", re.U)

text_type = str
string_types = (str,)


def setLogLevel(log_level: int | str) -> None:  # noqa: N802  jieba's name
    """Set the level of this module's logger (jieba's ``setLogLevel``)."""
    default_logger.setLevel(log_level)


def strdecode(sentence: Any) -> Any:
    """Decode bytes as jieba does: UTF-8, else GBK ignoring errors; anything else as is."""
    if not isinstance(sentence, str) and isinstance(sentence, (bytes, bytearray)):
        try:
            return sentence.decode("utf-8")
        except UnicodeDecodeError:
            return sentence.decode("gbk", "ignore")
    return sentence


def resolve_filename(f: Any) -> str:
    """``f.name``, or ``repr(f)`` for an object without one (jieba's helper)."""
    try:
        return f.name  # type: ignore[no-any-return]
    except AttributeError:
        return repr(f)


def _is_arrow(x: Any) -> bool:
    return hasattr(x, "__arrow_c_stream__") or hasattr(x, "__arrow_c_array__")


_NO_DICTIONARY: list[Any] = []


def _no_dictionary() -> Any:
    """A dictionary for text without word characters (any one gives the same words)."""
    if not _NO_DICTIONARY:
        _NO_DICTIONARY.append(_native._Native())
    return _NO_DICTIONARY[0]


def _get_abs_path(path: str) -> str:
    return os.path.normpath(os.path.join(os.getcwd(), path))


def _gen_entries(f: Any) -> tuple[list[str], list[int], int]:
    """``gen_pfdict``'s parse of a binary dictionary file: words and freqs in file order, total."""
    words: list[str] = []
    freqs: list[int] = []
    ltotal = 0
    f_name = resolve_filename(f)
    for lineno, line in enumerate(f, 1):
        try:
            line = line.strip().decode("utf-8")
            word, freq = line.split(" ")[:2]
            freq = int(freq)
            words.append(word)
            freqs.append(freq)
            ltotal += freq
        except ValueError:
            raise ValueError(
                f"invalid dictionary entry in {f_name} at Line {lineno}: {line}"
            ) from None
    f.close()
    return words, freqs, ltotal


class _Freq(Mapping[str, int]):
    """A read-only view of ``Tokenizer.FREQ`` (every word and word prefix, with its frequency).

    Iteration lists the dictionary's keys in UTF-8 byte order, then added keys
    (jieba's dict lists them in file order).
    """

    __slots__ = ("_tok",)

    def __init__(self, tok: Tokenizer) -> None:
        self._tok = tok

    def __getitem__(self, key: str) -> int:
        n = self._tok._native
        v = n.get(key) if n is not None and isinstance(key, str) else None
        if v is None:
            raise KeyError(key)
        return v  # type: ignore[no-any-return]

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        n = self._tok._native
        if n is None or not isinstance(key, str):
            return default
        v = n.get(key)
        return default if v is None else v

    def __contains__(self, key: object) -> bool:
        n = self._tok._native
        return n is not None and isinstance(key, str) and n.get(key) is not None

    def __len__(self) -> int:
        n = self._tok._native
        return 0 if n is None else len(n)

    def __iter__(self) -> Iterator[str]:
        n = self._tok._native
        return iter(()) if n is None else (k for k, _ in n.entries())

    def __repr__(self) -> str:
        return f"<pyroid.jieba FREQ of {self._tok!r}: {len(self)} entries>"


class Tokenizer:
    """A jieba tokenizer: a dictionary (jieba's default or a file) plus the words added to it.

    ``Tokenizer(dictionary=path)`` reads a jieba-format dictionary file
    (``word freq [tag]`` per line, UTF-8) on first use. The default dictionary
    is built in.
    """

    def __init__(self, dictionary: str | None = DEFAULT_DICT) -> None:
        self.lock = threading.RLock()
        if dictionary == DEFAULT_DICT:
            self.dictionary = dictionary
        else:
            self.dictionary = _get_abs_path(dictionary)  # type: ignore[arg-type]
        self._native: Any = None
        self.user_word_tag_tab: dict[str, str] = {}
        self.initialized = False
        self.tmp_dir: str | None = None
        self.cache_file: str | None = None
        # (word, freq, tag) for every add_word since the dictionary was (re)loaded, and the
        # dictionary path: pyroid.jieba.posseg / analyse replay them onto jieba's own tokenizer.
        self._ops: list[tuple[str, int, str | None]] = []
        self._generation = 0

    def __repr__(self) -> str:
        return f"<Tokenizer dictionary={self.dictionary!r}>"

    @property
    def FREQ(self) -> _Freq:  # noqa: N802  jieba's name
        """Every word and word prefix with its frequency (empty until initialized)."""
        return _Freq(self)

    @property
    def total(self) -> int:
        """The sum of the frequencies (0 until initialized)."""
        return 0 if self._native is None else self._native.total  # type: ignore[no-any-return]

    @staticmethod
    def gen_pfdict(f: Any) -> tuple[dict[str, int], int]:
        """Read a binary dictionary file into jieba's ``(FREQ, total)``."""
        words, freqs, total = _gen_entries(f)
        lfreq: dict[str, int] = {}
        for word, freq in zip(words, freqs, strict=True):
            lfreq[word] = freq
            for ch in range(len(word)):
                wfrag = word[: ch + 1]
                if wfrag not in lfreq:
                    lfreq[wfrag] = 0
        return lfreq, total

    def initialize(self, dictionary: str | None = None) -> None:
        """Load the dictionary now (jieba loads it on first use)."""
        if dictionary:
            abs_path = _get_abs_path(dictionary)
            if self.dictionary == abs_path and self.initialized:
                return
            self.dictionary = abs_path
            self.initialized = False
        else:
            abs_path = self.dictionary  # type: ignore[assignment]
        with self.lock:
            if self.initialized:
                return
            if abs_path == DEFAULT_DICT:
                native = _native._Native()
            else:
                native = self._load_file(self.get_dict_file())
            self._native = native
            self._ops = []
            self._generation += 1
            self.initialized = True

    @staticmethod
    def _load_file(f: Any) -> Any:
        data = f.read()
        fast = _native._Native.from_simple_bytes(data) if isinstance(data, bytes) else None
        if fast is not None:
            f.close()
            return fast
        import io

        buf = io.BytesIO(data)
        buf.name = resolve_filename(f)  # type: ignore[attr-defined]
        f.close()
        words, freqs, total = _gen_entries(buf)
        return _native._Native.from_entries(words, freqs, total)

    def check_initialized(self) -> None:
        """Load the dictionary if it is not loaded yet."""
        if not self.initialized:
            self.initialize()

    def _native_for(self, sentence: Any) -> Any:
        """The dictionary to cut ``sentence`` with, loading it as jieba does.

        jieba loads the dictionary in ``get_DAG``, at the first run of word
        characters (``re_han_default``); text without one is cut without it,
        so nothing is loaded (and a missing or broken dictionary file raises
        nothing). Arrow batches always load it.
        """
        if self.initialized:
            return self._native
        if isinstance(sentence, str):
            needs = re_han_default.search(sentence) is not None
        elif isinstance(sentence, (list, tuple)):
            needs = any(not isinstance(d, str) or re_han_default.search(d) for d in sentence)
        else:
            needs = True
        if needs:
            self.check_initialized()
            return self._native
        return _no_dictionary()

    def calc(self, sentence: str, DAG: dict[int, list[int]], route: dict[int, Any]) -> None:  # noqa: N803
        """Fill ``route`` with the maximum-probability route over ``DAG`` (jieba's ``calc``)."""
        from math import log

        N = len(sentence)  # noqa: N806
        route[N] = (0, 0)
        logtotal = log(self.total)
        freq = self.FREQ
        for idx in range(N - 1, -1, -1):
            route[idx] = max(
                (log(freq.get(sentence[idx : x + 1]) or 1) - logtotal + route[x + 1][0], x)
                for x in DAG[idx]
            )

    def get_DAG(self, sentence: str) -> dict[int, list[int]]:  # noqa: N802
        """For each character index, the end indices of the dictionary words starting there."""
        self.check_initialized()
        return self._native.get_dag(sentence)  # type: ignore[no-any-return]

    def _paddle(self, sentence: Any) -> Iterator[str] | None:
        if not _paddle_enabled():
            return None
        return _delegated_paddle_cut(sentence)

    def lcut(
        self,
        sentence: Any,
        cut_all: bool = False,
        HMM: bool = True,  # noqa: N803
        use_paddle: bool = False,
        *,
        threads: int | None = None,
    ) -> list[Any]:
        """Segment ``sentence`` into a list of words (a list of lists for a batch)."""
        if use_paddle:
            words = self._paddle(sentence)
            if words is not None:
                return list(words)
        sentence = strdecode(sentence)
        return self._native_for(sentence).cut(sentence, bool(cut_all), bool(HMM), threads=threads)  # type: ignore[no-any-return]

    def cut(
        self,
        sentence: Any,
        cut_all: bool = False,
        HMM: bool = True,  # noqa: N803
        use_paddle: bool = False,
        *,
        threads: int | None = None,
    ) -> Iterator[Any]:
        """Segment ``sentence``; yields the words (one list per document for a batch)."""
        yield from self.lcut(sentence, cut_all, HMM, use_paddle, threads=threads)

    def lcut_for_search(
        self,
        sentence: Any,
        HMM: bool = True,  # noqa: N803
        *,
        threads: int | None = None,
    ) -> list[Any]:
        """:meth:`lcut`, plus the 2- and 3-character dictionary words inside longer words."""
        sentence = strdecode(sentence)
        return self._native_for(sentence).cut_for_search(sentence, bool(HMM), threads=threads)  # type: ignore[no-any-return]

    def cut_for_search(
        self,
        sentence: Any,
        HMM: bool = True,  # noqa: N803
        *,
        threads: int | None = None,
    ) -> Iterator[Any]:
        """Generator form of :meth:`lcut_for_search`."""
        yield from self.lcut_for_search(sentence, HMM, threads=threads)

    _lcut = lcut
    _lcut_for_search = lcut_for_search

    def _lcut_no_hmm(self, sentence: Any) -> list[Any]:
        return self.lcut(sentence, False, False)

    def _lcut_all(self, sentence: Any) -> list[Any]:
        return self.lcut(sentence, True)

    def _lcut_for_search_no_hmm(self, sentence: Any) -> list[Any]:
        return self.lcut_for_search(sentence, False)

    def get_dict_file(self) -> Any:
        """The dictionary file, opened in binary mode.

        For the default dictionary this is jieba's own ``dict.txt`` (pyroid
        stores it in another form), so it needs jieba installed.
        """
        if self.dictionary == DEFAULT_DICT:
            jieba = _jieba_module("get_dict_file() of the default dictionary")
            return jieba.get_dict_file()
        return open(self.dictionary, "rb")

    def load_userdict(self, f: Any) -> None:
        """Add the words of a user dictionary: a path, or a file opened in binary or text mode.

        One word per line: ``word [freq] [tag]``, separated by spaces; a file
        given by path or opened in binary mode must be UTF-8.
        """
        self.check_initialized()
        if isinstance(f, str):
            f_name = f
            f = open(f, "rb")  # noqa: SIM115
        else:
            f_name = resolve_filename(f)
        for _lineno, ln in enumerate(f, 1):
            line = ln.strip()
            if not isinstance(line, str):
                try:
                    line = line.decode("utf-8").lstrip("﻿")
                except UnicodeDecodeError:
                    raise ValueError(f"dictionary file {f_name} must be utf-8") from None
            if not line:
                continue
            # match won't be None because there's at least one character
            word, freq, tag = re_userdict.match(line).groups()  # type: ignore[union-attr]
            if freq is not None:
                freq = freq.strip()
            if tag is not None:
                tag = tag.strip()
            self.add_word(word, freq, tag)

    def add_word(self, word: Any, freq: Any = None, tag: str | None = None) -> None:
        """Add ``word`` (or change its frequency).

        Without ``freq``, the frequency is :meth:`suggest_freq`'s, which makes
        the word cut whole.
        """
        self.check_initialized()
        word = strdecode(word)
        freq = int(freq) if freq is not None else self.suggest_freq(word, False)
        with self.lock:
            self._native.add_word(word, freq)
            if tag:
                self.user_word_tag_tab[word] = tag
            self._ops.append((word, freq, tag))
        if freq == 0:
            _native.add_force_split(word)

    def del_word(self, word: Any) -> None:
        """Remove ``word``: frequency 0, and the HMM may not produce it."""
        self.add_word(word, 0)

    def suggest_freq(self, segment: Any, tune: bool = False) -> int:
        """The frequency that makes ``segment`` cut as given.

        A str: the frequency that keeps it whole. A tuple of str: one that
        cuts their concatenation into them. With ``tune=True``, also set it
        (:meth:`add_word`). The HMM may still change the result; if so, cut
        with ``HMM=False``.
        """
        self.check_initialized()
        if isinstance(segment, str):
            word = segment
            freq = self._native.suggest_freq(word)
        else:
            segment = tuple(map(strdecode, segment))
            word = "".join(segment)
            freq = self._native.suggest_freq_split(list(segment))
        if tune:
            self.add_word(word, freq)
        return freq  # type: ignore[no-any-return]

    def tokenize(
        self,
        unicode_sentence: Any,
        mode: str = "default",
        HMM: bool = True,  # noqa: N803
        *,
        threads: int | None = None,
    ) -> Iterator[Any]:
        """Yield ``(word, start, end)`` with character offsets; ``mode="search"`` for finer words.

        For a batch (list, tuple or Arrow column), yields one list of tuples per document.
        """
        if not isinstance(unicode_sentence, (str, list, tuple)) and not _is_arrow(unicode_sentence):
            raise ValueError("jieba: the input parameter should be unicode.")
        yield from self._native_for(unicode_sentence).tokenize(
            unicode_sentence, mode != "default", bool(HMM), threads=threads
        )

    def set_dictionary(self, dictionary_path: str) -> None:
        """Use another main dictionary from now on (loaded on next use)."""
        with self.lock:
            abs_path = _get_abs_path(dictionary_path)
            if not os.path.isfile(abs_path):
                raise Exception("jieba: file does not exist: " + abs_path)
            self.dictionary = abs_path
            self.initialized = False


# ---- the delegated pieces ----------------------------------------------------------------


def _jieba_module(what: str) -> Any:
    """The real jieba, after a one-time FallbackWarning for ``what``."""
    from pyroid.jieba._delegate import jieba_module

    return jieba_module(what)


def _paddle_enabled() -> bool:
    from pyroid.jieba._delegate import paddle_enabled

    return paddle_enabled()


def _delegated_paddle_cut(sentence: Any) -> Iterator[str]:
    from pyroid.jieba._delegate import paddle_cut

    return paddle_cut(sentence)


def enable_paddle() -> None:
    """Enable jieba's paddle mode (delegated to jieba, with a FallbackWarning)."""
    from pyroid.jieba._delegate import enable_paddle as _enable

    _enable()


# ---- the default tokenizer and module-level functions ------------------------------------

dt = Tokenizer()


def get_FREQ(k: str, d: Any = None) -> Any:  # noqa: N802
    """``dt.FREQ.get(k, d)``."""
    return dt.FREQ.get(k, d)


add_word = dt.add_word
calc = dt.calc
cut = dt.cut
lcut = dt.lcut
cut_for_search = dt.cut_for_search
lcut_for_search = dt.lcut_for_search
del_word = dt.del_word
get_DAG = dt.get_DAG  # noqa: N816  jieba's name
get_dict_file = dt.get_dict_file
initialize = dt.initialize
load_userdict = dt.load_userdict
set_dictionary = dt.set_dictionary
suggest_freq = dt.suggest_freq
tokenize = dt.tokenize
user_word_tag_tab = dt.user_word_tag_tab


def _lcut_all(s: Any) -> list[Any]:
    return dt._lcut_all(s)


def _lcut(s: Any) -> list[Any]:
    return dt._lcut(s)


def _lcut_no_hmm(s: Any) -> list[Any]:
    return dt._lcut_no_hmm(s)


def _lcut_for_search(s: Any) -> list[Any]:
    return dt._lcut_for_search(s)


def _lcut_for_search_no_hmm(s: Any) -> list[Any]:
    return dt._lcut_for_search_no_hmm(s)


_parallel_threads: int | None = None


def _pcut(sentence: Any, cut_all: bool = False, HMM: bool = True) -> Iterator[str]:  # noqa: N803
    parts = strdecode(sentence).splitlines(True)
    for r in dt.lcut(parts, cut_all, HMM, threads=_parallel_threads):
        yield from r


def _pcut_for_search(sentence: Any, HMM: bool = True) -> Iterator[str]:  # noqa: N803
    parts = strdecode(sentence).splitlines(True)
    for r in dt.lcut_for_search(parts, HMM, threads=_parallel_threads):
        yield from r


def enable_parallel(processnum: int | None = None) -> None:
    """Make the module-level ``cut`` and ``cut_for_search`` segment line by line in parallel.

    As in jieba, this only changes those two functions (on ``dt``). pyroid
    uses threads (at most ``processnum``; all of its pool by default), not
    processes, and works on every platform. The words are the same.
    """
    global _parallel_threads, cut, cut_for_search
    dt.check_initialized()
    _parallel_threads = processnum
    cut = _pcut
    cut_for_search = _pcut_for_search


def disable_parallel() -> None:
    """Undo :func:`enable_parallel`."""
    global _parallel_threads, cut, cut_for_search
    _parallel_threads = None
    cut = dt.cut
    cut_for_search = dt.cut_for_search
