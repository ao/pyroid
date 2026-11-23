# Derived from jieba 0.42.1 (MIT), © 2013 Sun Junyi; modified.
"""A drop-in, Rust-backed replacement for jieba 0.42.1 (``import pyroid.jieba as jieba``).

``cut``, ``lcut``, ``cut_for_search``, ``lcut_for_search``, ``tokenize``,
``load_userdict``, ``add_word``, ``del_word``, ``suggest_freq`` and
:class:`Tokenizer` (default or custom dictionary) give the same words as jieba
0.42.1 for every input. The segmentation runs in Rust; jieba's default
dictionary is built in, so loading it takes no time and jieba itself is not
needed. ``pyroid.jieba.posseg``, ``pyroid.jieba.analyse`` and paddle mode
are forwarded to the real jieba, with a one-time :class:`pyroid.FallbackWarning`
(:class:`pyroid.PyroidError` if jieba is not installed).

Beyond jieba, the segmentation functions also take a list or tuple of str, or
an Arrow string column, and return a list with one list of words per document,
segmented in parallel on pyroid's shared thread pool (``threads=``).
"""

from __future__ import annotations

import logging
import operator
import os
import re
import threading
from collections.abc import Iterator, Mapping
from typing import Any

from pyroid import _versions
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

_versions.check(__name__, "jieba", f"jieba=={JIEBA_VERSION}")

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
    """``f.name``, or ``repr(f)`` for an object without one."""
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
    """A jieba tokenizer: a main dictionary plus the words added to it.

    ``Tokenizer()`` uses jieba's default dictionary (built in).
    ``Tokenizer(dictionary=path)`` reads a jieba-format dictionary file on first
    use: one ``word freq [tag]`` per line, UTF-8; for a repeated word the last
    frequency wins, and ``total`` adds up every line, as in jieba.

    Args:
        dictionary: ``None`` for the default dictionary, or a file path.
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
        """Every word and word prefix with its frequency (0 for a prefix); empty until loaded."""
        return _Freq(self)

    @property
    def total(self) -> int:
        """The sum of the frequencies (0 until initialized)."""
        return 0 if self._native is None else self._native.total  # type: ignore[no-any-return]

    @staticmethod
    def gen_pfdict(f: Any) -> tuple[dict[str, int], int]:
        """Read a binary dictionary file into jieba's ``(FREQ, total)`` (a plain dict).

        Raises:
            ValueError: A line is not ``word freq ...`` with an int ``freq``
                (``invalid dictionary entry in <file> at Line <n>: <line>``).
        """
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
        """Load the dictionary now (otherwise it loads on first use).

        Args:
            dictionary: Switch to this dictionary file first.

        Raises:
            ValueError: The dictionary file has an invalid line.
            OSError: The dictionary file cannot be read.
        """
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
        """For each character index of ``sentence``, the ends of the dictionary words there.

        An index where no dictionary word starts maps to itself.
        """
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
        """:meth:`cut` as a list (for a batch, a list of lists)."""
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
        """Segment ``sentence`` into words; yields them.

        Args:
            sentence: A str (bytes are decoded as UTF-8, else GBK), or a batch:
                a list or tuple of str, or an Arrow string array/column.
            cut_all: Full mode: every dictionary word in the text, overlapping
                (empty strings included, as jieba yields them).
            HMM: Find words that are not in the dictionary with the HMM.
            use_paddle: Use paddle mode if :func:`enable_paddle` enabled it
                (delegated to jieba); ignored otherwise, as in jieba.
            threads: For a batch: ``None`` for the whole pool, ``n`` for at
                most ``n`` tasks at once, ``1`` for the calling thread.

        Yields:
            The words, which concatenate to ``sentence`` (except in full
            mode); for a batch, one list of words per document (``None`` for
            an Arrow null).

        Raises:
            ValueError: The dictionary's total is 0, or a word on the route
                has a negative frequency (``math.log``'s error).
            pyroid.InputError: A batch holds a non-str item.
            UnicodeEncodeError: A str contains a lone surrogate.
        """
        yield from self.lcut(sentence, cut_all, HMM, use_paddle, threads=threads)

    def lcut_for_search(
        self,
        sentence: Any,
        HMM: bool = True,  # noqa: N803
        *,
        threads: int | None = None,
    ) -> list[Any]:
        """:meth:`cut_for_search` as a list (for a batch, a list of lists)."""
        sentence = strdecode(sentence)
        return self._native_for(sentence).cut_for_search(sentence, bool(HMM), threads=threads)  # type: ignore[no-any-return]

    def cut_for_search(
        self,
        sentence: Any,
        HMM: bool = True,  # noqa: N803
        *,
        threads: int | None = None,
    ) -> Iterator[Any]:
        """Segment for search engines: :meth:`cut`, plus the dictionary words inside long ones.

        Each word longer than 2 characters is preceded by the 2-character
        dictionary words inside it, and one longer than 3 also by the
        3-character ones.

        Args:
            sentence: A str, or a batch as for :meth:`cut`.
            HMM: Find words that are not in the dictionary with the HMM.
            threads: For a batch, as for :meth:`cut`.

        Raises:
            ValueError: As for :meth:`cut`.
        """
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
        """The main dictionary file, opened in binary mode.

        For the default dictionary this is jieba's own ``dict.txt`` (pyroid
        stores the dictionary in another form), so it is delegated to jieba.

        Raises:
            pyroid.PyroidError: The default dictionary and jieba is not installed.
        """
        if self.dictionary == DEFAULT_DICT:
            jieba = _jieba_module("get_dict_file() of the default dictionary")
            return jieba.get_dict_file()
        return open(self.dictionary, "rb")

    def load_userdict(self, f: Any) -> None:
        """Add the words of a user dictionary, as :meth:`add_word` does for each line.

        One entry per line: ``word [freq] [tag]``, separated by spaces (jieba's
        ``re_userdict``: a word may contain spaces; ``freq`` is ASCII digits;
        ``tag`` lowercase ASCII letters). Blank lines are skipped.

        Args:
            f: A path, or a file opened in binary mode (UTF-8, an initial BOM
                is ignored) or in text mode.

        Raises:
            ValueError: A file given by path or opened in binary mode is not
                UTF-8.
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
        """Add ``word`` to the dictionary, or change its frequency.

        Args:
            word: The word.
            freq: Its frequency (anything ``int()`` takes); ``None`` uses
                :meth:`suggest_freq`, which makes the word cut whole.
            tag: A part-of-speech tag, kept in ``user_word_tag_tab``.

        Raises:
            ValueError: ``int(freq)`` fails.
            OverflowError: ``freq`` or the new total does not fit in 64 bits.
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
        """Remove ``word``: set its frequency to 0 and stop the HMM from producing it."""
        self.add_word(word, 0)

    def suggest_freq(self, segment: Any, tune: bool = False) -> int:
        """The frequency that makes ``segment`` cut as given.

        Args:
            segment: A str, to keep it whole; or a tuple of str, to cut their
                concatenation into them.
            tune: Also set the word to that frequency (:meth:`add_word`).

        Returns:
            jieba's value. The HMM can still change how the word is cut; if
            so, cut with ``HMM=False``.

        Raises:
            ValueError: The dictionary's total is 0 (``math.log``'s error).
            ZeroDivisionError: The dictionary's total is 0.
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
        """Yield ``(word, start, end)`` for each word, with character offsets.

        Args:
            unicode_sentence: A str, or a batch as for :meth:`cut` (then one
                list of tuples per document).
            mode: ``"default"`` for :meth:`cut`'s words; anything else for
                :meth:`cut_for_search`'s.
            HMM: Find words that are not in the dictionary with the HMM.
            threads: For a batch, as for :meth:`cut`.

        Raises:
            ValueError: ``unicode_sentence`` is not a str or a batch
                (``jieba: the input parameter should be unicode.``), or as for
                :meth:`cut`.
        """
        if not isinstance(unicode_sentence, (str, list, tuple)) and not _is_arrow(unicode_sentence):
            raise ValueError("jieba: the input parameter should be unicode.")
        yield from self._native_for(unicode_sentence).tokenize(
            unicode_sentence, mode != "default", bool(HMM), threads=threads
        )

    def set_dictionary(self, dictionary_path: str) -> None:
        """Use another main dictionary file from now on (loaded on next use, without added words).

        Raises:
            Exception: The file does not exist (``jieba: file does not exist: <path>``).
        """
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
    """Enable jieba's paddle mode (delegated to jieba, with a FallbackWarning).

    Raises:
        pyroid.PyroidError: jieba or paddlepaddle-tiny is not installed.
    """
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
    """Make the module-level :func:`cut` and :func:`cut_for_search` cut lines in parallel.

    As in jieba, only those two functions change (on :data:`dt`). pyroid uses
    threads, at most ``processnum`` (the whole pool by default), and works on
    every platform; the words are the same.

    Args:
        processnum: The most threads to cut with; ``None`` for the whole pool.

    Raises:
        ValueError: ``processnum`` is below 1 (``"Number of processes must be
            at least 1"``, as jieba's ``multiprocessing.Pool``).
        TypeError: ``processnum`` is not an int (as jieba's ``Pool``).
    """
    global _parallel_threads, cut, cut_for_search
    dt.check_initialized()
    if processnum is not None:
        # jieba's multiprocessing.Pool(processnum) raises these, here and not at the next cut.
        if processnum < 1:
            raise ValueError("Number of processes must be at least 1")
        processnum = operator.index(processnum)
    _parallel_threads = processnum
    cut = _pcut
    cut_for_search = _pcut_for_search


def disable_parallel() -> None:
    """Undo :func:`enable_parallel`."""
    global _parallel_threads, cut, cut_for_search
    _parallel_threads = None
    cut = dt.cut
    cut_for_search = dt.cut_for_search
