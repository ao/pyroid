# Derived from jieba 0.42.1 (MIT), © 2013 Sun Junyi; modified.
r"""A drop-in, Rust-backed replacement for jieba 0.42.1's word segmentation.

```python
import pyroid.jieba as jieba

jieba.lcut("我来到北京清华大学")  # ['我', '来到', '北京', '清华大学']
jieba.lcut("我来到北京清华大学", cut_all=True)  # every dictionary word
jieba.lcut_for_search("小明硕士毕业于中国科学院计算所")
list(jieba.tokenize("永和服装饰品有限公司"))  # [('永和', 0, 2), ...]
jieba.load_userdict("userdict.txt")
jieba.add_word("石墨烯")
jieba.lcut(docs, threads=8)  # one list of words per document, in parallel
```

The words, offsets, frequencies and exceptions are jieba 0.42.1's for every input: the
blocks of ``re_han_default`` / ``re_skip_default``, the prefix-dictionary DAG, the
maximum-probability route (the same floating-point operations and tie-breaking), and the
BMES HMM for unknown words (``HMM=True``) are ported step for step, as are ``add_word``,
``del_word``, ``suggest_freq``, ``load_userdict``'s line format and ``Tokenizer`` with a
custom dictionary file. The default dictionary (jieba's ``dict.txt``, MIT) is built into
pyroid as a prebuilt trie, so loading it takes about 3 ms (jieba: about 0.4 s, with or without
its cache file) and writes no cache file; jieba itself is not needed.

Delegated to the real jieba (it must be installed; otherwise :class:`pyroid.PyroidError`),
with one :class:`pyroid.FallbackWarning` per feature: ``pyroid.jieba.posseg``
(part-of-speech tags), ``pyroid.jieba.analyse`` (TF-IDF and TextRank keywords),
paddle mode (``enable_paddle``, ``use_paddle=True``) and :func:`get_dict_file` for the
default dictionary. Before each delegated call, the words added through pyroid are
replayed onto jieba's tokenizer, so those features see them.

pyroid extensions: the segmentation functions also take a list or tuple of str, or an
Arrow string column, and return one list of words per document (``None`` for an Arrow
null), segmented in parallel on pyroid's shared pool (``threads=``: ``None`` for the whole
pool, ``n`` for at most ``n`` tasks, ``1`` for the calling thread).

Differences from jieba:

- Lone surrogates (``"\ud800"``) raise ``UnicodeEncodeError``, as everywhere in
  pyroid; jieba treats them as characters.
- Frequencies must fit in a signed 64-bit integer (``OverflowError`` otherwise); jieba
  takes any Python int.
- :func:`cut` is a generator, as in jieba, but it segments the whole input on the first
  ``next()``.
- ``FREQ`` is a read-only mapping (change it with :func:`add_word`); iterating it lists the
  dictionary's keys in UTF-8 byte order, not file order.
- :func:`enable_parallel` uses threads instead of processes and also works on Windows.
- :func:`enable_paddle` does not ``pip install`` paddlepaddle-tiny for you.
- Wrong argument types raise ``TypeError`` (:class:`pyroid.InputError` for the text)
  where jieba raises ``AttributeError``.
- There is no ``finalseg`` module; ``del_word`` and ``add_word(word, 0)`` still update the
  process-wide set of words the HMM must split, as jieba's ``Force_Split_Words``.
- jieba's log messages ("Building prefix dict ...") are not printed.
"""

from collections.abc import Iterator, Mapping
from re import Pattern
from typing import IO, Any, Final, TypeAlias, overload

from pyroid._typing import ArrowStrings

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
JIEBA_VERSION: Final[str]
__version__: Final[str]
__license__: Final[str]
DEFAULT_DICT: None
DEFAULT_DICT_NAME: str
re_userdict: Pattern[str]
re_eng: Pattern[str]
re_han_default: Pattern[str]
re_skip_default: Pattern[str]
text_type: type[str]
string_types: tuple[type[str]]
default_logger: Any

_Sentence: TypeAlias = str | bytes | bytearray
_Batch: TypeAlias = list[str] | tuple[str, ...]
_Token: TypeAlias = tuple[str, int, int]

def setLogLevel(log_level: int | str) -> None:  # noqa: N802
    """Set the level of this module's logger (jieba's ``setLogLevel``)."""

def strdecode(sentence: Any) -> Any:
    """Decode bytes as jieba does: UTF-8, else GBK ignoring errors; anything else as is."""

def resolve_filename(f: Any) -> str:
    """``f.name``, or ``repr(f)`` for an object without one."""

def enable_paddle() -> None:
    """Enable jieba's paddle mode (delegated to jieba, with a FallbackWarning).

    Raises:
        pyroid.PyroidError: jieba or paddlepaddle-tiny is not installed.
    """

class Tokenizer:
    """A jieba tokenizer: a main dictionary plus the words added to it.

    ``Tokenizer()`` uses jieba's default dictionary (built in).
    ``Tokenizer(dictionary=path)`` reads a jieba-format dictionary file on first
    use: one ``word freq [tag]`` per line, UTF-8; for a repeated word the last
    frequency wins, and ``total`` adds up every line, as in jieba.

    Args:
        dictionary: ``None`` for the default dictionary, or a file path.
    """

    lock: Any
    dictionary: str | None
    user_word_tag_tab: dict[str, str]
    initialized: bool
    tmp_dir: str | None
    cache_file: str | None
    def __init__(self, dictionary: str | None = None) -> None: ...
    @property
    def FREQ(self) -> Mapping[str, int]:  # noqa: N802
        """Every word and word prefix with its frequency (0 for a prefix); empty until loaded."""
    @property
    def total(self) -> int:
        """The sum of the frequencies (0 until initialized)."""
    @staticmethod
    def gen_pfdict(f: IO[bytes]) -> tuple[dict[str, int], int]:
        """Read a binary dictionary file into jieba's ``(FREQ, total)`` (a plain dict).

        Raises:
            ValueError: A line is not ``word freq ...`` with an int ``freq``
                (``invalid dictionary entry in <file> at Line <n>: <line>``).
        """
    def initialize(self, dictionary: str | None = None) -> None:
        """Load the dictionary now (otherwise it loads on first use).

        Args:
            dictionary: Switch to this dictionary file first.

        Raises:
            ValueError: The dictionary file has an invalid line.
            OSError: The dictionary file cannot be read.
        """
    def check_initialized(self) -> None:
        """Load the dictionary if it is not loaded yet."""
    def calc(self, sentence: str, DAG: dict[int, list[int]], route: dict[int, Any]) -> None:  # noqa: N803
        """Fill ``route`` with the maximum-probability route over ``DAG`` (jieba's ``calc``)."""
    def get_DAG(self, sentence: str) -> dict[int, list[int]]:  # noqa: N802
        """For each character index of ``sentence``, the ends of the dictionary words there.

        An index where no dictionary word starts maps to itself.
        """
    @overload
    def cut(
        self,
        sentence: _Sentence,
        cut_all: bool = False,
        HMM: bool = True,  # noqa: N803
        use_paddle: bool = False,
        *,
        threads: int | None = None,
    ) -> Iterator[str]:
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
    @overload
    def cut(
        self,
        sentence: _Batch,
        cut_all: bool = False,
        HMM: bool = True,  # noqa: N803
        use_paddle: bool = False,
        *,
        threads: int | None = None,
    ) -> Iterator[list[str]]: ...
    @overload
    def cut(
        self,
        sentence: ArrowStrings,
        cut_all: bool = False,
        HMM: bool = True,  # noqa: N803
        use_paddle: bool = False,
        *,
        threads: int | None = None,
    ) -> Iterator[list[str] | None]: ...
    @overload
    def lcut(
        self,
        sentence: _Sentence,
        cut_all: bool = False,
        HMM: bool = True,  # noqa: N803
        use_paddle: bool = False,
        *,
        threads: int | None = None,
    ) -> list[str]:
        """:meth:`cut` as a list (for a batch, a list of lists)."""
    @overload
    def lcut(
        self,
        sentence: _Batch,
        cut_all: bool = False,
        HMM: bool = True,  # noqa: N803
        use_paddle: bool = False,
        *,
        threads: int | None = None,
    ) -> list[list[str]]: ...
    @overload
    def lcut(
        self,
        sentence: ArrowStrings,
        cut_all: bool = False,
        HMM: bool = True,  # noqa: N803
        use_paddle: bool = False,
        *,
        threads: int | None = None,
    ) -> list[list[str] | None]: ...
    @overload
    def cut_for_search(
        self,
        sentence: _Sentence,
        HMM: bool = True,  # noqa: N803
        *,
        threads: int | None = None,
    ) -> Iterator[str]:
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
    @overload
    def cut_for_search(
        self,
        sentence: _Batch,
        HMM: bool = True,  # noqa: N803
        *,
        threads: int | None = None,
    ) -> Iterator[list[str]]: ...
    @overload
    def cut_for_search(
        self,
        sentence: ArrowStrings,
        HMM: bool = True,  # noqa: N803
        *,
        threads: int | None = None,
    ) -> Iterator[list[str] | None]: ...
    @overload
    def lcut_for_search(
        self,
        sentence: _Sentence,
        HMM: bool = True,  # noqa: N803
        *,
        threads: int | None = None,
    ) -> list[str]:
        """:meth:`cut_for_search` as a list (for a batch, a list of lists)."""
    @overload
    def lcut_for_search(
        self,
        sentence: _Batch,
        HMM: bool = True,  # noqa: N803
        *,
        threads: int | None = None,
    ) -> list[list[str]]: ...
    @overload
    def lcut_for_search(
        self,
        sentence: ArrowStrings,
        HMM: bool = True,  # noqa: N803
        *,
        threads: int | None = None,
    ) -> list[list[str] | None]: ...
    _lcut = lcut
    _lcut_for_search = lcut_for_search
    def _lcut_no_hmm(self, sentence: Any) -> list[Any]: ...
    def _lcut_all(self, sentence: Any) -> list[Any]: ...
    def _lcut_for_search_no_hmm(self, sentence: Any) -> list[Any]: ...
    def get_dict_file(self) -> IO[bytes]:
        """The main dictionary file, opened in binary mode.

        For the default dictionary this is jieba's own ``dict.txt`` (pyroid
        stores the dictionary in another form), so it is delegated to jieba.

        Raises:
            pyroid.PyroidError: The default dictionary and jieba is not installed.
        """
    def load_userdict(self, f: str | IO[bytes] | IO[str]) -> None:
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
    def add_word(self, word: str | bytes, freq: Any = None, tag: str | None = None) -> None:
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
    def del_word(self, word: str | bytes) -> None:
        """Remove ``word``: set its frequency to 0 and stop the HMM from producing it."""
    def suggest_freq(self, segment: str | tuple[str, ...] | list[str], tune: bool = False) -> int:
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
    @overload
    def tokenize(
        self,
        unicode_sentence: str,
        mode: str = "default",
        HMM: bool = True,  # noqa: N803
        *,
        threads: int | None = None,
    ) -> Iterator[_Token]:
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
    @overload
    def tokenize(
        self,
        unicode_sentence: _Batch,
        mode: str = "default",
        HMM: bool = True,  # noqa: N803
        *,
        threads: int | None = None,
    ) -> Iterator[list[_Token]]: ...
    @overload
    def tokenize(
        self,
        unicode_sentence: ArrowStrings,
        mode: str = "default",
        HMM: bool = True,  # noqa: N803
        *,
        threads: int | None = None,
    ) -> Iterator[list[_Token] | None]: ...
    def set_dictionary(self, dictionary_path: str) -> None:
        """Use another main dictionary file from now on (loaded on next use, without added words).

        Raises:
            Exception: The file does not exist (``jieba: file does not exist: <path>``).
        """

#: The default tokenizer; the module-level functions are its methods.
dt: Tokenizer
user_word_tag_tab: dict[str, str]

def get_FREQ(k: str, d: Any = None) -> Any:  # noqa: N802
    """``dt.FREQ.get(k, d)``."""

@overload
def cut(
    sentence: _Sentence,
    cut_all: bool = False,
    HMM: bool = True,  # noqa: N803
    use_paddle: bool = False,
    *,
    threads: int | None = None,
) -> Iterator[str]:
    """:meth:`Tokenizer.cut` on the default tokenizer."""

@overload
def cut(
    sentence: _Batch,
    cut_all: bool = False,
    HMM: bool = True,  # noqa: N803
    use_paddle: bool = False,
    *,
    threads: int | None = None,
) -> Iterator[list[str]]: ...
@overload
def cut(
    sentence: ArrowStrings,
    cut_all: bool = False,
    HMM: bool = True,  # noqa: N803
    use_paddle: bool = False,
    *,
    threads: int | None = None,
) -> Iterator[list[str] | None]: ...
@overload
def lcut(
    sentence: _Sentence,
    cut_all: bool = False,
    HMM: bool = True,  # noqa: N803
    use_paddle: bool = False,
    *,
    threads: int | None = None,
) -> list[str]:
    """:meth:`Tokenizer.lcut` on the default tokenizer."""

@overload
def lcut(
    sentence: _Batch,
    cut_all: bool = False,
    HMM: bool = True,  # noqa: N803
    use_paddle: bool = False,
    *,
    threads: int | None = None,
) -> list[list[str]]: ...
@overload
def lcut(
    sentence: ArrowStrings,
    cut_all: bool = False,
    HMM: bool = True,  # noqa: N803
    use_paddle: bool = False,
    *,
    threads: int | None = None,
) -> list[list[str] | None]: ...
@overload
def cut_for_search(
    sentence: _Sentence,
    HMM: bool = True,  # noqa: N803
    *,
    threads: int | None = None,
) -> Iterator[str]:
    """:meth:`Tokenizer.cut_for_search` on the default tokenizer."""

@overload
def cut_for_search(
    sentence: _Batch,
    HMM: bool = True,  # noqa: N803
    *,
    threads: int | None = None,
) -> Iterator[list[str]]: ...
@overload
def cut_for_search(
    sentence: ArrowStrings,
    HMM: bool = True,  # noqa: N803
    *,
    threads: int | None = None,
) -> Iterator[list[str] | None]: ...
@overload
def lcut_for_search(
    sentence: _Sentence,
    HMM: bool = True,  # noqa: N803
    *,
    threads: int | None = None,
) -> list[str]:
    """:meth:`Tokenizer.lcut_for_search` on the default tokenizer."""

@overload
def lcut_for_search(
    sentence: _Batch,
    HMM: bool = True,  # noqa: N803
    *,
    threads: int | None = None,
) -> list[list[str]]: ...
@overload
def lcut_for_search(
    sentence: ArrowStrings,
    HMM: bool = True,  # noqa: N803
    *,
    threads: int | None = None,
) -> list[list[str] | None]: ...
@overload
def tokenize(
    unicode_sentence: str,
    mode: str = "default",
    HMM: bool = True,  # noqa: N803
    *,
    threads: int | None = None,
) -> Iterator[_Token]:
    """:meth:`Tokenizer.tokenize` on the default tokenizer."""

@overload
def tokenize(
    unicode_sentence: _Batch,
    mode: str = "default",
    HMM: bool = True,  # noqa: N803
    *,
    threads: int | None = None,
) -> Iterator[list[_Token]]: ...
@overload
def tokenize(
    unicode_sentence: ArrowStrings,
    mode: str = "default",
    HMM: bool = True,  # noqa: N803
    *,
    threads: int | None = None,
) -> Iterator[list[_Token] | None]: ...
def add_word(word: str | bytes, freq: Any = None, tag: str | None = None) -> None:
    """:meth:`Tokenizer.add_word` on the default tokenizer."""

def del_word(word: str | bytes) -> None:
    """:meth:`Tokenizer.del_word` on the default tokenizer."""

def suggest_freq(segment: str | tuple[str, ...] | list[str], tune: bool = False) -> int:
    """:meth:`Tokenizer.suggest_freq` on the default tokenizer."""

def load_userdict(f: str | IO[bytes] | IO[str]) -> None:
    """:meth:`Tokenizer.load_userdict` on the default tokenizer."""

def calc(sentence: str, DAG: dict[int, list[int]], route: dict[int, Any]) -> None:  # noqa: N803
    """:meth:`Tokenizer.calc` on the default tokenizer."""

def get_DAG(sentence: str) -> dict[int, list[int]]:  # noqa: N802
    """:meth:`Tokenizer.get_DAG` on the default tokenizer."""

def get_dict_file() -> IO[bytes]:
    """:meth:`Tokenizer.get_dict_file` on the default tokenizer."""

def initialize(dictionary: str | None = None) -> None:
    """:meth:`Tokenizer.initialize` on the default tokenizer."""

def set_dictionary(dictionary_path: str) -> None:
    """:meth:`Tokenizer.set_dictionary` on the default tokenizer."""

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

def disable_parallel() -> None:
    """Undo :func:`enable_parallel`."""

def _lcut(s: Any) -> list[Any]: ...
def _lcut_all(s: Any) -> list[Any]: ...
def _lcut_no_hmm(s: Any) -> list[Any]: ...
def _lcut_for_search(s: Any) -> list[Any]: ...
def _lcut_for_search_no_hmm(s: Any) -> list[Any]: ...
def _pcut(sentence: Any, cut_all: bool = False, HMM: bool = True) -> Iterator[str]: ...  # noqa: N803
def _pcut_for_search(sentence: Any, HMM: bool = True) -> Iterator[str]: ...  # noqa: N803
