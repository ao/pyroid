# Derived from CPython's Lib/difflib.py (PSF-2.0), (c) Python Software Foundation, and typeshed's
# stdlib/difflib.pyi (Apache-2.0); modified. The module docstring is kept here on purpose: the API
# docs are built from this stub.
"""The standard library's `difflib`, in Rust.

```python
import pyroid.difflib as difflib          # was: import difflib

difflib.SequenceMatcher(None, "Acme Corp.", "ACME Corporation").ratio()
difflib.get_close_matches("appel", ["ape", "apple", "peach", "puppy"])
"".join(difflib.unified_diff(old_lines, new_lines, "a/setup.py", "b/setup.py"))
```

Results are the running interpreter's own difflib's (CPython 3.11 to 3.15): the same ratios (the
same float bits), matching blocks, opcodes, `b2j`/`bjunk`/`bpopular` contents and order, calls to
your `isjunk`, diff lines and HTML, and the same exceptions. Where CPython versions differ
(`Differ`'s line pairing in 3.14, `unified_diff(color=)` in 3.15, `HtmlDiff`'s markup), the
running version's behaviour is followed.

`SequenceMatcher`, `Differ` and `HtmlDiff` are subclasses of the standard library's classes, so
`isinstance` checks and your own subclasses work; overridden methods are called as difflib calls
them. `Match`, `IS_LINE_JUNK`, `IS_CHARACTER_JUNK`, `restore` and `diff_bytes` are the standard
library's own objects.

What runs natively: `SequenceMatcher` over a `str`, `bytes`, `bytearray`, or a `list`/`tuple` of
`str`, `int` or `bytes`; `get_close_matches` for a `str` word and `str` possibilities;
`Differ.compare`, `ndiff`, `unified_diff`, `context_diff` and `HtmlDiff`'s line pairing for lists
or tuples of `str`, with `linejunk` any function and `charjunk` `None` or `IS_CHARACTER_JUNK`.
Other element types run the standard library's code, with one `pyroid.FallbackWarning`; a custom
`charjunk` or an overridden `Differ`/`HtmlDiff` method runs the standard library's Python code
with this module's `SequenceMatcher` underneath.

`ratio_batch` and `close_matches_batch` score many strings at once on all cores.
`pyroid.patch("difflib")` makes the standard library's `difflib` use all of this in place.
"""

import difflib as _difflib
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import AnyStr, Final, Literal, TypeAlias, TypeVar, overload

from pyroid._typing import ArrowStrings

__all__ = [
    "IS_CHARACTER_JUNK",
    "IS_LINE_JUNK",
    "SUPPORTED_VERSIONS",
    "Differ",
    "HtmlDiff",
    "Match",
    "SequenceMatcher",
    "close_matches_batch",
    "context_diff",
    "diff_bytes",
    "get_close_matches",
    "ndiff",
    "ratio_batch",
    "restore",
    "unified_diff",
]

_T = TypeVar("_T")
_Tag: TypeAlias = Literal["replace", "delete", "insert", "equal"]

#: The CPython versions (major, minor) whose difflib this module reproduces. On another one,
#: every name here is the standard library's (with a `pyroid.FallbackWarning` at import).
SUPPORTED_VERSIONS: Final[tuple[tuple[int, int], ...]]

Match = _difflib.Match
IS_LINE_JUNK = _difflib.IS_LINE_JUNK
IS_CHARACTER_JUNK = _difflib.IS_CHARACTER_JUNK
restore = _difflib.restore
diff_bytes = _difflib.diff_bytes

class SequenceMatcher(_difflib.SequenceMatcher[_T]):
    """Compare two sequences of hashable elements, as `difflib.SequenceMatcher`.

    Ratcliff/Obershelp matching with CPython's junk and autojunk rules: `b2j` holds where each
    element of `b` occurs, `isjunk` is called once on each distinct element of `b` (in order of
    first occurrence), and with `autojunk` an element of a `b` of 200 or more elements that
    occurs more than 1% + 1 times is popular and not used to start a match.

    The instance attributes are CPython's (`a`, `b`, `isjunk`, `autojunk`, `b2j`, `bjunk`,
    `bpopular`, `fullbcount`, `matching_blocks`, `opcodes`); `b2j`, `bjunk`, `bpopular`,
    `fullbcount` and `matching_blocks` are built on first read. Pickling and copying give
    CPython's state.

    Args:
        isjunk: `None`, or a function called with each distinct element of `b`; true means junk.
        a: The first sequence.
        b: The second sequence, the one that is indexed.
        autojunk: Whether to drop popular elements of a long `b` (see above).
    """

    isjunk: Callable[[_T], object] | None
    a: Sequence[_T]
    b: Sequence[_T]
    autojunk: bool
    b2j: dict[_T, list[int]]
    bjunk: set[_T]
    bpopular: set[_T]
    fullbcount: dict[_T, int] | None
    matching_blocks: list[_difflib.Match] | None
    opcodes: list[tuple[_Tag, int, int, int, int]] | None
    @overload
    def __init__(
        self,
        isjunk: Callable[[_T], object] | None,
        a: Sequence[_T],
        b: Sequence[_T],
        autojunk: bool = True,
    ) -> None: ...
    @overload
    def __init__(self, *, a: Sequence[_T], b: Sequence[_T], autojunk: bool = True) -> None: ...
    @overload
    def __init__(
        self: SequenceMatcher[str],
        isjunk: Callable[[str], object] | None = None,
        a: Sequence[str] = "",
        b: Sequence[str] = "",
        autojunk: bool = True,
    ) -> None: ...
    def set_seqs(self, a: Sequence[_T], b: Sequence[_T]) -> None:
        """Set both sequences: `set_seq1(a)`, then `set_seq2(b)`."""

    def set_seq1(self, a: Sequence[_T]) -> None:
        """Set the first sequence.

        The index of `b` is kept, so comparing many `a` with one `b` this way is cheaper than
        making a matcher per pair.
        """

    def set_seq2(self, b: Sequence[_T]) -> None:
        """Set the second sequence and index it (`isjunk` is called here)."""

    def find_longest_match(
        self, alo: int = 0, ahi: int | None = None, blo: int = 0, bhi: int | None = None
    ) -> _difflib.Match:
        """The longest matching block in `a[alo:ahi]` and `b[blo:bhi]`.

        Of the longest blocks, the one that starts earliest in `a`, then in `b`; it is then
        extended by junk elements on both ends, as CPython does.

        Returns:
            `Match(a=i, b=j, size=k)` with `a[i:i+k] == b[j:j+k]`; `size` 0 if there is none.
        """

    def get_matching_blocks(self) -> list[_difflib.Match]:
        """The non-overlapping matching blocks, in order, ending with `Match(len(a), len(b), 0)`.

        The list is cached (and returned again) until a sequence changes.
        """

    def get_opcodes(self) -> list[tuple[_Tag, int, int, int, int]]:
        """How to turn `a` into `b`, as `(tag, i1, i2, j1, j2)` tuples.

        `tag` is one of `"replace"`, `"delete"`, `"insert"` and `"equal"`. Cached as in CPython.
        """

    def get_grouped_opcodes(self, n: int = 3) -> Iterable[list[tuple[str, int, int, int, int]]]:
        """The opcodes in groups of changes with up to `n` lines of context.

        These are the hunks of a unified diff.
        """

    def ratio(self) -> float:
        """`2.0 * M / T`, the similarity of the two sequences.

        `M` is the number of matched elements and `T` the number of elements in both sequences;
        `1.0` for two empty sequences. The same float as CPython's.
        """

    def quick_ratio(self) -> float:
        """An upper bound on `ratio()` from the elements' counts alone."""

    def real_quick_ratio(self) -> float:
        """An upper bound on `ratio()` from the lengths alone."""

if sys.version_info >= (3, 15):
    def unified_diff(
        a: Sequence[str],
        b: Sequence[str],
        fromfile: str = "",
        tofile: str = "",
        fromfiledate: str = "",
        tofiledate: str = "",
        n: int = 3,
        lineterm: str = "\n",
        *,
        color: bool = False,
    ) -> Iterator[str]:
        """The difference of two lists of lines as a unified diff, line by line.

        Args:
            a: The old lines (usually `file.readlines()`).
            b: The new lines.
            fromfile: The old file's name, for the `---` line.
            tofile: The new file's name, for the `+++` line.
            fromfiledate: Appended to the `---` line after a tab, if not empty.
            tofiledate: Appended to the `+++` line after a tab, if not empty.
            n: Lines of context around each change.
            lineterm: Ends the `---`, `+++` and `@@` lines; `""` for lines without newlines.
            color: Colour the output as `git diff --color` does (CPython 3.15), unless the
                environment says not to (`NO_COLOR`, ...).

        Raises:
            TypeError: At the first `next()`: lines or names that are not `str`, or a `str`
                for `a` or `b` (CPython 3.14+).
        """

else:
    def unified_diff(
        a: Sequence[str],
        b: Sequence[str],
        fromfile: str = "",
        tofile: str = "",
        fromfiledate: str = "",
        tofiledate: str = "",
        n: int = 3,
        lineterm: str = "\n",
    ) -> Iterator[str]:
        """The difference of two lists of lines as a unified diff, line by line.

        Args:
            a: The old lines (usually `file.readlines()`).
            b: The new lines.
            fromfile: The old file's name, for the `---` line.
            tofile: The new file's name, for the `+++` line.
            fromfiledate: Appended to the `---` line after a tab, if not empty.
            tofiledate: Appended to the `+++` line after a tab, if not empty.
            n: Lines of context around each change.
            lineterm: Ends the `---`, `+++` and `@@` lines; `""` for lines without newlines.

        Raises:
            TypeError: At the first `next()`: lines or names that are not `str`, or a `str`
                for `a` or `b` (CPython 3.14+).
        """

def context_diff(
    a: Sequence[str],
    b: Sequence[str],
    fromfile: str = "",
    tofile: str = "",
    fromfiledate: str = "",
    tofiledate: str = "",
    n: int = 3,
    lineterm: str = "\n",
) -> Iterator[str]:
    """The difference of two lists of lines as a context diff, line by line.

    The arguments are `unified_diff`'s (without `color`).
    """

class Differ(_difflib.Differ):
    """Line-by-line deltas with intraline marks (`ndiff`'s format), as `difflib.Differ`.

    Lines that differ are paired up by similarity (character-level `SequenceMatcher` ratios), and
    each pair gets `?` lines marking the changed characters. CPython 3.14 changed how lines are
    paired (a window of 10 lines around the diagonal instead of the best pair of the whole block);
    the running version's way is followed.

    Args:
        linejunk: `None`, or a function called with each distinct line of `b`; true means junk.
        charjunk: `None`, or a function called with characters; true means junk.
    """

    def __init__(
        self,
        linejunk: Callable[[str], object] | None = None,
        charjunk: Callable[[str], object] | None = None,
    ) -> None: ...
    def compare(self, a: Sequence[str], b: Sequence[str]) -> Iterator[str]:
        """The delta from `a` to `b`: lines starting with `"- "`, `"+ "`, `"  "` or `"? "`."""

def ndiff(
    a: Sequence[str],
    b: Sequence[str],
    linejunk: Callable[[str], object] | None = None,
    charjunk: Callable[[str], object] | None = ...,
) -> Iterator[str]:
    """`Differ(linejunk, charjunk).compare(a, b)`; `charjunk` is `IS_CHARACTER_JUNK` by default."""

class HtmlDiff(_difflib.HtmlDiff):
    """A side-by-side HTML table (or page) of the differences, as `difflib.HtmlDiff`.

    The markup, styles and line wrapping are the running interpreter's own; the line pairing
    underneath (`ndiff`) runs natively.

    Args:
        tabsize: Tab stops.
        wrapcolumn: Wrap lines at this column (`None`: no wrapping).
        linejunk: As for `ndiff`.
        charjunk: As for `ndiff`.
    """

    def __init__(
        self,
        tabsize: int = 8,
        wrapcolumn: int | None = None,
        linejunk: Callable[[str], object] | None = None,
        charjunk: Callable[[str], object] | None = ...,
    ) -> None: ...
    def make_file(
        self,
        fromlines: Sequence[str],
        tolines: Sequence[str],
        fromdesc: str = "",
        todesc: str = "",
        context: bool = False,
        numlines: int = 5,
        *,
        charset: str = "utf-8",
    ) -> str:
        """A complete HTML page with the table, its styles and a legend."""

    def make_table(
        self,
        fromlines: Sequence[str],
        tolines: Sequence[str],
        fromdesc: str = "",
        todesc: str = "",
        context: bool = False,
        numlines: int = 5,
    ) -> str:
        """The HTML table alone; with `context`, only changes and `numlines` lines around them."""

@overload
def get_close_matches(
    word: AnyStr,
    possibilities: Iterable[AnyStr],
    n: int = 3,
    cutoff: float = 0.6,
    *,
    threads: int | None = None,
) -> list[AnyStr]:
    """The best `n` of `possibilities` whose `ratio()` with `word` is at least `cutoff`.

    Best first, as CPython orders them (by score, then by the string, larger first). The objects
    returned are the possibilities' own. Native for a `str` word and `str` possibilities, which are
    scored on all cores for long lists.

    Args:
        word: What to look for.
        possibilities: Any iterable of candidates (read once).
        n: At most this many results.
        cutoff: The lowest ratio kept, in `[0.0, 1.0]`.
        threads: `None`: the whole shared pool; `n`: at most `n` tasks at once; `1`: this thread.

    Raises:
        ValueError: `n <= 0` or `cutoff` outside `[0.0, 1.0]`.
    """

@overload
def get_close_matches(
    word: Sequence[_T],
    possibilities: Iterable[Sequence[_T]],
    n: int = 3,
    cutoff: float = 0.6,
    *,
    threads: int | None = None,
) -> list[Sequence[_T]]: ...
@overload
def ratio_batch(
    a: str | Sequence[str],
    b: str | Sequence[str],
    *,
    isjunk: Callable[[str], object] | None = None,
    autojunk: bool = True,
    threads: int | None = None,
) -> list[float]:
    """`SequenceMatcher(isjunk, a[i], b[i], autojunk).ratio()` for every `i`, on all cores.

    ```python
    ratio_batch(["Acme Corp.", "Globex"], ["ACME Corporation", "Globex Inc"])
    ratio_batch("appel", candidates)  # one str is compared with every element of the other
    ```

    Args:
        a: The first sequences: a `str`, a list or tuple of `str`, or an Arrow string array.
        b: The second sequences, as `a`; both must have the same length unless one is a `str`.
        isjunk: `None`, or a function of one character, called once per distinct character of
            `b` before the work starts (so it must not depend on anything else).
        autojunk: As for `SequenceMatcher`.
        threads: `None`: the whole shared pool; `n`: at most `n` tasks at once; `1`: this thread.

    Returns:
        A list of floats, the same as `SequenceMatcher`'s; `None` where an Arrow input is null.

    Raises:
        ValueError: `a` and `b` have different lengths.
        pyroid.InputError: An argument is not text (lone surrogates raise `UnicodeEncodeError`).
    """

@overload
def ratio_batch(
    a: str | Sequence[str] | ArrowStrings,
    b: str | Sequence[str] | ArrowStrings,
    *,
    isjunk: Callable[[str], object] | None = None,
    autojunk: bool = True,
    threads: int | None = None,
) -> list[float | None]: ...
def close_matches_batch(
    words: str | Sequence[str] | ArrowStrings,
    possibilities: Sequence[str] | ArrowStrings,
    n: int = 3,
    cutoff: float = 0.6,
    *,
    threads: int | None = None,
) -> list[list[str] | None]:
    """`get_close_matches(word, possibilities, n, cutoff)` for every word, on all cores.

    Args:
        words: The words to look up: a `str`, a list or tuple of `str`, or an Arrow string array.
        possibilities: The candidates: a list or tuple of `str`, or an Arrow string array (nulls
            are skipped).
        n: At most this many results per word.
        cutoff: The lowest ratio kept, in `[0.0, 1.0]`.
        threads: `None`: the whole shared pool; `n`: at most `n` tasks at once; `1`: this thread.

    Returns:
        One list per word (the possibilities' own `str` objects for a list or tuple), or `None`
        for a null word.

    Raises:
        ValueError: `n <= 0` or `cutoff` outside `[0.0, 1.0]`.
        pyroid.InputError: An argument is not text.
    """
