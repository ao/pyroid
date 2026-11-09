"""NeMo Curator 1.4.0's CPU text heuristics in Rust, with the same scores (``pyroid.nemo``)."""

from collections.abc import Sequence
from typing import Final, overload

from pyroid._pyroid import Column
from pyroid._typing import ArrowStrings

__all__ = [
    "NEMO_CURATOR_VERSION",
    # filter scores and fastText helpers
    # string modifiers
    "fasttext_label",
    "normalize_newlines",
    "remove_boilerplate_strings",
    "remove_lines",
    "remove_markdown",
    "remove_quotation",
    "remove_urls",
    "slice_text",
]

NEMO_CURATOR_VERSION: Final[str]

# --- filter scores and fastText helpers -------------------------------------------------------

# --- string modifiers ---------------------------------------------------------------------------
# Each takes a str, a list or tuple of str, or an Arrow string array/column, and returns the same kind
# (a pyroid.Column for Arrow input); unchanged documents come back as the original objects and Arrow
# nulls stay null.

@overload
def remove_boilerplate_strings(
    x: str, /, *, remove_if_at_top_or_bottom: bool = True, threads: int | None = None
) -> str:
    """Drop boilerplate paragraphs, as ``BoilerPlateStringModifier`` does.

    Paragraphs are ``text.split("\\n\\n")``. A paragraph containing "lorem ipsum" (any case) makes the
    document empty. A paragraph with one of NeMo's policy strings ("terms of use", "privacy policy",
    "uses cookies", ...) is boilerplate: with ``remove_if_at_top_or_bottom=False`` it makes the document
    empty; with ``True`` the boilerplate paragraphs are removed when they form one block at the top or
    the bottom of the document, and the document is left as it is otherwise (also when every paragraph
    is boilerplate).

    Args:
        x: The documents.
        remove_if_at_top_or_bottom: As for ``BoilerPlateStringModifier``.
        threads: ``None`` uses pyroid's shared thread pool; ``n`` runs at most ``n`` tasks at once;
            ``1`` runs on the calling thread.

    Returns:
        The documents, the same kind as ``x``.

    Raises:
        pyroid.InputError: ``x`` is not one of the accepted kinds.
    """

@overload
def remove_boilerplate_strings(
    x: list[str], /, *, remove_if_at_top_or_bottom: bool = True, threads: int | None = None
) -> list[str]: ...
@overload
def remove_boilerplate_strings(
    x: tuple[str, ...], /, *, remove_if_at_top_or_bottom: bool = True, threads: int | None = None
) -> tuple[str, ...]: ...
@overload
def remove_boilerplate_strings(
    x: ArrowStrings, /, *, remove_if_at_top_or_bottom: bool = True, threads: int | None = None
) -> Column[str]: ...
@overload
def remove_markdown(x: str, /, *, threads: int | None = None) -> str:
    """Remove Markdown emphasis and links, as ``MarkdownRemover`` does.

    Line by line, ``**a**``, then ``*a*``, then ``_a_`` become ``a``, and ``[a](b)`` becomes ``b``
    (NeMo's four ``re.sub`` calls with lazy groups, in that order).

    Args:
        x: The documents.
        threads: As for :func:`remove_boilerplate_strings`.

    Returns:
        The documents, the same kind as ``x``.
    """

@overload
def remove_markdown(x: list[str], /, *, threads: int | None = None) -> list[str]: ...
@overload
def remove_markdown(x: tuple[str, ...], /, *, threads: int | None = None) -> tuple[str, ...]: ...
@overload
def remove_markdown(x: ArrowStrings, /, *, threads: int | None = None) -> Column[str]: ...
@overload
def normalize_newlines(x: str, /, *, threads: int | None = None) -> str:
    """Shorten newline runs, as ``NewlineNormalizer`` does.

    Three or more ``\\n`` in a row become ``\\n\\n``; then three or more ``\\r\\n`` in a row become
    ``\\r\\n\\r\\n``.

    Args:
        x: The documents.
        threads: As for :func:`remove_boilerplate_strings`.

    Returns:
        The documents, the same kind as ``x``.
    """

@overload
def normalize_newlines(x: list[str], /, *, threads: int | None = None) -> list[str]: ...
@overload
def normalize_newlines(x: tuple[str, ...], /, *, threads: int | None = None) -> tuple[str, ...]: ...
@overload
def normalize_newlines(x: ArrowStrings, /, *, threads: int | None = None) -> Column[str]: ...
@overload
def remove_quotation(x: str, /, *, threads: int | None = None) -> str:
    """Remove the quotes around a quoted document, as ``QuotationRemover`` does.

    A document that starts and ends with ``"`` and has more than 2 characters after ``strip()`` loses
    its first and last character, unless it contains a newline and its first line ends with ``"``.

    Args:
        x: The documents.
        threads: As for :func:`remove_boilerplate_strings`.

    Returns:
        The documents, the same kind as ``x``.
    """

@overload
def remove_quotation(x: list[str], /, *, threads: int | None = None) -> list[str]: ...
@overload
def remove_quotation(x: tuple[str, ...], /, *, threads: int | None = None) -> tuple[str, ...]: ...
@overload
def remove_quotation(x: ArrowStrings, /, *, threads: int | None = None) -> Column[str]: ...
@overload
def remove_urls(x: str, /, *, threads: int | None = None) -> str:
    """Delete URLs, as ``UrlRemover`` does.

    The same as ``re.sub(r"https?://\\S+|www\\.\\S+", "", text, flags=re.IGNORECASE)``, including
    ``re``'s case rules (``ſ``, U+017F, matches ``s``).

    Args:
        x: The documents.
        threads: As for :func:`remove_boilerplate_strings`.

    Returns:
        The documents, the same kind as ``x``.
    """

@overload
def remove_urls(x: list[str], /, *, threads: int | None = None) -> list[str]: ...
@overload
def remove_urls(x: tuple[str, ...], /, *, threads: int | None = None) -> tuple[str, ...]: ...
@overload
def remove_urls(x: ArrowStrings, /, *, threads: int | None = None) -> Column[str]: ...
@overload
def remove_lines(x: str, /, patterns: Sequence[str], *, threads: int | None = None) -> str:
    """Drop the lines equal to one of ``patterns``, as ``LineRemover(patterns)`` does.

    Lines are ``text.split("\\n")``; a line is dropped when it equals a pattern exactly.

    Args:
        x: The documents.
        patterns: The lines to drop.
        threads: As for :func:`remove_boilerplate_strings`.

    Returns:
        The documents, the same kind as ``x``.
    """

@overload
def remove_lines(
    x: list[str], /, patterns: Sequence[str], *, threads: int | None = None
) -> list[str]: ...
@overload
def remove_lines(
    x: tuple[str, ...], /, patterns: Sequence[str], *, threads: int | None = None
) -> tuple[str, ...]: ...
@overload
def remove_lines(
    x: ArrowStrings, /, patterns: Sequence[str], *, threads: int | None = None
) -> Column[str]: ...
@overload
def slice_text(
    x: str,
    /,
    left: int | str | None = 0,
    right: int | str | None = None,
    include_left: bool = True,
    include_right: bool = True,
    strip: bool = True,
    *,
    threads: int | None = None,
) -> str:
    """Slice each document, as ``Slicer(left, right, include_left, include_right, strip)`` does.

    An int bound is a Python slice index (in characters). A str ``left`` starts the slice at its first
    occurrence (after it when ``include_left`` is false), a str ``right`` ends it at its last occurrence
    (after it when ``include_right`` is true); the result is ``""`` when either string is missing.
    ``None`` means the start or the end. ``strip`` strips the result.

    Args:
        x: The documents.
        left: The start bound.
        right: The end bound.
        include_left: Keep a str ``left`` in the result.
        include_right: Keep a str ``right`` in the result.
        strip: Strip whitespace from the result.
        threads: As for :func:`remove_boilerplate_strings`.

    Returns:
        The documents, the same kind as ``x``.
    """

@overload
def slice_text(
    x: list[str],
    /,
    left: int | str | None = 0,
    right: int | str | None = None,
    include_left: bool = True,
    include_right: bool = True,
    strip: bool = True,
    *,
    threads: int | None = None,
) -> list[str]: ...
@overload
def slice_text(
    x: tuple[str, ...],
    /,
    left: int | str | None = 0,
    right: int | str | None = None,
    include_left: bool = True,
    include_right: bool = True,
    strip: bool = True,
    *,
    threads: int | None = None,
) -> tuple[str, ...]: ...
@overload
def slice_text(
    x: ArrowStrings,
    /,
    left: int | str | None = 0,
    right: int | str | None = None,
    include_left: bool = True,
    include_right: bool = True,
    strip: bool = True,
    *,
    threads: int | None = None,
) -> Column[str]: ...
@overload
def fasttext_label(x: str, /, label: object, *, threads: int | None = None) -> str:
    """Prefix each document with a fastText label, as ``FastTextLabelModifier(label)`` does.

    The result is ``f"{label} {text}"`` after replacing ``\\n`` and ``__label__`` in the text by
    spaces. Every document changes.

    Args:
        x: The documents.
        label: The label (formatted with ``format(label, "")``, as an f-string does).
        threads: As for :func:`remove_boilerplate_strings`.

    Returns:
        The documents, the same kind as ``x``.
    """

@overload
def fasttext_label(x: list[str], /, label: object, *, threads: int | None = None) -> list[str]: ...
@overload
def fasttext_label(
    x: tuple[str, ...], /, label: object, *, threads: int | None = None
) -> tuple[str, ...]: ...
@overload
def fasttext_label(x: ArrowStrings, /, label: object, *, threads: int | None = None) -> Column[str]: ...
