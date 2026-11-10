# Derived from ftfy 6.3.1 (Apache-2.0), © Robyn Speer; modified.
# Type stubs for pyroid.ftfy: the signatures of ftfy 6.3.1. The module
# docstring is kept here on purpose: the API docs are built from this stub.
"""A drop-in, Rust-backed replacement for ftfy 6.3.1 (``import pyroid.ftfy as ftfy``).

Results are byte-identical to ftfy 6.3.1 on the same CPython version.
``apply_plan``, ``guess_bytes``, ``fix_file`` and ``explain_unicode``, and
option values the Rust port does not handle (an unknown ``normalization``
form, a non-integer ``max_decode_length``), are delegated to the real ftfy
with a one-time ``pyroid.FallbackWarning``, or raise ``pyroid.PyroidError``
if it is not installed.

Differences from ftfy
---------------------
- ``max_decode_length`` below 1 raises ``ValueError``. ftfy loops forever on
  such a value.
- ``fix_text`` on a list or tuple raises ``TypeError`` (ftfy raises
  ``AttributeError``). For other non-``str`` inputs both raise ``TypeError``,
  with different messages.
- ``TextFixerConfig``, ``ExplanationStep`` and ``ExplainedText`` are pyroid's
  own classes. They have the same fields and defaults and compare equal to
  ftfy's (they are tuples), but are not instances of ftfy's classes. ftfy's
  ``TextFixerConfig`` objects are accepted wherever a config is.
"""

from collections.abc import Iterator
from typing import Any, BinaryIO, Final, Literal, NamedTuple, TextIO, overload

from pyroid._pyroid import Column
from pyroid._typing import ArrowStrings

__all__ = [
    "DELEGATED",
    "FTFY_VERSION",
    "ExplainedText",
    "ExplanationStep",
    "TextFixerConfig",
    "apply_plan",
    "explain_unicode",
    "fix_and_explain",
    "fix_encoding",
    "fix_encoding_and_explain",
    "fix_file",
    "fix_text",
    "fix_text_batch",
    "fix_text_segment",
    "ftfy",
    "guess_bytes",
]

#: The ftfy release whose behaviour this module reproduces.
FTFY_VERSION: Final[str]
#: The functions that are forwarded to the real ftfy, which must be installed for them.
DELEGATED: Final[tuple[str, ...]]
#: The message of the ``UnicodeError`` raised for a ``bytes`` argument (ftfy's text).
BYTES_ERROR_TEXT: Final[str]
_UNICODE_VERSION: Final[str]

class ExplanationStep(NamedTuple):
    """One step of an :class:`ExplainedText` explanation, as in ftfy.

    Attributes:
        action: ``"encode"``, ``"decode"``, ``"transcode"``, ``"apply"`` or
            ``"normalize"``.
        parameter: The encoding, or the name of the fixer or normalization form.
    """

    action: str
    parameter: str

class ExplainedText(NamedTuple):
    """Fixed text and the steps that fixed it, as in ftfy.

    Attributes:
        text: The fixed text.
        explanation: The steps in order, which :func:`apply_plan` replays on the
            original text, or ``None`` when the config has ``explain=False``.
    """

    text: str
    explanation: list[ExplanationStep] | None

class TextFixerConfig(NamedTuple):
    r"""ftfy's options, with the same fields and defaults.

    Pass one to any fixer, or give fields as keyword arguments. Instances of
    ``ftfy.TextFixerConfig`` are accepted too.

    Attributes:
        unescape_html: Decode HTML entities such as ``&amp;``. ``"auto"`` does
            it unless the text contains ``<`` (then it is probably HTML);
            ``True`` always, ``False`` never.
        remove_terminal_escapes: Remove ANSI terminal escape sequences, such as
            colour codes.
        fix_encoding: Fix mojibake: text decoded with the wrong encoding.
        restore_byte_a0: When fixing mojibake, allow a space to stand for a
            lost non-breaking space (byte 0xA0).
        replace_lossy_sequences: Replace mojibake whose bytes were partly lost
            (to ``�`` or ``?``) with ``�``.
        decode_inconsistent_utf8: Fix UTF-8 mojibake even when the rest of the
            text does not decode the same way.
        fix_c1_controls: Replace C1 control characters (U+0080 to U+009F) with
            the Windows-1252 characters they stand for.
        fix_latin_ligatures: Replace Latin ligatures such as ``ﬁ`` with their
            letters.
        fix_character_width: Replace fullwidth Latin characters and halfwidth
            Katakana with their standard forms.
        uncurl_quotes: Replace curly quotes with straight ones.
        fix_line_breaks: Replace ``\r\n``, ``\r`` and Unicode line breaks
            with ``\n``.
        fix_surrogates: Join UTF-16 surrogate pairs into the characters they
            encode, and replace lone surrogates with ``�``.
        remove_control_chars: Remove control characters that have no visible
            effect on text.
        normalization: The Unicode normalization form applied last, or ``None``
            for none.
        max_decode_length: The longest segment, in code points, that
            :func:`fix_text` fixes at once. Longer lines are split.
        explain: Record the steps taken. Functions that return no explanation
            default to ``False``.
    """

    unescape_html: str | bool = "auto"
    remove_terminal_escapes: bool = True
    fix_encoding: bool = True
    restore_byte_a0: bool = True
    replace_lossy_sequences: bool = True
    decode_inconsistent_utf8: bool = True
    fix_c1_controls: bool = True
    fix_latin_ligatures: bool = True
    fix_character_width: bool = True
    uncurl_quotes: bool = True
    fix_line_breaks: bool = True
    fix_surrogates: bool = True
    remove_control_chars: bool = True
    normalization: Literal["NFC", "NFD", "NFKC", "NFKD"] | None = "NFC"
    max_decode_length: int = 1000000
    explain: bool = True

def fix_text(text: str, config: TextFixerConfig | None = None, **kwargs: Any) -> str:
    """Fix mojibake and other problems in ``text``, as ``ftfy.fix_text`` does.

    The text is fixed line by line, in segments of at most
    ``max_decode_length`` code points, each as :func:`fix_text_segment` fixes
    it. With ``unescape_html="auto"``, HTML entities are decoded until the first
    line that contains ``<``.

    >>> fix_text("naÃ¯ve rÃ©sumÃ©")
    'naïve résumé'

    Args:
        text: The text to fix.
        config: The options (default: ``TextFixerConfig(explain=False)``).
        **kwargs: :class:`TextFixerConfig` fields that override ``config``.

    Returns:
        The fixed text.

    Raises:
        UnicodeError: ``text`` is ``bytes`` (decode it first).
        TypeError: ``text`` is not a str.
        ValueError: ``max_decode_length`` is below 1.
    """

def ftfy(text: str, config: TextFixerConfig | None = None, **kwargs: Any) -> str:
    """Fix ``text``: ftfy's alias for :func:`fix_text`."""

@overload
def fix_text_batch(
    texts: str, config: TextFixerConfig | None = None, *, threads: int | None = None, **kwargs: Any
) -> str:
    """:func:`fix_text` over many documents at once, on pyroid's thread pool.

    Not part of ftfy. Each result equals ``ftfy.fix_text(doc, config, **kwargs)``;
    documents that need no fixing come back as the original objects, and Arrow
    nulls stay null. Options the Rust port does not handle are delegated to
    ftfy for str/list/tuple input and raise ``pyroid.PyroidError`` for Arrow.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column
            (``string``, ``large_string`` or ``string_view``). Lone surrogates
            in a str are fixed as ftfy fixes them.
        config: As for :func:`fix_text` (default: ``explain=False``).
        threads: ``None`` uses pyroid's shared thread pool; ``n`` runs at most
            ``n`` tasks at once; ``1`` runs on the calling thread.
        **kwargs: Config fields, as for :func:`fix_text`.

    Returns:
        The same kind as ``texts``: str, list, tuple, or a :class:`pyroid.Column`
        of strings (``large_string`` for ``large_string`` input, else ``string``).

    Raises:
        pyroid.InputError: ``texts`` is not one of the accepted kinds, or holds
            a non-str item or a non-string Arrow type.
        UnicodeDecodeError: An Arrow row is not valid UTF-8.
    """

@overload
def fix_text_batch(
    texts: list[str],
    config: TextFixerConfig | None = None,
    *,
    threads: int | None = None,
    **kwargs: Any,
) -> list[str]: ...
@overload
def fix_text_batch(
    texts: tuple[str, ...],
    config: TextFixerConfig | None = None,
    *,
    threads: int | None = None,
    **kwargs: Any,
) -> tuple[str, ...]: ...
@overload
def fix_text_batch(
    texts: ArrowStrings,
    config: TextFixerConfig | None = None,
    *,
    threads: int | None = None,
    **kwargs: Any,
) -> Column[str]: ...
def fix_and_explain(
    text: str, config: TextFixerConfig | None = None, **kwargs: Any
) -> ExplainedText:
    """Fix ``text`` as one segment and say how, as ``ftfy.fix_and_explain`` does.

    Unlike :func:`fix_text`, the text is not split into lines.

    Args:
        text: The text to fix.
        config: The options (default: ``TextFixerConfig()``, with ``explain=True``).
        **kwargs: :class:`TextFixerConfig` fields that override ``config``.

    Returns:
        The fixed text and the steps taken (``None`` with ``explain=False``).

    Raises:
        UnicodeError: ``text`` is ``bytes``.
        ValueError: ``max_decode_length`` is below 1.
    """

def fix_encoding_and_explain(
    text: str, config: TextFixerConfig | None = None, **kwargs: Any
) -> ExplainedText:
    """Fix only the mojibake in ``text`` and say how, as ``ftfy.fix_encoding_and_explain`` does.

    Of the options, only ``fix_encoding`` and those that shape it apply
    (``restore_byte_a0``, ``replace_lossy_sequences``,
    ``decode_inconsistent_utf8``, ``fix_c1_controls``). With
    ``fix_encoding=False`` the text comes back unchanged, with no steps.

    Args:
        text: The text to fix.
        config: The options (default: ``TextFixerConfig()``).
        **kwargs: :class:`TextFixerConfig` fields that override ``config``.

    Returns:
        The fixed text and the steps taken (``None`` with ``explain=False``).

    Raises:
        UnicodeError: ``text`` is ``bytes``.
    """

def fix_encoding(text: str, config: TextFixerConfig | None = None, **kwargs: Any) -> str:
    """Fix only the mojibake in ``text``, as ``ftfy.fix_encoding`` does.

    >>> fix_encoding("Ã³")
    'ó'

    This is :func:`fix_encoding_and_explain` without the steps.
    """

def fix_text_segment(text: str, config: TextFixerConfig | None = None, **kwargs: Any) -> str:
    """Fix ``text`` as one segment, as ``ftfy.fix_text_segment`` does.

    This is :func:`fix_and_explain` without the steps.
    """

def apply_plan(text: str, plan: list[tuple[str, str]]) -> str:
    """Replay ``plan``, an explanation's steps, on ``text``, as ``ftfy.apply_plan`` does.

    Delegated to ftfy, with a one-time :class:`pyroid.FallbackWarning`.

    Raises:
        pyroid.PyroidError: ftfy is not installed.
    """

def guess_bytes(bstring: bytes) -> tuple[str, str]:
    """Guess the encoding of ``bstring`` and decode it, as ``ftfy.guess_bytes`` does.

    Delegated to ftfy, with a one-time :class:`pyroid.FallbackWarning`.

    Returns:
        The decoded text and the name of the encoding.

    Raises:
        pyroid.PyroidError: ftfy is not installed.
    """

def explain_unicode(text: str) -> None:
    """Print each code point of ``text`` with its name, as ``ftfy.explain_unicode`` does.

    Delegated to ftfy, with a one-time :class:`pyroid.FallbackWarning`.

    Raises:
        pyroid.PyroidError: ftfy is not installed.
    """

def fix_file(
    input_file: TextIO | BinaryIO,
    encoding: str | None = None,
    config: TextFixerConfig | None = None,
    **kwargs: Any,
) -> Iterator[str]:
    """Fix a file line by line, as ``ftfy.fix_file`` does.

    Delegated to ftfy, with a one-time :class:`pyroid.FallbackWarning`.

    Args:
        input_file: A file open in text or binary mode.
        encoding: The encoding of a binary file (default: guessed from its first
            line with :func:`guess_bytes`).
        config: The options (default: ``TextFixerConfig()``).
        **kwargs: :class:`TextFixerConfig` fields that override ``config``.

    Yields:
        The fixed lines.

    Raises:
        pyroid.PyroidError: ftfy is not installed.
    """
