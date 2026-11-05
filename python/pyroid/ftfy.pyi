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

FTFY_VERSION: Final[str]
DELEGATED: Final[tuple[str, ...]]
BYTES_ERROR_TEXT: Final[str]
_UNICODE_VERSION: Final[str]

class ExplanationStep(NamedTuple):
    action: str
    parameter: str

class ExplainedText(NamedTuple):
    text: str
    explanation: list[ExplanationStep] | None

class TextFixerConfig(NamedTuple):
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

def fix_text(text: str, config: TextFixerConfig | None = None, **kwargs: Any) -> str: ...
def ftfy(text: str, config: TextFixerConfig | None = None, **kwargs: Any) -> str: ...
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
) -> ExplainedText: ...
def fix_encoding_and_explain(
    text: str, config: TextFixerConfig | None = None, **kwargs: Any
) -> ExplainedText: ...
def fix_encoding(text: str, config: TextFixerConfig | None = None, **kwargs: Any) -> str: ...
def fix_text_segment(text: str, config: TextFixerConfig | None = None, **kwargs: Any) -> str: ...
def apply_plan(text: str, plan: list[tuple[str, str]]) -> str: ...
def guess_bytes(bstring: bytes) -> tuple[str, str]: ...
def explain_unicode(text: str) -> None: ...
def fix_file(
    input_file: TextIO | BinaryIO,
    encoding: str | None = None,
    config: TextFixerConfig | None = None,
    **kwargs: Any,
) -> Iterator[str]: ...
