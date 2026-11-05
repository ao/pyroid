# Derived from ftfy 6.3.1 (Apache-2.0), © Robyn Speer; modified.
# Type stubs for pyroid.ftfy: the signatures of ftfy 6.3.1. The module
# docstring is kept here on purpose: the API docs are built from this stub.
# ruff: noqa: PYI021
"""A drop-in, Rust-backed replacement for ftfy 6.3.1 (``import pyroid.ftfy as ftfy``).

Results are byte-identical to ftfy 6.3.1 on the same CPython version.
``apply_plan``, ``guess_bytes``, ``fix_file`` and ``explain_unicode``, and
option values the Rust port does not handle (an unknown ``normalization``
form, a non-integer ``max_decode_length``), are delegated to the real ftfy
with a one-time warning, or raise ``pyroid.PyroidError`` if it is not
installed.

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
from typing import Any, BinaryIO, Final, Literal, NamedTuple, TextIO

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
