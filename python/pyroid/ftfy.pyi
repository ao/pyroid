# Derived from ftfy 6.3.1 (Apache-2.0), © Robyn Speer; modified.
# Type stubs for pyroid.ftfy: the signatures of ftfy 6.3.1.

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
