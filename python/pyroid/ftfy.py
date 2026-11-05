# Derived from ftfy 6.3.1 (Apache-2.0), © Robyn Speer; modified.
"""A drop-in, Rust-backed replacement for ftfy 6.3.1 (``import pyroid.ftfy as ftfy``).

The functions here have ftfy's signatures and return byte-identical results on
the same CPython version. The work is done by a Rust port of ftfy; ftfy itself
is only needed for the few features listed in ``DELEGATED`` and for option
values the port does not handle (an unknown ``normalization`` form, a
non-integer ``max_decode_length``). Those calls go to the real ftfy, with a
one-time :class:`pyroid.FallbackWarning`, or raise :class:`pyroid.PyroidError`
if ftfy is not installed.
"""

from __future__ import annotations

import unicodedata
import warnings
from collections.abc import Iterator
from typing import Any, BinaryIO, Literal, NamedTuple, TextIO

from pyroid._pyroid import FallbackWarning, PyroidError
from pyroid._pyroid import ftfy as _native

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

#: The ftfy release whose behaviour this module reproduces.
FTFY_VERSION = "6.3.1"

#: Functions that are forwarded to the real ftfy (it must be installed).
DELEGATED = ("apply_plan", "explain_unicode", "fix_file", "guess_bytes")


class ExplanationStep(NamedTuple):
    """A step in an ExplainedText, explaining how to decode text (as in ftfy)."""

    action: str
    parameter: str

    def __repr__(self) -> str:
        return repr(tuple(self))


class ExplainedText(NamedTuple):
    """Fixed text and the steps that fixed it (``None`` when ``explain`` is off)."""

    text: str
    explanation: list[ExplanationStep] | None


class TextFixerConfig(NamedTuple):
    """ftfy's configuration object, with the same fields and defaults.

    See ftfy's documentation for the meaning of each option. As in ftfy,
    instances of ``ftfy.TextFixerConfig`` are accepted too.
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


BYTES_ERROR_TEXT = """Hey wait, this isn't Unicode.

ftfy is designed to fix problems with text. Treating bytes like they're
interchangeable with Unicode text is usually something that introduces
problems with text.

You should first decode these bytes from the encoding you think they're in.
If you're not sure what encoding they're in:

- First, try to find out. 'utf-8' is a good assumption.
- If the encoding is simply unknowable, try running your bytes through
  ftfy.guess_bytes. As the name implies, this may not always be accurate.

For more information on the distinction between bytes and text, read the
Python Unicode HOWTO:

    http://docs.python.org/3/howto/unicode.html
"""

_FIELDS = TextFixerConfig._fields
_BOOL_FIELDS = tuple(
    f for f in _FIELDS if f not in ("unescape_html", "normalization", "max_decode_length")
)
_FORMS = ("NFC", "NFD", "NFKC", "NFKD")
_MAX_LENGTH = 2**62


def _unicode_version() -> str:
    """The interpreter's Unicode version, or the newest one the kernel knows."""
    version = unicodedata.unidata_version
    supported = _native.supported_unicode_versions()
    if version in supported:
        return version
    newest = supported[-1]
    warnings.warn(
        f"pyroid.ftfy has no tables for Unicode {version} (this Python's unicodedata); "
        f"using Unicode {newest}. Output can differ from ftfy for characters added "
        f"after {newest}.",
        RuntimeWarning,
        stacklevel=3,
    )
    return newest


_UNICODE_VERSION = _unicode_version()


class _UnsupportedError(Exception):
    """A config value the Rust port does not handle; delegate to ftfy."""


def _native_config(config: Any) -> Any:
    """Translate a TextFixerConfig (ours or ftfy's) to a native config."""
    values = {f: getattr(config, f) for f in _FIELDS}
    unescape_value = values["unescape_html"]
    # ftfy tests `== "auto"` and then truthiness.
    unescape = "auto" if unescape_value == "auto" else ("always" if unescape_value else "never")
    norm = values["normalization"]
    if norm is not None and not (isinstance(norm, str) and norm in _FORMS):
        raise _UnsupportedError(f"normalization={norm!r}")
    length = values["max_decode_length"]
    if not isinstance(length, int):
        raise _UnsupportedError(f"max_decode_length={length!r}")
    if length < 1:
        # ftfy loops forever on such a value; refuse instead of hanging.
        raise ValueError(f"max_decode_length must be at least 1, not {length!r}")
    return _native._NativeConfig(
        unescape_html=unescape,
        normalization=norm,
        max_decode_length=min(length, _MAX_LENGTH),
        unicode_version=_UNICODE_VERSION,
        **{f: bool(values[f]) for f in _BOOL_FIELDS},
    )


_CACHE: dict[Any, Any] = {}


def _prepare(config: Any) -> Any:
    try:
        key = (type(config), tuple(getattr(config, f) for f in _FIELDS))
        hash(key)
    except TypeError:
        return _native_config(config)
    native = _CACHE.get(key)
    if native is None:
        native = _native_config(config)
        if len(_CACHE) > 256:
            _CACHE.clear()
        _CACHE[key] = native
    return native


_WARNED: set[str] = set()


def _ftfy(name: str, reason: str) -> Any:
    """Import the real ftfy for `name`, warning once, or raise PyroidError."""
    try:
        import ftfy as real
    except ImportError:
        raise PyroidError(
            f"pyroid.ftfy.{name}: {reason} is not supported by pyroid's Rust port, and "
            "ftfy is not installed to fall back to (pip install ftfy==6.3.1)"
        ) from None
    if real.__version__ != FTFY_VERSION and "version" not in _WARNED:
        _WARNED.add("version")
        warnings.warn(
            f"pyroid.ftfy is falling back to ftfy {real.__version__}, not {FTFY_VERSION}",
            RuntimeWarning,
            stacklevel=4,
        )
    if f"{name}:{reason}" not in _WARNED:
        _WARNED.add(f"{name}:{reason}")
        warnings.warn(
            f"pyroid.ftfy.{name}: {reason} is not supported by the Rust port; "
            "delegating to ftfy (slow path). This warning is shown once.",
            FallbackWarning,
            stacklevel=4,
        )
    return real


def _explained(result: Any) -> ExplainedText:
    text, steps = result
    if steps is None:
        return ExplainedText(text, None)
    return ExplainedText(text, [ExplanationStep(a, p) for a, p in steps])


def _config_from_kwargs(config: Any, kwargs: dict[str, Any]) -> Any:
    """Apply keyword arguments to a config, exactly as ftfy does."""
    if "fix_entities" in kwargs:
        warnings.warn(
            "`fix_entities` has been renamed to `unescape_html`",
            DeprecationWarning,
            stacklevel=3,
        )
        kwargs = kwargs.copy()
        kwargs["unescape_html"] = kwargs["fix_entities"]
        del kwargs["fix_entities"]
    return config._replace(**kwargs)


_DEFAULT_QUIET = _native_config(TextFixerConfig(explain=False))
_DEFAULT_EXPLAIN = _native_config(TextFixerConfig())


def fix_text(text: str, config: TextFixerConfig | None = None, **kwargs: Any) -> str:
    r"""Fix mojibake and other glitches in `text`, like ``ftfy.fix_text``.

        >>> fix_text('âœ” No problems')
        '✔ No problems'
        >>> fix_text('\uff2c\uff2f\uff35\uff24\u3000\uff2e\uff2f\uff29\uff33\uff25\uff33')
        'LOUD NOISES'

    The text is fixed line by line (segments of at most ``max_decode_length``
    code points). Options are given as a :class:`TextFixerConfig` or as keyword
    arguments.
    """
    if config is None and not kwargs and type(text) is str:
        return _native.fix_text(text, _DEFAULT_QUIET)  # type: ignore[no-any-return]
    if config is None:
        config = TextFixerConfig(explain=False)
    config = _config_from_kwargs(config, kwargs)
    if isinstance(text, bytes):
        raise UnicodeError(BYTES_ERROR_TEXT)
    try:
        native = _prepare(config)
    except _UnsupportedError as e:
        return _ftfy("fix_text", str(e)).fix_text(text, config)  # type: ignore[no-any-return]
    return _native.fix_text(text, native)  # type: ignore[no-any-return]


#: ftfy's alias for :func:`fix_text`.
ftfy = fix_text


def fix_text_batch(
    texts: Any, config: TextFixerConfig | None = None, *, threads: int | None = None, **kwargs: Any
) -> Any:
    """:func:`fix_text` over many documents at once, on pyroid's thread pool.

    Accepts a str, a list or tuple of str, or an Arrow string array/column and
    returns the same kind (a :class:`pyroid.Column` for Arrow); documents that
    need no fixing come back as the original objects. Every result equals
    ``ftfy.fix_text(doc, config, **kwargs)``. See ``ftfy.pyi`` for details.
    """
    if config is None and not kwargs:
        return _native.fix_text_batch(texts, _DEFAULT_QUIET, threads=threads)
    if config is None:
        config = TextFixerConfig(explain=False)
    config = _config_from_kwargs(config, kwargs)
    try:
        native = _prepare(config)
    except _UnsupportedError as e:
        if not isinstance(texts, (str, list, tuple)):
            raise PyroidError(
                f"pyroid.ftfy.fix_text_batch: {e} is not supported for Arrow (or other "
                "non-str/list/tuple) input; it is only delegated to ftfy for str, list and tuple"
            ) from None
        real = _ftfy("fix_text_batch", str(e))
        if isinstance(texts, str):
            return real.fix_text(texts, config)
        fixed = [real.fix_text(t, config) for t in texts]
        return fixed if isinstance(texts, list) else tuple(fixed)
    return _native.fix_text_batch(texts, native, threads=threads)


def fix_and_explain(
    text: str, config: TextFixerConfig | None = None, **kwargs: Any
) -> ExplainedText:
    """Fix `text` as one segment and explain what was done, like ``ftfy.fix_and_explain``."""
    if config is None:
        config = TextFixerConfig()
    if isinstance(text, bytes):
        raise UnicodeError(BYTES_ERROR_TEXT)
    config = _config_from_kwargs(config, kwargs)
    try:
        native = _prepare(config)
    except _UnsupportedError as e:
        return _explained(_ftfy("fix_and_explain", str(e)).fix_and_explain(text, config))
    return _explained(_native.fix_and_explain(text, native))


def fix_encoding_and_explain(
    text: str, config: TextFixerConfig | None = None, **kwargs: Any
) -> ExplainedText:
    """Fix only mojibake and explain how, like ``ftfy.fix_encoding_and_explain``."""
    if config is None:
        config = TextFixerConfig()
    if isinstance(text, bytes):
        raise UnicodeError(BYTES_ERROR_TEXT)
    config = _config_from_kwargs(config, kwargs)
    if not config.fix_encoding:
        return ExplainedText(text, [])
    try:
        native = _prepare(config)
    except _UnsupportedError as e:
        real = _ftfy("fix_encoding_and_explain", str(e))
        return _explained(real.fix_encoding_and_explain(text, config))
    return _explained(_native.fix_encoding_and_explain(text, native))


def fix_encoding(text: str, config: TextFixerConfig | None = None, **kwargs: Any) -> str:
    """Apply only the mojibake-fixing steps, like ``ftfy.fix_encoding``.

    >>> fix_encoding("Ã³")
    'ó'
    """
    if config is None:
        config = TextFixerConfig(explain=False)
    config = _config_from_kwargs(config, kwargs)
    return fix_encoding_and_explain(text, config).text


def fix_text_segment(text: str, config: TextFixerConfig | None = None, **kwargs: Any) -> str:
    """Fix `text` as one segment and discard the explanation, like ``ftfy.fix_text_segment``."""
    if config is None:
        config = TextFixerConfig(explain=False)
    config = _config_from_kwargs(config, kwargs)
    return fix_and_explain(text, config).text


# ---------------------------------------------------------------------------
# Delegated to the real ftfy
# ---------------------------------------------------------------------------


def apply_plan(text: str, plan: list[tuple[str, str]]) -> str:
    """Apply an explanation's plan to `text` (delegated to ftfy)."""
    return _ftfy("apply_plan", "apply_plan").apply_plan(text, plan)  # type: ignore[no-any-return]


def guess_bytes(bstring: bytes) -> tuple[str, str]:
    """Guess the encoding of `bstring` (delegated to ftfy)."""
    return _ftfy("guess_bytes", "guess_bytes").guess_bytes(bstring)  # type: ignore[no-any-return]


def explain_unicode(text: str) -> None:
    """Print the code points of `text` (delegated to ftfy)."""
    _ftfy("explain_unicode", "explain_unicode").explain_unicode(text)


def fix_file(
    input_file: TextIO | BinaryIO,
    encoding: str | None = None,
    config: TextFixerConfig | None = None,
    **kwargs: Any,
) -> Iterator[str]:
    """Fix the lines of a file (delegated to ftfy)."""
    real = _ftfy("fix_file", "fix_file")
    return real.fix_file(input_file, encoding, config, **kwargs)  # type: ignore[no-any-return]
