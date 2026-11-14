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

import sys
import unicodedata
import warnings
from collections.abc import Iterator
from typing import Any, BinaryIO, Literal, NamedTuple, TextIO

from pyroid import _versions
from pyroid._pyroid import FallbackWarning, PyroidError
from pyroid._pyroid import ftfy as _native

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
FTFY_VERSION = "6.3.1"

_versions.check(__name__, "ftfy", f"ftfy=={FTFY_VERSION}")

#: Functions that are forwarded to the real ftfy (it must be installed).
DELEGATED = ("apply_plan", "explain_unicode", "fix_file", "guess_bytes")


class ExplanationStep(NamedTuple):
    """One step of an :class:`ExplainedText` explanation, as in ftfy.

    Attributes:
        action: ``"encode"``, ``"decode"``, ``"transcode"``, ``"apply"`` or
            ``"normalize"``.
        parameter: The encoding, or the name of the fixer or normalization form.
    """

    action: str
    parameter: str

    def __repr__(self) -> str:
        return repr(tuple(self))


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


def _caller_stacklevel() -> int:
    """The `stacklevel` that points our caller's `warnings.warn` at the user's line.

    That is the first frame outside this module, however many of our functions are in between
    (e.g. `fix_encoding` -> `fix_encoding_and_explain` -> `_ftfy`). `skip_file_prefixes` would
    do this, but needs Python 3.12.
    """
    frame = sys._getframe(1)  # the function that calls warnings.warn: stacklevel 1
    level = 1
    while frame.f_back is not None and frame.f_globals.get("__name__") == __name__:
        frame = frame.f_back
        level += 1
    return level


def _ftfy(name: str, reason: str) -> Any:
    """Import the real ftfy for `name`, warning once, or raise PyroidError."""
    try:
        import ftfy as real
    except ImportError:
        real = None
    # Under pyroid.integrations.patch("ftfy"), `import ftfy` gives pyroid's own shim: fall back to
    # the real ftfy it wraps (None if ftfy is not installed), never to the shim (that recurses).
    if getattr(real, "__pyroid__", False):
        real = real.__wrapped__
    if real is None:
        raise PyroidError(
            f"pyroid.ftfy.{name}: {reason} is not supported by pyroid's Rust port, and "
            "ftfy is not installed to fall back to (pip install ftfy==6.3.1)"
        )
    if f"{name}:{reason}" not in _WARNED:
        _WARNED.add(f"{name}:{reason}")
        warnings.warn(
            f"pyroid.ftfy.{name}: {reason} is not supported by the Rust port; "
            "delegating to ftfy (slow path). This warning is shown once.",
            FallbackWarning,
            stacklevel=_caller_stacklevel(),
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
    """Fix only the mojibake in ``text``, as ``ftfy.fix_encoding`` does.

    >>> fix_encoding("Ã³")
    'ó'

    This is :func:`fix_encoding_and_explain` without the steps.
    """
    if config is None:
        config = TextFixerConfig(explain=False)
    config = _config_from_kwargs(config, kwargs)
    return fix_encoding_and_explain(text, config).text


def fix_text_segment(text: str, config: TextFixerConfig | None = None, **kwargs: Any) -> str:
    """Fix ``text`` as one segment, as ``ftfy.fix_text_segment`` does.

    This is :func:`fix_and_explain` without the steps.
    """
    if config is None:
        config = TextFixerConfig(explain=False)
    config = _config_from_kwargs(config, kwargs)
    return fix_and_explain(text, config).text


# ---------------------------------------------------------------------------
# Delegated to the real ftfy
# ---------------------------------------------------------------------------


def apply_plan(text: str, plan: list[tuple[str, str]]) -> str:
    """Replay ``plan``, an explanation's steps, on ``text``, as ``ftfy.apply_plan`` does.

    Delegated to ftfy, with a one-time :class:`pyroid.FallbackWarning`.

    Raises:
        pyroid.PyroidError: ftfy is not installed.
    """
    return _ftfy("apply_plan", "apply_plan").apply_plan(text, plan)  # type: ignore[no-any-return]


def guess_bytes(bstring: bytes) -> tuple[str, str]:
    """Guess the encoding of ``bstring`` and decode it, as ``ftfy.guess_bytes`` does.

    Delegated to ftfy, with a one-time :class:`pyroid.FallbackWarning`.

    Returns:
        The decoded text and the name of the encoding.

    Raises:
        pyroid.PyroidError: ftfy is not installed.
    """
    return _ftfy("guess_bytes", "guess_bytes").guess_bytes(bstring)  # type: ignore[no-any-return]


def explain_unicode(text: str) -> None:
    """Print each code point of ``text`` with its name, as ``ftfy.explain_unicode`` does.

    Delegated to ftfy, with a one-time :class:`pyroid.FallbackWarning`.

    Raises:
        pyroid.PyroidError: ftfy is not installed.
    """
    _ftfy("explain_unicode", "explain_unicode").explain_unicode(text)


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
    real = _ftfy("fix_file", "fix_file")
    return real.fix_file(input_file, encoding, config, **kwargs)  # type: ignore[no-any-return]
