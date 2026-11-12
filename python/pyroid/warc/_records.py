# Derived from warcio 1.8.2 (Apache-2.0), © Webrecorder, Ilya Kreymer et al.: statusandheaders.py,
# recordloader.py (ArcWarcRecord), digestverifyingreader.py (DigestChecker), exceptions.py and
# utils.py (to_native_str, headers_to_str_headers); modified.
"""warcio 1.8.2's record classes, ported: what ``pyroid.warc.ArchiveIterator`` yields."""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import quote

__all__ = [
    "ArcWarcRecord",
    "ArchiveLoadFailed",
    "DigestChecker",
    "StatusAndHeaders",
    "StatusAndHeadersParserException",
    "headers_to_str_headers",
    "supported_encodings",
    "to_native_str",
]


def to_native_str(value: Any, encoding: str = "utf-8") -> Any:
    """``value`` as a str: bytes are decoded, anything else is returned as is."""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode(encoding)
    return value


def headers_to_str_headers(headers: Any) -> list[tuple[Any, Any]]:
    """Dict or tuple headers of bytes or str as a new list of str tuples (latin-1 for bytes)."""
    ret = []
    h = headers.items() if isinstance(headers, Mapping) else headers
    for tup in h:
        k, v = tup
        if isinstance(k, bytes):
            k = k.decode("iso-8859-1")
        if isinstance(v, bytes):
            v = v.decode("iso-8859-1")
        ret.append((k, v))
    return ret


class ArchiveLoadFailed(Exception):  # noqa: N818 - warcio's name
    """The archive could not be read (warcio's ``warcio.exceptions.ArchiveLoadFailed``)."""

    def __init__(self, reason: object) -> None:
        self.msg = str(reason)
        super().__init__(self.msg)


class StatusAndHeadersParserException(Exception):  # noqa: N818 - warcio's name
    """A status line or headers block that does not parse (``verify_http=True``)."""

    def __init__(self, msg: str, statusline: Any) -> None:
        super().__init__(msg)
        self.statusline = statusline


class StatusAndHeaders:
    """A status line and a list of ``(name, value)`` headers (warcio's ``StatusAndHeaders``)."""

    ENCODE_HEADER_RX = re.compile(r'[=]["\']?([^;"]+)["\']?(?=[;]?)')

    statusline: str
    headers: list[tuple[str, str]]
    protocol: str
    total_len: int
    headers_buff: bytes | None

    def __init__(
        self,
        statusline: str,
        headers: Any,
        protocol: str = "",
        total_len: int = 0,
        is_http_request: bool = False,
    ) -> None:
        if is_http_request:
            protocol, statusline = statusline.split(" ", 1)

        self.statusline = statusline
        self.headers = headers_to_str_headers(headers)
        self.protocol = protocol
        self.total_len = total_len
        self.headers_buff = None

    @classmethod
    def _from_native(
        cls, statusline: str, headers: list[tuple[str, str]], protocol: str, total_len: int
    ) -> StatusAndHeaders:
        # pyroid's reader builds a fresh list of str tuples: nothing for __init__ to convert.
        self = cls.__new__(cls)
        self.statusline = statusline
        self.headers = headers
        self.protocol = protocol
        self.total_len = total_len
        self.headers_buff = None
        return self

    def get_header(self, name: str, default_value: Any = None) -> Any:
        """The value of the first header named ``name`` (any case), else ``default_value``."""
        name_lower = name.lower()
        for value in self.headers:
            if value[0].lower() == name_lower:
                return value[1]

        return default_value

    def add_header(self, name: str, value: str) -> None:
        """Append a header."""
        self.headers.append((name, value))

    def replace_header(self, name: str, value: str) -> str | None:
        """Set the last header named ``name`` (any case) to ``value``, or add it.

        Returns the old value, if any.
        """
        name_lower = name.lower()
        for index in range(len(self.headers) - 1, -1, -1):
            curr_name, curr_value = self.headers[index]
            if curr_name.lower() == name_lower:
                self.headers[index] = (curr_name, value)
                return curr_value

        self.headers.append((name, value))
        return None

    def remove_header(self, name: str) -> bool:
        """Remove the last header named ``name`` (any case); whether there was one."""
        name_lower = name.lower()
        for index in range(len(self.headers) - 1, -1, -1):
            if self.headers[index][0].lower() == name_lower:
                del self.headers[index]
                return True

        return False

    def get_statuscode(self) -> str:
        """The status code part of the status line."""
        code = self.statusline.split(" ", 1)[0]
        return code

    def validate_statusline(self, valid_statusline: str) -> bool:
        """Whether the status line starts with a positive code; if not, replace it."""
        code = self.get_statuscode()
        try:
            code_int = int(code)
            assert code_int > 0
            return True
        except (ValueError, AssertionError):
            self.statusline = valid_statusline
            return False

    def add_range(self, start: int, part_len: int, total_len: int) -> StatusAndHeaders:
        """Make this a ``206 Partial Content`` response for the given range."""
        content_range = f"bytes {start}-{start + part_len - 1}/{total_len}"

        self.statusline = "206 Partial Content"
        self.replace_header("Content-Range", content_range)
        self.replace_header("Content-Length", str(part_len))
        self.replace_header("Accept-Ranges", "bytes")
        return self

    def compute_headers_buffer(self, header_filter: Callable[..., Any] | None = None) -> None:
        """Set ``headers_buff`` to the headers as ASCII bytes."""
        self.headers_buff = self.to_ascii_bytes(header_filter)

    def __repr__(self) -> str:
        return (
            f"StatusAndHeaders(protocol = '{self.protocol}', statusline = '{self.statusline}', "
            f"headers = {self.headers})"
        )

    def __ne__(self, other: object) -> bool:
        return not (self == other)

    def __eq__(self, other: object) -> bool:
        if not other:
            return False

        return bool(
            self.statusline == other.statusline  # type: ignore[attr-defined]
            and self.headers == other.headers  # type: ignore[attr-defined]
            and self.protocol == other.protocol  # type: ignore[attr-defined]
        )

    def __str__(self, exclude_list: Any = None) -> str:
        return self.to_str(exclude_list)

    def __bool__(self) -> bool:
        return bool(self.statusline or self.headers)

    __nonzero__ = __bool__

    def to_str(self, filter_func: Callable[..., Any] | None = None) -> str:
        r"""The status line and headers, each line ending in ``\r\n``."""
        string = self.protocol

        if string and self.statusline:
            string += " "

        if self.statusline:
            string += self.statusline

        if string:
            string += "\r\n"

        for h in self.headers:
            if filter_func:
                h = filter_func(h)
                if not h:
                    continue

            string += ": ".join(h) + "\r\n"

        return string

    def to_bytes(
        self, filter_func: Callable[..., Any] | None = None, encoding: str = "utf-8"
    ) -> bytes:
        """``to_str()`` encoded, plus the blank line."""
        return self.to_str(filter_func).encode(encoding) + b"\r\n"

    def to_ascii_bytes(self, filter_func: Callable[..., Any] | None = None) -> bytes:
        """``to_bytes()`` in ASCII, %-encoding non-ASCII header values per RFC 8187."""
        try:
            string = self.to_str(filter_func)
            raw = string.encode("ascii")
        except (UnicodeEncodeError, UnicodeDecodeError):
            self.percent_encode_non_ascii_headers()
            string = self.to_str(filter_func)
            raw = string.encode("ascii")

        return raw + b"\r\n"

    def percent_encode_non_ascii_headers(self, encoding: str = "UTF-8") -> None:
        """%-encode the header values that are not ASCII (RFC 8187 / RFC 5987)."""

        def do_encode(m: re.Match[str]) -> str:
            return f"*={encoding}''" + quote(to_native_str(m.group(1)))

        for index in range(len(self.headers) - 1, -1, -1):
            curr_name, curr_value = self.headers[index]
            try:
                # test if header is ascii encodable, no action needed
                curr_value.encode("ascii")
            except Exception:
                # if single value header, (eg. no ';'), %-encode entire header
                if ";" not in curr_value:
                    new_value = quote(curr_value)

                else:
                    # %-encode value in ; name="value"
                    new_value = self.ENCODE_HEADER_RX.sub(do_encode, curr_value)
                    if new_value == curr_value:
                        new_value = quote(curr_value)

                self.headers[index] = (curr_name, new_value)

    # act like a (case-insensitive) dictionary of headers, much like other
    # python http headers apis including http.client.HTTPMessage
    # and requests.structures.CaseInsensitiveDict
    get = get_header
    __getitem__ = get_header
    __setitem__ = replace_header
    __delitem__ = remove_header

    def __contains__(self, key: str) -> bool:
        return bool(self[key])


class DigestChecker:
    """The digest checks of one record (warcio's ``DigestChecker``).

    ``kind`` is ``ArchiveIterator``'s ``check_digests`` argument: ``'raise'`` raises
    :class:`ArchiveLoadFailed` at the first problem, ``'log'`` also writes it to stderr.
    """

    def __init__(self, kind: Any = None) -> None:
        self._problem: list[str] = []
        self._passed: bool | None = None
        self.kind = kind

    @property
    def passed(self) -> bool | None:
        """``True`` when every digest checked so far matched, ``False`` after a problem."""
        return self._passed

    @passed.setter
    def passed(self, value: bool | None) -> None:
        self._passed = value

    @property
    def problems(self) -> list[str]:
        """The problems found so far."""
        return self._problem

    def problem(self, value: str, passed: bool = False) -> None:
        """Record a problem (raise or log it, as ``kind`` says)."""
        self._problem.append(value)
        if self.kind == "raise":
            raise ArchiveLoadFailed(value)
        if self.kind == "log":
            sys.stderr.write(value + "\n")
        self._passed = passed


_ENCODINGS: list[frozenset[str]] = []


def _brotli_ok() -> bool:
    # warcio's try_brotli_init: brotlicffi importable and not 1.0.3.0 <= version < 1.2.1.0.
    import importlib

    try:
        brotlicffi = importlib.import_module("brotlicffi")
    except ImportError:
        return False
    version = getattr(brotlicffi, "__version__", "")
    version_tuple = tuple(int(x) for x in re.findall(r"\d+", version))
    return not ((1, 0, 3, 0) <= version_tuple < (1, 2, 1, 0))


def supported_encodings() -> frozenset[str]:
    """``BufferedReader.get_supported_decompressors()``: the ``Content-Encoding`` values decoded."""
    if not _ENCODINGS:
        names = {"gzip", "deflate", "deflate_alt"}
        if _brotli_ok():
            names.add("br")
        _ENCODINGS.append(frozenset(names))
    return _ENCODINGS[0]


class ArcWarcRecord:
    """A WARC or ARC record (warcio's ``ArcWarcRecord``).

    Attributes: ``format`` (``'warc'`` or ``'arc'``), ``rec_type``, ``rec_headers``
    (:class:`StatusAndHeaders`), ``raw_stream``, ``http_headers`` (or ``None``),
    ``content_type``, ``length``, ``payload_length`` and ``digest_checker``.
    """

    format: str
    rec_type: str | None
    rec_headers: Any
    raw_stream: Any
    http_headers: Any
    content_type: str | None
    length: int | None
    payload_length: int
    digest_checker: Any

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        (
            self.format,
            self.rec_type,
            self.rec_headers,
            self.raw_stream,
            self.http_headers,
            self.content_type,
            self.length,
        ) = args
        self.payload_length = kwargs.get("payload_length", -1)
        self.digest_checker = kwargs.get("digest_checker")

    @classmethod
    def _from_native(
        cls,
        format: str,
        rec_type: str | None,
        rec_headers: Any,
        raw_stream: Any,
        http_headers: Any,
        content_type: str | None,
        length: int | None,
        payload_length: int,
        digest_checker: Any,
    ) -> ArcWarcRecord:
        self = cls.__new__(cls)
        self.format = format
        self.rec_type = rec_type
        self.rec_headers = rec_headers
        self.raw_stream = raw_stream
        self.http_headers = http_headers
        self.content_type = content_type
        self.length = length
        self.payload_length = payload_length
        self.digest_checker = digest_checker
        return self

    def content_stream(self) -> Any:
        """The payload, decoded: de-chunked and decompressed per the HTTP headers.

        Returns ``raw_stream`` itself when there is nothing to decode, else a new decoder
        over it on each call, as warcio.
        """
        http_headers = self.http_headers
        if not http_headers:
            return self.raw_stream

        encoding = http_headers.get_header("content-encoding")

        if encoding:
            encoding = encoding.lower()

            # 'x-gzip' is a legacy alias for 'gzip' (RFC 7230 4.2.3, RFC 2616
            # 3.5); treat it the same so an x-gzip payload is still decompressed.
            if encoding == "x-gzip":
                encoding = "gzip"

            if encoding not in supported_encodings():
                encoding = None

        chunked = http_headers.get_header("transfer-encoding") == "chunked"
        if chunked or encoding:
            decoder = self.raw_stream._decoder(chunked, encoding or None)
            if decoder is not None:
                return decoder
        return self.raw_stream
