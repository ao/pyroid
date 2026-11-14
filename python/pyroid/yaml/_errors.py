# Derived from PyYAML 6.0.3 (MIT), © Kirill Simonov and Ingy döt Net; modified.
"""PyYAML's exception classes and ``Mark``, for when PyYAML is not installed.

With PyYAML installed, :mod:`pyroid.yaml` uses PyYAML's own classes instead, so
``except yaml.YAMLError`` catches errors from both.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ComposerError",
    "ConstructorError",
    "EmitterError",
    "Mark",
    "MarkedYAMLError",
    "ParserError",
    "ReaderError",
    "RepresenterError",
    "ResolverError",
    "ScannerError",
    "SerializerError",
    "YAMLError",
]


class Mark:
    """A position in a YAML stream (PyYAML's ``yaml.error.Mark``)."""

    def __init__(
        self, name: Any, index: int, line: int, column: int, buffer: str | None, pointer: int | None
    ) -> None:
        self.name = name
        self.index = index
        self.line = line
        self.column = column
        self.buffer = buffer
        self.pointer = pointer

    def get_snippet(self, indent: int = 4, max_length: int = 75) -> str | None:
        if self.buffer is None or self.pointer is None:
            return None
        head = ""
        start = self.pointer
        while start > 0 and self.buffer[start - 1] not in "\0\r\n\x85\u2028\u2029":
            start -= 1
            if self.pointer - start > max_length / 2 - 1:
                head = " ... "
                start += 5
                break
        tail = ""
        end = self.pointer
        while end < len(self.buffer) and self.buffer[end] not in "\0\r\n\x85\u2028\u2029":
            end += 1
            if end - self.pointer > max_length / 2 - 1:
                tail = " ... "
                end -= 5
                break
        snippet = self.buffer[start:end]
        return (
            " " * indent
            + head
            + snippet
            + tail
            + "\n"
            + " " * (indent + self.pointer - start + len(head))
            + "^"
        )

    def __str__(self) -> str:
        snippet = self.get_snippet()
        where = '  in "%s", line %d, column %d' % (  # noqa: UP031 - PyYAML's formatting
            self.name,
            self.line + 1,
            self.column + 1,
        )
        if snippet is not None:
            where += ":\n" + snippet
        return where


class YAMLError(Exception):
    """Base class of the YAML errors."""


class MarkedYAMLError(YAMLError):
    """An error with a context, a problem and where they are in the stream."""

    def __init__(
        self,
        context: str | None = None,
        context_mark: Mark | None = None,
        problem: str | None = None,
        problem_mark: Mark | None = None,
        note: str | None = None,
    ) -> None:
        self.context = context
        self.context_mark = context_mark
        self.problem = problem
        self.problem_mark = problem_mark
        self.note = note

    def __str__(self) -> str:
        lines = []
        if self.context is not None:
            lines.append(self.context)
        if self.context_mark is not None and (
            self.problem is None
            or self.problem_mark is None
            or self.context_mark.name != self.problem_mark.name
            or self.context_mark.line != self.problem_mark.line
            or self.context_mark.column != self.problem_mark.column
        ):
            lines.append(str(self.context_mark))
        if self.problem is not None:
            lines.append(self.problem)
        if self.problem_mark is not None:
            lines.append(str(self.problem_mark))
        if self.note is not None:
            lines.append(self.note)
        return "\n".join(lines)


class ReaderError(YAMLError):
    """The stream holds bytes that do not decode or characters YAML does not allow."""

    def __init__(
        self, name: Any, position: int, character: Any, encoding: str, reason: str
    ) -> None:
        self.name = name
        self.character = character
        self.position = position
        self.encoding = encoding
        self.reason = reason

    def __str__(self) -> str:
        if isinstance(self.character, bytes):
            return "'%s' codec can't decode byte #x%02x: %s\n  in \"%s\", position %d" % (  # noqa: UP031
                self.encoding,
                ord(self.character),
                self.reason,
                self.name,
                self.position,
            )
        return 'unacceptable character #x%04x: %s\n  in "%s", position %d' % (  # noqa: UP031
            self.character,
            self.reason,
            self.name,
            self.position,
        )


class ScannerError(MarkedYAMLError):
    """The scanner could not make a token."""


class ParserError(MarkedYAMLError):
    """The tokens do not form a YAML document."""


class ComposerError(MarkedYAMLError):
    """Anchors, aliases or the document count are wrong."""


class ConstructorError(MarkedYAMLError):
    """A node could not be turned into a Python object."""


class ResolverError(YAMLError):
    """A bad resolver registration."""


class RepresenterError(YAMLError):
    """An object could not be represented as YAML."""


class SerializerError(YAMLError):
    """The serializer was used out of order."""


class EmitterError(YAMLError):
    """The emitter got events it cannot write."""


for _cls in (
    Mark,
    YAMLError,
    MarkedYAMLError,
    ReaderError,
    ScannerError,
    ParserError,
    ComposerError,
    ConstructorError,
    ResolverError,
    RepresenterError,
    SerializerError,
    EmitterError,
):
    _cls.__module__ = "pyroid.yaml"
del _cls
