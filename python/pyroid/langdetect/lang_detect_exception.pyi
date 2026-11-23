# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""langdetect's ``LangDetectException`` and ``ErrorCode`` (langdetect's own when installed)."""

from typing import ClassVar

__all__ = ["ErrorCode", "LangDetectException"]

class ErrorCode:
    """The error codes of :class:`LangDetectException`."""

    NoTextError: ClassVar[int]
    FormatError: ClassVar[int]
    FileLoadError: ClassVar[int]
    DuplicateLangError: ClassVar[int]
    NeedLoadProfileError: ClassVar[int]
    CantDetectError: ClassVar[int]
    CantOpenTrainData: ClassVar[int]
    TrainDataFormatError: ClassVar[int]
    InitParamError: ClassVar[int]

class LangDetectException(Exception):  # noqa: N818 (langdetect's name)
    """langdetect's exception; ``code`` is one of :class:`ErrorCode`'s values."""

    code: int
    def __init__(self, code: int, message: str) -> None: ...
    def get_code(self) -> int:
        """The error code."""
