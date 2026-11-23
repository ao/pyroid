# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""langdetect's ``LangDetectException`` and ``ErrorCode`` (langdetect's own with it installed)."""

from __future__ import annotations

from pyroid.langdetect._compat import ErrorCode, LangDetectException

__all__ = ["ErrorCode", "LangDetectException"]
