"""Which sqlparse objects pyroid.sqlparse uses, decided once at import.

With sqlparse 0.5.0 or newer installed, the token types (`tokens`), the token classes
(`sql`) and `exceptions.SQLParseError` are sqlparse's own objects, so trees built by either
library mix freely and `isinstance` / `is` checks hold both ways. With exactly the release
pyroid reproduces (0.6.1), the lexer singleton and the keyword dictionaries are shared too,
so `Lexer.get_default_instance()` customisations apply to both. Otherwise pyroid uses its
own copies of sqlparse 0.6.1's modules.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
import warnings

SQLPARSE_VERSION = "0.6.1"

#: The classes of ``sqlparse.sql`` pyroid builds trees from (sqlparse 0.5.0 added ``Over``).
CLASSES = (
    "Token",
    "TokenList",
    "Statement",
    "Identifier",
    "IdentifierList",
    "TypedLiteral",
    "Parenthesis",
    "SquareBrackets",
    "Assignment",
    "If",
    "For",
    "Comparison",
    "Comment",
    "Where",
    "Over",
    "Having",
    "Case",
    "Function",
    "Begin",
    "Operation",
    "Values",
    "Command",
)

_PYROID_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find() -> tuple[types.ModuleType | None, str | None]:
    """The installed sqlparse, if it can be shared, and its version."""
    current = sys.modules.get("sqlparse", None)
    if current is not None and (
        getattr(current, "__pyroid__", False) or current.__name__.startswith("pyroid.")
    ):
        return None, None  # pyroid's own module, put there by pyroid.patch()
    if current is None:
        if "sqlparse" in sys.modules:  # sys.modules["sqlparse"] = None blocks it
            return None, None
        try:
            if importlib.util.find_spec("sqlparse") is None:
                return None, None
        except (ImportError, ValueError):
            return None, None
    try:
        mod = importlib.import_module("sqlparse")
        for sub in ("sql", "tokens", "exceptions", "keywords", "lexer"):
            importlib.import_module(f"sqlparse.{sub}")
    except Exception:  # a broken install: pyroid works without it
        return None, None
    version = getattr(mod, "__version__", None)
    sql = getattr(mod, "sql", None)
    if sql is None or not all(isinstance(getattr(sql, n, None), type) for n in CLASSES):
        return None, version
    return mod, version


REAL, REAL_VERSION = _find()
#: The lexer singleton and keyword dictionaries are sqlparse's own (it is 0.6.1).
SHARE_LEXER = REAL is not None and REAL_VERSION == SQLPARSE_VERSION


def _stacklevel() -> int:
    """The ``stacklevel`` of the first frame outside pyroid and importlib."""
    frame: types.FrameType | None = sys._getframe(1)
    level = 1
    while frame is not None:
        name = frame.f_code.co_filename
        if not name.startswith((_PYROID_DIR, "<frozen importlib")):
            return level
        frame = frame.f_back
        level += 1
    return 1


def _warn_version() -> None:
    if REAL_VERSION is None or REAL_VERSION == SQLPARSE_VERSION:
        return
    if REAL is None:
        detail = (
            "it is older than 0.5.0, so pyroid uses its own token classes "
            "(isinstance checks against sqlparse.sql fail)"
        )
    else:
        detail = (
            "pyroid shares its token types and classes, but parses and formats as "
            f"{SQLPARSE_VERSION}, and the methods of the tokens are the installed release's"
        )
    warnings.warn(
        f"pyroid.sqlparse reproduces sqlparse {SQLPARSE_VERSION}, but sqlparse {REAL_VERSION} "
        f"is installed: {detail}. Install sqlparse=={SQLPARSE_VERSION} to compare like for like.",
        RuntimeWarning,
        stacklevel=_stacklevel(),
    )


_warn_version()
