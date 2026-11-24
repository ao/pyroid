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
import sys
import types

from pyroid import _versions

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


def _pyroids(mod: types.ModuleType) -> bool:
    """`mod` is pyroid's own module (put in ``sys.modules`` by pyroid.patch())."""
    return bool(getattr(mod, "__pyroid__", False)) or mod.__name__.startswith("pyroid.")


def _has_classes(mod: types.ModuleType) -> bool:
    """`mod.sql` has every class pyroid builds trees from (sqlparse 0.5.0 and later)."""
    sql = getattr(mod, "sql", None)
    return sql is not None and all(isinstance(getattr(sql, n, None), type) for n in CLASSES)


def _find() -> tuple[types.ModuleType | None, str | None]:
    """The installed sqlparse, if it can be shared, and its version."""
    current = sys.modules.get("sqlparse", None)
    if current is not None and _pyroids(current):
        return None, None
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
    if not _has_classes(mod):
        return None, version
    return mod, version


REAL, REAL_VERSION = _find()
#: The lexer singleton and keyword dictionaries are sqlparse's own (it is 0.6.1).
SHARE_LEXER = REAL is not None and REAL_VERSION == SQLPARSE_VERSION
#: The token classes have an ``is_newline`` slot (sqlparse 0.5.1 added it; 0.5.0's have none).
NEWLINE_SLOT = REAL is None or hasattr(REAL.sql.Token, "is_newline")


def _warn_classes() -> None:
    """Another release is installed whose classes pyroid cannot share (older than 0.5.0).

    pyroid.sqlparse's VersionMismatchWarning (pyroid._versions) names the release; this says
    what else differs.
    """
    if REAL is not None or REAL_VERSION is None or REAL_VERSION == SQLPARSE_VERSION:
        return
    # At the user's import line, as the VersionMismatchWarning after it: counting frames here
    # pointed past it (sys:1), since warnings.warn skips importlib's frames itself.
    _versions._warn(
        f"sqlparse {REAL_VERSION} is older than 0.5.0, so pyroid.sqlparse builds its trees from "
        f"its own copies of sqlparse {SQLPARSE_VERSION}'s token classes: isinstance checks "
        "against sqlparse.sql fail on them, and pyroid.patch('sqlparse') refuses to patch it.",
        RuntimeWarning,
    )


def patch_refusal(real: types.ModuleType) -> str | None:
    """Why ``pyroid.patch("sqlparse")`` must leave the sqlparse module `real` alone (or None).

    The patched functions make trees of the classes pyroid.sqlparse uses, so they may only go
    on the sqlparse whose classes, token types and filters those are.
    """
    if real is REAL or _pyroids(real):
        return None
    if not _has_classes(real):
        version = getattr(real, "__version__", None) or "unversioned"
        return (
            "patch('sqlparse') needs sqlparse 0.5.0 or later, and the installed sqlparse is "
            f"{version}: code using it (its filters, isinstance checks and token types) would "
            "get trees made of pyroid's own token classes, which it does not recognise. Upgrade "
            f"it with pip install -U 'sqlparse>=0.5.0' (pyroid reproduces {SQLPARSE_VERSION})."
        )
    return (
        "patch('sqlparse') cannot patch this sqlparse module: pyroid.sqlparse was imported "
        "before it could be, so pyroid's trees are made of other token classes than its own. "
        "Patch in a new process, before anything imports pyroid.sqlparse."
    )


_warn_classes()
