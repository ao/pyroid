"""A clear ``ModuleNotFoundError`` when a module's optional dependency is not installed."""

from __future__ import annotations

from typing import NoReturn


def missing(
    module: str, exc: ModuleNotFoundError, extra: str | None, *names: str, hint: str | None = None
) -> NoReturn:
    """Raise, from ``exc``, a ``ModuleNotFoundError`` that says what to install.

    Use it as ``except ModuleNotFoundError as e: missing(__name__, e, "cli", "pyarrow")``. If
    ``exc`` is about another module than ``names``, it is raised unchanged. The new error names
    the missing module (``e.name``), as ``import`` does, so ``pytest.importorskip`` and
    ``except ModuleNotFoundError`` treat it the same way.

    Args:
        module: The pyroid module that needs it, for the message.
        exc: The error of the failed import.
        extra: The pyroid extra that installs it (``"cli"`` for ``pyroid[cli]``).
        *names: The top-level modules this is about.
        hint: What to install, instead of ``pyroid[extra]``.
    """
    name = exc.name or ""
    if name not in names:
        raise exc
    fix = hint or f"pip install 'pyroid[{extra}]' (or pip install {name})"
    raise ModuleNotFoundError(f"{module} needs {name}: {fix}", name=name) from exc
