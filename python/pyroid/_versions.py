"""The installed-version check behind :class:`pyroid.VersionMismatchWarning` (not public API).

Every module that reproduces a release of another library calls :func:`check` once at import,
for each library whose behaviour it reproduces; the Thai and Japanese tokenizers and the URL
filter call it when they read the incumbent's data. :func:`pyroid.patch` imports the same
modules, so it warns through the same checks. Each library is checked once per process, by
whichever module asks first, and the warning points at the first frame outside pyroid: the
user's ``import`` or ``pyroid.patch(...)`` line, or the call that read the data.

The check never imports the incumbent. It costs one ``importlib.util.find_spec`` when the
incumbent is not installed, and reads its distribution metadata (``importlib.metadata``) only
when it is.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
import sys
import threading
import warnings
from collections.abc import Collection, Iterator, Mapping, Sequence
from types import FrameType

from pyroid._pyroid import VersionMismatchWarning

__all__ = ["check", "follow_installed", "quiet"]

_LOCK = threading.Lock()
# The libraries (top-level module names) already checked in this process.
_CHECKED: set[str] = set()
# Libraries whose checks accept every release in `modes`, in this context (follow_installed).
_FOLLOWED: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "pyroid._versions.followed", default=frozenset()
)
# Checks record the library without warning, in this context (quiet).
_QUIET: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "pyroid._versions.quiet", default=False
)
_MISSING = object()

_SILENCE = 'warnings.simplefilter("ignore", pyroid.VersionMismatchWarning)'


def _skipped() -> tuple[str, ...]:
    """The file prefixes a warning never points into: pyroid itself, and importlib's package."""
    import importlib

    prefixes = [os.path.dirname(__file__) + os.sep]
    if importlib.__file__:  # importlib.import_module(...) between the caller and the import
        prefixes.append(os.path.dirname(importlib.__file__) + os.sep)
    return tuple(prefixes)


_SKIPPED = _skipped()


def check(
    module: str,
    library: str,
    release: str,
    *,
    same: Collection[str] = (),
    modes: Mapping[str, str] | None = None,
    distributions: Sequence[str] = (),
) -> None:
    """Warn once per process if the installed `library` is not the `release` `module` reproduces.

    Nothing happens if `library` is not importable (no import is attempted), if it was already
    checked in this process, or if its distribution metadata is missing (a source checkout).

    Args:
        module: What reproduces `library`, as the message names it: ``"pyroid.ftfy"``.
        library: The incumbent's top-level module, ``"ftfy"``; the once-per-process key.
        release: The distribution and version reproduced, as a pip requirement:
            ``"ftfy==6.3.1"``, ``"rouge-score==0.1.2"``.
        same: Other releases (``"fasttext==0.9.3"``) whose results ``module`` gives too: no
            warning for them.
        modes: Other releases ``module`` reproduces when asked, and how to ask:
            ``{"trafilatura==1.11.0": 'compat="1.11.0"'}``. The warning says so, unless
            :func:`follow_installed` is in effect for `library` (the caller picks the mode).
        distributions: More distributions that install `library` (other builds), looked up
            after the ones named in `release`, `same` and `modes`.

    Warns:
        VersionMismatchWarning: The first distribution found is another release. The message
            names both versions and says what to install (or pass), and how to silence it.
    """
    if library in _CHECKED:
        return
    with _LOCK:
        if library in _CHECKED:
            return
        _CHECKED.add(library)
    if _QUIET.get() or not _importable(library):
        return
    found = _installed(release, same, modes or {}, distributions)
    if found is None or found == release or found in same:
        return
    dist, _, version = found.partition("==")
    advice = (modes or {}).get(found)
    if advice is None:
        todo = f"Install {release} to compare like for like"
    elif library in _FOLLOWED.get():
        return
    else:
        todo = f"Pass {advice} to reproduce it"
    name, _, wanted = release.partition("==")
    _warn(
        f"{module} reproduces {name} {wanted}, but {dist} {version} is installed: results can "
        f"differ from it. {todo} (or silence this with {_SILENCE})."
    )


def _importable(library: str) -> bool:
    """Whether `library` can be imported, without importing it."""
    module = sys.modules.get(library, _MISSING)
    if module is None:  # import blocked
        return False
    if module is not _MISSING:
        namespace = getattr(module, "__dict__", {})
        if namespace.get("__pyroid__", False):  # a pyroid.patch() shim: the module it wraps
            return namespace.get("__wrapped__") is not None
        return True
    import importlib.util

    try:
        return importlib.util.find_spec(library) is not None
    except (ImportError, ValueError):
        return False


def _installed(
    release: str, same: Collection[str], modes: Mapping[str, str], distributions: Sequence[str]
) -> str | None:
    """``"<distribution>==<version>"`` of the first candidate distribution installed, or None."""
    from importlib import metadata

    names = [r.partition("==")[0] for r in (release, *same, *modes)]
    for dist in dict.fromkeys([*names, *distributions]):
        try:
            version = metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
        if version:  # None for a METADATA without a version
            return f"{dist}=={version}"
    return None


# Count frames for `stacklevel` instead of passing `skip_file_prefixes`, which CPython 3.11's
# warnings.warn lacks (the tests set it on newer versions too).
_COUNT_FRAMES = sys.version_info < (3, 12)


def _warn(message: str) -> None:
    """``warnings.warn(message, VersionMismatchWarning)`` at the first frame outside pyroid."""
    if sys.version_info >= (3, 12) and not _COUNT_FRAMES:
        warnings.warn(message, VersionMismatchWarning, skip_file_prefixes=_SKIPPED)
    else:
        warnings.warn(message, VersionMismatchWarning, stacklevel=_stacklevel(sys._getframe()))


def _stacklevel(frame: FrameType) -> int:
    """The ``stacklevel`` that points ``warnings.warn``, called in `frame`, past `_SKIPPED`.

    Frames are counted as ``warnings.warn`` counts them: with a ``stacklevel`` above 1 it
    passes over importlib's bootstrap frames without counting them.
    """
    level = 1
    back = frame.f_back
    while back is not None:
        filename = back.f_code.co_filename
        if "importlib" in filename and "_bootstrap" in filename:
            back = back.f_back
            continue
        level += 1
        if not filename.startswith(_SKIPPED):
            break
        back = back.f_back
    return level


@contextlib.contextmanager
def follow_installed(library: str) -> Iterator[None]:
    """In this block, a check of `library` accepts every release in its `modes`.

    For callers that pick the mode from the installed release, or pass it explicitly:
    ``pyroid.patch("trafilatura")``, datatrove's ``Trafilatura`` step, the FineWeb recipe.
    """
    token = _FOLLOWED.set(_FOLLOWED.get() | {library})
    try:
        yield
    finally:
        _FOLLOWED.reset(token)


@contextlib.contextmanager
def quiet() -> Iterator[None]:
    """In this block, checks record their library as checked without warning.

    The start-up hook uses it in child processes of a process tree that already warned.
    """
    token = _QUIET.set(True)
    try:
        yield
    finally:
        _QUIET.reset(token)
