# Derived from NLTK 3.10.4 (Apache-2.0), © 2001-2026 NLTK Project; modified.
"""Finding and reading the ``punkt_tab`` models (NLTK's data, not pyroid's).

The models are not bundled: nltk_data publishes them without a licence. They are read from
the user's NLTK data, as ``nltk.tokenize.punkt.PunktTokenizer`` reads them:

- with NLTK installed, ``nltk.data.find`` locates ``tokenizers/punkt_tab/<lang>/`` (so
  ``nltk.data.path``, ``NLTK_DATA``, zip files, NLTK's permission checks and its
  ``LookupError`` text all apply) and ``nltk.tokenize.punkt.load_punkt_params`` reads it;
- without NLTK, the same default search path is scanned (``NLTK_DATA``, ``~/nltk_data``,
  ``sys.prefix``'s and the system's ``nltk_data`` directories), for an unpacked
  ``tokenizers/punkt_tab/<lang>/`` or a ``tokenizers/punkt_tab.zip``, and the four files are
  parsed with the same rules as NLTK's ``PunktDecoder``.
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
import unicodedata
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import Any

_FILES = ("collocations.tab", "sent_starters.txt", "abbrev_types.txt", "ortho_context.tab")

_WINDOWS_RESERVED_NAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"]
    + [f"COM{n}" for n in "123456789\u00b9\u00b2\u00b3"]
    + [f"LPT{n}" for n in "123456789\u00b9\u00b2\u00b3"]
)


def _is_windows_device_name(raw: str) -> bool:
    if os.name == "posix":
        return False
    leaf = raw.replace("/", "\\").rsplit("\\", 1)[-1]
    return leaf.split(".", 1)[0].strip().upper() in _WINDOWS_RESERVED_NAMES


def punkt_language(lang: Any) -> str:
    """``lang`` as NLTK's ``_punkt_language`` accepts it: one path segment naming a model.

    Raises:
        TypeError: ``lang`` is not a str.
        ValueError: ``lang`` cannot name a single ``punkt_tab`` model directory.
    """
    if not isinstance(lang, str):
        raise TypeError(f"A Punkt language name is a str, not {type(lang).__name__!r}")
    name: str = str.__str__(lang)
    lang = name
    if (
        not lang
        or not all(char.isalnum() or char in "-_" for char in lang)
        or unicodedata.normalize("NFC", lang) != lang
        or _is_windows_device_name(lang)
    ):
        raise ValueError(f"Invalid Punkt language name: {lang!r}")
    return name


def _rm_nl(s: str) -> str:
    return s[:-1] if s[-1] == "\n" else s


def _tups(lines: Iterable[str]) -> list[tuple[str, ...]]:
    return [tuple(_rm_nl(x).split("\t")) for x in lines]


class RawParams:
    """The four tables of one model, as NLTK's ``load_punkt_params`` builds them."""

    __slots__ = ("abbrev_types", "collocations", "ortho_context", "sent_starters")

    def __init__(
        self,
        collocations: set[Any],
        sent_starters: set[str],
        abbrev_types: set[str],
        ortho_context: defaultdict[str, int],
    ) -> None:
        self.collocations = collocations
        self.sent_starters = sent_starters
        self.abbrev_types = abbrev_types
        self.ortho_context = ortho_context


def _parse(open_file: Any) -> RawParams:
    """Read the four files with ``PunktDecoder``'s rules (text mode: universal newlines)."""
    with open_file("collocations.tab") as f:
        collocations = set(_tups(f))
    with open_file("sent_starters.txt") as f:
        sent_starters = {_rm_nl(x) for x in f}
    with open_file("abbrev_types.txt") as f:
        abbrev_types = {_rm_nl(x) for x in f}
    with open_file("ortho_context.tab") as f:
        ortho_context = defaultdict(int, {a: int(b) for a, b in _tups(f)})
    return RawParams(collocations, sent_starters, abbrev_types, ortho_context)


def default_paths() -> list[str]:
    """``nltk.data.path`` as NLTK 3.10.4 builds it at import time."""
    path = [os.path.expanduser(d) for d in os.environ.get("NLTK_DATA", "").split(os.pathsep) if d]
    if "APPENGINE_RUNTIME" not in os.environ and os.path.expanduser("~/") != "~/":
        path.append(os.path.expanduser("~/nltk_data"))
    if os.name == "nt":
        path += [
            os.path.join(sys.prefix, "nltk_data"),
            os.path.join(sys.prefix, "share", "nltk_data"),
            os.path.join(sys.prefix, "lib", "nltk_data"),
            os.path.join(os.environ.get("APPDATA", "C:\\"), "nltk_data"),
            r"C:\nltk_data",
            r"D:\nltk_data",
            r"E:\nltk_data",
        ]
    else:
        path += [
            os.path.join(sys.prefix, "nltk_data"),
            os.path.join(sys.prefix, "share", "nltk_data"),
            os.path.join(sys.prefix, "lib", "nltk_data"),
            "/usr/share/nltk_data",
            "/usr/local/share/nltk_data",
            "/usr/lib/nltk_data",
            "/usr/local/lib/nltk_data",
        ]
    return path


def _candidates(lang: str, paths: list[str]) -> Iterator[Any]:
    """Openers for ``lang``'s model, in NLTK's search order (directories, then zip files)."""
    rel = os.path.join("tokenizers", "punkt_tab", lang)
    for root in paths:
        if root and os.path.isfile(root) and root.endswith(".zip"):
            yield from _zip_opener(root, f"tokenizers/punkt_tab/{lang}/")
        elif not root or os.path.isdir(root):
            d = os.path.join(root, rel)
            if os.path.isdir(d):
                yield lambda name, d=d: open(os.path.join(d, name), encoding="utf-8")  # noqa: SIM115
    for root in paths:
        z = os.path.join(root, "tokenizers", "punkt_tab.zip")
        if (not root or os.path.isdir(root)) and os.path.isfile(z):
            yield from _zip_opener(z, f"punkt_tab/{lang}/")


def _zip_opener(path: str, entry: str) -> Iterator[Any]:
    try:
        zf = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile):
        return
    names = set(zf.namelist())
    if not any(n.startswith(entry) for n in names):
        zf.close()
        return

    def open_member(name: str) -> io.TextIOWrapper:
        member = entry + name
        if member not in names:
            raise LookupError(
                f"The Punkt model in {path + '/' + entry!r} has no {name!r}; "
                "reinstall it with nltk.download('punkt_tab')"
            )
        return io.TextIOWrapper(zf.open(member), encoding="utf-8")

    yield open_member


def _not_found(lang: str, paths: list[str]) -> LookupError:
    resource = f"tokenizers/punkt_tab/{lang}/"
    msg = (
        "  Resource 'punkt_tab' not found.\n"
        "  Please use the NLTK Downloader to obtain the resource:\n\n"
        "  >>> import nltk\n"
        "  >>> nltk.download('punkt_tab')\n"
        "  \n"
        "  (pyroid reads NLTK's punkt_tab models; NLTK itself is not installed, so its default"
        " search path was used.)\n"
        f"\n  Attempted to load '{resource}'\n"
        "\n  Searched in:" + "".join(f"\n    - {d!r}" for d in paths)
    )
    sep = "*" * 70
    return LookupError(f"\n{sep}\n{msg}\n{sep}\n")


def nltk_available() -> bool:
    """Whether NLTK can be imported (without importing it)."""
    if "nltk" in sys.modules:
        mod = sys.modules["nltk"]
        if getattr(mod, "__pyroid__", False):  # pyroid.patch("nltk")'s shim
            return getattr(mod, "__wrapped__", None) is not None
        return mod is not None
    try:
        return importlib.util.find_spec("nltk") is not None
    except (ImportError, ValueError):
        return False


def load(lang: str) -> RawParams:
    """The parameters of the ``punkt_tab`` model for ``lang`` (validated by the caller).

    Raises:
        LookupError: The model, or one of its four files, is not installed.
    """
    if nltk_available():
        find: Any = importlib.import_module("nltk.data").find
        load_punkt_params: Any = importlib.import_module("nltk.tokenize.punkt").load_punkt_params
        p = load_punkt_params(find(f"tokenizers/punkt_tab/{lang}/"))
        return RawParams(p.collocations, p.sent_starters, p.abbrev_types, p.ortho_context)
    paths = default_paths()
    for opener in _candidates(lang, paths):

        def open_file(name: str, opener: Any = opener) -> Any:
            try:
                return opener(name)
            except FileNotFoundError as missing:
                raise LookupError(
                    f"The Punkt model for {lang!r} has no {name!r}; "
                    "reinstall it with nltk.download('punkt_tab')"
                ) from missing

        return _parse(open_file)
    raise _not_found(lang, paths)
