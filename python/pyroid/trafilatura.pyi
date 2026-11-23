# Derived from trafilatura 2.3.1 and 1.11.0 (Apache-2.0), © Adrien Barbaresi; modified.
# Type stubs for pyroid.trafilatura: the signatures of trafilatura 2.3.1. The module docstring is
# kept here on purpose: the API docs are built from this stub.
"""A drop-in, Rust-backed replacement for trafilatura's text extraction (2.3.1 and 1.11.0).

:func:`extract` and :func:`bare_extraction` take trafilatura 2.3.1's arguments, with its
defaults, and return the same text, byte for byte. The whole path is ported to Rust: trafilatura's
``load_html`` and HTML repair, libxml2 2.14.6's HTML parser as lxml 6.1.3 drives it, the tree
cleaning, the main extractor, the readability and jusText 3.0.2 comparisons, the baseline and
recall fallbacks, deduplication and the text output. Nothing is imported from trafilatura or
lxml on that path. ``compat="1.11.0"`` reproduces trafilatura 1.11.0 instead, the release
datatrove 0.10.1 installs (see below).

Native: a ``str`` page (lone surrogates included), or ``bytes`` that are valid UTF-8 and not
compressed; ``output_format="txt"`` (``"python"`` for :func:`bare_extraction`); ``fast`` /
``no_fallback``, ``favor_precision``, ``favor_recall``, ``include_comments``, ``include_tables``,
``deduplicate`` (``True``, or an :class:`LRUCache`), ``url``, ``record_id``, ``tei_validation``,
``date_extraction_params`` (no effect without metadata), ``max_tree_size`` (the same
``ValueError``), and ``config`` / ``settingsfile`` (the size settings are read as trafilatura
reads them).

Delegated to the installed trafilatura 2.3.1 with a one-time ``pyroid.FallbackWarning`` (per
function and reason): the other output formats, ``include_formatting``, ``include_links``,
``include_images``, metadata (``with_metadata``, ``only_with_metadata``, ``url_blacklist``,
``author_blacklist``, :func:`extract_with_metadata`), ``target_language``, ``prune_xpath``,
``options=Extractor(...)``, ``deduplicate=trafilatura.deduplication.LRUCache(...)``, an lxml
tree or a response object as input, bytes that are compressed or not UTF-8, a negative size
setting, and ``sys.set_int_max_str_digits()`` limits other than 4300. Without trafilatura, or
with another version of it, these raise ``pyroid.PyroidError`` and say why.

trafilatura 1.11.0 (``compat="1.11.0"``)
----------------------------------------
The same options run natively, with 1.11.0's behaviour; the delegated ones go to an installed
trafilatura 1.11.0 (``PyroidError`` without it, or with another version). Where 1.11.0 differs:

- It has no ``fast``: :func:`extract` ignores it (1.11.0's ``extract`` takes ``**kwargs``) and
  :func:`bare_extraction` raises its ``TypeError`` for a true value. Its ``no_fallback`` is the
  same switch, without a ``DeprecationWarning``.
- ``max_tree_size`` works (its ``len(body)`` check) instead of raising; ``author_blacklist`` does
  not need metadata, so it has no effect here.
- ``deduplicate`` is a flag: any true value, an :class:`LRUCache` included, uses the module cache
  :data:`LRU_TEST`, as 1.11.0 always uses its own.
- ``config`` wins over ``settingsfile`` (which is read only with ``config=None``, and then on its
  own); the default ``config`` stands for 1.11.0's settings.cfg (``DEFAULT_CONFIGS["1.11.0"]``).
- :func:`bare_extraction` raises 1.11.0's ``AttributeError`` (``'NoneType' object has no
  attribute 'source'``) for a page that does not load, as 1.11.0 does when called directly; its
  ``commentsbody`` is ``None`` without ``include_comments``.
- There is no ``extract_with_metadata`` (``compat="1.11.0"`` raises ``ValueError``).
- :func:`bare_extraction`'s ``as_dict`` defaults to ``True``, as in 1.11.0. :func:`extract`
  ignores keywords it does not know, and raises 1.11.0's ``NameError`` for the old
  ``csv_output`` / ``json_output`` / ``tei_output`` / ``xml_output``, as 1.11.0's
  ``extract(**kwargs)`` does (its signature is shown as 2.3.1's; in 2.3.1 mode unknown keywords
  raise ``TypeError``).
- Positional arguments follow this module's (2.3.1's) order: pass 1.11.0's options by
  keyword. ``pyroid.patch("trafilatura")`` with trafilatura 1.11.0 installed gives functions
  with 1.11.0's exact signatures.

Differences from trafilatura
----------------------------
- Trafilatura's log messages (``logging.getLogger("trafilatura")``) are not emitted.
- ``config`` defaults to this module's :data:`DEFAULT_CONFIG`, not
  ``trafilatura.settings.DEFAULT_CONFIG``: changing trafilatura's object changes nothing here.
- :class:`LRUCache` keys are str without lone surrogates; its ``cache`` is a read-only copy.
- :func:`bare_extraction` returns a :class:`Document` whose ``body`` and ``commentsbody``
  become lxml elements when first read (the XML strings when lxml is not installed).
"""

from collections.abc import Mapping
from configparser import ConfigParser
from types import MappingProxyType
from typing import Any, Final, Literal, final, overload

from pyroid._typing import ArrowStrings

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_CONFIGS",
    "DELEGATED",
    "LRU_SIZE",
    "LRU_TEST",
    "REPRODUCED_VERSIONS",
    "TRAFILATURA_VERSION",
    "TRAFILATURA_VERSIONS",
    "Document",
    "LRUCache",
    "bare_extraction",
    "extract",
    "extract_batch",
    "extract_with_metadata",
    "reset_caches",
]

TRAFILATURA_VERSION: Final[str]
"""The trafilatura release reproduced by default (``"2.3.1"``, the default of ``compat=``)."""

TRAFILATURA_VERSIONS: Final[tuple[str, ...]]
"""The trafilatura releases pyroid reproduces, one ``compat=`` mode each (2.3.1 and 1.11.0)."""

REPRODUCED_VERSIONS: Final[Mapping[str, Mapping[str, str]]]
"""For each mode, the exact library versions it reproduces together.

``"2.3.1"``: trafilatura 2.3.1, lxml 6.1.3, libxml2 2.14.6 (bundled in lxml 6.1.3's wheels),
justext 3.0.2. ``"1.11.0"``: trafilatura 1.11.0, lxml 6.1.3, libxml2 2.14.6, justext 3.0.2, and
lxml-html-clean 0.4.6, charset-normalizer 3.5.2, courlan 1.4.0 and htmldate 1.11.0, as
``pip install datatrove[processing]==0.10.1`` resolves them (these four do not change the text).
"""

DELEGATED: Final[tuple[str, ...]]
"""Functions that always run the installed trafilatura: ``("extract_with_metadata",)``."""

LRU_SIZE: Final[int]
"""``trafilatura.settings.LRU_SIZE`` (4096): the size of :data:`LRU_TEST`."""

DEFAULT_CONFIG: Final[ConfigParser]
"""``trafilatura.settings.DEFAULT_CONFIG``: trafilatura 2.3.1's settings.cfg values."""

DEFAULT_CONFIGS: Final[Mapping[str, ConfigParser]]
"""Each mode's ``trafilatura.settings.DEFAULT_CONFIG`` (``DEFAULT_CONFIGS["2.3.1"]`` is
:data:`DEFAULT_CONFIG`). With ``compat="1.11.0"`` the default ``config`` stands for 1.11.0's."""

@final
class LRUCache:
    """Least Recently Used cache of string counts: ``trafilatura.deduplication.LRUCache`` in Rust.

    Pass one as ``deduplicate=`` to scope deduplication to it. It is safe to share between
    threads; an extraction holds it for the whole page, so concurrent extractions through one
    cache run one at a time. Instances pickle and copy with their entries.
    """

    def __new__(cls, maxsize: int = 128) -> LRUCache:
        """Create an empty cache.

        Args:
            maxsize: The most entries kept; the least recently used one is evicted past it.

        Raises:
            ValueError: ``maxsize`` is below 1 (``"maxsize must be at least 1"``).
            pyroid.InputError: ``maxsize`` is not an int.
        """

    @property
    def maxsize(self) -> int:
        """The most entries the cache keeps."""

    @property
    def cache(self) -> MappingProxyType[str, int]:
        """A read-only copy of the entries, oldest first (trafilatura's ``.cache``)."""

    def get(self, key: str) -> int:
        """Retrieve a value from the cache, or -1 if the key is absent."""

    def put(self, key: str, value: int) -> None:
        """Store a given key in the cache, evicting the oldest entry if full."""

    def increment(self, key: str) -> int:
        """Increment the stored count, return the previous count or -1."""

    def clear(self) -> None:
        """Delete all cache content."""

    def __bool__(self) -> Literal[True]:
        """Always true, even when empty: passed as ``deduplicate``, it enables deduplication."""

LRU_TEST: Final[LRUCache]
"""The cache ``deduplicate=True`` reads and updates (``trafilatura.deduplication.LRU_TEST``)."""

class Document:
    """What :func:`bare_extraction` returns: ``trafilatura.settings.Document``'s fields.

    A subclass of ``trafilatura.settings.Document`` when trafilatura 2.3.1 is installed (it is
    imported the first time this class is used), else a class with the same fields, constructor,
    ``from_dict`` and ``as_dict``. The extraction fills ``text``, ``raw_text`` (the same str),
    ``comments`` (with ``include_comments``), ``body`` and ``commentsbody``; the metadata fields
    are ``None``, as in trafilatura without metadata extraction. ``body`` and ``commentsbody``
    are parsed into lxml elements when first read (they are the strings
    ``etree.tostring(element, encoding="unicode")`` gives when lxml is not installed).

    With ``compat="1.11.0"`` the result is a subclass of trafilatura 1.11.0's ``Document``
    (when 1.11.0 is installed) or of a class with its fields: its constructor takes no
    arguments, setting an unknown attribute does nothing, and ``commentsbody`` is ``None``
    without ``include_comments``. A tree that holds control characters (1.11.0 keeps them, and
    lxml cannot build such a tree from Python) stays its XML string.
    """

    title: str | None
    author: str | None
    url: str | None
    hostname: str | None
    description: str | None
    sitename: str | None
    date: str | None
    categories: list[str] | None
    tags: list[str] | None
    fingerprint: str | None
    id: str | None
    license: str | None
    body: Any
    comments: str | None
    commentsbody: Any
    raw_text: str | None
    text: str | None
    language: str | None
    image: str | None
    pagetype: str | None
    filedate: str | None
    def __init__(
        self,
        *,
        title: str | None = None,
        author: str | None = None,
        url: str | None = None,
        hostname: str | None = None,
        description: str | None = None,
        sitename: str | None = None,
        date: str | None = None,
        categories: list[str] | None = None,
        tags: list[str] | None = None,
        fingerprint: str | None = None,
        idval: str | None = None,
        license_val: str | None = None,
        body: Any = None,
        comments: str | None = None,
        commentsbody: Any = None,
        raw_text: str | None = None,
        text: str | None = None,
        language: str | None = None,
        image: str | None = None,
        pagetype: str | None = None,
        filedate: str | None = None,
    ) -> None: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """Set a series of attributes using a dictionary."""

    def as_dict(self) -> dict[str, Any]:
        """Convert the document to a dictionary (``body`` and ``commentsbody`` as lxml elements)."""

def reset_caches() -> None:
    """Empty the module-level dedup cache :data:`LRU_TEST` (``trafilatura.meta.reset_caches``)."""

def extract(
    filecontent: str | bytes | Any,
    url: str | None = None,
    record_id: str | None = None,
    fast: bool = False,
    no_fallback: bool = False,
    favor_precision: bool = False,
    favor_recall: bool = False,
    include_comments: bool = True,
    output_format: str = "txt",
    tei_validation: bool = False,
    target_language: str | None = None,
    include_tables: bool = True,
    include_images: bool = False,
    include_formatting: bool | None = None,
    include_links: bool = False,
    deduplicate: bool | LRUCache | Any = False,
    date_extraction_params: dict[str, Any] | None = None,
    with_metadata: bool = False,
    only_with_metadata: bool = False,
    max_tree_size: int | None = None,
    url_blacklist: set[str] | None = None,
    author_blacklist: set[str] | None = None,
    settingsfile: str | None = None,
    prune_xpath: str | list[str] | None = None,
    config: ConfigParser = ...,
    options: Any = None,
    *,
    compat: str = "2.3.1",
) -> str | None:
    """Extract the main text of an HTML page: ``trafilatura.extract``.

    >>> page = "<html><body><article><p>" + "Some words. " * 30 + "</p></article></body></html>"
    >>> extract(page)[:22]
    'Some words. Some words'

    The arguments, defaults and results are trafilatura 2.3.1's; ``compat`` is pyroid's only
    addition. The text is NFC-normalized, as trafilatura's.

    Args:
        filecontent: The page: a str, or bytes (UTF-8, else delegated). lxml trees and response
            objects are delegated; other types give ``None``, as in trafilatura.
        url: The page's URL (only metadata and links use it).
        record_id: An ID for the metadata (not part of the txt output).
        fast: Skip the readability and jusText comparison.
        no_fallback: Deprecated alias of ``fast`` (trafilatura's ``DeprecationWarning``).
        favor_precision: Prefer less text but correct extraction.
        favor_recall: Prefer more text when unsure (it wins over ``favor_precision``).
        include_comments: Extract comments along with the main text.
        output_format: ``"txt"`` runs natively; the others are delegated.
        tei_validation: Only used for ``"xmltei"`` (delegated).
        target_language: Delegated (language detection).
        include_tables: Take ``<table>`` content into account.
        include_images: Delegated.
        include_formatting: ``True`` is delegated.
        include_links: Delegated.
        deduplicate: ``True`` deduplicates through the module cache :data:`LRU_TEST`; an
            :class:`LRUCache` scopes it to that cache (a trafilatura ``LRUCache`` is delegated).
        date_extraction_params: htmldate parameters (only used with metadata).
        with_metadata: Delegated.
        only_with_metadata: Delegated.
        max_tree_size: Deprecated: a truthy value raises trafilatura's ``ValueError``.
        url_blacklist: A non-empty set is delegated (it needs metadata).
        author_blacklist: A non-empty set is delegated (it needs metadata).
        settingsfile: A settings.cfg file overriding the defaults.
        prune_xpath: Delegated.
        config: A ``ConfigParser`` with trafilatura's settings (default :data:`DEFAULT_CONFIG`).
        options: A ``trafilatura.settings.Extractor`` is delegated; other values are ignored, as
            in trafilatura.
        compat: The trafilatura release to reproduce, one of :data:`TRAFILATURA_VERSIONS`
            (``"1.11.0"``: see the module docstring for how its arguments differ).

    Returns:
        The text, or ``None`` where trafilatura returns ``None`` (nothing extracted, too short,
        a duplicate, not HTML).

    Raises:
        ValueError: ``max_tree_size``; a page with a ``&#`` reference of more than 4300 digits
            (CPython's ``int()`` limit, as in trafilatura); a ``compat`` pyroid does not
            reproduce.
        AttributeError: An unknown ``output_format`` (from trafilatura, delegated).
        pyroid.PyroidError: The call needs trafilatura (see the module docstring), and it is
            not installed, or is another version than ``compat``.
    """

@overload
def bare_extraction(
    filecontent: str | bytes | Any,
    url: str | None = None,
    fast: bool = False,
    no_fallback: bool = False,
    favor_precision: bool = False,
    favor_recall: bool = False,
    include_comments: bool = True,
    output_format: str = "python",
    target_language: str | None = None,
    include_tables: bool = True,
    include_images: bool = False,
    include_formatting: bool | None = None,
    include_links: bool = False,
    deduplicate: bool | LRUCache | Any = False,
    date_extraction_params: dict[str, Any] | None = None,
    with_metadata: bool = False,
    only_with_metadata: bool = False,
    max_tree_size: int | None = None,
    url_blacklist: set[str] | None = None,
    author_blacklist: set[str] | None = None,
    as_dict: Literal[False] = False,
    prune_xpath: str | list[str] | None = None,
    config: ConfigParser = ...,
    options: Any = None,
    *,
    compat: str = "2.3.1",
) -> Document | None:
    """Extract a page's text and trees: ``trafilatura.bare_extraction``.

    The arguments are :func:`extract`'s, with ``output_format="python"`` (native; the others
    are delegated) and the deprecated ``as_dict`` (``DeprecationWarning``, then
    ``Document.as_dict()``).

    Returns:
        A :class:`Document` (``text`` is not NFC-normalized, as in trafilatura), its
        ``as_dict()`` with ``as_dict=True``, or ``None``.

    Raises:
        ValueError: As for :func:`extract`.
        pyroid.PyroidError: As for :func:`extract`.
    """

@overload
def bare_extraction(
    filecontent: str | bytes | Any,
    url: str | None = None,
    fast: bool = False,
    no_fallback: bool = False,
    favor_precision: bool = False,
    favor_recall: bool = False,
    include_comments: bool = True,
    output_format: str = "python",
    target_language: str | None = None,
    include_tables: bool = True,
    include_images: bool = False,
    include_formatting: bool | None = None,
    include_links: bool = False,
    deduplicate: bool | LRUCache | Any = False,
    date_extraction_params: dict[str, Any] | None = None,
    with_metadata: bool = False,
    only_with_metadata: bool = False,
    max_tree_size: int | None = None,
    url_blacklist: set[str] | None = None,
    author_blacklist: set[str] | None = None,
    as_dict: bool = False,
    prune_xpath: str | list[str] | None = None,
    config: ConfigParser = ...,
    options: Any = None,
    *,
    compat: str = "2.3.1",
) -> Document | dict[str, Any] | None: ...
def extract_with_metadata(
    filecontent: str | bytes | Any,
    url: str | None = None,
    record_id: str | None = None,
    fast: bool = False,
    favor_precision: bool = False,
    favor_recall: bool = False,
    include_comments: bool = True,
    output_format: str = "txt",
    tei_validation: bool = False,
    target_language: str | None = None,
    include_tables: bool = True,
    include_images: bool = False,
    include_formatting: bool | None = None,
    include_links: bool = False,
    deduplicate: bool | LRUCache | Any = False,
    date_extraction_params: dict[str, Any] | None = None,
    url_blacklist: set[str] | None = None,
    author_blacklist: set[str] | None = None,
    settingsfile: str | None = None,
    prune_xpath: str | list[str] | None = None,
    config: ConfigParser = ...,
    options: Any = None,
    *,
    compat: str = "2.3.1",
) -> Any:
    """``trafilatura.extract_with_metadata``: always delegated (metadata extraction).

    Returns trafilatura's ``Document``. Needs trafilatura 2.3.1; a pyroid :class:`LRUCache`
    passed as ``deduplicate`` is lent to it and updated.

    Raises:
        pyroid.PyroidError: trafilatura is not installed, or is another version.
        ValueError: ``compat="1.11.0"`` (trafilatura 1.11.0 has no ``extract_with_metadata``).
    """

@overload
def extract_batch(
    htmls: str,
    *,
    threads: int | None = None,
    compat: str = "2.3.1",
    **kwargs: Any,
) -> str | None:
    """:func:`extract` over many pages, in order, on pyroid's thread pool without the GIL.

    Not part of trafilatura. ``extract_batch(htmls, **kwargs)`` equals
    ``[extract(html, **kwargs) for html in htmls]``, including with ``deduplicate``: with
    ``deduplicate=False`` the pages are extracted in parallel; with ``True`` or an
    :class:`LRUCache` they are parsed in parallel and extracted one after the other in order
    through the cache, so the result (and the cache afterwards) is the loop's. A batch that holds
    a non-str item (``bytes``, as :func:`extract` takes them), or uses a delegated option or
    ``no_fallback``, runs that loop.

    Args:
        htmls: A list or tuple of pages (str), an Arrow string array/column (pyarrow, polars,
            ``pyroid.Column``, ...), or one page (str).
        threads: ``None`` uses pyroid's whole thread pool; ``n`` at most ``n`` tasks at once;
            ``1`` the calling thread only.
        compat: As for :func:`extract`.
        **kwargs: :func:`extract`'s keyword arguments.

    Returns:
        A list (also for a tuple or Arrow input) with one text or ``None`` per page, in order;
        ``None`` for each Arrow null. For a str, ``extract(htmls, **kwargs)``.

    Raises:
        ValueError: As for :func:`extract` (the note names the page); ``threads <= 0``.
        TypeError: An argument :func:`extract` does not take.
        pyroid.InputError: ``htmls`` is not a str, list, tuple or Arrow string column, or a list
            or tuple holds ``None`` (the message gives its index).
    """

@overload
def extract_batch(
    htmls: list[str] | tuple[str, ...] | ArrowStrings,
    *,
    threads: int | None = None,
    compat: str = "2.3.1",
    **kwargs: Any,
) -> list[str | None]: ...
