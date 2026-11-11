"""Outcome capture, normalization, digests and the call codec of the ``trafilatura`` parity suite.

An *outcome* is ``pyroid_parity.json_repair_compare.Outcome``: the normalized return value or the exception (type
name and message), plus the warning categories raised, leaving out ``pyroid.FallbackWarning``. Return values are
normalized so that a candidate need not build lxml objects:

* ``extract`` returns ``str`` or ``None``, compared as they are;
* ``bare_extraction`` returns a ``trafilatura.settings.Document`` (or its ``as_dict()``): it becomes
  ``{"$document": {slot: value}}`` over Document's 21 slots in order. ``body`` and ``commentsbody`` become
  ``{"$xml": etree.tostring(element, encoding="unicode")}``; a candidate may hold that XML string itself there.

The pinned references store ``digest(outcome)`` (sha256 of the outcome's canonical JSON) and the text length, not
the text: the corpus pages are third-party content and the references stay small.

Recorded upstream calls (``pyroid_parity.trafilatura_record``) store their arguments with ``encode_arg``: plain
values as in ``json_repair_compare.encode``, strings and bytes of 4 KiB or more as ``{"$blob": [sha256, "str" |
"bytes"]}`` (kept once, in a separate blob table), and the objects Trafilatura's tests pass as tagged values that
``decode_arg`` rebuilds: lxml trees (``$tree``, a node list, so the rebuilt tree is the same element by element),
``ConfigParser`` (``$config``), ``LRUCache`` (``$lru``, its state at call time), ``Extractor`` (``$extractor``) and
sets (``$set``). Building those needs trafilatura and lxml; nothing else here imports them.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from pyroid_parity import json_repair_compare as JC
from pyroid_parity.json_repair_compare import Outcome, capture, describe_difference, same_outcome

__all__ = [
    "DOCUMENT_SLOTS",
    "Blobs",
    "Outcome",
    "capture",
    "capture_call",
    "capture_with_warnings",
    "decode_arg",
    "describe_difference",
    "digest",
    "encode_arg",
    "normalize",
    "same_outcome",
]

#: trafilatura.settings.Document.__slots__ in 2.3.1, in order.
DOCUMENT_SLOTS = (
    "title",
    "author",
    "url",
    "hostname",
    "description",
    "sitename",
    "date",
    "categories",
    "tags",
    "fingerprint",
    "id",
    "license",
    "body",
    "comments",
    "commentsbody",
    "raw_text",
    "text",
    "language",
    "image",
    "pagetype",
    "filedate",
)
XML_SLOTS = frozenset({"body", "commentsbody"})
BLOB_MIN = 4096


# ----------------------------------------------------------------------------------------------- normalizing


def _is_lxml(x: Any) -> bool:
    return type(x).__module__.startswith("lxml.")


def _xml(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, str):
        return {"$xml": x}
    if _is_lxml(x):
        from lxml import etree

        return {"$xml": etree.tostring(x, encoding="unicode")}
    return {"$object": f"{type(x).__module__}.{type(x).__qualname__}"}


def _value(key: str, v: Any) -> Any:
    if key in XML_SLOTS:
        return _xml(v)
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [_value("", e) for e in v]
    if isinstance(v, dict):
        return {str(k): _value(str(k), e) for k, e in v.items()}
    if _is_lxml(v):
        return _xml(v)
    return {"$object": f"{type(v).__module__}.{type(v).__qualname__}"}


def normalize(result: Any) -> Any:
    """A comparable form of what ``extract`` / ``bare_extraction`` returned (see the module docstring)."""
    if result is None or isinstance(result, str):
        return result
    if isinstance(result, dict):
        return {"$dict": {str(k): _value(str(k), v) for k, v in result.items()}}
    if all(hasattr(result, s) for s in ("text", "body", "raw_text")):
        return {"$document": {s: _value(s, getattr(result, s, None)) for s in DOCUMENT_SLOTS}}
    return {"$object": f"{type(result).__module__}.{type(result).__qualname__}"}


def capture_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Outcome:
    """``fn(*args, **kwargs)`` as a normalized outcome (every exception caught: type and message are compared)."""
    return capture(lambda: normalize(fn(*args, **kwargs)))


def capture_with_warnings(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Outcome, list[str]]:
    """``capture_call`` plus the exact category (``module.qualname``) of every warning raised, in order."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = Outcome("ok", normalize(fn(*args, **kwargs)))
        except Exception as e:  # noqa: BLE001  exception types and messages are part of parity
            out = Outcome("raise", (type(e).__name__, str(e)))
    names = [f"{w.category.__module__}.{w.category.__qualname__}" for w in caught]
    cats = tuple(sorted({w.category.__name__ for w in caught if not JC.is_fallback_warning(w.category)}))
    return Outcome(out.kind, out.value, cats), names


def outcome_json(o: Outcome) -> str:
    """Canonical JSON of an outcome (what ``digest`` hashes)."""
    return json.dumps(o.to_json(), ensure_ascii=True, separators=(",", ":"))


def digest(o: Outcome) -> str:
    return hashlib.sha256(outcome_json(o).encode("ascii")).hexdigest()


def text_length(o: Outcome) -> int | None:
    return len(o.value) if o.kind == "ok" and isinstance(o.value, str) else None


def first_text_difference(got: str, want: str) -> str:
    pos = next(
        (i for i, (a, b) in enumerate(zip(got, want, strict=False)) if a != b), min(len(got), len(want))
    )
    lo = max(0, pos - 40)
    return f"at {pos} (lengths {len(got)} vs {len(want)}): {got[lo : pos + 40]!r} vs {want[lo : pos + 40]!r}"


# --------------------------------------------------------------------------------------------- call codec


class Blobs:
    """Content-addressed store for large ``str`` / ``bytes`` arguments (sha256 -> bytes).

    ``root`` is the directory the upstream tests run from (recording) or were rebuilt in (replay): a str argument
    naming a file under it is stored as ``{"$testfile": relative path}``. ``default_config`` is what
    ``{"$config": "default"}`` replays as where trafilatura is not installed: the candidate's own default for
    ``config`` (``None`` only if the candidate's is).
    """

    def __init__(self, items: Iterable[tuple[str, bytes]] = (), root: str | None = None) -> None:
        self.data: dict[str, bytes] = dict(items)
        self.root = root
        self.default_config: Any = None

    def put(self, raw: bytes) -> str:
        key = hashlib.sha256(raw).hexdigest()
        self.data.setdefault(key, raw)
        return key

    def get(self, key: str) -> bytes:
        return self.data[key]


def _tree_nodes(root: Any) -> list[Any]:
    """An element tree as nested lists, built without recursion.

    Element: ``["e", tag, [[name, value], ...], text, tail, children]``; comment ``["c", text, tail]``;
    processing instruction ``["p", target, text, tail]``.
    """
    from lxml import etree

    def leaf(el: Any) -> list[Any]:
        if isinstance(el, etree._Comment):
            return ["c", el.text, el.tail]
        if isinstance(el, etree._ProcessingInstruction):
            return ["p", el.target, el.text, el.tail]
        return ["e", el.tag, [list(kv) for kv in el.attrib.items()], el.text, el.tail, []]

    out = leaf(root)
    out[4] = None  # the root's tail is not part of the input
    stack = [(root, out)]
    while stack:
        el, node = stack.pop()
        if node[0] != "e":
            continue
        for child in el:
            n = leaf(child)
            node[5].append(n)
            stack.append((child, n))
    return out


def _build_tree(nodes: list[Any], html: bool) -> Any:
    from lxml import etree
    from lxml import html as lhtml

    parser = lhtml.html_parser if html else etree.XMLParser()

    def make(n: list[Any]) -> Any:
        if n[0] == "c":
            el = etree.Comment(n[1])
            el.tail = n[2]
        elif n[0] == "p":
            el = etree.ProcessingInstruction(n[1], n[2])
            el.tail = n[3]
        else:
            el = parser.makeelement(n[1], dict(n[2]))
            el.text, el.tail = n[3], n[4]
        return el

    root = make(nodes)
    stack = [(root, nodes)]
    while stack:
        el, n = stack.pop()
        if n[0] != "e":
            continue
        for c in n[5]:
            child = make(c)
            el.append(child)
            stack.append((child, c))
    return root


def _lru_items(lru: Any) -> list[tuple[Any, Any]]:
    """(key, value) of a trafilatura LRUCache, least recently used first.

    2.3.1 keeps an OrderedDict in that order; 1.11.0 a circular doubly linked list under ``root``, whose links are
    ``[prev, next, key, value]`` with the least recent one after the root.
    """
    root = getattr(lru, "root", None)
    if root is None:
        return list(lru.cache.items())
    out, link = [], root[1]
    while link is not root:
        out.append((link[2], link[3]))
        link = link[1]
    return out


def encode_arg(x: Any, blobs: Blobs, tags: set[str]) -> Any:
    """``x`` as JSON (large strings into ``blobs``); adds ``"unreplayable"`` to ``tags`` when it cannot."""
    t = type(x)
    if t is str and blobs.root and x.startswith(blobs.root.rstrip("/") + "/"):
        tags.add("test-file")
        return {"$testfile": x[len(blobs.root.rstrip("/")) + 1 :]}
    if t is str and len(x) >= BLOB_MIN:
        return {"$blob": [blobs.put(x.encode("utf-8", "surrogatepass")), "str"]}
    if t is bytes and len(x) >= BLOB_MIN:
        return {"$blob": [blobs.put(x), "bytes"]}
    if x is None or t in (bool, int, float, str, bytes):
        return JC.encode(x)
    if t is list:
        return [encode_arg(v, blobs, tags) for v in x]
    if t is tuple:
        return {"$tuple": [encode_arg(v, blobs, tags) for v in x]}
    if t is dict:
        if all(type(k) is str and not k.startswith("$") for k in x):
            return {k: encode_arg(v, blobs, tags) for k, v in x.items()}
        return {"$dict": [[encode_arg(k, blobs, tags), encode_arg(v, blobs, tags)] for k, v in x.items()]}
    if t in (set, frozenset):
        items = sorted((encode_arg(v, blobs, tags) for v in x), key=lambda e: json.dumps(e, sort_keys=True))
        return {"$set": [t.__name__, items]}
    mod = t.__module__
    if mod.startswith("lxml."):
        from lxml import etree

        if not isinstance(x, etree._Element):
            tags.add("unreplayable")
            return {"$object": [f"{mod}.{t.__qualname__}", "lxml object"]}
        html = mod == "lxml.html"
        if not html and t is not etree._Element:
            tags.add("unreplayable")
        tags.add("tree-input")
        return {"$tree": {"html": html, "nodes": _tree_nodes(x)}}
    if mod.startswith("trafilatura"):
        name = t.__qualname__
        if name == "LRUCache":
            with x.lock:
                items = [[encode_arg(k, blobs, tags), v] for k, v in _lru_items(x)]
            return {"$lru": {"maxsize": x.maxsize, "items": items}}
        if name == "Extractor" or any(c.__qualname__ == "Extractor" for c in t.__mro__[1:]):
            if name != "Extractor":
                tags.add("extractor-subclass")
            slots = {s: encode_arg(getattr(x, s, None), blobs, tags) for s in x.__slots__}
            return {"$extractor": slots}
    if mod == "configparser":
        from trafilatura.settings import DEFAULT_CONFIG

        if x is DEFAULT_CONFIG and not getattr(blobs, "default_config_mutated", False):
            return {"$config": "default"}
        sections = {name: dict(x.items(name, raw=True)) for name in x.sections()}
        return {"$config": {"defaults": dict(x.defaults()), "sections": sections}}
    tags.add("unreplayable")
    return {"$object": [f"{mod}.{t.__qualname__}", repr(x)[:200]]}


def decode_arg(x: Any, blobs: Blobs) -> Any:
    """Inverse of ``encode_arg``: a fresh object each call (trees, caches and configs are never shared)."""
    if isinstance(x, list):
        return [decode_arg(v, blobs) for v in x]
    if not isinstance(x, dict):
        return x
    if len(x) == 1:
        ((tag, v),) = x.items()
        if tag == "$testfile":
            if blobs.root is None:
                raise ValueError(
                    "a recorded call names an upstream test file: rebuild the tests and set root"
                )
            return str(Path(blobs.root) / v)
        if tag == "$blob":
            raw = blobs.get(v[0])
            return raw.decode("utf-8", "surrogatepass") if v[1] == "str" else raw
        if tag == "$tuple":
            return tuple(decode_arg(e, blobs) for e in v)
        if tag == "$dict":
            return {decode_arg(k, blobs): decode_arg(e, blobs) for k, e in v}
        if tag == "$set":
            items = [decode_arg(e, blobs) for e in v[1]]
            return frozenset(items) if v[0] == "frozenset" else set(items)
        if tag == "$tree":
            return _build_tree(v["nodes"], v["html"])
        if tag == "$lru":
            from trafilatura.deduplication import LRUCache

            lru = LRUCache(maxsize=v["maxsize"])
            for k, n in v["items"]:  # oldest first, so the rebuilt cache has the same order
                lru.put(decode_arg(k, blobs), n)
            return lru
        if tag == "$config":
            from configparser import ConfigParser

            if v == "default":
                try:
                    from trafilatura.settings import DEFAULT_CONFIG
                except ImportError:
                    # No trafilatura here: the candidate's own default stands for trafilatura's. Not None,
                    # unless that is the candidate's default: 1.11.0's bare_extraction raises on config=None.
                    return blobs.default_config
                return DEFAULT_CONFIG
            cp = ConfigParser()
            cp.read_dict({"DEFAULT": v["defaults"], **v["sections"]})
            return cp
        if tag == "$extractor":
            from trafilatura.settings import Extractor

            ex = object.__new__(Extractor)
            for slot, e in v.items():
                setattr(ex, slot, decode_arg(e, blobs))
            return ex
        if tag.startswith("$"):
            return JC.decode(x)
    return {k: decode_arg(v, blobs) for k, v in x.items()}
