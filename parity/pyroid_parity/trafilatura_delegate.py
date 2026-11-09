"""Run one option case against a candidate and against trafilatura in a fresh process; print JSON.

    python -m pyroid_parity.trafilatura_delegate MODULE GROUP CASE
    python -m pyroid_parity.trafilatura_delegate --shim TOP      # signatures of TOP.integrations' trafilatura shim

``GROUP`` is ``delegated`` or ``edge`` (``specs.trafilatura_cases``) or ``native`` (the corpus option sets and
``bare:<name>`` for the bare ones), all of the mode in $PYROID_TRAFILATURA_COMPAT (default 2.3.1).
The candidate gets ``DOCS`` in order with the case's options, each call with fresh argument objects, under
``warnings.simplefilter("always")``: pyroid's "once per process, function and reason" rule is its own, so the
count of ``pyroid.FallbackWarning`` over the calls is what a user sees. A fresh process is the only way to see
the first call of a reason. The result is one JSON object: per call, the candidate's and trafilatura's outcomes
(``trafilatura_compare.outcome_json``, FallbackWarning left out) and the exact category names of the candidate's
warnings.

``$``-values in a case are built here, because they need trafilatura or lxml: ``$input`` (``tree``: an lxml tree
from ``lxml.html.fromstring``; ``utf-8-bytes``, ``latin-1-bytes``), ``$config:K=V,...`` (``use_config()`` with
those settings), ``$settingsfile:K=V,...`` (the same, written to a file), ``$extractor:precision``
(``Extractor(precision=True, comments=False)``), ``$lrucache:N`` (a new ``LRUCache(maxsize=N)``) and ``$fn``
(``bare_extraction`` instead of ``extract``).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from pyroid_parity import impl as I
from pyroid_parity import specs
from pyroid_parity import trafilatura_compare as T

_BODY = (
    "<p>Trafilatura extracts the main text of a page. This paragraph is long enough to count as content: "
    "it has <b>bold words</b>, <i>italics</i> and <a href='/docs/intro.html'>a relative link</a> to follow, "
    "and it keeps going for a while so that the extractors do not treat it as boilerplate. Café, naïve, "
    "résumé and Zürich keep the encoding cases honest.</p>"
    "<p>A second paragraph with an <a href='https://example.org/other'>absolute link</a>, a footnote<sup>1</sup> "
    "and some more ordinary sentences. Readers who reach this point have seen enough text for the length checks "
    "to pass, which is the point of writing it out in full.</p>"
    "<img src='/img/photo.jpg' alt='A photo of the river'/>"
    "<h2>Numbers</h2><table><tr><th>Year</th><th>Value</th></tr><tr><td>2019</td><td>12</td></tr>"
    "<tr><td>2023</td><td>17</td></tr></table>"
    "<ul><li>first item of a list</li><li>second item, <code>with code</code></li></ul>"
    "<blockquote>A quoted passage that stands apart from the rest.</blockquote>"
)
_ARTICLE = (
    "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>River notes | Example News</title>"
    "<meta name='author' content='Jane Doe'><meta name='description' content='Notes on a river.'>"
    "<meta property='article:published_time' content='2024-05-01T10:00:00Z'>"
    "<link rel='canonical' href='https://example.org/blocked'></head><body>"
    "<nav><a href='/'>Home</a> <a href='/news'>News</a></nav>"
    f"<article><h1>River notes</h1>{_BODY}</article>"
    "<div class='comments' id='comments'><h3>Comments</h3><p>First! A reader comment that is long enough to "
    "be kept as a comment by the extractor, with a few more words.</p></div>"
    "<footer class='footer'><p>Copyright Example News. All rights reserved.</p></footer></body></html>"
)
_GERMAN = (
    "<html><head><title>Zweite Seite</title><meta name='author' content='Max Muster'></head><body>"
    "<div id='main' class='post-content'><h1>Ein Beitrag</h1>"
    "<p>Dies ist ein deutscher Absatz mit genug Text, damit er als Inhalt zählt. Er erwähnt Straße, Größe und "
    "Übermut, und er läuft noch eine Weile weiter, damit die Längenprüfungen bestanden werden.</p>"
    f"{_BODY}</div><div class='share'>Teilen: Facebook Twitter</div></body></html>"
)
_SHORT = "<html><body><div class='nav'><a href='/a'>A</a></div><p>Too short.</p></body></html>"
#: The pages each case runs on, in order: an English article with metadata, comments, a table, links and an
#: image; a German page (the target_language case drops it); a page too short to extract.
DOCS = (_ARTICLE, _GERMAN, _SHORT)


def case_kwargs(group: str, name: str, mode: str | None = None) -> dict[str, Any]:
    """A case of the mode checked (``specs.trafilatura_mode()``)."""
    mode = mode or specs.trafilatura_mode()
    if group in ("delegated", "edge"):
        return dict(specs.trafilatura_cases(mode, group)[name])
    if group == "native":
        if name.startswith("bare:"):
            bare = specs.TRAFILATURA_BARE_SETS_BY_MODE[mode][name.removeprefix("bare:")]
            return {"$fn": "bare_extraction", **bare}
        return dict(specs.TRAFILATURA_OPTION_SETS_BY_MODE[mode][name])
    raise SystemExit(f"unknown group {group!r}")


def _settings(spec: str) -> Any:
    from trafilatura.settings import use_config

    cfg = use_config()
    for item in spec.split(","):
        key, _, value = item.partition("=")
        cfg.set("DEFAULT", key, value)
    return cfg


def build(kwargs: dict[str, Any], html: str, tmp: Path) -> tuple[str, Any, dict[str, Any]]:
    """(function name, first argument, keyword arguments) with fresh objects for one call."""
    kw = dict(kwargs)
    fn = kw.pop("$fn", "extract")
    first: Any = html
    if (kind := kw.pop("$input", None)) is not None:
        if kind == "tree":
            from lxml import html as lhtml

            first = lhtml.fromstring(html)
        elif kind == "utf-8-bytes":
            first = html.encode("utf-8")
        elif kind == "latin-1-bytes":
            first = html.replace("charset='utf-8'", "charset='iso-8859-1'").encode(
                "latin-1", "xmlcharrefreplace"
            )
        else:
            raise SystemExit(f"unknown $input {kind!r}")
    for key, value in list(kw.items()):
        if not (isinstance(value, str) and value.startswith("$")):
            continue
        tag, _, arg = value.partition(":")
        if tag == "$config":
            kw[key] = _settings(arg)
        elif tag == "$settingsfile":
            path = tmp / "settings.cfg"
            with path.open("w", encoding="utf-8") as f:
                _settings(arg).write(f)
            kw[key] = str(path)
        elif tag == "$extractor":
            from trafilatura.settings import Extractor

            kw[key] = Extractor(precision=True, comments=False)
        elif tag == "$lrucache":
            from trafilatura.deduplication import LRUCache

            kw[key] = LRUCache(maxsize=int(arg))
        else:
            raise SystemExit(f"unknown marker {value!r}")
    return fn, first, kw


def run(module: str, group: str, name: str) -> dict[str, Any]:
    import trafilatura

    cand = I.resolve("trafilatura", module).obj
    oracle = I.resolve("trafilatura", "trafilatura").obj
    kwargs = case_kwargs(group, name)
    stateful = specs.trafilatura_stateful(kwargs)
    calls = []
    with tempfile.TemporaryDirectory(prefix="pyroid-traf-delegate-") as d:
        tmp = Path(d)
        for i, html in enumerate(DOCS):
            # Each call on its own: with deduplicate=True both sides start from an empty cache (in a self-check
            # the candidate and the oracle are the same module and share it).
            if stateful:
                cand.reset_caches()
            fn, first, kw = build(kwargs, html, tmp)
            got, categories = T.capture_with_warnings(getattr(cand, fn), first, **kw)
            if stateful:
                oracle.reset_caches()
            fn, first, kw = build(kwargs, html, tmp)
            want = T.capture_call(getattr(oracle, fn), first, **kw)
            calls.append(
                {
                    "doc": i,
                    "fn": fn,
                    "got": T.outcome_json(got),
                    "want": T.outcome_json(want),
                    "warnings": categories,
                }
            )
    return {
        "module": module,
        "group": group,
        "case": name,
        "mode": specs.trafilatura_mode(),
        "trafilatura": trafilatura.__version__,
        "calls": calls,
    }


def shim_shapes(top: str) -> dict[str, Any]:
    """Signatures of ``extract`` / ``bare_extraction`` in the module ``<top>.integrations.patch("trafilatura")``
    puts in ``sys.modules``, as (name, kind, default) with ``config``'s default as "DEFAULT_CONFIG" only for
    trafilatura's own object (or for None where trafilatura is not installed)."""
    import importlib
    import inspect
    import warnings

    try:
        integrations = importlib.import_module(f"{top}.integrations")
    except ImportError as e:
        return {"skip": f"{top}.integrations is not importable: {e}"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            integrations.patch("trafilatura")
        except (ValueError, KeyError) as e:
            return {"skip": f"{top}.integrations has no trafilatura shim: {e}"}
    shim = sys.modules["trafilatura"]
    real = getattr(shim, "__wrapped__", None)
    try:
        from trafilatura.settings import DEFAULT_CONFIG
    except ImportError:
        DEFAULT_CONFIG = None
    shapes: dict[str, Any] = {}
    for name in specs.TRAFILATURA_SIGNATURES:
        fn = getattr(shim, name, None)
        if fn is None:
            shapes[name] = None
            continue
        out = []
        for p in inspect.signature(fn).parameters.values():
            d = p.default
            if p.name == "config":
                d = "DEFAULT_CONFIG" if d is DEFAULT_CONFIG else repr(d)
            elif d is inspect.Parameter.empty:
                d = "<required>"
            out.append([p.name, p.kind.name, d])
        shapes[name] = out
    real_version = getattr(real, "__version__", None) if real is not None else None
    return {
        "shapes": shapes,
        "pyroid": getattr(shim, "__pyroid__", False) is True,
        "real_version": real_version,
    }


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if args[:1] == ["--shim"]:
        print(json.dumps(shim_shapes(args[1]), ensure_ascii=True))
        return 0
    module, group, name = args[:3]
    print(json.dumps(run(module, group, name), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
