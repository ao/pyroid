"""Shared inputs for the pyroid.trafilatura tests: small pages and a Hypothesis HTML strategy.

Not a test module: the test_trafilatura*.py modules load it by path (``load()`` in each), since
``--import-mode=importlib`` keeps ``python/tests`` off ``sys.path``.
"""

from __future__ import annotations

from typing import Any

from hypothesis import strategies as st

PARAGRAPH = (
    "Trafilatura keeps the main text of a page and drops the navigation, the footer and the "
    "other boilerplate around it. This paragraph is long enough to count as content, and it "
    "mentions Café, naïve, Zürich and 日本語 so that the encodings stay honest."
)

#: Small pages covering the main paths: article, comments, forum posts, tables, lists, too
#: short, not HTML, lone surrogates, control characters, a faulty <html/>, a BOM, duplicates.
PAGES: dict[str, str] = {
    "article": (
        "<!DOCTYPE html><html lang='en'><head><title>River notes</title></head><body>"
        "<nav><a href='/'>Home</a> <a href='/news'>News</a></nav>"
        f"<article><h1>River notes</h1><p>{PARAGRAPH}</p><p>A second paragraph with "
        "<b>bold</b>, <i>italics</i> and <a href='/x'>a link</a>, which keeps going for a "
        "while so the length checks pass without any doubt at all.</p>"
        "<ul><li>first item of a list</li><li>second item</li></ul>"
        "<blockquote>A quoted passage.</blockquote></article>"
        "<div class='comments' id='comments'><h3>Comments</h3><p>First! A reader comment "
        "that is long enough to be kept as a comment by the extractor.</p></div>"
        "<footer><p>Copyright Example News. All rights reserved.</p></footer></body></html>"
    ),
    "table": (
        "<html><body><div class='post-content'><h2>Numbers</h2>"
        f"<p>{PARAGRAPH}</p><table><tr><th>Year</th><th>Value</th></tr>"
        "<tr><td>2019</td><td>12</td></tr><tr><td>2023</td><td>17</td></tr></table>"
        "</div></body></html>"
    ),
    "forum": (
        "<html><head><script type='application/ld+json'>{\"@type\": "
        '"DiscussionForumPosting", "headline": "A thread"}</script></head><body>'
        "<div class='comments'><div class='comment'><p>"
        + "Post one is about rivers. " * 12
        + "</p></div><div class='comment'><p>"
        + "Post two answers it. " * 12
        + "</p></div></div></body></html>"
    ),
    "short": "<html><body><div class='nav'><a href='/a'>A</a></div><p>Too short.</p></body></html>",
    "empty": "",
    "not_html": "just some text without any markup at all",
    "fragment": "<p>" + "word " * 80 + "</p>",
    "surrogate": f"<html><body><p>lone \ud800 surrogate {PARAGRAPH}</p></body></html>",
    "controls": f"<html><body><p>a NUL \x00 and \x01 controls {PARAGRAPH}</p></body></html>",
    "faulty_html": f"<html lang='en'/>\n<body><article><p>{PARAGRAPH}</p></article></body>",
    "bom": f"﻿<html><body><div class='comments'><p>{'c ' * 200}</p></div><p>{PARAGRAPH}</p>"
    "</body></html>",
    "repeated": "<html><body>"
    + "<p>The same paragraph, repeated on purpose, long enough to be checked.</p>" * 6
    + "</body></html>",
    "entities": f"<html><body><p>&amp; &lt;tag&gt; &#169; &#x1F600; &nbsp; {PARAGRAPH}</p>"
    "</body></html>",
}

#: Extraction option sets: trafilatura's default, datatrove's call, and the main switches.
OPTION_SETS: dict[str, dict[str, Any]] = {
    "default": {},
    "datatrove": {"favor_precision": True, "include_comments": False, "deduplicate": True},
    "recall": {"favor_recall": True},
    "fast": {"fast": True},
    "no_tables": {"include_tables": False},
    "no_comments": {"include_comments": False},
    "precision_comments": {"favor_precision": True},
}

_WORDS = [
    "the", "river", "page", "Trafilatura", "extracts", "text", "and", "of", "to", "café",
    "naïve", "Zürich", "日本語", "текст", "مرحبا", "😀", "&amp;", "&lt;", "&#169;", "&nbsp;",
    "&#x1F600;", "&#0;", "Share", "Comments", "Copyright", "Read more", "Login", "Menu", "Home",
    "Next", "Reply", "\ud800", "\x00", "\t", "\n", "  ", "\xa0", "\u2028", "Ø",
]  # fmt: skip
_CLASSES = [
    "", "content", "article", "post", "entry-content", "comments", "comment", "footer", "nav",
    "sidebar", "share", "related", "ad", "main", "byline", "author", "date", "hidden",
    "post-content", "discussion", "teaser", "widget", "cookie",
]  # fmt: skip
_CONTAINERS = [
    "div", "section", "article", "main", "p", "blockquote", "ul", "ol", "li", "table", "tr",
    "td", "th", "h1", "h2", "h3", "h4", "header", "footer", "nav", "aside", "pre", "code",
    "form", "figure", "span", "a", "b", "i", "em", "strong", "dl", "dt", "dd", "details",
    "summary", "q", "del", "sup", "font", "center", "noscript", "button", "label", "time",
]  # fmt: skip
_VOID = ["br", "hr", "img src='/i.png' alt='an image'", "input", "wbr", "meta", "link"]
_JUNK = [
    "<!-- a comment -->", "</div>", "</p>", "<p", "<![CDATA[x]]>", "<?php echo 1 ?>", "&",
    "<script>var x = '<p>not text</p>';</script>", "<style>p { color: red }</style>",
    "<!DOCTYPE html>", "<html/>", "</html>", "<body>", "<title>A title</title>", "\x0c",
]  # fmt: skip

words = st.lists(
    st.one_of(st.sampled_from(_WORDS), st.text(max_size=6)), min_size=1, max_size=30
).map(" ".join)


def _element(children: Any) -> Any:
    tag = st.sampled_from(_CONTAINERS)
    cls = st.sampled_from(_CLASSES)
    ident = st.sampled_from(_CLASSES)

    def render(t: str, c: str, i: str, kids: list[str]) -> str:
        attrs = (f" class='{c}'" if c else "") + (f" id='{i}'" if i else "")
        return f"<{t}{attrs}>{''.join(kids)}</{t}>"

    return st.builds(render, tag, cls, ident, st.lists(children, max_size=5))


nodes = st.recursive(
    st.one_of(
        words,
        st.sampled_from(_VOID).map(lambda v: f"<{v}>"),
        st.sampled_from(_JUNK),
    ),
    _element,
    max_leaves=25,
)


@st.composite
def documents(draw: st.DrawFn) -> str:
    """An HTML page from a tag grammar: boilerplate classes, entities, junk, odd characters."""
    body = "".join(draw(st.lists(nodes, min_size=1, max_size=6)))
    if draw(st.booleans()):
        body += f"<article><p>{PARAGRAPH}</p>{draw(nodes)}</article>"
    head = draw(
        st.sampled_from(["", "<head><title>T</title></head>", "<head><meta charset='utf-8'>"])
    )
    shell = draw(st.sampled_from(["full", "body", "bare", "doctype"]))
    if shell == "full":
        return f"<html>{head}<body>{body}</body></html>"
    if shell == "body":
        return f"<body>{body}</body>"
    if shell == "doctype":
        return f"<!DOCTYPE html>\n<html lang='en'>{head}<body>{body}</body></html>"
    return body


#: trafilatura 2.3.1's outcome for every page (rows) and option set (columns, in OPTION_SETS
#: order): sha256 of ``repr(("ok", result))`` or ``repr(("raise", type, message))``, cut to 16 hex
#: digits. Recorded with trafilatura itself (test_trafilatura_parity.py checks them against it).
#: Option sets with deduplicate=True start from an empty cache.
PINNED: dict[str, str] = {
    "article": "d119fce5158a00b7 e90d03e86dc7299d d119fce5158a00b7 d119fce5158a00b7 "
    "d119fce5158a00b7 e90d03e86dc7299d d119fce5158a00b7",
    "table": "14f41940ea4e7f18 14f41940ea4e7f18 14f41940ea4e7f18 14f41940ea4e7f18 "
    "4851a5a9b709d295 14f41940ea4e7f18 14f41940ea4e7f18",
    "forum": "b01b425079df9572 9c63cb0c2c90900a b01b425079df9572 b01b425079df9572 "
    "b01b425079df9572 b01b425079df9572 b01b425079df9572",
    "short": "56f6710b5e381c93 949913fd1bf79061 56f6710b5e381c93 56f6710b5e381c93 "
    "56f6710b5e381c93 56f6710b5e381c93 949913fd1bf79061",
    "empty": " ".join(["9c63cb0c2c90900a"] * 7),
    "not_html": " ".join(["9c63cb0c2c90900a"] * 7),
    "fragment": " ".join(["9c63cb0c2c90900a"] * 7),
    "surrogate": " ".join(["5c2faeb38fd026a0"] * 7),
    "controls": " ".join(["436e6e5e2adf5ae4"] * 7),
    "faulty_html": " ".join(["c9a87099c19321c5"] * 7),
    "bom": "5db3ec12c75f9954 c9a87099c19321c5 5db3ec12c75f9954 5db3ec12c75f9954 "
    "5db3ec12c75f9954 c9a87099c19321c5 5db3ec12c75f9954",
    "repeated": " ".join(["0e967eaa3b84bbee"] * 7),
    "entities": " ".join(["7c1191c8dea18f6c"] * 7),
}

#: Calls pyroid hands to trafilatura: case -> (function, keyword arguments).
DELEGATED_CALLS: dict[str, tuple[str, dict[str, Any]]] = {
    "output_format=xml": ("extract", {"output_format": "xml"}),
    "output_format=json": ("extract", {"output_format": "json"}),
    "output_format=python": ("extract", {"output_format": "python"}),
    "output_format=markdown": ("extract", {"output_format": "markdown"}),
    "include_formatting": ("extract", {"include_formatting": True}),
    "include_links": ("extract", {"include_links": True}),
    "include_images": ("extract", {"include_images": True}),
    "with_metadata": ("extract", {"with_metadata": True}),
    "only_with_metadata": ("extract", {"only_with_metadata": True}),
    "url_blacklist": ("extract", {"url_blacklist": {"https://example.org/x"}}),
    "author_blacklist": ("extract", {"author_blacklist": {"Jane"}}),
    "target_language": ("extract", {"target_language": "en"}),
    "prune_xpath": ("extract", {"prune_xpath": "//p"}),
    "bare:output_format=txt": ("bare_extraction", {"output_format": "txt"}),
    "bare:with_metadata": ("bare_extraction", {"with_metadata": True}),
    "metadata": ("extract_with_metadata", {}),
    "batch:include_links": ("extract_batch", {"include_links": True}),
}


#: OPTION_SETS for trafilatura 1.11.0, which has no ``fast``: its switch is ``no_fallback``.
OPTION_SETS_111: dict[str, dict[str, Any]] = {
    name: ({"no_fallback": True} if kw == {"fast": True} else kw)
    for name, kw in OPTION_SETS.items()
}

#: Calls compat="1.11.0" hands to trafilatura 1.11.0 (it has no ``extract_with_metadata``).
DELEGATED_CALLS_111: dict[str, tuple[str, dict[str, Any]]] = {
    case: call
    for case, call in DELEGATED_CALLS.items()
    if case not in ("metadata", "author_blacklist")  # 1.11.0 needs no metadata for author_blacklist
}

#: PINNED for trafilatura 1.11.0 and OPTION_SETS_111 (recorded with trafilatura 1.11.0).
PINNED_111: dict[str, str] = {
    "article": "d119fce5158a00b7 e90d03e86dc7299d d119fce5158a00b7 d119fce5158a00b7 "
    "d119fce5158a00b7 e90d03e86dc7299d d119fce5158a00b7",
    "table": "b3d4525cf85e19a1 b3d4525cf85e19a1 b3d4525cf85e19a1 b3d4525cf85e19a1 "
    "4851a5a9b709d295 b3d4525cf85e19a1 b3d4525cf85e19a1",
    "forum": "86bfa0473983dfbf b01b425079df9572 86bfa0473983dfbf 86bfa0473983dfbf "
    "86bfa0473983dfbf b01b425079df9572 86bfa0473983dfbf",
    "short": " ".join(["56f6710b5e381c93"] * 7),
    "empty": " ".join(["9c63cb0c2c90900a"] * 7),
    "not_html": " ".join(["9c63cb0c2c90900a"] * 7),
    "fragment": " ".join(["9c63cb0c2c90900a"] * 7),
    "surrogate": " ".join(["5c2faeb38fd026a0"] * 7),
    "controls": " ".join(["9c63cb0c2c90900a"] * 7),
    "faulty_html": " ".join(["58d72e5a0863c917"] * 7),
    "bom": "5db3ec12c75f9954 19901e145035859c 5db3ec12c75f9954 5db3ec12c75f9954 "
    "5db3ec12c75f9954 19901e145035859c 5db3ec12c75f9954",
    "repeated": " ".join(["0e967eaa3b84bbee"] * 7),
    "entities": " ".join(["7c1191c8dea18f6c"] * 7),
}
