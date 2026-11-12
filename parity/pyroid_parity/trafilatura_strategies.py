"""Hypothesis strategies for the trafilatura suite: grammar-generated HTML pages (importable without pytest).

    from pyroid_parity import trafilatura_strategies as S
    S.documents()     # whole pages as str: head, body, blocks, inline markup, comments, scripts, malformed parts

The grammar aims at what Trafilatura's XPaths and cleaning look at: block and inline tags, tables, lists, forms,
``class``/``id`` tokens from its boilerplate and content expressions (post, article, comment, footer, nav, share,
...), ``hidden``/``style``/``aria-hidden``/``role`` attributes, comments, scripts, styles and ``noscript``. Text is
drawn from sentences long enough to pass Trafilatura's length thresholds, with repeats (the dedup cache counts
segments within a page too) and with what lxml's str input path treats specially: lone surrogates, NUL and other
control characters, a BOM, entities (named, numeric, invalid) and long single tokens (kept to a few thousand
characters for CI). Pages are made malformed on purpose: unclosed and stray end tags, misnesting, uppercase tags,
attribute values without quotes, processing instructions and CDATA.

Hypothesis never draws lone surrogates or NUL in a structured grammar by chance, so they are explicit pieces here.
"""

from __future__ import annotations

from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

# Tokens Trafilatura's XPath expressions test in class/id attributes (main_extractor, xpaths, htmlprocessing).
CLASS_TOKENS = (
    "post",
    "post-content",
    "entry-content",
    "article",
    "article-body",
    "content",
    "main",
    "story",
    "text",
    "body",
    "comment",
    "comments",
    "commentlist",
    "comment-body",
    "reply",
    "footer",
    "header",
    "nav",
    "navigation",
    "menu",
    "breadcrumb",
    "sidebar",
    "widget",
    "share",
    "social",
    "sharing",
    "related",
    "author",
    "byline",
    "date",
    "meta",
    "tags",
    "hidden",
    "ad",
    "ads",
    "advert",
    "banner",
    "cookie",
    "consent",
    "newsletter",
    "subscribe",
    "teaser",
    "caption",
    "copyright",
    "paywall",
    "premium",
    "disqus_thread",
    "jp-relatedposts",
)
BLOCK_TAGS = ("p", "div", "section", "article", "main", "aside", "header", "footer", "nav", "span", "font")
INLINE_TAGS = (
    "b",
    "i",
    "em",
    "strong",
    "u",
    "s",
    "small",
    "sup",
    "sub",
    "mark",
    "q",
    "abbr",
    "code",
    "kbd",
    "span",
)
WORDS = (
    "the",
    "river",
    "runs",
    "through",
    "a",
    "valley",
    "where",
    "people",
    "have",
    "lived",
    "for",
    "centuries",
    "and",
    "farmers",
    "grow",
    "wheat",
    "café",
    "naïve",
    "Straße",
    "Zürich",
    "日本語",
    "中文",
    "Привет",
    "العربية",
    "ไทย",
    "\N{EM DASH}",
    "\N{LEFT DOUBLE QUOTATION MARK}quoted\N{RIGHT DOUBLE QUOTATION MARK}",
    "it\N{RIGHT SINGLE QUOTATION MARK}s",
    "\N{HORIZONTAL ELLIPSIS}",
    "1,234",
    "2024",
    "e.g.",
    "http://example.org/x",
    "#hashtag",
    "@user",
)
SPECIALS = (
    "\x00",
    "\x01",
    "\x08",
    "\x0b",
    "\x0c",
    "\x1f",
    "\x7f",
    "\x85",
    "\N{NO-BREAK SPACE}",
    "\N{SOFT HYPHEN}",
    "\N{ZERO WIDTH SPACE}",
    "\N{LINE SEPARATOR}",
    "\N{ZERO WIDTH NO-BREAK SPACE}",
    "\ufffe",
    "\ud800",
    "\udfff",
    "\ud83d",
    "\ude00",
    "\U0001f600",
    "\r\n",
    "\r",
    "\t",
)
ENTITIES = (
    "&amp;",
    "&lt;",
    "&gt;",
    "&quot;",
    "&apos;",
    "&nbsp;",
    "&eacute;",
    "&hellip;",
    "&mdash;",
    "&#8217;",
    "&#x20AC;",
    "&#0;",
    "&#xD800;",
    "&#1114112;",
    "&bogus;",
    "&amp",
    "&",
    "&#;",
)


XHTML_DOCTYPE = "<!DOCTYPE html PUBLIC '-//W3C//DTD XHTML 1.0 Strict//EN' 'http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd'>"


def _sentence(draw: st.DrawFn) -> str:
    words = draw(st.lists(st.sampled_from(WORDS), min_size=3, max_size=30))
    return " ".join(words).capitalize() + draw(st.sampled_from([".", "!", "?", ":", "", " ..."]))


@st.composite
def text(draw: st.DrawFn) -> str:
    """A run of text: sentences, entities, specials and now and then one long token."""
    parts = []
    for _ in range(draw(st.integers(1, 6))):
        kind = draw(st.integers(0, 9))
        if kind <= 5:
            parts.append(_sentence(draw))
        elif kind == 6:
            parts.append(draw(st.sampled_from(ENTITIES)))
        elif kind == 7:
            parts.append(draw(st.sampled_from(SPECIALS)))
        elif kind == 8:
            parts.append(draw(st.sampled_from("xa-0")) * draw(st.integers(50, 3000)))
        else:
            parts.append(draw(st.text(max_size=12)))
    return " ".join(parts)


@st.composite
def attrs(draw: st.DrawFn) -> str:
    out = []
    for _ in range(draw(st.integers(0, 3))):
        kind = draw(st.integers(0, 9))
        if kind <= 3:
            tokens = draw(st.lists(st.sampled_from(CLASS_TOKENS), min_size=1, max_size=3))
            out.append(f'class="{" ".join(tokens)}"')
        elif kind <= 5:
            out.append(f"id={draw(st.sampled_from(CLASS_TOKENS))}")
        elif kind == 6:
            out.append(
                draw(
                    st.sampled_from(
                        [
                            'style="display:none"',
                            "hidden",
                            'aria-hidden="true"',
                            'role="navigation"',
                            'role="main"',
                            'itemprop="articleBody"',
                            'itemprop="author"',
                            'data-x="1"',
                        ]
                    )
                )
            )
        elif kind == 7:
            out.append(f"title='{draw(st.sampled_from(WORDS))}'")
        elif kind == 8:
            out.append(f'class="{draw(st.sampled_from(CLASS_TOKENS))}')  # unclosed quote
        else:
            out.append(f'lang="{draw(st.sampled_from(["en", "de", "fr", "zh", "", "x-bogus"]))}"')
    return (" " + " ".join(out)) if out else ""


@st.composite
def inline(draw: st.DrawFn) -> str:
    t = draw(text())
    kind = draw(st.integers(0, 7))
    if kind <= 2:
        return t
    if kind == 3:
        return f"<a href='{draw(st.sampled_from(['/a', 'https://example.org/b', '#c', 'javascript:x()']))}'>{t}</a>"
    if kind == 4:
        return f"<img src='/i.jpg' alt='{draw(st.sampled_from(WORDS))}'>{t}"
    if kind == 5:
        return f"{t}<br>{draw(text())}"
    tag = draw(st.sampled_from(INLINE_TAGS))
    if kind == 6:
        return f"<{tag}>{t}</{tag}>"
    return f"<{tag}>{t}"  # left open


@st.composite
def block(draw: st.DrawFn, depth: int = 0) -> str:
    kind = draw(st.integers(0, 15 if depth < 3 else 6))
    if kind <= 3:
        tag = draw(st.sampled_from(BLOCK_TAGS))
        body = "".join(draw(st.lists(inline(), min_size=1, max_size=4)))
        close = draw(st.sampled_from([f"</{tag}>", f"</{tag}>", f"</{tag}>", "", f"</{tag.upper()}>"]))
        return f"<{tag}{draw(attrs())}>{body}{close}"
    if kind == 4:
        h = draw(st.integers(1, 6))
        return f"<h{h}{draw(attrs())}>{draw(text())}</h{h}>"
    if kind == 5:
        return f"<!-- {draw(text())} -->"
    if kind == 6:
        return draw(
            st.sampled_from(
                [
                    "<script>var x = '<p>not text</p>';</script>",
                    "<style>p { color: red }</style>",
                    "<noscript><p>Enable JavaScript to see this page.</p></noscript>",
                    '<script type="application/ld+json">{"@type": "DiscussionForumPosting"}</script>',
                    "<iframe src='/x'></iframe>",
                    "<?php echo 1; ?>",
                    "<![CDATA[ raw ]]>",
                    "</div>",
                    "</p></p>",
                    "<hr>",
                    "<b><i>misnested</b></i>",
                ]
            )
        )
    inner = depth + 1
    if kind <= 8:
        tag = draw(st.sampled_from(BLOCK_TAGS))
        children = "".join(draw(st.lists(block(inner), min_size=1, max_size=4)))
        return f"<{tag}{draw(attrs())}>{children}</{tag}>"
    if kind == 9:
        tag = draw(st.sampled_from(["ul", "ol", "dl"]))
        item = "dd" if tag == "dl" else "li"
        items = "".join(
            f"<{item}>{draw(inline())}" + draw(st.sampled_from([f"</{item}>", ""])) for _ in range(3)
        )
        return f"<{tag}{draw(attrs())}>{items}</{tag}>"
    if kind == 10:
        rows = []
        for _ in range(draw(st.integers(1, 4))):
            cells = "".join(
                f"<{draw(st.sampled_from(['td', 'th']))}{draw(attrs())}>{draw(inline())}" for _ in range(3)
            )
            rows.append(f"<tr>{cells}" + draw(st.sampled_from(["</tr>", ""])))
        nested = draw(block(inner)) if draw(st.booleans()) else ""
        return f"<table{draw(attrs())}>{''.join(rows)}<tr><td>{nested}</td></tr></table>"
    if kind == 11:
        return (
            f"<form{draw(attrs())}><label>{draw(text())}</label><input type=text name=q value='x'>"
            "<select><option>One</option><option>Two</option></select><button>Send</button></form>"
        )
    if kind == 12:
        return f"<blockquote>{draw(inline())}</blockquote>" + draw(
            st.sampled_from(["", f"<pre>{draw(text())}</pre>"])
        )
    if kind == 13:
        return f"<figure><img src='/f.png'><figcaption>{draw(text())}</figcaption></figure>"
    if kind == 14:
        para = draw(block(inner))
        return para * draw(st.integers(2, 4))  # repeated segments: the dedup cache counts them
    return f"<details><summary>{draw(text())}</summary>{draw(block(inner))}</details>"


@st.composite
def documents(draw: st.DrawFn) -> str:
    """A whole page as str (sometimes a bare fragment, sometimes with a byte-order mark or junk first)."""
    head = []
    if draw(st.booleans()):
        head.append(f"<title>{draw(text())}</title>")
    if draw(st.booleans()):
        head.append("<meta charset='utf-8'><meta name='author' content='Jane Doe'>")
    if draw(st.booleans()):
        head.append("<script>window.x = 1;</script><style>.a{}</style>")
    body = "".join(draw(st.lists(block(), min_size=1, max_size=8)))
    start = draw(
        st.sampled_from(
            [
                "<!DOCTYPE html>",
                "",
                "\N{ZERO WIDTH NO-BREAK SPACE}<!DOCTYPE html>",
                "<?xml version='1.0' encoding='utf-8'?>",
                XHTML_DOCTYPE,
                "  \n",
            ]
        )
    )
    shape = draw(st.integers(0, 5))
    if shape == 0:
        return start + body  # a fragment
    if shape == 1:
        return f"{start}<html><body{draw(attrs())}>{body}"  # no end tags
    lang = draw(st.sampled_from(["", " lang='en'", " lang='de'", " LANG=fr"]))
    return f"{start}<html{lang}><head>{''.join(head)}</head><body{draw(attrs())}>{body}</body></html>"


def option_names() -> SearchStrategy[str]:
    from pyroid_parity import specs

    return st.sampled_from(sorted(specs.TRAFILATURA_OPTION_SETS))
