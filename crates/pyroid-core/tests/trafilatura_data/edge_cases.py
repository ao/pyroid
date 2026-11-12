"""Hand-written pages for the rare paths of trafilatura 2.3.1 and 1.11.0
(tests/trafilatura_edge.rs).

    /target/venv-dt/bin/python -I edge_cases.py edge.json
    /target/venv-t111/bin/python -I edge_cases.py edge-1.11.0.json

Each page runs through trafilatura.extract and bare_extraction with several option sets; the
fixture keeps the pages (as surrogatepass UTF-8 in hex where they hold lone surrogates) and the
results: the text, None, or the exception extract raises (2.3.1: its type; 1.11.0: "Type:
message"). 1.11.0 also runs the pages of CASES_111, for paths only it has (its baseline's
AttributeError, the int() of a row span in xmltotxt, the TypeErrors it catches, and characters
lxml refuses to assign in every place the extraction copies text from); it takes
`no_fallback` for `fast`, and its commentsbody is None without comments.
"""

# ruff: noqa: T201  (a recording script: it reports on stdout)

import json
import sys

import trafilatura
from lxml import etree

V1 = trafilatura.__version__.startswith("1.")


OPTION_SETS = {
    "default": {},
    "precision": {"favor_precision": True, "include_comments": False},
    "recall": {"favor_recall": True},
    "precision_comments": {"favor_precision": True, "include_comments": True},
    "fast": {"fast": True},
    "notables": {"include_tables": False},
}

LONG = (
    "The council met on Tuesday to discuss the new budget, and after a long debate the members "
    "agreed on a plan that keeps the library open on weekends while the pool closes in winter. "
)


def page(body, head="", title="t"):
    return f"<html><head><title>{title}</title>{head}</head><body>{body}</body></html>"


def ld(obj):
    return f'<script type="application/ld+json">{json.dumps(obj)}</script>'


CASES = {
    "control char joined by delete_element": page(
        f"<article><p>{LONG}</p><nav>menu</nav>&#1;tail {LONG}</article>"
    ),
    "form feed in a block for baseline": page(f"<div><p>short</p><div>a&#12;b {LONG}</div></div>"),
    "unit separator for html2txt": page(
        f"<main><p>{LONG}</p></main><div><div>x&#31;y</div>{LONG * 6}</div>"
    ),
    "colon tag in a list item": page(
        f"<article><p>{LONG}</p><ul><li><o:p>Item text one that is long enough</o:p></li>"
        f"<li>two</li></ul><p>{LONG}</p></article>"
    ),
    "json-ld body with markup": page(
        "<div><p>Short teaser.</p></div>",
        ld(
            {
                "@type": "NewsArticle",
                "articleBody": (
                    f"&lt;p&gt;{LONG}&lt;/p&gt;&lt;p&gt;Second &amp;amp; last {LONG}&lt;/p&gt;"
                ),
            }
        ),
    ),
    "json-ld lone surrogate and entity": page(
        "<div><p>Short teaser.</p></div>",
        '<script type="application/ld+json">{"@type": "Article", "articleBody": "&am\\ud800p; '
        + LONG
        + ' \\ud83d\\ude00 end"}</script>',
    ),
    "json-ld graph, steps, answers, teaser": page(
        "<div><p>Short teaser.</p></div>",
        ld(
            {
                "@graph": [
                    {
                        "@type": "Recipe",
                        "recipeInstructions": [
                            {"@type": "HowToStep", "text": LONG},
                            {"itemListElement": [{"text": "Stir well and serve."}]},
                            "Plain step.",
                        ],
                    },
                    {"@type": "FAQPage", "mainEntity": [{"acceptedAnswer": {"text": LONG}}]},
                    {"@type": ["Product", "Thing"], "description": "A product teaser " + LONG},
                ]
            }
        ),
    ),
    "json-ld teaser only": page(
        "<div>x</div>", ld({"@type": "VideoObject", "description": LONG * 2})
    ),
    "json-ld deep nesting": page(
        "<div>x</div>",
        '<script type="application/ld+json">'
        + "[" * 1200
        + '{"articleBody": "x"}'
        + "]" * 1200
        + "</script>",
    ),
    "discourse preload": page(
        "<div id='data-preloaded' data-preloaded='"
        + json.dumps(
            {
                "topic_1": json.dumps(
                    {
                        "post_stream": {
                            "posts": [
                                {"cooked": f"<p>{LONG}</p>"},
                                {"cooked": "<p>Second post here.</p>"},
                            ]
                        }
                    }
                )
            }
        ).replace("'", "&#39;")
        + "'></div><div>x</div>"
    ),
    "discourse preload with a number for posts": page(
        "<div id='data-preloaded' data-preloaded='"
        + json.dumps({"topic_1": json.dumps({"post_stream": {"posts": 5}})})
        + "'></div><div>x</div>"
    ),
    "huge colspan": page(
        f"<article><p>{LONG}</p><table><tr><td colspan='{'9' * 5000}'>cell</td><td>b</td></tr>"
        "</table></article>"
    ),
    "huge char reference in text": page(
        f"<article><p>{LONG} &amp;#{'1' * 5000} end</p><p>{LONG}</p></article>"
    ),
    "forum thread": page(
        f"<div class='post'><p>{LONG}</p></div><div id='comments'>"
        f"<div class='comment'><p>First reply {LONG}</p></div>"
        f"<div class='comment'><p>Second reply that says something else entirely.</p></div></div>",
        ld({"@type": "DiscussionForumPosting", "headline": "h"}),
    ),
    "comments section": page(
        f"<article><p>{LONG}</p><p>{LONG}</p></article>"
        "<section id='comments'><p>Nice article, thanks for writing it.</p>"
        f"<p>I disagree with the second point.</p><div id='respond'>Leave a reply</div></section>"
    ),
    "readability wins": page(
        "<div><a href='/'>Home</a> <a href='/b'>Blog</a></div>"
        + "".join(
            f"<div class='para'>{LONG} sentence {i}, with commas, and more, words.</div>"
            for i in range(6)
        )
    ),
    "video iframe": page(
        f"<div class='content'><p>{LONG}</p>"
        f"<iframe src='https://www.youtube.com/embed/x'></iframe><p>{LONG}</p></div>"
    ),
    "tables": page(
        "<article><table><caption>Results <b>2026</b></caption>"
        "<thead><tr><th>Name</th><th colspan='2'>Score</th></tr></thead>"
        "<tbody><tr><td rowspan='2'>Ann</td><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr>"
        "<tr><td>Bob</td><td><ul><li>x</li><li>y</li></ul></td>"
        "<td><table><tr><td>inner</td></tr></table></td></tr></tbody></table>"
        f"<p>{LONG}</p></article>"
    ),
    "math, code, quotes": page(
        f"<article><p>{LONG}<math display='block'>"
        "<annotation encoding='application/x-tex'>x^2</annotation></math> after</p>"
        "<pre>def f(x):\n    return {x}</pre><pre><span class='hljs-keyword'>if</span> x</pre>"
        "<blockquote><p>Quoted words here.</p></blockquote>"
        "<p>Inline <sup></sup><sub>2</sub> and "
        "<strong class='schema-faq-question'>Question?</strong></p>"
        f"<p>{LONG}</p></article>"
    ),
    "literal graphic and ref tags": page(
        f"<div><p>{LONG}</p></div><div class='x'>"
        "<graphic src='//img.example/a.png' alt='Alt'>cap</graphic>"
        f"<ref target='u'>link [1]</ref> {LONG * 3}</div>"
    ),
    "lists and details": page(
        f"<article><dl><dt>Term</dt><dd>Definition {LONG}</dd></dl>"
        "<ol><li>One<ul><li>Nested</li></ul></li><li>Two</li></ol>"
        f"<details><summary>More</summary><p>{LONG}</p></details></article>"
    ),
    "xml declaration": '<?xml version="1.0" encoding="utf-8"?>'
    + page(f"<article><p>{LONG}</p><p>{LONG}</p></article>"),
    "lone surrogate in the page": page(
        f"<article><p>{LONG} \udc80 tail</p><p>{LONG}</p></article>"
    ),
    "deep nesting": page("<div>" * 300 + LONG + "</div>" * 300 + f"<p>{LONG}</p>"),
    "fragment, not a page": f"<p>{LONG}</p><p>{LONG}</p>",
    "no html at all": "just some text " * 20,
}


CASES_111 = {
    "json-ld body of one paragraph": page(
        "<div><p>Short teaser.</p></div>", ld({"@type": "Article", "articleBody": f"<p>{LONG}</p>"})
    ),
    "json-ld body of two paragraphs": page(
        "<div><p>Short teaser.</p></div>",
        ld({"@type": "Article", "articleBody": f"<p>{LONG}</p><p>{LONG}</p>"}),
    ),
    "json-ld body is a number": page(
        "<div><p>Short teaser.</p></div>", ld({"@type": "Article", "articleBody": 5})
    ),
    "json-ld body is zero": page(
        "<div><p>Short teaser.</p></div>", ld({"@type": "Article", "articleBody": 0.0})
    ),
    "json-ld body is a list": page(
        "<div><p>Short teaser.</p></div>", ld({"@type": "Article", "articleBody": [LONG]})
    ),
    "json-ld body with a lone surrogate": page(
        "<div><p>Short teaser.</p></div>",
        '<script type="application/ld+json">{"articleBody": "' + LONG + ' \\udc80 end"}</script>',
    ),
    "json-ld body with a control character": page(
        "<div><p>Short teaser.</p></div>",
        '<script type="application/ld+json">{"articleBody": "' + LONG + ' \\u0001 end"}</script>',
    ),
    "json-ld bodies then an article": page(
        f"<article><p>{LONG}</p></article>",
        ld({"@type": "Article", "articleBody": "Tiny."}) + ld({"articleBody": "Also tiny."}),
    ),
    "row with a bad span in a code block": page(
        f"<article><p>{LONG}</p><pre lang='x'><row span='abc'></row></pre><p>{LONG}</p></article>"
    ),
    "row with a spaced span in a code block": page(
        f"<article><p>{LONG}</p><pre lang='x'><row span=' 3 '></row><row span='1_0'></row></pre>"
        f"<p>{LONG}</p></article>"
    ),
    "colspan past sys.maxsize": page(
        f"<article><p>{LONG}</p><table><tr><td colspan='99999999999999999999'>cell one</td></tr>"
        f"<tr><td>cell two</td></tr></table><p>{LONG}</p></article>"
    ),
    "colspan too large to allocate": page(
        f"<article><p>{LONG}</p><table><tr><th colspan='4611686018427387904'>head</th></tr>"
        f"<tr><td>cell</td></tr></table><p>{LONG}</p></article>"
    ),
    "colspan not a number": page(
        f"<article><p>{LONG}</p><table><tr><td colspan='2x'>a</td></tr></table></article>"
    ),
    "colspans with header rows": page(
        f"<article><p>{LONG}</p><table><tr><th>a</th><th colspan='+2'>b</th></tr>"
        "<tr><td>1</td><td>2</td></tr>"
        f"<tr><td colspan='-1'>3</td></tr></table><p>{LONG}</p></article>"
    ),
    "hi without rend in a paragraph": page(
        f"<article><p>{LONG} <hi>x</hi> end</p><p>{LONG}</p></article>"
    ),
    "attribute with a colon in a list item": page(
        f"<article><p>{LONG}</p><ul><li>Item <x data:y='1'>sub</x> long enough text here</li>"
        f"<li>two</li></ul><p>{LONG}</p></article>"
    ),
    "doctype with a slash": "<!DOCTYPE a/b>x/><html><body><article><p>"
    + LONG
    + "</p></article></body></html>",
    "raw control characters": page(
        f"<article><p>{LONG}\x01 \x0c tail</p><p>{LONG}\x1f</p></article>"
    ),
    "formatting outside paragraphs": page(
        f"<div><p>{LONG}</p><b>bold words</b><span>span words {LONG}</span></div>"
    ),
    # 1.11.0 keeps control characters through its repair: lxml's ValueError on assigning them
    # can come from any step that copies text, and where it is caught decides the result
    "control character after </html>": (
        "<html><body>the<article><p>Trafilatura keeps the main text of a page and drops the "
        "navigation, the footer and the other boilerplate around it. This paragraph is long "
        "enough to count as content, and it mentions Café, naïve, Zürich and 日本語 so that the "
        "encodings stay honest.</p>"
        "<div>the<div></html></div><div></div>\x08</div></article></body></html>"
    ),
}

# the same characters in every place the extraction copies text from
BAD_CHARS = {"bs": "\x08", "ff": "\x0c", "nul": "\x00", "fffe": "\ufffe", "sur": "\udc80"}
BAD_PLACES = {
    "p text": lambda x: page(f"<article><p>{LONG}{x} tail</p><p>{LONG}</p></article>"),
    "between paragraphs": lambda x: page(f"<article><p>{LONG}</p>{x}<p>{LONG}</p></article>"),
    "span in p": lambda x: page(f"<article><p>{LONG} <span>a{x}b</span> end</p></article>"),
    "bold in p": lambda x: page(f"<article><p>{LONG} <b>bold {x}</b> end</p></article>"),
    "link in div": lambda x: page(f"<article><div><a href='/x'>link{x}</a> {LONG}</div></article>"),
    "list item": lambda x: page(
        f"<article><p>{LONG}</p><ul><li>item{x} one</li><li>two</li></ul></article>"
    ),
    "list item attribute": lambda x: page(
        f"<article><p>{LONG}</p><ul><li>Item <x data-y='a{x}b'>sub</x> text long enough</li>"
        "</ul></article>"
    ),
    "heading": lambda x: page(f"<article><h2>Title{x}</h2><p>{LONG}</p></article>"),
    "table cell": lambda x: page(
        f"<article><p>{LONG}</p><table><tr><td>cell{x}</td><td>b</td></tr></table></article>"
    ),
    "blockquote": lambda x: page(
        f"<article><p>{LONG}</p><blockquote>quote{x}</blockquote></article>"
    ),
    "pre": lambda x: page(f"<article><p>{LONG}</p><pre>code {x}</pre></article>"),
    "div text": lambda x: page(f"<article><div>{LONG}{x}</div></article>"),
    "tail of an empty span": lambda x: page(
        f"<article><p>{LONG}</p><span></span>{x}<p>{LONG}</p></article>"
    ),
    "tail of a removed nav": lambda x: page(
        f"<article><p>{LONG}</p><nav>menu</nav>{x} tail</article>"
    ),
    "after </html>": lambda x: page(f"<article><p>{LONG}</p></article>") + f"<div></div>{x}",
    "comments": lambda x: page(
        f"<article><p>{LONG}</p></article><div id='comments'><p>comment{x} here</p></div>"
    ),
    "title": lambda x: page(f"<article><p>{LONG}</p></article>", title=f"t{x}"),
    "class attribute": lambda x: page(f"<article class='a{x}b'><p>{LONG}</p></article>"),
    "body text": lambda x: page(f"{x}<article><p>{LONG}</p></article>"),
    "readability page": lambda x: page(
        "<div><a href='/'>Home</a> <a href='/b'>Blog</a></div>"
        + "".join(
            (
                f"<div class='para'>{LONG} sentence {i}, with commas, and more.</div>"
                f"{x if i == 2 else ''}"
            )
            for i in range(4)
        )
    ),
    "short page": lambda x: page(f"<div><p>Short {x} teaser.</p></div>"),
}
FAMILY = set()
# (three option sets each: the fixture stays small)
FAMILY_SETS = ("default", "precision", "fast")
for _place, _make in BAD_PLACES.items():
    for _cname, _char in BAD_CHARS.items():
        CASES_111[f"{_cname} in {_place}"] = _make(_char)
        FAMILY.add(f"{_cname} in {_place}")


def kwargs(kw, bare):
    kw = dict(kw)
    if V1:
        if "fast" in kw:
            kw["no_fallback"] = kw.pop("fast")
        if bare:
            kw["as_dict"] = False
    return kw


def raised(err):
    return {"raises": f"{type(err).__name__}: {err}" if V1 else type(err).__name__}


def tostr(e):
    return None if e is None else etree.tostring(e, encoding="unicode")


def main():
    out = []
    cases = {**CASES, **CASES_111} if V1 else CASES
    for name, html in cases.items():
        rec = {
            "name": name,
            "html": html.encode("utf-8", "surrogatepass").hex(),
            "extract": {},
            "bare": {},
        }
        for k, kw in OPTION_SETS.items():
            if name in FAMILY and k not in FAMILY_SETS:
                continue
            try:
                rec["extract"][k] = trafilatura.extract(html, **kwargs(kw, False))
            except Exception as err:
                rec["extract"][k] = raised(err)
            try:
                d = trafilatura.bare_extraction(html, **kwargs(kw, True))
                rec["bare"][k] = (
                    None
                    if d is None
                    else {
                        "text": d.text,
                        "raw_text": d.raw_text,
                        "comments": d.comments,
                        "body": tostr(d.body),
                        "commentsbody": tostr(d.commentsbody),
                    }
                )
            except Exception as err:
                rec["bare"][k] = raised(err)
        out.append(rec)
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=True, indent=0)
    nulls = sum(v is None for r in out for v in r["extract"].values())
    raises = sum(isinstance(v, dict) for r in out for v in r["extract"].values())
    print(
        f"trafilatura {trafilatura.__version__}: {len(out)} pages x {len(OPTION_SETS)} option "
        f"sets: {nulls} None, {raises} raised -> {sys.argv[1]}"
    )


if __name__ == "__main__":
    main()
