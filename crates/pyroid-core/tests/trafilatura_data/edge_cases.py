"""Hand-written pages for the rare paths of trafilatura 2.3.1 (tests/trafilatura_edge.rs).

    /target/venv-dt/bin/python -I edge_cases.py edge.json

Each page runs through trafilatura.extract and bare_extraction with several option sets; the
fixture keeps the pages (as surrogatepass UTF-8 in hex where they hold lone surrogates) and the
results: the text, None, or the name of the exception extract raises.
"""

import json
import sys

import trafilatura
from lxml import etree

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
    "control char joined by delete_element": page(f"<article><p>{LONG}</p><nav>menu</nav>&#1;tail {LONG}</article>"),
    "form feed in a block for baseline": page(f"<div><p>short</p><div>a&#12;b {LONG}</div></div>"),
    "unit separator for html2txt": page(f"<main><p>{LONG}</p></main><div><div>x&#31;y</div>{LONG * 6}</div>"),
    "colon tag in a list item": page(f"<article><p>{LONG}</p><ul><li><o:p>Item text one that is long enough</o:p></li><li>two</li></ul><p>{LONG}</p></article>"),
    "json-ld body with markup": page(
        "<div><p>Short teaser.</p></div>",
        ld({"@type": "NewsArticle", "articleBody": f"&lt;p&gt;{LONG}&lt;/p&gt;&lt;p&gt;Second &amp;amp; last {LONG}&lt;/p&gt;"}),
    ),
    "json-ld lone surrogate and entity": page(
        "<div><p>Short teaser.</p></div>",
        '<script type="application/ld+json">{"@type": "Article", "articleBody": "&am\\ud800p; '
        + LONG
        + ' \\ud83d\\ude00 end"}</script>',
    ),
    "json-ld graph, steps, answers, teaser": page(
        "<div><p>Short teaser.</p></div>",
        ld({"@graph": [
            {"@type": "Recipe", "recipeInstructions": [{"@type": "HowToStep", "text": LONG},
                                                       {"itemListElement": [{"text": "Stir well and serve."}]}, "Plain step."]},
            {"@type": "FAQPage", "mainEntity": [{"acceptedAnswer": {"text": LONG}}]},
            {"@type": ["Product", "Thing"], "description": "A product teaser " + LONG},
        ]}),
    ),
    "json-ld teaser only": page("<div>x</div>", ld({"@type": "VideoObject", "description": LONG * 2})),
    "json-ld deep nesting": page("<div>x</div>", '<script type="application/ld+json">' + "[" * 1200 + '{"articleBody": "x"}' + "]" * 1200 + "</script>"),
    "discourse preload": page(
        "<div id='data-preloaded' data-preloaded='"
        + json.dumps({"topic_1": json.dumps({"post_stream": {"posts": [{"cooked": f"<p>{LONG}</p>"}, {"cooked": "<p>Second post here.</p>"}]}})}).replace("'", "&#39;")
        + "'></div><div>x</div>"
    ),
    "discourse preload with a number for posts": page(
        "<div id='data-preloaded' data-preloaded='"
        + json.dumps({"topic_1": json.dumps({"post_stream": {"posts": 5}})})
        + "'></div><div>x</div>"
    ),
    "huge colspan": page(f"<article><p>{LONG}</p><table><tr><td colspan='{'9' * 5000}'>cell</td><td>b</td></tr></table></article>"),
    "huge char reference in text": page(f"<article><p>{LONG} &amp;#{'1' * 5000} end</p><p>{LONG}</p></article>"),
    "forum thread": page(
        f"<div class='post'><p>{LONG}</p></div><div id='comments'><div class='comment'><p>First reply {LONG}</p></div>"
        f"<div class='comment'><p>Second reply that says something else entirely.</p></div></div>",
        ld({"@type": "DiscussionForumPosting", "headline": "h"}),
    ),
    "comments section": page(
        f"<article><p>{LONG}</p><p>{LONG}</p></article><section id='comments'><p>Nice article, thanks for writing it.</p>"
        f"<p>I disagree with the second point.</p><div id='respond'>Leave a reply</div></section>"
    ),
    "readability wins": page(
        "<div><a href='/'>Home</a> <a href='/b'>Blog</a></div>"
        + "".join(f"<div class='para'>{LONG} sentence {i}, with commas, and more, words.</div>" for i in range(6))
    ),
    "video iframe": page(f"<div class='content'><p>{LONG}</p><iframe src='https://www.youtube.com/embed/x'></iframe><p>{LONG}</p></div>"),
    "tables": page(
        "<article><table><caption>Results <b>2026</b></caption><thead><tr><th>Name</th><th colspan='2'>Score</th></tr></thead>"
        "<tbody><tr><td rowspan='2'>Ann</td><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr>"
        "<tr><td>Bob</td><td><ul><li>x</li><li>y</li></ul></td><td><table><tr><td>inner</td></tr></table></td></tr></tbody></table>"
        f"<p>{LONG}</p></article>"
    ),
    "math, code, quotes": page(
        f"<article><p>{LONG}<math display='block'><annotation encoding='application/x-tex'>x^2</annotation></math> after</p>"
        "<pre>def f(x):\n    return {x}</pre><pre><span class='hljs-keyword'>if</span> x</pre><blockquote><p>Quoted words here.</p></blockquote>"
        f"<p>Inline <sup></sup><sub>2</sub> and <strong class='schema-faq-question'>Question?</strong></p><p>{LONG}</p></article>"
    ),
    "literal graphic and ref tags": page(
        f"<div><p>{LONG}</p></div><div class='x'><graphic src='//img.example/a.png' alt='Alt'>cap</graphic><ref target='u'>link [1]</ref> {LONG * 3}</div>"
    ),
    "lists and details": page(
        f"<article><dl><dt>Term</dt><dd>Definition {LONG}</dd></dl><ol><li>One<ul><li>Nested</li></ul></li><li>Two</li></ol>"
        f"<details><summary>More</summary><p>{LONG}</p></details></article>"
    ),
    "xml declaration": '<?xml version="1.0" encoding="utf-8"?>' + page(f"<article><p>{LONG}</p><p>{LONG}</p></article>"),
    "lone surrogate in the page": page(f"<article><p>{LONG} \udc80 tail</p><p>{LONG}</p></article>"),
    "deep nesting": page("<div>" * 300 + LONG + "</div>" * 300 + f"<p>{LONG}</p>"),
    "fragment, not a page": f"<p>{LONG}</p><p>{LONG}</p>",
    "no html at all": "just some text " * 20,
}


def main():
    out = []
    for name, html in CASES.items():
        rec = {"name": name, "html": html.encode("utf-8", "surrogatepass").hex(), "extract": {}, "bare": {}}
        for k, kw in OPTION_SETS.items():
            try:
                rec["extract"][k] = trafilatura.extract(html, **kw)
            except Exception as err:
                rec["extract"][k] = {"raises": type(err).__name__}
            try:
                d = trafilatura.bare_extraction(html, **kw)
                rec["bare"][k] = None if d is None else {
                    "text": d.text, "raw_text": d.raw_text, "comments": d.comments,
                    "body": etree.tostring(d.body, encoding="unicode"),
                    "commentsbody": etree.tostring(d.commentsbody, encoding="unicode"),
                }
            except Exception as err:
                rec["bare"][k] = {"raises": type(err).__name__}
        out.append(rec)
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=True, indent=0)
    nulls = sum(v is None for r in out for v in r["extract"].values())
    raises = sum(isinstance(v, dict) for r in out for v in r["extract"].values())
    print(f"{len(out)} pages x {len(OPTION_SETS)} option sets: {nulls} None, {raises} raised -> {sys.argv[1]}")


if __name__ == "__main__":
    main()
