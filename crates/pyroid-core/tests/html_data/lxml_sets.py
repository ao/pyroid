"""Reference data for `pyroid_core::html` from lxml 6.1.3 (the parity tests in tests/html_*.rs).

Run with the incumbent's interpreter (trafilatura 2.3.1, lxml 6.1.3), e.g. /target/venv-dt:

    python lxml_sets.py fuzz OUTDIR [--count N] [--seed S] [--mutate SETDIR]
        random HTML-like documents (or random edits of SETDIR's raw.bin pages) through
        trafilatura.utils.load_html, written as a sample set (raw.bin, repaired.bin, lxml.tree,
        meta.jsonl) for tests/html_lxml.rs
    python lxml_sets.py cases OUTDIR
        the hand-written cases below, as a sample set
    python lxml_sets.py fixture OUT.jsonl [--fuzz N]
        lxml.html.fromstring on str and on bytes for the cases plus N small fuzz documents:
        the committed fixture tests/html_data/fromstring.jsonl (tests/html_fromstring.rs)
    python lxml_sets.py large OUT.jsonl [--sweep]
        documents over 10 MB that reach libxml2's limits, described by their segments so that
        nothing large is stored, with the sha1 of lxml's tree (tests/html_large.rs); --sweep
        writes every length around each limit instead (about 570 documents, 5 minutes)
    python lxml_sets.py hashes SETDIR OUTDIR
        a set without lxml.tree (for sets like hold, where the trees would not fit the disk
        budget): OUTDIR/raw.bin links to SETDIR's, OUTDIR/meta.jsonl has the sha1 of each tree
    python lxml_sets.py repaired SETDIR OUT.bin
        the strings load_html passes to lxml (decode_file, repair_faulty_html) for SETDIR's
        raw.bin, for sets that have no repaired.bin
    python lxml_sets.py bench SETDIR [RUNS]
        time lxml.html.fromstring on SETDIR's repaired.bin, as tests/html_bench.rs does

Set formats: `.bin` is a u32 little-endian count, then per document a u32 little-endian
length and the surrogatepass UTF-8. `lxml.tree` has one JSON array per line, depth-first:
["D", index] starts a document (nothing follows when load_html returned None); ["S", tag]
opens an element, then ["A", name, value] per attribute, ["T", text] if its .text is not
empty, its children, and ["E"]; ["L", tail] follows an element whose .tail is not empty
(never the root); ["X", class, text] is a non-element node.

Derived from the P10 prototype's py/port_fuzz.py, port_cases.py, dump_lxml.py and
port_bench.py.
"""

# ruff: noqa: T201  (a recording script: it reports on stdout)

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import sys
import time
from pathlib import Path

from lxml import etree
from lxml.html import fromstring
from trafilatura import utils as tu

# ---- set I/O and dumps --------------------------------------------------------------------


def write_bin(path: Path, strings: list[str]) -> None:
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(strings)))
        for s in strings:
            b = s.encode("utf-8", "surrogatepass")
            f.write(struct.pack("<I", len(b)))
            f.write(b)


def read_bin(path: Path) -> list[str]:
    data = path.read_bytes()
    (n,) = struct.unpack_from("<I", data, 0)
    off, out = 4, []
    for _ in range(n):
        (ln,) = struct.unpack_from("<I", data, off)
        out.append(data[off + 4 : off + 4 + ln].decode("utf-8", "surrogatepass"))
        off += 4 + ln
    return out


def dump(el: etree._Element, out: list[str], root: bool = False) -> None:
    """The subtree in the tree format, iteratively (lxml trees can be deep)."""
    j = lambda *a: out.append(json.dumps(a, ensure_ascii=False))  # noqa: E731
    stack: list[tuple[etree._Element, bool]] = [(el, False)]
    while stack:
        node, closing = stack.pop()
        if closing:
            j("E")
            if node.tail and not (root and node is el):
                j("L", node.tail)
            continue
        if not isinstance(node.tag, str):
            j("X", type(node).__name__, node.text or "")
            if node.tail and not (root and node is el):
                j("L", node.tail)
            continue
        j("S", node.tag)
        for k, v in node.attrib.items():
            j("A", k, v)
        if node.text:
            j("T", node.text)
        stack.append((node, True))
        stack.extend((c, False) for c in reversed(node))


def load_html_traced(html: str) -> tuple[etree._Element | None, str, str]:
    """trafilatura.utils.load_html for a str, also returning the repaired string and the lxml
    call that built the tree ("str" or "bytes")."""
    s = tu.decode_file(html)
    beginning = s[:50].lower()
    check_flag = tu.is_dubious_html(beginning)
    s = tu.repair_faulty_html(s, beginning)
    path, tree, fallback = "str", None, False
    try:
        tree = tu.fromstring(s, parser=tu.HTML_PARSER)
    except ValueError:
        tree, path, fallback = tu.fromstring_bytes(s), "bytes", True
    except Exception:  # as load_html: logged, tree stays None
        pass
    if (tree is None or len(tree) < 1) and not fallback:
        tree, path = tu.fromstring_bytes(s), "bytes"
    if tree is not None and check_flag is True and len(tree) < 2:
        tree = None
    return tree, s, path


def write_set(out: Path, docs: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    raws, reps, lines, meta = [], [], [], []
    for i, html in enumerate(docs):
        tree, s, path = load_html_traced(html)
        ref = tu.load_html(html)
        same = (ref is None and tree is None) or (
            ref is not None and tree is not None and etree.tostring(ref) == etree.tostring(tree)
        )
        if not same:
            raise SystemExit(
                f"doc {i}: load_html_traced disagrees with trafilatura.utils.load_html"
            )
        raws.append(html)
        reps.append(s)
        lines.append(json.dumps(["D", i]))
        if tree is not None:
            dump(tree, lines, root=True)
        meta.append({"index": i, "path": path, "root": None if tree is None else tree.tag})
    write_bin(out / "raw.bin", raws)
    write_bin(out / "repaired.bin", reps)
    (out / "lxml.tree").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", errors="surrogatepass"
    )
    (out / "meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))
    paths = [m["path"] for m in meta]
    print(
        f"{len(docs)} docs -> {out}; str {paths.count('str')}, bytes {paths.count('bytes')}, "
        f"no tree {sum(m['root'] is None for m in meta)}"
    )


# ---- generator (the paths real pages rarely take) -----------------------------------------

TAGS = [
    "a",
    "abbr",
    "address",
    "area",
    "b",
    "base",
    "basefont",
    "big",
    "blockquote",
    "body",
    "br",
    "button",
    "caption",
    "center",
    "col",
    "colgroup",
    "dd",
    "del",
    "div",
    "dl",
    "dt",
    "em",
    "embed",
    "fieldset",
    "font",
    "form",
    "frame",
    "frameset",
    "h1",
    "h2",
    "h3",
    "head",
    "hr",
    "html",
    "i",
    "iframe",
    "img",
    "input",
    "ins",
    "isindex",
    "kbd",
    "label",
    "legend",
    "li",
    "link",
    "listing",
    "map",
    "menu",
    "meta",
    "noembed",
    "noframes",
    "noscript",
    "object",
    "ol",
    "optgroup",
    "option",
    "p",
    "param",
    "plaintext",
    "pre",
    "q",
    "s",
    "samp",
    "script",
    "select",
    "small",
    "span",
    "strike",
    "strong",
    "style",
    "sub",
    "sup",
    "table",
    "tbody",
    "td",
    "textarea",
    "tfoot",
    "th",
    "thead",
    "title",
    "tr",
    "tt",
    "u",
    "ul",
    "var",
    "xmp",
    "section",
    "article",
    "nav",
    "main",
    "header",
    "footer",
    "source",
    "wbr",
    "svg",
    "math",
    "template",
    "picture",
    "figure",
]
RAW = ["script", "style", "textarea", "title", "xmp", "iframe", "noembed", "noframes"]
ENTS = [
    "&amp;",
    "&amp",
    "&ampx",
    "&lt",
    "&gt;",
    "&nbsp;",
    "&nbsp",
    "&copy2020",
    "&notin;",
    "&notit;",
    "&not",
    "&NotEqualTilde;",
    "&CounterClockwiseContourIntegral;",
    "&quot",
    "&apos;",
    "&foo;",
    "&",
    "&#",
    "&#x",
    "&#x;",
    "&#65;",
    "&#65",
    "&#x41;",
    "&#X41",
    "&#0;",
    "&#128;",
    "&#x80;",
    "&#x9f;",
    "&#xD800;",
    "&#55357;&#56832;",
    "&#x110000;",
    "&#99999999999;",
    "&#1;",
    "&#x0B;",
    "&#13;",
    "&#x1F600;",
    "&eacute",
    "&Eacute;",
    "&amp;amp;",
    "&lt;script&gt;",
    "&AMP;",
    "&Amp;",
    "&ngE;",
    "&nGt;",
    "&bne;",
    "&fjlig;",
    "&#xFFFE;",
    "&#xfffF;",
]
WORDS = [
    "hello",
    "world",
    "x",
    "Grüße",
    "café",
    "中文",
    "ال",
    "\U0001f600",
    " ",
    "  ",
    "\t",
    "\n",
    "\r\n",
    "\r",
    " \r ",
    " ",  # noqa: RUF001  (a no-break space on purpose)
    "​",
    "\x7f",
    "\x85",
    "\x9f",
    "﻿",
    "﷐",
    "\U0010fffd",
    "-",
    "--",
    "->",
    "<",
    "< b",
    "<3",
    "<>",
    "a<b",
    "a&b",
    ">",
    '"',
    "'",
    "=",
    "/",
    "\x01",
    "\x0b",
    "\x1f",
]
SPACE = [" ", "  ", "\n", "\t", "\r\n", "\r", " ", "　", ""]  # noqa: RUF001  (a no-break space on purpose)


class Gen:
    def __init__(self, rnd: random.Random) -> None:
        self.r = rnd

    def pick(self, xs):
        return self.r.choice(xs)

    def name(self) -> str:
        r = self.r.random()
        if r < 0.75:
            return self.pick(TAGS)
        if r < 0.85:
            return self.pick(TAGS).upper()
        if r < 0.9:
            return "x" * self.r.randint(95, 130)
        if r < 0.95:
            return self.pick(["fb:like", "my-elem", "h7", "dïv", "a1", "o:p", "foo<bar", "td\x00"])
        return self.pick(TAGS) + self.pick(["", "/", "é"])

    def attr_value(self) -> str:
        v = "".join(self.pick(WORDS + ENTS) for _ in range(self.r.randint(0, 3)))
        q = self.r.random()
        if q < 0.4:
            return '"' + v.replace('"', "") + '"'
        if q < 0.7:
            return "'" + v.replace("'", "") + "'"
        if q < 0.8:
            return '"' + v  # unterminated
        return v.replace(" ", "").replace(">", "")

    def attrs(self) -> str:
        out = []
        for _ in range(self.r.choice([0, 0, 1, 1, 2, 3, 5, 20])):
            n = self.pick(
                [
                    "class",
                    "id",
                    "href",
                    "CLASS",
                    "checked",
                    "selected",
                    "data-x",
                    "x",
                    "=x",
                    '"a',
                    "on",
                    "disabled",
                    "nowrap",
                    "a" * 120,
                    "été",
                    "xml:lang",
                    "/",
                    "lang",
                ]
            )
            if self.r.random() < 0.7:
                out.append(n + self.pick(["=", " = ", "= ", "=\n"]) + self.attr_value())
            else:
                out.append(n)
        sep = self.pick([" ", "  ", "\n", "/", " / ", "\t"])
        return "".join(sep + a for a in out)

    def start(self) -> str:
        return "<" + self.name() + self.attrs() + self.pick([">", ">", ">", "/>", " />", " >", ""])

    def end(self) -> str:
        if self.r.random() < 0.85:
            return "</" + self.name() + self.pick([">", " >", " x=1>", "/>", ""])
        return self.pick(["</>", "</ x>", "</3>", "</", "</!x>", "</-->"])

    def text(self) -> str:
        return "".join(self.pick(WORDS + ENTS + SPACE) for _ in range(self.r.randint(1, 6)))

    def misc(self) -> str:
        return self.pick(
            [
                "<!-- c -->",
                "<!---->",
                "<!-->",
                "<!--->",
                "<!-- a --!> b",
                "<!-- x",
                "<!x>",
                "<?php echo 1 ?>",
                "<!DOCTYPE html>",
                '<!doctype HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "x">',
                "<![CDATA[x]]>",
                "<!-- <script> -->",
                "<!--x--y-->",
                "<!DOCTYPE html SYSTEM 'a'>",
                "<!DOCTYPE x PUBLIC>",
            ]
        )

    def raw(self) -> str:
        t = self.pick(RAW)
        body = "".join(
            self.pick(
                [
                    "var a = 1 < 2;",
                    "<!--",
                    "-->",
                    "<script>",
                    "</script >",
                    "</scriptx>",
                    "</SCRIPT>",
                    f"</{t}",
                    "&amp;",
                    "<b>",
                    "</b>",
                    "\r\n",
                    "x",
                    "--",
                    "<!-- <script> </script> -->",
                    f"</{t.upper()}>",
                    "\x00",
                ]
            )
            for _ in range(self.r.randint(0, 6))
        )
        close = self.pick([f"</{t}>", f"</{t}>", f"</{t.upper()} >", f"</{t}/>", ""])
        return f"<{t}{self.attrs()}>" + body + close

    def nest(self) -> str:
        t = self.pick(["div", "font", "b", "span", "p", "td", "li", "table"])
        n = self.r.choice([250, 254, 255, 256, 257, 300])
        return f"<{t}>" * n + "deep" + f"</{t}>" * self.r.randint(0, n)

    def content(self, n: int) -> str:
        out = []
        for _ in range(n):
            r = self.r.random()
            if r < 0.35:
                out.append(self.start())
            elif r < 0.55:
                out.append(self.end())
            elif r < 0.8:
                out.append(self.text())
            elif r < 0.88:
                out.append(self.misc())
            elif r < 0.97:
                out.append(self.raw())
            else:
                out.append(
                    self.pick(
                        [
                            "<html>",
                            "</html>",
                            "<body>",
                            "</body>",
                            "<head>",
                            "</head>",
                            "<html/>",
                            "<body/>",
                            "<head/>",
                            "<html lang=x>",
                            "<body class=y>",
                        ]
                    )
                )
        return "".join(out)

    def doc(self) -> str:
        r = self.r.random()
        prefix = ""
        if r < 0.1:
            prefix = '<?xml version="1.0" encoding="utf-8"?>'
        elif r < 0.15:
            prefix = "<?xml version='1.0'?>"
        elif r < 0.2:
            prefix = "﻿"
        elif r < 0.5:
            prefix = self.pick(
                [
                    "<!DOCTYPE html>",
                    "<!doctype html>\n",
                    ' \n<!DOCTYPE html PUBLIC "x" "y">',
                    "<!DOCTYPE html/>\n",
                    "< ! DOCTYPE html x/y>",
                ]
            )
        shape = self.r.random()
        if shape < 0.4:
            body = (
                self.pick(["<html>", "<HTML lang=en>", "<html>\n", "<html/>\n"])
                + self.pick(["", "<head>", "<head><title>t</title>"])
                + self.content(self.r.randint(1, 40))
                + self.pick(
                    ["", "</html>", "</body></html>", "</html>\n<script>x</script>", "</html>tail"]
                )
            )
        elif shape < 0.7:
            body = self.content(self.r.randint(1, 40))
        elif shape < 0.8:
            body = "<html><body>" + self.nest() + self.content(5)
        else:
            body = self.pick(["<div>", "<p>", "<span>", "", "<b>"]) + self.content(
                self.r.randint(1, 12)
            )
        s = prefix + self.pick(SPACE) + body
        q = self.r.random()
        if q < 0.05:
            i = self.r.randint(0, len(s))
            s = s[:i] + self.pick(["\ud800", "\udc00", "\U0001f600", "\ud83d", "😀"]) + s[i:]
            if self.r.random() < 0.5:
                s += "\U0001f600"
        elif q < 0.1:
            s = s.encode("latin-1", "ignore").decode("latin-1")
        elif q < 0.15:
            s = "".join(c for c in s if ord(c) < 128)
        if self.r.random() < 0.03:
            s = s * 3
        return s


def mutate(g: Gen, s: str) -> str:
    """Random edits of a real page: cut, duplicate or move spans, insert generated markup,
    truncate, or splice in surrogates and astral characters around the 4000-byte reads."""
    r = g.r
    for _ in range(r.randint(1, 4)):
        n = len(s)
        if n == 0:
            break
        i, j = sorted((r.randrange(n + 1), r.randrange(n + 1)))
        op = r.random()
        if op < 0.2:
            s = s[:i] + s[j:]
        elif op < 0.35:
            s = s[:j] + s[i:j] + s[j:]
        elif op < 0.6:
            s = s[:i] + g.content(r.randint(1, 6)) + s[i:]
        elif op < 0.7:
            s = s[: max(j, 1)]
        elif op < 0.8:
            k = r.choice(
                [1990, 1995, 1998, 1999, 2000, 2001, 997, 998, 999, 1000, 1001, 3999, 4000]
            ) + r.randint(-3, 3)
            s = s[:k] + r.choice(["\ud800", "\udc00", "\U0001f600", "\U0001f600", "\ud83d"]) + s[k:]
        elif op < 0.85:
            s = "﻿" + s
        elif op < 0.9:
            s = s.replace("\n", "\r\n") if r.random() < 0.5 else s.replace("\n", "\r")
        else:
            s = s[:i] + g.nest() + s[i:]
    return s


CASES = [
    "<html><body><p>a<p>b</body></html>",
    "<p>one</p>",
    "<p>one</p><p>two</p>",
    "<b>x</b> tail",
    "text only, no html word here",
    "<html><head><title>t</title></head><body>x</body></html> after",
    "<html><body>x</body></html><p>after root</p>",
    "<body>a</body><body>b<i>c</i></body>",
    "<head><meta charset=x></head><head><link></head><p>z",
    "<table><tr><td>1<td>2<tr><td>3</table>",
    "<ul><li>a<li>b</ul><dl><dt>x<dd>y<dt>z</dl>",
    "<select><option>a<option>b<optgroup><option>c</select>",
    "<div><p>a<div>b</div>c</p></div>",
    "<script>if (a<b && c>d) x='</scr'+'ipt>';</script><p>after",
    "<script><!--<script>x</script>y</script>z-->q</script>w",
    "<textarea>&amp; <b>not a tag</b></textarea><title>&lt;t&gt;</title>",
    "<plaintext><p>all text</p></plaintext>",
    "&#0;&#x80;&#x9F;&#xD800;&#x110000;&#1;&#xFFFE;&notit;&notin;&amp&ampx",
    "<a href='x&copy=1&amp;y=2&nbsp' title=&lt;>z</a>",
    "<p>a\r\nb\rc</p>",
    "<div " + "x" * 120 + "=1>long attribute name</div>",
    "<" + "y" * 130 + ">long tag name</" + "y" * 130 + ">",
    "<html/><body/>x",
    "<div><body/>y</div>z",
    "﻿<html><body>bom</body></html>",
    "﻿\ud800<html>",
    '<?xml version="1.0" encoding="utf-8"?><html><body>xml decl</body></html>',
    '<?xml version="1.0" encoding="utf-8"?><p>a&#1;</p><body>b</body><body>c</body>',
    "<p>a&#11;b</p><body>x</body>",
    "<div>" * 300 + "deep",
    "<html><body>" + "<font>" * 260 + "x",
    "",
    "   ",
    "\n\t",
    "<!-- only a comment -->",
    "<!DOCTYPE html>",
    "<html>",
    "</p>",
    "<p>a</p><body>&#1;x</body><body>y</body>",
    "<body>a</body>&#xFFFE;<body>b</body>",
    "<head></head><body><p>x</body>tail<body>t2</body>",
    "<p>x</p>😀<p>pair</p>",
    "<p>" + "€" * 1999 + "\ud83d<p>split pair",
]


def cmd_fuzz(a) -> None:
    g = Gen(random.Random(a.seed))
    base = read_bin(Path(a.mutate) / "raw.bin") if a.mutate else None
    docs = [mutate(g, g.pick(base)) if base else g.doc() for _ in range(a.count)]
    write_set(Path(a.outdir), docs)


def cmd_cases(a) -> None:
    write_set(Path(a.outdir), CASES)


def fromstring_result(x: str | bytes) -> dict:
    try:
        el = fromstring(x, parser=tu.HTML_PARSER)
    except Exception as e:  # the exception class is the result
        return {"error": type(e).__name__, "message": str(e)}
    lines: list[str] = []
    dump(el, lines, root=True)
    return {"tree": "\n".join(lines) + "\n"}


def text_or_hex(s: str) -> dict:
    try:
        s.encode("utf-8")
        return {"text": s}
    except UnicodeEncodeError:
        return {"hex": s.encode("utf-8", "surrogatepass").hex()}


def cmd_fixture(a) -> None:
    g = Gen(random.Random(a.seed))
    docs = list(CASES)
    while len(docs) < len(CASES) + a.fuzz:
        d = g.doc()
        if len(d) <= 600:
            docs.append(d)
    with open(a.out, "w", encoding="utf-8") as f:
        for d in docs:
            rec = {
                "input": text_or_hex(d),
                "str": fromstring_result(d),
                "bytes": fromstring_result(d.encode("utf-8", "surrogatepass")),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{len(docs)} documents -> {a.out}")


# ---- documents over 10 MB ------------------------------------------------------------------

M = 10_000_000


def large_specs() -> list[tuple[str, list[tuple[str, int]]]]:
    """(label, segments): the document is the concatenation of segment * count."""
    near = [-4096, -300, -64, 0, 64, 300, 4096]
    s: list[tuple[str, list[tuple[str, int]]]] = [
        (f"text {d:+}", [("<html><body><p>", 1), ("a", M + d), ("</p><p>after</p>", 1)])
        for d in [*near, -20_000, 20_000]
    ]
    for d in near:
        s.append((f"attr dq {d:+}", [('<div title="', 1), ("v", M + d), ('">x</div><p>after', 1)]))
        s.append(
            (
                f"attr unquoted {d:+}",
                [("<div title=", 1), ("v", M + d), (" id=a>x</div><p>after", 1)],
            )
        )
        s.append((f"comment {d:+}", [("<p>a</p><!--", 1), ("c", M + d), ("--><p>after", 1)]))
        s.append((f"script {d:+}", [("<script>", 1), ("x", M + d), ("</script><p>after", 1)]))
        s.append((f"blanks in tag {d:+}", [("<p>a</p><div", 1), (" ", M + d), ("id=x>y</div>", 1)]))
        s.append((f"leading blanks {d:+}", [(" ", M + d), ("<p>x", 1)]))
        s.append(
            (f"bogus doctype {d:+}", [("<!DOCTYPE html x", 1), ("y", M + d), ("><p>after", 1)])
        )
    for d in (-4096, 0, 4096, 40_000):
        # Text with line feeds and multi-byte characters: libxml2's accelerator stops at both.
        s.append(
            (f"text lf {d:+}", [("<p>", 1), ("abcdefghi\n", (M + d) // 10), ("</p><p>after", 1)])
        )
        s.append((f"text latin1 {d:+}", [("<p>", 1), ("é", (M + d) // 2), ("</p><p>after", 1)]))
        s.append((f"text ucs2 {d:+}", [("<p>", 1), ("€", (M + d) // 3), ("</p><p>after", 1)]))
        s.append(
            (f"text ucs4 {d:+}", [("<p>", 1), ("\U0001f600", (M + d) // 4), ("</p><p>after", 1)])
        )
        s.append(
            (f"attr ucs2 {d:+}", [("<a href='", 1), ("€", (M + d) // 3), ("'>x</a><p>after", 1)])
        )
    # Many tokens, none large: no limit is reached.
    s.append(("small tokens", [("<p class=x>abc &amp; def</p>\n", 450_000)]))
    s.append(("small tokens ucs2", [("<p>€&amp;</p>", 1_100_000)]))
    # Text node over 10 MB built from many chunks: "Text node too long".
    s.append(
        ("text node limit", [("<p>", 1), ("a" * 999 + "&amp;", 10_050), ("</p><p>after</p>", 1)])
    )
    # Attribute value over 10 MB in many chunks: "value too long", the parse goes on.
    s.append(
        (
            "attr value too long",
            [("<div title='", 1), ("v" * 999 + "&amp;", 10_050), ("'>x</div><p>after", 1)],
        )
    )
    # A lone surrogate past the first read in a large str, and right before the limit.
    s.append(
        ("surrogate early", [("<p>", 1), ("a", 5000), ("\ud800", 1), ("b", M + 1000), ("</p>", 1)])
    )
    s.append(("surrogate late", [("<p>", 1), ("€", M // 3 + 100), ("\udc00", 1), ("x</p>", 1)]))
    s.append(
        ("surrogate past limit", [("<p>", 1), ("a", M + 50_000), ("\ud800", 1), ("</p><p>z", 1)])
    )
    # A deep then wide document with the limit reached in a later text run.
    s.append(
        (
            "after many elements",
            [("<div><span>x</span></div>", 200_000), ("<p>", 1), ("t", M), ("</p>z", 1)],
        )
    )
    return s


def sweep_specs() -> list[tuple[str, list[tuple[str, int]]]]:
    """Every length around the point where each token kind starts to halt the parse."""
    s: list[tuple[str, list[tuple[str, int]]]] = []
    for d in range(-72, 9):
        s.append((f"text {d:+}", [("<html><body><p>", 1), ("a", M + d), ("</p><p>after</p>", 1)]))
        s.append((f"script {d:+}", [("<script>", 1), ("x", M + d), ("</script><p>after", 1)]))
    for d in range(-8, 73):
        s.append((f"attr dq {d:+}", [('<div title="', 1), ("v", M + d), ('">x</div><p>after', 1)]))
        s.append((f"comment {d:+}", [("<p>a</p><!--", 1), ("c", M + d), ("--><p>after", 1)]))
    s.extend((f"leading blanks {d:+}", [(" ", M + d), ("<p>x", 1)]) for d in range(-64, 65, 2))
    for d in range(256, 4160, 64):
        s.append((f"blanks in tag {d:+}", [("<p>a</p><div", 1), (" ", M + d), ("id=x>y</div>", 1)]))
        s.append(
            (f"bogus doctype {d:+}", [("<!DOCTYPE html x", 1), ("y", M + d), ("><p>after", 1)])
        )
    s.extend(
        (f"text ucs2 {d:+}", [("<p>", 1), ("€", (M + d) // 3), ("</p><p>after", 1)])
        for d in range(-4200, 200, 100)
    )
    return s


def cmd_large(a) -> None:
    with open(a.out, "w", encoding="utf-8") as f:
        for label, segs in sweep_specs() if a.sweep else large_specs():
            doc = "".join(seg * n for seg, n in segs)
            rec: dict = {"label": label, "segments": [[text_or_hex(seg), n] for seg, n in segs]}
            for key, x in (("str", doc), ("bytes", doc.encode("utf-8", "surrogatepass"))):
                r = fromstring_result(x)
                if "tree" in r:
                    t = r.pop("tree").encode("utf-8", "surrogatepass")
                    r = {"sha1": hashlib.sha1(t).hexdigest(), "len": len(t)}
                rec[key] = r
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(label, rec["str"], rec["bytes"], flush=True)


def cmd_hashes(a) -> None:
    """A set without lxml.tree: OUTDIR/raw.bin links to SETDIR's, and meta.jsonl holds per
    document the sha1 of its tree's lines (null when load_html returns None)."""
    src = Path(a.setdir)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    link = out / "raw.bin"
    if not link.exists():
        link.symlink_to((src / "raw.bin").resolve())
    with (out / "meta.jsonl").open("w") as f:
        for i, html in enumerate(read_bin(src / "raw.bin")):
            tree, _, path = load_html_traced(html)
            digest = None
            if tree is not None:
                lines: list[str] = []
                dump(tree, lines, root=True)
                digest = hashlib.sha1(
                    ("\n".join(lines) + "\n").encode("utf-8", "surrogatepass")
                ).hexdigest()
            f.write(json.dumps({"index": i, "path": path, "sha1": digest}) + "\n")
    print(f"hashes -> {out}")


def cmd_repaired(a) -> None:
    out = []
    for html in read_bin(Path(a.setdir) / "raw.bin"):
        s = tu.decode_file(html)
        out.append(tu.repair_faulty_html(s, s[:50].lower()))
    write_bin(Path(a.out), out)
    print(f"{len(out)} repaired strings -> {a.out}")


def cmd_bench(a) -> None:
    d = Path(a.setdir)
    docs = read_bin(d / "repaired.bin")
    nbytes = sum(len(s.encode("utf-8", "surrogatepass")) for s in docs)
    per_doc = [float("inf")] * len(docs)
    best = float("inf")
    clock = time.perf_counter
    for r in range(a.runs):
        t0 = clock()
        for i, s in enumerate(docs):
            t = clock()
            # As load_html: some documents raise. A plain try rather than contextlib.suppress,
            # which would add its own cost to the timed call.
            try:  # noqa: SIM105
                fromstring(s, parser=tu.HTML_PARSER)
            except Exception:
                pass
            per_doc[i] = min(per_doc[i], clock() - t)
        total = clock() - t0
        best = min(best, total)
        print(f"run {r}: {total:.3f} s total, {1e3 * total / len(docs):.3f} ms/doc")
    print(
        f"lxml: {len(docs)} docs, {nbytes / 1e6:.1f} MB; best run {best:.3f} s = "
        f"{1e3 * best / len(docs):.3f} ms/doc; "
        f"sum of per-doc minima {1e3 * sum(per_doc) / len(docs):.3f} ms/doc"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fuzz")
    p.add_argument("outdir")
    p.add_argument("--count", type=int, default=20000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--mutate", help="SETDIR whose raw.bin documents are mutated instead of generating"
    )
    p.set_defaults(f=cmd_fuzz)
    p = sub.add_parser("cases")
    p.add_argument("outdir")
    p.set_defaults(f=cmd_cases)
    p = sub.add_parser("fixture")
    p.add_argument("out")
    p.add_argument("--fuzz", type=int, default=150)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(f=cmd_fixture)
    p = sub.add_parser("large")
    p.add_argument("out")
    p.add_argument(
        "--sweep", action="store_true", help="every length around each threshold instead"
    )
    p.set_defaults(f=cmd_large)
    p = sub.add_parser("hashes")
    p.add_argument("setdir")
    p.add_argument("outdir")
    p.set_defaults(f=cmd_hashes)
    p = sub.add_parser("repaired")
    p.add_argument("setdir")
    p.add_argument("out")
    p.set_defaults(f=cmd_repaired)
    p = sub.add_parser("bench")
    p.add_argument("setdir")
    p.add_argument("runs", type=int, nargs="?", default=5)
    p.set_defaults(f=cmd_bench)
    a = ap.parse_args()
    sys.setrecursionlimit(10_000)
    a.f(a)


if __name__ == "__main__":
    main()
