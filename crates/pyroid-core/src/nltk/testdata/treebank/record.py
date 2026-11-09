"""Record NLTK 3.10.4's word tokenizers and detokenizer.

pyroid's `nltk::treebank` and `nltk::regexp` replay the recordings
(`replay.rs` next to this file). Run in the dev container, never on the
host (`-B`: no `__pycache__` in the source tree):

    cd /work/crates/pyroid-core/src/nltk/testdata/treebank
    export NLTK_DATA=/target/nltk_data
    /target/venv-313/bin/python -B record.py golden .
    /target/venv-313/bin/python -B record.py regexp .
    /target/venv-313/bin/python -B record.py large DIR
    /target/venv-313/bin/python -B record.py codepoints DIR
    /target/venv-313/bin/python -B record.py codepoints DIR --regexp
    /target/venv-312/bin/python -B record.py codepoints DIR --re-only

`golden` writes `golden.jsonl` and `regexp` writes `golden-regexp.jsonl`
(both committed). `large` writes every sentence of the first 5,000 FineWeb
documents and of up to 1,000 documents of each FineWeb-2 file (plus some
whole documents). `codepoints` puts every code point except the surrogates
into the TEMPLATES; `--re-only` keeps the templates whose result depends on
`re`'s `\\w` and `(?i)`, which change with CPython's Unicode version, and
records only the tokenizers; `--regexp` records REGEXP_TEMPLATES through
the regexp tokenizers. `--template T` (repeatable) replaces the list and
`--suffix=S` ends the file name.

Every file starts with a header line: {"kind": "rows" | "codepoints" |
"regexp" | "regexp-codepoints", "python", "unidata", "nltk", "regex"}. Then:

* "rows": one object per text "t". "tb"/"nl": the TreebankWordTokenizer /
  NLTKWordTokenizer tokens joined by " "; "tbp"/"nlp": the same with
  convert_parentheses=True (absent when equal to "tb"/"nl"); "tbs"/"nls":
  span_tokenize as a flat [start, end, ...] list, or {"e": exception type,
  "m": message}; "dnl": detokenize(nl tokens); "dtb": detokenize(tb tokens)
  (absent when equal to "dnl"); "dnlp": detokenize(nlp tokens,
  convert_parentheses=True) (absent when equal to "dnl"). Optional "x": edited
  token lists, with "dx" (detokenize), "dxp" (with convert_parentheses=True,
  absent when equal to "dx") and "ax" (align_tokens(x, t), spans or error).
* "codepoints": one object per template "tpl" and result "r" (the fields
  above, as a dict), with "cps": the [first, last] code point ranges giving
  that result. In "r" every occurrence of the code point is written as
  U+0000. "d1" is detokenize([text]).
* "regexp": one object per text "t" with "f" (per REGEXP_PATTERNS:
  [tokenize, span_tokenize] with gaps=False), "g" (per pattern, with
  gaps=True: tokenize and span_tokenize with discard_empty=True, then
  False), "named" (wordpunct_tokenize, WordPunctTokenizer spans,
  WhitespaceTokenizer, blankline_tokenize, BlanklineTokenizer,
  SpaceTokenizer, TabTokenizer: tokens, then spans) and "lines" (per
  LINE_MODES: LineTokenizer tokens and spans). Spans are flat lists.
* "regexp-codepoints": as "codepoints", with the "regexp" fields in "r".
"""

import argparse
import json
import multiprocessing
import random
import sys
import unicodedata
from pathlib import Path

import nltk
import regex
from nltk.tokenize import (
    BlanklineTokenizer,
    LineTokenizer,
    NLTKWordTokenizer,
    RegexpTokenizer,
    SpaceTokenizer,
    TabTokenizer,
    TreebankWordTokenizer,
    WhitespaceTokenizer,
    WordPunctTokenizer,
    blankline_tokenize,
    sent_tokenize,
    wordpunct_tokenize,
)
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize.util import align_tokens

# align_tokens of the edited lists is recorded for texts up to this length
# (its error message repeats the text).
AX_MAX = 48

TB = TreebankWordTokenizer()
NL = NLTKWordTokenizer()
DT = TreebankWordDetokenizer()

FINEWEB = "/data/fineweb/fineweb-dev-10k.parquet"
FINEWEB2 = sorted(Path("/data/fineweb2").glob("*.parquet")) + sorted(
    Path("/target/fw2").glob("*.parquet")
)
# Punkt models for the FineWeb-2 languages (others are split as English).
PUNKT = {
    "ces": "czech",
    "dan": "danish",
    "deu": "german",
    "ekk": "estonian",
    "ell": "greek",
    "fin": "finnish",
    "fra": "french",
    "ita": "italian",
    "mal": "malayalam",
    "nld": "dutch",
    "nob": "norwegian",
    "pol": "polish",
    "por": "portuguese",
    "rus": "russian",
    "slv": "slovene",
    "spa": "spanish",
    "swe": "swedish",
    "tur": "turkish",
}


def header(kind):
    return {
        "kind": kind,
        "python": sys.version.split()[0],
        "unidata": unicodedata.unidata_version,
        "nltk": nltk.__version__,
        "regex": regex.__version__,
    }


def outcome(f):
    """A flat span list, or the exception NLTK raised."""
    try:
        return [x for span in f() for x in span]
    except Exception as e:  # noqa: BLE001 - the exception kind is the result
        return {"e": type(e).__name__, "m": str(e)}


def tokenize_fields(text):
    r = {}
    tb = TB.tokenize(text)
    nl = NL.tokenize(text)
    r["tb"] = " ".join(tb)
    tbp = " ".join(TB.tokenize(text, convert_parentheses=True))
    if tbp != r["tb"]:
        r["tbp"] = tbp
    r["nl"] = " ".join(nl)
    nlp = NL.tokenize(text, convert_parentheses=True)
    if " ".join(nlp) != r["nl"]:
        r["nlp"] = " ".join(nlp)
    return r, tb, nl, nlp


def row(text, edits=()):
    r = {"t": text}
    fields, tb, nl, nlp = tokenize_fields(text)
    r.update(fields)
    r["tbs"] = outcome(lambda: TB.span_tokenize(text))
    r["nls"] = outcome(lambda: NL.span_tokenize(text))
    r["dnl"] = DT.detokenize(nl)
    dtb = DT.detokenize(tb)
    if dtb != r["dnl"]:
        r["dtb"] = dtb
    dnlp = DT.detokenize(nlp, convert_parentheses=True)
    if dnlp != r["dnl"]:
        r["dnlp"] = dnlp
    if edits:
        edits = [list(x) for x in edits]
        r["x"] = edits
        r["dx"] = [DT.detokenize(x) for x in edits]
        dxp = [DT.detokenize(x, convert_parentheses=True) for x in edits]
        if dxp != r["dx"]:
            r["dxp"] = dxp
        if len(text) <= AX_MAX:
            r["ax"] = [outcome(lambda x=x: align_tokens(x, text)) for x in edits]
    return r


# ---------------------------------------------------------------------------
# Inputs

# Pieces for edited token lists and for the fuzz texts.
TOKEN_PIECES = [
    '"', "''", "``", "`", "'", "'''", "(", ")", "[", "]", "{", "}", "<", ">",
    "-LRB-", "-RRB-", "-LSB-", "-RSB-", "-LCB-", "-RCB-", "--", "-", "...", "..",
    ".", ",", ":", ";", "!", "?", "#", "$", "%", "&", "@", "*", "n't", "N'T",
    "'s", "'S", "'m", "'M", "'d", "'D", "'ll", "'LL", "'re", "'RE", "'ve", "'VE",
    "can", "not", "Can", "NOT", "d", "'ye", "gim", "me", "gon", "na", "got", "ta",
    "lem", "more", "'n", "wan", "'t", "is", "was", "'T", "IS", "WAS", "a", "b",
    "x", "1", "2", "", " ", "\t", "\n", "\x1c", "\x85", "\xa0", "\u3000", "a b",
    "a\tb", "\u00ab", "\u00bb", "\u201c", "\u201d", "\u2018", "\u2019", "\u201e",
    "\u2013", "\u2014", "\u017f", "\u0130", "\u0131", "\u0301", "\u00b2",
    "\u0663", "\u00e9", ".'", '."', ",'", ',"', "'.", "$5", "#1",
]

TEXT_PIECES = TOKEN_PIECES + [
    "cannot", "CanNot", "d'ye", "gimme", "gonna", "gotta", "lemme", "more'n",
    "wanna", "'tis", "'Twas", "'TIS", "don't", "DON'T", "can't", "I'm", "it's",
    "they'll", "we've", "you're", "he'd", "O'Neil", "'90s", "rock'n'roll",
    "3:00", "1,000", "3.88", "e.g.", "U.S.", "Dr.", "a.", "b?", "c!", " ",
    "  ", "\r\n", "\x0b", "\x0c", "\x1d", "\x1e", "\x1f", "\u2028", "\u2029",
    "\u200b", "\u2012", "\u2015", "word", "Word", "WORD", "_", "\u00a0",
    "g\u0130mme", "g\u0131mme", "'t\u0130s", "'ti\u017f", "\u017f", "'\u017f",
    "a\u0301", "\u0301'", "'\u0301", "x\u00b2", "\u00b2x", "\u0660", "\U0001f600",
]

NLTK_SENTENCES = [
    "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks.",
    "They'll save and invest more.",
    "hi, my name can't hello,",
    "Good muffins cost $3.88 (roughly 3,36 euros)\nin New York.  Please buy me\ntwo of them.\nThanks.",
    "Good muffins cost $3.88\nin New (York).  Please (buy) me\ntwo of them.\n(Thanks).",
    "The DUP is similar to the \"religious right\" in the United States and takes a hardline stance on social issues",
    "The DUP is similar to the \"religious right\" in the United States and takes a ``hardline'' stance on social issues",
    "I said, \"I'd like to buy some ''good muffins\" which cost $3.88\n each in New (York).\"",
    "On a $50,000 mortgage of 30 years at 8 percent, the monthly payment would be $366.88.",
    "\"We beat some pretty good teams to get here,\" Slocum said.",
    "Well, we couldn't have this predictable, cliche-ridden, \"Touched by an Angel\" (a show creator John Masius worked on) wanna-be if she didn't.",
    "I cannot cannot work under these conditions!",
    "The company spent $30,000,000 last year.",
    "The company spent 40.75% of its income last year.",
    "He arrived at 3:00 pm.",
    "I bought these items: books, pencils, and pens.",
    "Though there were 150, 100 of them were old.",
    "There were 300,000, but that wasn't enough.",
    "It's more'n enough.",
    "\u00abNow that I can do.\u00bb",
    "The unicode 201C and 201D \u201cLEFT(RIGHT) DOUBLE QUOTATION MARK\u201d is also OPEN_PUNCT and CLOSE_PUNCT.",
    "How \"are\" you?",
    "Hello (world)",
    "<A sentence> with (many) [kinds] of {parentheses}. \"Sometimes it's inside (quotes)\". (\"Sometimes the otherway around\").",
    "Sentence ending with (parentheses)",
    "(Sentence) starting with parentheses.",
    "I've",
    "Don't",
    "I'd",
    "I called Dr. Jones. I called Dr. Jones.",
    "Ich muss unbedingt daran denken, Mehl, usw. fur einen Kuchen einzukaufen. Ich muss.",
    "The plane, bound for St Petersburg, crashed in Egypt's Sinai desert just 23 minutes after take-off from Sharm el-Sheikh on Saturday.",
    "This is a, *weird sentence with *asterisks in it.",
    "The food\u2014which was delicious\u2014reminded me of home.",
    "The score was 3\u20131.",
    "He said \u2015 wait for it \u2015 a new thing.",
    "The number is 12\u201234.",
    "state-of-the-art",
    "Why did dotdot.. not get tokenized but dotdotdot... did? How about manydots.....",
    "The 'v', I've been fooled but I'll seek revenge.",
    "'v' 're'",
    "'Hard' to tell",
    "He said 'hello' to me",
    "It's o'clock already",
    "O'Connor went home",
    "Hello, world.",
    "\"Shippers are saying `the party's over,'\" said Mr. LaLonde.",
]

DETOK_LISTS = [
    ["hello", ",", "i", "ca", "n't", "feel", "my", "feet", "!", "Help", "!", "!"],
    ["hello", ",", "i", "can't", "feel", ";", "my", "feet", "!", "Help", "!", "!",
     "He", "said", ":", "Help", ",", "help", "?", "!"],
    ["Hello", ",", "world", "."], ["I", "'m", "sure", "."], ["You", "'ll", "see", "."],
    ["I", "do", "n't", "know", "."], ["``", "Hello", "''"],
    ["He", "said", "``", "hi", "''", "."], ["``", "Yes", ",", "''", "he", "said", "."],
    ["The", "dog", "'s", "bone", "."], ["-LRB-", "hello", "-RRB-"], ["foo", "--", "bar"],
    ["`", "Hello", "'"], ["``", "Shippers", "are", "saying", "`", "the", "party", "'s",
     "over", ",", "'", "''", "said", "Mr.", "LaLonde", "."],
    ["word", ",", "'", "''"], ["a", ";", "b"], ["a", ":", "b"], ["we", "'ve", "seen", "."],
    ["who", "'d", "know", "."], ["word", ".", "''"], ["word", ",", "''"], ["word", "!", "''"],
    ["word", "?", "''"], ["``", "A", "B", "C", "''"], ["``", "Hi", ",", "''", "said", "Jo", "."],
    ["a", ",", "b"], ["a", "."], ["``", "a", "''"],
    ["Lorem", "ipsum", "dolor", "sit", "amet", ".", "consectetur", "adipiscing", "elit", "."],
    ["I", "called", "Dr.", "Jones", ".", "I", "called", "Dr.", "Jones", "."],
    ["wait", "...", "what"], [], [""], [" "], ["", ""],
    ["can", "not"], ["Can", "\tNOT"], ["d", "'ye"], ["gim", "me"], ["gon", "na"],
    ["got", "ta"], ["lem", "me"], ["more", "'n"], ["wan", "na"], ["wan", "na", "x"],
    ["wan", "na\x1c"], ["wan", "na\x85"], ["'t", "is"], ["'T", "WAS"], ["x", "'t", "is", "y"],
    ["can\u3000not"], ["CAN", "NOT"], ["can", "not", "can", "not"], ["\u0130t", "'s"],
    ["'t", "i\u017f"], ["'t", "\u0130s"], ["g\u0130m", "me"], ["\u0301can", "not"],
    ["can", "not\u0301"], ["\u00b2can", "not"],
]


def adversarial():
    out = []
    quotes = ['"', "''", "``", "`", "'", "'''", '""', "```", "````", "\u00ab", "\u00bb",
              "\u201c", "\u201d", "\u2018", "\u2019", "\u201e", "' '", "'\"", "\"'", "`'"]
    places = ["{q}a b", "a {q}b", "a{q} b", "a b{q}", "({q}a)", "[{q}a", "<{q}a", "{{{q}a",
              "a.{q}", "a.{q} ", "a{q}.", "{q}", "a {q} b", "{q}{q}a{q}{q}", "{q}.",
              " {q}", "{q} ", "a{q}s b", "a{q}ll b", "x {q}tis", "{q}\n", "a, {q}b{q}, c",
              "{q}Hello,{q} he said{q}.", "{q}{q}", "a{q}", "{q}a", ". {q}a"]
    for q in quotes:
        for p in places:
            out.append(p.replace("{q}", q))
    clitics = ["'s", "'S", "'m", "'M", "'d", "'D", "'ll", "'LL", "'re", "'RE", "'ve", "'VE",
               "n't", "N'T", "N't", "n'T", "'Ll", "'rE", "'\u017f", "'\u0130", "'\u0131",
               "'t", "'T", "'n", "'nt", "'em", "'"]
    for c in clitics:
        for p in ["x{c} y", "x{c}", "{c} y", "{c}", "x {c} y", "x{c}.", "x{c},", "x{c}{c} y",
                  "'{c}x", "x{c}\n", "x{c}\ty", "x{c}\u00a0y", "\u0301{c} y", "x\u0301{c} y",
                  "x{c}\u0301 y", "_{c} y", "1{c} y", "\u00b2{c} y"]:
            out.append(p.replace("{c}", c))
    words = ["cannot", "d'ye", "gimme", "gonna", "gotta", "lemme", "more'n", "wanna",
             "'tis", "'twas"]

    def variants(w):
        vs = {w, w.upper(), w.title(), w.capitalize(), w.swapcase()}
        vs.add(w.replace("i", "\u0130"))
        vs.add(w.replace("i", "\u0131"))
        vs.add(w.replace("i", "I"))
        vs.add(w.replace("s", "\u017f"))
        vs.add(w.replace("s", "S"))
        vs.add(w.replace("k", "\u212a"))
        vs.add("".join(ch.upper() if k % 2 else ch for k, ch in enumerate(w)))
        return sorted(vs)

    # Each variant in "{w}", "x {w} y" and six of the other places.
    word_places = [" {w} ", "{w}.", "({w})", "\"{w}\"", "'{w}'", "{w}x", "x{w}",
                   "\u0301{w}", "{w}\u0301", "\u00b2{w}", "{w}\u00b2", "_{w}", "{w}_", "1{w}",
                   "{w}1", "{w}-be", "-{w}", "{w}\n", "\u0660{w}", "\u00aa{w}", "{w}\u2160",
                   "\u02b0{w}", "{w}\u0903", "x '{w}", "x {w}'"]
    rng = random.Random(7)
    for w in words:
        for v in variants(w):
            for p in ["{w}", "x {w} y"] + rng.sample(word_places, 6):
                out.append(p.replace("{w}", v))
    for ws in ["\x1c", "\x1d", "\x1e", "\x1f", "\x85", "\xa0", "\n", "\t", "\u2028",
               "\u3000", "\u200b", "\u180e", "\ufeff", ""]:
        out += [f"wanna{ws}", f"wanna{ws}x", f"x wanna{ws}", f"WanNa{ws}be", f"can{ws}not",
                f"'t{ws}is", f"a.{ws}", f"a{ws}.", f"a,{ws}", f"a.){ws}", f"a.'{ws}x",
                f"a{ws}'s", f"a{ws}b", f"{ws}", f"{ws}{ws}", f"''{ws}.", f"a{ws}--{ws}b",
                f"(a{ws}b)", f"\"{ws}a{ws}\""]
    k = 0
    for end in ["", " ", "\t", "\n", "\x0b", "\x0c", "\r", "\x1c", "\x85", "\xa0", "\u2028",
                "\u3000", "  ", " \n", "\n ", "\n\n", "\r\n", " \x1c", "\x1c ", "\n\x1c"]:
        for closers in ["", ")", "]", "}", ">", '"', "'", "\u00bb", "\u201d", "\u2019",
                        ") )", ")'\"", "'\u2019", " )", ") ", '"\u00bb', ")x", ".", ")."]:
            forms = [f"a.{closers}{end}", f"a..{closers}{end}", f".{closers}{end}",
                     f"a,{closers}{end}", f"a:{closers}{end}", f"Dr. X.{closers}{end}"]
            out += [forms[0], forms[1 + k % 5]]
            k += 1
    out += ["a..", "a...", "a....", "a.....", "a......", "a.. b", "... a", "a ... b",
            ". . .", "a...b", "a. . .", "a...\n", "...", "..", ".", "a.b.c.", "e.g. x",
            "U.S.A.", "a .", "a . ", "a.\u2026", "\u2026", "a\u2026b"]
    out += ["\u2012", "\u2013", "\u2014", "\u2015", "\u2011", "\u2010", "\u2212", "--",
            "---", "----", "-----", "-", "a--b", "a\u2014b", "a - b", "a -- b", "--a--",
            "a\u2012\u2015b", "-LRB-", "-RRB-", "(-LRB-)", "a-LRB-b"]
    out += ["a\u0301's", "\u0301'a", "'\u0301a", "x'\u0301", "can\u0301not", "'\u0301",
            "\u0301'", "e\u0301'll", "'e\u0301tude", "na\u0308ive's"]
    out += ["\u00b2cannot", "cannot\u00b2", "a,\u00b2", "a:\u00b2", ",\u00b2", "a,\u0663",
            "a,1", "a:1", "3:00", "1,000", "a, 1", ",1", ":1", "a,\u00bd", "a,\u2160",
            "a,\U0001d7ce", "a:\u0966"]
    for b in "()[]{}<>":
        out += [f"{b}", f"a{b}b", f" {b} ", f"{b}{b}", f"a.{b}", f"{b}\"a", f"{b}''a"]
    out += ["(a)", "[a]", "{a}", "<a>", "((a))", "a (b) c.", "[a.]", "(a.)", "{\"a\"}",
            "-LRB- a -RRB-", "-LSB-", "-RSB-", "-LCB-", "-RCB-"]
    out += ["", " ", "\t", "\n", "\x1c", "\x85", "\u3000", " \x1c ", "\x1c\x1d\x1e\x1f",
            "\u200b", "\ufeff", "\r\n"]
    out += ["a;b", "@a", "#1", "$5", "50%", "a&b", "a*b", "a?b", "a!b", "a?!", "?", "!",
            "a ; b", "a@b.com", "#$%&;@", "**", "a*", "*a*"]
    return list(dict.fromkeys(out))


def fuzz_texts(seed, n, max_pieces=8):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        k = rng.randint(1, max_pieces)
        out.append("".join(rng.choice(TEXT_PIECES) for _ in range(k)))
    return out


def edit_tokens(rng, toks):
    t = list(toks)
    k = rng.randrange(6)
    if k == 0:
        rng.shuffle(t)
    elif k == 1 or not t:
        for _ in range(rng.randint(1, 4)):
            t.insert(rng.randint(0, len(t)), rng.choice(TOKEN_PIECES))
    elif k == 2:
        for _ in range(rng.randint(1, max(1, len(t) // 3))):
            if t:
                del t[rng.randrange(len(t))]
    elif k == 3:
        for _ in range(rng.randint(1, 3)):
            t[rng.randrange(len(t))] = rng.choice(TOKEN_PIECES)
    elif k == 4 and len(t) > 1:
        i = rng.randrange(len(t) - 1)
        t[i : i + 2] = [t[i] + rng.choice(["", " ", "\t", "'", "."]) + t[i + 1]]
    else:
        i = rng.randrange(len(t))
        t.insert(i, rng.choice(TOKEN_PIECES))
        rng.shuffle(t)
    return t


def edits_for(text, seed, n):
    rng = random.Random(f"{seed}:{text}")
    nl = NL.tokenize(text)
    return [edit_tokens(rng, nl) for _ in range(n)]


def read_texts(path, n):
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["text"])
    return [t for t in table.column("text").to_pylist()[:n] if t is not None]


def language(path):
    return PUNKT.get(Path(path).name[:3], "english")


def sentences(doc, lang):
    return sent_tokenize(doc, lang)


# ---------------------------------------------------------------------------
# Commands


def write_rows(path, rows, kind="rows"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(header(kind), ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def golden(out_dir):
    rows = []
    for s in NLTK_SENTENCES:
        rows.append(row(s, edits_for(s, "nltk", 2)))
    for toks in DETOK_LISTS:
        rows.append(row(" ".join(toks), [toks]))
    for k, s in enumerate(adversarial()):
        rows.append(row(s, edits_for(s, "adv", 1) if k % 2 and len(s) > 2 else ()))
    rng = random.Random(20261109)
    fw = []
    for doc in read_texts(FINEWEB, 400):
        fw += sentences(doc, "english")
    for s in rng.sample(fw, 130):
        rows.append(row(s, edits_for(s, "fw", 1)))
    for path in FINEWEB2:
        lang = language(path)
        sents = [s for doc in read_texts(path, 60) for s in sentences(doc, lang) if len(s) < 300]
        for s in rng.sample(sents, min(3, len(sents))):
            rows.append(row(s, edits_for(s, "fw2", 1)))
    for s in fuzz_texts(1, 700):
        rows.append(row(s, edits_for(s, "fuzz", 1)))
    for s in fuzz_texts(2, 50, max_pieces=40):
        rows.append(row(s, edits_for(s, "fuzz2", 2)))
    write_rows(Path(out_dir) / "golden.jsonl", rows)
    print(f"golden: {len(rows)} rows", file=sys.stderr)


def large_row(args):
    text, with_edits = args
    return json.dumps(row(text, edits_for(text, "large", 2) if with_edits else ()),
                      ensure_ascii=False)


def large(out_dir, docs, fw2_docs, whole_docs):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [("fineweb", FINEWEB, docs, "english")] + [
        (Path(p).stem, p, fw2_docs, language(p)) for p in FINEWEB2
    ]
    with multiprocessing.Pool() as pool:
        for name, path, n, lang in jobs:
            items = []
            for k, doc in enumerate(read_texts(path, n)):
                for j, s in enumerate(sentences(doc, lang)):
                    items.append((s, j % 10 == 0))
                if k < whole_docs:
                    items.append((doc, False))
            with open(out_dir / f"large-{name}.jsonl", "w", encoding="utf-8") as f:
                f.write(json.dumps(header("rows")) + "\n")
                for line in pool.imap(large_row, items, chunksize=64):
                    f.write(line + "\n")
            print(f"{name}: {len(items)} rows", file=sys.stderr)


# The code point templates; RE_TEMPLATES are those that depend on `re`.
TEMPLATES = [
    "a{c}b", "{c}'s", "'{c}", "{c}.", "{c}cannot{c}", "{c}``", "g{c}mme x 'ti{c}",
    "can{c}not", "wanna{c}", "x{c}'t{c}is", "{c}\"",
]
RE_TEMPLATES = ["{c}cannot{c}", "g{c}mme x 'ti{c}", "can{c}not", "wanna{c}", "x{c}'t{c}is"]


def codepoint_result(tpl, cp, re_only):
    c = chr(cp)
    text = tpl.replace("{c}", c)

    def enc(s):
        return s.replace(c, "\x00")

    def enc_spans(x):
        return {"e": x["e"], "m": enc(x["m"])} if isinstance(x, dict) else x

    tb = TB.tokenize(text)
    nl = NL.tokenize(text)
    r = {"tb": enc(" ".join(tb)), "nl": enc(" ".join(nl))}
    if not re_only:
        r["tbs"] = enc_spans(outcome(lambda: TB.span_tokenize(text)))
        r["nls"] = enc_spans(outcome(lambda: NL.span_tokenize(text)))
        r["dnl"] = enc(DT.detokenize(nl))
        r["d1"] = enc(DT.detokenize([text]))
    return json.dumps(r, ensure_ascii=False, sort_keys=True)


def codepoint_slice(args):
    # mode: False (all treebank fields), True (--re-only) or "regexp".
    tpl, lo, hi, mode = args
    runs = []  # [key, first, last]
    for cp in range(lo, hi):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        if mode == "regexp":
            key = regexp_codepoint_result(tpl, cp)
        else:
            key = codepoint_result(tpl, cp, bool(mode))
        if runs and runs[-1][0] == key and runs[-1][2] == cp - 1:
            runs[-1][2] = cp
        else:
            runs.append([key, cp, cp])
    return runs


def codepoints(out_dir, mode, templates=None, suffix=""):
    """mode: False (TEMPLATES), True (RE_TEMPLATES, tokenizers only) or "regexp".

    `templates` replaces the mode's list; `suffix` ends the file name.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not templates:
        templates = {False: TEMPLATES, True: RE_TEMPLATES, "regexp": REGEXP_TEMPLATES}[mode]
    suffix = {False: "", True: "-re", "regexp": "-regexp"}[mode] + suffix
    kind = "regexp-codepoints" if mode == "regexp" else "codepoints"
    step = 0x1000
    path = out_dir / f"codepoints-{unicodedata.unidata_version}{suffix}.jsonl"
    with multiprocessing.Pool() as pool, open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(header(kind)) + "\n")
        for tpl in templates:
            jobs = [(tpl, lo, min(lo + step, 0x110000), mode) for lo in range(0, 0x110000, step)]
            groups = {}
            for runs in pool.imap(codepoint_slice, jobs):
                for key, a, b in runs:
                    ranges = groups.setdefault(key, [])
                    if ranges and ranges[-1][1] == a - 1:
                        ranges[-1][1] = b
                    else:
                        ranges.append([a, b])
            for key, ranges in groups.items():
                f.write(json.dumps({"tpl": tpl, "r": json.loads(key), "cps": ranges},
                                   ensure_ascii=False) + "\n")
            print(f"{tpl!r}: {len(groups)} results", file=sys.stderr)


# ---------------------------------------------------------------------------
# nltk.tokenize.regexp and nltk.tokenize.simple (pyroid's nltk::regexp)

# The RegexpTokenizer patterns pyroid runs natively (nltk::regexp::Pattern).
REGEXP_PATTERNS = [
    r"\w+", r"\w+|[^\w\s]+", r"\w+|\$[\d\.]+|\S+", r"[\w']+", r"\S+", r"\s+",
    r"[a-zA-Z]+", r"(?:(?<!\s)\s*)?\n\s*\n\s*",
]
LINE_MODES = ["discard", "keep", "discard-eof"]


def flat(spans):
    return [x for span in spans for x in span]


def regexp_fields(text):
    """The "regexp" record of `text` (without "t")."""
    r = {}
    # gaps=False: findall and span_tokenize, per pattern.
    r["f"] = [[RegexpTokenizer(p).tokenize(text), flat(RegexpTokenizer(p).span_tokenize(text))]
              for p in REGEXP_PATTERNS]
    # gaps=True with discard_empty True and False: tokenize and span_tokenize.
    r["g"] = [[f(RegexpTokenizer(p, gaps=True, discard_empty=d))
               for d in (True, False)
               for f in (lambda t: t.tokenize(text), lambda t: flat(t.span_tokenize(text)))]
              for p in REGEXP_PATTERNS]
    r["named"] = [
        wordpunct_tokenize(text), flat(WordPunctTokenizer().span_tokenize(text)),
        WhitespaceTokenizer().tokenize(text), flat(WhitespaceTokenizer().span_tokenize(text)),
        blankline_tokenize(text), flat(BlanklineTokenizer().span_tokenize(text)),
        SpaceTokenizer().tokenize(text), flat(SpaceTokenizer().span_tokenize(text)),
        TabTokenizer().tokenize(text), flat(TabTokenizer().span_tokenize(text)),
    ]
    r["lines"] = [[LineTokenizer(m).tokenize(text), flat(LineTokenizer(m).span_tokenize(text))]
                  for m in LINE_MODES]
    return r


REGEXP_PIECES = TEXT_PIECES + [
    "\n", "\n\n", "\n \n", " \n\t\n ", "\r\n\r\n", "\\n", "\t\t", "\x0b", "\x0c", "\x1c\n",
    "\n\x1c\n", " ", " ", "\x85\n", "$", "$1", "$.", "$3.88", "$٣", "x$",
    "'x'", "can't", "ABC", "Àb", "a_b", "²", "\U0001d7ce",
]


def regexp_texts():
    texts = [
        "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks.",
        "Alas, it has not rained today. When, do you think, will it rain again?",
        "<p>Although this is <b>not</b> the case here, we must not relax our vigilance!</p>",
        "a\tb c\n\t d", "", " ", "\n", "\n\n", "a\n\nb", "a \n \n b", "\n\na", "a\n\n",
        "a\n \t\n", "a\n\x1c\nb", "a\n\x85\n", "a  \n\nb\n\n\nc", "a\nb", "a\n  \n\nb", "a\n \n",
        "a\\nb", "a\nb\n", "\n a \n", "a  b", " a", "a ", "  ", "$abc a$1 $3.88, ,abc",
    ]
    rng = random.Random(3)
    for _ in range(200):
        texts.append("".join(rng.choice(REGEXP_PIECES) for _ in range(rng.randint(1, 7))))
    for doc in read_texts(FINEWEB, 6):
        texts.append(doc[:160])
    for path in FINEWEB2[::5]:
        texts.append(read_texts(path, 1)[0][:100])
    return list(dict.fromkeys(texts))


def regexp_golden(out_dir):
    rows = [{"t": t, **regexp_fields(t)} for t in regexp_texts()]
    write_rows(Path(out_dir) / "golden-regexp.jsonl", rows, kind="regexp")
    print(f"golden-regexp: {len(rows)} rows", file=sys.stderr)


REGEXP_TEMPLATES = ["a{c}b", "a\n{c}\nb", "{c}$1", "$1{c}", "{c}'x"]


def placeholder(value, c):
    """`value` with every occurrence of `c` in its strings written as U+0000."""
    if isinstance(value, str):
        return value.replace(c, "\x00")
    if isinstance(value, list):
        return [placeholder(v, c) for v in value]
    if isinstance(value, dict):
        return {k: placeholder(v, c) for k, v in value.items()}
    return value


def regexp_codepoint_result(tpl, cp):
    c = chr(cp)
    fields = regexp_fields(tpl.replace("{c}", c))
    return json.dumps(placeholder(fields, c), ensure_ascii=False, sort_keys=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("golden")
    g.add_argument("dir")
    rg = sub.add_parser("regexp")
    rg.add_argument("dir")
    lg = sub.add_parser("large")
    lg.add_argument("dir")
    lg.add_argument("--docs", type=int, default=5000)
    lg.add_argument("--fw2-docs", type=int, default=1000)
    lg.add_argument("--whole-docs", type=int, default=200)
    cp = sub.add_parser("codepoints")
    cp.add_argument("dir")
    cp.add_argument("--re-only", action="store_true")
    cp.add_argument("--regexp", action="store_true")
    cp.add_argument("--template", action="append", help="instead of the mode's templates")
    cp.add_argument("--suffix", default="", help="appended to the file name")
    a = p.parse_args()
    if a.cmd == "golden":
        golden(a.dir)
    elif a.cmd == "regexp":
        regexp_golden(a.dir)
    elif a.cmd == "large":
        large(a.dir, a.docs, a.fw2_docs, a.whole_docs)
    else:
        mode = "regexp" if a.regexp else a.re_only
        codepoints(a.dir, mode, a.template, a.suffix)


if __name__ == "__main__":
    main()
