"""Record NLTK 3.10.4's Punkt results for the replay tests of ``punkt.rs``.

Run in the dev container, with nltk 3.10.4 and regex 2026.9.29 installed::

    PY=/target/venv-313/bin/python
    $PY -I record.py goldens [DIR]          # params-*.json, golden-*.jsonl (committed)
    $PY -I record.py unicode [DIR]          # golden-unicode-<version>.jsonl (run per CPython)
    NLTK_DATA=/target/nltk_data $PY -I record.py pretrained DIR [DOCS]
                                            # punkt_tab models: DIR/<lang>.jsonl (never committed)
    [NLTK_DATA=...] $PY -I record.py fuzz DIR N [SEED]
                                            # 2N random texts, for PYROID_PUNKT_GOLDEN_DIR=DIR

DIR defaults to this script's directory (``pretrained`` needs one). Inputs:
NLTK's own test and doctest sentences; FineWeb (``/data/fineweb``) and
FineWeb-2 (``/data/fineweb2``, ``/target/fw2``) documents, both ODC-By;
hand-written edge cases; and seeded random texts. ``goldens`` trains its
parameter sets with NLTK's ``PunktTrainer`` on FineWeb text and keeps, of
each ``ortho_context``, the types the recorded inputs can look up.

Each line of a ``golden-*.jsonl`` file is one input:

* ``p``: the parameter set (``params-<p>.json``; ``empty`` has none);
  ``l``: ``sent_end_chars`` as a string when not ``.?!``;
* ``t``: the text, or ``k``: a token list (``sentences_from_tokens`` only);
* ``s`` / ``n``: ``span_tokenize(t)`` with / without realignment, as a flat
  list of code point offsets (``n`` is ``"=s"`` when the same);
* ``b``: ``text_contains_sentbreak(t)``;
* ``g``: ``sentences_from_text_legacy(t)``, or ``"=s"`` / ``"=n"`` when it is
  ``tokenize(t)`` with / without realignment;
* ``d``: ``debug_decisions(t)``, each ``[period_index, text, type1, type2,
  type1_in_abbrs, type1_is_initial, type2_is_sent_starter,
  type2_ortho_heuristic, ortho flag bits, collocation, REASONS index,
  break_decision]``;
* ``w``: the word tokens of ``t`` (``_tokenize_words``), for short texts;
* ``ks``: ``sentences_from_tokens(t.split())``; ``r``: the same for ``k``.

A generator that raised is ``{"e": type, "m": message, "y": what it yielded}``
(``d`` ends with ``{"e", "m"}`` after the dicts yielded). The pretrained
records have the same rows, without ``p``.
"""

# A recording script: it reports on stdout, and its inputs are deliberately confusable text
# (curly quotes, look-alike letters) in long literal lines.
# ruff: noqa: T201, RUF001, RUF005, E501, PERF401

import json
import os
import random
import sys
import unicodedata

import nltk
import regex
from nltk.tokenize import punkt

assert nltk.__version__ == "3.10.4", nltk.__version__
assert regex.__version__ == "2026.9.29", regex.__version__

HERE = os.path.dirname(os.path.abspath(__file__))
FINEWEB = "/data/fineweb/fineweb-dev-10k.parquet"
FINEWEB2 = "/data/fineweb2/{}.parquet"
FW2_TEST = "/target/fw2/{}-test-5k.parquet"

REASONS = [
    punkt.REASON_DEFAULT_DECISION,
    punkt.REASON_KNOWN_COLLOCATION,
    punkt.REASON_ABBR_WITH_ORTHOGRAPHIC_HEURISTIC,
    punkt.REASON_ABBR_WITH_SENTENCE_STARTER,
    punkt.REASON_INITIAL_WITH_ORTHOGRAPHIC_HEURISTIC,
    punkt.REASON_INITIAL_WITH_SPECIAL_ORTHOGRAPHIC_HEURISTIC,
]
assert punkt.REASON_NUMBER_WITH_ORTHOGRAPHIC_HEURISTIC in REASONS
ORTHO_BITS = {
    "BEG-UC": punkt._ORTHO_BEG_UC,
    "MID-UC": punkt._ORTHO_MID_UC,
    "UNK-UC": punkt._ORTHO_UNK_UC,
    "BEG-LC": punkt._ORTHO_BEG_LC,
    "MID-LC": punkt._ORTHO_MID_LC,
    "UNK-LC": punkt._ORTHO_UNK_LC,
}
DEFAULT_ENDS = "".join(punkt.PunktLanguageVars.sent_end_chars)


def texts(path, n=None, column="text"):
    import pyarrow.parquet as pq

    col = pq.read_table(path, columns=[column]).column(0).to_pylist()
    return col if n is None else col[:n]


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------


def lang_vars(ends):
    if ends is None or ends == DEFAULT_ENDS:
        return punkt.PunktLanguageVars()
    cls = type("LV", (punkt.PunktLanguageVars,), {"sent_end_chars": tuple(ends)})
    return cls()


def tokenizer(params, ends=None):
    t = punkt.PunktSentenceTokenizer(lang_vars=lang_vars(ends))
    if params is not None:
        t._params = params
    return t


def params_from_json(d):
    p = punkt.PunktParameters()
    p.abbrev_types = set(d["abbrev_types"])
    p.collocations = {tuple(c) for c in d["collocations"]}
    p.sent_starters = set(d["sent_starters"])
    for k, v in d["ortho_context"]:
        p.ortho_context[k] = v
    return p


def params_to_json(p):
    return {
        "abbrev_types": sorted(p.abbrev_types),
        "collocations": sorted(list(c) for c in p.collocations),
        "sent_starters": sorted(p.sent_starters),
        "ortho_context": sorted([k, v] for k, v in p.ortho_context.items() if v),
    }


def word_tokens(tok, text):
    return [t.tok for t in tok._tokenize_words(text)]


def lookup_keys(tok, text=None, tokens=None):
    """Every type a decision on `text` (or on `tokens`) can look up."""
    words = []
    if text is not None:
        words += word_tokens(tok, text)
        for _, context in tok._match_potential_end_contexts(text):
            words += word_tokens(tok, context)
        words += text.split()
    if tokens is not None:
        words += tokens
    keys = set()
    for w in words:
        t = punkt.PunktToken(w)
        keys.add(t.type)
        keys.add(t.type_no_period)
    return keys


def prune(params, keys):
    p = punkt.PunktParameters()
    p.abbrev_types = set(params.abbrev_types)
    p.collocations = set(params.collocations)
    p.sent_starters = set(params.sent_starters)
    for k, v in params.ortho_context.items():
        if k in keys and v:
            p.ortho_context[k] = v
    return p


# ---------------------------------------------------------------------------
# Recording one input
# ---------------------------------------------------------------------------


def generator(gen, conv=lambda x: x):
    out = []
    try:
        for item in gen:
            out.append(conv(item))
    except (IndexError, AssertionError) as e:
        return {"e": type(e).__name__, "m": str(e), "y": out}
    return out


def flat(spans):
    return [i for s in spans for i in s]


def decision(d):
    bits = 0
    for name in d["type2_ortho_contexts"]:
        bits |= ORTHO_BITS[name]
    heuristic = d["type2_ortho_heuristic"]
    assert heuristic in (True, False, "unknown"), heuristic
    return [
        d["period_index"],
        d["text"],
        d["type1"],
        d["type2"],
        int(d["type1_in_abbrs"]),
        int(d["type1_is_initial"]),
        int(d["type2_is_sent_starter"]),
        heuristic,
        bits,
        int(d["collocation"]),
        REASONS.index(d["reason"]),
        d["break_decision"],
    ]


def record_text(tok, text, *, debug=True, legacy=True, words=None):
    s = list(tok.span_tokenize(text))
    n = list(tok.span_tokenize(text, realign_boundaries=False))
    assert tok.tokenize(text) == [text[a:b] for a, b in s]
    row = {"t": text, "s": flat(s), "n": "=s" if n == s else flat(n)}
    row["b"] = tok.text_contains_sentbreak(text)
    if legacy:
        g = generator(tok.sentences_from_text_legacy(text))
        if g == [text[a:b] for a, b in s]:
            g = "=s"
        elif g == [text[a:b] for a, b in n]:
            g = "=n"
        row["g"] = g
    if debug:
        out = []
        try:
            for d in tok.debug_decisions(text):
                out.append(decision(d))
        except IndexError as e:
            out.append({"e": type(e).__name__, "m": str(e)})
        row["d"] = out
    if words if words is not None else len(text) <= 400:
        row["w"] = word_tokens(tok, text)
    row["ks"] = generator(tok.sentences_from_tokens(text.split()), len)
    return row


def record_tokens(tok, tokens):
    return {"k": tokens, "r": generator(tok.sentences_from_tokens(tokens), len)}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{path}: {len(rows)} rows, {os.path.getsize(path):,} bytes")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

NLTK_TEXTS = [
    # punkt.py's module docstring and _match_potential_end_contexts doctest
    """Punkt knows that the periods in Mr. Smith and Johann S. Bach
do not mark sentence boundaries.  And sometimes sentences
can start with non-capitalized words.  i is a good variable
name.""",
    """(How does it deal with this parenthesis?)  "It should be part of the
previous sentence." "(And the same with this one.)" ('And this one!')
"('(And (this)) '?)" [(and this. )]""",
    "Very bad acting!!! I promise.",
    # tokenize.doctest
    "See Section 3).  Or Section 2).  ",
    "See Section 3.)  Or Section 2.)  ",
    ". This input starts with a dot. This used to cause issues.",
    # data.doctest, sentiment.doctest, smoke tests
    "Hello.  This is a test.  It works!",
    "It was one of the worst movies I've seen, despite good reviews. "
    "Unbelievably bad acting!! Poor direction. VERY poor production. "
    "The movie was bad. Very bad movie. VERY bad movie. VERY BAD movie. VERY BAD movie!",
    "One. Two! Three?",
    "Hello world. This is a test.",
    "Hello world. This is a test. Dr. Smith went home. It works well. "
    "Another sentence here. And one more follows. ",
    'He said "hello" and left. It\'s fine.',
    "Dr. Smith isn't here.",
    # test_tokenize.py
    "Subject: Some subject. Attachments: Some attachments",
    "Subject: Some subject! Attachments: Some attachments",
    "This is just a normal sentence, just like any other.",
    "Subject: Some subject^ Attachments: Some attachments",
    "this is a test. . new sentence.",
    "This. . . That",
    "This..... That",
    "This... That",
    "This.. . That",
    "This. .. That",
    "This. ,. That",
    "This!!! That",
    "This! That",
    "1. This is R .\n2. This is A .\n3. That's all",
    "1. This is R .\t2. This is A .\t3. That's all",
    "Hello.\tThere",
    "“First sentence.” Next one.",
    "He said “this is great.” Then left.",
    "She asked, “Are you coming?” He nodded.",
    "Il a dit «bonjour.» Puis il est parti.",
    'He said "hi." Then left.',
    "This is one sentence. This is another.",
    "উপরাষ্ট্রপতি শ্রী এম ভেঙ্কাইয়া নাইডু সোমবার আই আই টি দিল্লির হীরক জয়ন্তী উদযাপনের উদ্বোধন করেছেন। অনলাইনের মাধ্যমে এই অনুষ্ঠানে কেন্দ্রীয় মানব সম্পদ উন্নয়নমন্ত্রী শ্রী রমেশ পোখরিয়াল ‘নিশাঙ্ক’  উপস্থিত ছিলেন। এই উপলক্ষ্যে উপরাষ্ট্রপতি হীরকজয়ন্তীর লোগো এবং ২০৩০-এর জন্য প্রতিষ্ঠানের লক্ষ্য ও পরিকল্পনার নথি প্রকাশ করেছেন।",
]

# sent_tokenize() expectations in NLTK's tests (English model).
NLTK_ENGLISH_EXPECTED = [
    ("this is a test. . new sentence.", ["this is a test.", ".", "new sentence."]),
    ("This. . . That", ["This.", ".", ".", "That"]),
    ("This..... That", ["This..... That"]),
    ("This... That", ["This... That"]),
    ("This.. . That", ["This.. .", "That"]),
    ("This. .. That", ["This.", ".. That"]),
    ("This. ,. That", ["This.", ",.", "That"]),
    ("This!!! That", ["This!!!", "That"]),
    ("This! That", ["This!", "That"]),
    (
        "1. This is R .\n2. This is A .\n3. That's all",
        ["1.", "This is R .", "2.", "This is A .", "3.", "That's all"],
    ),
    (
        "1. This is R .\t2. This is A .\t3. That's all",
        ["1.", "This is R .", "2.", "This is A .", "3.", "That's all"],
    ),
    ("Hello.\tThere", ["Hello.", "There"]),
    ("“First sentence.” Next one.", ["“First sentence.”", "Next one."]),
    ("He said “this is great.” Then left.", ["He said “this is great.”", "Then left."]),
    ("She asked, “Are you coming?” He nodded.", ["She asked, “Are you coming?”", "He nodded."]),
    ("Il a dit «bonjour.» Puis il est parti.", ["Il a dit «bonjour.»", "Puis il est parti."]),
    ('He said "hi." Then left.', ['He said "hi."', "Then left."]),
    ("This is one sentence. This is another.", ["This is one sentence.", "This is another."]),
    ("Hello.  This is a test.  It works!", ["Hello.", "This is a test.", "It works!"]),
    ("One. Two! Three?", ["One.", "Two!", "Three?"]),
    (
        NLTK_TEXTS[0],
        [
            "Punkt knows that the periods in Mr. Smith and Johann S. Bach\ndo not mark sentence boundaries.",
            "And sometimes sentences\ncan start with non-capitalized words.",
            "i is a good variable\nname.",
        ],
    ),
]

QUOTES = ['"', "'", ")", "]", "}", "‘", "’", "“", "”", "«", "»"]
NON_WORD = [")", '"', ";", "}", "]", "*", ":", "@", "'", "(", "{", "["] + QUOTES[5:]
SPACES = [
    " ", "  ", "\t", "\n", "\n\n", "\r\n", "\r", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x1f",
    "\x85", "\xa0", "\u1680", "\u2000", "\u2028", "\u2029", "\u202f", "\u205f", "\u3000",
    "\u200b", "\u180e", "\ufeff",
]  # fmt: skip
FOLLOWERS = (
    NON_WORD + SPACES + ["", "x", "X", ".", "..", "--", "-", ",", "?", "!", "1", "\u0301", "²"]
)


def adversarial():
    out = []
    for e in [".", "?", "!"]:
        for f in FOLLOWERS:
            out += [f"Word{e}{f} Next word.", f"Word{e}{f}", f"(Word{e}){f} Next.", f"a{e}{f}b{e}"]
    for q in QUOTES:
        out += [
            f"He said {q}hi.{q} Then left.",
            f"Sent1.{q}{q} Sent2.",
            f"Sent1.{q}--Sent2.",
            f"Sent1.{q}",
            f"Sent1.{q}\nSent2.",
            f"Sent1.{q} {q}Sent2.{q}  ",
            f"Sent1!{q}\u3000Sent2?{q}\x1cSent3.",
            f"{q}Sent1.{q}{q}\n\n{q}Sent2.{q}",
        ]
    out += [
        # ellipses and multi-character punctuation
        "This. . . That", "a . . . b", "a. . . . . b", "a.. . b", "a... b", "a.... B", "a . b",
        ". . . .", ".. .. ..", "x. .", ". .x", "x.\u3000.\u3000.\u3000y", ". .\x1c.", "a--b. --C",
        "a---. B", "--. --", "- - -. A", "Wait... what?! Really?!? Yes!!!", "Hm.. ok. . .fine",
        # numbers
        "It is 3.5. Next", "Page 3. Next", "-1. Next", "1,000. Next", "No. 5 is here.",
        "3.5.6. X", ".5. X", ",5. x", "Item 12. it", "On 12. Jan. 2020 it", "٣. ٣",
        "x². Y", "1st. Then", "2.) Two", "10-12. Next", "-.5. Next",
        # initials
        "J. S. Bach was here.", "J. Bach.", "A. B. C. D.", "Ä. Ö. Ü. test", "1. 2. 3.",
        "John F. Kennedy. He", "e\u0301. next", "\u0301. X", "_. X", "ß. Weiter",
        # abbreviations, hyphens
        "pre-Dr. Smith", "ex-Mr. X", "a-b-c-dr. X", "Dr.-Ing. Y", "Dr. Smith. Dr. smith",
        "e.g. this", "U.S. Army", "u.s. army", "etc.) Next", "(etc.) Next", "Mr.Smith. Hi",
        "St. Louis. St. louis", "vs. them", "Fig. 3 shows", "Inc. The", "Ltd. the",
        # paragraphs and line ends
        "One.\n\nTwo.", "One.\r\nTwo.", "One.\r\n\r\nTwo", "One.\n \nTwo", "One.\n\x1c\nTwo",
        "One.\x1cTwo", "One. \x1c", "One.\x85Two", "One.\u2028Two", "One.\u2029Two",
        "a.\n\x1c", "a.\n\x1d\nb", "a.\n\x1c\n\x1cb", "a.\n\x1c\n\x1c a", "a. \n\x1f\n\x1fb c",
        "\x1c\n\x1c", "One.\n\x0b\nTwo", "One.\r\rTwo", "\n\nOne.\n\n", " \t One.  Two.  \t ",
        "One.\u3000\u3000Two.\u3000", "One.\xa0Two", "One.\u200bTwo", "One. \ufeffTwo",
        # case, Unicode
        "Café. Next", "Cafe\u0301. Next", "İstanbul. İstanbul is", "Go to İst. Now",
        "ΟΔΟΣ. Next", "ΟΔΟΣ. οδος", "ΑΣ.Β. Γ", "Σ. Σ", "ǅemal. ǆemal ǅ",
        "Straße. STRASSE. straße", "ﬁne. ﬃ", "这是. 中文. 日本語。 Next", "😀. 😀! 😀",
        "x\U0001d7ce. Y", "Ⓐ. ⓐ", "ʕ. ʕord",
        # empty and blank
        "", " ", "\n", "\x1c", "  .  ", ".", "!", "?", "..", "a", "a.", " a. ", "\x1c\x1d",
        "\u3000", ". ", "! !", "?\n?", ".\x1c",
        # punctuation inside words and commas
        "a, b", "a,", "a,b", "a,.", "a,--b", "a,)b", "a,,b", "Hi., There", "Hi.,There",
        "a.b.c. D", "x.y! Z", "`quote`. Next", "&amp. Next", "#tag. Next", "@user. Next",
        "*star*. Next", "Hi!) There", "Hi!? There", "Hi?! There", "What?!.. There",
        "Hi’. Next", "x.” Y", "x.» Y", "x.«Y", "x.(y) Z", "x.[1] Y", "x.{a} B",
        "x.;y Z", "x.:y Z", "x.@y Z", "x.*y Z", "x.'y Z", "x.\"y Z", "x.‘y’ Z",
    ]  # fmt: skip
    return out


# Long texts, recorded once (without debug_decisions past 1200 characters).
LONG = [
    "a" * 3000 + ". B",
    "." * 2000,
    ". " * 1000,
    "-" * 2000,
    "x." * 1000,
    "!" * 2000 + " X",
    "Ab. " * 600,
    "a" * 5000,
    "(" * 500 + "x." + ")" * 500 + " Y",
    "Dr. " * 200 + "x",
    ". " * 300 + "x",
]


def fuzz_pool():
    words = [
        "the", "The", "THE", "Mr", "mr", "Mrs", "Dr", "dr", "Smith", "smith", "S", "J", "Bach",
        "e.g", "i.e", "U.S", "u.s", "etc", "vs", "St", "Louis", "apple", "Apple", "is", "Is",
        "iPad", "İstanbul", "istanbul", "İst", "ΟΔΟΣ", "οδος", "Straße", "STRASSE", "ǅemal",
        "café", "cafe\u0301", "٣٤", "12", "3.5", "-1", "1,000", "²", "x²", "中文",
        "日本語", "😀", "However", "but", "And", "army", "Army", "no", "No", "Jan", "fig",
        "a", "I", "i", "x", "Z", "ß", "ʕ", "Ᲊord", "Ɤbc", "_", "a_b",
    ] + [f"w{i}" for i in range(0, 128, 3)] + [f"W{i}" for i in range(1, 128, 3)]  # fmt: skip
    punct = [
        ".", ".", ".", "?", "!", "..", "...", ". . .", ". .", "--", "-", ",", ";", ":", "(",
        ")", "[", "]", "{", "}", '"', "'", "`", "‘", "’", "“", "”", "«", "»", "*", "@", "#",
        "&", "।", "^", "。", "%", "/", "\\",
    ]  # fmt: skip
    spaces = [" "] * 8 + SPACES
    return words, punct, spaces


def fuzz_texts(seed, n):
    rng = random.Random(seed)
    words, punct, spaces = fuzz_pool()
    out = []
    for i in range(n):
        if i % 4 == 3:
            alphabet = "aAbZ.?!,;:-'\"()[]«»‘’“” \n\t\x1c\x85\u3000É\u0301İΣ1²٣_.."
            out.append("".join(rng.choice(alphabet) for _ in range(rng.randint(1, 30))))
            continue
        pieces = []
        for _ in range(rng.randint(1, 24)):
            r = rng.random()
            pool = words if r < 0.42 else punct if r < 0.75 else spaces
            pieces.append(rng.choice(pool))
        out.append("".join(pieces))
    return out


def token_lists(seed, n):
    rng = random.Random(seed)
    words, punct, _ = fuzz_pool()
    pool = (
        words
        + punct
        + [w + "." for w in words]
        + ["", "\n", "12\n", "a.\n", "..\n", "Mr.\n", " ", ". ."]
    )
    out = [
        ["Mr.", ""],
        ["Dr.", "", "x"],
        ["J.", ""],
        ["3.", ""],
        ["...", ""],
        ["", "."],
        [],
        [""],
        ["Hello.", "b", "Mr.", ""],
        ["12\n", "Next"],
        ["dr.", "12\n"],
        ["dr.", "W5"],
        ["J.", "Bach", "."],
    ]
    for _ in range(n):
        out.append([rng.choice(pool) for _ in range(rng.randint(1, 12))])
    return out


SYNTH = {
    "abbrev_types": [
        "dr", "mr", "mrs", "e.g", "i.e", "u.s", "etc", "vs", "st", "no", "fig", "jan", "inc",
        "ltd", "z.b", "bzw", "i\u0307st", "οδος", "straße", "a.m", "p.m", "ca", "w1", "w2",
        "w3", "w4", "w9", "w12", "ing", "-", "", "1st",
    ],
    "collocations": [
        ["s", "bach"], ["j", "s"], ["dr", "smith"], ["##number##", "jan"], ["mr", "##number##"],
        ["u.s", "army"], ["e.g", "the"], ["st", "louis"], ["etc", "and"], ["w6", "w7"],
        ["w15", "the"], ["j", "bach"], ["i", "##number##"],
    ],
    "sent_starters": [
        "the", "but", "and", "however", "it", "we", "this", "##number##", "i\u0307stanbul",
        "das", "smith", "w4", "w5", "w10", "w13", "w16", "w19", "army",
    ],
    "ortho_context": sorted(
        [[f"w{i}", i] for i in range(128)]
        + [
            ["the", 2 | 32], ["but", 2 | 32 | 64], ["and", 2 | 16 | 32], ["however", 2],
            ["smith", 4 | 8], ["bach", 4], ["apple", 16 | 32], ["is", 32], ["ipad", 16],
            ["i\u0307stanbul", 2 | 4], ["istanbul", 32], ["οδος", 2 | 32], ["army", 4 | 32],
            ["louis", 4], ["no", 2 | 16 | 32], ["jan", 4 | 8], ["##number##", 32 | 2],
            ["straße", 2], ["café", 2 | 32], ["a", 32 | 4], ["i", 4], ["x", 64], ["z", 8],
            ["ß", 16], ["ʕ", 32], ["ᲊord", 16], ["ɤbc", 32], ["日本語", 2], ["😀", 16],
            ["s", 4], ["j", 4], ["mr", 2 | 4], ["dr", 2 | 4 | 32],
        ]
    ),
}  # fmt: skip


def fineweb_excerpts(docs, start, n, seed, cut=1800):
    rng = random.Random(seed)
    out = []
    for d in docs[start : start + n]:
        if len(d) > cut:
            a = rng.randrange(0, len(d) - cut)
            d = d[a : a + cut]
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def train(text, include_all_collocs=False):
    trainer = punkt.PunktTrainer()
    trainer.INCLUDE_ALL_COLLOCS = include_all_collocs
    trainer.train(text, finalize=True)
    return trainer.get_params()


def cmd_goldens(out_dir):
    en_docs = texts(FINEWEB)
    de_docs = texts(FINEWEB2.format("deu_Latn"))
    trained = {
        "en": train("\n\n".join(en_docs[:900])),
        "en-colloc": train("\n\n".join(en_docs[900:1500]), include_all_collocs=True),
        "de": train("\n\n".join(de_docs[:180])),
    }
    synth = params_from_json(SYNTH)

    fuzz = fuzz_texts(1, 1950)
    adv = adversarial()
    jobs = {
        "nltk": [(p, None, t) for p in ["empty", "en", "de", "synth"] for t in NLTK_TEXTS]
        + [("empty", ".?!।", NLTK_TEXTS[-1]), ("empty", ".?!^", NLTK_TEXTS[16])],
        "adversarial": [(p, None, t) for p in ["empty", "synth"] for t in adv]
        + [("synth", None, t) for t in LONG]
        + [("en", None, t) for t in adv[::5]]
        + [("synth", ".?!।^;", t) for t in adv[::4]]
        + [("synth", "!?。", t) for t in adv[1::4]],
        "fuzz": [("synth", None, t) for t in fuzz[:1300]]
        + [("synth", ".?!।^。", t) for t in fuzz[1300:1650]]
        + [("en", None, t) for t in fuzz[1650:1800]]
        + [("empty", "?", t) for t in fuzz[1800:1950]],
        "fineweb": [("en", None, t) for t in fineweb_excerpts(en_docs, 6000, 90, 2, 1400)]
        + [("en-colloc", None, t) for t in fineweb_excerpts(en_docs, 7000, 40, 3, 1400)]
        + [("de", None, t) for t in fineweb_excerpts(de_docs, 180, 60, 4, 1400)],
    }
    token_jobs = [(p, None, k) for p in ["empty", "synth"] for k in token_lists(5, 300)]
    token_jobs += [("en", None, k) for k in token_lists(6, 60)]

    # Keep, of each trained ortho_context, the types these inputs can look up.
    keys = {name: set() for name in trained}
    for jobs_ in jobs.values():
        for p, ends, t in jobs_:
            if p in keys:
                keys[p] |= lookup_keys(tokenizer(None, ends), text=t)
    for p, ends, k in token_jobs:
        if p in keys:
            keys[p] |= lookup_keys(tokenizer(None, ends), tokens=k)
    params = {"empty": None, "synth": synth}
    for name, p in trained.items():
        params[name] = prune(p, keys[name])
        path = os.path.join(out_dir, f"params-{name}.json")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(params_to_json(params[name]), f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")
        print(f"{path}: {os.path.getsize(path):,} bytes")
    path = os.path.join(out_dir, "params-synth.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(params_to_json(synth), f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")

    toks = {}

    def tok(p, ends):
        key = (p, ends)
        if key not in toks:
            toks[key] = tokenizer(params[p], ends)
        return toks[key]

    for name, jobs_ in jobs.items():
        rows = []
        for p, ends, t in jobs_:
            row = {"p": p}
            if ends is not None:
                row["l"] = ends
            debug = len(t) <= 1200 and (name != "fineweb" or len(rows) % 3 == 0)
            row.update(record_text(tok(p, ends), t, debug=debug))
            rows.append(row)
        write_jsonl(os.path.join(out_dir, f"golden-{name}.jsonl"), rows)
    rows = []
    for p, ends, k in token_jobs:
        row = {"p": p}
        row.update(record_tokens(tok(p, ends), k))
        rows.append(row)
    write_jsonl(os.path.join(out_dir, "golden-tokens.jsonl"), rows)


# Characters whose isupper / islower / lower() differ between CPython 3.12..3.15.
UNICODE_CHARS = [
    "ʕ", "Ᲊ", "ᲊ", "Ɤ", "Ꟍ", "ꟍ", "꟎", "꟏", "꟒",
    "Ꟛ", "ꟛ", "Ƛ", "꟱", "\U00010d50", "\U00010d70", "\U00016ea0",
    "\U00016ebb", "A", "a",
]  # fmt: skip
# Their lowercase in the newest version.
UNICODE_LOWER = {
    "Ᲊ": "ᲊ", "Ɤ": "ɤ", "Ꟍ": "ꟍ", "꟎": "꟏",
    "꟒": "ꟓ", "Ꟛ": "ꟛ", "Ƛ": "ƛ", "\U00010d50": "\U00010d70",
    "\U00016ea0": "\U00016ebb", "A": "a",
}  # fmt: skip


def unicode_params():
    p = punkt.PunktParameters()
    p.abbrev_types = {"dr", "mr"} | {
        UNICODE_LOWER.get(c, c) + "bc" for c in UNICODE_CHARS if c in UNICODE_LOWER
    }
    p.sent_starters = {UNICODE_LOWER.get(c, c) + "ord" for c in UNICODE_CHARS[::2]}
    for i, c in enumerate(UNICODE_CHARS):
        lc = UNICODE_LOWER.get(c, c)
        p.ortho_context[lc + "ord"] = [16, 32, 2 | 32, 4 | 32, 64][i % 5]
        p.ortho_context[c + "ord"] = [2, 4, 16 | 4, 0, 32][i % 5]
    return p


def unicode_texts():
    out = []
    for c in UNICODE_CHARS:
        out += [
            f"Ask the dr. {c}ord here.",
            f"J. {c}ord here.",
            f"Go {c}bc. then more.",
            f"Go {c}bc. Then more.",
            f"{c}. {c}ord",
            f"Mr. {c}",
            f"3. {c}ord",
            f"The {c}{c}. {c.upper()}ord. ...  {c}ord",
        ]
    return out


def cmd_unicode(out_dir):
    params = unicode_params()
    with open(
        os.path.join(out_dir, "params-unicode.json"), "w", encoding="utf-8", newline="\n"
    ) as f:
        json.dump(params_to_json(params), f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    tok = tokenizer(params)
    rows = []
    for t in unicode_texts():
        row = {"v": unicodedata.unidata_version}
        row.update(record_text(tok, t, words=True))
        rows.append(row)
    write_jsonl(os.path.join(out_dir, f"golden-unicode-{unicodedata.unidata_version}.jsonl"), rows)


LANGS = {
    "english": FINEWEB,
    "czech": FW2_TEST.format("ces_Latn"),
    "danish": FW2_TEST.format("dan_Latn"),
    "dutch": FW2_TEST.format("nld_Latn"),
    "estonian": FW2_TEST.format("ekk_Latn"),
    "finnish": FW2_TEST.format("fin_Latn"),
    "french": FINEWEB2.format("fra_Latn"),
    "german": FINEWEB2.format("deu_Latn"),
    "greek": FW2_TEST.format("ell_Grek"),
    "italian": FW2_TEST.format("ita_Latn"),
    "malayalam": FW2_TEST.format("mal_Mlym"),
    "norwegian": FW2_TEST.format("nob_Latn"),
    "polish": FW2_TEST.format("pol_Latn"),
    "portuguese": FW2_TEST.format("por_Latn"),
    "russian": FINEWEB2.format("rus_Cyrl"),
    "slovene": FW2_TEST.format("slv_Latn"),
    "spanish": FINEWEB2.format("spa_Latn"),
    "swedish": FW2_TEST.format("swe_Latn"),
    "turkish": FW2_TEST.format("tur_Latn"),
}


def pretrained_docs(path, n):
    docs = texts(path)
    if len(docs) >= n:
        return docs[:n]
    # A small sample: the documents, then their lines and paragraphs.
    out = list(docs)
    for d in docs:
        out += [p for p in d.split("\n\n") if len(p) >= 40 and p != d]
    for d in docs:
        out += [line for line in d.split("\n") if len(line) >= 40 and line not in d.split("\n\n")]
    return out[:n]


def cmd_pretrained(out_dir, n):
    from nltk.tokenize import PunktTokenizer

    os.makedirs(out_dir, exist_ok=True)
    english = PunktTokenizer("english")
    for text, want in NLTK_ENGLISH_EXPECTED:
        assert english.tokenize(text) == want, (text, english.tokenize(text))
    for lang, path in LANGS.items():
        tok = PunktTokenizer(lang)
        docs = pretrained_docs(path, n)
        if lang == "english":
            docs = [t for t, _ in NLTK_ENGLISH_EXPECTED] + NLTK_TEXTS + adversarial() + docs
        else:
            docs = NLTK_TEXTS + docs
        rows = [
            record_text(tok, t, debug=i % 4 == 0, legacy=i % 4 == 0, words=False)
            for i, t in enumerate(docs)
        ]
        write_jsonl(os.path.join(out_dir, f"{lang}.jsonl"), rows)


def unicode_fuzz_texts(seed, n):
    """Random texts over characters of many scripts and classes."""
    rng = random.Random(seed)
    ranges = [
        (0x20, 0x7E), (0x09, 0x0D), (0x1C, 0x1F), (0x80, 0x24F), (0x250, 0x2FF),
        (0x300, 0x36F), (0x370, 0x3FF), (0x400, 0x52F), (0x530, 0x58F), (0x590, 0x6FF),
        (0x900, 0x9FF), (0xD00, 0xD7F), (0x10A0, 0x10FF), (0x1C80, 0x1CBF), (0x1E00, 0x1FFF),
        (0x2000, 0x206F), (0x2100, 0x218F), (0x2460, 0x24FF), (0x2C00, 0x2C7F), (0x3000, 0x303F),
        (0x4E00, 0x4E3F), (0xA640, 0xA69F), (0xA720, 0xA7FF), (0xFB00, 0xFB06), (0xFF00, 0xFF5E),
        (0x10400, 0x1044F), (0x10D40, 0x10D8F), (0x16EA0, 0x16EDF), (0x1D400, 0x1D7FF),
        (0x1E900, 0x1E95F), (0x1F600, 0x1F64F),
    ]  # fmt: skip
    common = list(". . . ?!.,;:-'\"()[]{}«»‘’“” \n\n\t") + ["Mr.", "Dr.", "e.g.", "J.", "3.", "..."]
    out = []
    for _ in range(n):
        chars = []
        for _ in range(rng.randint(1, 60)):
            if rng.random() < 0.45:
                chars.append(rng.choice(common))
            else:
                a, b = rng.choice(ranges)
                cp = rng.randint(a, b)
                if 0xD800 <= cp < 0xE000:
                    cp = 0x41
                chars.append(chr(cp))
        out.append("".join(chars))
    return out


def cmd_fuzz(out_dir, n, seed):
    """A larger seeded run for local replay (PYROID_PUNKT_GOLDEN_DIR); not committed."""
    os.makedirs(out_dir, exist_ok=True)
    params = {"empty": None, "synth": params_from_json(SYNTH)}
    if os.environ.get("NLTK_DATA"):
        from nltk.tokenize import PunktTokenizer

        params["english"] = PunktTokenizer("english")._params
    for name, p in params.items():
        if p is not None:
            with open(os.path.join(out_dir, f"params-{name}.json"), "w", encoding="utf-8") as f:
                json.dump(params_to_json(p), f, ensure_ascii=False)
    texts_ = fuzz_texts(seed, n) + unicode_fuzz_texts(seed + 1, n)
    rows = []
    for i, t in enumerate(texts_):
        p = list(params)[i % len(params)]
        ends = [None, None, ".?!।^。;", "!?"][i % 4]
        row = {"p": p, "v": unicodedata.unidata_version}
        if ends is not None:
            row["l"] = ends
        row.update(record_text(tokenizer(params[p], ends), t, words=True))
        rows.append(row)
    write_jsonl(os.path.join(out_dir, f"golden-fuzzbig-{unicodedata.unidata_version}.jsonl"), rows)
    rows = []
    for i, k in enumerate(token_lists(seed + 2, n // 4)):
        p = list(params)[i % len(params)]
        rows.append(
            {"p": p, "v": unicodedata.unidata_version, **record_tokens(tokenizer(params[p]), k)}
        )
    write_jsonl(
        os.path.join(out_dir, f"golden-tokensbig-{unicodedata.unidata_version}.jsonl"), rows
    )


def main(argv):
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "goldens":
        cmd_goldens(argv[2] if len(argv) > 2 else HERE)
    elif cmd == "unicode":
        cmd_unicode(argv[2] if len(argv) > 2 else HERE)
    elif cmd == "fuzz" and len(argv) > 3:
        cmd_fuzz(argv[2], int(argv[3]), int(argv[4]) if len(argv) > 4 else 7)
    elif cmd == "pretrained" and len(argv) > 2:
        cmd_pretrained(argv[2], int(argv[3]) if len(argv) > 3 else 2000)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
