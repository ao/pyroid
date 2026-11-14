"""A deterministic corpus of realistic broken LLM outputs for the ``json_repair`` parity suite.

    from pyroid_parity import json_repair_corpus as JC
    JC.cases(20_000)        # list[Case]; any n is a prefix of the same sequence (each case has its own seed)
    JC.case(123)            # one case

    python -m pyroid_parity.json_repair_corpus --stats             # recipes, sizes, how often json.loads fails
    python -m pyroid_parity.json_repair_corpus --show 0 5           # print cases 0..4
    python -m pyroid_parity.json_repair_corpus --pin                # rewrite corpus-outcomes.json (json_repair)

Each case starts from a valid JSON document shaped like real model / tool-call / API output (tool calls with
arguments as an escaped JSON string, arrays of records, extraction and classification results, code in strings,
multilingual text, numbers in every JSON form), serialised in one of several styles (compact, indented, ASCII-only,
Python ``repr``), and then broken the way LLM output breaks:

* **truncate**: a streaming prefix, cut anywhere or at a sensitive spot (mid-escape, mid-``\\u`` escape,
  mid-number, mid-literal, right after ``:`` / ``,`` / an opening quote or bracket);
* **mutate**: 1-4 random deletions / insertions / replacements from a JSON-relevant alphabet;
* **structural**: 1-3 edits (drop a quote / brace / bracket / comma / colon, other quote styles ``'`` ``\u201c``
  ``\u201d`` ``\u201e`` ``\u2018`` ``\u2019`` and backticks, trailing commas, unquoted keys, Python literals,
  ``#`` ``//`` ``/* */`` comments, duplicate keys, unescaped inner quotes, raw newlines in strings, NaN /
  Infinity, odd numbers, ``...`` ellipses, extra or missing closers);
* **wrap**: markdown fences (closed or not), prose before / after, ``<json>`` tags, parentheses (Python tuples),
  a JSON document quoted as a string, several JSON blocks in one answer;
* combinations of the above (a fenced answer cut mid-stream, a wrapped document with structural damage).

Stdlib only: ``python/tests`` loads this file by path. ``random.Random`` with an int seed, ``random()``,
``randrange``, ``choice`` and ``shuffle`` produce the same sequence on every CPython 3.x, and no set or hash
order is ever iterated, so the corpus is identical across interpreters and ``PYTHONHASHSEED`` values
(``--pin`` records its sha256; the parity suite checks it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SEED = 645
DEFAULT_SIZE = 20_000
CHUNK = 250
DATA = Path(__file__).resolve().parents[1] / "data" / "json_repair-0.64.9"
PIN = DATA / "corpus-outcomes.json"
PIN_FORMAT = "pyroid-json_repair-corpus/1"


@dataclass(frozen=True)
class Case:
    id: str
    recipe: str
    text: str


# ------------------------------------------------------------------------------------------- vocabulary

WORDS = (  # noqa: SIM905  a word list reads better as text
    "the quick brown fox jumps over lazy dog weather forecast report summary customer order invoice "
    "payment status pending shipped delivered cancelled refund account user profile settings language "
    "model response request token context window prompt answer question reason because however therefore "
    "temperature pressure humidity wind speed north south east west city country region street building "
    "meeting agenda project deadline budget review approve reject comment issue ticket priority high low "
    "medium critical minor feature release version update patch install configure deploy server client "
    "database table column index query result error warning info debug trace event metric value total"
).split()
NAMES = ("Alice", "Bob", "Carol", "Dave", "Eve", "Mallory", "Oscar", "Peggy", "Trent", "Victor", "Wendy")
CITIES = ("Paris", "London", "New York", "Tokyo", "Berlin", "Lagos", "Lima", "Mumbai", "Cairo", "Sydney")
FUNCTIONS = (
    "get_weather",
    "search_web",
    "create_ticket",
    "send_email",
    "lookup_order",
    "run_sql",
    "translate",
)
LABELS = ("PERSON", "ORG", "LOCATION", "DATE", "PRODUCT", "EVENT", "MONEY")
SENTIMENTS = ("positive", "negative", "neutral", "mixed")
# Text a model writes inside JSON strings: apostrophes, inner quotes, newlines, tabs, backslashes, markup.
PHRASES = (
    "It's fine",
    "don't worry",
    'the user said "great job"',
    "line one\nline two",
    "a\ttabbed\tvalue",
    "C:\\Users\\alice\\Documents\\report.txt",
    "regex: \\d+\\.\\d*",
    "path/to/file.json",
    "<b>bold</b> & <i>italic</i>",
    "50% off: {not json}",
    "[1] see note",
    "she wrote: 'hello'",
    "ends with a backslash \\",
    "\\u0041 is not an escape here",
    "multi\n\nparagraph\n",
    "key: value, other: thing",
    "trailing space ",
    "",
)
UNICODE = (
    "d\u00e9j\u00e0 vu",
    "na\u00efve",
    "\u00fcber",
    "\u4e2d\u6587\u6587\u672c",
    "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8",
    "\ud55c\uad6d\uc5b4",
    "\u0627\u0644\u0639\u0631\u0628\u064a\u0629",
    "\u0420\u0443\u0441\u0441\u043a\u0438\u0439",
    "\u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac",
    "\u0939\u093f\u0928\u094d\u0926\u0940",
    "\U0001f389\U0001f44d\U0001f3fd",
    "\U0001f600 smile",
    "\u20ac100",
    "\N{EM DASH} dash",
    "\N{HORIZONTAL ELLIPSIS}",
    "\N{LEFT DOUBLE QUOTATION MARK}curly\N{RIGHT DOUBLE QUOTATION MARK}",
    "\N{NO-BREAK SPACE}nbsp",
    "\N{IDEOGRAPHIC SPACE}wide space",
    "x\N{ZERO WIDTH SPACE}y",
    "\N{LINE SEPARATOR}ls",
)
NUMBERS: tuple[Any, ...] = (
    0,
    1,
    -1,
    42,
    -17,
    2024,
    1_000_000,
    12345678901234567890,
    -9223372036854775809,
    0.5,
    -0.25,
    3.14159,
    -0.0,
    1e-07,
    6.02e23,
    1.5e300,
    100.0,
    2.5e-05,
)
# Characters for random mutations: structure, quotes of every kind, escapes, digits, literals, comments.
ALPHABET = (
    "{}[]:,\"'\\ \n\t0123456789.-+eE"
    "truefalsnTFN#/*`()_xu"
    "\u201c\u201d\u201e\u2018\u2019\u00ab\u00bb\u00a0\u3000\u00e9\u4e2d\U0001f600"
)


def _word(r: random.Random) -> str:
    return r.choice(WORDS)


def _sentence(r: random.Random, lo: int = 3, hi: int = 12) -> str:
    words = [_word(r) for _ in range(r.randint(lo, hi))]
    s = " ".join(words).capitalize() + r.choice((".", "!", "?", "", "..."))
    if r.random() < 0.35:
        s += " " + r.choice(PHRASES)
    if r.random() < 0.25:
        s += " " + r.choice(UNICODE)
    return s


def _string(r: random.Random) -> str:
    x = r.random()
    if x < 0.4:
        return _sentence(r)
    if x < 0.6:
        return r.choice(PHRASES)
    if x < 0.75:
        return r.choice(UNICODE)
    if x < 0.85:
        return r.choice(NAMES)
    return _word(r)


def _number(r: random.Random) -> Any:
    x = r.random()
    if x < 0.4:
        return r.choice(NUMBERS)
    if x < 0.7:
        return r.randint(-1000, 100_000)
    return round(r.uniform(-1000, 1000), r.randint(0, 6))


def _scalar(r: random.Random) -> Any:
    x = r.random()
    if x < 0.45:
        return _string(r)
    if x < 0.75:
        return _number(r)
    if x < 0.9:
        return r.random() < 0.5
    return None


def _key(r: random.Random) -> str:
    x = r.random()
    if x < 0.7:
        return _word(r) if r.random() < 0.6 else f"{_word(r)}_{_word(r)}"
    if x < 0.85:
        return _word(r).capitalize() + _word(r).capitalize()
    if x < 0.93:
        return r.choice(("id", "type", "name", "value", "data", "items", "error", "message"))
    return r.choice(
        ("user name", "e-mail", "x.y", "a:b", "d\u00e9j\u00e0 vu", "\u540d\u524d", "$ref", "@id", "")
    )


def _value(r: random.Random, depth: int) -> Any:
    if depth <= 0 or r.random() < 0.55:
        return _scalar(r)
    if r.random() < 0.5:
        return {_key(r): _value(r, depth - 1) for _ in range(r.randint(0, 5))}
    return [_value(r, depth - 1) for _ in range(r.randint(0, 5))]


# ------------------------------------------------------------------------------------------- documents


def _tool_call(r: random.Random) -> Any:
    args = {_key(r): _scalar(r) for _ in range(r.randint(1, 4))}
    if r.random() < 0.5:
        args["location"] = r.choice(CITIES)
        args["unit"] = r.choice(("celsius", "fahrenheit"))
    return {"name": r.choice(FUNCTIONS), "arguments": args}


def _chat_completion(r: random.Random) -> Any:
    calls = [
        {
            "id": f"call_{r.getrandbits(48):012x}",
            "type": "function",
            "function": {
                "name": r.choice(FUNCTIONS),
                # Arguments as a JSON *string*: escaped quotes inside the document.
                "arguments": json.dumps(_tool_call(r)["arguments"], ensure_ascii=r.random() < 0.5),
            },
        }
        for _ in range(r.randint(1, 2))
    ]
    return {
        "id": f"chatcmpl-{r.getrandbits(40):010x}",
        "object": "chat.completion",
        "created": 1_700_000_000 + r.randrange(10**7),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": None, "tool_calls": calls},
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": r.randint(5, 900), "completion_tokens": r.randint(1, 300)},
    }


def _records(r: random.Random) -> Any:
    out = []
    for i in range(r.randint(1, 6)):
        rec: dict[str, Any] = {
            "id": i + 1,
            "name": r.choice(NAMES),
            "email": f"{r.choice(NAMES).lower()}@example.com",
            "score": round(r.random(), r.randint(1, 4)),
            "tags": [_word(r) for _ in range(r.randint(0, 3))],
            "active": r.random() < 0.7,
            "manager": None if r.random() < 0.5 else r.randint(1, 9),
        }
        if r.random() < 0.3:
            rec["notes"] = _sentence(r)
        out.append(rec)
    return out if r.random() < 0.6 else {"results": out, "count": len(out), "next": None}


def _extraction(r: random.Random) -> Any:
    text = _sentence(r, 6, 16)
    ents = []
    for _ in range(r.randint(0, 4)):
        w = r.choice(NAMES + CITIES)
        start = r.randrange(len(text) + 1)
        ents.append({"text": w, "label": r.choice(LABELS), "start": start, "end": start + len(w)})
        if r.random() < 0.5:
            ents[-1]["confidence"] = round(r.random(), 3)
    return {"text": text, "entities": ents, "language": r.choice(("en", "fr", "zh", "und"))}


def _classification(r: random.Random) -> Any:
    return {
        "label": r.choice(SENTIMENTS),
        "confidence": round(r.random(), 2),
        "reasoning": _sentence(r, 8, 20),
        "flags": {"toxic": r.random() < 0.1, "spam": r.random() < 0.1, "language": "en"},
        "scores": {s: round(r.random(), 3) for s in SENTIMENTS},
    }


def _code(r: random.Random) -> Any:
    snippets = (
        'def greet(name):\n    print(f"Hello, {name}!")\n    return {"ok": True}\n',
        "SELECT name, COUNT(*) FROM users WHERE city = 'Paris' GROUP BY name;",
        'const x = {"a": [1, 2, 3]};\nconsole.log(`value: ${x.a[0]}`);',
        "if (a < b && c > d) { return '\\n'; }",
        'echo "path: C:\\\\temp" | grep -E "\\d{3}"',
    )
    return {
        "language": r.choice(("python", "sql", "javascript", "bash")),
        "code": r.choice(snippets),
        "explanation": _sentence(r),
    }


def _config(r: random.Random) -> Any:
    return {_key(r): _value(r, 3) for _ in range(r.randint(1, 6))} | {"empty": r.choice(({}, [], "", None))}


def _numbers(r: random.Random) -> Any:
    return {_word(r): _number(r) for _ in range(r.randint(2, 8))} | {"values": [_number(r) for _ in range(4)]}


def _i18n(r: random.Random) -> Any:
    return {
        "title": r.choice(UNICODE),
        "body": " ".join(r.choice(UNICODE) for _ in range(r.randint(1, 4))),
        "translations": {lang: r.choice(UNICODE) for lang in ("fr", "zh", "ja", "ar")},
    }


def _toplevel(r: random.Random) -> Any:
    x = r.random()
    if x < 0.4:
        return [f"Step {i + 1}: {_sentence(r, 2, 6)}" for i in range(r.randint(1, 5))]
    if x < 0.6:
        return _string(r)
    if x < 0.8:
        return _number(r)
    return [_scalar(r) for _ in range(r.randint(0, 6))]


DOCUMENTS: tuple[tuple[Callable[[random.Random], Any], float], ...] = (
    (_tool_call, 0.14),
    (_chat_completion, 0.08),
    (_records, 0.16),
    (_extraction, 0.1),
    (_classification, 0.1),
    (_code, 0.07),
    (_config, 0.14),
    (_numbers, 0.07),
    (_i18n, 0.07),
    (_toplevel, 0.07),
)


def _pick(r: random.Random, weighted: tuple[tuple[Any, float], ...]) -> Any:
    x, acc = r.random() * sum(w for _, w in weighted), 0.0
    for item, w in weighted:
        acc += w
        if x < acc:
            return item
    return weighted[-1][0]


def _serialize(r: random.Random, doc: Any) -> str:
    x = r.random()
    if x < 0.06:
        return repr(doc)  # Python literal: single quotes, True / False / None
    indent = r.choice((None, None, 2, 4, "\t", 1))
    seps = (",", ":") if indent is None and r.random() < 0.3 else None
    return json.dumps(doc, indent=indent, separators=seps, ensure_ascii=r.random() < 0.3)


# -------------------------------------------------------------------------------------------- breakage

_ESC = re.compile(r"\\")
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_LIT = re.compile(r"\b(?:true|false|null)\b")
_KEY = re.compile(r'"([A-Za-z_][A-Za-z0-9_ ]*)"(\s*):')
_STR = re.compile(r'"(?:[^"\\\n]|\\.)*"')


def _positions(text: str, pattern: re.Pattern[str], where: Callable[[re.Match[str]], int]) -> list[int]:
    return [where(m) for m in pattern.finditer(text)]


def truncate(r: random.Random, text: str) -> str:
    """A streaming prefix, cut at a random or a sensitive position."""
    if len(text) < 2:
        return text
    kind = r.random()
    spots: list[int] = []
    if kind < 0.15:
        spots = [m.end() for m in _ESC.finditer(text)]  # right after a backslash
    elif kind < 0.25:
        spots = [m.start() + 2 + r.randrange(4) for m in re.finditer(r"\\u", text)]  # inside \uXXXX
    elif kind < 0.4:
        spots = _positions(text, _NUM, lambda m: m.start() + 1 + r.randrange(max(1, len(m.group()))))
    elif kind < 0.5:
        spots = _positions(text, _LIT, lambda m: m.start() + 1 + r.randrange(len(m.group()) - 1))
    elif kind < 0.62:
        spots = [i + 1 for i, c in enumerate(text) if c in ":,[{"]
    elif kind < 0.72:
        spots = [m.start() + 1 for m in _STR.finditer(text)]  # just after an opening quote
    cut = r.choice(spots) if spots else r.randrange(1, len(text))
    cut = max(1, min(cut, len(text) - 1))
    out = text[:cut]
    if r.random() < 0.1:
        out += r.choice((" ", "\n", "\n\n", "  \n"))
    return out


def mutate(r: random.Random, text: str) -> str:
    """1-4 random character deletions, insertions or replacements."""
    s = list(text)
    for _ in range(r.randint(1, 4)):
        op = r.random()
        if not s or op < 0.34:
            s.insert(r.randrange(len(s) + 1), r.choice(ALPHABET))
        elif op < 0.67:
            del s[r.randrange(len(s))]
        else:
            s[r.randrange(len(s))] = r.choice(ALPHABET)
    return "".join(s)


def _replace_nth(r: random.Random, text: str, old: str, new: str) -> str:
    idx = [m.start() for m in re.finditer(re.escape(old), text)]
    if not idx:
        return text
    i = r.choice(idx)
    return text[:i] + new + text[i + len(old) :]


def _sub_some(
    r: random.Random, pattern: re.Pattern[str], repl: Callable[[re.Match[str]], str], text: str
) -> str:
    p = r.choice((1.0, 0.5, 0.2))
    return pattern.sub(lambda m: repl(m) if r.random() < p else m.group(), text)


QUOTE_STYLES = (
    ("'", "'"),
    ("\u201c", "\u201d"),
    ("\u201d", "\u201d"),
    ("\u201e", "\u201c"),
    ("\u2018", "\u2019"),
    ("`", "`"),
    ("\u00ab", "\u00bb"),
)


def _swap_quotes(r: random.Random, text: str) -> str:
    left, right = r.choice(QUOTE_STYLES)
    return _sub_some(r, _STR, lambda m: left + m.group()[1:-1] + right, text)


def _comment(r: random.Random, text: str) -> str:
    lines = text.split("\n")
    i = r.randrange(len(lines))
    style = r.random()
    if style < 0.35:
        lines[i] += "  # " + _sentence(r, 1, 4)
    elif style < 0.7:
        lines[i] += " // " + _sentence(r, 1, 4)
    elif style < 0.9:
        return _replace_nth(r, text, ",", ", /* " + _sentence(r, 1, 3) + " */")
    else:
        lines.insert(i, r.choice(("// ", "# ", "/* ")) + _sentence(r, 1, 5) + r.choice(("", " */")))
    return "\n".join(lines)


def _duplicate_key(r: random.Random, text: str) -> str:
    keys = [m.group(1) for m in _KEY.finditer(text)]
    if not keys:
        return text
    return _replace_nth(r, text, "{", "{" + json.dumps(r.choice(keys)) + ": " + json.dumps(_scalar(r)) + ", ")


def _inner_quotes(r: random.Random, text: str) -> str:
    return _replace_nth(r, text, '\\"', '"') if '\\"' in text else _replace_nth(r, text, "'", '"')


def _odd_number(r: random.Random, text: str) -> str:
    forms = ("NaN", "Infinity", "-Infinity", "undefined", "1_000", "1,234", "1/2", ".5", "-.5", "1.", "007",
             "+1", "1e", "0x1F", "1.2.3", "10-20", "\u00b2", "\u0663")  # fmt: skip
    nums = list(_NUM.finditer(text))
    if not nums:
        return text
    m = r.choice(nums)
    return text[: m.start()] + r.choice(forms) + text[m.end() :]


def _ellipsis(r: random.Random, text: str) -> str:
    return _replace_nth(r, text, "]", r.choice((", ...]", "...]", ", \u2026]", " // more items\n]")))


STRUCTURAL: tuple[tuple[str, Callable[[random.Random, str], str]], ...] = (
    ("drop-quote", lambda r, t: _replace_nth(r, t, '"', "")),
    ("drop-open-brace", lambda r, t: _replace_nth(r, t, "{", "")),
    ("drop-close-brace", lambda r, t: _replace_nth(r, t, "}", "")),
    ("drop-bracket", lambda r, t: _replace_nth(r, t, r.choice("[]"), "")),
    ("drop-comma", lambda r, t: _replace_nth(r, t, ",", "")),
    ("drop-colon", lambda r, t: _replace_nth(r, t, ":", r.choice(("", " ")))),
    ("quotes", _swap_quotes),
    ("trailing-comma", lambda r, t: _replace_nth(r, t, r.choice("}]"), "," + r.choice("}]"))),
    ("unquoted-keys", lambda r, t: _sub_some(r, _KEY, lambda m: m.group(1) + m.group(2) + ":", t)),
    (
        "python-literals",
        lambda r, t: _sub_some(
            r, _LIT, lambda m: {"true": "True", "false": "False"}.get(m.group(), "None"), t
        ),
    ),
    ("comment", _comment),
    ("duplicate-key", _duplicate_key),
    ("inner-quotes", _inner_quotes),
    ("raw-newline", lambda r, t: _replace_nth(r, t, "\\n", "\n")),
    ("odd-number", _odd_number),
    ("ellipsis", _ellipsis),
    ("extra-closer", lambda r, t: t + r.choice(("}", "]", "}}", "]}", "\n}"))),
    ("missing-closers", lambda r, t: t.rstrip().rstrip("}]") if t.rstrip()[-1:] in "}]" else t[:-1]),
    ("empty-value", lambda r, t: re.sub(r":\s*[^,{}\[\]\n]+", ": ", t, count=1)),
    ("lowercase-literals", lambda r, t: t.replace("true", r.choice(("TRUE", "True", "tRUE")))),
)


def structural(r: random.Random, text: str) -> tuple[str, list[str]]:
    names = []
    for _ in range(r.randint(1, 3)):
        name, fn = r.choice(STRUCTURAL)
        text, names = fn(r, text), [*names, name]
    return text, names


PROSE_BEFORE = (
    "Here is the JSON you asked for:\n\n",
    "Sure! Here's the result:\n",
    "```json\n",
    "Output:\n",
    "json\n",
    "JSON: ",
    "Based on the input, the answer is ",
    "I found the following information.\n\n```\n",
    "\ufeff",
    "   \n",
)
PROSE_AFTER = (
    "\n\nLet me know if you need anything else!",
    "\n```",
    "\n```\n\nThe `score` field is between 0 and 1.",
    " (this is an estimate)",
    "\n\nNote: values may be approximate; I'm not 100% sure.",
    "\n",
    "",
)


def wrap(r: random.Random, text: str) -> tuple[str, str]:
    x = r.random()
    if x < 0.35:
        fence = r.choice(("```json", "```JSON", "``` json", "```javascript", "```"))
        close = r.choice(("\n```", "\n```\n", "", "```"))
        before, after = r.choice((*PROSE_BEFORE[:5], "")), r.choice(PROSE_AFTER)
        return before + fence + "\n" + text + close + after, "fence"
    if x < 0.6:
        return r.choice(PROSE_BEFORE) + text + r.choice(PROSE_AFTER), "prose"
    if x < 0.7:
        return "(" + text + r.choice((")", ", " + text + ")", "")), "parens"
    if x < 0.78:
        return "<json>" + text + r.choice(("</json>", "")), "tag"
    if x < 0.86:
        return json.dumps(text), "quoted"  # a JSON document returned as a JSON string
    return text + r.choice(PROSE_AFTER), "suffix"


def multi(r: random.Random) -> str:
    docs = [_serialize(r, _pick(r, DOCUMENTS)(r)) for _ in range(r.randint(2, 3))]
    style = r.random()
    if style < 0.4:
        return "\n\n".join(f"Result {i + 1}:\n```json\n{d}\n```" for i, d in enumerate(docs))
    if style < 0.7:
        return "\n".join(d.replace("\n", "") for d in docs)  # JSON lines
    if style < 0.85:
        return "".join(docs)
    return ", ".join(docs)


RECIPES: tuple[tuple[str, float], ...] = (
    ("valid", 0.05),
    ("truncate", 0.2),
    ("mutate", 0.15),
    ("structural", 0.25),
    ("wrap", 0.1),
    ("wrap+structural", 0.1),
    ("wrap+truncate", 0.07),
    ("structural+truncate", 0.03),
    ("multi", 0.05),
)
MAX_CHARS = 4000


def case(i: int, seed: int = SEED) -> Case:
    """Case ``i`` of the corpus (independent of every other case)."""
    r = random.Random(seed * 1_000_003 + i)
    recipe = _pick(r, RECIPES)
    text = _serialize(r, _pick(r, DOCUMENTS)(r))[:MAX_CHARS]
    tags = [recipe]
    if recipe == "multi":
        text = multi(r)
    if "structural" in recipe:
        text, names = structural(r, text)
        tags += names
    if recipe.startswith("wrap"):
        text, how = wrap(r, text)
        tags.append(how)
    if recipe == "mutate":
        text = mutate(r, text)
    if "truncate" in recipe:
        text = truncate(r, text)
    return Case(f"{i:05d}", "/".join(tags), text)


def cases(n: int = DEFAULT_SIZE, seed: int = SEED) -> list[Case]:
    return [case(i, seed) for i in range(n)]


def inputs_sha256(cs: list[Case]) -> str:
    h = hashlib.sha256()
    for c in cs:
        h.update(c.id.encode() + b"\0" + c.text.encode("utf-8", "surrogatepass") + b"\0")
    return h.hexdigest()


# ------------------------------------------------------------------------------------- pinned outcomes


def outcome_digests(
    cs: list[Case], options: dict[str, dict[str, Any]], fn: Any = None
) -> dict[str, list[str]]:
    """Per option set, one sha256 (16 hex digits) per CHUNK cases over json_repair's encoded outcomes."""
    from pyroid_parity import json_repair_compare as C

    if fn is None:
        import json_repair

        fn = json_repair.repair_json
    out: dict[str, list[str]] = {}
    for name, kw in options.items():
        digests = []
        for start in range(0, len(cs), CHUNK):
            h = hashlib.sha256()
            for c in cs[start : start + CHUNK]:
                h.update(
                    json.dumps(C.capture(fn, c.text, **kw).to_json(), ensure_ascii=True).encode() + b"\n"
                )
            digests.append(h.hexdigest()[:16])
        out[name] = digests
    return out


def chunk_digest(outcomes: list[Any]) -> str:
    """The digest ``outcome_digests`` computes, from already encoded outcomes (``Outcome.to_json()``)."""
    h = hashlib.sha256()
    for o in outcomes:
        h.update(json.dumps(o, ensure_ascii=True).encode() + b"\n")
    return h.hexdigest()[:16]


def pin(n: int = DEFAULT_SIZE) -> dict[str, Any]:
    import importlib.metadata as md
    import platform

    from pyroid_parity.specs import JSON_REPAIR_OPTION_SETS

    cs = cases(n)
    return {
        "format": PIN_FORMAT,
        "json_repair": md.version("json-repair"),
        "python": platform.python_version(),
        "seed": SEED,
        "size": n,
        "chunk": CHUNK,
        "inputs_sha256": inputs_sha256(cs),
        "option_sets": JSON_REPAIR_OPTION_SETS,
        "chunks": outcome_digests(cs, JSON_REPAIR_OPTION_SETS),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pyroid_parity.json_repair_corpus")
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--show", nargs=2, type=int, metavar=("START", "STOP"))
    ap.add_argument("--pin", action="store_true", help=f"write {PIN.name} (json_repair outcome digests)")
    a = ap.parse_args(argv)
    if a.show:
        for i in range(*a.show):
            c = case(i)
            print(f"--- {c.id} {c.recipe}\n{c.text}")
    if a.stats:
        cs = cases(a.size)
        recipes = Counter(c.recipe.split("/")[0] for c in cs)
        lengths = sorted(len(c.text) for c in cs)
        valid = 0
        for c in cs:
            try:
                json.loads(c.text)
                valid += 1
            except ValueError:
                pass
        print(f"{len(cs)} cases, sha256 {inputs_sha256(cs)}")
        print(f"chars: median {lengths[len(lengths) // 2]}, max {lengths[-1]}, total {sum(lengths)}")
        print(f"json.loads accepts {valid} ({100 * valid / len(cs):.1f} %)")
        print("recipes:", dict(sorted(recipes.items())))
    if a.pin:
        p = pin(a.size)
        PIN.write_text(json.dumps(p, indent=1, ensure_ascii=True) + "\n", "utf-8")
        print(f"wrote {PIN} ({PIN.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
