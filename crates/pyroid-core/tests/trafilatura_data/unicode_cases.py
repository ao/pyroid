"""Unicode-version test vectors for pyroid_core::trafilatura (tests/trafilatura_unicode.rs).

    python3.15 -I unicode_cases.py build GEN_DIR inputs.json
    for v in 311 312 313 314 315; do
        /target/venv-traf$v/bin/python -I unicode_cases.py record inputs.json out-$v.json
    done
    python3 -I unicode_cases.py merge inputs.json out-*.json unicode.json

`build` (newest interpreter) picks the characters whose properties change between CPython
3.11 and 3.15, from the dumps of `src/trafilatura/generate_tables.py` in GEN_DIR, and builds:
strings for `str.lower()` (with the final sigma), `remove_control_characters`, NFC, and small
HTML pages whose extracted text depends on the version (new letters, digits, unassigned
characters, decomposed sequences). `record` runs CPython and trafilatura 2.3.1 on them;
`merge` writes the committed fixture: inputs and, per Unicode version, the outputs.

The pages of trafilatura 1.11.0 (unicode-1.11.0.json), on the same inputs:

    python3 -I unicode_cases.py inputs unicode.json inputs.json
    for v in 311 312 314 315; do
        /target/venv-t111-$v/bin/python -I unicode_cases.py record inputs.json out-$v.json
    done
    /target/venv-t111/bin/python -I unicode_cases.py record inputs.json out-313.json
    python3 -I unicode_cases.py merge inputs.json out-*.json unicode-1.11.0.json
"""

import json
import random
import sys
import unicodedata

VERSIONS = ["14.0.0", "15.0.0", "15.1.0", "16.0.0", "17.0.0"]
OPTION_SETS = {
    "default": {},
    "precision": {"favor_precision": True, "include_comments": False},
    "recall": {"favor_recall": True},
    "fast": {"fast": True},
}


def ranges_set(rs):
    out = set()
    for a, b in rs:
        out.update(range(a, b + 1))
    return out


def build(gen_dir, out):
    import glob

    dumps = {}
    for p in glob.glob(f"{gen_dir}/*.json"):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        dumps[r["unicode"]] = r
    assert sorted(dumps) == VERSIONS
    rnd = random.Random(20261109)
    printable = {v: ranges_set(dumps[v]["printable"]) for v in VERSIONS}
    changed_print = sorted(set.union(*printable.values()) - set.intersection(*printable.values()))
    lowers = {v: {int(k): x for k, x in dumps[v]["lower"].items()} for v in VERSIONS}
    changed_lower = sorted({k for v in VERSIONS for k in lowers[v]} - set.intersection(*(set(lowers[v]) for v in VERSIONS)))
    cased_any = sorted({int(k) for k in lowers["17.0.0"]})
    ign = sorted(ranges_set(dumps["17.0.0"]["case_ignorable"]))
    ign_changed = sorted(ranges_set(dumps["17.0.0"]["case_ignorable"]) - ranges_set(dumps["14.0.0"]["case_ignorable"]))
    sample = lambda xs, k: [xs[i] for i in sorted(rnd.sample(range(len(xs)), min(k, len(xs))))]
    pick = lambda xs: chr(rnd.choice(xs))

    lower_cases = []
    pool = changed_lower + sample(cased_any, 200)
    for _ in range(300):
        s = "".join(
            rnd.choice(
                [lambda: pick(pool), lambda: "Σ", lambda: pick(ign), lambda: pick(ign_changed or ign), lambda: " ",
                 lambda: rnd.choice("AbΣσ.")]
            )()
            for _ in range(rnd.randint(1, 8))
        )
        lower_cases.append(s)
    lower_cases += ["ΤΗΣ", "ΣΑ", "Σ", "ΑΣ.", "ΑΣΑ", "İSTANBUL", "Kelvin", "ΑΣͅ", "ᾼΣ"]

    print_cases = []
    ppool = sample(changed_print, 400)
    for _ in range(200):
        s = "".join(
            rnd.choice([lambda: pick(ppool), lambda: rnd.choice("ab \t\x01\x0c\x1c­​￾"), lambda: "x"])()
            for _ in range(rnd.randint(1, 10))
        )
        print_cases.append(s)

    # NFC: decompositions of characters assigned after 14.0
    assigned14 = ranges_set(dumps["14.0.0"]["printable"])
    nfc_cases = []
    for cp in range(0x110000):
        c = chr(cp)
        dec = unicodedata.decomposition(c)
        if dec and not dec.startswith("<") and cp not in assigned14 and cp > 0x7F:
            parts = "".join(chr(int(h, 16)) for h in dec.split())
            nfc_cases.append(unicodedata.normalize("NFD", parts))
            nfc_cases.append("a" + parts + "́b")
    nfc_cases += ["é", "ą́", "각"]

    # pages: version-dependent text inside otherwise ordinary articles
    filler = (
        "This paragraph is long enough to be kept by the extractor, with ordinary English words and "
        "enough of them for the length thresholds of every stage to pass without trouble at all. "
    )
    new_letters = [cp for cp in changed_print if unicodedata.category(chr(cp)).startswith("L")]
    new_digits = [cp for cp in changed_print if unicodedata.category(chr(cp)) == "Nd"]
    pages = []
    for k in range(12):
        nl = "".join(chr(rnd.choice(new_letters)) for _ in range(5))
        weird = "".join(chr(rnd.choice(ppool)) for _ in range(6))
        dig = chr(rnd.choice(new_digits)) if new_digits else "2"
        dec = rnd.choice(nfc_cases)
        body = [
            f"<p>{filler}{nl} {weird} {dec}</p>",
            f"<p>{filler}</p>",
            f"<ul><li>{nl}Facebook</li><li> Twitter</li><li>{weird}Print</li><li>{filler}</li></ul>",
            f"<table><tr><td colspan=\"{dig}\">{filler}</td><td>{nl}</td></tr><tr><td>a</td><td>b</td><td>c</td></tr></table>",
            f"<p>ΤΗΣ ΚΑΙ ΤΟ ΣΤΗΝ ΜΕ ΓΙΑ ΑΠΟ ΠΟΥ ΣΕ {nl} ΤΗΣ ΚΑΙ ΤΟ ΣΤΗΝ ΜΕ ΓΙΑ ΑΠΟ ΠΟΥ ΣΕ ΤΗΣ ΚΑΙ ΤΟ ΣΤΗΝ ΜΕ ΓΙΑ ΑΠΟ ΠΟΥ</p>",
        ]
        rnd.shuffle(body)
        if k % 3 == 0:
            ld = json.dumps({"@type": "Article", "articleBody": f"{filler}{nl} {weird} &lt;p&gt;{dec}&lt;/p&gt;" * 2})
            head = f'<script type="application/ld+json">{ld}</script>'
            main = f"<div><span>{nl}{weird}</span></div>"
        else:
            head = ""
            main = f"<article>{''.join(body)}</article>"
        pages.append(f"<html><head><title>t</title>{head}</head><body><nav>menu</nav>{main}<footer>f</footer></body></html>")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"lower": lower_cases, "printable": print_cases, "nfc": nfc_cases, "pages": pages}, f, ensure_ascii=True)
    print(f"{len(lower_cases)} lower, {len(print_cases)} printable, {len(nfc_cases)} nfc, {len(pages)} pages -> {out}")


def record(inp, out):
    import trafilatura
    from trafilatura.utils import remove_control_characters

    v1 = trafilatura.__version__.startswith("1.")
    with open(inp, encoding="utf-8") as f:
        cases = json.load(f)

    def kw(k):
        # 1.11.0 calls `fast` no_fallback
        return {("no_fallback" if v1 and a == "fast" else a): b for a, b in OPTION_SETS[k].items()}

    rec = {
        "unicode": unicodedata.unidata_version,
        "python": sys.version.split()[0],
        "trafilatura": trafilatura.__version__,
        "lower": [s.lower() for s in cases["lower"]],
        "printable": [remove_control_characters(s) for s in cases["printable"]],
        "nfc": [unicodedata.normalize("NFC", s) for s in cases["nfc"]],
        "pages": {k: [trafilatura.extract(p, **kw(k)) for p in cases["pages"]] for k in OPTION_SETS},
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=True)
    print(f"trafilatura {rec['trafilatura']}, {rec['python']} (Unicode {rec['unicode']}) -> {out}")


def merge(inp, outs, final):
    """The fixture; for trafilatura 1.11.0 (unicode-1.11.0.json) only the pages, as the string
    vectors are CPython's alone and in unicode.json already."""
    with open(inp, encoding="utf-8") as f:
        cases = json.load(f)
    per = {}
    for p in outs:
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        per[r["unicode"]] = r
    assert sorted(per) == VERSIONS, sorted(per)
    v1 = all(r.get("trafilatura", "2").startswith("1.") for r in per.values())
    keys = ("pages",) if v1 else ("lower", "printable", "nfc", "pages")
    per = {v: {k: r[k] for k in ("python", *keys)} for v, r in per.items()}
    if v1:
        cases = {"pages": cases["pages"]}
    for key in keys[:-1]:
        n = sum(len({json.dumps(per[v][key][i]) for v in VERSIONS}) > 1 for i in range(len(cases[key])))
        print(f"{key}: {n} of {len(cases[key])} differ between versions")
    for k in OPTION_SETS:
        n = sum(len({json.dumps(per[v]["pages"][k][i]) for v in VERSIONS}) > 1 for i in range(len(cases["pages"])))
        print(f"pages/{k}: {n} of {len(cases['pages'])} differ between versions")
    with open(final, "w", encoding="utf-8") as f:
        json.dump({"inputs": cases, "outputs": per}, f, ensure_ascii=True, indent=0)


def inputs(fixture, out):
    "The inputs of a committed fixture (to record another trafilatura release on them)."
    with open(fixture, encoding="utf-8") as f:
        cases = json.load(f)["inputs"]
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=True)


if __name__ == "__main__":
    {"build": lambda: build(sys.argv[2], sys.argv[3]),
     "record": lambda: record(sys.argv[2], sys.argv[3]),
     "inputs": lambda: inputs(sys.argv[2], sys.argv[3]),
     "merge": lambda: merge(sys.argv[2], sys.argv[3:-1], sys.argv[-1])}[sys.argv[1]]()
