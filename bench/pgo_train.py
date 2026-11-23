#!/usr/bin/env python3
"""PGO training workload: every pyroid module, called the way the benchmarks call it, on docs they never time.

    python bench/pgo_train.py --corpus-dir pgo/corpus                  # inputs from bench/pgo_corpus.py
    python bench/pgo_train.py --corpus-dir pgo/corpus --threads 1,4 --docs 2000

Run it once per interpreter the release wheels serve, with an instrumented build of pyroid installed
(rustc ``-Cprofile-generate``); ``llvm-profdata merge`` then combines the profiles. bench/pgo_build.sh and
.github/workflows/release.yml do both.

Each module runs on list[str] and Arrow input and single str calls, at every ``--threads`` value:
ftfy (FineWeb text, plus mojibake made from it), translit, fastText LID (.bin and .ftz), the Gopher, C4 and
FineWeb filters and the formatters, MinHash and exact and sentence dedup, pyroid.text, pyroid.jieba,
pyroid.json_repair (broken LLM output from its parity corpus generator, with another seed), pyroid.metrics
(ROUGE, BLEU, chrF), pyroid.datasketch, pyroid.yaml (generated documents loaded and dumped in several
styles), pyroid.tabulate (tables of numbers, FineWeb words and mixed cells in the common formats),
pyroid.dateutil (dates and times in common formats, valid and not), pyroid.jmespath (queries over a
boto3-style response and over records), pyroid.difflib (diffs of FineWeb lines, name ratios and close
matches), pyroid.packaging (versions, specifiers, markers and requirements), and the FineWeb-2 filter
and MinHash configs of every language pyroid tokenizes natively here (Japanese only with SudachiDict
installed).

Needs only pyroid and numpy. pyarrow is optional: without it, as on Windows ARM64, the Arrow calls are
skipped. Nothing is checked or timed: an instrumented build is never shipped.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import random
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parity"))  # pyroid_parity: stdlib-only specs
from pyroid_parity import json_repair_corpus as JC
from pyroid_parity import specs

# json_repair_corpus.SEED (645) makes the gated corpus; training uses another.
TRAIN_SEED = 1

try:
    import pyarrow as pa
except ImportError:
    pa = None


def read_jsonl(p: Path, n: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if n is not None and len(rows) >= n:
                break
    return rows


def kinds(texts: list[str]) -> list[Any]:
    """The input kinds the harness times: list[str], and an Arrow string array when pyarrow is there."""
    return [texts] + ([pa.array(texts, pa.string())] if pa is not None else [])


def mojibake(texts: list[str]) -> list[str]:
    """UTF-8 read as Latin-1 or Windows-1252, the mix ftfy is for, from the docs that have non-ASCII text."""
    out = []
    for i, t in enumerate(t for t in texts if not t.isascii()):
        b = t.encode("utf-8")
        out.append(b.decode("latin-1") if i % 2 else b.decode("cp1252", "replace"))
    return out


def broken_json(n: int) -> list[str]:
    """Broken LLM output from the json_repair parity corpus's generator, with a seed the suite never uses."""
    return [c.text for c in JC.cases(n, seed=TRAIN_SEED)]


def shingles(text: str, n: int = 5) -> list[bytes]:
    w = text.split()
    return [" ".join(w[i : i + n]).encode("utf-8") for i in range(max(len(w) - n + 1, 1))]


def native(lang: str) -> bool:
    import pyroid.filters as F

    try:
        F.apply(["Probe."], [F.GopherQuality(language=lang)])
    except NotImplementedError:
        return False
    return True


class Trainer:
    def __init__(self, corpus: Path, docs: int, per_doc: int, threads: list[int]) -> None:
        self.rows = read_jsonl(corpus / "fineweb.jsonl", docs)
        self.texts = [r["text"] for r in self.rows]
        self.per_doc = per_doc
        self.threads = threads
        self.corpus = corpus
        self.langs = {
            p.stem.removeprefix("fineweb2-"): [r["text"] for r in read_jsonl(p)]
            for p in sorted(corpus.glob("fineweb2-*.jsonl"))
        }
        self.timings: dict[str, float] = {}

    def section(self, name: str, fn: Callable[[], None]) -> None:
        t0 = time.perf_counter()
        fn()
        self.timings[name] = time.perf_counter() - t0
        print(f"  {name}: {self.timings[name]:.1f} s", flush=True)

    def ftfy(self) -> None:
        import pyroid.ftfy as FX

        bad = mojibake(self.texts)
        for t in self.threads:
            for docs in kinds(self.texts) + kinds(bad):
                FX.fix_text_batch(docs, threads=t)
        for s in self.texts[: self.per_doc] + bad[: self.per_doc]:
            FX.fix_text(s)
        for s in bad[:200]:
            FX.fix_encoding(s)
            FX.fix_and_explain(s)

    def translit(self) -> None:
        import pyroid.translit as T

        other = [t for texts in self.langs.values() for t in texts]
        for t in self.threads:
            for docs in kinds(self.texts) + kinds(other):
                T.anyascii(docs, threads=t)
                T.unidecode(docs, threads=t)
        for s in self.texts[: self.per_doc] + other[: self.per_doc]:
            T.anyascii(s)
            T.unidecode(s)

    def fasttext(self) -> None:
        import pyroid.fasttext as FT

        flat = [t.replace("\n", " ") for t in self.texts]
        other = [t.replace("\n", " ") for texts in self.langs.values() for t in texts]
        for name in specs.LID_MODELS:
            m = FT.load_model(str(self.corpus / name))
            for t in self.threads:
                for k in (1, 5):
                    m.predict(flat, k=k, threads=t)
                    m.predict(other, k=k, threads=t)
            for s in flat[: self.per_doc // 2] + other[: self.per_doc // 2]:
                m.predict(s)

    def filters(self) -> None:
        import pyroid.filters as F

        chain = [
            F.GopherRepetition(),
            F.GopherQuality(),
            F.C4Quality(filter_no_terminal_punct=False),
            F.FineWebQuality(),
        ]
        some = self.texts[: max(len(self.texts) // 5, 1)]
        for t in self.threads:
            for docs in kinds(self.texts):
                F.apply(docs, chain, threads=t)
            for docs in kinds(some):
                for fn, _, cfg in specs.FILTER_CASES.values():
                    getattr(F, fn)(docs, threads=t, **cfg)
                    getattr(F, f"{fn}_reasons")(docs, threads=t, **cfg)
                F.c4_lines(docs, threads=t)
                F.pii_replace(docs, threads=t)
                F.symbol_lines(docs, threads=t)
        for s in self.texts[: self.per_doc]:
            F.gopher_repetition_reasons(s)
            F.gopher_quality_reasons(s)
            F.c4_quality_reasons(s)
            F.fineweb_quality_reasons(s)

    def dedup(self) -> None:
        import pyroid.dedup as D

        some = self.texts[: max(len(self.texts) // 5, 1)]
        for t in self.threads:
            for docs in kinds(self.texts):
                D.minhash_signatures(docs, threads=t, **specs.MINHASH_CONFIG)
                D.exact_dedup(docs, threads=t)
                D.simplify_text(docs, threads=t)
            D.MinHashDedup(threads=t, **specs.MINHASH_CONFIG).fit(self.texts)
            D.sentence_dedup(some, threads=t)
        for s in self.texts[: self.per_doc]:
            D.minhash_signatures(s, **specs.MINHASH_CONFIG)

    def text(self) -> None:
        import pyroid.text as TX

        for t in self.threads:
            for docs in kinds(self.texts):
                TX.normalize_whitespace(docs, threads=t)
                TX.min_chars(docs, 500, threads=t)
        for s in self.texts[: self.per_doc]:
            TX.normalize_whitespace(s)
            TX.min_chars(s, 500)

    def multilingual(self) -> None:
        import pyroid.dedup as D
        import pyroid.filters as F

        build = {
            "gopher_repetition": F.GopherRepetition,
            "fineweb_quality": F.FineWebQuality,
            "gopher_quality": F.GopherQuality,
        }
        for lang, texts in self.langs.items():
            if not native(lang):
                print(f"    {lang}: not native here, skipped", flush=True)
                continue
            cases = specs.fineweb2_filter_cases(lang)
            steps = [build[fn](**cfg) for fn, _, cfg in cases.values() if fn in build]
            mh = {**specs.MINHASH_CONFIG_FINEWEB2, "language": lang}
            for t in self.threads:
                F.apply(texts, steps, threads=t)
                for fn, _, cfg in cases.values():
                    # datatrove's own errors (C4 on some scripts) are raised as datatrove raises them.
                    with contextlib.suppress(ValueError):
                        getattr(F, f"{fn}_reasons")(texts, threads=t, **cfg)
                D.minhash_signatures(texts, threads=t, **mh)
                D.minhash_signatures(texts, threads=t, **{**specs.MINHASH_CONFIG, "language": lang})

    def jieba(self) -> None:
        import pyroid.jieba as J

        zh = self.langs.get("cmn_Hani", [])
        for t in self.threads:
            for docs in kinds(zh):
                J.lcut(docs, threads=t)
                J.lcut_for_search(docs, threads=t)
                J.lcut(docs, HMM=False, threads=t)
        for s in zh[: self.per_doc // 4]:
            J.lcut(s)
            J.lcut(s, cut_all=True)
            J.lcut_for_search(s)
            list(J.tokenize(s))
            list(J.tokenize(s, mode="search"))

    def json_repair(self) -> None:
        import pyroid.json_repair as JR

        docs = broken_json(max(len(self.rows) // 2, 1))
        for t in self.threads:
            JR.repair_json_batch(docs, threads=t)
            JR.repair_json_batch(docs, return_objects=True, threads=t)
        for s in docs[: self.per_doc]:
            JR.repair_json(s)
            JR.loads(s)
            JR.repair_json(s, ensure_ascii=False, skip_json_loads=True)

    def metrics(self) -> None:
        from pyroid.metrics import rouge_scorer, sacrebleu

        n = max(len(self.texts) // 5, 1)
        hyp = [t[:300] for t in self.texts[:n]]
        ref = [t[150:450] or t for t in self.texts[:n]]
        zh = self.langs.get("cmn_Hani", [])
        for stem in (False, True):
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=stem)
            for t in self.threads:
                scorer.score_batch(ref, hyp, threads=t)
            for a, b in zip(ref[:200], hyp[:200], strict=True):
                scorer.score(a, b)
        for t in self.threads:
            sacrebleu.sentence_bleu_batch(hyp, [ref], threads=t)
            sacrebleu.sentence_chrf_batch(hyp, [ref], threads=t)
            sacrebleu.BLEU().corpus_score(hyp, [ref], threads=t)
            sacrebleu.BLEU(tokenize="intl").corpus_score(hyp, [ref], threads=t)
            sacrebleu.CHRF().corpus_score(hyp, [ref], threads=t)
            sacrebleu.CHRF(word_order=2).corpus_score(hyp, [ref], threads=t)
            if zh:
                sacrebleu.BLEU(tokenize="zh").corpus_score(
                    [s[:200] for s in zh], [[s[100:300] for s in zh]], threads=t
                )
        for a, b in zip(hyp[:200], ref[:200], strict=True):
            sacrebleu.sentence_bleu(a, [b])
            sacrebleu.sentence_chrf(a, [b])

    def datasketch(self) -> None:
        import pyroid.datasketch as DS

        n = max(len(self.texts) // 5, 1)
        sh = [shingles(t) for t in self.texts[:n]]
        for t in self.threads:
            for scheme in (None, "affine32"):
                kw = {"scheme": scheme} if scheme else {}
                ms = DS.MinHash.bulk(sh, num_perm=128, threads=t, **kw)
                sig = DS.minhash_batch(self.texts[:n], num_perm=128, ngram=5, threads=t, **kw)
            lsh = DS.MinHashLSH(threshold=0.9, num_perm=128)
            lsh.insert_batch(range(n), sig, threads=t)
            lsh.query_batch(sig, threads=t)
        lsh = DS.MinHashLSH(threshold=0.5, num_perm=128)
        for i, m in enumerate(ms[:500]):
            lean = DS.LeanMinHash(m)
            lsh.insert(i, lean)
            lsh.query(lean)
            m.jaccard(ms[0])

    def yaml(self) -> None:
        # bench/yaml/ would be imported as a namespace package named yaml; pyroid.yaml has to
        # see that PyYAML is not installed.
        here = Path(__file__).resolve().parent
        saved = sys.path[:]
        sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != here]
        try:
            import pyroid.yaml as Y
        finally:
            sys.path[:] = saved

        rnd = random.Random(TRAIN_SEED)
        words = [w for t in self.texts[:300] for w in t.split()[:30]] or ["x"]
        odd = ["yes", "no", "on", "1.0", "0x1F", "~", "", "a: b", "- x", "#c", "1e3", "2026-11-16", ".inf"]

        def value(depth: int) -> Any:
            k = rnd.randrange(10 if depth < 3 else 7)
            if k == 0:
                return rnd.randint(-(10**6), 10**6)
            if k == 1:
                return rnd.random() * 10 ** rnd.randint(-3, 6)
            if k == 2:
                return rnd.choice([True, False, None])
            if k == 3:
                return " ".join(rnd.choices(words, k=rnd.randint(1, 8)))
            if k == 4:
                return rnd.choice(words)
            if k in (5, 6):
                return rnd.choice(odd)
            if k in (7, 8):
                return {rnd.choice(words): value(depth + 1) for _ in range(rnd.randint(0, 6))}
            return [value(depth + 1) for _ in range(rnd.randint(0, 6))]

        docs = [value(0) for _ in range(3000)]
        styles: list[dict[str, Any]] = [{}, {"default_flow_style": True}, {"sort_keys": False}]
        styles += [{"allow_unicode": True, "indent": 4, "width": 40}, {"explicit_start": True}]
        texts = [Y.safe_dump(d, **styles[i % len(styles)]) for i, d in enumerate(docs)]
        stream = Y.safe_dump_all(docs[:1000])
        list(Y.safe_load_all(stream))
        list(Y.safe_load_all(stream.encode()))
        small = ["a: 1\n", "name: test\nversion: 1.2.3\ndeps: [a, b, c]\nopts: {x: true, y: null}\n" * 4]
        small += [Y.safe_dump(docs[:12])]
        # Single calls in the hundreds of thousands, as callers make them: with fewer, the
        # per-call path stays cold next to the rest of the profile.
        for _ in range(10):
            for t, d in zip(texts, docs, strict=True):
                Y.safe_load(t)
                Y.safe_dump(d)
        for _ in range(20_000):
            for s in small:
                Y.safe_dump(Y.safe_load(s))
                Y.load(s, Loader=Y.SafeLoader)
        for bad in ["a: [1, 2", "a: b: c", "\t- x", "key: 'open", "--- !!python/object:x {}", "&a [*b]"]:
            with contextlib.suppress(Y.YAMLError):
                Y.safe_load(bad)

    def tabulate(self) -> None:
        import pyroid.tabulate as TB

        rnd = random.Random(TRAIN_SEED)
        words = [w for t in self.texts[:500] for w in t.split()[:50]] or ["x"]

        def cell(kind: str, i: int, j: int) -> Any:
            if kind == "numbers":
                return i * 7 + j if j % 2 == 0 else rnd.random() * 10 ** rnd.randint(-4, 7)
            if kind == "text":
                if j % 3 == 2:
                    return str(rnd.randint(0, 10**7))
                return " ".join(rnd.choices(words, k=rnd.randint(1, 5)))
            mixed = [words[i % len(words)], -i, rnd.random() * 1000, None if i % 6 == 0 else "y" * (i % 4)]
            return [*mixed, f"{i}.5"][j % 5]

        formats = ["simple", "plain", "pipe", "github", "grid", "fancy_grid", "psql", "rst", "orgtbl", "html"]
        formats += ["latex", "tsv"]
        for n, cols in ((1, 3), (6, 4), (800, 8), (30_000, 9)):
            for kind in ("numbers", "text", "mixed"):
                rows = [[cell(kind, i, j) for j in range(cols)] for i in range(n)]
                headers = [f"h{j}" for j in range(cols)]
                for fmt in formats if n <= 800 else formats[:5]:
                    TB.tabulate(rows, headers=headers, tablefmt=fmt)
                TB.tabulate(rows, headers="firstrow", floatfmt=".3f", intfmt=",", missingval="-")
                TB.tabulate(rows, showindex=True, numalign="right", stralign="center")
        for _ in range(self.per_doc):
            TB.tabulate([[rnd.random(), rnd.randint(0, 99), rnd.choice(words)]], headers=["a", "b", "c"])

    def dateutil(self) -> None:
        from pyroid.dateutil import parser as P

        rnd = random.Random(TRAIN_SEED)
        formats = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y%m%dT%H%M%SZ"]
        formats += ["%Y-%m-%dT%H:%M:%S+05:30", "%d/%m/%Y", "%m/%d/%y %I:%M %p", "%d %B %Y", "%B %d, %Y %H:%M"]
        formats += ["%a, %d %b %Y %H:%M:%S +0000", "%A %d %b %Y", "%b %d %Y %I%p", "%Y.%m.%d", "%d-%b-%y"]
        formats += ["%H:%M:%S", "%Y-W%W", "%c"]
        start = datetime.datetime(1975, 1, 1)
        span = 60 * 365 * 86400
        stamps = [
            start + datetime.timedelta(seconds=rnd.randint(0, span), microseconds=rnd.randint(0, 10**6 - 1))
            for _ in range(20_000)
        ]
        dates = [d.strftime(formats[i % len(formats)]) for i, d in enumerate(stamps)]
        dates += ["not a date", "32/13/2020", "2020-02-30", "", "12:61", "Mon 31 Feb", "1e5"] * 50
        specs_ = ("auto", "seconds", "milliseconds")
        isos = [d.isoformat(sep=" " if i % 3 else "T", timespec=specs_[i % 3]) for i, d in enumerate(stamps)]
        isos += [d.date().isoformat() for d in stamps[:2000]] + ["2021-W05-3", "20210203T1504", "2021-13-01"]
        prose = [f"{t[:60]} {d}" for t, d in zip(self.texts, dates, strict=False)]
        for t in self.threads:
            P.parse_batch(dates, threads=t, errors="coerce")
            P.parse_batch(dates, threads=t, errors="coerce", dayfirst=True, ignoretz=True)
            P.parse_batch(prose[:2000], threads=t, errors="coerce", fuzzy=True)
            P.isoparse_batch(isos, threads=t, errors="coerce")
            if pa is not None:
                P.parse_batch(pa.array(dates), threads=t, errors="coerce", ignoretz=True)
                P.isoparse_batch(pa.array(isos[:5000]), threads=t, errors="coerce")
        # About two million single calls, as callers make them: with tens of thousands the
        # per-call path stays cold next to the rest of the profile.
        tzinfos = {"EST": -18000, "CET": 3600}
        for _ in range(50):
            for i, s in enumerate(dates):
                with contextlib.suppress(ValueError, OverflowError):
                    P.parse(s)
                if i % 8 == 0:
                    with contextlib.suppress(ValueError, OverflowError):
                        P.parse(s + " EST", tzinfos=tzinfos, yearfirst=True)
            for s in isos:
                with contextlib.suppress(ValueError, OverflowError):
                    P.isoparse(s)

    def jmespath(self) -> None:
        # This script's folder holds bench/jmespath/, which Python would import as a namespace
        # package named jmespath; pyroid.jmespath has to see that jmespath is not installed.
        here = Path(__file__).resolve().parent
        saved = sys.path[:]
        sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != here]
        try:
            import pyroid.jmespath as JM
        finally:
            sys.path[:] = saved

        rnd = random.Random(TRAIN_SEED)
        words = [w for t in self.texts[:200] for w in t.split()[:20]] or ["x"]
        states = ["running", "stopped", "pending"]
        response = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": f"i-{r:04x}{i:04x}",
                            "InstanceType": rnd.choice(["t3.micro", "m7g.large", "c7i.4xlarge"]),
                            "State": {"Name": rnd.choice(states), "Code": rnd.randint(0, 80)},
                            "LaunchTime": f"2026-{rnd.randint(1, 12):02d}-{rnd.randint(1, 28):02d}",
                            "Tags": [
                                {"Key": "Name", "Value": rnd.choice(words)},
                                {"Key": "env", "Value": "web-1"},
                            ],
                        }
                        for i in range(rnd.randint(1, 4))
                    ]
                }
                for r in range(12)
            ]
        }
        small = [
            "Reservations",
            "Reservations[0].Instances[0].State.Name",
            "Reservations[].Instances[].InstanceId",
            "Reservations[].Instances[?State.Name=='running'].InstanceId[]",
            "Reservations[].Instances[].{id: InstanceId, type: InstanceType}",
            "Reservations[].Instances[].Tags[?Key=='Name'].Value[]",
            "sort_by(Reservations[].Instances[], &LaunchTime)[-1].InstanceId",
            "!contains(Reservations[].Instances[].InstanceType, 'x') && length(Reservations) > `3`",
            "join(', ', Reservations[].Instances[].InstanceId)",
            "Reservations[*].Instances[*].State | [0][0]",
        ]
        # Single calls in the hundreds of thousands, as callers make them: with fewer, the
        # per-call path stays cold next to the rest of the profile.
        for e in small:
            compiled = JM.compile(e)
            for _ in range(50_000):
                compiled.search(response)
            for _ in range(20_000):
                JM.search(e, response)
        for i in range(20_000):
            e = f"Reservations[].Instances[?State.Name=='{states[i % 3]}' && Tags[?Value=='w{i}']]"
            JM.search(e, response)
        records = [
            {
                "id": i,
                "name": words[i % len(words)],
                "age": rnd.randint(1, 99),
                "score": rnd.random() * 100,
                "tags": ["a", "b", "c"][: i % 4],
                "address": {"city": rnd.choice(words), "zip": f"{i:06d}"},
            }
            for i in range(50_000)
        ]
        queries = [
            "[*].address.city",
            "[?age > `40`].name",
            "[?age > `40` && address.zip != '000001'].id",
            "[*].{n: name, c: address.city}",
            "sort_by(@, &age)[0].id",
            "max_by(@, &score).id",
            "min_by(@, &age).name",
            "sum([*].age)",
            "avg([*].score)",
            "[?contains(tags, 'a')] | length(@)",
            "[*].tags[]",
            "sort([*].name)[-1]",
            "reverse(@)[0].id",
            "[*].[id, age]",
            "@[?score < `50`] | [*].address",
        ]
        for e in queries:
            JM.compile(e).search(records)

    def difflib(self) -> None:
        import pyroid.difflib as D

        rnd = random.Random(TRAIN_SEED)
        words = [w for t in self.texts[:2000] for w in t.split()[:20] if w.isalpha()] or ["x"]
        names = list(dict.fromkeys(words))[:20_000]

        def typo(s: str) -> str:
            i = rnd.randrange(max(1, len(s)))
            return [s[:i] + s[i + 1 :], s[:i] + rnd.choice("aeiorst") + s[i:], s.upper(), s[::-1]][i % 4]

        def edited(xs: list[str]) -> list[str]:
            out = []
            for x in xs:
                k = rnd.randrange(10)
                if k == 0:
                    continue
                out.append(typo(x) + "\n" if k == 1 else x)
                if k == 2:
                    out.append(rnd.choice(xs))
            return out

        lines = [s + "\n" for t in self.texts[:3000] for s in t.split(". ") if s]
        files = [lines[i : i + rnd.randint(20, 300)] for i in range(0, len(lines), 300)][:200]
        for a in files:
            b = edited(a)
            list(D.unified_diff(a, b))
            list(D.context_diff(a, b, n=1))
            list(D.ndiff(a[:80], b[:80]))
            sm = D.SequenceMatcher(None, a, b)
            sm.get_opcodes()
            list(sm.get_grouped_opcodes())
        for a in files[:30]:
            b = edited(a)
            D.HtmlDiff().make_file(a[:100], b[:100])
            D.HtmlDiff(wrapcolumn=60).make_table(a[:60], b[:60], context=True)
        typos = [typo(n) for n in names]
        paras = [t[:500] for t in self.texts[:2000]]
        changed = [typo(typo(p)) for p in paras]
        for t in self.threads:
            D.ratio_batch(typos, names, threads=t)
            D.ratio_batch(paras, changed, threads=t)
            D.close_matches_batch(typos[:300], names[:5000], threads=t)
            if pa is not None:
                D.ratio_batch(pa.array(typos), pa.array(names), threads=t)
        # Single calls in the hundreds of thousands, as callers make them: with fewer, the
        # per-call path stays cold next to the rest of the profile.
        for _ in range(20):
            for x, y in zip(typos, names, strict=True):
                D.SequenceMatcher(None, x, y).ratio()
        for x, y in zip(typos[:50_000], names, strict=False):
            sm = D.SequenceMatcher(lambda c: c == " ", x, y)
            sm.quick_ratio()
            sm.get_matching_blocks()
            D.SequenceMatcher(None, x.encode(), y.encode()).ratio()
        for x, y in zip(paras, changed, strict=True):
            D.SequenceMatcher(None, x, y).ratio()
        for q in typos[:300]:
            D.get_close_matches(q, names[:10_000])

    def packaging(self) -> None:
        # bench/packaging/ would be imported as a namespace package named packaging; pyroid.packaging has to
        # see that packaging is not installed.
        here = Path(__file__).resolve().parent
        saved = sys.path[:]
        sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != here]
        try:
            import pyroid.packaging as RP
            from pyroid.packaging import markers as RM
            from pyroid.packaging import requirements as RR
            from pyroid.packaging import specifiers as RS
            from pyroid.packaging import utils as RU
            from pyroid.packaging import version as RV
        finally:
            sys.path[:] = saved

        rnd = random.Random(TRAIN_SEED)

        def version(i: int) -> str:
            v = ".".join(str(rnd.randint(0, 40)) for _ in range(rnd.randint(1, 4)))
            k = i % 11
            if k == 1:
                v += rnd.choice(["a", "b", "rc", "alpha", "c", "pre"]) + str(rnd.randint(0, 9))
            elif k == 2:
                v += f".post{rnd.randint(0, 9)}"
            elif k == 3:
                v += f".dev{rnd.randint(0, 9)}"
            elif k == 4:
                v += f"+local.{rnd.randint(0, 99)}"
            elif k == 5:
                v = f"{rnd.randint(1, 3)}!{v}"
            elif k == 6:
                v = "v" + v + "-1"
            elif k == 7:
                v += rnd.choice([" ", "-", "_"]) + "foo"
            return v

        versions = [version(i) for i in range(100_000)]
        specs = [">=1.0,<2", "~=1.4.2", "!=1.5.*", "==2.*", ">3,<=20.1", "===1.0", ">=4.2,!=4.3.*,<30"]
        specs += ["~=12.4", "", "<0.5.dev0", ">=1.0a1"]
        markers = ['python_version < "3.10"', 'sys_platform == "linux" and platform_machine != "aarch64"']
        markers += ["extra == 'test' or python_full_version >= '3.12.1'", 'os_name == "nt"']
        markers += ['implementation_name == "cpython" and python_version in "3.11 3.12"']
        reqs = ["requests>=2.8.1", "requests[security,socks]>=2.8.1,==2.8.*; python_version < '2.7'"]
        reqs += ["numpy (>=1.26)", "pip @ https://example.org/pip-24.0-py3-none-any.whl", "attrs"]
        reqs += ["foo[bar]~=1.2; extra == 'baz'", "Django>=4.2,<5.0", "zope.interface ; os_name != 'nt'"]
        for t in self.threads:
            RP.parse_batch(versions, errors="none", threads=t)
            valid = [v for v in RP.parse_batch(versions, errors="none", threads=t) if v is not None]
            RP.sort_versions(valid, threads=t)
            for s in specs:
                RP.contains_batch(s, valid, threads=t)
                RP.filter_batch(s, valid, threads=t)
            if pa is not None:
                RP.parse_batch(pa.array(versions), errors="none", threads=t)
                RP.contains_batch(specs[0], pa.array([str(v) for v in valid]), threads=t)
        # Single calls in the hundreds of thousands, as callers make them: with fewer, the
        # per-call path stays cold next to the rest of the profile.
        objs = []
        for _ in range(5):
            for s in versions:
                with contextlib.suppress(RV.InvalidVersion):
                    objs.append(RV.Version(s))
        sample = objs[:100_000]
        sorted(sample)
        set(sample)
        for v in sample:
            str(v)
            _ = (v.public, v.base_version, v.is_prerelease, v.release, v.local)
        parsed = RP.parse_batch(versions, errors="none")
        strs = [s for s, v in zip(versions, parsed, strict=True) if v is not None]
        sorted(strs[:50_000], key=RV.Version)
        for s in specs * 20_000:
            ss = RS.SpecifierSet(s)
            ss.contains(sample[rnd.randrange(len(sample))])
            ss.contains("1.6.2", prereleases=True)
        for m in markers * 20_000:
            RM.Marker(m).evaluate({"extra": "test"})
        for r in reqs * 20_000:
            str(RR.Requirement(r))
        for n in ("Foo.Bar_baz", "requests", "Zope-Interface", "a__b") * 50_000:
            RU.canonicalize_name(n)
        for s in versions[:50_000]:
            with contextlib.suppress(RV.InvalidVersion):
                RU.canonicalize_version(s)
        for f in ("pyroid-1.0.1-cp311-abi3-manylinux_2_34_x86_64.whl", "pip-24.0.tar.gz") * 20_000:
            with contextlib.suppress(RU.InvalidWheelFilename, RU.InvalidSdistFilename):
                RU.parse_wheel_filename(f) if f.endswith(".whl") else RU.parse_sdist_filename(f)

    def run(self) -> dict[str, float]:
        warnings.simplefilter("ignore")  # FallbackWarning and friends: the paths still run
        sections = ["ftfy", "translit", "fasttext", "filters", "dedup", "text", "multilingual", "jieba"]
        sections += ["json_repair", "metrics", "datasketch", "yaml", "tabulate", "dateutil", "jmespath"]
        sections += ["difflib", "packaging"]
        for name in sections:
            self.section(name, getattr(self, name))
        return self.timings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", type=Path, required=True, help="the folder bench/pgo_corpus.py wrote")
    ap.add_argument("--docs", type=int, default=10_000, help="FineWeb docs for the batch calls")
    ap.add_argument("--per-doc", type=int, default=1_000, help="docs for the single-str calls")
    ap.add_argument("--threads", default="1,8", help="comma-separated thread counts")
    a = ap.parse_args()
    import pyroid

    threads = [int(x) for x in a.threads.split(",")]
    print(
        f"pgo_train: pyroid {pyroid.__version__}, Python {sys.version.split()[0]}"
        f"{' free-threaded' if not getattr(sys, '_is_gil_enabled', lambda: True)() else ''},"
        f" threads {threads}, pyarrow {'yes' if pa is not None else 'no (Arrow calls skipped)'}",
        flush=True,
    )
    t0 = time.perf_counter()
    Trainer(a.corpus_dir, a.docs, a.per_doc, threads).run()
    print(f"pgo_train: done in {time.perf_counter() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
