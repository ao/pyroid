"""pyroid.langdetect against langdetect 1.0.9 on real text: FineWeb and FineWeb-2.

For every profile language, the first ``--docs`` documents of a FineWeb-2 file (FineWeb for
English) are detected as they are and as derived short texts (first 50 characters, a tweet,
a title, a 3-word query) and mixed-language texts, with seed 0, and the first ``--seeded``
documents again with seeds 1..``--seeds``. Each check compares ``detect`` and ``detect_langs``
(language and ``float.hex`` of every probability, or the exception's type, message and code)
between langdetect and pyroid in the same worker process, and ``detect_batch`` /
``detect_langs_batch`` of each language's texts against pyroid's own per-call results.

    python parity/langdetect_corpus.py --fw2 /target/fw2 --fineweb /data/fineweb/fineweb-100k.parquet \\
        --out /target/langdetect-parity.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from typing import Any

# langdetect code -> FineWeb-2 subset (English: FineWeb).
LANGS = {
    "af": "afr_Latn", "ar": "arb_Arab", "bg": "bul_Cyrl", "bn": "ben_Beng", "ca": "cat_Latn",
    "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn", "de": "deu_Latn", "el": "ell_Grek",
    "en": "fineweb", "es": "spa_Latn", "et": "ekk_Latn", "fa": "fas_Arab", "fi": "fin_Latn",
    "fr": "fra_Latn", "gu": "guj_Gujr", "he": "heb_Hebr", "hi": "hin_Deva", "hr": "hrv_Latn",
    "hu": "hun_Latn", "id": "ind_Latn", "it": "ita_Latn", "ja": "jpn_Jpan", "kn": "kan_Knda",
    "ko": "kor_Hang", "lt": "lit_Latn", "lv": "lvs_Latn", "mk": "mkd_Cyrl", "ml": "mal_Mlym",
    "mr": "mar_Deva", "ne": "npi_Deva", "nl": "nld_Latn", "no": "nob_Latn", "pa": "pan_Guru",
    "pl": "pol_Latn", "pt": "por_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl", "sk": "slk_Latn",
    "sl": "slv_Latn", "so": "som_Latn", "sq": "als_Latn", "sv": "swe_Latn", "sw": "swh_Latn",
    "ta": "tam_Taml", "te": "tel_Telu", "th": "tha_Thai", "tl": "fil_Latn", "tr": "tur_Latn",
    "uk": "ukr_Cyrl", "ur": "urd_Arab", "vi": "vie_Latn", "zh-cn": "cmn_Hani", "zh-tw": "yue_Hani",
}
EXTRA = ["", " ", " \t\n ", "1234 5678", "http://example.com/x", "a@b.cd", "😀😀😀", "　", "...", "\ud800"]


def variants(doc: str, other: str) -> dict[str, str]:
    words = doc.split()
    mid = len(words) // 2
    first_line = doc.strip().split("\n", 1)[0]
    return {
        "whole": doc,
        "short50": doc[:50],
        "tweet": doc[:140],
        "title": first_line[:80],
        "query": " ".join(words[mid : mid + 3]),
        "mixed": doc[:300] + " " + other[:300],
    }


_LD: Any = None
_PL: Any = None


def _init() -> None:
    global _LD, _PL
    import langdetect

    import pyroid.langdetect

    langdetect.DetectorFactory.seed = 0
    pyroid.langdetect.DetectorFactory.seed = 0
    _LD, _PL = langdetect, pyroid.langdetect


def _outcome(fn: Any, text: str) -> Any:
    try:
        r = fn(text)
    except Exception as exc:
        return ["err", type(exc).__name__, str(exc), getattr(exc, "code", None)]
    if isinstance(r, list):
        return [[x.lang, x.prob.hex()] for x in r]
    return r


def _check(item: tuple[str, str, int, str]) -> tuple[str, str, bool, bool, Any, float, float]:
    lang, variant, seed, text = item
    _LD.DetectorFactory.seed = seed
    _PL.DetectorFactory.seed = seed
    t0 = time.perf_counter()
    a = (_outcome(_LD.detect, text), _outcome(_LD.detect_langs, text))
    t1 = time.perf_counter()
    b = (_outcome(_PL.detect, text), _outcome(_PL.detect_langs, text))
    t2 = time.perf_counter()
    detail = None if a == b else {"seed": seed, "text": text[:200], "langdetect": a, "pyroid": b}
    return lang, variant, a[0] == b[0], a[1] == b[1], detail, t1 - t0, t2 - t1


def _batch(item: tuple[str, list[str]]) -> tuple[str, int, int]:
    lang, texts = item
    _PL.DetectorFactory.seed = 0
    one = [_outcome(_PL.detect_langs, t) for t in texts]
    many = _PL.detect_langs_batch(texts, errors="none")
    langs = [None if r is None else [[x.lang, x.prob.hex()] for x in r] for r in many]
    expect = [None if isinstance(r, list) and r[:1] == ["err"] else r for r in one]
    det = _PL.detect_batch(texts, errors="none")
    det_expect = [None if r is None else (r[0][0] if r else "unknown") for r in expect]
    bad = sum(x != y for x, y in zip(langs, expect)) + sum(x != y for x, y in zip(det, det_expect))
    return lang, len(texts), bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fw2", required=True)
    ap.add_argument("--fineweb", required=True)
    ap.add_argument("--docs", type=int, default=2000)
    ap.add_argument("--seeded", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import pyarrow.parquet as pq

    docs: dict[str, list[str]] = {}
    missing = []
    for code, subset in LANGS.items():
        path = args.fineweb if subset == "fineweb" else os.path.join(args.fw2, f"{subset}.parquet")
        if not os.path.exists(path):
            missing.append(subset)
            continue
        pf = pq.ParquetFile(path)
        texts: list[str] = []
        for batch in pf.iter_batches(columns=["text"], batch_size=1024):
            texts.extend(batch.column(0).to_pylist())
            if len(texts) >= args.docs:
                break
        docs[code] = texts[: args.docs]
    codes = sorted(docs)
    items: list[tuple[str, str, int, str]] = []
    batches: list[tuple[str, list[str]]] = []
    for k, code in enumerate(codes):
        other = docs[codes[(k + 1) % len(codes)]]
        per_lang: list[str] = []
        for i, doc in enumerate(docs[code]):
            for variant, text in variants(doc, other[i % len(other)]).items():
                items.append((code, variant, 0, text))
                per_lang.append(text)
                if i < args.seeded and variant in ("whole", "short50"):
                    items.extend((code, variant, s, text) for s in range(1, args.seeds + 1))
        batches.append((code, per_lang))
    items.extend(("extra", "extra", s, t) for t in EXTRA for s in range(5))
    t0 = time.time()
    counts: Counter[tuple[str, str, str]] = Counter()
    mismatches: list[Any] = []
    secs = [0.0, 0.0]
    with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:
        for lang, variant, same_detect, same_langs, detail, ta, tb in pool.imap_unordered(
            _check, items, chunksize=64
        ):
            counts[(lang, variant, "n")] += 1
            counts[(lang, variant, "detect")] += same_detect
            counts[(lang, variant, "detect_langs")] += same_langs
            secs[0] += ta
            secs[1] += tb
            if detail is not None and len(mismatches) < 200:
                mismatches.append({"lang": lang, "variant": variant, **detail})
        batch_results = pool.map(_batch, batches)
    total = sum(v for (l, v_, k), v in counts.items() if k == "n")
    same = sum(v for (l, v_, k), v in counts.items() if k == "detect_langs")
    report = {
        "python": sys.version,
        "checks": total,
        "detect_equal": sum(v for (l, v_, k), v in counts.items() if k == "detect"),
        "detect_langs_equal": same,
        "batch_texts": sum(n for _, n, _ in batch_results),
        "batch_mismatches": sum(b for _, _, b in batch_results),
        "languages": codes,
        "missing_subsets": missing,
        "per_language": {
            f"{l}/{v}": [counts[(l, v, "n")], counts[(l, v, "detect")], counts[(l, v, "detect_langs")]]
            for (l, v, k) in sorted(counts)
            if k == "n"
        },
        "mismatches": mismatches,
        "seconds_langdetect_cpu": round(secs[0], 1),
        "seconds_pyroid_cpu": round(secs[1], 1),
        "wall_seconds": round(time.time() - t0, 1),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_language", "mismatches")}))
    print("mismatches:", len(mismatches))
    for m in mismatches[:5]:
        print(json.dumps(m, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
