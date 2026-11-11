#!/usr/bin/env python3
"""Download the pinned benchmark/parity inputs into ``$PYROID_DATA`` (default ``/data``).

    cd bench && uv run --frozen python fetch.py   # fetch + verify the inputs
    ... python fetch.py --verify                 # verify the inputs only (no network)
    ... python fetch.py --verify-all             # verify every pinned file, parity/ references included
                                                 # (the opt-in fineweb2-train files only with --only)
    ... python fetch.py --pin                    # first fetch / intentional update: write hashes to manifest.json

Layout under /data (total budget 1.5 GB):
    fineweb/fineweb-100k.parquet       headline slice: first 100,000 rows of FineWeb sample/10BT/000_00000.parquet
    fineweb/fineweb-dev-10k.parquet    dev slice: the first 10,000 of those rows
    fineweb2/<lang>.parquet            first rows of FineWeb-2 test splits (non-English adversarial input)
    fineweb2/<lang>-train-20k.parquet  first 20,000 rows of a FineWeb-2 train file (--only fineweb2-train)
    models/lid.176.bin, lid.176.ftz    fastText language-ID models (CC BY-SA 3.0)
    metrics/cnndm-3.0.0-test.parquet   CNN/DailyMail 3.0.0 test split, as published (pyroid.metrics ROUGE parity)
    metrics/summeval.parquet           SummEval: 16 systems' summaries of 100 CNN/DM articles, 11 references each
    metrics/wmt21.parquet              WMT21 newstest system outputs + references, 8 language pairs (BLEU/chrF)
    trafilatura/CC-MAIN-*.parquet      Common Crawl pages (id, url, html) of 3 crawls, as datatrove hands them to
                                       Trafilatura: WARC byte ranges through WarcReader + URLFilter (~150 MB)
    trafilatura/upstream-tests-{2.3.1,1.11.0}.parquet   Trafilatura's tests/ at v2.3.1 and v1.11.0 (path, git blob
                                       sha1, bytes)
    warc/cc/*.gz                       Common Crawl byte ranges (~80 MB): WARC files of 16 crawls (2013-2026), WET, WAT, the
                                       2012 ARC crawl, truncated and empty ones (pyroid_parity.warc_corpus)
    warc/warcio-1.8.2/*                warcio's test files at v1.8.2 (Apache-2.0)

The parity/ files in the manifest are generated, not fetched: ``python -m pyroid_parity.refs`` writes them and
verifies what it wrote, so fetch.py checks them only with ``--verify-all``.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from pyroid_parity import data as D

FINEWEB = {
    "repo": "HuggingFaceFW/fineweb",
    "revision": "9bb295ddab0e05d785b879661af7260fed5140fc",
    "file": "sample/10BT/000_00000.parquet",
    "license": "ODC-By 1.0",
}
# Content sha256 over id\0text\0...: reproducible across parquet writers.
EXPECTED_CONTENT = {
    100_000: "1a93378229bbed5bcfa552ef67c9068d0c8d70f56204d40ac62c41fb3616567c",
    5_000: "f227def5ef38b53612a8ce2d5a968bd4da21809c714535912599f0388e20a949",
}
FINEWEB2 = {
    "repo": "HuggingFaceFW/fineweb-2",
    "revision": "af9c13333eb981300149d5ca60a8e9d659b276b9",
    "license": "ODC-By 1.0",
    "rows": 250,
    "langs": [
        "fra_Latn",
        "deu_Latn",
        "spa_Latn",
        "rus_Cyrl",
        "cmn_Hani",
        "jpn_Jpan",
        "arb_Arab",
        "hin_Deva",
        "vie_Latn",
        "tha_Thai",
    ],
}
# Larger inputs for the uncommitted word-tokenizer recordings (fetched only with --only fineweb2-train):
# the first rows of the first train file, at the same revision.
FINEWEB2_TRAIN = {"rows": 20_000, "langs": ["vie_Latn", "tha_Thai"]}
TRAIN_SUFFIX = f"-train-{FINEWEB2_TRAIN['rows'] // 1000}k.parquet"
MODELS = {
    "models/lid.176.bin": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
    "models/lid.176.ftz": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
}
# Eval data for the pyroid.metrics parity suite (ROUGE: summarization; BLEU/chrF: machine translation).
CNNDM = {
    "repo": "abisee/cnn_dailymail",
    "revision": "96df5e686bee6baa90b8bee7c28b81fa3fa6223d",
    "file": "3.0.0/test-00000-of-00001.parquet",
    "license": "Apache-2.0",
}
SUMMEVAL = {
    "repo": "mteb/summeval",
    "revision": "bfc121155064afa2d81b5505682ffc0d96f4334c",
    "file": "data/test-00000-of-00001-35901af5f6649399.parquet",
    "license": "MIT",
}
WMT21 = {
    "repo": "wmt-conference/wmt21-news-systems",
    "revision": "f41750dadbf2ba6ab01ca5380347d253298a3427",
    "testset": "newstest2021",
    "langpairs": ["en-de", "de-en", "en-zh", "zh-en", "en-ru", "ru-en", "en-ja", "ja-en"],
    "license": "WMT21 shared-task release (research use; not redistributed)",
}
# The trafilatura suite's real pages: Common Crawl WARC slices (byte ranges on gzip member boundaries, sha256 of
# the raw bytes) through datatrove 0.10.1's WarcReader + URLFilter (pyroid_parity.trafilatura_cc). One file per
# crawl; slices from 20 %, 40 % and 60 % into the older WARC files, whose records are sorted by host.
_CC26 = "crawl-data/CC-MAIN-2026-39/segments/1788492699475.64/warc/CC-MAIN-20260904131603-20260904161603-00100.warc.gz"
_CC23 = "crawl-data/CC-MAIN-2023-50/segments/1700679099281.67/warc/CC-MAIN-20231128083443-20231128113443-00100.warc.gz"
_CC19 = "crawl-data/CC-MAIN-2019-35/segments/1566027312025.20/warc/CC-MAIN-20190817203056-20190817225056-00100.warc.gz"
TRAFILATURA_CC = {
    "trafilatura/CC-MAIN-2026-39.parquet": {
        "crawl": "CC-MAIN-2026-39",
        "warc": _CC26,
        "slices": [(0, 196_203_906, "c76da056343574fa5e309fefe519a842251d11338cc4c64629743f6f0aab96a2")],
        # pyarrow's defaults, as P10 wrote /target/cc/sample.parquet: this file is that one, byte for byte.
        "writer": {"compression": "zstd"},
        "p10": "the P10 sample (sha256 a9cadd81...): eval = rows 0-1999, dev = 2000-2999, hold = 3000-4676",
    },
    "trafilatura/CC-MAIN-2023-50.parquet": {
        "crawl": "CC-MAIN-2023-50",
        "warc": _CC23,
        "slices": [
            (240_083_421, 250_072_163, "2ec2d4641d1421a6a8ac52fe374b719b1127e1fbe4e9f574832ec1d7c54b8bc4"),
            (480_170_626, 490_135_704, "7fe3bd35631b1fec6c7d7b6f4304f4d0e446cbe8fa15672463e63e3285bd58b4"),
            (720_221_902, 730_161_499, "a6050c1149500acd8a9681c37aeb4dafd973ec2d5fbb69e5fcdab11a43a845bc"),
        ],
        "writer": {
            "compression": "zstd",
            "compression_level": 9,
            "row_group_size": 5_000,
            "write_statistics": False,
        },
    },
    "trafilatura/CC-MAIN-2019-35.parquet": {
        "crawl": "CC-MAIN-2019-35",
        "warc": _CC19,
        "slices": [
            (213_435_718, 223_416_245, "925ecf76671b41386f3bda7fa02cf3031c748d8d7904e994899ca35a8c17b2f3"),
            (426_878_038, 436_635_070, "3ec0c1fee4072538d66928e5f0571ebdb953514d4a10d1f5269c0db382f3a282"),
            (640_340_436, 650_284_519, "93a6028a2152ee4e8f1b6ad8e2af69c2b1b46ff02e83a9c22c6e32033ab9b2d4"),
        ],
        "writer": {
            "compression": "zstd",
            "compression_level": 9,
            "row_group_size": 5_000,
            "write_statistics": False,
        },
    },
}
# Trafilatura's own tests at the commits tagged v2.3.1 and v1.11.0 (one per pyroid.trafilatura mode), minus the
# evaluation set (tests/eval/, 135 MB) and its data files: replayed as golden calls by pyroid_parity.trafilatura_record.
_TESTS_LICENSE = "Apache-2.0 (the tests); tests/cache/ holds third-party web pages (not redistributed)"
TRAFILATURA_TESTS = {
    "2.3.1": {
        "repo": "adbar/trafilatura",
        "tag": "v2.3.1",
        "commit": "e2f85c965662c7df26b73290ef175b2c2a5fdca6",
        "paths": "tests/ except tests/eval/ and tests/evaldata.json",
        "license": _TESTS_LICENSE,
    },
    "1.11.0": {
        "repo": "adbar/trafilatura",
        "tag": "v1.11.0",
        "commit": "60647e5eeff77df9ef4b7432a936f580736b3075",
        "paths": "tests/ except tests/eval/, tests/evaldata.json and tests/evaldata.py",
        "license": _TESTS_LICENSE,
    },
}
_TESTS_LEFT_OUT = {"tests/evaldata.json", "tests/evaldata.py"}
BUDGET = 1_500_000_000
# The --only steps; each downloads the pinned files under its directory (see step_of).
STEPS = ("fineweb", "fineweb2", "fineweb2-train", "models", "metrics", "trafilatura", "warc")
DEFAULT_STEPS = "fineweb,fineweb2,models,metrics,trafilatura,warc"
# Steps whose files are fetched (and verified) only when --only names them.
OPT_IN_STEPS = frozenset(set(STEPS) - set(DEFAULT_STEPS.split(",")))


def step_of(rel: str) -> str | None:
    """The --only step that downloads the pinned file ``rel``; None for the generated files: the
    parity/ references and the models no step downloads (models/nemo-quality-dev10k.ftz, which the
    nemo step of pyroid_parity trains)."""
    top = rel.split("/", 1)[0]
    if top == "fineweb2" and rel.endswith(TRAIN_SUFFIX):
        return "fineweb2-train"
    if top == "models" and rel not in MODELS:
        return None
    return top if top in STEPS else None


def log(msg: str) -> None:
    print(f"[fetch {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_parquet(table, rel: str) -> Path:
    import pyarrow.parquet as pq

    out = D.DATA / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    # Fixed writer options -> deterministic bytes for a given pyarrow version (pinned in uv.lock).
    pq.write_table(
        table, tmp, compression="zstd", compression_level=3, row_group_size=5_000, write_statistics=False
    )
    tmp.replace(out)
    return out


def range_read(repo: str, revision: str, file: str, n_rows: int):
    """Read the first ``n_rows`` rows, in file order, via HTTP range reads of whole row groups."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    with fs.open(f"datasets/{repo}@{revision}/{file}", "rb", block_size=8 << 20) as f:
        pf = pq.ParquetFile(f)
        batches, have = [], 0
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg)
            batches.append(t)
            have += t.num_rows
            if have >= n_rows:
                break
        return pa.concat_tables(batches).slice(0, n_rows).combine_chunks()


def fetch_fineweb(pin: bool) -> None:
    rel100, rel10 = "fineweb/fineweb-100k.parquet", "fineweb/fineweb-dev-10k.parquet"
    if not (D.DATA / rel100).exists():
        log("FineWeb: range-reading the first 100k rows (~30-60 s)")
        t = range_read(FINEWEB["repo"], FINEWEB["revision"], FINEWEB["file"], 100_000)
        ids, texts = t.column("id").to_pylist(), t.column("text").to_pylist()
        got = D.content_sha256(ids, texts)
        if got != EXPECTED_CONTENT[100_000]:
            sys.exit(f"FineWeb 100k content sha256 {got} != pinned {EXPECTED_CONTENT[100_000]}")
        if D.content_sha256(ids[:5000], texts[:5000]) != EXPECTED_CONTENT[5_000]:
            sys.exit("FineWeb 5k-prefix content sha256 mismatch")
        write_parquet(t, rel100)
        write_parquet(t.slice(0, 10_000), rel10)
    if pin:
        import pyarrow.parquet as pq

        for rel, n in ((rel100, 100_000), (rel10, 10_000)):
            t = pq.read_table(D.DATA / rel, columns=["id", "text"])
            ids, texts = t.column("id").to_pylist(), t.column("text").to_pylist()
            assert len(ids) == n
            D.pin(
                rel,
                rows=n,
                content_sha256=D.content_sha256(ids, texts),
                chars=sum(map(len, texts)),
                source={**FINEWEB, "rows": f"first {n} in file order"},
            )


def fetch_fineweb2(pin: bool) -> None:
    for lang in FINEWEB2["langs"]:
        rel = f"fineweb2/{lang}.parquet"
        if not (D.DATA / rel).exists():
            log(f"FineWeb-2 {lang}: first {FINEWEB2['rows']} rows of the test split")
            file = f"data/{lang}/test/000_00000.parquet"
            t = range_read(FINEWEB2["repo"], FINEWEB2["revision"], file, FINEWEB2["rows"])
            write_parquet(t.select(["id", "text"]), rel)
        if pin:
            ids = D.read_parquet_columns(D.DATA / rel, ["id", "text"])
            D.pin(
                rel,
                rows=len(ids["id"]),
                content_sha256=D.content_sha256(ids["id"], ids["text"]),
                source={
                    "repo": FINEWEB2["repo"],
                    "revision": FINEWEB2["revision"],
                    "file": f"data/{lang}/test/000_00000.parquet",
                    "license": FINEWEB2["license"],
                },
            )


def fetch_fineweb2_train(pin: bool) -> None:
    n = FINEWEB2_TRAIN["rows"]
    for lang in FINEWEB2_TRAIN["langs"]:
        rel = f"fineweb2/{lang}{TRAIN_SUFFIX}"
        file = f"data/{lang}/train/000_00000.parquet"
        if not (D.DATA / rel).exists():
            log(f"FineWeb-2 {lang}: first {n} rows of {file} (~45 MB of row groups)")
            t = range_read(FINEWEB2["repo"], FINEWEB2["revision"], file, n)
            write_parquet(t.select(["id", "text"]), rel)
        if pin:
            cols = D.read_parquet_columns(D.DATA / rel, ["id", "text"])
            assert len(cols["id"]) == n
            D.pin(
                rel,
                rows=n,
                content_sha256=D.content_sha256(cols["id"], cols["text"]),
                chars=sum(map(len, cols["text"])),
                source={
                    "repo": FINEWEB2["repo"],
                    "revision": FINEWEB2["revision"],
                    "file": file,
                    "license": FINEWEB2["license"],
                    "rows": f"first {n} in file order",
                },
            )


def download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:  # pinned https URLs
        while block := r.read(1 << 20):
            f.write(block)
    tmp.replace(out)


def fetch_models(pin: bool) -> None:
    for rel, url in MODELS.items():
        if not (D.DATA / rel).exists():
            log(f"downloading {url}")
            download(url, D.DATA / rel)
        if pin:
            D.pin(rel, source={"url": url, "license": "CC BY-SA 3.0"})


def hf_url(src: dict) -> str:
    return f"https://huggingface.co/datasets/{src['repo']}/resolve/{src['revision']}/{src['file']}"


def wmt21_files() -> dict[str, list[str]]:
    """Language pair -> [reference files..., system output files...] at the pinned revision (sorted)."""
    import json

    url = f"https://api.github.com/repos/{WMT21['repo']}/git/trees/{WMT21['revision']}?recursive=1"
    with urllib.request.urlopen(url, timeout=120) as r:  # pinned commit: the listing cannot change
        tree = json.load(r)
    assert not tree.get("truncated"), "GitHub tree listing truncated"
    out: dict[str, list[str]] = {}
    for lp in WMT21["langpairs"]:
        stem = f"{WMT21['testset']}.{lp}"
        refs = sorted(x["path"] for x in tree["tree"] if x["path"].startswith(f"txt/references/{stem}.ref."))
        hyps = sorted(
            x["path"] for x in tree["tree"] if x["path"].startswith(f"txt/system-outputs/{stem}.hyp.")
        )
        out[lp] = refs + hyps
    return out


def _lines(raw: bytes) -> list[str]:
    text = raw.decode("utf-8")
    return text[:-1].split("\n") if text.endswith("\n") else text.split("\n")


def build_wmt21():
    """One row per (language pair, system, segment): hyp and the references in file order (A, B, C, ...)."""
    import pyarrow as pa

    rows: dict[str, list] = {"lp": [], "system": [], "seg": [], "hyp": [], "refs": [], "ref_names": []}
    base = f"https://raw.githubusercontent.com/{WMT21['repo']}/{WMT21['revision']}/"
    for lp, files in wmt21_files().items():
        stem = f"{WMT21['testset']}.{lp}"
        refs, hyps = [], []
        for path in files:
            with urllib.request.urlopen(base + path, timeout=120) as r:
                lines = _lines(r.read())
            name = path.rsplit("/", 1)[1].removeprefix(stem + ".")
            (refs if name.startswith("ref.") else hyps).append((name.split(".")[1], lines))
        n = len(refs[0][1])
        bad = [nm for nm, lines in refs + hyps if len(lines) != n]
        assert not bad, f"{lp}: line counts differ from ref {refs[0][0]} ({n}): {bad}"
        log(f"WMT21 {lp}: {len(hyps)} systems x {n} segments, references {[nm for nm, _ in refs]}")
        for system, lines in hyps:
            for i, hyp in enumerate(lines):
                rows["lp"].append(lp)
                rows["system"].append(system)
                rows["seg"].append(i)
                rows["hyp"].append(hyp)
                rows["refs"].append([lines_r[i] for _, lines_r in refs])
                rows["ref_names"].append([nm for nm, _ in refs])
    return pa.table(rows)


def fetch_metrics(pin: bool) -> None:
    for rel, src in (("metrics/cnndm-3.0.0-test.parquet", CNNDM), ("metrics/summeval.parquet", SUMMEVAL)):
        if not (D.DATA / rel).exists():
            log(f"downloading {src['repo']} {src['file']}")
            download(hf_url(src), D.DATA / rel)
        if pin:
            import pyarrow.parquet as pq

            D.pin(rel, rows=pq.ParquetFile(D.DATA / rel).metadata.num_rows, source=src)
    rel = "metrics/wmt21.parquet"
    if not (D.DATA / rel).exists():
        log("WMT21: downloading the newstest2021 references and system outputs from GitHub")
        write_parquet(build_wmt21(), rel)
    if pin:
        import pyarrow.parquet as pq

        t = pq.read_table(D.DATA / rel, columns=["lp", "system", "hyp"])
        D.pin(
            rel,
            rows=t.num_rows,
            systems=len(set(zip(t.column("lp").to_pylist(), t.column("system").to_pylist(), strict=True))),
            source=WMT21,
        )


def build_cc_pages(rel: str, spec: dict) -> None:
    """Fetch the slices of one crawl, run datatrove's reader and URL filter, write (id, url, html)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyroid_parity import trafilatura_cc as CC

    pages = CC.Pages()
    rows: dict[str, list[str]] = {"id": [], "url": [], "html": []}
    for start, end, sha in spec["slices"]:
        log(f"{spec['crawl']}: bytes [{start:,}, {end:,}) of {spec['warc'].rsplit('/', 1)[1]}")
        n = len(rows["id"])
        for doc_id, url, html in pages(CC.fetch(CC.Slice(spec["crawl"], spec["warc"], start, end, sha))):
            rows["id"].append(doc_id)
            rows["url"].append(url)
            rows["html"].append(html)
        log(f"{spec['crawl']}: {len(rows['id']) - n} pages")
    out = D.DATA / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    pq.write_table(pa.table(rows), tmp, **spec["writer"])
    tmp.replace(out)


def git_blob_sha1(data: bytes) -> str:
    import hashlib

    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def build_trafilatura_tests(rel: str, src: dict) -> None:
    """Trafilatura's tests/ at a pinned commit (each file checked against its git blob sha1)."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    url = f"https://api.github.com/repos/{src['repo']}/git/trees/{src['commit']}?recursive=1"
    with urllib.request.urlopen(url, timeout=120) as r:  # pinned commit: the listing cannot change
        tree = json.load(r)
    assert not tree.get("truncated"), "GitHub tree listing truncated"
    files = sorted(
        (x["path"], x["sha"])
        for x in tree["tree"]
        if x["type"] == "blob"
        and x["path"].startswith("tests/")
        and not x["path"].startswith("tests/eval/")
        and x["path"] not in _TESTS_LEFT_OUT
    )
    log(f"Trafilatura {src['tag']} tests: {len(files)} files from GitHub")
    base = f"https://raw.githubusercontent.com/{src['repo']}/{src['commit']}/"
    blobs = []
    for path, sha in files:
        with urllib.request.urlopen(base + urllib.parse.quote(path), timeout=120) as r:
            data = r.read()
        if git_blob_sha1(data) != sha:
            sys.exit(f"{path}: git blob sha1 differs from the tree at {src['commit']}")
        blobs.append(data)
    table = pa.table(
        {"path": [p for p, _ in files], "git_sha1": [s for _, s in files], "data": blobs},
        schema=pa.schema([("path", pa.string()), ("git_sha1", pa.string()), ("data", pa.binary())]),
    )
    out = D.DATA / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    pq.write_table(
        table, tmp, compression="zstd", compression_level=9, row_group_size=5_000, write_statistics=False
    )
    tmp.replace(out)


def fetch_trafilatura(pin: bool) -> None:
    import pyarrow.parquet as pq

    for rel, spec in TRAFILATURA_CC.items():
        if not (D.DATA / rel).exists():
            build_cc_pages(rel, spec)
        t = pq.read_table(D.DATA / rel, columns=["id", "html"])
        ids, htmls = t.column("id").to_pylist(), t.column("html").to_pylist()
        content = D.content_sha256(ids, htmls)
        if pin:
            from pyroid_parity import trafilatura_cc as CC

            D.pin(
                rel,
                rows=len(ids),
                content_sha256=content,
                chars=sum(map(len, htmls)),
                source={
                    "crawl": spec["crawl"],
                    "warc": CC.CC_DATA + spec["warc"],
                    "slices": [list(s) for s in spec["slices"]],
                    "pipeline": CC.PIPELINE,
                    "license": CC.LICENSE,
                    **({"p10": spec["p10"]} if "p10" in spec else {}),
                },
                writer=spec["writer"],
            )
        elif content != D.load_manifest()["files"].get(rel, {}).get("content_sha256"):
            sys.exit(f"{rel}: content sha256 {content} differs from the manifest")
        del t, ids, htmls
    for version, src in TRAFILATURA_TESTS.items():
        rel = f"trafilatura/upstream-tests-{version}.parquet"
        if not (D.DATA / rel).exists():
            build_trafilatura_tests(rel, src)
        if pin:
            D.pin(rel, rows=pq.ParquetFile(D.DATA / rel).metadata.num_rows, source=src)


def fetch_warc(pin: bool) -> None:
    """The warc suite's inputs: Common Crawl slices and warcio's test files, each checked on download."""
    from pyroid_parity import warc_corpus

    warc_corpus.fetch(pin, log)


def verify(steps: set[str] | set[str | None] | None = None, skip: tuple[str, ...] = ()) -> int:
    """Re-hash the pinned files of ``steps`` (all of them for None), except those under ``skip``;
    returns the number that are bad."""
    manifest = D.load_manifest()
    files = {
        r: e
        for r, e in manifest["files"].items()
        if (steps is None or step_of(r) in steps) and not (skip and r.startswith(skip))
    }
    bad, checked = 0, 0
    for rel, entry in files.items():
        p = D.DATA / rel
        ok = p.exists() and p.stat().st_size == entry["bytes"] and D.sha256_file(p) == entry["sha256"]
        checked += entry["bytes"]
        print(f"{'ok ' if ok else 'BAD'} {entry['bytes']:>12,}  {rel}")
        bad += not ok
    total = sum(e["bytes"] for e in manifest["files"].values())
    print(
        f"verified {len(files)} of {len(manifest['files'])} pinned files ({checked / 1e6:,.1f} MB); "
        f"{total / 1e6:,.1f} MB pinned in all (budget {BUDGET / 1e6:,.0f} MB)"
    )
    if total > BUDGET:
        print("over the 1.5 GB budget")
        bad += 1
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="only verify the inputs against manifest.json")
    ap.add_argument(
        "--verify-all",
        action="store_true",
        help="only verify every pinned file against manifest.json, the generated parity/ references included"
        " (the opt-in steps' files only when --only names them)",
    )
    ap.add_argument(
        "--except",
        dest="skip",
        action="append",
        default=[],
        metavar="PREFIX",
        help="with --verify-all: leave out the pinned files under PREFIX (built and checked elsewhere)",
    )
    ap.add_argument("--pin", action="store_true", help="(re)write sha256 entries for fetched files")
    ap.add_argument("--only", default=DEFAULT_STEPS, help=f"comma-separated subset of {', '.join(STEPS)}")
    a = ap.parse_args()
    steps = set(a.only.split(","))
    if unknown := steps - set(STEPS):
        ap.error(f"--only: unknown step(s) {sorted(unknown)}; choose from {list(STEPS)}")
    if a.verify_all:
        # step_of() is None for the generated files (parity/ references, the nemo model).
        wanted: set[str | None] = {*(set(STEPS) - OPT_IN_STEPS), *(steps & OPT_IN_STEPS), None}
        return 1 if verify(wanted, skip=tuple(a.skip)) else 0
    if not a.verify:
        if "fineweb" in steps:
            fetch_fineweb(a.pin)
        if "fineweb2" in steps:
            fetch_fineweb2(a.pin)
        if "fineweb2-train" in steps:
            fetch_fineweb2_train(a.pin)
        if "models" in steps:
            fetch_models(a.pin)
        if "metrics" in steps:
            fetch_metrics(a.pin)
        if "trafilatura" in steps:
            fetch_trafilatura(a.pin)
        if "warc" in steps:
            fetch_warc(a.pin)
    return 1 if verify(steps) else 0


if __name__ == "__main__":
    sys.exit(main())
