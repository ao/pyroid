#!/usr/bin/env python3
"""Download the pinned benchmark/parity inputs into the shared ``/data`` volume (run in Docker, never on the host).

    DX_NAME=harness .orchestration/dx.sh 'cd bench && uv run --frozen python fetch.py'   # fetch + verify the inputs
    ... python fetch.py --verify                 # verify the inputs only (no network)
    ... python fetch.py --verify-all             # verify every pinned file, parity/ references included
    ... python fetch.py --pin                    # first fetch / intentional update: write hashes to manifest.json

Layout under /data (total budget 1.5 GB):
    fineweb/fineweb-100k.parquet       headline slice: first 100,000 rows of FineWeb sample/10BT/000_00000.parquet
    fineweb/fineweb-dev-10k.parquet    dev slice: the first 10,000 of those rows
    fineweb2/<lang>.parquet            first rows of FineWeb-2 test splits (non-English adversarial input)
    models/lid.176.bin, lid.176.ftz    fastText language-ID models (CC BY-SA 3.0)
    metrics/cnndm-3.0.0-test.parquet   CNN/DailyMail 3.0.0 test split, as published (pyroid.metrics ROUGE parity)
    metrics/summeval.parquet           SummEval: 16 systems' summaries of 100 CNN/DM articles, 11 references each
    metrics/wmt21.parquet              WMT21 newstest system outputs + references, 8 language pairs (BLEU/chrF)

The parity/ files in the manifest are generated, not fetched: ``python -m pyroid_parity.refs`` writes them and
verifies what it wrote, so fetch.py checks them only with ``--verify-all``.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

from pyroid_parity import data as D

FINEWEB = {
    "repo": "HuggingFaceFW/fineweb",
    "revision": "9bb295ddab0e05d785b879661af7260fed5140fc",
    "file": "sample/10BT/000_00000.parquet",
    "license": "ODC-By 1.0",
}
# Content sha256 over id\0text\0... (R2 §5): reproducible across parquet writers. 100k/5k measured by R2.
EXPECTED_CONTENT = {
    100_000: "1a93378229bbed5bcfa552ef67c9068d0c8d70f56204d40ac62c41fb3616567c",
    5_000: "f227def5ef38b53612a8ce2d5a968bd4da21809c714535912599f0388e20a949",
}
FINEWEB2 = {
    "repo": "HuggingFaceFW/fineweb-2",
    "revision": "af9c13333eb981300149d5ca60a8e9d659b276b9",
    "license": "ODC-By 1.0",
    "rows": 250,
    "langs": ["fra_Latn", "deu_Latn", "spa_Latn", "rus_Cyrl", "cmn_Hani", "jpn_Jpan", "arb_Arab", "hin_Deva"],
}
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
BUDGET = 1_500_000_000
# Manifest prefix of the files each --only step downloads.
INPUTS = {"fineweb": "fineweb/", "fineweb2": "fineweb2/", "models": "models/", "metrics": "metrics/"}


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


def verify(prefixes: tuple[str, ...] | None = None) -> int:
    """Re-hash the pinned files under ``prefixes`` (all of them for None); returns the number that are bad."""
    manifest = D.load_manifest()
    files = {r: e for r, e in manifest["files"].items() if prefixes is None or r.startswith(prefixes)}
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
        help="only verify every pinned file against manifest.json, the generated parity/ references included",
    )
    ap.add_argument("--pin", action="store_true", help="(re)write sha256 entries for fetched files")
    ap.add_argument("--only", default="fineweb,fineweb2,models,metrics", help="comma-separated subset")
    a = ap.parse_args()
    steps = set(a.only.split(","))
    if unknown := steps - INPUTS.keys():
        ap.error(f"--only: unknown step(s) {sorted(unknown)}; choose from {sorted(INPUTS)}")
    if a.verify_all:
        return 1 if verify() else 0
    if not a.verify:
        if "fineweb" in steps:
            fetch_fineweb(a.pin)
        if "fineweb2" in steps:
            fetch_fineweb2(a.pin)
        if "models" in steps:
            fetch_models(a.pin)
        if "metrics" in steps:
            fetch_metrics(a.pin)
    return 1 if verify(tuple(INPUTS[s] for s in sorted(steps))) else 0


if __name__ == "__main__":
    sys.exit(main())
