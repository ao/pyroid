#!/usr/bin/env python3
"""Select and download the pyroid.yaml real-file corpus, then write its manifest.

    python -I bench/yaml/build_corpus.py search  --work /target/data/build
    python -I bench/yaml/build_corpus.py github  --work /target/data/build --corpus /target/data/corpus
    python -I bench/yaml/build_corpus.py hf      --work /target/data/build --corpus /target/data/corpus
    python -I bench/yaml/build_corpus.py synth   --work /target/data/build --corpus /target/data/corpus
    python -I bench/yaml/build_corpus.py manifest --work /target/data/build --corpus /target/data/corpus \
        --status /target/data/oracle/corpus/index.jsonl --out parity/data/yaml-corpus/manifest.json

This is the one-off selection step; ``fetch_corpus.py`` rebuilds the same corpus from the manifest.
Needs git and, for ``synth``/``manifest``, PyYAML 6.0.3 (the recorder venv).

* ``search``: popular GitHub repositories from the unauthenticated search API (10 requests a minute),
  the queries in ``QUERIES``, in rank order, plus the repositories in ``FIXED`` (their licence and
  default branch from the repos API). Writes ``WORK/repos.json``. openapi-directory is sampled (see
  ``select_openapi``; blob sizes from the trees API at the pinned commit).
* ``github``: per repository, a shallow partial clone (``--depth 1 --filter=blob:none --no-checkout``)
  pins the commit and lists the tree; the paths the family rules pick are checked out (git fetches
  just those blobs), their exact blob bytes are copied to ``CORPUS/<family>/<owner>__<repo>/<path>``
  and the clone is deleted. Symlinks, submodules and Git LFS pointers are skipped. Writes
  ``WORK/sources/<owner>__<repo>.json``.
* ``hf``: the most downloaded and most liked Hugging Face models and datasets; the YAML front matter of
  README.md at the listed revision goes to ``CORPUS/hf-card/{models,datasets}/<id>.yaml``.
* ``synth``: multi-document streams of at least 1, 10 and 100 MB built from real files
  (``fetch_corpus.build_synthetic``) into ``CORPUS/synthetic/``; the parts are files that PyYAML and
  rapidyaml both parse (``part_ok``), so every row of ``baseline_bench.py`` can time them.
* ``manifest``: one entry per file (identical contents are kept once, first source wins), the
  sources with commit and licence, the synthetic recipes; ``--status`` adds whether
  ``yaml.safe_load_all`` succeeded, taken from ``record_oracle.py``'s index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_corpus import build_synthetic, download, front_matter, synthetic_part

# (query, pages of 100) for /search/repositories, sorted by stars.
QUERIES = [
    ("stars:>=20000", 6),
    ("language:Python stars:>=3000", 4),
    ('language:"Jupyter Notebook" stars:>=1500', 2),
    ("topic:kubernetes stars:>=1500", 1),
    ("topic:docker stars:>=2000", 1),
    ("topic:self-hosted stars:>=2000", 1),
    ("topic:mkdocs stars:>=300", 1),
]
# How many of the ranked popular repositories contribute GitHub Actions workflows.
GHA_REPOS = 400
# openapi-directory: largest spec kept in the openapi family, and how many of the largest go to big.
OPENAPI_MAX_BYTES = 500_000
OPENAPI_BIG = 5

YML = r"\.ya?ml$"
# Repositories taken for a family of their own: (repo, family, path regex, max file count or None).
FIXED = [
    ("kubernetes/examples", "k8s", r".*" + YML, None),
    ("kubernetes/website", "k8s", r"^content/en/examples/.*" + YML, None),
    ("argoproj/argo-workflows", "k8s", r"^examples/.*" + YML, None),
    ("istio/istio", "k8s", r"^samples/.*" + YML, None),
    (
        "GoogleCloudPlatform/microservices-demo",
        "k8s",
        r"^(kubernetes-manifests|kustomize|helm-chart)/.*" + YML,
        None,
    ),
    ("bitnami/charts", "helm", r"^bitnami/[^/]+/(values\.yaml|Chart\.yaml|templates/[^/]+\.yaml)$", None),
    ("prometheus-community/helm-charts", "helm", r"^charts/.*" + YML, None),
    ("grafana/helm-charts", "helm", r"^charts/.*" + YML, None),
    ("docker/awesome-compose", "compose", r"(^|/)(docker-)?compose[^/]*" + YML, None),
    ("ansible/ansible-examples", "ansible", r".*" + YML, None),
    ("APIs-guru/openapi-directory", "openapi", r"^APIs/.*/(openapi|swagger)\.yaml$", None),
    ("github/rest-api-description", "big", r"^descriptions/api\.github\.com/api\.github\.com\.yaml$", None),
    ("stripe/openapi", "big", r"^openapi/spec3\.yaml$", None),
    ("cloudflare/api-schemas", "big", r"^openapi\.yaml$", None),
]

# Rules for the popular repositories: (family, path regex, max files per repository), first match wins.
POPULAR_RULES = [
    ("gha", r"^\.github/workflows/[^/]+" + YML, 20),
    ("compose", r"(^|/)(docker-)?compose[^/]*" + YML, 10),
    ("conda", r"(^|/)(environment|conda)[^/]*" + YML, 10),
    ("mkdocs", r"(^|/)mkdocs[^/]*" + YML, 3),
    ("pnpm-lock", r"^pnpm-lock\.yaml$", 1),
    (
        "config",
        (
            r"^(\.pre-commit-config\.ya?ml|\.readthedocs\.ya?ml|\.?codecov\.ya?ml|\.golangci\.ya?ml"
            r"|\.goreleaser\.ya?ml|\.gitlab-ci\.yml|\.travis\.yml|action\.ya?ml|\.clang-format|\.clang-tidy"
            r"|\.yamllint(\.ya?ml)?|pubspec\.yaml|CITATION\.cff|\.markdownlint\.ya?ml"
            r"|\.github/(dependabot|FUNDING|release|labeler|stale|config)\.ya?ml"
            r"|\.github/(ISSUE|DISCUSSION)_TEMPLATE/[^/]+\.ya?ml)$"
        ),
        15,
    ),
]
HF_LISTS = [
    ("models", "downloads", 300),
    ("models", "likes", 300),
    ("datasets", "downloads", 300),
    ("datasets", "likes", 200),
]
SYNTH_SIZES = [
    ("synthetic/mixed-1mb.yaml", 1_000_000),
    ("synthetic/mixed-10mb.yaml", 10_000_000),
    ("synthetic/mixed-100mb.yaml", 100_000_000),
]
SYNTH_FAMILIES = [
    "k8s",
    "helm",
    "gha",
    "compose",
    "conda",
    "mkdocs",
    "config",
    "ansible",
    "hf-card",
    "openapi",
]
# The yaml-test-suite data release the oracle also covers (parity/data/yaml-test-suite-data-2022-01-17/).
YAML_TEST_SUITE = {
    "repo": "https://github.com/yaml/yaml-test-suite",
    "tag": "data-2022-01-17",
    "commit": "6e6c296ae9c9d2d5c4134b4b64d01b29ac19ff6f",
    "tree": "493d388a50385d992cfaa628a88171aa1965c50b",
    "tarball": "https://github.com/yaml/yaml-test-suite/archive/refs/tags/data-2022-01-17.tar.gz",
    "tarball_sha256": "ccae49c1d010d27c20d315969a97752f95b14642fc1e73755db4eb2df4bb0256",
    "licence": "MIT",
    "cases": 402,
}
UA = {
    "User-Agent": "pyroid-yaml-corpus (https://github.com/ao/pyroid)",
    "Accept": "application/vnd.github+json",
}


def log(*a: object) -> None:
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def get_json(url: str) -> object:
    for attempt in range(10):
        req = urllib.request.Request(url, headers=UA)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in (403, 429, 500, 502, 503) or attempt == 9:
                raise
            reset = e.headers.get("X-RateLimit-Reset")
            wait = (
                float(reset) - time.time() + 2
                if reset and e.headers.get("X-RateLimit-Remaining") == "0"
                else 30
            )
            log(f"HTTP {e.code} for {url}, waiting {wait:.0f}s")
            time.sleep(max(5, min(wait, 3700)))
    raise RuntimeError("unreachable")


def slug(full_name: str) -> str:
    return full_name.replace("/", "__")


def cmd_search(a: argparse.Namespace) -> None:
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    popular: dict[str, dict] = {}
    for q, pages in QUERIES:
        for page in range(1, pages + 1):
            url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(
                {"q": q, "sort": "stars", "order": "desc", "per_page": 100, "page": page}
            )
            items = get_json(url)["items"]
            log(f"{q} page {page}: {len(items)}")
            for it in items:
                popular.setdefault(
                    it["full_name"],
                    {
                        "repo": it["full_name"],
                        "stars": it["stargazers_count"],
                        "query": q,
                        "licence": (it.get("license") or {}).get("spdx_id") or "none",
                        "archived": it["archived"],
                        "size_kb": it["size"],
                    },
                )
            time.sleep(7)
    fixed = {}
    for repo, *_ in FIXED:
        it = get_json(f"https://api.github.com/repos/{repo}")
        fixed[repo] = {
            "repo": it["full_name"],
            "licence": (it.get("license") or {}).get("spdx_id") or "none",
            "stars": it["stargazers_count"],
        }
        time.sleep(1)
    (work / "repos.json").write_text(
        json.dumps({"popular": list(popular.values()), "fixed": fixed}, indent=1)
    )
    log(f"{len(popular)} popular repositories, {len(fixed)} fixed")


def git(*args: str, cwd: Path | None = None, timeout: int = 1800, inp: bytes | None = None) -> bytes:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_LITERAL_PATHSPECS="1", GIT_LFS_SKIP_SMUDGE="1")
    r = subprocess.run(
        ["git", *args], cwd=cwd, input=inp, capture_output=True, timeout=timeout, env=env, check=False
    )
    if r.returncode:
        raise RuntimeError(f"git {' '.join(args[:3])}: {r.stderr.decode(errors='replace')[-400:]}")
    return r.stdout


def select_popular(paths: list[str], gha: bool) -> list[tuple[str, str]]:
    picked: list[tuple[str, str]] = []
    counts: dict[str, int] = {}
    for p in sorted(paths):
        for fam, rx, cap in POPULAR_RULES:
            if fam == "gha" and not gha:
                continue
            if re.search(rx, p):
                if counts.get(fam, 0) < cap:
                    counts[fam] = counts.get(fam, 0) + 1
                    picked.append((fam, p))
                break
    return picked


def openapi_sizes(repo: str, commit: str) -> dict[str, int]:
    tree = get_json(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1")
    if tree.get("truncated"):
        raise RuntimeError("tree listing truncated")
    return {e["path"]: e["size"] for e in tree["tree"] if e["type"] == "blob"}


def select_openapi(paths: list[str], sizes: dict[str, int]) -> list[tuple[str, str]]:
    """Pick from APIs-guru/openapi-directory: APIs/<provider>[/<service>]/<version>/<file>.

    Only the highest <version> of each API (natural sort) is considered. Of those, every second API in
    path order up to OPENAPI_MAX_BYTES goes to ``openapi`` and the OPENAPI_BIG largest to ``big``.
    """
    best: dict[str, tuple[str, str]] = {}
    for p in paths:
        parts = p.split("/")
        api, version = "/".join(parts[:-2]), parts[-2]
        cur = best.get(api)
        if cur is None or _vkey(version) > _vkey(cur[0]):
            best[api] = (version, p)
    latest = [best[api][1] for api in sorted(best)]
    big = set(sorted(latest, key=lambda p: (-sizes[p], p))[:OPENAPI_BIG])
    picked = [("big", p) for p in sorted(big)]
    picked += [
        ("openapi", p)
        for i, p in enumerate(latest)
        if i % 2 == 0 and p not in big and sizes[p] <= OPENAPI_MAX_BYTES
    ]
    return picked


def _vkey(v: str) -> list[tuple[int, int | str]]:
    return [(0, int(t)) if t.isdigit() else (1, t) for t in re.split(r"(\d+)", v) if t]


def process_repo(job: dict, corpus: Path, work: Path) -> dict | None:
    repo = job["repo"]
    out = work / "sources" / f"{slug(repo)}.json"
    if out.exists():
        return json.loads(out.read_text())
    tmp = work / "clones" / slug(repo)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        git(
            "clone",
            "--quiet",
            "--depth",
            "1",
            "--filter=blob:none",
            "--no-checkout",
            "--single-branch",
            f"https://github.com/{repo}.git",
            str(tmp),
        )
        commit = git("rev-parse", "HEAD", cwd=tmp).decode().strip()
        entries = []
        for rec in git("ls-tree", "-r", "-z", "HEAD", cwd=tmp).split(b"\0"):
            if not rec:
                continue
            meta, path = rec.split(b"\t", 1)
            mode, typ, oid = meta.split()
            if typ == b"blob" and mode in (b"100644", b"100755"):
                try:
                    entries.append((path.decode("utf-8"), oid.decode()))
                except UnicodeDecodeError:  # no raw URL for it; skip
                    pass
        oids = dict(entries)
        paths = list(oids)
        if job["kind"] == "fixed":
            rx = re.compile(job["rx"])
            sel = sorted(p for p in paths if rx.search(p))
            if job["family"] == "openapi":
                picked = select_openapi(sel, openapi_sizes(repo, commit))
            else:
                picked = [(job["family"], p) for p in sel]
        else:
            picked = select_popular(paths, job["gha"])
        files = []
        if picked:
            listing = tmp.with_name(tmp.name + ".paths")
            listing.write_bytes(b"\0".join(p.encode() for _, p in picked))
            git(
                "checkout",
                "--quiet",
                "HEAD",
                f"--pathspec-from-file={listing}",
                "--pathspec-file-nul",
                cwd=tmp,
            )
            batch = b"".join(oids[p].encode() + b"\n" for _, p in picked)
            raw = git("cat-file", "--batch", cwd=tmp, inp=batch)
            pos = 0
            for fam, p in picked:
                header_end = raw.index(b"\n", pos)
                _oid, _typ, size = raw[pos:header_end].split()
                size = int(size)
                data = raw[header_end + 1 : header_end + 1 + size]
                pos = header_end + 1 + size + 1
                if data.startswith(b"version https://git-lfs.github.com/spec/"):
                    continue
                if fam == "helm" and "/templates/" in p and b"{{" in data:
                    continue
                rel = f"{fam}/{slug(repo)}/{p}"
                dest = corpus / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                files.append(
                    {
                        "path": rel,
                        "family": fam,
                        "repo_path": p,
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "bytes": len(data),
                    }
                )
        rec = {
            "repo": f"https://github.com/{repo}",
            "name": repo,
            "commit": commit,
            "licence": job["licence"],
            "kind": job["kind"],
            "files": files,
        }
    except Exception as e:  # noqa: BLE001 - recorded in the source record; the run goes on
        log(f"{repo}: {type(e).__name__}: {str(e)[:300]}")
        rec = {
            "repo": f"https://github.com/{repo}",
            "name": repo,
            "error": f"{type(e).__name__}: {str(e)[:300]}",
            "kind": job["kind"],
            "files": [],
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.with_name(tmp.name + ".paths").unlink(missing_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    return rec


def cmd_github(a: argparse.Namespace) -> None:
    work, corpus = Path(a.work), Path(a.corpus)
    repos = json.loads((work / "repos.json").read_text())
    jobs = [
        {"repo": r, "kind": "fixed", "family": fam, "rx": rx, "licence": repos["fixed"][r]["licence"]}
        for r, fam, rx, _cap in FIXED
    ]
    fixed_names = {r for r, *_ in FIXED}
    rank = 0
    for r in repos["popular"]:
        if r["repo"] in fixed_names:
            continue
        jobs.append({"repo": r["repo"], "kind": "popular", "gha": rank < GHA_REPOS, "licence": r["licence"]})
        rank += 1
    if a.limit:
        jobs = jobs[: a.limit]
    log(f"{len(jobs)} repositories")
    done = 0
    with ThreadPoolExecutor(a.jobs) as pool:
        for rec in pool.map(lambda j: process_repo(j, corpus, work), jobs):
            done += 1
            if done % 50 == 0:
                log(f"{done}/{len(jobs)}")
            del rec


def cmd_hf(a: argparse.Namespace) -> None:
    work, corpus = Path(a.work), Path(a.corpus)
    seen: dict[tuple[str, str], dict] = {}
    for kind, sort, n in HF_LISTS:
        url = f"https://huggingface.co/api/{kind}?" + urllib.parse.urlencode(
            {"sort": sort, "direction": -1, "limit": n, "full": "true"}
        )
        for it in get_json(url):
            lic = next(
                (t.split(":", 1)[1] for t in it.get("tags", []) if t.startswith("license:")), "unspecified"
            )
            seen.setdefault(
                (kind, it["id"]),
                {"kind": kind, "id": it["id"], "sha": it["sha"], "licence": lic, "rank": f"{sort}"},
            )
    log(f"{len(seen)} Hugging Face repositories")

    def one(item: dict) -> dict:
        prefix = "" if item["kind"] == "models" else "datasets/"
        url = f"https://huggingface.co/{prefix}{item['id']}/resolve/{item['sha']}/README.md"
        rec = {
            "repo": f"https://huggingface.co/{prefix}{item['id']}",
            "name": f"hf:{prefix}{item['id']}",
            "commit": item["sha"],
            "licence": item["licence"],
            "kind": "hf",
            "files": [],
        }
        try:
            data = download(url, tries=4)
        except Exception as e:  # noqa: BLE001 - the card is listed as skipped
            rec["error"] = f"{type(e).__name__}: {e}"
            return rec
        fm = front_matter(data)
        if fm is None:
            rec["error"] = "no front matter"
            return rec
        rel = f"hf-card/{item['kind']}/{item['id']}.yaml"
        (corpus / rel).parent.mkdir(parents=True, exist_ok=True)
        (corpus / rel).write_bytes(fm)
        rec["files"].append(
            {
                "path": rel,
                "family": "hf-card",
                "url": url,
                "extract": "front-matter",
                "sha256": hashlib.sha256(fm).hexdigest(),
                "bytes": len(fm),
            }
        )
        return rec

    with ThreadPoolExecutor(8) as pool:
        recs = list(pool.map(one, sorted(seen.values(), key=lambda x: (x["kind"], x["id"]))))
    (work / "sources").mkdir(parents=True, exist_ok=True)
    (work / "sources" / "_hf.json").write_text(json.dumps(recs, indent=1))
    log(
        f"{sum(len(r['files']) for r in recs)} cards with front matter, {sum('error' in r for r in recs)} skipped"
    )


def load_sources(work: Path) -> list[dict]:
    recs: list[dict] = []
    for p in sorted((work / "sources").glob("*.json")):
        d = json.loads(p.read_text())
        recs.extend(d if isinstance(d, list) else [d])
    return recs


def unique_files(recs: list[dict]) -> tuple[list[tuple[dict, dict]], list[dict]]:
    """(source, file) pairs with each content once (first in source order), and the dropped files."""
    seen: set[str] = set()
    kept, dropped = [], []
    for r in recs:
        for f in r["files"]:
            if f["sha256"] in seen:
                dropped.append(f)
            else:
                seen.add(f["sha256"])
                kept.append((r, f))
    return kept, dropped


def cmd_dedup(a: argparse.Namespace) -> None:
    corpus = Path(a.corpus)
    kept, dropped = unique_files(load_sources(Path(a.work)))
    for f in dropped:
        (corpus / f["path"]).unlink(missing_ok=True)
    log(f"{len(kept)} files kept, {len(dropped)} duplicate contents removed")


def part_ok(yaml: object, ryml: object, data: bytes) -> bool:
    """A synthetic part: valid UTF-8 that CSafeLoader loads to non-null documents, the same documents
    inside a stream as on its own, and that rapidyaml parses (so every benchmark row can use the file)."""
    try:
        docs = list(yaml.load_all(data.decode("utf-8"), Loader=yaml.CSafeLoader))
        if not docs or any(d is None for d in docs):
            return False
        joined = synthetic_part(b"a: 1\n") + synthetic_part(data) + synthetic_part(b"z: 2\n")
        if list(yaml.load_all(joined, Loader=yaml.CSafeLoader)) != [{"a": 1}, *docs, {"z": 2}]:
            return False
        ryml.parse_in_arena(data)
    except Exception:  # noqa: BLE001 - any failure rules the file out
        return False
    return True


def cmd_synth(a: argparse.Namespace) -> None:
    import ryml

    import yaml

    work, corpus = Path(a.work), Path(a.corpus)
    files = [f for _, f in unique_files(load_sources(work))[0]]
    by_fam: dict[str, list[str]] = {}
    for f in files:
        if 200 <= f["bytes"] <= 200_000:
            by_fam.setdefault(f["family"], []).append(f["path"])
    # Round robin over the families, files in path order, so every stream mixes all of them.
    queues = [sorted(by_fam.get(fam, [])) for fam in SYNTH_FAMILIES]
    order: list[str] = []
    while any(queues):
        order.extend(q.pop(0) for q in queues if q)
    parts: list[str] = []
    total = 0
    for p in order:
        data = (corpus / p).read_bytes()
        if data.startswith(b"\xef\xbb\xbf") or any(line.startswith(b"%") for line in data.split(b"\n")):
            continue
        if not part_ok(yaml, ryml, data):
            continue
        parts.append(p)
        total += len(data)
        if total >= 12_000_000:
            break
    log(f"{len(parts)} parts, {total} bytes")
    out = []
    for rel, size in SYNTH_SIZES:
        data = build_synthetic(corpus, parts, size)
        ryml.parse_in_arena(data)  # raises if the stream as a whole is not accepted
        (corpus / rel).parent.mkdir(parents=True, exist_ok=True)
        (corpus / rel).write_bytes(data)
        out.append(
            {
                "path": rel,
                "family": "synthetic",
                "target_bytes": size,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "parts": parts,
            }
        )
        log(rel, len(data))
    (work / "synthetic.json").write_text(json.dumps(out, indent=1))


def raw_url(name: str, commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{name}/{commit}/{urllib.parse.quote(path)}"


def cmd_manifest(a: argparse.Namespace) -> None:
    work, corpus = Path(a.work), Path(a.corpus)
    status: dict[str, dict] = {}
    if a.status:
        for line in Path(a.status).read_text().splitlines():
            r = json.loads(line)
            status[r["path"]] = r
    recs = load_sources(work)
    failed = [{"repo": r["repo"], "error": r["error"]} for r in recs if "error" in r]
    kept, dropped = unique_files(recs)
    sources, files = {}, []
    for r, f in kept:
        if (corpus / f["path"]).stat().st_size != f["bytes"]:
            raise SystemExit(f"{f['path']}: size differs from the source record")
        e = {
            "path": f["path"],
            "family": f["family"],
            "source": r["name"],
            "url": f.get("url") or raw_url(r["name"], r["commit"], f["repo_path"]),
            "sha256": f["sha256"],
            "bytes": f["bytes"],
        }
        if f.get("extract"):
            e["extract"] = f["extract"]
        st = status.get(f["path"])
        if st is not None:
            e["safe_load_all"] = "ok" if st["error"] is None else st["error"]["type"]
        files.append(e)
        sources[r["name"]] = {"repo": r["repo"], "commit": r["commit"], "licence": r["licence"]}
    synth = json.loads((work / "synthetic.json").read_text()) if (work / "synthetic.json").exists() else []
    for s in synth:
        st = status.get(s["path"])
        if st is not None:
            s["safe_load_all"] = "ok" if st["error"] is None else st["error"]["type"]
    files.sort(key=lambda e: e["path"])
    man = {
        "schema": 1,
        "about": "pyroid.yaml real-file corpus: files at pinned commits, rebuilt by bench/yaml/fetch_corpus.py. "
        "safe_load_all: 'ok' or the exception class yaml.safe_load_all(bytes) raised (PyYAML 6.0.3, "
        "pure-Python SafeLoader).",
        "yaml_test_suite": YAML_TEST_SUITE,
        "sources": dict(sorted(sources.items())),
        "files": files,
        "synthetic": synth,
        "skipped_sources": failed,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # One entry per line: diffable, and small enough to review.
    lines = ["{", f' "schema": {man["schema"]},', f' "about": {json.dumps(man["about"])},']
    lines += [f' "yaml_test_suite": {json.dumps(YAML_TEST_SUITE)},', ' "sources": {']
    items = list(man["sources"].items())
    lines += [
        f"  {json.dumps(k)}: {json.dumps(v)}{',' if i < len(items) - 1 else ''}"
        for i, (k, v) in enumerate(items)
    ]
    lines += [" },", ' "files": [']
    lines += [f"  {json.dumps(e)}{',' if i < len(files) - 1 else ''}" for i, e in enumerate(files)]
    lines += [" ],", ' "synthetic": [']
    lines += [f"  {json.dumps(s)}{',' if i < len(synth) - 1 else ''}" for i, s in enumerate(synth)]
    lines += [" ],", f' "skipped_sources": {json.dumps(failed)}', "}"]
    out.write_text("\n".join(lines) + "\n")
    json.loads(out.read_text())
    log(
        f"{len(files)} files ({len(dropped)} duplicates dropped), {len(sources)} sources, {len(synth)} synthetic -> {out}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["search", "github", "hf", "dedup", "synth", "manifest"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--corpus")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="github: first N repositories only (testing)")
    ap.add_argument("--status", help="manifest: record_oracle.py index.jsonl")
    ap.add_argument("--out", help="manifest: output path")
    a = ap.parse_args()
    steps = {
        "search": cmd_search,
        "github": cmd_github,
        "hf": cmd_hf,
        "dedup": cmd_dedup,
        "synth": cmd_synth,
        "manifest": cmd_manifest,
    }
    steps[a.step](a)


if __name__ == "__main__":
    main()
