#!/usr/bin/env python3
"""Record PyYAML 6.0.3's results for every file of a YAML corpus: the oracle for pyroid.yaml's parity tests.

    python -P bench/yaml/record_oracle.py /target/data/corpus /target/data/oracle/corpus --jobs 16
    python -P bench/yaml/record_oracle.py /target/data/yaml-test-suite/data-2022-01-17 \
        /target/data/oracle/yaml-test-suite --name in.yaml --suite

Needs ``pyyaml==6.0.3`` with its libyaml extension. Runs under ``PYTHONHASHSEED=0`` (it re-executes
itself with it when hash randomisation is on), so the ``sort_keys=False`` dumps of ``!!set`` values,
whose order is the set's iteration order, are reproducible. Each input file is read as bytes and
handed to PyYAML as bytes (marks are named ``<byte string>``). For each file REL:

* ``OUT/load/REL.json``: ``yaml.load_all(data, Loader=yaml.SafeLoader)`` (the pure-Python loader) as
  :func:`canonical` JSON: ``{"docs": [...], "error": null | {...}}``. ``docs`` are the documents the
  generator yielded, also those before an exception. ``error`` has the class (``module.qualname``),
  ``str(exc)`` and, for ``MarkedYAMLError`` and ``ReaderError``, their fields.
* ``OUT/cload/REL.json``: the same for ``yaml.CSafeLoader``, written only when it differs.
* ``OUT/dump/OPT/REL``: ``yaml.safe_dump_all(docs, **OPTIONS[OPT])`` as UTF-8, for files that loaded
  without error. ``OUT/cdump/OPT/REL``: ``yaml.dump_all(docs, Dumper=yaml.CSafeDumper, **OPTIONS[OPT])``,
  written only when it differs from the pure-Python dump.
* ``OUT/index.jsonl``: one line per file, sorted by path: sha256, size, document count, error, value
  features (sets, pairs, binary, timestamps, shared nodes), digests of every dump, the CSafeLoader
  and CSafeDumper verdicts, and for files that fail to load the outcome with ``str`` and stream input.
* ``OUT/csafe-differences.json``: every file where CSafeLoader's result differs from SafeLoader's,
  with the kind of difference; ``OUT/meta.json``: versions, options, counts.

The canonical encoding (:func:`canonical`) is plain JSON that keeps Python types apart: ``None``,
``bool``, ``int`` (bare JSON integers; ``{"$int": hex}`` past 4000 digits) and ``str`` are bare; floats
are ``{"$float": repr(x)}``; ``{"$bytes": base64}``, ``{"$date": iso}``, ``{"$datetime": iso,
"$tz": repr(tzinfo) | null}``, ``{"$tuple": [...]}`` (``!!omap``/``!!pairs`` give lists of tuples),
``{"$dict": [[key, value], ...]}`` in insertion order, ``{"$set": [...]}`` sorted by the elements'
encoding; lists are bare JSON arrays. A list, dict or set reached more than once (an alias, or a merge
key's shared values) carries ``"$id": n`` at its first visit (as ``{"$list": [...], "$id": n}`` for a
list) and is ``{"$ref": n}`` afterwards, which also covers recursive structures; n counts from 0 in
visiting order. Anything else is ``{"$other": type, "repr": repr(x)}``.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import io
import json
import math
import os
import pickle
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

OPTIONS: dict[str, dict] = {
    "default": {},
    "flow-none": {"default_flow_style": None},
    "sort-false": {"sort_keys": False},
    "unicode": {"allow_unicode": True},
    "indent4-w120": {"indent": 4, "width": 120},
    "explicit-start": {"explicit_start": True},
}
# Files above this size are dumped in separate tasks (one per option set) from a pickle of the documents.
SPLIT_BYTES = 4_000_000


class _Encoder:
    def __init__(self, root: object) -> None:
        self.shared = _shared_ids(root)
        self.ids: dict[int, int] = {}
        self.features: set[str] = {"shared"} if self.shared else set()

    def enc(self, o: object) -> object:
        t = type(o)
        if o is None or t is bool or t is str:
            return o
        if t is int:
            if abs(o) >= 10**4000:
                return {"$int": hex(o)}
            return o
        if t is float:
            if math.isnan(o) or math.isinf(o):
                self.features.add("float-special")
            return {"$float": repr(o)}
        if t is list or t is dict or t is set:
            k = id(o)
            n = None
            if k in self.shared:
                if k in self.ids:
                    return {"$ref": self.ids[k]}
                n = self.ids[k] = len(self.ids)
            if t is list:
                body: object = [self.enc(x) for x in o]
                if n is None:
                    return body
                return {"$list": body, "$id": n}
            if t is dict:
                out = {"$dict": [[self.enc(a), self.enc(b)] for a, b in o.items()]}
            else:
                self.features.add("set")
                items = [self.enc(x) for x in o]
                out = {"$set": sorted(items, key=lambda x: json.dumps(x, sort_keys=True))}
            if n is not None:
                out["$id"] = n
            return out
        if t is tuple:
            self.features.add("pairs")
            return {"$tuple": [self.enc(x) for x in o]}
        if t is bytes:
            self.features.add("binary")
            return {"$bytes": base64.b64encode(o).decode("ascii")}
        if t is datetime.datetime:
            self.features.add("datetime")
            return {"$datetime": o.isoformat(), "$tz": None if o.tzinfo is None else repr(o.tzinfo)}
        if t is datetime.date:
            self.features.add("date")
            return {"$date": o.isoformat()}
        self.features.add("other")
        return {"$other": f"{t.__module__}.{t.__qualname__}", "repr": repr(o)}


def _shared_ids(root: object) -> set[int]:
    counts: dict[int, int] = {}
    stack = [root]
    while stack:
        o = stack.pop()
        t = type(o)
        if t is list or t is dict or t is set:
            k = id(o)
            if k in counts:
                counts[k] += 1
                continue
            counts[k] = 1
            if t is dict:
                for a, b in o.items():
                    stack.append(a)
                    stack.append(b)
            else:
                stack.extend(o)
        elif t is tuple:
            stack.extend(o)
    return {k for k, c in counts.items() if c > 1}


def error_record(e: BaseException) -> dict:
    t = type(e)
    rec: dict = {"type": f"{t.__module__}.{t.__qualname__}", "str": str(e)}
    if isinstance(e, yaml.MarkedYAMLError):

        def mark(m: yaml.Mark | None) -> list | None:
            return None if m is None else [m.name, m.index, m.line, m.column]

        rec["fields"] = {
            "context": e.context,
            "context_mark": mark(e.context_mark),
            "problem": e.problem,
            "problem_mark": mark(e.problem_mark),
            "note": e.note,
        }
    elif isinstance(e, yaml.reader.ReaderError):
        ch = e.character
        rec["fields"] = {
            "name": e.name,
            "position": e.position,
            "character": ch if isinstance(ch, int) else repr(ch),
            "encoding": e.encoding,
            "reason": e.reason,
        }
    return rec


def run_load(data: object, loader: type) -> tuple[list, dict | None]:
    docs: list = []
    try:
        for d in yaml.load_all(data, Loader=loader):
            docs.append(d)  # noqa: PERF402 - keeps the documents yielded before an exception
    except Exception as e:  # noqa: BLE001 - the oracle records whatever PyYAML raises
        return docs, error_record(e)
    return docs, None


def canonical(docs: list, error: dict | None) -> tuple[str, set[str]]:
    """Return the canonical JSON of a load result (see the module docstring) and the value features."""
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 20000))
    try:
        enc = _Encoder(docs)
        body = [enc.enc(d) for d in docs]
    finally:
        sys.setrecursionlimit(old)
    text = json.dumps(
        {"docs": body, "error": error}, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    )
    return text + "\n", enc.features


def run_dump(docs: list, opts: dict, c: bool) -> tuple[bytes | None, dict | None]:
    try:
        if c:
            s = yaml.dump_all(docs, Dumper=yaml.CSafeDumper, **opts)
        else:
            s = yaml.safe_dump_all(docs, **opts)
    except Exception as e:  # noqa: BLE001 - recorded like a load error
        return None, error_record(e)
    return s.encode("utf-8", "surrogatepass"), None


def _write(path: Path, data: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _strip_marks(err: dict) -> tuple:
    f = err.get("fields") or {}
    return (err["type"], f.get("context"), f.get("problem"), f.get("note"))


def compare(py: tuple[str, dict | None, list], c: tuple[str, dict | None, list]) -> list[str]:
    """Kinds of difference between two load results (text, error, docs-json)."""
    kinds = []
    ptext, perr, pdocs = py
    ctext, cerr, cdocs = c
    if ptext == ctext:
        return kinds
    if (perr is None) != (cerr is None):
        kinds.append("ok-vs-error")
    if pdocs != cdocs:
        kinds.append("docs" if perr is None and cerr is None else "docs-before-error")
    if perr is not None and cerr is not None and perr != cerr:
        pf, cf = perr.get("fields"), cerr.get("fields")
        if perr["type"] != cerr["type"]:
            kinds.append("class")
        elif perr["str"] == cerr["str"]:
            pass
        elif not (pf and cf and "problem" in pf and "problem" in cf) or _strip_marks(perr) != _strip_marks(
            cerr
        ):
            kinds.append("message")
        else:
            pm = [(m or [None])[1:] for m in (pf["context_mark"], pf["problem_mark"])]
            cm = [(m or [None])[1:] for m in (cf["context_mark"], cf["problem_mark"])]
            kinds.append("mark-position" if pm != cm else "snippet-only")
    return kinds or ["other"]


def _docs_json(text: str) -> str:
    return text[: text.rindex(',"error":')]


def dump_task(out: Path, rel: str, opt_names: list[str], docs: list | None, pickle_path: str | None) -> dict:
    if docs is None:
        # Only ever a pickle this run wrote itself (load_task, OUT/tmp/), deleted at the end of the run.
        with open(pickle_path, "rb") as f:
            docs = pickle.load(f)
    res: dict = {"dump": {}, "cdump": {}}
    for name in opt_names:
        opts = OPTIONS[name]
        t0 = time.perf_counter()
        pyb, pyerr = run_dump(docs, opts, c=False)
        dt = time.perf_counter() - t0
        if pyerr is None:
            _write(out / "dump" / name / rel, pyb)
            res["dump"][name] = {"sha256": _sha(pyb), "bytes": len(pyb), "s": round(dt, 4)}
        else:
            res["dump"][name] = {"error": pyerr}
        cb, cerr = run_dump(docs, opts, c=True)
        if (cb is not None and cb == pyb) or (cb is None and pyb is None and cerr == pyerr):
            res["cdump"][name] = "same"
        else:
            res["cdump"][name] = {"error": cerr} if cerr else {"sha256": _sha(cb), "bytes": len(cb)}
            if cb is not None:
                _write(out / "cdump" / name / rel, cb)
    return res


def load_task(root: str, out: str, rel: str, suite: bool) -> dict:
    root_p, out_p = Path(root), Path(out)
    src = root_p / rel
    data = src.read_bytes()
    rec: dict = {"path": rel, "sha256": _sha(data), "bytes": len(data)}
    if suite:
        rec["expect_error"] = (src.parent / "error").exists()
    t0 = time.perf_counter()
    docs, err = run_load(data, yaml.SafeLoader)
    rec["load_s"] = round(time.perf_counter() - t0, 4)
    text, feats = canonical(docs, err)
    _write(out_p / "load" / (rel + ".json"), text)
    rec["docs"] = len(docs)
    rec["error"] = None if err is None else {"type": err["type"], "str": err["str"]}
    rec["features"] = sorted(feats)
    rec["load_sha256"] = _sha(text.encode())

    t0 = time.perf_counter()
    cdocs, cerr = run_load(data, yaml.CSafeLoader)
    rec["cload_s"] = round(time.perf_counter() - t0, 4)
    ctext, _ = canonical(cdocs, cerr)
    kinds = compare((text, err, _docs_json(text)), (ctext, cerr, _docs_json(ctext)))
    if kinds:
        _write(out_p / "cload" / (rel + ".json"), ctext)
        rec["csafe"] = {
            "kinds": kinds,
            "error": None if cerr is None else {"type": cerr["type"], "str": cerr["str"]},
            "docs": len(cdocs),
        }
    else:
        rec["csafe"] = "same"
    del cdocs

    if err is not None:
        variants = {}
        try:
            s = data.decode("utf-8")
        except UnicodeDecodeError:
            s = None
        for name, inp in (("str", s), ("stream", io.BytesIO(data))):
            if inp is None:
                continue
            vdocs, verr = run_load(inp, yaml.SafeLoader)
            vtext, _ = canonical(vdocs, verr)
            variants[name] = {"same_docs": _docs_json(vtext) == _docs_json(text), "error": verr}
        rec["variants"] = variants
        return rec
    if len(data) > SPLIT_BYTES:
        pk = out_p / "tmp" / (rel + ".pickle")
        pk.parent.mkdir(parents=True, exist_ok=True)
        with open(pk, "wb") as f:
            pickle.dump(docs, f, protocol=pickle.HIGHEST_PROTOCOL)
        rec["_pickle"] = str(pk)
        return rec
    rec.update(dump_task(out_p, rel, list(OPTIONS), docs, None))
    return rec


def find_inputs(root: Path, name: str | None) -> list[str]:
    rels = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not os.path.islink(os.path.join(dirpath, d)))
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            if os.path.islink(p) or (name and fn != name):
                continue
            rels.append(os.path.relpath(p, root))
    return sorted(rels)


def main() -> int:
    if sys.flags.hash_randomization and os.environ.get("PYTHONHASHSEED") != "0":
        os.execve(
            sys.executable, [sys.executable, "-P", "-s", *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"}
        )
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("corpus", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--name", help="only files with this name (e.g. in.yaml)")
    ap.add_argument(
        "--suite",
        action="store_true",
        help="yaml-test-suite layout: note whether an 'error' file is next to the input",
    )
    ap.add_argument("--only", action="append", help="only paths starting with this prefix (repeatable)")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    if not yaml.__with_libyaml__:
        raise SystemExit("PyYAML was built without libyaml: CSafeLoader is missing")

    rels = find_inputs(a.corpus, a.name)
    if a.only:
        rels = [r for r in rels if r.startswith(tuple(a.only))]
    sizes = {r: (a.corpus / r).stat().st_size for r in rels}
    order = sorted(rels, key=lambda r: -sizes[r])
    a.out.mkdir(parents=True, exist_ok=True)
    print(
        f"{len(rels)} files, {sum(sizes.values()) / 1e6:.1f} MB, {a.jobs} jobs", file=sys.stderr, flush=True
    )
    t0 = time.time()
    recs: dict[str, dict] = {}
    with ProcessPoolExecutor(a.jobs) as pool:
        futs = {pool.submit(load_task, str(a.corpus), str(a.out), r, a.suite): r for r in order}
        dumps = {}
        for i, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001 - a recorder bug for one file; listed in meta.json
                rec = {"path": r, "recorder_error": f"{type(e).__name__}: {e}"}
            recs[r] = rec
            if "_pickle" in rec:
                for name in OPTIONS:
                    dumps[pool.submit(dump_task, a.out, r, [name], None, rec["_pickle"])] = r
            if i % 1000 == 0:
                print(f"{i}/{len(rels)} loaded, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        for fut in as_completed(dumps):
            r = dumps[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001 - as above
                recs[r]["recorder_error"] = f"dump: {type(e).__name__}: {e}"
                continue
            for k in ("dump", "cdump"):
                recs[r].setdefault(k, {}).update(res[k])
    for r in recs.values():
        if "_pickle" in r:
            Path(r.pop("_pickle")).unlink(missing_ok=True)
            r["dump"] = {k: r["dump"][k] for k in OPTIONS if k in r.get("dump", {})}
            r["cdump"] = {k: r["cdump"][k] for k in OPTIONS if k in r.get("cdump", {})}

    with open(a.out / "index.jsonl", "w") as f:
        for r in sorted(recs):
            f.write(json.dumps(recs[r], ensure_ascii=True) + "\n")
    diffs = [
        {
            "path": r["path"],
            "kinds": r["csafe"]["kinds"],
            "safe": r["error"],
            "csafe": r["csafe"]["error"],
            "docs": [r["docs"], r["csafe"]["docs"]],
        }
        for r in (recs[k] for k in sorted(recs))
        if isinstance(r.get("csafe"), dict)
    ]
    (a.out / "csafe-differences.json").write_text(json.dumps(diffs, indent=1, ensure_ascii=True) + "\n")
    cdump_diffs = sorted(
        r["path"] for r in recs.values() if any(v != "same" for v in r.get("cdump", {}).values())
    )
    ok = [r for r in recs.values() if r.get("error") is None and "recorder_error" not in r]
    meta = {
        "python": sys.version,
        "platform": platform.platform(),
        "pyyaml": yaml.__version__,
        "libyaml": yaml._yaml.get_version_string(),
        "hash_seed": os.environ.get("PYTHONHASHSEED"),
        "options": OPTIONS,
        "input": "bytes",
        "corpus": str(a.corpus),
        "files": len(recs),
        "bytes": sum(sizes.values()),
        "load_ok": len(ok),
        "load_error": len(recs) - len(ok),
        "recorder_errors": sorted(r["path"] for r in recs.values() if "recorder_error" in r),
        "csafe_differs": len(diffs),
        "cdump_differs": len(cdump_diffs),
        "cdump_differs_paths": cdump_diffs,
        "seconds": round(time.time() - t0, 1),
    }
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(
        json.dumps({k: v for k, v in meta.items() if k not in ("options", "cdump_differs_paths")}),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
