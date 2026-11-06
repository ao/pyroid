"""The ``pyroid`` command line (also ``python -m pyroid``).

Streams Parquet or JSONL in Arrow batches; input may be a file or a directory
of files (read in name order). Output is one file. A file's format follows its
name: ``.parquet``/``.pq`` is Parquet; ``.jsonl``/``.ndjson``/``.json`` is JSON
Lines, optionally compressed as ``.gz``, ``.bz2``, ``.zst`` or ``.lz4``.

    pyroid clean IN OUT --recipe fineweb [--threads N] [--text-column text]
                 [--excluded DROPPED] [--stats]
    pyroid langid IN OUT [--model lid.176.bin] [--keep en,fr --min-score 0.65] [--top-k K]
    pyroid dedup IN OUT [--method minhash|exact] [--normalize] [--excluded DROPPED]
    pyroid translit IN OUT [--mode anyascii|unidecode] [--output-column COL]
    pyroid run [-p ftfy,fasttext] (SCRIPT | -m MODULE) [ARGS...]

``clean --recipe fineweb`` is the benchmark's FineWeb chain
(``pyroid.integrations.recipes.FineWeb``): ftfy -> fastText LID (en > 0.65) -> Gopher
repetition/quality, C4 and FineWeb filters (datatrove's decisions) -> MinHash dedup over the
survivors. Kept rows get the cleaned text and datatrove's ``language`` / ``language_score``
columns (replacing any of that name); ``--excluded`` gets the dropped rows with a
``filter_reason`` (``lid``, ``<stage>:<datatrove reason>``, ``minhash``). Needs pyarrow and numpy:
``pip install 'pyroid[cli]'``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyroid

RECIPES = ("fineweb",)
# The suffixes pyroid.integrations._io reads and writes (only JSON Lines can be compressed).
_FORMATS = ".parquet/.pq or .jsonl/.ndjson/.json[.gz|.bz2|.zst|.lz4]"


def _need_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise SystemExit(
            "pyroid: this command needs pyarrow (and numpy): pip install 'pyroid[cli]'"
        ) from None


class _Stats:
    def __init__(self, command: str) -> None:
        self.command = command
        self.docs_in = 0
        self.docs_out = 0
        self.removed: Counter[str] = Counter()
        self.t0 = time.perf_counter()

    def as_dict(self) -> dict[str, Any]:
        wall = time.perf_counter() - self.t0
        by_stage: Counter[str] = Counter()
        for reason, n in self.removed.items():
            by_stage[reason.split(":", 1)[0]] += n
        return {
            "command": self.command,
            "docs_in": self.docs_in,
            "docs_out": self.docs_out,
            "removed": dict(sorted(self.removed.items(), key=lambda kv: (-kv[1], kv[0]))),
            "removed_by_stage": dict(sorted(by_stage.items(), key=lambda kv: (-kv[1], kv[0]))),
            "wall_s": round(wall, 3),
            "pyroid": pyroid.__version__,
        }

    def report(self, a: argparse.Namespace) -> None:
        d = self.as_dict()
        if a.stats_json:
            Path(a.stats_json).write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")
        if not a.stats:
            return
        w = sys.stderr.write
        w(f"pyroid {d['command']}: {d['docs_in']:,} in, {d['docs_out']:,} out, {d['wall_s']} s\n")
        width = max((len(k) for k in d["removed"]), default=0)
        for k, n in d["removed"].items():
            pct = 100 * n / d["docs_in"] if d["docs_in"] else 0.0
            w(f"  dropped {k:<{width}}  {n:>9,}  {pct:5.1f}%\n")


def _check_paths(a: argparse.Namespace) -> None:
    out = Path(a.output)
    if out.exists() and not a.overwrite:
        raise SystemExit(f"pyroid: {out} exists (pass --overwrite to replace it)")
    for p in (a.output, getattr(a, "excluded", None)):
        if p and Path(p).resolve() == Path(a.input).resolve():
            raise SystemExit("pyroid: an output path is the input path")


def _writers(a: argparse.Namespace) -> tuple[Any, Any]:
    from pyroid.integrations import _io

    out = _io.Writer(a.output, compression=a.compression)
    excl = (
        _io.Writer(a.excluded, compression=a.compression) if getattr(a, "excluded", None) else None
    )
    return out, excl


def _text(table: Any, a: argparse.Namespace) -> Any:
    if a.text_column not in table.column_names:
        raise SystemExit(
            f"pyroid: no column {a.text_column!r} (columns: {', '.join(table.column_names)})"
        )
    return table.column(a.text_column)


def _run(a: argparse.Namespace, body: Any) -> int:
    """Shared driver: writers that are dropped on failure, then the stats."""
    _need_pyarrow()
    _check_paths(a)
    from pyroid.integrations import _io

    files = _io.input_files(a.input)
    stats = _Stats(a.command)
    out, excl = _writers(a)
    try:
        body(a, files, out, excl, stats)
    except BaseException:
        out.abort()
        if excl is not None:
            excl.abort()
        raise
    schema = getattr(stats, "schema", None)
    out.close(schema)
    if excl is not None:
        excl.close(getattr(stats, "excluded_schema", None))
    stats.report(a)
    return 0


# ------------------------------------------------------------------------------------------ clean


def _clean(a: argparse.Namespace, files: list[Path], out: Any, excl: Any, stats: Any) -> None:
    import pyarrow as pa
    import pyarrow.compute as pc

    from pyroid.integrations import _io, recipes

    recipe = recipes.FineWeb(
        lid_model=a.lid_model,
        threads=a.threads,
        minhash=None if a.no_dedup else recipes.MINHASH_CONFIG,
    )
    dedup = recipe.new_dedup()
    spool = spool_path = None
    try:
        for t in _io.read_tables(files, a.batch_size):
            ann = recipe.annotate(_text(t, a))
            t = _io.set_column(t, a.text_column, ann.text)
            t = _io.set_column(t, "language", ann.language)
            t = _io.set_column(t, "language_score", ann.language_score)
            stats.docs_in += t.num_rows
            drop = pc.invert(ann.keep)
            reasons = ann.reason.filter(drop)
            stats.removed.update(reasons.to_pylist())
            stats.schema = t.schema
            if excl is not None:
                dropped = t.filter(drop).append_column("filter_reason", reasons)
                stats.excluded_schema = dropped.schema
                excl.write(dropped)
            kept = t.filter(ann.keep)
            if dedup is None:
                out.write(kept)
                stats.docs_out += kept.num_rows
                continue
            dedup.add(kept.column(a.text_column))
            if spool is None:
                fd, spool_path = tempfile.mkstemp(
                    dir=a.tmp_dir or Path(a.output).resolve().parent,
                    prefix=".pyroid-",
                    suffix=".arrow",
                )
                os.close(fd)
                spool = pa.ipc.new_file(spool_path, kept.schema)
            spool.write_table(kept)
        if dedup is None or spool is None:
            return
        spool.close()
        spool = None
        keep = pa.array(dedup.keep_mask(), pa.bool_())
        with pa.memory_map(spool_path) as src:
            reader = pa.ipc.open_file(src)
            pos = 0
            for i in range(reader.num_record_batches):
                b = pa.Table.from_batches([reader.get_batch(i)])
                m = keep.slice(pos, b.num_rows)
                pos += b.num_rows
                out.write(b.filter(m))
                kept_n = pc.sum(m).as_py() or 0
                stats.docs_out += kept_n
                if excl is not None:
                    dups = b.filter(pc.invert(m))
                    excl.write(
                        dups.append_column(
                            "filter_reason", pa.array(["minhash"] * dups.num_rows, pa.string())
                        )
                    )
                stats.removed["minhash"] += b.num_rows - kept_n
        stats.removed = Counter({k: v for k, v in stats.removed.items() if v})
    finally:
        if spool is not None:
            spool.close()
        if spool_path and os.path.exists(spool_path):
            os.unlink(spool_path)


# ----------------------------------------------------------------------------------------- langid


def _langid(a: argparse.Namespace, files: list[Path], out: Any, excl: Any, stats: Any) -> None:
    import pyarrow as pa
    import pyarrow.compute as pc

    from pyroid.integrations import _io, recipes

    keep_langs = [x.strip() for x in a.keep.split(",") if x.strip()] if a.keep else None
    recipe = recipes.FineWeb(
        lid_model=a.model,
        languages=keep_langs,
        language_threshold=a.min_score if keep_langs else -math.inf,
        ftfy_config=False,
        filters=[],
        minhash=None,
        threads=a.threads,
    )
    for t in _io.read_tables(files, a.batch_size):
        text = _text(t, a)
        ann = recipe.annotate(text)
        t = _io.set_column(t, "language", ann.language)
        t = _io.set_column(t, "language_score", ann.language_score)
        if a.top_k > 1:
            labels, probs = recipe.model.predict(
                pc.replace_substring(text, "\n", " "), k=a.top_k, threads=a.threads
            )
            t = _io.set_column(
                t,
                "top_languages",
                pa.array(
                    [
                        None if lb is None else [x.removeprefix("__label__") for x in lb]
                        for lb in labels
                    ],
                    pa.list_(pa.string()),
                ),
            )
            t = _io.set_column(
                t,
                "top_language_scores",
                pa.array(
                    [None if p is None else [float(x) for x in p] for p in probs],
                    pa.list_(pa.float64()),
                ),
            )
        stats.docs_in += t.num_rows
        drop = pc.invert(ann.keep)
        reasons = ann.reason.filter(drop)
        stats.removed.update(reasons.to_pylist())
        stats.schema = t.schema
        if excl is not None:
            dropped = t.filter(drop).append_column("filter_reason", reasons)
            stats.excluded_schema = dropped.schema
            excl.write(dropped)
        kept = t.filter(ann.keep)
        stats.docs_out += kept.num_rows
        out.write(kept)


# ------------------------------------------------------------------------------------------ dedup


def _dedup(a: argparse.Namespace, files: list[Path], out: Any, excl: Any, stats: Any) -> None:
    import pyarrow as pa
    import pyarrow.compute as pc

    import pyroid.dedup
    from pyroid.integrations import _io

    def emit(t: Any, mask: Any) -> None:
        stats.docs_in += t.num_rows
        stats.schema = t.schema
        kept = t.filter(mask)
        stats.docs_out += kept.num_rows
        out.write(kept)
        n_drop = t.num_rows - kept.num_rows
        stats.removed[a.method] += n_drop
        if excl is not None:
            dropped = t.filter(pc.invert(mask))
            dropped = dropped.append_column(
                "filter_reason", pa.array([a.method] * n_drop, pa.string())
            )
            stats.excluded_schema = dropped.schema
            excl.write(dropped)

    if a.method == "exact":
        state = pyroid.dedup.ExactDedup(normalize=a.normalize)
        for t in _io.read_tables(files, a.batch_size):
            emit(t, pa.array(state.add(_text(t, a), threads=a.threads), pa.bool_()))
    else:
        cfg = {
            k: v
            for k, v in (
                ("n_grams", a.n_grams),
                ("num_buckets", a.num_buckets),
                ("hashes_per_bucket", a.hashes_per_bucket),
                ("seed", a.seed),
                ("hash_fc", a.hash_fc),
                ("precision", a.precision),
            )
            if v is not None
        }
        index = pyroid.dedup.MinHashDedup(threads=a.threads, **cfg)
        for t in _io.read_tables(files, a.batch_size):  # pass 1: signatures only
            index.add(_text(t, a))
        keep = pa.array(index.keep_mask(), pa.bool_())
        pos = 0
        for t in _io.read_tables(files, a.batch_size):  # pass 2: the rows
            emit(t, keep.slice(pos, t.num_rows))
            pos += t.num_rows
        if pos != len(keep):
            raise RuntimeError("the input changed between the two passes")
    stats.removed = Counter({k: v for k, v in stats.removed.items() if v})


# --------------------------------------------------------------------------------------- translit


def _translit(a: argparse.Namespace, files: list[Path], out: Any, excl: Any, stats: Any) -> None:
    import pyroid.translit
    from pyroid.integrations import _io

    for t in _io.read_tables(files, a.batch_size):
        text = _text(t, a)
        if a.mode == "anyascii":
            col = pyroid.translit.anyascii(text, threads=a.threads)
        else:
            col = pyroid.translit.unidecode(text, a.errors, a.replace_str, threads=a.threads)
        t = _io.set_column(t, a.output_column or a.text_column, col)
        stats.docs_in += t.num_rows
        stats.docs_out += t.num_rows
        stats.schema = t.schema
        out.write(t)


# ------------------------------------------------------------------------------------------- main


def _positive(s: str) -> int:
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return v


def _threads(s: str) -> int | None:
    v = int(s)
    if v < 0:
        raise argparse.ArgumentTypeError("must be >= 0 (0 = the whole pool)")
    return v or None


def build_parser() -> argparse.ArgumentParser:
    """The argument parser (``pyroid run`` is dispatched before it)."""
    ap = argparse.ArgumentParser(
        prog="pyroid",
        description="Rust-fast text cleaning for LLM datasets: Parquet/JSONL in and out.",
        epilog="Also: pyroid run [-p ftfy,fasttext] (SCRIPT | -m MODULE) [ARGS...] runs a "
        "program with pyroid.integrations.patch() applied.",
    )
    ap.add_argument("--version", action="version", version=f"pyroid {pyroid.__version__}")
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def common(p: argparse.ArgumentParser, excluded: bool = True) -> None:
        p.add_argument("input", help=f"input file ({_FORMATS}), or a directory of such files")
        p.add_argument("output", help=f"output file ({_FORMATS})")
        p.add_argument("--text-column", default="text", help="the text column (default: text)")
        p.add_argument(
            "--threads",
            type=_threads,
            default=None,
            help="pyroid threads (default 0: the whole pool, sized by PYROID_NUM_THREADS)",
        )
        p.add_argument(
            "--batch-size",
            type=_positive,
            default=10_000,
            help="rows per Arrow batch (default 10000)",
        )
        if excluded:
            p.add_argument("--excluded", metavar="PATH", help="also write the dropped rows here")
        p.add_argument("--stats", action="store_true", help="print kept/dropped counts by reason")
        p.add_argument("--stats-json", metavar="PATH", help="write the stats as JSON")
        p.add_argument("--compression", help="Parquet compression (default: pyarrow's, snappy)")
        p.add_argument("--overwrite", action="store_true", help="replace an existing output file")

    p = sub.add_parser("clean", help="run a cleaning recipe (ftfy, LID, quality filters, MinHash)")
    common(p)
    p.add_argument("--recipe", choices=RECIPES, required=True)
    p.add_argument(
        "--lid-model", help="fastText lid.176.bin (default: $PYROID_LID_MODEL or download)"
    )
    p.add_argument("--no-dedup", action="store_true", help="skip the MinHash stage")
    p.add_argument(
        "--tmp-dir", help="where survivors wait for the dedup decision (default: OUT's dir)"
    )
    p.set_defaults(body=_clean)

    p = sub.add_parser(
        "langid", help="add fastText language columns; optionally keep some languages"
    )
    common(p)
    p.add_argument("--model", help="a fastText model (default: lid.176.bin, as for clean)")
    p.add_argument("--keep", metavar="LANGS", help="comma-separated labels to keep, e.g. en,fr")
    p.add_argument(
        "--min-score", type=float, default=0.65, help="with --keep: datatrove's language_threshold"
    )
    p.add_argument(
        "--top-k", type=_positive, default=1, help="also add top_languages/_scores lists"
    )
    p.set_defaults(body=_langid)

    p = sub.add_parser(
        "dedup", help="drop near-duplicates (MinHash, datatrove's) or exact duplicates"
    )
    common(p)
    p.add_argument("--method", choices=("minhash", "exact"), default="minhash")
    p.add_argument(
        "--normalize", action="store_true", help="exact: compare simplify_text()-normalized"
    )
    g = p.add_argument_group("MinHash (datatrove MinhashConfig; defaults: datatrove's)")
    g.add_argument("--n-grams", type=_positive)
    g.add_argument("--num-buckets", type=_positive)
    g.add_argument("--hashes-per-bucket", type=_positive)
    g.add_argument("--seed", type=int)
    g.add_argument("--hash-fc", choices=("xxhash", "sha1"))
    g.add_argument("--precision", type=int, choices=(32, 64))
    p.set_defaults(body=_dedup)

    p = sub.add_parser("translit", help="transliterate to ASCII (anyascii data, ISC)")
    common(p, excluded=False)
    p.add_argument("--mode", choices=("anyascii", "unidecode"), default="anyascii")
    p.add_argument("--errors", default="ignore", help="unidecode: ignore|replace|strict|preserve")
    p.add_argument("--replace-str", default="?", help="unidecode: replacement for errors=replace")
    p.add_argument("--output-column", help="write here instead of replacing the text column")
    p.set_defaults(body=_translit)
    return ap


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``pyroid`` console script."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "run":
        from pyroid.integrations import run

        return run(args[1:])
    a = build_parser().parse_args(args)
    try:
        return _run(a, a.body)
    except (FileNotFoundError, ValueError) as e:
        if os.environ.get("PYROID_DEBUG"):
            raise
        print(f"pyroid {a.command}: {e}", file=sys.stderr)  # noqa: T201
        return 1


if __name__ == "__main__":
    sys.exit(main())
