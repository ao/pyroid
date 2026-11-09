# pyroid parity suites

Differential tests that compare any implementation against the incumbents pyroid replaces, using pinned inputs
and pinned reference outputs. You choose the implementation by module path (`--impl`), so a new module plugs in
without editing the suites. Every run writes `parity-report.json`, which feeds the docs Compatibility page.

| suite | reference (oracle) | checks | tolerance |
|---|---|---|---|
| `ftfy` | ftfy 6.3.1 (live + pinned outputs) | 161-case golden corpus (incl. ftfy's known failures), Hypothesis fuzzing with lone surrogates / surrogate pairs / mojibake / entities / controls, `fix_text_batch`, dev slice (10k) + adversarial set (3.2k) | byte-identical, same exception type |
| `fasttext` | fasttext-predict 0.9.2.4 on CPython 3.13 | top-5 labels + probs on lid.176.bin and lid.176.ftz for 13.2k docs, live Hypothesis (incl. surrogates, NUL, empty), newline → `ValueError` | labels identical, \|Δp\| ≤ 1e-6 |
| `filters` | datatrove 0.10.1 (+ spaCy 3.8.18 tokenizer) | Gopher repetition, Gopher quality, C4 (default and FineWeb config), FineWeb quality on 13.2k docs | same keep/drop, same reason string, same rewritten text (C4), same exception |
| `dedup` | datatrove 0.10.1 MinHash, FineWeb config (sha1/64, 14×8, 5-grams) | 112 signature values per doc; removed-doc set over dev + adversarial (445 removals, mostly injected near-dups) | identical |
| `metrics` | rouge-score 0.1.2 (+ NLTK 3.10.4 Porter), sacrebleu 2.6.0 | ROUGE-1/2/3/4/L/Lsum on CNN/DM test (11.5k lead-3 vs highlights, 2k whole articles) and SummEval (1.6k system summaries, 11 refs, `score_multi`), with and without stemming; BLEU (13a/intl/char/none/zh/trg_lang, smoothing, lowercase, effective order, max order, `None` reference holes) and chrF/chrF++ per WMT21 system (141 corpora) and per segment (163k); live Hypothesis vs both incumbents (sacrebleu only on GIL builds: it loads lxml, which re-enables the GIL) | identical `repr`/`str`/`format`/signature strings (exact floats) |
| `json_repair` | json_repair 0.64.5 (live + calls recorded from its own test suite + pinned digests) | 672 public calls from json_repair's 466 upstream tests ([`data/json_repair-0.64.5`](data/json_repair-0.64.5/README.md)), 20k generated broken LLM outputs × 8 option sets, Hypothesis grammar fuzzing with lone surrogates, `loads` / `load` / `from_file` | `str` byte-identical; objects and repair logs type-exact (key order, int vs float, NaN); same exception type and message |
| `trafilatura` | one mode per release (`--trafilatura-compat`): trafilatura 2.3.1, or 1.11.0 as `datatrove[processing]` resolves it, each with lxml 6.1.3 + justext 3.0.2 (live + pinned digests + calls recorded from its own tests) | 7,192 Common Crawl pages of 3 crawls (2019, 2023, 2026) × 7 `extract` option sets, datatrove's call among them (stateful: dedup in page order), also through `extract_batch`, and 2 `bare_extraction` sets; the public calls of the release's own tests, replayed test by test (2.3.1: 4,553 from 3,938 tests, [`data/trafilatura-2.3.1`](data/trafilatura-2.3.1/README.md); 1.11.0: 633 from 29 tests, [`data/trafilatura-1.11.0`](data/trafilatura-1.11.0/README.md)); Hypothesis grammar-generated HTML; unported options in fresh processes | text byte-identical (`None` is not `""`), `Document` fields identical, same exception type and message; exactly one `pyroid.FallbackWarning` per delegated option, none for the ported ones |
| `warc` | warcio 1.8.2 (+ brotlicffi 1.2.1.0) and datatrove 0.10.1's `WarcReader` (+ faust-cchardet 3.2.0, python-magic 0.4.27) (live + pinned digests) | 34 Common Crawl slices (WARC of 16 crawls 2013–2026, WET, WAT, the 2012 ARC crawl, truncated, empty; 18.6k records) and warcio's 16 test files, [`data/warcio-1.8.2`](data/warcio-1.8.2/README.md); 1,085 seeded synthetic files (framing, gzip damage, lengths, headers, HTTP, chunked, encodings, digests, ARC, charsets, big records) and Hypothesis WARC streams, each under up to 29 read plans × options × openers (file, BytesIO, unseekable, short reads, `gzip.GzipFile`, datatrove's open); `process_record` dicts, `iter_documents`, `iter_batches`, `WarcReader` runs | every record field, read, offset, stderr / log message and exception (with what follows it) identical, event by event; Documents identical |

## Plugging in a module

1. Implement the suite contract in your module. The contracts are in the docstring of
   [`pyroid_parity/impl.py`](pyroid_parity/impl.py):
   - `ftfy`: ftfy's own API.
   - `fasttext`: `load_model(path).predict(text, k, threshold) -> (labels, probs)`, like fastText.
   - `filters`: `gopher_quality(texts, **config) -> (keep, reasons[, texts_out])` and so on, mask-shaped.
   - `dedup`: `minhash_signatures(texts, **config)` and `minhash_dedup(texts, **config) -> removed mask`.
   - `trafilatura`: trafilatura 2.3.1's `extract` and `bare_extraction` (same signatures plus a keyword-only `compat="2.3.1"` last), `reset_caches()` (empties the dedup cache `deduplicate=True` uses) and optionally `extract_batch(htmls, *, threads=None, compat="2.3.1", **kwargs)`; a `pyroid.integrations` shim keeps trafilatura's exact signatures.
   - `warc`: warcio 1.8.2's reader classes (`ArchiveIterator` and friends, `StatusAndHeaders`, `DigestChecker`, `ArchiveLoadFailed`, `WARCIO_VERSION`), `iter_documents(source, *, compression="infer")` (datatrove's read path) and optionally `iter_batches`; `<top>.integrations.datatrove.WarcReader` is checked against datatrove's.

   Function names, configs and the MinHash parameters are in [`pyroid_parity/specs.py`](pyroid_parity/specs.py).
   If your API differs, add an adapter in `pyroid_parity/adapters/<suite>_<module>.py`, which is how datatrove is wrapped.
2. Build a wheel into `dist/` (for example `maturin build --release --out dist`), then run the suite against the
   installed wheel, using the parity env plus your wheel (on Linux, or in a Linux container):

   ```bash
   cd parity
   uv run --frozen --with ../dist/pyroid-*.whl \
     pytest -m ftfy --impl pyroid.ftfy --parity-report parity-ftfy.json
   ```
   - `--impl` takes `pyroid.ftfy`, `pyroid.fasttext`, `pyroid.filters`, `pyroid.dedup` (the last dotted component names the suite), or the explicit form `SUITE=MODULE`.
   - Without `--impl`, a suite self-checks its reference against the pinned outputs.
   - Use `-m <suite>` or `tests/test_<suite>.py` to select suites.
3. For free-threaded builds and 3.15, sync a second env, e.g. `UV_PROJECT_ENVIRONMENT=.venv-314t uv sync --frozen -p 3.14t --no-default-groups`.
   - The core deps (ftfy, pytest, hypothesis, pyarrow, numpy) install on 3.12–3.15t.
   - The pinned references let the fasttext/filters/dedup checks run where the oracles don't install. fasttext-predict stops at cp313/cp313t, and datatrove's extras aren't needed.
   - The live-oracle checks skip there and say so in the report.
4. `HYPOTHESIS_PROFILE=ci` gives 2000 examples per property (default `dev`: 200).
5. To read the report:
   - `python -m pyroid_parity.report render parity-report.json` prints a Markdown table and, under GitHub Actions, appends it to `$GITHUB_STEP_SUMMARY`. It is the only writer of the summary: pytest runs never touch it.
   - `validate` checks it against [`schema/parity-report.schema.json`](schema/parity-report.schema.json).
   - `merge a.json b.json -o all.json` combines interpreters.

`--parity-limit N` restricts every corpus to its first N docs (0 = all). Reference self-checks of the slow suites
(filters, dedup) sample 1000 docs by default. The removed-set check is corpus-level, so it runs only with `--parity-limit 0`.

## Report schema (`parity-report.json`, schema_version 1)
- Top level: `environment` (python, free_threaded, gil_enabled, platform, in_docker), `manifest_sha256` (which pinned data was used) and `status`.
- `suites.<name>`:
  - `candidate`: module, dist, version, adapter, `is_reference`.
  - `reference`: oracle name, version and tolerance.
  - `inputs`: pinned file → sha256.
  - `checks.<check>`: `status` pass/fail/skip, `same`, `diff`, up to 5 `examples` of diffs, `skip_reason`.
- Status rules:
  - A check passes with ≥1 comparison and 0 diffs.
  - A suite fails if any check fails.
  - Skips are explicit, never silent.

## Pinned data
All inputs and references live under `$PYROID_DATA` (default `/data`) and are pinned by sha256 in
[`../bench/manifest.json`](../bench/manifest.json). Loaders refuse mismatched bytes.

| file | what |
|---|---|
| `fineweb/fineweb-dev-10k.parquet` | first 10k rows of FineWeb `sample/10BT/000_00000.parquet` @ `9bb295dd` (ODC-By) |
| `parity/adversarial.parquet` | 3,232 docs: 2,000 non-English (FineWeb-2 test splits, 8 languages), 600 mojibake/HTML/control variants, 600 near-duplicates of dev docs (edit rates 0–40 %, truncation, append), 32 edge cases (empty, whitespace-only, 5k-char word, …) |
| `metrics/cnndm-3.0.0-test.parquet`, `metrics/summeval.parquet`, `metrics/wmt21.parquet` | eval data for the `metrics` suite: CNN/DailyMail 3.0.0 test (Apache-2.0), SummEval (MIT), WMT21 newstest2021 system outputs + references for 8 language pairs (research release, not redistributed) |
| `parity/{ftfy,fasttext,filters,minhash,metrics}-ref.parquet` | reference outputs; the manifest records the oracle versions |
| `trafilatura/CC-MAIN-{2026-39,2023-50,2019-35}.parquet` | Common Crawl pages (id, url, html) as datatrove 0.10.1 hands them to Trafilatura: WARC byte ranges through `WarcReader` + `URLFilter` (Common Crawl terms of use; see [`data/trafilatura-2.3.1`](data/trafilatura-2.3.1/README.md)) |
| `trafilatura/upstream-tests-{2.3.1,1.11.0}.parquet` | Trafilatura's `tests/` at v2.3.1 and v1.11.0, from GitHub |
| `parity/trafilatura-ref.parquet`, `parity/trafilatura-upstream{,-blobs}.parquet` | 2.3.1's outcome digests per page and option set; its tests' calls and golden outcomes |
| `parity/trafilatura-1.11.0-{ref,upstream,upstream-blobs}.parquet` | the same for 1.11.0 (`refs trafilatura111 trafilatura111_upstream`, in the `trafilatura111` group) |
| `warc/cc/*.gz` | 34 Common Crawl byte ranges (86.9 MB): WARC, WET, WAT and ARC files of 17 crawls, on gzip member boundaries or cut on purpose (Common Crawl terms of use; see [`data/warcio-1.8.2`](data/warcio-1.8.2/README.md)) |
| `warc/warcio-1.8.2/*` | warcio's `test/data` at v1.8.2 (16 files, Apache-2.0) |

- `../bench/fetch.py` fetches the inputs. It verifies the inputs it downloads, and `--verify-all` checks every pinned file.
- In the oracle env, on Linux, `python -m pyroid_parity.refs all` rebuilds the references and checks each one
  against the manifest, exiting 1 if any differ. Add `--pin` to record new hashes instead (adversarial generation is
  seeded and deterministic).

### Per-platform references
The fastText and metrics references are pinned once per platform (`<sys.platform>-<arch>`,
`pyroid_parity.data.platform_key`). Such manifest entries carry `"platform"`, and `data.load_manifest` only sees
the current platform's, so the same goes for every loader, the suites, `bench/fetch.py --verify-all` and the bench
gates.

| reference | linux-aarch64 | linux-x86_64 | chosen by |
|---|---|---|---|
| fastText | `parity/fasttext-ref.parquet` | `parity/fasttext-ref.linux-x86_64.parquet` | `specs.lid_ref` |
| metrics | `parity/metrics-ref.parquet` | `parity/metrics-ref.linux-x86_64.parquet` | `metrics_cases.metrics_ref` |

The x86_64 pins now come from real x86 hardware. All references were regenerated on 2026-11-06 on an AWS EC2
c7i.4xlarge (Intel Xeon Platinum 8488C, AVX2 + FMA). The fastText file came out byte-identical to the earlier pin,
which was made under Rosetta 2 emulation (FMA but no AVX2); the metrics file was re-pinned.

- fastText: fasttext-predict 0.9.2.4's own floats differ by CPU (measured in Docker, 2026-11-06). Its GCC `-O3`
  build contracts the quantized-matrix loops (`addcode`, `mulcode`, `Vector::norm`) into fused multiply-adds on
  aarch64, where FMA is baseline, and can't on baseline x86-64. On the 26,464 pinned predictions, x86_64 against
  aarch64: the labels are identical, the `lid.176.bin` probabilities bit-identical, and on `lid.176.ftz` 10,837 of
  13,232 docs differ by up to 5.96e-7. pyroid follows the same per-architecture rule (`fasttext::matrix::fused`),
  and is bit-identical to the oracle of its own platform on all 26,464.
- metrics: sacrebleu's BLEU goes through `math.exp`/`math.log`, so it depends on glibc's libm. Under Rosetta, libm
  took its SSE2 paths, and 165 of the 694,757 results (all BLEU scores, 1–2 ulps) differed from linux-aarch64.
  ROUGE, chrF and every count were identical. On the c7i, glibc uses its FMA variants of `exp`/`log`, and the x86
  metrics reference now equals the aarch64 one: only Rosetta 2's emulation differed. With those variants masked
  (`GLIBC_TUNABLES`), the same machine reproduces the old Rosetta pin exactly. The x86_64 file stays a separate
  entry.
- Every other reference (adversarial set, ftfy, filters, MinHash) is identical on both platforms. On an AWS
  Graviton4 (c8g.4xlarge), all 8 references reproduce the linux-aarch64 pins, made on Apple Silicon, byte for byte.

pyroid and the oracles call the same libm, so they agree with each other on any CPU, but a file regenerated on
another CPU can still differ from its pin. If the parity-data job reports a mismatch for one of these files only,
re-pin it there (`python -m pyroid_parity.refs fasttext --pin`, or `metrics --pin`).

## The trafilatura suite
`tests/test_trafilatura.py` checks one trafilatura release at a time, the mode pyroid.trafilatura reproduces with
`compat=`. Each mode has its own dependency group, and the groups conflict (`[tool.uv] conflicts`):

| mode | group | oracle | run with |
|---|---|---|---|
| 2.3.1 (default) | `trafilatura` (default group) | trafilatura 2.3.1, lxml 6.1.3 (libxml2 2.14.6), justext 3.0.2 | `pytest -m trafilatura --impl pyroid.trafilatura` |
| 1.11.0 | `trafilatura111` | trafilatura 1.11.0, lxml 6.1.3, justext 3.0.2, lxml-html-clean 0.4.6, charset-normalizer 3.5.2, courlan 1.4.0, htmldate 1.11.0: what a fresh `datatrove[processing]==0.10.1` resolves | `uv run --no-group trafilatura --group trafilatura111 pytest -m trafilatura --impl pyroid.trafilatura --trafilatura-compat 1.11.0` |

`--trafilatura-compat` (or `$PYROID_TRAFILATURA_COMPAT`) picks that release's references, option matrix and
recorded calls, requires its exact pins in the live oracle, and passes `compat=` to a candidate that takes it.
Both modes share the pages, Common Crawl HTML as datatrove 0.10.1 hands it to Trafilatura
(`bench/fetch.py --only trafilatura`, `pyroid_parity.trafilatura_cc`):

| file | crawl | pages | HTML (chars) | on disk |
|---|---|---:|---:|---:|
| `trafilatura/CC-MAIN-2026-39.parquet` | CC-MAIN-2026-39, WARC 00100, bytes 0-196,203,906 (the P10 sample, byte for byte) | 4,677 | 618.6 M | 112.9 MB |
| `trafilatura/CC-MAIN-2023-50.parquet` | CC-MAIN-2023-50, WARC 00100, three 10 MB slices | 926 | 134.2 M | 16.6 MB |
| `trafilatura/CC-MAIN-2019-35.parquet` | CC-MAIN-2019-35, WARC 00100, three 10 MB slices | 1,589 | 140.0 M | 17.8 MB |

Option sets (`specs.TRAFILATURA_OPTION_SETS_BY_MODE`, all with `output_format="txt"`): `datatrove`
(`favor_precision=True, include_comments=False, deduplicate=True`, datatrove's call), `default`, `fast` (1.11.0 has
no `fast`: `no_fallback` there), `favor_recall`, `precision_comments`, `no_tables` and `dedup` (`deduplicate=True`).
The two dedup sets are stateful: each file is one datatrove task, run in page order after `reset_caches()`. Pinned references hold a digest per page and set, not
the text. Options pyroid does not port (other output formats, formatting, links, images, metadata) are checked
in fresh processes: trafilatura's result with exactly one `pyroid.FallbackWarning`; the ported sets must not warn.

- **CI:** the parity matrix has `trafilatura` and `trafilatura@1.11.0` (a suite in another mode); the GIL legs
  install that mode's group, and parity-data builds both modes' references.
- **Free-threaded legs:** the groups install on 3.14t and 3.15t from wheels (measured on linux-aarch64, 2.3.1), but
  importing lxml re-enables the GIL (htmldate, courlan and justext import it too). CI leaves them out there: the
  pinned checks run with the GIL off, and the live ones (Hypothesis, delegation) skip.
- **One pin for all platforms:** the references were made on linux-aarch64. On the first 100 pages of each file
  under all 9 option sets, linux-x86_64 (Rosetta 2 emulation, 2026-11-09) gave the same 2,700 digests; a mismatch
  on real x86 hardware would show in the parity-data job.
- **Mutant:** `python -m pyroid_parity.trafilatura_mutant` runs the suite against a deliberately broken wrapper
  of the installed trafilatura and exits 1 unless every check fails; `tests/test_harness_trafilatura.py` runs it
  in the parity-data job, once per mode.
- **At scale (not CI):** `python -m pyroid_parity.trafilatura_scale OUT_DIR [--impl pyroid.trafilatura]
  [--compat 1.11.0] --pages N` streams slices of other WARC files of several crawls, compares, and keeps only
  JSON counts (with the mode's group and `--group cc`).

## The warc suite
`tests/test_warc.py` checks a module with warcio 1.8.2's reader API against warcio itself, and the datatrove level
against datatrove 0.10.1's `WarcReader`:

```bash
cd parity
uv run --frozen --with ../dist/pyroid-*.whl pytest -m warc --impl pyroid.warc
uv run --frozen --with ../dist/pyroid-*.whl pytest -m warc --impl pyroid.warc --parity-limit 0   # every record
```

warcio and brotlicffi are core dependencies (pure Python and a cp314t wheel), so the free-threaded legs compare
against warcio live; datatrove, faust-cchardet and python-magic are the `warc` group (a default group). libmagic
(Debian/Ubuntu `libmagic1`) is needed for records without `WARC-Identified-Payload-Type`, which datatrove hands to
`magic.from_buffer`; without it a stub raises at those records on both sides, and the pinned datatrove checks skip.

A run drives each input under a set of `warc_compare.Config`s: iterator options (`no_record_parse`, `verify_http`,
`arc2warc`, `ensure_http_headers`, `check_digests` False / True / `log` / `raise`, `block_size` 1 to 65536), read
plans (iterate, `content_stream().read()`, 16 KiB blocks, `raw_stream`, offsets, seeded mixes of `read(n)` /
`readline(n)` on both streams, early stop) and openers (file, unbuffered file, BytesIO, no `tell`, short reads, a
logging wrapper, `gzip.GzipFile`, datatrove's `DataFolder.open`). The trace of a run lists every observable thing
in order; the candidate's must equal warcio's event for event. Common Crawl slices run 12 configs, the other inputs
29 (31,291 traces per implementation in the default run); the default reads the first 100 records of each slice.
In Docker on Apple Silicon (indicative), the default run took 113 s as a self-check and 202 s against
`warc_mutant` without defects, which exercises every live comparison with warcio as the candidate; with
`--parity-limit 0`, 168 s and 488 s.

- **Pins:** `data/warcio-1.8.2/ref-digests.json` holds warcio's and datatrove's trace digests
  (`python -m pyroid_parity.warc_pins write`; `check` recomputes them). They were made on CPython 3.13 with zlib
  1.3.2, brotlicffi 1.2.1.0 and libmagic 5.44 (Debian 12). Checks whose inputs depend on a version that differs
  here (the synthetic gzip bytes on zlib, libmagic's answers) skip and say why.
- **Without `--impl`:** the live checks skip (warcio against itself), the pinned ones run, and `mutant` proves that
  three broken wrappers of warcio are caught.
- **Proving the harness:** `pyroid_parity.warc_mutant` is a stand-in candidate with warcio's API (`--impl
  warc=pyroid_parity.warc_mutant`). `python -m pyroid_parity.warc_mutant faithful` runs the whole suite live
  against it without defects (no check may fail); `prove` turns on one defect per kind of check and exits 1
  unless each of those checks fails.
- **Documented difference:** pyroid.warc verifies md5, sha1, sha224, sha256, sha384 and sha512 digests. Other
  algorithms that hashlib (OpenSSL) knows, such as sha3_256 or blake2b, count as an unknown algorithm in pyroid.
  `digest-other-algorithms` accepts either outcome and nothing else.
- **At scale (not CI):** `python -m pyroid_parity.warc_scale OUT.json FILE... --impl pyroid.warc --workers N`
  compares every record of whole files (both read paths and the datatrove level) as per-record digests, in parallel,
  and writes counts and the first diffs. On the 100 MB CC-MAIN-2026-39 prefix (8,631 records), warcio against
  itself took 2.5 s per side for `content_stream()` reads over the raw file (Docker VM on Apple Silicon,
  indicative).

## Oracle choices
- fastText: fasttext-predict 0.9.2.4.
  - It is the package datatrove's LID loads (`fasttext.FastText._FastText`), built from the unmodified fastText C++ predict path.
  - It ships manylinux/musllinux wheels for aarch64 and x86_64, cp39–cp313 plus cp313t, so it installs with no compiler.
  - `fasttext==0.9.3` is sdist-only and needs g++, and its `predict` breaks on numpy 2.
  - Cross-check, measured in Docker: fasttext-predict-v2 1.1.0 on CPython 3.14 reproduces all 26,464 pinned predictions with max |Δp| = 0.
  - An independent build (`fasttext` 0.9.3 from source) was not run.
- datatrove: 0.10.1, without the `processing` extra. That extra pulls `fasttext-numpy2-wheel`, which has no aarch64 wheels. We add spaCy and tokenizers instead.
  - datatrove 0.10.1 splits English words with spaCy `blank("en")`, not NLTK. A Rust port must reproduce spaCy's tokenizer exceptions/prefix/suffix/infix rules to get identical Gopher/FineWeb decisions.
