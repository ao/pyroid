# pyroid parity suites

Differential tests that compare **any implementation** against the incumbents pyroid replaces, using pinned inputs
and pinned reference outputs. You choose the implementation by module path (`--impl`), so product tracks plug in
without editing the suites. Every session writes `parity-report.json`, which feeds the docs Compatibility page.

| suite | reference (oracle) | checks | tolerance |
|---|---|---|---|
| `ftfy` | ftfy 6.3.1 (live + pinned outputs) | 161-case golden corpus (incl. ftfy's known failures), Hypothesis fuzzing with lone surrogates / surrogate pairs / mojibake / entities / controls, `fix_text_batch`, dev slice (10k) + adversarial set (3.2k) | byte-identical, same exception type |
| `fasttext` | fasttext-predict 0.9.2.4 on CPython 3.13 | top-5 labels + probs on lid.176.bin and lid.176.ftz for 13.2k docs, live Hypothesis (incl. surrogates, NUL, empty), newline → `ValueError` | labels identical, \|Δp\| ≤ 1e-6 |
| `filters` | datatrove 0.10.1 (+ spaCy 3.8.18 tokenizer) | Gopher repetition, Gopher quality, C4 (default and FineWeb config), FineWeb quality on 13.2k docs | same keep/drop, same reason string, same rewritten text (C4), same exception |
| `dedup` | datatrove 0.10.1 MinHash, FineWeb config (sha1/64, 14×8, 5-grams) | 112 signature values per doc; removed-doc set over dev + adversarial (445 removals, mostly injected near-dups) | identical |
| `metrics` | rouge-score 0.1.2 (+ NLTK 3.10.4 Porter), sacrebleu 2.6.0 | ROUGE-1/2/3/4/L/Lsum on CNN/DM test (11.5k lead-3 vs highlights, 2k whole articles) and SummEval (1.6k system summaries, 11 refs, `score_multi`), with and without stemming; BLEU (13a/intl/char/none/zh/trg_lang, smoothing, lowercase, effective order, max order, `None` reference holes) and chrF/chrF++ per WMT21 system (141 corpora) and per segment (163k); live Hypothesis vs both incumbents (sacrebleu only on GIL builds: it loads lxml, which re-enables the GIL) | identical `repr`/`str`/`format`/signature strings (exact floats) |

## Plugging in a product track

1. **Implement the suite contract** in your module. The contracts are in the docstring of
   [`pyroid_parity/impl.py`](pyroid_parity/impl.py):
   - `ftfy`: ftfy's own API.
   - `fasttext`: `load_model(path).predict(text, k, threshold) -> (labels, probs)`, like fastText.
   - `filters`: `gopher_quality(texts, **config) -> (keep, reasons[, texts_out])` and so on, mask-shaped.
   - `dedup`: `minhash_signatures(texts, **config)` and `minhash_dedup(texts, **config) -> removed mask`.

   Function names, configs and the MinHash parameters are in [`pyroid_parity/specs.py`](pyroid_parity/specs.py).
   If your API differs, ask the harness owner for an adapter in `pyroid_parity/adapters/<suite>_<module>.py`, which is how datatrove is wrapped.
2. **Build your wheel in Docker**, then run the suite against the installed wheel, using the parity env plus your wheel:

   ```bash
   DX_NAME=<track> /Users/devel/src/ao/pyroid/.orchestration/dx.sh \
     'cd parity && UV_PROJECT_ENVIRONMENT=/target/venv-parity uv run --frozen --with ../dist/pyroid-*.whl \
        pytest -m ftfy --impl pyroid.ftfy --parity-report /target/parity-ftfy.json'
   ```
   - `--impl` takes `pyroid.ftfy`, `pyroid.fasttext`, `pyroid.filters`, `pyroid.dedup` (the last dotted component names the suite), or the explicit form `SUITE=MODULE`.
   - Without `--impl`, a suite self-checks its reference against the pinned outputs.
   - Use `-m <suite>` or `tests/test_<suite>.py` to select suites.
3. **Free-threaded / 3.15:** sync a second env, e.g. `UV_PROJECT_ENVIRONMENT=/target/venv-parity-314t uv sync --frozen -p 3.14t --no-default-groups`.
   - The core deps (ftfy, pytest, hypothesis, pyarrow, numpy) install on 3.12–3.15t.
   - The pinned references let the fasttext/filters/dedup checks run where the oracles don't install. fasttext-predict stops at cp313/cp313t, and datatrove's extras aren't needed.
   - The live-oracle checks skip there and say so in the report.
4. **Hypothesis depth:** `HYPOTHESIS_PROFILE=ci` gives 2000 examples per property (default `dev`: 200).
5. **Read the report:**
   - `python -m pyroid_parity.report render parity-report.json` prints a Markdown table and, under GitHub Actions, appends it to `$GITHUB_STEP_SUMMARY`. It is the only writer of the summary: pytest runs never touch it.
   - `validate` checks it against [`schema/parity-report.schema.json`](schema/parity-report.schema.json).
   - `merge a.json b.json -o all.json` combines interpreters.

`--parity-limit N` restricts every corpus to its first N docs (0 = all). Reference self-checks of the slow suites
(filters, dedup) sample 1000 docs by default. The removed-set check is corpus-level, so it runs only with `--parity-limit 0`.

## Report schema (`parity-report.json`, schema_version 1)
- **Top level:** `environment` (python, free_threaded, gil_enabled, platform, in_docker), `manifest_sha256` (which pinned data was used) and `status`.
- **`suites.<name>`:**
  - `candidate`: module, dist, version, adapter, `is_reference`.
  - `reference`: oracle name, version and tolerance.
  - `inputs`: pinned file → sha256.
  - `checks.<check>`: `status` pass/fail/skip, `same`, `diff`, up to 5 `examples` of diffs, `skip_reason`.
- **Status rules:**
  - A check passes with ≥1 comparison and 0 diffs.
  - A suite fails if any check fails.
  - Skips are explicit, never silent.

## Pinned data
All inputs and references live in the shared Docker volume `/data` and are pinned by sha256 in
[`../bench/manifest.json`](../bench/manifest.json). Loaders refuse mismatched bytes.

| file | what |
|---|---|
| `fineweb/fineweb-dev-10k.parquet` | first 10k rows of FineWeb `sample/10BT/000_00000.parquet` @ `9bb295dd` (ODC-By) |
| `parity/adversarial.parquet` | 3,232 docs: 2,000 non-English (FineWeb-2 test splits, 8 languages), 600 mojibake/HTML/control variants, 600 near-duplicates of dev docs (edit rates 0–40 %, truncation, append), 32 edge cases (empty, whitespace-only, 5k-char word, …) |
| `metrics/cnndm-3.0.0-test.parquet`, `metrics/summeval.parquet`, `metrics/wmt21.parquet` | eval data for the `metrics` suite: CNN/DailyMail 3.0.0 test (Apache-2.0), SummEval (MIT), WMT21 newstest2021 system outputs + references for 8 language pairs (research release, not redistributed) |
| `parity/{ftfy,fasttext,filters,minhash,metrics}-ref.parquet` | reference outputs; the manifest records the oracle versions |

- **Fetch inputs:** `../bench/fetch.py` (it verifies the inputs it downloads; `--verify-all` checks every pinned file).
- **Regenerate** (oracle env, Docker): `python -m pyroid_parity.refs all` rebuilds the references and checks each one
  against the manifest, exiting 1 if any differ; add `--pin` to record new hashes instead (adversarial generation is
  seeded and deterministic).

### Per-platform references
The fastText reference is pinned once per platform (`<sys.platform>-<arch>`, `pyroid_parity.data.platform_key`):
`parity/fasttext-ref.parquet` for linux-aarch64, `parity/fasttext-ref.linux-x86_64.parquet` for linux-x86_64
(`specs.lid_ref`). Such manifest entries carry `"platform"`, and `data.load_manifest` (so every loader, the suites,
`bench/fetch.py --verify-all` and the bench gates) only sees the current platform's.
- **Why** (measured in Docker, 2026-11-06): fasttext-predict 0.9.2.4's own floats differ by CPU. Its GCC `-O3` build
  contracts the quantized-matrix loops (`addcode`, `mulcode`, `Vector::norm`) into fused multiply-adds on aarch64,
  where FMA is baseline, and can't on baseline x86-64. On the 26,464 pinned predictions, x86_64 vs aarch64: labels
  identical, `lid.176.bin` probabilities bit-identical, `lid.176.ftz` 10,837 of 13,232 docs differ by up to 5.96e-7.
  pyroid follows the same per-architecture rule (`fasttext::matrix::fused`), and is bit-identical to the oracle of
  its own platform on all 26,464.
- **Assumption:** glibc. The x86_64 pin was generated under Rosetta 2 emulation, which exposes FMA but not AVX2, so
  glibc's libm took its SSE2 paths; a CPU with AVX2 + FMA runs glibc's FMA variants of `exp`/`log`, which can round
  differently in the last bit. Both pyroid and fastText call the same libm, so they still agree with each other, but
  the CI runner's regenerated file could differ from this pin. If the parity-data job reports a mismatch for this
  file only, re-pin it there (`python -m pyroid_parity.refs fasttext --pin`).
- Every other reference (adversarial set, ftfy, filters, MinHash) is identical on both platforms.

## Oracle choices
- **fastText → fasttext-predict 0.9.2.4.**
  - It is the package datatrove's LID loads (`fasttext.FastText._FastText`), built from the unmodified fastText C++ predict path.
  - It ships manylinux/musllinux wheels for aarch64 and x86_64, cp39–cp313 plus cp313t, so it installs in `pyroid-dev` with no compiler.
  - `fasttext==0.9.3` is sdist-only and needs g++, which isn't in the image, and its `predict` breaks on numpy 2.
  - Cross-check, measured in Docker: **fasttext-predict-v2 1.1.0 on CPython 3.14 reproduces all 26,464 pinned predictions with max |Δp| = 0.**
  - An independent build (`fasttext` 0.9.3 from source) was not run; see open issues in the P1 report.
- **datatrove → 0.10.1, without the `processing` extra.** That extra pulls `fasttext-numpy2-wheel`, which has no aarch64 wheels. We add spaCy and tokenizers instead.
  - **datatrove 0.10.1 splits English words with spaCy `blank("en")`, not NLTK.** A Rust port must reproduce spaCy's tokenizer exceptions/prefix/suffix/infix rules to get identical Gopher/FineWeb decisions.
