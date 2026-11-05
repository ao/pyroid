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
   - `python -m pyroid_parity.report render parity-report.json` prints a Markdown table, which is also appended to `$GITHUB_STEP_SUMMARY`.
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
| `parity/{ftfy,fasttext,filters,minhash}-ref.parquet` | reference outputs; the manifest records the oracle versions |

- **Regenerate** (oracle env, Docker): `python -m pyroid_parity.refs all --pin` (adversarial generation is seeded and deterministic).
- **Fetch inputs:** `../bench/fetch.py`.

## Oracle choices
- **fastText → fasttext-predict 0.9.2.4.**
  - It is the package datatrove's LID loads (`fasttext.FastText._FastText`), built from the unmodified fastText C++ predict path.
  - It ships manylinux/musllinux wheels for aarch64 and x86_64, cp39–cp313 plus cp313t, so it installs in `pyroid-dev` with no compiler.
  - `fasttext==0.9.3` is sdist-only and needs g++, which isn't in the image, and its `predict` breaks on numpy 2.
  - Cross-check, measured in Docker: **fasttext-predict-v2 1.1.0 on CPython 3.14 reproduces all 26,464 pinned predictions with max |Δp| = 0.**
  - An independent build (`fasttext` 0.9.3 from source) was not run; see open issues in the P1 report.
- **datatrove → 0.10.1, without the `processing` extra.** That extra pulls `fasttext-numpy2-wheel`, which has no aarch64 wheels. We add spaCy and tokenizers instead.
  - **datatrove 0.10.1 splits English words with spaCy `blank("en")`, not NLTK.** A Rust port must reproduce spaCy's tokenizer exceptions/prefix/suffix/infix rules to get identical Gopher/FineWeb decisions.
