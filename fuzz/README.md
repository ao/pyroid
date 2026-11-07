# Fuzzing pyroid-core

[cargo-fuzz] (libFuzzer) targets for the text kernels in `crates/pyroid-core`.
This directory is its own Cargo workspace, so the root workspace, its lints
and its CI jobs never see it. The Arrow import fuzzing in `arrow/` is
documented in [`arrow/README.md`](arrow/README.md).

The targets need a nightly Rust toolchain, a C++ compiler (for libFuzzer) and `cargo-fuzz`
(`rustup toolchain install nightly`, `cargo install cargo-fuzz`). They have been run on Linux.

## Targets

Every target turns a few leading bytes into a configuration (all-zero bytes
give the reference library's defaults, and only values the Python API
accepts are generated) and uses the rest as the text. Besides "no panic",
each one checks properties that the reference library has too, so a failure
is a pyroid bug, not a quirk of the reference. The doc comment at the top of
each file is the full list.

| Target | Input layout | What it checks |
|---|---|---|
| `ftfy` | 4 config bytes (bits flip `TextFixerConfig` defaults, `explain=False` when zero), 1 `max_decode_length` byte, text | every str and `*_wtf8` entry point; WTF-8 validity of outputs and the `InvalidWtf8` offset; `*_and_explain` text equals the plain call; `Cow::Borrowed` exactly when unchanged; `fix_text` equals ftfy's per-line loop over `fix_text_segment`; `fix_text_segment` and `fix_encoding` are idempotent; `is_bad == (badness > 0)` |
| `filters` | version byte, step count, step kinds, 4 x 24-byte step configs, text | GopherRepetition, GopherQuality, C4 (default and FineWeb's `filter_no_terminal_punct=False`), FineWeb and a fuzzed `Pipeline`: deterministic; which filters can fail and when; drop reasons belong to a check that is on; C4's kept text is stripped and free of what the config removes; `Pipeline::run` equals running its steps one by one |
| `tokenizer` | version byte, text | spaCy-English tokens: non-empty, ordered, on char boundaries, covering the text with only single-space gaps; whitespace tokens are whitespace, others have no outer whitespace; same tokens twice (span cache); every `words` entry point; the sentence functions agree; `WhitespaceTokenizer` is `str.split()` |
| `translit` | replacement length byte, replacement, text | `anyascii`, `unidecode` (drop, replace, keep), `first_missing`, `ascii_prefix_len` and the `*_wtf8` variants against a per-character model; ASCII output where promised; `Borrowed` exactly for ASCII input |
| `dedup_simplify` | norm-config bits, version, n-grams, buckets, hashes, 4-byte seed, hash bits, text | `simplify_text` vs `simplify_text_into`; `Borrowed` exactly when unchanged; stripped output; idempotence only for the configs where datatrove's `simplify_text` is idempotent; MinHash signature size, presence (iff at least `n_grams` words), determinism, and equality with the token/shingle entry points and a naive `u128` computation |

| `fasttext` | a `.bin`/`.ftz` model file; its last byte `n` selects the `n % 48` bytes before it as an extra query text (the loader ignores trailing bytes) | `Model::from_bytes`, then `predict`/`predict_into`, `sentence_vector`, `word_vector`, `subwords`, `line`, `line_ids`, `tokenize` and the id lookups on whatever loads: heap ≤ 1 MiB + 300 × file size to load (plus 8 KiB per query byte to infer); errors in fastText's order; ≤ k distinct in-range labels, best first, ≥ threshold, probabilities NaN or in [0, 1 + 2e-5] (hs: (1 + 1e-5)^depth); `predict_into` agrees bit for bit; vectors have `dim` floats. Seeds: tiny synthetic models from the `fasttext_load.rs` builders. Header comment of `fuzz_targets/fasttext.rs` for details |

The Arrow C Data importer target lives in its own cargo-fuzz project,
[`arrow/`](arrow/README.md), because it compiles pyroid-py's PyO3 sources and
links libpython.

### Idempotence of `ftfy.fix_text`

`fix_text_segment` and `fix_encoding` loop until nothing changes, so they are
idempotent in ftfy and the target asserts it. `fix_text` is not: it fixes
line by line, in chunks of at most `max_decode_length` code points, and turns
`unescape_html="auto"` off for good at the first line holding `<`. A first
pass can change where those boundaries fall (a chunk cut through mojibake
decodes differently once the text is shorter; `fix_line_breaks` turns `\r`
into new lines), so a second pass may change the text again. Of the 5,530
inputs in `corpus/ftfy`, 241 are not idempotent (239 through
`max_decode_length` chunks, 2 through new lines, none within one segment),
and ftfy 6.3.1 gives the same first and second pass on all 241 (before
`cmin`: 336 of 10,409, 334 through chunks and 2 through new lines). So the
target checks `fix_text` against its own segment loop instead. Set
`PYROID_FUZZ_REPORT=<file>` to log the inputs where
`fix_text(fix_text(x)) != fix_text(x)`.

### Differential runs against the references

The fuzzer cannot call Python, so the reference comparison runs over a
corpus afterwards. With `PYROID_FUZZ_DUMP=<file>` set, every target writes
one JSON object per input with its results (computed with Unicode 15.1, the
version of CPython 3.13; strings hex-encoded):

```sh
PYROID_FUZZ_DUMP=/target/ftfy-dump.jsonl cargo +nightly fuzz run -a -s none ftfy -- -runs=0
```

and a short script then recomputes them with the reference in a CPython
3.13 venv (`ftfy==6.3.1`, `datatrove==0.10.1` with `regex`, `xxhash`,
`numpy` and `tokenizers`, `spacy==3.8.18`, `anyascii==0.3.3`):
`ftfy.fix_text` / `fix_and_explain` / `fix_encoding_and_explain` /
`badness.badness`; the datatrove filters chained like `Pipeline`;
`split_into_words`, `span_tokenize` and the spaCy tokens; `simplify_text`,
`word_tokenize` and `MinhashDedupSignature.get_signature`; and
`anyascii.anyascii`. Texts containing `IS_ALPHA` are skipped for the
tokenizer (datatrove's documented history effect).

## Running

From the repository root:

```sh
cd fuzz
cargo +nightly fuzz build -a -s none ftfy
cargo +nightly fuzz run -a -s none ftfy -- -max_total_time=900 -rss_limit_mb=1536 \
  -max_len=8192 -timeout=10 -print_final_stats=1
```

To run them in a Linux container instead, mount the checkout, for example with
`docker run --rm -it -v "$PWD":/src -w /src rust:latest bash`, and install the nightly toolchain
and `cargo-fuzz` inside it.

Always name the target (`cargo fuzz build <target>`, `run <target>`). Keep
runs single-process (no `-jobs`/`-workers`) and the RSS limit at 1.5 GB or
less on a shared machine.

- `-a` turns on debug assertions and overflow checks (off in cargo-fuzz's
  default release build), which is what catches arithmetic bugs.
- `-s none` drops AddressSanitizer: `pyroid-core` is `#![forbid(unsafe_code)]`,
  so it buys little and costs about half the speed. Use the default
  sanitizer when fuzzing code with `unsafe` (the Arrow FFI in `arrow/`).
- libFuzzer grows inputs slowly; add `-len_control=0` for a run that tries
  long texts (many lines, `max_decode_length` chunks) from the start.
- Use the same `-a -s none` for `tmin` and `cmin`, or cargo-fuzz rebuilds.
- Reproduce a crash: `cargo +nightly fuzz run -a -s none <target> artifacts/<target>/crash-<sha>`
  (add `-- -runs=1`; the panic message names the broken property).
- Minimise it: `cargo +nightly fuzz tmin -a -s none <target> artifacts/<target>/crash-<sha>`.
  Turn the minimised input into a regression test in `crates/pyroid-core`
  before fixing; artifacts are gitignored.
- Shrink a corpus after a long run: `cargo +nightly fuzz cmin -a -s none <target>`.
  Keep each committed corpus under about 1 MB: add `-- -max_len=2048` to
  truncate the few long inputs (the ftfy and filters corpora were cut that
  way, losing 1.5% and 3% of their features).
- Coverage: `cargo +nightly fuzz coverage <target>` (writes `coverage/`,
  gitignored).

## Corpora

`corpus/<target>/` holds the minimised corpora (`cargo fuzz cmin`), started
from small real inputs: ftfy's own test cases, the tokenizer and
`simplify_text` golden files under `crates/pyroid-core/tests/`, and a few
synthetic documents and multilingual strings. File names are content hashes;
the files are raw fuzz inputs (header bytes, then text), so they are binary
and must be left alone by whitespace and spelling hooks.

[cargo-fuzz]: https://github.com/rust-fuzz/cargo-fuzz
