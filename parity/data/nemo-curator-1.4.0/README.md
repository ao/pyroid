NeMo Curator 1.4.0's outputs for the `nemo` suite. Nothing from NeMo is committed here: the references are
generated files under `$PYROID_DATA`, pinned by sha256 in [`bench/manifest.json`](../../../bench/manifest.json).
NeMo Curator is Apache-2.0, © NVIDIA CORPORATION.

| file | what |
|---|---|
| `parity/nemo-ref.parquet` | 888,518 rows, 23.8 MB: one per case and document. Columns `case, id, outcome, type, score, keep, error, message, digest, length, changed` (`pyroid_parity.nemo_cases.REF_COLUMNS`). linux-aarch64; another platform pins `parity/nemo-ref.<platform>.parquet` (`nemo_cases.nemo_ref`), because it holds fastText floats |
| `models/nemo-quality-dev10k.ftz` | 1.7 MB fastText quality model for FastTextQualityFilter, trained here (below) |

## Cases

`pyroid_parity/nemo_cases.py` lists 74, each with the `pyroid.nemo` function that reproduces it:

- 28 filter cases: the ScoreFilter stages of NeMo's `config/text/heuristic_filter_english_pipeline.yaml`, in order
  and with its arguments. The recorder checks the list against the YAML installed with NeMo.
- 22 other filter configs: `WordCountFilter(min_words=80)`, `CommonEnglishWordsFilter(stop_at_false=False)` and
  `(min_num_common_words=5)`, `RepeatingTopNGramsFilter(n=1)`, `RepeatingDuplicateNGramsFilter(n=1..4)`,
  `PornographicUrlsFilter()`, six `SubstringFilter` cases, and seven `lang="zh"` cases (NeMo splits words with
  `jieba.cut`), which run on the adversarial set's 250 Mandarin documents and 32 edge cases only.
- 20 modifier configs: UnicodeReformatter with NeMo's defaults, with ftfy's (`config=TextFixerConfig()`) and with
  NFKC + quote uncurling; BoilerPlateStringModifier in both modes; MarkdownRemover, NewlineNormalizer,
  QuotationRemover, UrlRemover; LineRemover (frequent dev-slice lines and the empty line); nine Slicer cases;
  FastTextLabelModifier.
- 4 fastText cases: FastTextLangId on `lid.176.bin` and `lid.176.ftz` (`min_langid_score=0.3`), the same with
  `lang="en"` (only `keep` differs), and FastTextQualityFilter (`alpha=3, seed=42`).

Every case runs on dev 10k + adversarial 3,232 in that order, except the `lang="zh"` ones.

## Rows

- Filters: `type` and `score` are the Python type and exact text of `score_document`'s value (`repr` for
  numbers, the string itself for FastTextLangId's `str([score, lang])`); `keep` is `keep_document` of it. When
  NeMo raises, `outcome` is `raise` with the exception's type and message.
- Modifiers: `digest` (the first 32 hex digits of the output's sha256), `length` (code points), and `changed`.
- FastTextQualityFilter's `keep`: one filter built with `seed=42` (which seeds numpy's global RNG), then
  `keep_document` once per document, dev then adversarial. Batched draws give the same values.

## How the files are made

In the NeMo env (parity/README.md, "The nemo suite"), in Docker:

```bash
cd parity
PYROID_DATA=/target/parity/data python -m pyroid_parity.refs nemo --pin --workers 4   # about 50 s
PYROID_DATA=/target/parity/data python -m pyroid_parity.refs nemo --workers 4         # regenerate and verify
```

The step re-runs itself with `PYTHONHASHSEED=0`: RepeatingTopNGramsFilter keeps the first of the most frequent
n-grams in `set` iteration order, which follows the string hash seed, and 45 % / 66 % / 78 % of the dev documents
tie for n = 2 / 3 / 4. With the seed fixed, the tie order is the same on CPython 3.12, 3.13, 3.14, 3.14t, 3.15 and
3.15t (checked over the corpus for n = 2, 3, 4), so the references hold on every interpreter of the matrix.

The quality model is trained by the same step with fasttext 0.9.3: one line per dev document, labelled
`__label__hq` if the document passes all 28 heuristic filters (from the rows just recorded; 9,545 of 10,000) and
`__label__lq` otherwise, prepared with `FastTextLabelModifier(label).modify_document(text)`. Recipe:
`train_supervised(lr=0.1, epoch=5, dim=32, loss="softmax", thread=1, seed=0)`, then
`quantize(cutoff=50000, qnorm=True, retrain=False)`. With one thread the model is the same file on every run. The
fasttext 0.9.3 build in the NeMo env (g++ 12, linux-aarch64) stops with "Encountered NaN" for `dim` below 32 on
this data, and for `dim=32` with `minCount > 1` or word bigrams; the unquantized `dim=32` model is 52 MB.

Both files reproduced byte for byte on a second run (2026-11-10, linux-aarch64, Docker on Apple Silicon).

## NeMo's pipelines, stock against stock

`python -m pyroid_parity.nemo_pipeline all` (XennaExecutor in batch mode, `RayClient(num_cpus=4)`,
`PYTHONHASHSEED=0`, 13,230 input documents in 11 files; 2026-11-10, linux-aarch64, Docker on Apple Silicon):

| pipeline | runs | kept | wall time | output digest |
|---|---|---:|---|---|
| heuristic (`heuristic_filter_english_pipeline.yaml`, scores kept) | 3 | 10,405 | 79 s, 77 s, 63 s | `ac8a07dc0a32e33b`, the same each run |
| fastText + modifiers | 3 | 13,123 | 33 s, 33 s, 30 s | `bc6138283d733a63`, the same each run |

The digest covers kept ids, every column with its dtype (`id`, `text` and `language` `string[pyarrow]`, scores
`double[pyarrow]` / `int64[pyarrow]`) and every cell. In the heuristic runs NeMo's `_stage_perf` adds up to
about 60 s of filter time, spread over 4 workers; most of the rest of the wall time goes to starting and stopping
each of the 30 stages' workers in turn (batch mode), about 2 s per stage in the log. Without a fixed hash seed,
two stock runs gave different RepeatingTopNGramsFilter scores (an earlier probe on 2,000 dev documents).

## What NeMo does on the adversarial documents

- 7 whitespace-only edge cases (`""`, `" "`, `"\n"`, `"\n\n\n"`, `"\r\n\r\n"`, `"\t\t\t"`, two spaces and an
  ideographic space) make SymbolsToWordsFilter, RepeatedLinesFilter, RepeatedLinesByCharFilter, PunctuationFilter,
  WordsWithoutAlphabetsFilter, MeanWordLengthFilter, EllipsisFilter and BulletsFilter raise ZeroDivisionError, and
  LongWordFilter ValueError (`max()` of nothing); RepeatedParagraphsByCharFilter raises on `""` only.
- In the heuristic pipeline two of them, `"\n"` and `"\n\n\n"`, get past the first stage: NonAlphaNumericFilter's
  alphanumeric class includes `"\n"`, so their non-alphanumeric ratio is 0. SymbolsToWordsFilter then raises,
  which fails the whole pipeline. `pyroid_parity.nemo_pipeline` leaves those two (`adv-edge-002`,
  `adv-edge-003`) out of its inputs; on the other 13,230 documents the stock pipeline keeps exactly the 10,405
  that pass all 28 filters in these references.
- With `lang="zh"`, SymbolsToWordsFilter raises ZeroDivisionError on 273 of the 282 documents: jieba returns
  whitespace runs as tokens, and the filter divides by `len(word.strip())`.
- `ScoreFilter` on a `string[pyarrow]` text column (what ParquetReader gives) stores int scores as `int64[pyarrow]`
  and float scores as `double[pyarrow]`: pandas 3.1.1's `Series.apply` keeps the Arrow backend.
