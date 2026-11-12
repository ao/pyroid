# pyroid examples

Ways to plug pyroid into what you already run. Each one gives the same output as the tool it
replaces; the checks below are how that is verified.

| Path | What |
|---|---|
| `datatrove/fineweb.py` | datatrove's FineWeb pipeline on `pyroid.integrations.datatrove`'s drop-in steps (`--stock` runs datatrove's own, for comparison). |
| `datatrove/verify.py` | The benchmark's datatrove chain (`bench/pipeline.py`) with pyroid's steps swapped in; prints the per-doc digest, which must equal the stock chain's. Needs a pyroid checkout and the pinned `/data`. |
| `cli/digest.py` | The same digest, computed from `pyroid clean ... --excluded ...` outputs. |
| `notebooks/` | "Clean FineWeb in 60 seconds": Hugging Face `datasets` + `pyroid.integrations.hf`, Colab-ready. |
| `space/` | A minimal Gradio app (paste text or upload a file; see what each stage decides). Not deployed. |

## The command line

```bash
pip install 'pyroid[cli]'
pyroid clean fineweb.parquet clean.parquet --recipe fineweb --excluded dropped.parquet --stats
pyroid langid docs.jsonl.gz tagged.parquet --keep en,fr --min-score 0.65
pyroid dedup clean.parquet dedup.parquet --method minhash --excluded dups.parquet
pyroid translit names.parquet ascii.parquet --output-column name_ascii
```

## Existing code, unchanged

```bash
pyroid run -p ftfy,fasttext train.py --epochs 3   # `import ftfy` / `import fasttext` get pyroid's
```

or, inside Python, `import pyroid.integrations; pyroid.integrations.patch("ftfy")` before the
`import ftfy` you want redirected (`unpatch()` undoes it).
