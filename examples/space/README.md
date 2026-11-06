---
title: pyroid FineWeb cleaner
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: "3.13"
app_file: app.py
license: mit
short_description: FineWeb's text-cleaning recipe, in Rust, with pyroid
---

# pyroid FineWeb cleaner

A minimal [Gradio](https://www.gradio.app) demo of [pyroid](https://github.com/ao/pyroid)'s
FineWeb recipe: the stages, settings and order of datatrove's FineWeb pipeline (ftfy,
fastText language ID, the Gopher repetition, Gopher quality, C4 and FineWeb quality
filters, MinHash dedup), with the same results.

- **Paste text**: one document through the chain, with each stage's verdict (did ftfy
  change it, the language and its probability, which filter dropped it and why) and the
  text as it leaves the chain.
- **Upload a file**: a parquet or JSONL file with a `text` column through the whole recipe,
  MinHash included (`pyroid.integrations.hf.fineweb`), and the cleaned file back, in the
  same format, with the counts per stage and per reason.

## Limits

This is a public, anonymous demo, so every input is capped (constants at the top of
`app.py`):

| Cap | Value |
|---|---|
| Pasted text | 100,000 characters |
| Uploaded file | 20 MiB (Gradio rejects bigger uploads; the handler checks again) |
| Rows cleaned per upload | the first 5,000 |
| Decoded data of those rows (parquet compresses) | 128 MiB |
| Requests running at once | 1, with at most 16 waiting |
| Cleaned files and uploads kept | deleted once over an hour old (checked hourly, and on each upload) |

## Run it locally

```bash
pip install -r requirements.txt
python app.py
```

Gradio serves it on <http://127.0.0.1:7860>, on this machine only; `app.py` never sets a
server address or a share link (Hugging Face Spaces sets its own address).

On first use the app downloads fastText's language ID model `lid.176.bin` (131 MB,
CC BY-SA 3.0, by Facebook AI Research) into `~/.cache/pyroid` and checks its sha256. To
use a copy you already have, set `PYROID_LID_MODEL=/path/to/lid.176.bin`. Set
`PYROID_SPACE_THREADS` to cap pyroid's threads (default: every CPU).

pyroid 1.0 is not on PyPI yet: until it is, replace the `pyroid` line of
`requirements.txt` with the URL of a pyroid 1.0 wheel.
