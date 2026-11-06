# Notebooks

## Clean FineWeb in 60 seconds

[`clean_fineweb_in_60_seconds.ipynb`](clean_fineweb_in_60_seconds.ipynb) streams a sample
of [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (`sample-10BT`) and
runs FineWeb's cleaning recipe on it with `pyroid.integrations.hf`: ftfy, fastText
language ID, the Gopher/C4/FineWeb quality filters and MinHash dedup, the same stages,
settings and order as datatrove's FineWeb pipeline, with the same output. It prints the
counts per stage and per reason, the time it took, a few dropped documents per stage, and
writes the kept documents to parquet.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ao/pyroid/blob/main/examples/notebooks/clean_fineweb_in_60_seconds.ipynb)

pyroid 1.0 is not on PyPI yet: until it is, edit the first code cell to install a pyroid
1.0 wheel instead of `pyroid[hf]`.

### Run it locally, offline

Environment variables point the notebook at local files (and skip the install cell):

| Variable | Meaning |
|---|---|
| `PYROID_NOTEBOOK_DATA` | a FineWeb parquet file (a `text` column) to read instead of streaming |
| `PYROID_LID_MODEL` | the path of fastText's `lid.176.bin` (else it is downloaded once) |
| `PYROID_NOTEBOOK_DOCS` | how many documents (default 10,000) |
| `PYROID_NOTEBOOK_OUT` | where to write the cleaned parquet file |

To execute it headless (`pip install nbclient ipykernel`):

```bash
PYROID_NOTEBOOK_DATA=/path/to/fineweb.parquet PYROID_LID_MODEL=/path/to/lid.176.bin \
PYROID_NOTEBOOK_OUT=/tmp/fineweb-clean.parquet HF_DATASETS_OFFLINE=1 \
    jupyter execute --output=/tmp/executed.ipynb clean_fineweb_in_60_seconds.ipynb
```

pyroid's test suite does the same (`python/tests/test_integrations_hf_notebook.py`) on its
pinned 10,000-document FineWeb slice and checks the counts against datatrove's pipeline.

Save the notebook without outputs (Edit > Clear All Outputs): the test checks that too.
