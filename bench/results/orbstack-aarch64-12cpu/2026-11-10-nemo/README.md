# NeMo Curator 1.4.0 stages vs pyroid.integrations.nemo_curator (2026-11-10)

Docker (OrbStack VM on a Mac, aarch64, 12 vCPUs), CPython 3.12, NeMo Curator 1.4.0 (pandas 3.1.1,
Ray 2.60, cosmos-xenna 0.2.0, fasttext 0.9.3 built with g++ 12), pyroid 1.0.0rc1 release build of
`feat/nemo`. Indicative: a shared VM, not a dedicated machine. `PYTHONHASHSEED=0` for every run.

- `stages-1thread.json`: `bench/nemo_bench.py --docs 10000 --batch 1000`, pinned to one CPU
  (`taskset -c 0`). NeMo's `ScoreFilter` / `Modify` with NeMo's object against pyroid's stage with the
  same object (`threads=1`); whole DataFrames compared equal before timing; minimum of 2 (NeMo) and
  5 (pyroid) passes over the 10 batches.
- `pipeline-100k.json`: `bench/nemo_pipeline_bench.py`, NeMo's
  `heuristic_filter_english_pipeline.yaml` (ParquetReader, no writer) on the first 100,000 documents
  of `fineweb/fineweb-100k.parquet` in 64 files, `XennaExecutor` in batch mode with a Ray cluster
  of 1 or 8 CPUs, one fresh container per run, each started once the 1-minute load average was
  below 3. Both variants kept the same 95,774 documents (same ids and text).
