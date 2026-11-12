# WARC test data

- `*.warc`, `*.warc.gz`, `*.arc`, `*.arc.gz`, `example-bad.warc.gz.bad`: warcio's own test
  files, from https://github.com/webrecorder/warcio/tree/v1.8.2/test/data (Apache-2.0,
  Webrecorder, Ilya Kreymer et al.), unchanged.
- `synth/`: small synthetic files (encodings, chunking, odd lengths and headers, digests,
  truncated, corrupt and multi-record gzip members, ARC), written by `gen.py`.
- `expected.json.gz`: what warcio 1.8.2 gives on CPython 3.13.18 (zlib 1.3.2) for every file
  above with several `ArchiveIterator` options and read plans, written by `gen.py`; checked
  by `tests/warc_oracle.rs`.

Regenerate in the dev container (warcio 1.8.2 installed):

    /target/venv313/bin/python -I crates/pyroid-core/tests/data/warc/gen.py
