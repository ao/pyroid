# jmespath 1.1.0 test data

Copied unchanged from the jmespath.py repository at tag `1.1.0`
(https://github.com/jmespath/jmespath.py, MIT, see `LICENSE`):

- `compliance/*.json`: the JMESPath compliance suite as jmespath.py ships it (the same files as
  https://github.com/jmespath/jmespath.test at 53abcc37, except `benchmarks.json`).
- `legacy/literal.json`: jmespath.py's legacy literal cases.
- `tests/*.py`: jmespath.py's own unit tests (`test_lexer`, `test_parser`, `test_functions`,
  `test_search`, `test_custom_functions`, `test_compliance`). `python/tests/test_jmespath_parity.py`
  runs them against `pyroid.jmespath`.

`real-queries.json` is a seeded sample made by `bench/jmespath/real_queries.py`: 358 JMESPath
expressions from botocore 1.44.3's paginator (`result_key`, `output_token`, `more_results`,
`non_aggregate_keys`) and waiter (`argument`) definitions and from AWS CLI 1.46.3's examples
(`--query`), both Apache-2.0, each with a response generated from the operation's output shape
(timestamps as `{"$datetime": ...}`, blobs as `{"$bytes": ...}`).

The GitHub tag's `jmespath/` sources are byte-identical to the PyPI sdist
(`jmespath-1.1.0.tar.gz`, sha256 472c87d80f36026ae83c6ddd0f1d05d4e510134ed462851fd5f754c8c3cbb88d).
