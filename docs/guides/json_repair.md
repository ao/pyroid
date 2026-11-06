# `pyroid.json_repair`: drop-in json_repair

`pyroid.json_repair` is a Rust port of [json_repair](https://github.com/mangiucugna/json_repair)
0.64.5 by Stefano Baccianella (MIT), the library that turns broken JSON from LLMs (missing quotes,
single quotes, comments, trailing commas, truncation, markdown fences, prose around it, Python
literals) into valid JSON. Its parser is ported line by line, so it returns **the same strings,
objects and repair logs as json_repair 0.64.5** ([parity](../compatibility.md#json_repair)). It
also runs in linear time on the inputs where json_repair's lookaheads are quadratic, and deep
nesting never overflows the stack.

## Change one import

```diff
-import json_repair
+import pyroid.json_repair as json_repair

 json_repair.repair_json("{'a': 1, b: [1, 2,}")    # '{"a": 1, "b": [1, 2]}'
 json_repair.loads('{"title": "Notes on the Analytical Eng')
 # {'title': 'Notes on the Analytical Eng'}
```

## Repairing and decoding

`repair_json`, `loads`, `load` and `from_file` take json_repair's arguments:

| Function | Returns |
|---|---|
| `repair_json(json_str, ...)` | the repaired JSON as a string (`""` when no JSON was found) |
| `loads(json_str, ...)` | the decoded value, like `json.loads` after a repair |
| `load(fd, ...)` | the same, reading an open text file |
| `from_file(filename, ...)` | the same, opening the file by name |

Valid JSON comes back as `json.dumps(json.loads(text))`; anything else goes through json_repair's
heuristics.

```python
import pyroid.json_repair as json_repair

reply = 'Sure! Here is the JSON:\n```json\n{"name": "Ada", "langs": ["en", "fr"\n```'
json_repair.repair_json(reply)  # '{"name": "Ada", "langs": ["en", "fr"]}'
json_repair.loads("{'ok': True, 'value': None, // comment\n 'n': 1,}")
# {'ok': True, 'value': None, 'n': 1}

json_repair.repair_json("[1, 2", logging=True)
# ([1, 2], [{'text': 'While parsing an array we missed the closing ], ignoring it',
#            'context': '[1, 2'}])
json_repair.repair_json("{'city': 'Zürich'", ensure_ascii=False)  # '{"city": "Zürich"}'
```

The options are json_repair's:

- `return_objects=True` returns the value instead of a string (what `loads` does).
- `logging=True` returns `(value, repair_log)`: a list of `{"text": ..., "context": ...}` dicts,
  empty when nothing was repaired.
- `skip_json_loads=True` skips the `json.loads` validity check, so valid JSON goes through the
  repair parser too.
- `stream_stable=True` keeps the repairs of a growing, streamed prefix stable.
- `strict=True` raises `ValueError` on structural problems instead of repairing them, with
  json_repair's messages (`"Missing ':' after key in strict mode while parsing object."`).
- `json_fd=` reads the text from an open file (prefer `load`); `chunk_length` is accepted and
  ignored, because pyroid reads the file at once.
- Other keyword arguments go to `json.dumps` for the string result. `ensure_ascii`, `indent`,
  `separators`, `sort_keys` and `allow_nan` are reproduced in Rust; any others call `json.dumps`.

Lone surrogates are handled as json_repair handles them (`"['a\ud800"` gives `'["a\\ud800"]'`).
Unicode-dependent checks (`str.isspace`, `isalpha`, `isdigit`, `isalnum`) follow the running
interpreter's Unicode version, 14.0 to 17.0; a newer Python gets a `RuntimeWarning` and the 17.0
tables.

## Many documents: `repair_json_batch`

`repair_json_batch` is pyroid's addition: `repair_json` over a batch on pyroid's thread pool,
without the GIL. Each result equals `json_repair.repair_json(doc, ...)` with the same
`skip_json_loads`, `stream_stable`, `strict` and `json.dumps` arguments.

```python
import pyarrow as pa
import pyroid.json_repair as json_repair

docs = ["{'a': 1", '{"b": 2}', "[1, 2,"]
fixed = json_repair.repair_json_batch(docs)  # ['{"a": 1}', '{"b": 2}', '[1, 2]']
fixed[1] is docs[1]  # True: a document that needs no change is the same object
json_repair.repair_json_batch(docs, threads=4, indent=2)  # at most 4 tasks at once

column = json_repair.repair_json_batch(pa.array(["{'a': 1", None, "[1, 2,"]))
column.to_pylist()  # ['{"a": 1}', None, '[1, 2]']
```

It takes a `str`, a `list` or `tuple` of `str`, or an Arrow string column (pyarrow, polars,
`datasets`) and returns the same kind: a [`pyroid.Column`](../reference/pyroid.md#pyroid.Column)
for Arrow input, with nulls kept. In strict mode a failing document raises for the whole batch,
and the exception's note names its index. `json.dumps` arguments other than `ensure_ascii`,
`indent`, `separators`, `sort_keys` and `allow_nan` work for `str`, `list` and `tuple` input only
(one `repair_json` call per document).

## Deep nesting: `NestingError`

pyroid's parser follows up to `MAX_DEPTH` (1,000) levels of broken nesting. Deeper input raises
`NestingError`, which is a [`pyroid.PyroidError`](../reference/pyroid.md#pyroid.PyroidError), a
`ValueError` and a `RecursionError`, so `except` clauses written for json_repair still catch it.
json_repair itself raises `ValueError` past about 140 to 330 levels, depending on
`sys.getrecursionlimit()` and the caller's stack.

```python
import pyroid.json_repair as json_repair

json_repair.repair_json("[" * 1000)  # '[]' (json_repair raises ValueError)
try:
    json_repair.repair_json("[" * 1001)
except json_repair.NestingError as e:
    print(e)  # Input nesting exceeds the supported parser recursion depth.
```

## Delegated features

These are handed to the real json_repair, which must then be installed; the first call emits a
[`pyroid.FallbackWarning`](../concepts/fallbacks.md), and without json_repair the call raises
[`pyroid.PyroidError`](../reference/pyroid.md#pyroid.PyroidError):

- every `schema=` call, which needs jsonschema or pydantic as in json_repair (listed in
  `pyroid.json_repair.DELEGATED`);
- input that is not a `str` (for example `bytes`);
- the rare text holding a character in U+10F800..U+10FFFF, raw or as an escaped surrogate pair:
  the port uses that range to store lone surrogates. In `repair_json_batch`, such documents in a
  list or tuple are delegated one by one; Arrow input raises `PyroidError` instead.

```python
json_repair.repair_json(b"{'a': 1")  # json_repair's result, and:
# FallbackWarning: pyroid.json_repair.repair_json: a bytes input is not supported by the Rust
# port; delegating to json_repair (slow path). This warning is shown once.
```

json_repair's command line (`python -m json_repair`) and its internal modules are not provided.

## Linear time on pathological input

json_repair has lookaheads and string copies that make some inputs quadratic. The port keeps
every heuristic but memoizes those scans, and stays linear on all 13 super-linear input families
found (json_repair's log-log slopes are 1.5 to 2.0). A few of them, at sizes doubling from
16 KiB to 1 MiB (CPython 3.12; times at the largest size json_repair finished;
*Docker/OrbStack VM on Apple Silicon (12 vCPU), indicative*):

| Input family (json_repair's cause) | Slope, json_repair / pyroid | json_repair → pyroid |
|---|---|---|
| `truncated_string` (`string_acc +=` copies) | 1.64 / 1.00 | 512 KB: 6.2 s → 9.2 ms; 1 MB: > 20 s → 19 ms |
| `escaped_code_string` (`[:-1]` per escape) | 1.71 / 1.02 | 1 MB: 18.1 s → 17.6 ms |
| `markdown_in_array` (a scan per `]`) | 2.00 / 1.02 | 128 KB: 10.0 s → 2.4 ms |
| `single_quoted_apostrophes` (pairwise quote walk) | 2.04 / 1.05 | 64 KB: 15.9 s → 2.2 ms |
| `records_missing_separators` (input rebuilt per split) | 1.47 / 0.97 | 1 MB: 6.1 s → 40 ms |
| `nested_open_containers` (adversarial, `{[` × 60) | O(depth·n) / 1.02 | 512 KB: 12.5 s → 9.2 ms |

## Differences from json_repair

From the module's documentation (`pyroid/json_repair.pyi`):

- Nesting: json_repair raises `ValueError` past about 140-330 levels of broken input (it depends
  on `sys.getrecursionlimit()` and the caller's stack). pyroid repairs up to `MAX_DEPTH` levels;
  deeper broken input raises `NestingError`, which is both a `ValueError` and a `RecursionError`.
  Valid JSON is decoded by `json.loads` for objects, as in json_repair (so its `RecursionError`,
  past about 10,000 levels on 3.12, is the same), and by pyroid at any depth for string results
  (json_repair raises there too).
- `load`/`from_file` read the file once (`fd.read()`) instead of in `chunk_length` pieces; the
  result is the same, `chunk_length` is ignored and unseekable streams work.
- json_repair's command line (`python -m json_repair`) is not part of this module.

And by design, the [delegated features](#delegated-features) need json_repair installed.

## Parity

0 differences in 184,403 comparisons on CPython 3.12 and 184,254 on free-threaded 3.15t (where
the `schema=` replays are skipped: no jsonschema). The suite replays json_repair's own test suite
as 672 recorded calls, 20,000 seeded broken LLM outputs under 8 option sets, Hypothesis grammar
fuzzing with lone surrogates under 9 option sets, and `load` and `from_file`. Details:
[Compatibility](../compatibility.md#json_repair).

## Speed

Single documents, CPython 3.12, best time per call (3.15t is within ±10%). *Docker/OrbStack VM on
Apple Silicon (12 vCPU), indicative*:

| Document | json_repair | pyroid | Speedup |
|---|---:|---:|---:|
| valid ~1 KB, `repair_json` | 11.3 µs | 6.6 µs | 1.7× |
| valid ~1 KB, `loads` | 6.2 µs | 6.4 µs | 1.0× |
| broken ~1 KB, `repair_json` | 600 µs | 20.1 µs | 30× |
| broken ~1 KB, `loads` | 589 µs | 22.2 µs | 27× |
| valid 1.2 MB, `repair_json` | 9.9 ms | 8.1 ms | 1.2× |
| valid 1.2 MB, `loads` | 5.7 ms | 5.8 ms | 1.0× |
| broken 1 MB, `repair_json` | 544 ms | 17.1 ms | 32× |
| broken 1 MB, `loads` | 548 ms | 18.6 ms | 30× |

On valid JSON there is little to gain: both libraries start with CPython's `json.loads`, so
`loads` of valid JSON costs the same. The speedup is on broken JSON, where json_repair's parser
runs in Python.

Batches: 10,000 broken documents of about 1.2 KB on free-threaded CPython 3.15t (GIL off), same
label:

| Run | Documents/s | vs json_repair loop |
|---|---:|---:|
| json_repair, loop | 1,204 | 1× |
| json_repair, 8-thread pool | 4,681 | 3.9× |
| pyroid `repair_json`, loop | 41,414 | 34× |
| `repair_json_batch`, `threads=1` | 46,221 | 38× |
| `repair_json_batch`, `threads=8` | 156,134 | 130× |

The 8-thread run is limited by the shared, loaded VM. See also
[Benchmarks](../benchmarks.md#pyroidjson_repair).

API: [`pyroid.json_repair` reference](../reference/json_repair.md).
