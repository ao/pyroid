# `pyroid.json_repair`: drop-in json_repair

`pyroid.json_repair` is a Rust port of [json_repair](https://github.com/mangiucugna/json_repair)
0.64.9 by Stefano Baccianella (MIT). json_repair turns broken JSON from LLMs into valid JSON:
missing quotes, single quotes, comments, trailing commas, truncation, markdown fences, prose
around it, Python literals. Its parser is ported line by line, so pyroid returns the same
strings, objects and repair logs as json_repair 0.64.9 ([parity](../compatibility.md#json_repair)).
It repairs broken documents about 15 times faster, stays linear on the inputs where json_repair is
still quadratic, and deep nesting never overflows the stack.

## Change one import

```diff
-import json_repair
+import pyroid.json_repair as json_repair

 json_repair.repair_json("{'a': 1, b: [1, 2,}")    # '{"a": 1, "b": [1, 2]}'
 json_repair.loads('{"title": "Notes on the Analytical Eng')
 # {'title': 'Notes on the Analytical Eng'}
```

For code you don't own, `pyroid.patch("json_repair")` makes a later `import json_repair` (and
`from json_repair.json_repair import ...`) give these functions; see
[`pyroid.patch()`](patch.md).

## Repairing and decoding

`repair_json`, `loads`, `load` and `from_file` take json_repair's arguments:

| Function | Returns |
|---|---|
| `repair_json(json_str, ...)` | the repaired JSON as a string (`""` when no JSON was found) |
| `loads(json_str, ...)` | the decoded value, like `json.loads` after a repair |
| `load(fd, ...)` | the same, reading an open text file |
| `from_file(filename, ...)` | the same, opening the file by name |

Valid JSON comes back as `json.dumps(json.loads(text))`. Anything else goes through json_repair's
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
- `json_fd=` reads the text from an open file (prefer `load`). `chunk_length` is accepted and
  ignored, because pyroid reads the file at once.
- Other keyword arguments go to `json.dumps` for the string result. `ensure_ascii`, `indent`,
  `separators`, `sort_keys` and `allow_nan` are reproduced in Rust; any others call `json.dumps`.

Lone surrogates are handled as json_repair handles them (`"['a\ud800"` gives `'["a\\ud800"]'`).
Unicode-dependent checks (`str.isspace`, `isalpha`, `isdigit`, `isalnum`) follow the running
interpreter's Unicode version, 14.0 to 17.0. A newer Python gets a `RuntimeWarning` and the 17.0
tables.

## Many documents: `repair_json_batch`

`repair_json_batch` is pyroid's addition. It runs `repair_json` over a batch on pyroid's thread
pool, without the GIL. Each result equals `json_repair.repair_json(doc, ...)` with the same
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
`datasets`) and returns the same kind. For Arrow input that is a
[`pyroid.Column`](../reference/pyroid.md#pyroid.Column), with nulls kept. In strict mode a
failing document raises for the whole batch, and the exception's note names its index or row.

`json.dumps` arguments other than `ensure_ascii`, `indent`, `separators`, `sort_keys` and
`allow_nan` work for `str`, `list` and `tuple` input only (one `repair_json` call per document).

An Arrow column cannot hold lone surrogates. json_repair keeps an escape such as `\ud800` as a
lone surrogate, and so does an escaped pair such as `\ud83d\ude00` when its own parser reads it.
With the default `ensure_ascii=True` the result writes it back as the escape, so nothing changes.
With `ensure_ascii=False`, Arrow input raises `UnicodeEncodeError` for such a result, and the
note names the row; pass a list to get json_repair's `str`.

## Deep nesting: `NestingError`

pyroid's parser follows up to `MAX_DEPTH` (1,000) levels of broken nesting: arrays and objects
in any mix, where an object that json_repair reads again as an array (`[{[{`) is one level. Deeper
input raises
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

These are handed to the real json_repair, which must then be installed. The first call emits a
[`pyroid.FallbackWarning`](../concepts/fallbacks.md), and without json_repair the call raises
[`pyroid.PyroidError`](../reference/pyroid.md#pyroid.PyroidError):

- every `schema=` call, which needs jsonschema or pydantic as in json_repair (listed in
  `pyroid.json_repair.DELEGATED`);
- input that is not a `str` (for example `bytes`);
- the rare text holding a character in U+10F800..U+10FFFF, raw or as an escaped surrogate pair,
  because the port uses that range to store lone surrogates. In `repair_json_batch`, only such
  documents are delegated, one by one, from a list, a tuple or an Arrow column.

```python
json_repair.repair_json(b"{'a': 1")  # json_repair's result, and:
# FallbackWarning: pyroid.json_repair.repair_json: a bytes input is not supported by the Rust
# port; delegating to json_repair (slow path). This warning is shown once.
```

json_repair's command line (`python -m json_repair`) and its internal modules are not provided.

## Pathological input

json_repair has lookaheads and string copies that make some inputs quadratic. The port keeps
every heuristic but memoizes those scans, so it stays linear on all 13 input families found by
reading json_repair 0.64.5's code. json_repair 0.64.9 made 9 of them linear too. Two are still
quadratic there, and two grow with the nesting depth times the length:

| Input family (json_repair's cause) | Slope, json_repair / pyroid | json_repair → pyroid |
|---|---|---|
| `records_missing_separators` (the input is rebuilt at each split) | 2.0 / 1.0 | 1 MB: 5.1 s → 33 ms |
| `regex_class_quotes` (adversarial: quotes inside `[...]`) | 2.0 / 1.0 | 16 KB: 6.3 s → 0.38 ms; 32 KB: over 20 s → 0.79 ms |
| `nested_open_tuples` (adversarial: 100 levels of `(`) | depth × length / 1.0 | 1 MB: 18.3 s → 36 ms |
| `nested_open_containers` (adversarial: `{[` × 60) | depth × length / 1.0 | 256 KB: 7.1 s → 4.8 ms |

On the other nine families both are linear and pyroid is 7 to 39 times faster at 1 MB (for
example `truncated_string` 6.8×, `markdown_in_array` 17×, `single_quoted_apostrophes` 29×). The
slopes are log-log fits over the 3 largest of sizes doubling from 2 KiB to 1 MiB.
`bench/json_repair_bench.py`, CPython 3.12, on an EC2 c8g.16xlarge devbox (shared, indicative).

These families come from reading json_repair's code. Fuzzing finds other inputs, made of a short
unit of garbage repeated, that are quadratic in both. There json_repair's parser backtracks:
it parses a value to the end of the text, drops it and parses again a few characters further on,
or rescans an unclosed run of brackets from each bracket. pyroid follows the same steps to give the
same output, so it is faster by a constant factor only. Two of them, repeated to 4,096 and
16,384 characters (json_repair 0.64.9, same machine):

| Input | json_repair, 4 K / 16 K | pyroid, 4 K / 16 K |
|---|---|---|
| `\x0c\0\0\0c{]=` repeated | 1.25 s / 20.1 s | 13 ms / 210 ms |
| `{"a": "` then `{a: [` repeated (brackets inside a string value) | 0.57 s / 9.2 s | 7.7 ms / 119 ms |

## Differences from json_repair

From the module's documentation (`pyroid/json_repair.pyi`):

- json_repair raises `ValueError` past about 140-330 levels of broken input (it depends on
  `sys.getrecursionlimit()` and the caller's stack). pyroid repairs up to `MAX_DEPTH` levels, and
  deeper broken input raises `NestingError`, which is both a `ValueError` and a `RecursionError`.
- Valid JSON is decoded by `json.loads` for objects, as in json_repair, so its `RecursionError`
  past about 10,000 levels on 3.12 is the same. For string results pyroid decodes it at any depth
  (json_repair raises there too).
- `load`/`from_file` read the file once (`fd.read()`) instead of in `chunk_length` pieces. The
  result is the same and `chunk_length` is ignored.
- json_repair's command line (`python -m json_repair`) is not part of this module.

And by design, the [delegated features](#delegated-features) need json_repair installed.

## Versions

Each pyroid release reproduces one json_repair release, normally the newest one, and names it in
`pyroid.json_repair.JSON_REPAIR_VERSION`. pyroid 1.0.0 reproduced 0.64.5; this release
reproduces 0.64.9. When another version of json-repair is installed, `import pyroid.json_repair`
emits one [`pyroid.VersionMismatchWarning`](../compatibility.md#installed-versions) that names
both versions. Delegated calls always run the installed json_repair.

What changed between json_repair 0.64.5 and 0.64.9, and so between pyroid 1.0.0 and this release:

- 0.64.6: valid JSON holding an empty string is returned as JSON: `repair_json('""')` is now
  `'""'`, not `''`. `''` still means that no value was found.
- 0.64.6: a stream that cannot seek (`seekable()` is false) is read whole and repaired as a
  string, so the valid-JSON shortcuts for strings apply to it.
- 0.64.6: `schema=` rejects invalid schemas with `ValueError("Invalid schema: ...")` and leaves
  `const`, `enum`, `default` and `examples` values alone. These calls are delegated, so they follow
  the installed json_repair.
- 0.64.8: two commas after a number in an object no longer turn the number into a string:
  `{"a": 1,, "b": 2}` gives `{"a": 1, "b": 2}` (it was `{"a": "1,", "b": 2}`), and a stray number
  between the commas is dropped. Strict mode raises on it.
- 0.64.8: a number that ends in a bare exponent drops it: `[1e, 2]` gives `[1, 2]` (it was
  `[1, "e", 2]`) and `[1e+]` gives `[1]` (it was `["1e"]`). Strict mode raises on it.
- 0.64.9: faster on long inputs. Its results were the same as 0.64.8's on the 20,000 generated
  broken LLM outputs of the parity suite and on 3,152 fuzzing inputs.

On those 20,000 outputs, 39 give a different result between json_repair 0.64.5 and 0.64.9 under
at least one option set, most of them because of the number changes.

## Parity

There were 0 differences against json_repair 0.64.9 in 184,552 comparisons on CPython 3.12 and
184,386 each on free-threaded 3.14t and 3.15t, where the `schema=` replays are skipped for lack of
jsonschema. The suite replays json_repair's own test suite as 821 recorded calls, 20,000
seeded broken LLM outputs under 8 option sets, Hypothesis grammar fuzzing with lone surrogates
under 9 option sets, and `load` and `from_file`. Details: [Compatibility](../compatibility.md#json_repair).

## Speed

These timings come from an EC2 c8g.16xlarge devbox (Graviton4, shared with other work),
indicative. Single documents, json_repair 0.64.9, CPython 3.12, best time per call:

| Document | json_repair | pyroid | Speedup |
|---|---:|---:|---:|
| valid ~1 KB, `repair_json` | 12.6 µs | 7.5 µs | 1.7× |
| valid ~1 KB, `loads` | 6.6 µs | 6.6 µs | 1.0× |
| broken ~1 KB, `repair_json` | 367 µs | 21.0 µs | 17× |
| broken ~1 KB, `loads` | 358 µs | 23.6 µs | 15× |
| valid 1.2 MB, `repair_json` | 10.1 ms | 7.6 ms | 1.3× |
| valid 1.2 MB, `loads` | 5.7 ms | 5.8 ms | 1.0× |
| broken 1 MB, `repair_json` | 232 ms | 16.0 ms | 14.5× |
| broken 1 MB, `loads` | 228 ms | 17.6 ms | 13× |

On valid JSON there is little to gain. Both libraries start with CPython's `json.loads`, so
`loads` of valid JSON costs the same. A decoder in Rust would not change that much: on the same
devbox, orjson, the fastest Python JSON decoder, took valid ~1 KB and 1.9 MB documents 1.3× and
1.7× faster than `json.loads`, and pyroid's own Rust decoder (with the conversion to Python
objects) was slower than `json.loads` on both, so `loads` keeps calling `json.loads`. The speedup
is on broken JSON, where json_repair's parser runs in Python. json_repair 0.64.9 is itself faster than 0.64.5 there: on the same machine 0.64.5
took 625 µs for the broken ~1 KB document and 425 ms for the broken 1 MB one.

For batches, 10,000 broken documents of about 1.2 KB on free-threaded CPython 3.15t (GIL off),
on the same machine while other work kept the load average between 30 and 40:

| Run | Documents/s | vs json_repair loop |
|---|---:|---:|
| json_repair, loop | 2,828 | 1× |
| json_repair, 8-thread pool | 16,127 | 5.7× |
| pyroid `repair_json`, loop | 48,056 | 17× |
| `repair_json_batch`, `threads=1` | 49,131 | 17× |
| `repair_json_batch`, `threads=8` | 260,003 | 92× |

Raw results: `bench/results/aws-c8g.16xlarge-devbox/2026-11-14-json_repair/`. See also
[Benchmarks](../benchmarks.md#other-modules-languages-and-integrations).

API: [`pyroid.json_repair` reference](../reference/json_repair.md).
