# `pyroid.tabulate`: drop-in tabulate

`pyroid.tabulate` is a Rust port of [tabulate](https://github.com/astanin/python-tabulate)
0.10.0 by Sergey Astanin and contributors (MIT), the library that prints tables for CLIs, logs,
reports and `pandas.DataFrame.to_markdown()`. It returns the same text byte for byte, for every
format and option, and raises the same exceptions with the same messages
([parity](../compatibility.md#tabulate)).

```console
$ pip install pyroid
```

## Change one import

```diff
-from tabulate import tabulate
+from pyroid.tabulate import tabulate

 print(tabulate([["spam", 41.9999], ["eggs", "451.0"]], headers=["item", "qty"], tablefmt="github"))
 # | item   |      qty |
 # |--------|----------|
 # | spam   |  41.9999 |
 # | eggs   | 451      |
```

The module has tabulate's other names too: `tabulate_formats`, `simple_separated_format`,
`TableFormat`, `Line`, `DataRow`, `SEPARATING_LINE`, `multiline_formats`, `JupyterHTMLStr`, and
the settings `MIN_PADDING`, `PRESERVE_WHITESPACE` and `WIDE_CHARS_MODE`, which `tabulate()` reads
on each call as tabulate does. `python -m pyroid.tabulate` is tabulate's command line.

For code you don't own, `DataFrame.to_markdown()` included, patch `tabulate`:

```python
import pyroid

pyroid.patch("tabulate")  # import tabulate gives pyroid's module
df.to_markdown()  # pandas now draws the table with pyroid
```

`pyroid run -p tabulate script.py` does the same for a whole program and the processes it starts
([pyroid.patch()](patch.md)). Settings written on the patched module
(`tabulate.MIN_PADDING = 0`) reach pyroid.tabulate too.

## What is the same

Everything `tabulate()` takes, with tabulate's defaults:

- input: lists of lists or tuples (ragged rows too), lists of dicts, dataclasses and named
  tuples, dicts of columns, numpy arrays and record arrays, pandas DataFrames (with their index),
  DB-API cursors, generators;
- `headers` as a list, `"firstrow"`, `"keys"` or a dict; `showindex` as `"default"`, `"always"`,
  `"never"`, a bool or the values to show;
- all 37 formats in `tabulate_formats`, `simple_separated_format`, and custom `TableFormat`s,
  callables included;
- `floatfmt`, `intfmt` and `missingval` as one value or one per column; `numalign`, `stralign`,
  `colglobalalign`, `colalign`, `headersglobalalign`, `headersalign` and `rowalign`, including the
  `UserWarning` tabulate gives for a bare string;
- `disable_numparse` as a bool or a list of columns, `preserve_whitespace`, `maxcolwidths`,
  `maxheadercolwidths`, `break_long_words` and `break_on_hyphens`;
- multiline cells, `SEPARATING_LINE` rows, ANSI colour codes and terminal hyperlinks, and wide
  characters measured with the installed `wcwidth` when `WIDE_CHARS_MODE` is on.

Column types follow tabulate's rules, quirks and all: `"1,000"` is an int, an empty string does
not stop a column from being numeric, `"42992e1"` is a float unless you pass `disable_numparse`,
numpy's unsigned ints are typed as floats, and `"infinity"` is text while `"inf"` is a number.

## What runs where

| Step | Runs |
|---|---|
| Column types, number formatting (Python's format-spec language for floats and ints), ANSI stripping, widths of ASCII text, alignment, drawing every built-in format | in Rust |
| Turning a plain list of rows, or a DataFrame of floats, signed ints or objects, into rows and adding the index | in Rust |
| Other inputs (dicts, dataclasses, numpy arrays, generators, other DataFrames) | tabulate's own normalisation code, in Python |
| `maxcolwidths` and `maxheadercolwidths` wrapping | tabulate's own text wrapper, in Python |
| Cells Rust does not type itself (Decimal, Fraction, dates, bytes, subclasses, ints beyond 64 bits, numpy scalars other than float, int, uint and bool) | tabulate's `_type` and `_format`, per cell |
| Widths of text that is not ASCII, with `WIDE_CHARS_MODE` on | the installed `wcwidth.wcswidth` |
| Callables of a custom `TableFormat` | called from Rust, as tabulate calls them |

The Python parts are tabulate's code, copied; a test checks they still match tabulate 0.10.0's
source, AST for AST.

These calls go to the real tabulate, which must then be installed. The first call for each
reason emits a [`pyroid.FallbackWarning`](../concepts/fallbacks.md), and without tabulate the
call raises [`pyroid.PyroidError`](../reference/pyroid.md#pyroid.PyroidError):

- text with lone surrogates (`"\ud800"`), in a cell, a header or a custom format's output;
- an alignment that is not a `str`, `None` or a `bool`;
- a custom `TableFormat` whose lines and rows are not strings or callables, or whose `padding` is
  not an int, and a `MIN_PADDING` that is not an int;
- a `missingval` that is not a `str`.

## Speed

Measured on a dedicated AWS EC2 c7i.4xlarge (Intel Xeon, 16 vCPU), Amazon Linux 2023, Docker,
CPython 3.12.16, tabulate 0.10.0 with wcwidth 0.9.4, best of several runs; each pyroid result was
checked equal to tabulate's first:

| Table | tabulate | pyroid | Speedup |
|---|---:|---:|---:|
| 5 rows x 3, mixed, `simple` | 97 µs | 12 µs | 7.8x |
| 1,000 x 8, numbers, `simple` | 24.3 ms | 1.40 ms | 17x |
| 1,000 x 8, text, `simple` | 41.6 ms | 1.58 ms | 26x |
| 1,000 x 8, mixed, `grid` | 31.7 ms | 1.12 ms | 28x |
| 100,000 x 10, numbers, `pipe` | 3.29 s | 232 ms | 14x |
| 100,000 x 10, text, `pipe` | 5.40 s | 310 ms | 17x |
| 100,000 x 10, mixed, `github` | 3.98 s | 181 ms | 22x |
| `DataFrame.to_markdown()`, 1,000 x 8 numbers | 35.2 ms | 2.04 ms | 17x |
| `DataFrame.to_markdown()`, 100,000 x 10 numbers | 4.71 s | 308 ms | 15x |

All formats and sizes are on the [benchmarks page](../benchmarks.md#pyroidtabulate)
(`bench/results/aws-c7i.4xlarge/2026-11-16/tabulate.json`, made by
`bench/tabulate/tabulate_bench.py`). A one-row table of three numbers takes 11 µs (tabulate:
33 µs), most of it fixed per-call work. pyroid.tabulate runs on free-threaded CPython without
enabling the GIL.

## Versions {#versions}

pyroid.tabulate reproduces tabulate 0.10.0, the newest release. With another tabulate installed,
`pyroid.patch("tabulate")` warns once that the patched functions give 0.10.0's results. tabulate
0.9.0 (2022) is still common; 0.10.0 changed what it prints in these cases:

- an empty string in a number column counts as missing, so the column stays right-aligned (0.9.0:
  the column became text);
- `"1,000"` and `"1,000.5"` are numbers;
- numpy integer scalars are ints, not floats;
- `mediawiki` writes `style="text-align: right;"` (0.9.0: `align="right"`) and `asciidoc`
  `cols="<5,>7"` (0.9.0: `cols="5<,7>"`);
- `github` and the `*outline` formats fold multiline cells;
- new: the `colon_grid` format and the `colglobalalign`, `headersglobalalign`, `headersalign`,
  `break_long_words` and `break_on_hyphens` arguments; `PRESERVE_WHITESPACE` became the
  `preserve_whitespace` argument.

On the 48 public CSV files of the parity corpus, 0.9.0 and 0.10.0 draw 8 files differently in
every format, every `mediawiki` and `asciidoc` table differs, and 1 of the 48
`DataFrame.to_markdown()` outputs differs.

## Differences from tabulate {#differences-from-tabulate}

No differences in output are known. Some behaviour around the edges:

- With `maxcolwidths` narrower than a wide character (say `maxcolwidths=1` and `"日本"`),
  tabulate's text wrapper loops forever, and so does pyroid, which runs the same wrapper.
- `pyroid.tabulate` provides tabulate's private helpers that its tests use (`_CustomTextWrap`,
  `_strip_ansi`, `_type`, `_column_type`, `_align_column`, ...); `_align_column` and
  `_align_cell_veritically` run pyroid's Rust code. Other private helpers (`_format_table`,
  `_visible_width`, ...) are not provided; under `pyroid.patch("tabulate")` they come from the
  installed tabulate.
- The fallbacks above need tabulate installed.
