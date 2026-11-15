# `pyroid.dateutil`: drop-in `dateutil.parser`

`pyroid.dateutil.parser` is a Rust port of the date parser in
[python-dateutil](https://github.com/dateutil/dateutil) (BSD-3-Clause). `parse` reads dates the
way people write them (`"Thu Sep 25 10:36:28 BRST 2003"`, `"25/09/2003"`, `"10h36m"`), and
`isoparse` reads ISO 8601. pyroid tokenizes and resolves the string in Rust and returns what
dateutil returns: the same `datetime`, the installed dateutil's own time zone objects
(`tz.UTC`, `tz.tzoffset(...)`, `tz.tzlocal()`), and dateutil's own `ParserError` with the same
message. `parse` is the same in python-dateutil 2.9.0.post0, 3.0.0 and 4.0.0, and pyroid
reproduces all three ([parity](../compatibility.md#dateutil)).

## Change one import

```diff
-from dateutil import parser
+from pyroid.dateutil import parser

 parser.parse("Thu Sep 25 10:36:28 BRST 2003", tzinfos={"BRST": -10800})
 # datetime.datetime(2003, 9, 25, 10, 36, 28, tzinfo=tzoffset('BRST', -10800))
 parser.isoparse("2003-09-25T10:49:41+03:00")
 # datetime.datetime(2003, 9, 25, 10, 49, 41, tzinfo=tzoffset(None, 10800))
```

Code you cannot edit can use [`pyroid.patch("dateutil")`](#pyroidpatchdateutil).

## What is ported

Everything in `dateutil.parser`, with dateutil's signatures:

- `parse(timestr, parserinfo=None, **kwargs)` with every keyword: `default`, `ignoretz`,
  `tzinfos` (a dict of name to `tzinfo`, `TZ` string or offset in seconds, or a function),
  `dayfirst`, `yearfirst`, `fuzzy` and `fuzzy_with_tokens`.
- `parser(info=None).parse(...)` and `parser._parse(...)`, which returns dateutil's
  `_result` object.
- `parserinfo(dayfirst=False, yearfirst=False)`, including subclasses that change the word lists
  (`JUMP`, `WEEKDAYS`, `MONTHS`, `HMS`, `AMPM`, `UTCZONE`, `PERTAIN`, `TZOFFSET`). These run in
  Rust like the default ones.
- `isoparse(dt_str)` and `isoparser(sep=None)` with `parse_isodate`, `parse_isotime` and
  `parse_tzstr`.
- `ParserError` and `UnknownTimezoneWarning`. With python-dateutil installed these are
  dateutil's own classes, so `except dateutil.parser.ParserError` catches pyroid's errors.

`timestr` can be a `str`, `bytes`, `bytearray` or a text stream, as in dateutil. Values that
`datetime` rejects raise the interpreter's own message, wrapped in `ParserError` as dateutil
wraps it (`"day is out of range for month: Feb 30 2021"`). Python's `str` methods follow the running interpreter's Unicode version (14.0 to 17.0),
so digits from other scripts (`"٢٠٢٤-٠١-٠٢"`) and letters count as they do for dateutil.

```python
import datetime

from pyroid.dateutil import parser

parser.parse("Today is January 1, 2047 at 8:21:00AM", fuzzy_with_tokens=True)
# (datetime.datetime(2047, 1, 1, 8, 21), ('Today is ', ' ', ' ', 'at '))
parser.parse("01/05/09", dayfirst=True)  # datetime.datetime(2009, 5, 1, 0, 0)
parser.parse("10:00", default=datetime.datetime(2003, 9, 25))  # 2003-09-25 10:00


class German(parser.parserinfo):
    MONTHS = [
        ("Jan", "Januar"),
        ("Feb", "Februar"),
        ("Mär", "März"),
        ("Apr", "April"),
        ("Mai",),
        ("Jun", "Juni"),
        ("Jul", "Juli"),
        ("Aug", "August"),
        ("Sep", "September"),
        ("Okt", "Oktober"),
        ("Nov", "November"),
        ("Dez", "Dezember"),
    ]


parser.parse("25. März 2003", German())  # datetime.datetime(2003, 3, 25, 0, 0)
```

The rest of python-dateutil (`dateutil.tz`, `relativedelta`, `rrule`, `easter`) is not ported:
use dateutil's.

## Time zones and python-dateutil

pyroid returns the time zone objects dateutil would: `tz.UTC` for `Z`, `UTC`, `GMT` and zero
offsets, `tz.tzoffset(name, seconds)` for offsets, `tz.tzlocal()` for a name in `time.tzname`,
and whatever `tzinfos` gives. With python-dateutil installed they are made by its own `tz`
module, so they are its classes and its cached instances.

python-dateutil is not a dependency. Without it, pyroid uses copies of dateutil 4.0.0's
`tzutc`, `tzoffset` and `tzlocal` (same behaviour, `repr` and equality, but other classes), and
its own `ParserError` and `UnknownTimezoneWarning` (same messages). A `str` value in `tzinfos`
needs `dateutil.tz.tzstr` and raises `pyroid.PyroidError` without it.

## Versions: `isoparse` in 2.9 and 4.0

`parse` did not change between python-dateutil 2.9.0.post0, 3.0.0 and 4.0.0. `isoparse` did, in
4.0.0: each numeric field must now be exactly N ASCII digits. 2.9.0.post0 and 3.0.0 read the
fields with `int()`, which takes spaces, signs and underscores, so `"2014- 4-21"`,
`"2_14-04-21"` and an offset of `+-1:00` parse there and raise in 4.0.0, and
`"20200728121240"` reads as 21:24. The error messages differ too.

`isoparser(sep=None, *, compat=None)` picks the release:

| `compat` | `isoparse` of |
|---|---|
| `None` (the default) | the installed python-dateutil: 4.0.0, or 2.9.0.post0 for 2.9.0.post0 and 3.0.0; 4.0.0 when it is not installed |
| `"4.0.0"` | python-dateutil 4.0.0 |
| `"2.9.0.post0"` or `"3.0.0"` | python-dateutil 2.9.0.post0 and 3.0.0 |

The module-level `isoparse` uses `compat=None`, so it agrees with the dateutil installed next to
it. With another python-dateutil release installed (2.8.2, say), importing the module warns once
that results can differ from it.

## Many strings: `parse_batch` and `isoparse_batch`

These are pyroid's additions. They parse a batch on pyroid's thread pool, without the GIL, and
build the results on the calling thread.

```python
import pyarrow as pa
from pyroid.dateutil import parser

parser.parse_batch(["2003-09-25", "Sep 26 2003 10:00", "nonsense"], errors="coerce")
# [datetime.datetime(2003, 9, 25, 0, 0), datetime.datetime(2003, 9, 26, 10, 0), None]

column = parser.parse_batch(pa.array(["2003-09-25 10:00", None, "Sep 26 2003 10:00:00.5"]))
column.type  # 'timestamp[us]'
pa.array(column)  # [2003-09-25 10:00:00.000000, null, 2003-09-26 10:00:00.500000]

parser.parse_batch(pa.array(["2003-09-25 10:00Z", "2003-09-25 10:00+05:30"])).type
# 'timestamp[us, tz=UTC]'
parser.isoparse_batch(["2003-09-25T10:49:41", "2003-W39-4"], compat="4.0.0")
```

- Input: a `str`, a `list` or `tuple` of `str`, or an Arrow string column (pyarrow, polars,
  `datasets`). Output: the same kind. Arrow input gives a
  [`pyroid.Column`](../reference/pyroid.md#pyroid.Column) of `timestamp[us]`.
- Each result is what `parse(s, ...)` gives with the same arguments (`fuzzy_with_tokens` is not
  accepted), except that `default=None` means today at midnight read once for the batch, not per
  string.
- `errors="raise"` (the default) raises the first failing string's exception, with a note naming
  its index. `errors="coerce"` gives `None` (an Arrow null) for strings that raise `ValueError`,
  such as `ParserError`, or `OverflowError`.
- The Arrow time zone rule: when every result is naive, the column is `timestamp[us]` and holds
  the wall time; when every result is aware, it is `timestamp[us, tz=UTC]` and holds the instant
  (each `datetime` converted to UTC with its `utcoffset()`). Results that mix naive and aware
  raise `ValueError`; pass `ignoretz=True`, or a list, for those. Nulls stay null.
- `threads=None` uses the shared pool, `threads=n` at most `n` tasks. Batches of 64 strings or
  fewer run on the calling thread.

## `pyroid.patch("dateutil")`

`pyroid.patch("dateutil")` (or [`pyroid run -p dateutil`](patch.md)) sets pyroid's `parse` and
`isoparse` on python-dateutil's own modules: `dateutil.parser.parse`,
`dateutil.parser._parser.parse`, `dateutil.parser.isoparse` and
`dateutil.parser.isoparser.isoparse`. They behave as dateutil's module-level functions: they parse
with dateutil's own `DEFAULTPARSER`, `parser` class and `DEFAULT_ISOPARSER`, so changes made to
those (a word added to `DEFAULTPARSER.info`) still apply, and `isoparse` follows the installed
release. Everything else stays dateutil's: `dateutil.tz`, `relativedelta`, `rrule`, `easter`, and
the `parser` and `isoparser` classes and their methods.

```python
import pyroid

pyroid.patch("dateutil")

from dateutil import parser

parser.parse("Thu Sep 25 10:36:28 2003")  # parsed by pyroid
pyroid.unpatch("dateutil")
```

`pandas.to_datetime` calls `dateutil.parser.DEFAULTPARSER._parse`, not `parse`, so patching does
not change or speed up pandas. Names imported before the patch (`from dateutil.parser import
parse`) keep dateutil's function.

## Fallbacks

These run dateutil's own algorithm (pyroid's copy of dateutil 4.0.0's parser, so they need no
python-dateutil), with a one-time [`pyroid.FallbackWarning`](../concepts/fallbacks.md) per reason:

- a `parser` subclass that overrides one of its private methods (`_parse`,
  `_parse_numeric_token`, `_build_naive`, ...), or a `parserinfo` subclass that overrides a method
  (`convertyear`, `validate`, ...) rather than the word lists;
- a parserinfo whose tables are not dicts of `str` to small `int`s;
- a number past 64 bits, or a fractional number with more than 26 significant digits (where the
  `decimal` context rounds or raises), or a thread whose `decimal` precision is below what the
  arithmetic needs;
- a text stream whose `read(1)` does not return one-character strings;
- a string holding both lone surrogates and characters in U+10F800..U+10FFFF.

An unexpected keyword argument runs dateutil's code too (no warning), so the `TypeError` is
dateutil's. `parse_batch` and `isoparse_batch` handle these strings one by one in the same way.

## Parity

There were 0 differences in all of these comparisons:

- python-dateutil's own `test_parser.py` and `test_isoparser.py`, run against dateutil and against
  pyroid: the same outcome for each of 830 tests with 4.0.0 and 813 with 2.9.0.post0.
- The parity suite: those tests, 747 strings from dateutil's and pandas' issue trackers under 10
  option sets, Hypothesis, `parse_batch` and a corpus sample. 222,675 comparisons on CPython 3.12
  and on free-threaded 3.14t, and 222,658 against 2.9.0.post0.
- Every string of a corpus of 9,246,745 real ones from 35 public sources, through `parse` under
  3 option sets and through `isoparse`. The sources are system and web server logs, email and
  HTTP headers, CSV date columns, and CLDR's date formats in 760 locales. That is 147,947,920
  comparisons, against 4.0.0 under `TZ=UTC`, `Europe/London` and `America/New_York` and
  against 2.9.0.post0.

Details: [Compatibility](../compatibility.md#dateutil).

## Speed

These timings are from a dedicated AWS EC2 c7i.4xlarge (Intel Xeon, 16 vCPU), Amazon Linux
2023, in Docker. The inputs are 608,343 real strings, up to 20,000 from each corpus source,
those that dateutil cannot parse included, and every result was first checked against
dateutil's. Against python-dateutil 2.9.0.post0 on CPython 3.12 (`bench/dateutil_bench.py`):

| Call | µs per string | vs dateutil |
|---|---:|---:|
| `dateutil.parser.parse`, loop | 27.0 | 1× |
| pyroid `parse`, loop | 1.76 | 15× |
| `parse_batch`, list, 1 thread / all 16 | 1.19 / 0.60 | 23× / 45× |
| `parse_batch`, Arrow, 1 thread / all 16 | 0.73 / 0.27 | 37× / 101× |
| `pandas.to_datetime(format="mixed")` (its own parser, other results) | 29.1 | 0.9× |

A loop of single calls also pays for a Python call per string and, for each string that fails, a
raised and caught exception. Arrow output skips making a `datetime` per row on the calling
thread, which is why it scales with threads. On free-threaded CPython 3.14t, dateutil itself on 8
threads takes 4.9 µs per string (5.3×), and pyroid's Arrow batch 0.27 µs.

On the 139,939 strings among them that `datetime.fromisoformat` accepts, `isoparse` takes
0.97 µs against dateutil's 3.5 µs (3.6×; on 3.14t, 5.5× on the 126,638 strings that its
`fromisoformat` accepts). `datetime.fromisoformat` takes 0.13 µs, but it accepts other strings
and returns other objects. The [benchmarks page](../benchmarks.md#pyroiddateutil) has every row.

## Differences from dateutil

- `tz.tzlocal()`: for a local zone without daylight saving time (`TZ=UTC`, for one), pyroid
  returns the same `tzlocal()` instance for every result while the zone stays the same; dateutil
  makes a new, equal one per call. With daylight saving time, dateutil's own steps run.
- `parserinfo`: pyroid's parserinfo tables are a `dict` subclass that reports changes, and pyroid
  re-reads a parserinfo after any change to its attributes or tables. A parserinfo made by
  python-dateutil itself (under `patch()`, or passed in) is checked on every call, by the identity
  and size of each table, so replacing one value in place (`info._months["jan"] = 5`) is not seen
  until something else changes.
- `UnknownTimezoneWarning` points at the caller's line, not at a line inside dateutil.
- `parse_batch` and `isoparse_batch` take `str` only (no `bytes` or streams), without lone
  surrogates, as every pyroid batch function does.

API: [`pyroid.dateutil.parser` reference](../reference/dateutil.md).
