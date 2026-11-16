# `pyroid.difflib`: drop-in difflib

`pyroid.difflib` is a Rust port of the standard library's
[`difflib`](https://docs.python.org/3/library/difflib.html): `SequenceMatcher`'s
Ratcliff/Obershelp matching, `get_close_matches`, `unified_diff`, `context_diff`, `ndiff`,
`Differ` and `HtmlDiff`. Results are the running interpreter's own, for CPython 3.11 to 3.15:
the same ratios (the same float bits), matching blocks, opcodes, diff lines and HTML
([parity](../compatibility.md#difflib)). rapidfuzz is fast too, but it computes other scores
(Levenshtein-based); pyroid.difflib is for code that needs difflib's exact numbers and output.

```console
$ pip install pyroid
```

## Change one import

```diff
-import difflib
+import pyroid.difflib as difflib

 difflib.SequenceMatcher(None, "Acme Corp.", "ACME Corporation").ratio()
 difflib.get_close_matches("appel", ["ape", "apple", "peach", "puppy"])
 "".join(difflib.unified_diff(old_lines, new_lines, "a/setup.py", "b/setup.py"))
 difflib.HtmlDiff(wrapcolumn=80).make_file(old_lines, new_lines)
```

`SequenceMatcher`, `Differ` and `HtmlDiff` are subclasses of the standard library's classes, so
`isinstance(m, difflib.SequenceMatcher)` holds for both modules and your own subclasses keep
working: an overridden `find_longest_match`, `_qformat` or `_format_line` is called where difflib
calls it. `Match`, `IS_LINE_JUNK`, `IS_CHARACTER_JUNK`, `restore` and `diff_bytes` are the
standard library's own objects.

To make code you don't own use pyroid, patch the standard library:

```python
import pyroid

pyroid.patch("difflib")
```

This patches `difflib` in place rather than replacing it: pyroid's methods are set on
`difflib.SequenceMatcher`, `difflib.Differ` and `difflib.HtmlDiff` themselves, and its
`get_close_matches`, `unified_diff`, `context_diff` and `_mdiff` on the module. So classes and
functions imported before the patch, subclasses defined earlier and `isinstance` checks all keep
working, and `ndiff`, `HtmlDiff`, unittest's `assertEqual` diffs and pytest's assertion diffs run
natively. `pyroid run -p difflib -m pytest` does the same for a whole program and the processes it
starts ([pyroid.patch()](patch.md)); `pyroid.unpatch("difflib")` puts everything back. Matchers
made or used under the patch keep working after it: `unpatch` gives them the attributes
difflib's own methods read (it looks through the live objects once, with `gc.get_objects()`).

## What is the same

Every number and line is the interpreter's own difflib's. `ratio()`, `quick_ratio()` and
`real_quick_ratio()` compute `2.0 * M / T` with the same float operations. `isjunk` is called once
on each distinct element of `b` in the order of first occurrence, as `__chain_b` calls it, and
autojunk drops the same popular elements (more than 1% + 1 occurrences in a `b` of 200 or more).
`b2j`, `bjunk`, `bpopular`, `fullbcount`, `matching_blocks` and `opcodes` are there with
CPython's contents and key order, `get_matching_blocks()` and `get_opcodes()` return their cached
lists, and `get_grouped_opcodes()` changes the cached opcodes as CPython's does. Pickling and
copying give CPython's state. `get_close_matches` returns the possibilities' own objects, best
first, with ties broken as `heapq.nlargest` breaks them.

Where CPython versions differ, the running version's behaviour is followed:

- 3.14 changed how `Differ` (and so `ndiff` and `HtmlDiff`) pairs up similar lines: it searches a
  window of 10 lines around the diagonal instead of the best pair of the whole block, so many
  `ndiff` outputs differ between 3.13 and 3.14. pyroid has both.
- 3.14 makes `unified_diff` and `context_diff` reject a `str` for `a` or `b`; 3.15 adds
  `unified_diff(color=True)`.
- `HtmlDiff`'s styles and page template changed in 3.12.10, 3.13.3, 3.14 and 3.15 (and 3.15 marks
  changed lines with `class="diff_changed"`). pyroid uses the interpreter's own templates.

On another CPython version (3.16 and later), `pyroid.difflib` is the standard library's difflib
and says so with one [`pyroid.FallbackWarning`](../concepts/fallbacks.md) at import. There,
`get_close_matches` accepts `threads=` and ignores it, `pyroid.patch("difflib")` patches nothing
(its `PatchWarning` says so), and `ratio_batch` and `close_matches_batch` still run natively, with
3.15's `SequenceMatcher`.

## What runs in Rust

- `SequenceMatcher` over a `str`, a `bytes` or `bytearray`, or a `list` or `tuple` whose elements
  are `str`, `int` (of any size), `bool` or `bytes`: types whose equality pyroid knows without
  calling Python. Subclasses of these (a `str` subclass, an `IntEnum`, a named tuple) count too,
  unless they define their own comparisons, `__hash__`, `__len__`, `__getitem__` or `__iter__`.
  Lone surrogates are fine. A list that mixes `str` and `bytes` elements runs natively too, except
  under `python -b`, where comparing them warns: there it runs difflib's code, which does.
- `get_close_matches` for a `str` word and `str` possibilities (or such subclasses), on all
  cores for long lists.
- `Differ.compare`, `ndiff`, `unified_diff`, `context_diff`, `HtmlDiff.make_table` and
  `make_file` for lists or tuples of `str`, with any `linejunk` (called from Rust as difflib calls
  it) and a `charjunk` of `None` or `IS_CHARACTER_JUNK` (the defaults).

The rest runs the standard library's Python code, rebound to `pyroid.difflib` so that it uses
pyroid's `SequenceMatcher` underneath: a custom `charjunk`, a `Differ` or `HtmlDiff` subclass
that overrides one of its helper methods, a `SequenceMatcher` subclass that overrides
`find_longest_match` or `get_matching_blocks`. Sequences of other element types (tuples, floats,
objects with their own `__eq__`) run the standard library's `SequenceMatcher` code, with one
`pyroid.FallbackWarning` that names the type that triggered it, for example "a list with an
element of type 'float'".

pyroid.difflib runs on free-threaded CPython without enabling the GIL. Lists are read in one step,
so another thread changing a list during a call cannot crash it. A long call (matching long
sequences, diffing large files, a batch) acts on Ctrl-C within about 0.1 s and raises
`KeyboardInterrupt`; the matcher it ran on is unchanged and stays usable. In a batch on several
threads, an item a pool thread has started is finished first.

## Many strings at once

`ratio_batch` and `close_matches_batch` take lists, tuples or Arrow string arrays (pyarrow,
polars, ...) and use all cores:

```python
import pyroid.difflib as difflib

difflib.ratio_batch(["Acme Corp.", "Globex"], ["ACME Corporation", "Globex Inc"])
# [SequenceMatcher(None, "Acme Corp.", "ACME Corporation").ratio(), ...]
difflib.ratio_batch("appel", candidates)  # one str against every element of the other
difflib.close_matches_batch(queries, vocabulary, n=3, cutoff=0.6)
# [get_close_matches(q, vocabulary, 3, 0.6) for q in queries]
```

Both give exactly what the loops in the comments give. `isjunk=` in `ratio_batch` is called once
per distinct character before the work starts, so it must be a plain function of the character.
Arrow nulls give `None`. Pass `threads=n` to use at most `n` tasks at once.

## Speed

Measured on a dedicated AWS EC2 c7i.4xlarge (Intel Xeon, 16 vCPU) with CPython 3.12.
`SequenceMatcher(None, a, b).ratio()` on a pair of company or place names takes 2.1 µs in a
Python loop (difflib: 10 µs). Most of that is creating the matcher object; `ratio_batch` takes
0.64 µs per pair on one thread and 0.19 µs on 16 threads. On 500-character pairs, the loop takes
27 µs (difflib: 398 µs). `get_close_matches` over 100,000 place names takes 10 ms on one thread
and 5 ms on 16 (difflib: 144 ms). Diffs of argparse.py between CPython 3.12 and 3.14 (2,700
lines) take 0.8 ms with `unified_diff` (difflib: 5.6 ms), 2.7 ms with `ndiff` (31 ms) and 8.5 ms
with `HtmlDiff().make_file` (52 ms). The [benchmarks page](../benchmarks.md#pyroiddifflib) has
the table. For context, rapidfuzz's `fuzz.ratio` takes 0.25 µs per name pair on one thread, but
it computes a different score.

## Differences from difflib {#differences-from-difflib}

- A sequence is read when it is first needed (`set_seq2(b)` reads `b`, the first ratio or
  opcodes read `a`), and later changes made to that same list in place are not seen. CPython reads
  `a` again in each `quick_ratio()` and reads elements on demand; call `set_seq1`/`set_seq2` with
  a new list instead.
- `b2j`, `bjunk` and `bpopular` are built on first read; editing them does not change the
  matching (in CPython it would).
- So `vars()` of a matcher differs: `b2j`, `bjunk`, `bpopular`, `fullbcount` and
  `matching_blocks` are in it only once read, and pyroid keeps its native state there under
  `_pyroid_` names (also in `dir()`). Pickles and copies have CPython's attributes only.
- On 3.11 to 3.13, `Differ` pairs lines by recursion, and CPython raises `RecursionError` on long
  runs of similar lines (149 such lines at a recursion limit of 120, say). `HtmlDiff(wrapcolumn=...)`
  wraps a line by recursion in every version, so a line of about 1,000 wrapped rows raises it too.
  pyroid keeps the same order without recursion and returns the diff.
- `get_close_matches` reads a non-list, non-tuple `possibilities` into a list before it starts.
- A copied or unpickled `SequenceMatcher` has CPython's state and runs CPython's code until its
  next `set_seq2` (the results are the same).
- `ratio_batch` and `close_matches_batch` follow pyroid's text-input rules: lone surrogates raise
  `UnicodeEncodeError` there (they are fine everywhere else).
