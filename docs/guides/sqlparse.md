# `pyroid.sqlparse`: drop-in sqlparse

`pyroid.sqlparse` is a Rust port of [sqlparse](https://github.com/andialbrecht/sqlparse) 0.6.1
(BSD-3-Clause), the non-validating SQL parser and formatter behind Django's SQL handling,
django-debug-toolbar, database shells and many SQL tools. 0.6.1 is sqlparse's final release.
The lexer, the statement splitter, the grouping passes and the formatting filters are ported
line by line, so `format`, `split`, `parse` and `lexer.tokenize` give sqlparse 0.6.1's text,
statements, token trees and exceptions ([parity](../compatibility.md#sqlparse)), 15 to 160 times
faster per call on one core ([speed](#speed)).

```console
$ pip install pyroid
```

## Change one import

```diff
-import sqlparse
+import pyroid.sqlparse as sqlparse

 sqlparse.format("select a, b from t where x = 1", reindent=True, keyword_case="upper")
 # 'SELECT a,\n       b\nFROM t\nWHERE x = 1'
 sqlparse.split("select 1; select 2;")  # ['select 1;', 'select 2;']
 stmt, = sqlparse.parse("select a.b as c from t")
 stmt.get_type()  # 'SELECT'
```

Everything else stays as it was: `sqlparse.sql.Identifier`, `stmt.tokens`, `token.ttype is
sqlparse.tokens.Keyword`, `get_real_name()`, `flatten()`, `except
sqlparse.exceptions.SQLParseError`, `sqlparse.engine.FilterStack` with your own filters,
`sqlparse.lexer.Lexer.get_default_instance().add_keywords(...)`.

To make code you don't own use pyroid (Django, django-debug-toolbar, SQL tools), patch it:

```python
import pyroid

pyroid.patch("sqlparse")  # sqlparse.format, split, parse, parsestream are pyroid's
```

`pyroid run -p sqlparse manage.py sqlmigrate app 0001` does the same for a whole program and the
processes it starts ([pyroid.patch()](patch.md)). With sqlparse installed, the patch sets
pyroid's `parse`, `parsestream`, `format`, `split`, `lexer.tokenize` and
`engine.grouping.group` on sqlparse's own modules, so code that imported `sqlparse` before the
patch (Django does at start-up) calls pyroid, and sqlparse's own `FilterStack` pipelines (the
debug toolbar's SQL panel) lex and group in Rust around their Python filters. The DoS limits
are then read from `sqlparse.engine.grouping`. Without sqlparse, `import sqlparse` gives
pyroid's modules.

## The token objects

`parse` returns trees of real Python objects, made the way sqlparse makes them. Which classes
they are depends on what is installed:

| Installed | Token classes (`sql`), token types (`tokens`), `SQLParseError` | Lexer singleton and keyword dictionaries |
|---|---|---|
| sqlparse 0.6.1 | sqlparse's own | sqlparse's own: customising either library's lexer customises both |
| sqlparse 0.5.0 to 0.6.0 | sqlparse's own | pyroid's copies of 0.6.1's |
| no sqlparse (or older than 0.5.0) | pyroid's copies of 0.6.1's | pyroid's copies |

So with sqlparse installed, `isinstance(t, sqlparse.sql.Identifier)` and `t.ttype is
sqlparse.tokens.Keyword` hold for pyroid's trees, and trees mix freely: sqlparse's filters run
on pyroid's trees, `pyroid.sqlparse.engine.grouping.group` groups a statement sqlparse's splitter
made, pickles and `copy.deepcopy` work as with sqlparse's own trees. The methods of the tokens
(`get_real_name`, `token_next_by`, ...) are the installed release's Python code; with another
release than 0.6.1 installed, pyroid warns once at import (a `RuntimeWarning`), because parsing
and formatting follow 0.6.1.

The objects carry sqlparse's state exactly, including what sqlparse leaves stale: a group's
`value` is the joined values of its children when it was built (and extended), not `str(group)`;
`normalized` is set once; grouping an operator sets `ttype` without touching `is_keyword`.

## What runs in Rust

The lexer (sqlparse's delimited-spans pass and its regex table, matched by an engine that gives
`re`'s results: Unicode `\w` and `\b`, IGNORECASE folding such as `ı`/`ſ`/`K`, lookbehinds), the
splitter, all 25 grouping passes, and every filter: keyword and identifier case, string
truncation, comment and whitespace stripping, spaces around operators, `reindent`,
`reindent_aligned`, the python and php output formats, and the serializer. `format` and `split`
build no Python objects at all; `parse` builds the token objects at the end.

Python runs only where sqlparse's behaviour is Python's own:

- `formatter.validate_options` checks the options (so the errors are sqlparse's);
- `Lexer.get_default_instance()` is asked for the lexer on each call, and its `_SQL_REGEX` and
  `_keywords` are compiled again when they change (`set_SQL_REGEX`, `add_keywords`, `clear`,
  `default_initialization`). A pattern that uses syntax pyroid's matcher does not reproduce
  (`re.VERBOSE`, possessive quantifiers, ...) runs in Python's `re` for that entry only, with one
  [`pyroid.FallbackWarning`](../concepts/fallbacks.md);
- a `Lexer` with overridden methods, or a text holding both lone surrogates and characters from
  U+10F800 on, runs sqlparse's own Python code (pyroid ships it, so sqlparse need not be
  installed), with one `FallbackWarning`;
- when sqlparse's code raises on some input (an `IndexError` in a filter on malformed SQL, say),
  pyroid runs that code for the same exception.

## Batches

```python
docs = ["select a from t", "insert into t values (1)", ...]
sqlparse.format_batch(docs, reindent=True)        # == [sqlparse.format(d, reindent=True) for d in docs]
sqlparse.split_batch(docs, strip_semicolon=True)  # == [sqlparse.split(d, strip_semicolon=True) ...]
sqlparse.format_batch(pa.array(docs), threads=8)  # a pyroid.Column of strings; nulls stay null
```

`format_batch` and `split_batch` take a list, a tuple or an Arrow string array or column, run on
the shared thread pool, check for Ctrl-C between chunks, and raise what the loop would raise.

## Limits, deep input, threads

The DoS limits work as in sqlparse 0.6.1: grouping a statement that nests deeper than
`engine.grouping.MAX_GROUPING_DEPTH` (100) or has a token list longer than `MAX_GROUPING_TOKENS`
(10,000) raises `SQLParseError("Maximum grouping depth exceeded (100).")` or `("Maximum number
of tokens exceeded (10000).")`. Set them on `pyroid.sqlparse.engine.grouping` (or, under the
patch, on `sqlparse.engine.grouping`); `None` disables them. `split` does not group, so the
limits do not apply to it.

Deep statements are grouped on a helper thread with a large stack, so a thread with a small stack
(`threading.stack_size`, musl) never crashes. With the limits disabled, nesting past about 980
levels raises `SQLParseError("Maximum recursion depth exceeded")`, the error sqlparse raises when
CPython's recursion limit stops it.

Long calls check for Ctrl-C while lexing and between statements, and let other threads take the
GIL every few milliseconds. On free-threaded CPython the module runs without the GIL; the lexer
snapshot and the token-type table are the only shared state and are locked. As in sqlparse, a
call made while another thread re-initialises the lexer may see its table half built.

## Differences from sqlparse

- With the limits disabled, the nesting depth at which `Maximum recursion depth exceeded` is
  raised is close to sqlparse's but not the same (sqlparse's depends on the interpreter's
  recursion limit and the caller's stack).
- The lexer reads the whole text before the first statement is grouped (sqlparse lexes as the
  splitter asks); results are the same, but Python regex entries of a customised table are all
  called before the first statement comes back, and `parsestream` reads the DoS limits once, when
  it starts.
- A keyword dictionary is re-read when its size changes or the lexer's lists change. Replacing the
  value of an existing key in place (`KEYWORDS["LIMIT"] = tokens.Name`) is seen after the next
  `add_keywords`, `set_SQL_REGEX`, `clear` or `default_initialization`.
- `format_batch` and `split_batch` are not part of sqlparse.

## Versions

pyroid reproduces 0.6.1, the final release, with no mode for 0.5.x. The releases since 0.5.3
changed results here and there: the DoS limits (0.5.4, raising since 0.5.5), the rewritten
splitter (0.6.0: nested BEGIN ... END, BEGIN TRANSACTION, `;` inside blocks), backslash escaping
in the python and php output formats and `$` escaping in php (security fixes), linear-time
lexing of unclosed comments and dollar quotes, `END FOR`/`END CASE`/`IF EXISTS`/`REGEXP BINARY`
tokens, new keywords, and fixes in grouping and `reindent` for malformed input. See sqlparse's
changelog for the list.

## Speed

EC2 c8g.16xlarge devbox (Graviton4, 64 vCPU), shared, indicative: CPython 3.12.16, a release
build of pyroid (not the PGO-built release wheels), sqlparse 0.6.1, both libraries in one
process, best of 7 rounds, load average about 2
([full table](https://github.com/ao/pyroid/blob/main/bench/results/aws-c8g.16xlarge/2026-11-17/sqlparse.md),
`bench/sqlparse/sqlparse_bench.py`).

| Input | Call | sqlparse (µs) | pyroid (µs) | Speedup |
|---|---|---:|---:|---:|
| 1-line SELECT | `format()` | 96 | 6.3 | 15× |
| 1-line SELECT | `format(reindent=True)` | 375 | 12.8 | 29× |
| 1-line SELECT | `split()` | 94 | 3.0 | 31× |
| 1-line SELECT | `parse()` | 300 | 20.0 | 15× |
| 4-line SELECT | `format()` | 743 | 18.8 | 39× |
| 4-line SELECT | `format(reindent=True)` | 2,804 | 64.7 | 43× |
| 4-line SELECT | `split()` | 716 | 11.0 | 65× |
| 4-line SELECT | `parse()` | 2,271 | 124.2 | 18× |
| INSERT, 200 rows | `format()` | 15,259 | 186.0 | 82× |
| INSERT, 200 rows | `format(reindent=True)` | 42,601 | 884.8 | 48× |
| INSERT, 200 rows | `split()` | 15,346 | 97.3 | 158× |
| INSERT, 200 rows | `parse()` | 38,729 | 2,127.2 | 18× |
| migration, 120 statements | `format()` | 14,474 | 331.4 | 44× |
| migration, 120 statements | `format(reindent=True)` | 56,260 | 1,247.8 | 45× |
| migration, 120 statements | `split()` | 13,975 | 163.4 | 86× |
| migration, 120 statements | `parse()` | 44,748 | 2,859.6 | 16× |

Batches of 20,000 one-line SELECTs: `format_batch(docs, reindent=True)` took
38 ms on all 64 cores and 440 ms on one, against 19.0 s for a loop over
sqlparse (502× and 43×); `split_batch` 14 ms and 71 ms against 4.9 s.

`parse` gains less than `format` and `split` because it ends by creating sqlparse's Python
objects (about 190 for the 4-line SELECT), and the cycle collector then has to free them, as it
does for sqlparse's own trees.
