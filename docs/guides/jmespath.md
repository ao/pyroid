# `pyroid.jmespath`: drop-in jmespath

`pyroid.jmespath` is a Rust port of [jmespath](https://github.com/jmespath/jmespath.py) 1.1.0
(MIT), the JMESPath implementation behind boto3's and botocore's paginators and waiters, the
AWS CLI's `--query` and Ansible's `json_query`. Its lexer, parser, interpreter and functions are
ported method by method, so pyroid returns the same results, ASTs and exceptions as jmespath
1.1.0 ([parity](../compatibility.md#jmespath)). It evaluates expressions on your Python objects
directly, without converting the document.

```console
$ pip install pyroid
```

## Change one import

```diff
-import jmespath
+import pyroid.jmespath as jmespath

 jmespath.search("Reservations[].Instances[].InstanceId", response)
 expression = jmespath.compile("items[?price > `10`].name | sort(@)")
 expression.search(data)
 jmespath.search("{a: a}", data, jmespath.Options(dict_cls=collections.OrderedDict))
```

Everything else stays as it was: `except jmespath.exceptions.JMESPathError`,
`jmespath.functions.Functions` subclasses with `@signature` functions, `compile(...).parsed`,
`jmespath.parser.Parser().parse(...)`, `jmespath.lexer.Lexer().tokenize(...)`. With jmespath
installed, the exception classes and `_Expression` are jmespath's own, so errors from pyroid and
from jmespath are the same classes. jmespath is not required: without it, pyroid has copies.

To make code you don't own use pyroid (boto3, botocore, the AWS CLI, Ansible), patch it:

```python
import pyroid

pyroid.patch("jmespath")  # jmespath.search and jmespath.compile are pyroid.jmespath's
```

`pyroid run -p jmespath script.py` does the same for a whole program and the processes it starts
([pyroid.patch()](patch.md)). boto3's `paginator.search(...)`, botocore's paginators and waiters
call `jmespath.compile` and `jmespath.search` on the `jmespath` module, so they pick it up.

## What is the same

Results are the objects jmespath returns: the input's own sub-objects where jmespath returns
them (`a.b`, `items[0]`, the elements of a projection), new lists and dicts where it builds them,
and the cached expression's literal objects for literals. `int` and `float` come out as jmespath
computes them (`sum`, `avg`, `to_number`, `ceil` and `floor` go through CPython's own `sum`, `/`,
`int()`, `float()` and `math`, so 3.12's compensated float sums are the interpreter's), `sort_by`,
`min_by` and `max_by` keep jmespath's order for equal keys, and multi-select hashes keep key order
(or use your `dict_cls`).

jmespath's Python-specific rules hold too: a field is `value.get(key)` on any object that has a
`get` method, projections and indexes need `isinstance(x, list)` (a tuple gives `null`), the
function type checks go by `type(x).__name__` (so an `OrderedDict` is an object, a list subclass
is not an array), `0` and `1` never equal `false` and `true`, ordering works on numbers
(`numbers.Number`, so `Decimal` too) and strings, and comparing a string with a number raises
Python's `TypeError`. Objects other than the builtin types go through the same Python operations
jmespath applies to them. Type names added to `jmespath.functions.REVERSE_TYPES_MAP` or
`TYPES_MAP` count as in jmespath (Ansible's `json_query` adds its tagged `str`, `list` and `dict`
subclasses); with jmespath installed, `pyroid.jmespath.functions` holds the same two dicts.

Errors have jmespath's classes (`ParseError`, `LexerError`, `IncompleteExpressionError`,
`EmptyExpressionError`, `ArityError`, `VariadictArityError`, `JMESPathTypeError`,
`UnknownFunctionError`), messages and attributes:

```python
>>> jmespath.compile("foo]baz")
jmespath.exceptions.ParseError: Unexpected token: ]: Parse error at column 3, token "]" (RBRACKET), for expression:
"foo]baz"
    ^
```

Compiled expressions are cached as jmespath 1.1.0 caches them: the 512 most recent, the oldest
out first, shared by `search`, `compile` and `Parser().parse`, so `compile(e) is compile(e)`.
An expression is found as jmespath's dict finds it, so a `str` subclass with its own `__eq__` or
`__hash__` (a case-insensitive string, say) gets the entry that compares equal to it. While such
an expression is cached, each lookup compares hashes with every cached entry, which adds about
0.4 µs when 512 are cached (EC2 c8g.16xlarge devbox, shared, CPython 3.12); once it leaves the
cache, lookups go by the text again.
A deeply nested expression raises `RecursionError` at the nesting depth where jmespath's
recursive parser and interpreter run out of frames, computed from `sys.getrecursionlimit()` and
the caller's stack depth (on the main thread; see the differences below). A long chain such as
`a || b || c ...` or `a[][]...` compiles at any length, as in jmespath (both parsers build it in a
loop), and searching it raises `RecursionError` where jmespath's interpreter does.

## Custom functions

```python
from pyroid.jmespath import functions  # or jmespath.functions


class Custom(functions.Functions):
    @functions.signature({"types": ["string"]})
    def _func_shout(self, s):
        return s.upper()


jmespath.search("shout(name)", {"name": "x"}, jmespath.Options(custom_functions=Custom()))
```

A subclass of pyroid's `Functions` or of jmespath's runs the built-in functions it inherits
natively and its own functions in Python, with jmespath's arguments and checks. Expression
references (`&field`) reach your function as `_Expression` objects whose `visit(expression,
value)` evaluates natively. A subclass that overrides `call_function` or the argument checks gets
every call through its own `call_function`, as in jmespath.

A function added to the `FUNCTION_TABLE` of `jmespath.functions.Functions` or of pyroid's
`Functions` is found without custom functions too, as jmespath's default `Functions()` finds it,
and runs through that class's `call_function`.

## What runs in Rust

All of it, for `str` expressions: the lexer, the parser, the interpreter and the 26 built-in
functions. Python runs only where jmespath's results are defined by Python: `json.loads` for
literals and quoted identifiers (once per expression, then cached), `json.dumps` in
`to_string`, CPython's numeric builtins, the `__eq__`, `__lt__`, `get` and `values` of objects
other than `dict`, `list`, `str`, `int`, `float`, `bool` and `None`, and custom functions.

An expression that is not a `str` goes through jmespath's own lexer, which iterates over it: empty
values, unhashable values and `bytes` raise what jmespath raises, and anything else (a tuple of
characters, say) is handed to jmespath with one [`pyroid.FallbackWarning`](../concepts/fallbacks.md),
or raises `pyroid.PyroidError` without jmespath.

pyroid.jmespath runs on free-threaded CPython without enabling the GIL; the expression cache is
the only shared state and is locked.

## Speed

On a dedicated AWS EC2 c7i.4xlarge (Intel Xeon) with CPython 3.12: a one-field `search()` on a
boto3-style response takes 0.14 µs (jmespath: 1.5 µs) and a compiled one 0.09 µs; projections,
paths, filters and multi-selects on the same response run 10 to 26 times faster than jmespath's.
Over 10,000 records, projections and filters run 15 to 20 times faster, `sort_by` and `max_by` 12
to 13 times, `sum` 7 times; over 1,000,000 records, 4 to 17 times. A new expression every call
(parse and search) takes 5.5 µs (jmespath: 55 µs). The
[benchmarks page](../benchmarks.md#pyroidjmespath) has the table.

Most of what remains per call is CPython's own work: building the result lists and dicts,
reference counting, and the `==`, `get` and `sum` calls jmespath's semantics require.

## Versions

pyroid.jmespath reproduces jmespath 1.1.0 (January 2026). jmespath 1.0.1, still common in older
lockfiles, differs in two places: `avg()` divides by `float(len(arg))` (a different last bit for
integer sums past 2^53, `OverflowError` for sums past about 1e308), and its parser cache holds 128
expressions and evicts a random half (with the `random` module) when full. `pyroid.patch("jmespath")`
with another jmespath installed says so in its `PatchWarning`.

## Differences from jmespath {#differences-from-jmespath}

- Deep nesting in threads other than the main thread: pyroid keeps its native stack within 192
  KiB there (96 KiB on musl), so a deeply nested expression raises `RecursionError` at about 100 to
  410 levels, depending on the construct, where jmespath allows 250 to 500 at the default
  recursion limit (CPython 3.12 on aarch64 Linux). On the main thread the depths are jmespath's.
- With a raised recursion limit, pyroid stops nested expressions (parentheses, `!`,
  multi-selects, function calls, filters, projections) at 500 to 1,000 levels. Chains of `|`,
  `||`, `&&`, comparisons and `[]` go as deep as the recursion limit and a 6 MiB native stack
  allow on the main thread: 3,200 to 10,000 terms at a limit of 20,000, where jmespath on CPython
  3.12 stops at 2,500 to 5,000.
- Type names removed from jmespath's own entries in `REVERSE_TYPES_MAP` or `TYPES_MAP` still
  count: pyroid checks those entries natively and reads the dicts only for other types.
- The `PendingDeprecationWarning` for a backtick literal that is not JSON points at the line that
  called `search`, `compile`, `Parser().parse` or `Lexer().tokenize`; jmespath's points into its
  lexer.
- `ParsedResult.parsed` is built on first access and is not read back: editing its dicts in place
  does not change what `search` evaluates (assigning a new AST to `.parsed` does).
- The `_Expression` objects pyroid makes carry a native interpreter: `expref.interpreter` is not a
  `TreeInterpreter` object (its other attributes come from one).
- Iterating a dict for `*` takes a snapshot of its values, so a custom function that resizes the
  dict being projected does not raise jmespath's `RuntimeError: dictionary changed size during
  iteration`.
- `pyroid.jmespath.visitor.TreeInterpreter` is jmespath's Python interpreter (copied), for code
  that drives or subclasses it; `search` and `ParsedResult.search` do not go through it.
- Without custom functions, the built-in functions run natively without reading
  `Functions.FUNCTION_TABLE` (jmespath's or pyroid's), since reading it would add several dict
  lookups to every search that calls a function. So replacing or removing a built-in's entry
  there, or replacing `call_function` or the argument checks on that class itself, does not
  change what `search` does. Functions added to the table are found. To change a built-in,
  pass a `Functions` subclass as `Options(custom_functions=...)`.
