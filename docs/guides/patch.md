# `pyroid.patch()`: speed up code you don't own

Many libraries call ftfy or fastText internally: CLIP-style tokenizers run `ftfy.fix_text` on
every string, language detectors call `fasttext.load_model`. You can't change their imports, but
you can change what `import ftfy` returns. `pyroid.patch()` does that, explicitly and reversibly:

```python
import pyroid

pyroid.patch("ftfy", "fasttext")  # before the imports you want redirected

import fasttext
import ftfy

ftfy.fix_text("naÃ¯ve rÃ©sumÃ©")  # 'naïve résumé', computed by pyroid.ftfy
model = fasttext.load_model("lid.176.bin")  # a pyroid.fasttext.Model
model.predict("Bonjour tout le monde")  # (('__label__fr',), array([0.99457073]))
```

After `patch()`, `import ftfy` gives a stand-in module whose functions are pyroid's own function
objects. A patched call gives exactly what the pyroid module gives and costs what calling pyroid
costs. [`pyroid.ftfy`](ftfy.md) is byte-identical to ftfy 6.3.1, and
[`pyroid.fasttext`](fasttext.md) bit-identical to fastText's predict path. On three unmodified
third-party call sites the output was identical patched and unpatched
([below](#verified-call-sites), [parity](../compatibility.md#integrations)).

Nothing happens until you ask. Installing pyroid changes no imports (there are no `.pth` files),
and `patch()` is the only switch. Each patched name emits one
[`PatchWarning`](#patchwarning) per process, saying what was replaced and how to undo it.

## What can be patched

| `patch()` name | Redirected to | What `import <name>` gives from pyroid | Still from the real package |
|---|---|---|---|
| `"ftfy"` | [`pyroid.ftfy`](ftfy.md) (ftfy 6.3.1) | `fix_text`, `fix_text_segment`, `fix_encoding`, `fix_and_explain`, `fix_encoding_and_explain`, `TextFixerConfig`, `ExplainedText`, `ExplanationStep`, the `ftfy` alias, `fix_text_batch` | `apply_plan`, `explain_unicode`, `fix_file`, `guess_bytes`, `ftfy.fixes` and the other submodules |
| `"fasttext"` | [`pyroid.fasttext`](fasttext.md) (fastText 0.9.2's inference, as fasttext-predict 0.9.2.4) | `load_model`, `tokenize`, `fasttext.FastText.load_model`, and `fasttext.FastText._FastText(path)` (which returns a pyroid model) | training (`train_supervised`, ...) and `_FastText()` without a model file |
| `"anyascii"` | [`pyroid.translit.anyascii`](translit.md) (anyascii 0.3.3) | `anyascii` | the rest of the package |

`fasttext.FastText._FastText(path)` is how datatrove and others load models, and
`isinstance(model, _FastText)` is true for pyroid's models too. Calls that pyroid hands to the
real package (an ftfy option value the Rust port does not support, `_FastText()` without a model
file) go to the real one, with the usual one-time
[`pyroid.FallbackWarning`](../concepts/fallbacks.md).

### Why there is no `"unidecode"`

[`pyroid.translit.unidecode`](translit.md) has Unidecode's signature but transliterates with
anyascii's tables, so its output differs from Unidecode's. It is not a drop-in, and `patch()`
says so:

```python
import pyroid

pyroid.patch("unidecode")
# ValueError: cannot patch 'unidecode': the supported names are 'ftfy', 'fasttext', 'anyascii'
# (pyroid.translit.unidecode transliterates with anyascii's tables, so its output differs from
# Unidecode's: it is not a drop-in)
```

## How it works

`patch("ftfy")` imports the real ftfy (if it is installed) and puts a shim in
`sys.modules["ftfy"]`: a module named like the real one that holds pyroid's functions. So a later
`import ftfy`, `from ftfy import fix_text` or `import fasttext.FastText` gets the shim, and:

- every attribute the shim does not provide comes from the real package, such as
  `ftfy.guess_bytes` and `fasttext.train_supervised`;
- real submodules keep working, as attributes and as `import ftfy.fixes`. The shim shares the
  real package's `__path__` and a copy of its `__spec__`, so `importlib.util.find_spec` and
  `importlib.resources` see the real package;
- `ftfy.__pyroid__` is `True`, `ftfy.__wrapped__` is the real module, and `ftfy.__version__` /
  `anyascii.__version__` are the versions pyroid reproduces (`"6.3.1"`, `"0.3.3"`);
- modules that imported the real package before `patch()` keep it, so call `patch()` before
  importing the library whose imports you want redirected.

```python
import pyroid

pyroid.patch("ftfy")

import ftfy
import ftfy.fixes

import pyroid.ftfy

ftfy.fix_text is pyroid.ftfy.fix_text  # True
ftfy  # <module 'ftfy' (pyroid shim; wrapping <module 'ftfy' from '.../ftfy/__init__.py'>)>
ftfy.guess_bytes  # <function guess_bytes at 0x...>: the real ftfy's
```

Patching a name that is already patched does nothing. `patch()` raises `ValueError` for an
unknown name (and patches nothing then), `TypeError` without names, and `ImportError` if the
pyroid module behind a name can't be imported (`pyroid.fasttext` needs numpy).

### Without the real package

The shim also works when the real package is not installed. What pyroid provides works, and any
other attribute raises an `AttributeError` that says so. `importlib.util.find_spec("fasttext")`
still finds the shim. This makes libraries usable where the real package has no wheel, such as
fastText on CPython 3.14+ and on free-threaded builds. fast-langdetect, whose import fails there
unpatched, works patched on 3.15t. ftfy's delegated functions (`apply_plan`, `explain_unicode`,
`fix_file`, `guess_bytes`) raise [`pyroid.PyroidError`](../concepts/errors.md) without ftfy.

## Undo: `unpatch`, `patched`, `patching`

```python
import pyroid
import pyroid.integrations

pyroid.patch("ftfy")
pyroid.integrations.patched()  # ('ftfy',)
pyroid.unpatch()  # every patched name
pyroid.integrations.patched()  # ()

with pyroid.integrations.patching("anyascii"):
    import anyascii

    anyascii.anyascii("Björk")  # 'Bjork', from pyroid.translit
pyroid.integrations.patched()  # ()
```

- `unpatch(*names)` gives `sys.modules` back exactly the entries it had for each name and its
  submodules before `patch()`: the real module if it had been imported, else no entry.
  Submodules imported meanwhile are dropped, so the next import is fresh. With no names it
  undoes every patch, and names that are not patched are ignored. Modules that imported a shim
  keep it, and it keeps working.
- `patched()` returns the names currently patched.
- `patching(*names)` patches for the duration of a `with` block, and on exit unpatches only the
  names this block patched (a name patched before stays patched).

`pyroid.patch` and `pyroid.unpatch` are `pyroid.integrations.patch` and
`pyroid.integrations.unpatch`. `patched`, `patching` and `PatchWarning` live in
`pyroid.integrations`. They import nothing until called, and they are thread-safe (one lock).

## `PatchWarning`

`patch()` warns once per process for each name, at the line that called it:

```console
$ cat script.py
import pyroid

pyroid.patch("ftfy")
$ python script.py
/tmp/run/script.py:3: PatchWarning: pyroid.integrations.patch: `import ftfy` now gives pyroid's drop-in (pyroid.ftfy, which reproduces ftfy 6.3.1). Everything else (apply_plan, explain_unicode, fix_file, guess_bytes, ftfy.fixes and the other submodules) still comes from the installed ftfy. Modules that imported ftfy before this call keep the real one. Undo with pyroid.integrations.unpatch('ftfy'). This warning is shown once.
  pyroid.patch("ftfy")
```

It warns once more if the installed package is another version than the one pyroid reproduces.
The patched functions then give that version's results, and the rest is the installed version's.
`PatchWarning` is a `UserWarning`, distinct from `pyroid.FallbackWarning`. To silence it:

```python
import warnings

import pyroid.integrations

warnings.simplefilter("ignore", pyroid.integrations.PatchWarning)
```

## `pyroid run`: programs and their child processes

`patch()` affects one interpreter. Data pipelines often start more, through `multiprocessing`
workers or `subprocess` calls. `pyroid run` runs a program with the patch applied in it and in
every Python process it starts, without editing a line:

```console
$ pyroid run -p ftfy,fasttext train.py --epochs 3
```

```text
usage: pyroid run [-p NAMES] (-m MODULE | SCRIPT) [ARGS...]

Run a Python script or module, as `python SCRIPT` / `python -m MODULE` would, with
`import ftfy` / `import fasttext` / `import anyascii` giving pyroid's drop-ins. The Python
processes it starts (multiprocessing workers, subprocesses) are patched too.

options:
  -p, --patch NAMES  comma-separated modules to patch: ftfy, fasttext, anyascii
                     (default: $PYROID_PATCH, else ftfy,fasttext)
  -m MODULE          run a library module as a script
  -h, --help         show this help and exit
```

With this `train.py`:

```python
import multiprocessing as mp
import subprocess
import sys

import ftfy


def worker(_: int) -> bool:
    import ftfy

    return ftfy.__pyroid__


if __name__ == "__main__":
    print("main:", ftfy.__pyroid__, sys.argv[1:], flush=True)
    with mp.get_context("spawn").Pool(2) as pool:
        print("spawn workers:", pool.map(worker, range(2)), flush=True)
    child = [sys.executable, "-c", "import ftfy; print('subprocess:', ftfy.__pyroid__)"]
    subprocess.run(child, check=True)
```

`pyroid run -p ftfy,fasttext train.py --epochs 3` prints (after the two `PatchWarning`s):

```text
main: True ['--epochs', '3']
spawn workers: [True, True]
subprocess: True
```

`pyroid run`:

- patches `NAMES` (comma-separated). Without `-p` it patches the names in `$PYROID_PATCH`, else
  `ftfy,fasttext`, and then skips `fasttext`, with a `PatchWarning`, if numpy is missing;
- sets `PYROID_PATCH` to those names and puts pyroid's start-up hook directory first on
  `PYTHONPATH`. The hook is a `sitecustomize.py`, which Python imports at start-up. It applies
  `$PYROID_PATCH` in every Python process started from there (`multiprocessing` spawn and
  forkserver workers, `subprocess`), then runs the environment's own `sitecustomize`, if there is
  one. Forked workers inherit the patch anyway;
- runs the target in the same process, as `python SCRIPT ARGS` / `python -m MODULE ARGS` would
  (`sys.argv`, `sys.path[0]`, `__name__ == "__main__"`). It exits with the target's exit code: 0
  when it finishes, 2 for a usage error or a missing script, 1 when the module can't be found.

`PatchWarning` is shown once per process tree, not in every worker. The hook does nothing in an
interpreter that can't import pyroid, nor in one started with `-I` or `-E` (which ignore
`PYTHONPATH`).

### Without `pyroid run`

The same works with two environment variables, for example in a job script:

```console
$ export PYROID_PATCH=ftfy,fasttext
$ export PYTHONPATH="$(python -c 'import pyroid.integrations._patch as p; print(p.HOOK_DIR)')"
$ python train.py
```

## Verified call sites

Each library ran unmodified in a fresh interpreter, once with the real package and once after
`patch()`, in Docker on Linux aarch64:

| Library | Call site | Compared | Result |
|---|---|---:|---|
| open_clip 3.3.0 (torch 2.15, CPU) | ViT-B-32 tokenizer: `ftfy.fix_text` on every text | 592 texts, 392 of them changed by ftfy | identical token ids; `tokenizer.ftfy.fix_text is pyroid.ftfy.fix_text` |
| fast-langdetect 1.1.0 | `fasttext.load_model` + `predict`, lite model | 600 texts | identical on CPython 3.13; works patched on 3.15t, where the unpatched import fails (no fastText wheel) |
| datatrove 0.10.1 | `FT176LID(k=3)`: `from fasttext.FastText import _FastText` | 592 documents | identical to fasttext-predict |

fast-langdetect, patched:

```python
import pyroid

pyroid.patch("fasttext")

from fast_langdetect import detect

detect("Bonjour, comment allez-vous aujourd'hui ?", model="lite", k=1)
# [{'lang': 'fr', 'score': 0.9748966097831726}]
```

open_clip, patched:

```python
import pyroid

pyroid.patch("ftfy")

import open_clip

tokenizer = open_clip.get_tokenizer("ViT-B-32")
ids = tokenizer(["a photo of a cat", "a diagram"])
```

For datatrove itself, use the [drop-in pipeline steps](datatrove.md) rather than patching
fastText. They batch the whole step, not only the model call.

## Limitations

- A shim is not the real module object: code that already holds the real module (imported before
  `patch()`), or compares module objects, is not redirected.
- `importlib.reload` of a shim is not supported.
- Only `ftfy`, `fasttext` and `anyascii` can be patched. pyroid's filters and dedup reach
  datatrove pipelines through the [drop-in steps](datatrove.md) instead.
- Patches made with `patch()` stay in the process that made them; child processes need
  [`pyroid run`](#pyroid-run-programs-and-their-child-processes) or `PYROID_PATCH` (forked
  children inherit them).

## Speed

A patched call is a call to pyroid's own function, so it runs at the pyroid module's speed. See
[`pyroid.ftfy`](ftfy.md#speed) and [`pyroid.fasttext`](fasttext.md#speed-and-memory). Code that
calls `fix_text` or `predict` one string at a time gets pyroid's single-call speed, not its
batch speed. To use the batch functions you need to change the calling code.

API: [`pyroid.integrations` patch reference](../reference/integrations.md#patch).
