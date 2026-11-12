# `pyroid`

The top-level package holds the shared types: the Arrow result type, the base error, the input error, and the fallback warning, plus `patch`/`unpatch` ([guide](../guides/patch.md)). Functions live in submodules (`import pyroid.ftfy`).

::: pyroid
    options:
      show_root_heading: false
      members: [PyroidError, InputError, FallbackWarning, Column, patch, unpatch]
