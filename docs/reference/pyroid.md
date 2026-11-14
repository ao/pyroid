# `pyroid`

The top-level package holds the shared types: the Arrow result type, the base error, the input error, the fallback warning and the version warning, plus `patch`/`unpatch` ([guide](../guides/patch.md)). Functions live in submodules (`import pyroid.ftfy`).

::: pyroid
    options:
      show_root_heading: false
      members: [PyroidError, InputError, FallbackWarning, VersionMismatchWarning, Column, patch, unpatch]
