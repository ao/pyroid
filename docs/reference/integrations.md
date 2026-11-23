# `pyroid.integrations`

Plug-ins for the tools you already run. `import pyroid` imports none of them, and each needs an
extra: `pyroid[datatrove]`, `pyroid[nemo]`, `pyroid[hf]` or `pyroid[cli]` (for `recipes`). Guides:
[datatrove steps](../guides/datatrove.md), [NeMo Curator stages](../guides/nemo-curator.md), [Hugging Face `datasets`](../guides/huggingface.md),
[the command line](../guides/cli.md), [`pyroid.patch()`](../guides/patch.md).

These modules have no stubs: this page is generated from their source.

## Patching: `pyroid.integrations` {#patch}

`pyroid.patch` and `pyroid.unpatch` are the same functions as `pyroid.integrations.patch` and
`pyroid.integrations.unpatch`.

::: pyroid.integrations
    options:
      show_root_heading: false
      heading_level: 3
      members: [patch, unpatch, patched, patching, run, PatchWarning]

### Constants

The names `patch()` accepts, `pyroid run`'s defaults, and the start-up hook's directory (put it on
`PYTHONPATH`, with the names in `PYROID_PATCH`, to patch Python processes started without
`pyroid run`):

::: pyroid.integrations._patch
    options:
      show_root_heading: false
      show_docstring_description: false
      heading_level: 4
      members: [SUPPORTED, DEFAULT_RUN_NAMES, ENV_VAR, HOOK_DIR]

## `pyroid.integrations.datatrove` {#datatrove}

Each step subclasses datatrove's step of the same name; its constructor takes datatrove's
arguments plus the keyword-only `batch_size` and `threads` (and `model_path` for
`LanguageFilter`, `compat` for `Trafilatura`).

::: pyroid.integrations.datatrove
    options:
      show_root_heading: false
      heading_level: 3
      show_bases: false
      show_if_no_docstring: false
      members:
        - FTFYFormatter
        - LanguageFilter
        - GopherRepetitionFilter
        - GopherQualityFilter
        - C4QualityFilter
        - FineWebQualityFilter
        - MinhashDedupSignature
        - PIIFormatter
        - C4BadWordsFilter
        - URLFilter
        - SymbolLinesFormatter
        - SentenceDedupSignature
        - SentenceFindDedups
        - SentenceDedupFilter
        - SentenceDedupBuildIndex
        - Trafilatura
        - WarcReader

## `pyroid.integrations.nemo_curator` {#nemo_curator}

Each class subclasses NeMo Curator 1.4.0's class of the same name and takes its arguments; the
stages also take the keyword `threads`.

::: pyroid.integrations.nemo_curator
    options:
      show_root_heading: false
      heading_level: 3
      show_bases: false
      show_if_no_docstring: false
      members:
        - ScoreFilter
        - Score
        - Filter
        - Modify
        - FastTextLangId
        - FastTextQualityFilter
        - UnicodeReformatter

## `pyroid.integrations.hf` {#hf}

::: pyroid.integrations.hf
    options:
      show_root_heading: false
      heading_level: 3
      show_if_no_docstring: false
      members:
        - LidModel
        - ANNOTATION_COLUMNS
        - fix_text
        - langid
        - quality_mask
        - keep_mask
        - annotate
        - Stats
        - FineWebResult
        - fineweb

## `pyroid.integrations.recipes` {#recipes}

::: pyroid.integrations.recipes
    options:
      show_root_heading: false
      heading_level: 3
      show_if_no_docstring: false
      members:
        - LID_176_URL
        - LID_176_MIRRORS
        - LID_176_SHA256
        - LID_176_SIZE
        - MINHASH_CONFIG
        - FTFY_FORMATTER
        - lid_model_path
        - Annotation
        - FineWeb
        - FINEWEB_TRAFILATURA
        - FineWebWarc
        - WarcAnnotation
        - TaskExtractor
