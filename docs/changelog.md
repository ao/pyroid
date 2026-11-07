# Changelog

The changelog is kept in
[`CHANGELOG.md`](https://github.com/ao/pyroid/blob/main/CHANGELOG.md) in the repository, and
release notes are on the [GitHub releases](https://github.com/ao/pyroid/releases) page. From 1.0
on, both are generated from [Conventional Commits](https://www.conventionalcommits.org/) by
release-please.

## 1.0.0 (unreleased)

A rewrite. The 0.7.x API is removed (see [Migrating from 0.7](migrating-from-0.7.md)), and pyroid
becomes a text-cleaning core for LLM datasets.

- Modules: [`pyroid.ftfy`](guides/ftfy.md), [`pyroid.fasttext`](guides/fasttext.md),
  [`pyroid.filters`](guides/filters.md), [`pyroid.dedup`](guides/dedup.md),
  [`pyroid.translit`](guides/translit.md), [`pyroid.metrics`](guides/metrics.md) and
  [`pyroid.json_repair`](guides/json_repair.md). The filters and dedup follow datatrove's
  behaviour in [3,999 of its 4,343 language codes](guides/languages.md).
- Integrations: the [`pyroid` command line](guides/cli.md), [`pyroid.patch()`](guides/patch.md),
  drop-in [datatrove steps](guides/datatrove.md) and
  [Hugging Face `datasets` helpers](guides/huggingface.md).
