# Changelog

The changelog is kept in
[`CHANGELOG.md`](https://github.com/ao/pyroid/blob/main/CHANGELOG.md) in the repository, and
release notes are on the [GitHub releases](https://github.com/ao/pyroid/releases) page. From 1.0
on, both are generated from [Conventional Commits](https://www.conventionalcommits.org/) by
release-please.

## 1.0.0 (unreleased)

A rewrite: the 0.7.x API is removed (see [Migrating from 0.7](migrating-from-0.7.md)), and pyroid
becomes a text-cleaning core for LLM datasets: [`pyroid.ftfy`](guides/ftfy.md),
[`pyroid.fasttext`](guides/fasttext.md), [`pyroid.filters`](guides/filters.md), with
[`pyroid.dedup`](guides/dedup.md) and [`pyroid.translit`](guides/translit.md) in preview.
