<!--
The PR title becomes the squash-merge commit message and feeds the changelog (release-please),
so write it as a Conventional Commit: `feat(ftfy): ...`, `fix: ...`, `perf(dedup): ...`, `docs: ...`.
Mark breaking changes with `!` (`feat!: ...`) and explain them under "Breaking changes".
-->

## Summary

<!-- What does this change and why? Link the issue: "Fixes #123". -->

## Checklist

- [ ] Tests added or updated. `cargo test` and `pytest python/tests` pass locally against a `maturin develop` build.
- [ ] Parity: the relevant parity suite passes and no parity case was removed or loosened (see CONTRIBUTING.md).
- [ ] Performance-sensitive change: benchmark before and after attached (see CONTRIBUTING.md), or not applicable.
- [ ] Public API change: type stubs (`.pyi`) and docs updated, or not applicable.
- [ ] Lint is clean: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `ruff check`, `ruff format --check`, `mypy`.
- [ ] I used AI assistance for this PR and reviewed every line myself, or I didn't use AI assistance.

## Breaking changes

<!-- Delete this section if there are none. -->
