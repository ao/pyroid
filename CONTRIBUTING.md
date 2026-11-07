# Contributing to pyroid

Thanks for helping. pyroid is a Rust core for cleaning LLM text datasets, exposed to Python through PyO3. Every
module must match its reference implementation exactly (ftfy, fastText, datatrove, ...), so correctness comes
first and speed second.

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Report security problems privately,
as described in [SECURITY.md](SECURITY.md), never in a public issue.

## Before you start

- For anything bigger than a small fix, open an issue first so we can agree on the API and scope.
- If pyroid's output differs from ftfy, fastText or datatrove, open a **Parity mismatch** issue. That's a bug,
  and the issue's input usually becomes a new parity test case.
- The 0.7.x code lives on the frozen [`legacy`](https://github.com/ao/pyroid/tree/legacy) branch. We don't accept
  changes to it.

## Development setup

You need [Rust](https://rustup.rs/) (stable) and [uv](https://docs.astral.sh/uv/). uv installs Python for you,
including the free-threaded builds. Docker isn't needed.

```bash
git clone https://github.com/ao/pyroid && cd pyroid

uv venv --python 3.12                # a venv in .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
uv pip install --group dev          # maturin, pytest, ruff, mypy (pinned in pyproject.toml)
maturin develop --uv                 # compiles the extension (debug build) into .venv
python -c "import pyroid; print(pyroid.__version__)"   # PEP 440: Cargo.toml's 1.0.0-dev is 1.0.0.dev0
```

Re-run `maturin develop --uv` after every Rust change. Add `--release` only when you measure performance: debug
builds are much slower.

To test on a free-threaded interpreter, use a second venv:

```bash
uv venv --python 3.14t .venv-ft
uv pip install --python .venv-ft --group test
VIRTUAL_ENV=.venv-ft maturin develop --uv
.venv-ft/bin/python -c "import sys, pyroid; assert not sys._is_gil_enabled()"   # importing pyroid must not re-enable the GIL
```

## Tests and lint

```bash
cargo test                           # Rust unit tests of the pure-Rust core (the default workspace member)
pytest python/tests                  # Python tests against the extension you built
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
ruff check && ruff format --check
mypy python/pyroid                   # the public API is typed through .pyi stubs
```

CI runs the same checks on Linux, macOS and Windows, on CPython 3.11, 3.14t, 3.15 and 3.15t, against the built
wheel rather than the source tree.

## Pinned test data

The parity suites and benchmarks read pinned corpora (FineWeb slices) and models (fastText `lid.176`) from
`$PYROID_DATA` (default `/data`). `bench/fetch.py` downloads them and checks each file's sha256 against
`bench/manifest.json`. The parity reference outputs are generated, not downloaded: `python -m pyroid_parity.refs all`
(in `parity/`) rebuilds them and checks them against the same manifest.

```bash
export PYROID_DATA=$HOME/.cache/pyroid-data # any directory outside the source tree
cd bench
uv run --frozen python fetch.py              # download and verify the inputs (about 300 MB)
uv run --frozen python fetch.py --verify     # re-check the inputs you already have (no network)
uv run --frozen python fetch.py --verify-all # ... and the generated parity references too
uv run --frozen python fetch.py --only models
```

## Parity suites

`parity/` holds the implementation-agnostic parity suites: ftfy's golden corpus, Hypothesis fuzzing (including
lone surrogates), fastText probabilities within 1e-6, and datatrove filter and dedup decisions. A suite runs
against whatever module you pass with `--impl`, so the same tests check the reference and pyroid.
`parity/README.md` has the details.

```bash
cd parity
uv sync --frozen
uv pip install -e ..                         # build your working tree of pyroid into parity's venv
uv run --frozen pytest tests -m ftfy --impl pyroid.ftfy                         # one suite against pyroid
uv run --frozen pytest tests --impl filters=pyroid.filters --impl dedup=pyroid.dedup
```

A pull request must not remove or loosen a parity case. If a difference from the reference is intentional,
document it and say so in the PR.

## Benchmarks

`bench/` runs equivalence-gated benchmarks: outputs are checked against the reference before anything is
timed. Build pyroid in release mode for this (`uv pip install ..` from `bench/`, or `maturin develop --uv
--release`). `bench/README.md` lists the stages and their options.

For a performance PR, attach before and after numbers from the same machine and say what it is (CPU, cores, OS,
laptop or VM). The README's headline numbers come from dedicated AWS EC2 instances (a c7i.4xlarge and a
c8g.4xlarge), not from laptops or shared VMs, and any other number there says where it was measured.

## Commits and pull requests

We use [Conventional Commits](https://www.conventionalcommits.org/). Pull requests are squash-merged, so the
PR title is the commit that lands, and [release-please](https://github.com/googleapis/release-please) builds
the changelog and the next version number from it.

| Type | Use it for | In the changelog? |
|---|---|---|
| `feat` | new functions or options | yes |
| `fix` | bug fixes, including parity fixes | yes |
| `perf` | speedups without behaviour changes | yes |
| `docs`, `test`, `refactor`, `build`, `ci`, `chore` | everything else | no |

- Add a scope when it helps: `feat(ftfy): ...`, `fix(dedup): ...`.
- Mark breaking changes with `!` (`feat(filters)!: return masks instead of lists`) and add a
  `BREAKING CHANGE:` paragraph to the PR description.
- Don't change version numbers or edit `CHANGELOG.md` by hand. The release PR does both.

Before you open the PR:
- the tests and lint above pass;
- new behaviour has tests, and parity cases where a reference exists;
- the public API has type stubs and docstrings;
- the PR template checklist is filled in, including whether you used AI assistance. AI-assisted PRs are welcome
  if you have reviewed and understood every line.

## Releases

Maintainers release through the `release.yml` workflow. See [docs/RELEASING.md](docs/RELEASING.md).
