Copied from https://github.com/yaml/yaml-test-suite at tag `data-2022-01-17` (commit `6e6c296ae9c9d2d5c4134b4b64d01b29ac19ff6f`, tree `493d388a50385d992cfaa628a88171aa1965c50b`, the same tree as the head of the `data` branch, `6ad3d2c62885d82fc349026c136ef560838fdf3d`). It is the latest data release, generated from the suite's `v2022-01-17` release. GitHub's tarball of the tag, `https://github.com/yaml/yaml-test-suite/archive/refs/tags/data-2022-01-17.tar.gz`, had sha256 `ccae49c1d010d27c20d315969a97752f95b14642fc1e73755db4eb2df4bb0256` when it was fetched (2026-11-14).
License: MIT, Copyright (c) 2016-2020 Ingy döt Net. The data branch carries no licence file, so `LICENSE` is the `License` file of the `v2022-01-17` tag (sha256 `c9562189164244554a69ab3f29d2d93ed9492c165723aaaa5fffc932cdbbfc85`).

Every test directory (`229Q/`, ...) is copied unmodified: 350 directories, 402 test cases (69 of them in numbered subdirectories `00/`, `01/`, ... of a directory that holds several), 1,887 files, 93 KB. The tarball's `name/` and `tags/` folders are 1,500 symlinks; they are replaced by two tab-separated files so that the tree holds no symlinks: `names.tsv` (test name, test id) and `tags.tsv` (tag, test id or `id/NN`).

| file | in | what |
|---|---|---|
| `in.yaml` | 402 | the YAML input |
| `===` | 402 | the test's title |
| `test.event` | 402 | the expected parse events, one per line |
| `in.json` | 282 | the expected value(s) as JSON, one JSON text per document |
| `out.yaml` | 249 | the input re-emitted in a normalised style |
| `emit.yaml` | 55 | the output of an emitter where it differs from `out.yaml` |
| `error` | 94 | present (empty) when the input is invalid YAML |
| `lex.token` | 1 | expected tokens |

The suite describes YAML 1.2. PyYAML implements YAML 1.1, so its results on these inputs are recorded rather than compared with `in.json` (see `bench/yaml/record_oracle.py`).
