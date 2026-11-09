#!/usr/bin/env bash
# Profile-guided wheels with the steps release.yml's wheel jobs take, for a Linux box or container.
#
#   bash bench/pgo_build.sh dist/pgo -i python3.12 -i python3.14t           # from the repository root
#   bash bench/pgo_build.sh dist/pgo -i python3.15                           # (a second maturin run)
#
# 1. the training inputs: $PGO_CORPUS if it exists, else bench/pgo_corpus.py writes them there (needs the
#    bench env: `uv run --project bench --frozen --no-default-groups`);
# 2. an instrumented build (rustc -Cprofile-generate) with the maturin arguments given;
# 3. training: bench/pgo_train.sh for each of $PGO_TRAIN_PYTHONS (raw profiles accumulate in $PGO_DIR/raw,
#    so several builds into the same $PGO_DIR share one profile, as the two maturin runs of a release do);
# 4. llvm-profdata merge (rustup's llvm-tools, the LLVM of this rustc);
# 5. the real build with -Cprofile-use and the same maturin arguments.
# Knobs: PGO_DIR [/tmp/pyroid-pgo], PGO_CORPUS [$PGO_DIR/corpus], PGO_TRAIN_PYTHONS [3.12 3.14t 3.15 3.15t],
#        PGO_STEPS [instrument,train,merge,use: any subset, in that order], PGO_TRAIN_ARGS, LLVM_PROFDATA.
set -euo pipefail
OUT=$1; shift
PGO=${PGO_DIR:-/tmp/pyroid-pgo}
CORPUS=${PGO_CORPUS:-$PGO/corpus}
STEPS=,${PGO_STEPS:-instrument,train,merge,use},
cd "$(dirname "$0")/.."
if [[ -z "${LLVM_PROFDATA:-}" ]]; then
    rustup component add llvm-tools >/dev/null
    LLVM_PROFDATA=$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/^host: //p')/bin/llvm-profdata
fi
if [[ $STEPS == *,instrument,* ]]; then
    RUSTFLAGS="-Cprofile-generate=$PGO/raw" maturin build --release --locked --out "$PGO/gen" "$@"
fi
if [[ $STEPS == *,train,* ]]; then
    if [[ ! -f "$CORPUS/MANIFEST.json" ]]; then
        uv run --project bench --frozen --no-default-groups python bench/pgo_corpus.py --out "$CORPUS"
    fi
    # shellcheck disable=SC2086 # a list of interpreters
    sh bench/pgo_train.sh "$PGO/gen" "$CORPUS" "$PGO/raw" ${PGO_TRAIN_PYTHONS:-3.12 3.14t 3.15 3.15t}
fi
if [[ $STEPS == *,merge,* ]]; then
    "$LLVM_PROFDATA" merge -o "$PGO/merged.profdata" "$PGO/raw"
    "$LLVM_PROFDATA" show "$PGO/merged.profdata" | tail -n 4
fi
if [[ $STEPS == *,use,* ]]; then
    RUSTFLAGS="-Cprofile-use=$PGO/merged.profdata" maturin build --release --locked --out "$OUT" "$@"
    echo "PGO wheels in $OUT (profile: $PGO/merged.profdata)"
fi
