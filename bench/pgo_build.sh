#!/usr/bin/env bash
# Profile-guided release wheels: the recipe a release job would follow (Docker only here; never on the host).
#
#   DX_NAME=perf .orchestration/dx.sh 'bash bench/pgo_build.sh /target/wheels/pgo -i python3.12 -i python3.15t'
#
# 1. instrumented build (rustc -Cprofile-generate) for the given interpreters, installed with `uv pip`
#    into $PGO_TRAIN_PYTHONS (space-separated interpreters/venv pythons that have pyarrow + pyroid_parity);
# 2. training: bench/pgo_train.py on headline[10000:20000] (not the benchmarked dev slice) per interpreter;
# 3. llvm-profdata merge (rustup's llvm-tools component: the LLVM matching rustc);
# 4. the real build with -Cprofile-use. Pass the same maturin args as an ordinary release build.
# Knobs: PGO_DIR [/tmp/pyroid-pgo], CARGO_TARGET_DIR (a separate one avoids rebuilding the normal cache),
#        PGO_TRAIN_PYTHONS [python], LLVM_PROFDATA [from `rustup component add llvm-tools`].
set -euo pipefail
OUT=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)
PGO=${PGO_DIR:-/tmp/pyroid-pgo}
rm -rf "$PGO"; mkdir -p "$PGO/raw" "$PGO/gen"
if [[ -z "${LLVM_PROFDATA:-}" ]]; then
    rustup component add llvm-tools >/dev/null
    LLVM_PROFDATA=$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/^host: //p')/bin/llvm-profdata
fi
cd "$HERE/.."
RUSTFLAGS="-Cprofile-generate=$PGO/raw" maturin build --release --locked --out "$PGO/gen" "$@"
for py in ${PGO_TRAIN_PYTHONS:-python}; do
    uv pip install -q --python "$py" --no-index --no-deps --reinstall --find-links "$PGO/gen" pyroid
    (cd "$HERE" && "$py" pgo_train.py)
done
"$LLVM_PROFDATA" merge -o "$PGO/merged.profdata" "$PGO/raw"
RUSTFLAGS="-Cprofile-use=$PGO/merged.profdata" maturin build --release --locked --out "$OUT" "$@"
echo "PGO wheels in $OUT (profile: $PGO/merged.profdata)"
