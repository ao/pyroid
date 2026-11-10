#!/bin/sh
# Train instrumented pyroid wheels: for each interpreter, a fresh venv with the matching wheel, then
# bench/pgo_train.py. Each run writes a raw profile into <raw-dir> (LLVM_PROFILE_FILE). POSIX sh, so it
# also runs in the Alpine container release.yml trains the musllinux wheels in.
#
#   sh bench/pgo_train.sh <wheel-dir> <corpus-dir> <raw-dir> 3.12 3.14t 3.15 3.15t    # from the repository root
#
# The wheel for each interpreter is picked by flavour, as release.yml's test job does: cp311-abi3 for
# 3.11-3.14, cp314-cp314t for 3.14t, cp315-abi3.abi3t for 3.15 and 3.15t.
# Knobs: PGO_VENVS [<raw-dir>/../venvs], PGO_KEEP_VENVS=1 (keep each venv after its run),
#        PGO_TRAIN_ARGS (extra pgo_train.py arguments).
# Needs uv. Installs numpy, pyarrow (not on Windows ARM64, where it has no wheels) and sudachidict-core
# (Japanese is trained only with SudachiDict installed).
set -eu
WHEELS=$1
CORPUS=$2
RAW=$3
shift 3
mkdir -p "$RAW"
VENVS=${PGO_VENVS:-$RAW/../venvs}
LLVM_PROFILE_FILE="$RAW/%m_%p.profraw"
export LLVM_PROFILE_FILE
# The training pins, as bench/uv.lock and parity/pyproject.toml pin them.
DEPS="numpy==2.5.4 sudachidict-core==20261015"
ARROW="pyarrow==26.0.1; sys_platform != 'win32' or platform_machine != 'ARM64'"
for py in "$@"; do
    case "$py" in
        3.1[1-4]) flavour=-cp311-abi3- ;;
        3.14t) flavour=-cp314-cp314t- ;;
        3.15 | 3.15t) flavour=-cp315-abi3.abi3t- ;;
        *) echo "pgo_train.sh: no wheel flavour for Python $py" >&2; exit 1 ;;
    esac
    set -- "$WHEELS"/*"$flavour"*.whl
    if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
        echo "pgo_train.sh: expected one *${flavour}*.whl in $WHEELS" >&2
        exit 1
    fi
    wheel=$1
    venv=$VENVS/$py
    uv venv -q --clear --no-project --managed-python --python "$py" "$venv"
    if [ -x "$venv/Scripts/python.exe" ]; then python=$venv/Scripts/python.exe; else python=$venv/bin/python; fi
    # shellcheck disable=SC2086 # DEPS is a list of requirement words
    uv pip install -q --python "$python" $DEPS "$ARROW"
    uv pip install -q --python "$python" --no-index --no-deps "$wheel"
    echo "pgo_train.sh: Python $py with $(basename "$wheel")"
    # shellcheck disable=SC2086 # PGO_TRAIN_ARGS is a list of words
    "$python" bench/pgo_train.py --corpus-dir "$CORPUS" ${PGO_TRAIN_ARGS:-}
    [ "${PGO_KEEP_VENVS:-0}" = 1 ] || rm -rf "$venv"
done
n=$(find "$RAW" -name '*.profraw' | wc -l | tr -d ' ')
echo "pgo_train.sh: $n raw profile(s) in $RAW"
[ "$n" -gt 0 ]
