#!/usr/bin/env bash
# Prepare an Amazon Linux 2023 EC2 instance (or an amazonlinux:2023 container) for bench/ec2/run.sh.
# Runs as root, natively on the instance (no Docker there). Idempotent: every step checks before it installs, so a
# re-run after a failure only does what is missing.
#
#   bash bootstrap.sh              # on the instance, through `ssm send-command` (see ec2.sh bootstrap)
#
# Installs, all under $PYROID_PREFIX [/opt/pyroid] except the dnf packages:
#   dnf: git gcc gcc-c++ tar zstd (the brief), plus util-linux (lscpu, taskset), procps-ng (free) and diffutils, which the AMI
#        already has; the amazonlinux:2023 container image does not.
#   rustup + the $RUST_TOOLCHAIN [stable] toolchain (minimal profile)
#   uv $UV_VERSION [0.13.7, as in the pyroid-dev image]
#   CPython via uv: $PYTHONS [3.12.16 3.13.18 3.15.2 3.15.2t, the pyroid-dev image's patch versions]. 3.15 (GIL) is
#        there only because CI builds the cp315-abi3.abi3t wheel with it (maturin picks abi3t from a 3.15 interpreter).
#   maturin $MATURIN_VERSION [1.16.0, as ci.yml] as a uv tool
# Writes $PYROID_PREFIX/env.sh (sourced by run.sh) and $PYROID_PREFIX/logs/bootstrap.json (versions).
set -euo pipefail

PREFIX=${PYROID_PREFIX:-/opt/pyroid}
UV_VERSION=${UV_VERSION:-0.13.7}
MATURIN_VERSION=${MATURIN_VERSION:-1.16.0}
RUST_TOOLCHAIN=${RUST_TOOLCHAIN:-stable}
PYTHONS=${PYTHONS:-3.12.16 3.13.18 3.15.2 3.15.2t}
DNF_PKGS=${DNF_PKGS:-git gcc gcc-c++ tar zstd util-linux procps-ng diffutils}

mkdir -p "$PREFIX"/{bin,logs}
LOG=$PREFIX/logs/bootstrap.log
exec > >(tee -a "$LOG") 2>&1
log() { echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }
export HOME=${HOME:-/root}
log "start on $(uname -srm), prefix $PREFIX"

cat >"$PREFIX/env.sh" <<EOF
# Written by bench/ec2/bootstrap.sh; sourced by bench/ec2/run.sh.
export PYROID_PREFIX=$PREFIX
export HOME=\${HOME:-/root}
export RUSTUP_HOME=$PREFIX/rustup CARGO_HOME=$PREFIX/cargo
export UV_PYTHON_INSTALL_DIR=$PREFIX/python UV_CACHE_DIR=$PREFIX/uv-cache
export UV_TOOL_DIR=$PREFIX/uv-tools UV_TOOL_BIN_DIR=$PREFIX/bin UV_LINK_MODE=copy
# Only the interpreters installed here; never AL2023's system python3.9.
export UV_PYTHON_PREFERENCE=only-managed
export CARGO_INCREMENTAL=0 CARGO_TERM_COLOR=never
export PATH=$PREFIX/bin:$PREFIX/cargo/bin:\$PATH
EOF
# shellcheck source=/dev/null
source "$PREFIX/env.sh"

missing=()
for p in $DNF_PKGS; do rpm -q "$p" >/dev/null 2>&1 || missing+=("$p"); done
if ((${#missing[@]})); then
    log "dnf install ${missing[*]}"
    dnf install -y -q "${missing[@]}"
else
    log "dnf: all of $DNF_PKGS installed"
fi

if [[ ! -x $CARGO_HOME/bin/rustup ]]; then
    log "rustup ($RUST_TOOLCHAIN, minimal)"
    curl --proto '=https' --tlsv1.2 -sSf --retry 3 https://sh.rustup.rs |
        sh -s -- -y -q --profile minimal --default-toolchain "$RUST_TOOLCHAIN" --no-modify-path
else
    log "rustup present; ensuring $RUST_TOOLCHAIN"
    rustup toolchain install "$RUST_TOOLCHAIN" --profile minimal
    rustup default "$RUST_TOOLCHAIN"
fi

if [[ "$("$PREFIX/bin/uv" --version 2>/dev/null)" != "uv $UV_VERSION"* ]]; then
    log "uv $UV_VERSION"
    curl -LsSf --retry 3 "https://astral.sh/uv/$UV_VERSION/install.sh" |
        env UV_INSTALL_DIR="$PREFIX/bin" UV_NO_MODIFY_PATH=1 sh
else
    log "uv $UV_VERSION present"
fi

log "CPython $PYTHONS"
# shellcheck disable=SC2086
uv python install $PYTHONS
# PYTHONS lists exact versions; check each resolves to what was asked for.
for v in $PYTHONS; do
    p=$(uv python find "$v")
    "$p" -c 'import sys, sysconfig; print(sys.version.split()[0], "free-threaded" if sysconfig.get_config_var("Py_GIL_DISABLED") else "GIL", sys.executable)'
done

if [[ "$(maturin --version 2>/dev/null)" != "maturin $MATURIN_VERSION" ]]; then
    log "maturin $MATURIN_VERSION"
    uv tool install --force --python "$(uv python find 3.13)" "maturin==$MATURIN_VERSION"
else
    log "maturin $MATURIN_VERSION present"
fi

py=$(uv python find 3.13)
"$py" - "$PREFIX/logs/bootstrap.json" <<'EOF'
import json, platform, subprocess, sys, time
def out(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"
doc = {
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "uname": " ".join(platform.uname()),
    "rustc": out("rustc", "-vV"),
    "cargo": out("cargo", "-V"),
    "uv": out("uv", "--version"),
    "maturin": out("maturin", "--version"),
    "gcc": out("gcc", "--version").splitlines()[0],
    "git": out("git", "--version"),
    "glibc": out("ldd", "--version").splitlines()[0],
    "pythons": out("uv", "python", "list", "--only-installed").splitlines(),
}
open(sys.argv[1], "w").write(json.dumps(doc, indent=1) + "\n")
print(json.dumps(doc, indent=1))
EOF
log "done"
