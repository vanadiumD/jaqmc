#!/usr/bin/env bash
#SBATCH --job-name=diag-env312-sigill
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --output=diag-env312-sigill-%j.out
#SBATCH --error=diag-env312-sigill-%j.err

set -uo pipefail

ENV312="${ENV312:-$HOME/.pyenv/versions/env312}"
PY="$ENV312/bin/python"

READLINE_ROOT="${READLINE_ROOT:-/data/app/foss/2025a/software/libreadline/8.2-GCCcore-14.2.0}"
SQLITE_ROOT="${SQLITE_ROOT:-/data/app/foss/2025a/software/SQLite/3.47.2-GCCcore-14.2.0}"
NCURSES_ROOT="${NCURSES_ROOT:-/data/app/foss/2025a/software/ncurses/6.5-GCCcore-14.2.0}"

overall_rc=0

echo "============================================================"
echo " env312 SIGILL diagnostic on V100 compute node"
echo "============================================================"
echo "Date:                 $(date --iso-8601=seconds 2>/dev/null || date)"
echo "Host:                 $(hostname)"
echo "User:                 ${USER:-unset}"
echo "Job ID:               ${SLURM_JOB_ID:-unset}"
echo "Partition:            ${SLURM_JOB_PARTITION:-unset}"
echo "Node list:            ${SLURM_JOB_NODELIST:-unset}"
echo "CPUs per task:        ${SLURM_CPUS_PER_TASK:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "ENV312:               $ENV312"
echo

module purge 2>/dev/null || true

unset PYTHONHOME
unset PYTHONPATH
unset PYENV_VERSION

export PYTHONNOUSERSITE=1

# Keep the runtime paths matching the env312 build.
export LD_LIBRARY_PATH="\
$READLINE_ROOT/lib:\
$SQLITE_ROOT/lib:\
$NCURSES_ROOT/lib64:\
${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"

if [[ ! -x "$PY" ]]; then
    echo "ERROR: env312 Python not found:"
    echo "  $PY"
    exit 2
fi

echo "===== 1. Compute-node CPU ====="
if command -v lscpu >/dev/null 2>&1; then
    lscpu
else
    echo "lscpu unavailable"
fi

echo
echo "===== 2. Selected CPU flags ====="
grep -m1 -E '^flags|^Features' /proc/cpuinfo 2>/dev/null || true

echo
echo "===== 3. GPU / driver ====="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "nvidia-smi unavailable"
fi

echo
echo "===== 4. Python executable sanity ====="
echo "Python path: $PY"

"$PY" --version
rc=$?
echo "python --version rc=$rc"
if [[ $rc -ne 0 ]]; then
    overall_rc=1
fi

echo
echo "===== 5. Python sysconfig / build flags ====="

set +e
PYTHONFAULTHANDLER=1 "$PY" -u - <<'PY'
import sys
import sysconfig

print("sys.version:", sys.version, flush=True)
print("sys.executable:", sys.executable, flush=True)

for key in (
    "CC",
    "CFLAGS",
    "CFLAGS_NODIST",
    "CONFIG_ARGS",
    "LDFLAGS",
    "LDFLAGS_NODIST",
    "CCSHARED",
    "SOABI",
    "LIBDIR",
):
    try:
        print(f"{key}: {sysconfig.get_config_var(key)}", flush=True)
    except Exception as exc:
        print(f"{key}: ERROR {exc!r}", flush=True)
PY
rc=$?
set -e 2>/dev/null || true

echo "sysconfig rc=$rc"
if [[ $rc -eq 132 ]]; then
    echo "SIGILL: even sysconfig import crashes."
    overall_rc=1
elif [[ $rc -ne 0 ]]; then
    overall_rc=1
fi

echo
echo "===== 6. Isolated module import tests ====="
echo "Each module is tested in a separate Python process."
echo

modules=(
    sys
    os
    math
    json
    ctypes
    _ctypes
    readline
    sqlite3
    _sqlite3
    ssl
    _ssl
    bz2
    _bz2
    lzma
    _lzma
    zlib
)

declare -A MOD_RC

set +e
for mod in "${modules[@]}"; do
    echo "------------------------------------------------------------"
    echo "MODULE: $mod"

    PYTHONFAULTHANDLER=1 "$PY" -u -c "
import sys
print('before import: $mod', flush=True)
import $mod
m = sys.modules['$mod']
print('after import:  $mod', flush=True)
print('file:', getattr(m, '__file__', 'built-in'), flush=True)
"

    rc=$?
    MOD_RC["$mod"]=$rc

    if [[ $rc -eq 0 ]]; then
        echo "RESULT: PASS"
    elif [[ $rc -eq 132 ]]; then
        echo "RESULT: SIGILL (signal 4, shell rc=132)"
        overall_rc=1
    elif [[ $rc -eq 139 ]]; then
        echo "RESULT: SIGSEGV (signal 11, shell rc=139)"
        overall_rc=1
    else
        echo "RESULT: FAILED rc=$rc"
        overall_rc=1
    fi
    echo
done
set -e 2>/dev/null || true

echo
echo "===== 7. Import summary ====="
for mod in "${modules[@]}"; do
    rc="${MOD_RC[$mod]:-999}"
    if [[ "$rc" -eq 0 ]]; then
        status="PASS"
    elif [[ "$rc" -eq 132 ]]; then
        status="SIGILL"
    elif [[ "$rc" -eq 139 ]]; then
        status="SIGSEGV"
    else
        status="FAIL($rc)"
    fi
    printf "%-12s %s\n" "$mod" "$status"
done

echo
echo "===== 8. Locate Python extension modules without importing them ====="

LIBDYN="$ENV312/lib/python3.12/lib-dynload"
if [[ ! -d "$LIBDYN" ]]; then
    # pyenv-virtualenv may refer to the base interpreter's lib-dynload.
    base_prefix=""
    set +e
    base_prefix="$("$PY" -c 'import sys; print(sys.base_prefix)' 2>/dev/null)"
    set -e 2>/dev/null || true
    if [[ -n "$base_prefix" && -d "$base_prefix/lib/python3.12/lib-dynload" ]]; then
        LIBDYN="$base_prefix/lib/python3.12/lib-dynload"
    fi
fi

echo "lib-dynload: $LIBDYN"

declare -A patterns=(
    [_ctypes]="_ctypes*.so"
    [readline]="readline*.so"
    [_sqlite3]="_sqlite3*.so"
    [_ssl]="_ssl*.so"
    [_bz2]="_bz2*.so"
    [_lzma]="_lzma*.so"
)

for mod in _ctypes readline _sqlite3 _ssl _bz2 _lzma; do
    echo
    echo "------------------------------------------------------------"
    echo "EXTENSION: $mod"

    shopt -s nullglob
    matches=( "$LIBDYN"/${patterns[$mod]} )
    shopt -u nullglob

    if ((${#matches[@]} == 0)); then
        echo "NOT FOUND"
        continue
    fi

    for so in "${matches[@]}"; do
        echo "Path: $so"

        echo "--- file ---"
        file "$so" 2>/dev/null || true

        echo "--- ldd ---"
        if command -v ldd >/dev/null 2>&1; then
            ldd "$so" || true
        fi

        echo "--- GNU property / ISA notes ---"
        if command -v readelf >/dev/null 2>&1; then
            readelf -n "$so" 2>/dev/null \
                | grep -A8 -Ei 'GNU_PROPERTY|ISA|x86' \
                || echo "(no explicit ISA note found)"
        fi

        echo "--- objdump architecture ---"
        if command -v objdump >/dev/null 2>&1; then
            objdump -f "$so" 2>/dev/null | head -20 || true
        fi
    done
done

echo
echo "===== 9. External foss libraries ====="

external_libs=(
    "$READLINE_ROOT/lib/libreadline.so.8.2"
    "$SQLITE_ROOT/lib/libsqlite3.so.0.8.6"
    "$NCURSES_ROOT/lib64/libncurses.so.6"
)

for so in "${external_libs[@]}"; do
    echo
    echo "------------------------------------------------------------"
    echo "LIBRARY: $so"

    if [[ ! -e "$so" ]]; then
        echo "MISSING"
        continue
    fi

    file "$so" 2>/dev/null || true

    echo "--- ldd ---"
    ldd "$so" 2>/dev/null || true

    echo "--- GNU property / ISA notes ---"
    if command -v readelf >/dev/null 2>&1; then
        readelf -n "$so" 2>/dev/null \
            | grep -A8 -Ei 'GNU_PROPERTY|ISA|x86' \
            || echo "(no explicit ISA note found)"
    fi
done

echo
echo "===== 10. Direct dynamic-loader tests ====="

set +e

echo "--- Readline library via ctypes.CDLL ---"
PYTHONFAULTHANDLER=1 "$PY" -u - <<PY
import ctypes
print("before CDLL(readline)", flush=True)
lib = ctypes.CDLL("$READLINE_ROOT/lib/libreadline.so.8.2")
print("after CDLL(readline)", flush=True)
PY
rc=$?
echo "readline CDLL rc=$rc"
[[ $rc -eq 0 ]] || overall_rc=1

echo
echo "--- SQLite library via ctypes.CDLL ---"
PYTHONFAULTHANDLER=1 "$PY" -u - <<PY
import ctypes
print("before CDLL(sqlite)", flush=True)
lib = ctypes.CDLL("$SQLITE_ROOT/lib/libsqlite3.so.0.8.6")
print("after CDLL(sqlite)", flush=True)
PY
rc=$?
echo "sqlite CDLL rc=$rc"
[[ $rc -eq 0 ]] || overall_rc=1

echo
echo "--- ncurses library via ctypes.CDLL ---"
PYTHONFAULTHANDLER=1 "$PY" -u - <<PY
import ctypes
print("before CDLL(ncurses)", flush=True)
lib = ctypes.CDLL("$NCURSES_ROOT/lib64/libncurses.so.6")
print("after CDLL(ncurses)", flush=True)
PY
rc=$?
echo "ncurses CDLL rc=$rc"
[[ $rc -eq 0 ]] || overall_rc=1

set -e 2>/dev/null || true

echo
echo "===== 11. Environment ====="
echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"
echo "PYTHONHOME=${PYTHONHOME:-unset}"
echo "PYTHONPATH=${PYTHONPATH:-unset}"
echo "PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-unset}"

echo
echo "============================================================"
if [[ $overall_rc -eq 0 ]]; then
    echo "RESULT: all diagnostic checks passed."
else
    echo "RESULT: one or more checks failed."
    echo
    echo "Most useful section:"
    echo "  ===== 7. Import summary ====="
    echo
    echo "Interpretation examples:"
    echo "  ctypes SIGILL   -> inspect _ctypes / libffi"
    echo "  readline SIGILL -> inspect readline.so -> libreadline -> ncurses"
    echo "  sqlite3 SIGILL  -> inspect _sqlite3.so -> libsqlite3 -> zlib"
    echo "  ssl SIGILL      -> inspect _ssl.so -> OpenSSL"
fi
echo "============================================================"

exit "$overall_rc"
