#!/usr/bin/env bash
#SBATCH --job-name=jaqmc-hchain
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=jaqmc-hchain-%j.out
#SBATCH --error=jaqmc-hchain-%j.err

# Universal single-node JaQMC H-chain worker.
# Resource directives above are safe defaults.  Override them at submission time,
# e.g. sbatch --gres=gpu:4 --cpus-per-task=12 --mem=64G this_file ...

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV312="${ENV312:-$HOME/.pyenv/versions/env312}"
PY="${PY:-$ENV312/bin/python}"
JAQMC="${JAQMC:-$ENV312/bin/jaqmc}"
RUNNER="${JAQMC_RUNNER:-$SCRIPT_DIR/run_jaqmc_hchain.py}"
DEFAULT_YML="${JAQMC_YML:-$SCRIPT_DIR/jaqmc_hchain_repro.yml}"

module purge 2>/dev/null || true

# Same isolation principle as the env312 diagnostic, but intentionally DO NOT
# re-add the old site FOSS LD_LIBRARY_PATH.  env312 has its private RPATHs and
# pip jax[cuda12] should not be shadowed by unrelated cluster CUDA/site libs.
unset PYTHONHOME
unset PYTHONPATH
unset PYENV_VERSION
unset LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"

# Keep JAX's normal allocator for realistic performance unless explicitly
# overridden by the submitter, e.g. JAQMC_XLA_PREALLOCATE=false.
if [[ -n "${JAQMC_XLA_PREALLOCATE:-}" ]]; then
  export XLA_PYTHON_CLIENT_PREALLOCATE="$JAQMC_XLA_PREALLOCATE"
else
  unset XLA_PYTHON_CLIENT_PREALLOCATE || true
fi

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Python not executable: $PY" >&2
  exit 2
fi
if [[ ! -x "$JAQMC" ]]; then
  echo "ERROR: JaQMC CLI not executable: $JAQMC" >&2
  exit 2
fi
if [[ ! -f "$RUNNER" ]]; then
  echo "ERROR: runner missing: $RUNNER" >&2
  exit 2
fi

# If caller does not supply --yml, inject the bundled default.
has_yml=0
for arg in "$@"; do
  if [[ "$arg" == "--yml" ]]; then
    has_yml=1
    break
  fi
done
args=("$@")
if [[ $has_yml -eq 0 ]]; then
  args=(--yml "$DEFAULT_YML" "${args[@]}")
fi

# If caller does not supply --python-bin/--jaqmc-bin, inject the pinned env312.
has_py=0
has_jaq=0
for arg in "${args[@]}"; do
  [[ "$arg" == "--python-bin" ]] && has_py=1
  [[ "$arg" == "--jaqmc-bin" ]] && has_jaq=1
done
[[ $has_py -eq 0 ]] && args+=(--python-bin "$PY")
[[ $has_jaq -eq 0 ]] && args+=(--jaqmc-bin "$JAQMC")

echo "============================================================"
echo " JaQMC periodic H-chain job"
echo "============================================================"
echo "Date:                 $(date --iso-8601=seconds 2>/dev/null || date)"
echo "Host:                 $(hostname)"
echo "Job ID:               ${SLURM_JOB_ID:-unset}"
echo "Partition:            ${SLURM_JOB_PARTITION:-unset}"
echo "Node list:            ${SLURM_JOB_NODELIST:-unset}"
echo "CPUs per task:        ${SLURM_CPUS_PER_TASK:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "ENV312:               $ENV312"
echo "Python:               $PY"
echo "JaQMC:                $JAQMC"
echo

"$PY" --version
"$PY" -m pip show jaqmc jax jaxlib 2>/dev/null | sed -n '1,40p' || true

echo
echo_gpu_rc=0
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || echo_gpu_rc=$?
else
  echo "ERROR: nvidia-smi unavailable" >&2
  exit 2
fi
[[ $echo_gpu_rc -eq 0 ]] || true

echo
echo "===== Launch ====="
exec "$PY" -u "$RUNNER" "${args[@]}"
