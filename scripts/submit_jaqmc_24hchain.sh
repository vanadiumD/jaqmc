#!/usr/bin/env bash
#
# Submit one 4×V100 JaQMC H-chain task.
#
# The login-node process does NOT wait for the Slurm job.
# All outputs are stored under:
#
#   /data/home/vanadium/jaqmc/result/
#
# Example:
#
#   /data/home/vanadium/jaqmc/result/
#   └── hchain_4gpu_20260821_145000/
#       ├── settings.txt
#       ├── job_id.txt
#       ├── sbatch_submit.txt
#       ├── logs/
#       └── gpu4/
#           ├── run/
#           └── perf/
#

set -Eeuo pipefail


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT="${PROJECT_ROOT:-/data/home/vanadium/jaqmc}"
SCRIPT_ROOT="${JAQMC_SCRIPT_DIR:-$PROJECT_ROOT/scripts}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/result}"

JOB_SCRIPT="${JOB_SCRIPT:-$SCRIPT_ROOT/jaqmc_hchain_job.sbatch}"
RUNNER="${RUNNER:-$SCRIPT_ROOT/run_jaqmc_hchain.py}"
YML="${YML:-$SCRIPT_ROOT/jaqmc_hchain_repro.yml}"

ENV312="${ENV312:-/data/home/vanadium/.pyenv/versions/env312}"
PY="${PY:-$ENV312/bin/python}"


# ============================================================
# Slurm / benchmark configuration
# ============================================================

PARTITION="${PARTITION:-v100}"

GPU_COUNT="${GPU_COUNT:-4}"
CPUS_PER_GPU="${CPUS_PER_GPU:-3}"

# This is HOST RAM, not GPU VRAM.
HOST_MEM_PER_GPU_GB="${HOST_MEM_PER_GPU_GB:-16}"

TIME_LIMIT="${TIME_LIMIT:-48:00:00}"

GLOBAL_BATCH="${GLOBAL_BATCH:-2048}"

PRETRAIN_STEPS="${PRETRAIN_STEPS:-40000}"
TRAIN_STEPS="${TRAIN_STEPS:-40000}"

PRETRAIN_BURN_IN="${PRETRAIN_BURN_IN:-10}"
TRAIN_BURN_IN="${TRAIN_BURN_IN:-10}"

TELEMETRY_INTERVAL="${TELEMETRY_INTERVAL:-0.25}"


# ============================================================
# Output directory
# ============================================================

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

ROOT="${ROOT:-$RESULT_ROOT/hchain_${GPU_COUNT}gpu_${TIMESTAMP}}"

CASE_DIR="$ROOT/gpu${GPU_COUNT}"
RUN_DIR="$CASE_DIR/run"
PERF_DIR="$CASE_DIR/perf"
LOG_DIR="$ROOT/logs"

mkdir -p \
    "$RESULT_ROOT" \
    "$ROOT" \
    "$CASE_DIR" \
    "$RUN_DIR" \
    "$PERF_DIR" \
    "$LOG_DIR"


# ============================================================
# Sanity checks
# ============================================================

for path in \
    "$JOB_SCRIPT" \
    "$RUNNER" \
    "$YML"
do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: required file missing:"
        echo "  $path" >&2
        exit 2
    fi
done

if [[ ! -x "$PY" ]]; then
    echo "ERROR: env312 Python missing:"
    echo "  $PY" >&2
    exit 2
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch not found." >&2
    echo "Run this script on a Slurm login node." >&2
    exit 2
fi

if ! [[ "$GPU_COUNT" =~ ^[0-9]+$ ]] || (( GPU_COUNT < 1 )); then
    echo "ERROR: invalid GPU_COUNT=$GPU_COUNT" >&2
    exit 2
fi

if (( GLOBAL_BATCH % GPU_COUNT != 0 )); then
    echo "ERROR:" >&2
    echo "GLOBAL_BATCH=$GLOBAL_BATCH is not divisible by GPU_COUNT=$GPU_COUNT." >&2
    exit 2
fi


# ============================================================
# Derived resource allocation
# ============================================================

CPUS=$((CPUS_PER_GPU * GPU_COUNT))
HOST_MEM_GB=$((HOST_MEM_PER_GPU_GB * GPU_COUNT))
PER_GPU_BATCH=$((GLOBAL_BATCH / GPU_COUNT))


# ============================================================
# Save benchmark configuration
# ============================================================

cat > "$ROOT/settings.txt" <<EOF
project_root=$PROJECT_ROOT
script_root=$SCRIPT_ROOT
result_root=$RESULT_ROOT

job_script=$JOB_SCRIPT
runner=$RUNNER
yml=$YML
env312=$ENV312
python=$PY

root=$ROOT
run_dir=$RUN_DIR
perf_dir=$PERF_DIR
log_dir=$LOG_DIR

partition=$PARTITION

gpu_count=$GPU_COUNT
cpus_per_gpu=$CPUS_PER_GPU
cpus=$CPUS

host_mem_per_gpu_gb=$HOST_MEM_PER_GPU_GB
host_mem_gb=$HOST_MEM_GB

global_batch=$GLOBAL_BATCH
per_gpu_batch=$PER_GPU_BATCH

pretrain_steps=$PRETRAIN_STEPS
train_steps=$TRAIN_STEPS

pretrain_burn_in=$PRETRAIN_BURN_IN
train_burn_in=$TRAIN_BURN_IN

telemetry_interval=$TELEMETRY_INTERVAL
time_limit=$TIME_LIMIT
EOF


# ============================================================
# Print submission configuration
# ============================================================

echo
echo "============================================================"
echo " JaQMC H-chain asynchronous smoke submission"
echo "============================================================"
echo
echo "Project root : $PROJECT_ROOT"
echo "Result root  : $ROOT"
echo
echo "GPU          : ${GPU_COUNT} × V100"
echo "CPU          : $CPUS"
echo "Host RAM     : ${HOST_MEM_GB}G"
echo
echo "Global batch : $GLOBAL_BATCH"
echo "Batch / GPU  : $PER_GPU_BATCH"
echo
echo "Pretrain     : $PRETRAIN_STEPS steps"
echo "Train        : $TRAIN_STEPS steps"
echo
echo "The submit script will return immediately."
echo "The Slurm job continues independently."
echo


# ============================================================
# Submit
#
# IMPORTANT:
# There is deliberately NO --wait here.
# ============================================================

SBATCH_OUTPUT="$(
    sbatch \
        --parsable \
        --partition="$PARTITION" \
        --nodes=1 \
        --ntasks=1 \
        --gres="gpu:${GPU_COUNT}" \
        --cpus-per-task="$CPUS" \
        --mem="${HOST_MEM_GB}G" \
        --time="$TIME_LIMIT" \
        --job-name="jq-h24-${GPU_COUNT}g" \
        --chdir="$PROJECT_ROOT" \
        --output="$LOG_DIR/gpu${GPU_COUNT}-%j.out" \
        --error="$LOG_DIR/gpu${GPU_COUNT}-%j.err" \
        --export="ALL,JAQMC_SCRIPT_DIR=$SCRIPT_ROOT,ENV312=$ENV312,JAQMC_RUNNER=$RUNNER,JAQMC_YML=$YML" \
        "$JOB_SCRIPT" \
            --yml "$YML" \
            --save-path "$RUN_DIR" \
            --perf-dir "$PERF_DIR" \
            --batch-size "$GLOBAL_BATCH" \
            --pretrain-steps "$PRETRAIN_STEPS" \
            --train-steps "$TRAIN_STEPS" \
            --pretrain-burn-in "$PRETRAIN_BURN_IN" \
            --train-burn-in "$TRAIN_BURN_IN" \
            --expected-gpus "$GPU_COUNT" \
            --telemetry-interval "$TELEMETRY_INTERVAL" \
            --override pretrain.run.timing_warmup_steps=0 \
            --override train.run.timing_warmup_steps=0 \
            --override pretrain.run.save_step_interval=1000000 \
            --override train.run.save_step_interval=1000000
)"


# Slurm --parsable may return:
#
#   3069001
#
# or:
#
#   3069001;clustername
#
JOB_ID="${SBATCH_OUTPUT%%;*}"


# ============================================================
# Record submission metadata
# ============================================================

printf '%s\n' "$SBATCH_OUTPUT" \
    > "$ROOT/sbatch_submit.txt"

printf '%s\n' "$JOB_ID" \
    > "$ROOT/job_id.txt"


cat > "$ROOT/monitor_commands.txt" <<EOF
# Job status
squeue -j $JOB_ID

# Detailed job information
scontrol show job $JOB_ID

# Accounting information after/during execution
sacct -j $JOB_ID -X \\
  --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES,MaxRSS

# Follow stdout
tail -f $LOG_DIR/gpu${GPU_COUNT}-${JOB_ID}.out

# Follow stderr
tail -f $LOG_DIR/gpu${GPU_COUNT}-${JOB_ID}.err

# Expected performance result after completion
cat $PERF_DIR/perf_summary.csv

# GPU telemetry
ls -lh $PERF_DIR
EOF


# ============================================================
# Return immediately
# ============================================================

echo "============================================================"
echo " Submitted successfully"
echo "============================================================"
echo
echo "Job ID:"
echo "  $JOB_ID"
echo
echo "Result directory:"
echo "  $ROOT"
echo
echo "Slurm stdout:"
echo "  $LOG_DIR/gpu${GPU_COUNT}-${JOB_ID}.out"
echo
echo "Slurm stderr:"
echo "  $LOG_DIR/gpu${GPU_COUNT}-${JOB_ID}.err"
echo
echo "Performance output:"
echo "  $PERF_DIR/perf_summary.csv"
echo
echo "Check status with:"
echo "  squeue -j $JOB_ID"
echo
echo "The login-shell submit process is now finished."