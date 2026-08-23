#!/usr/bin/env bash
# Submit 1 -> 2 -> 4 GPU JaQMC H-chain smoke jobs serially and aggregate results.
# This script runs on the LOGIN node and calls sbatch --wait for each case.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${JOB_SCRIPT:-$SCRIPT_DIR/jaqmc_hchain_job.sbatch}"
YML="${YML:-$SCRIPT_DIR/jaqmc_hchain_repro.yml}"
ENV312="${ENV312:-$HOME/.pyenv/versions/env312}"
PY="${PY:-$ENV312/bin/python}"

PARTITION="${PARTITION:-v100}"
GPU_LIST_STR="${GPU_LIST:-1 2 4}"
CPUS_PER_GPU="${CPUS_PER_GPU:-3}"
MEM_PER_GPU_GB="${MEM_PER_GPU_GB:-16}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
GLOBAL_BATCH="${GLOBAL_BATCH:-2048}"
SMOKE_PRETRAIN_STEPS="${SMOKE_PRETRAIN_STEPS:-2}"
SMOKE_TRAIN_STEPS="${SMOKE_TRAIN_STEPS:-5}"
SMOKE_PRETRAIN_BURN_IN="${SMOKE_PRETRAIN_BURN_IN:-10}"
SMOKE_TRAIN_BURN_IN="${SMOKE_TRAIN_BURN_IN:-10}"
TELEMETRY_INTERVAL="${TELEMETRY_INTERVAL:-0.25}"
SMOKE_ROOT="${SMOKE_ROOT:-$PWD/jaqmc_hchain_smoke_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$SMOKE_ROOT/logs"

if [[ ! -f "$JOB_SCRIPT" ]]; then
  echo "ERROR: job script missing: $JOB_SCRIPT" >&2
  exit 2
fi
if [[ ! -f "$YML" ]]; then
  echo "ERROR: YAML missing: $YML" >&2
  exit 2
fi
if [[ ! -x "$PY" ]]; then
  echo "ERROR: env312 Python missing: $PY" >&2
  exit 2
fi

printf '%s\n' \
  "smoke_root=$SMOKE_ROOT" \
  "partition=$PARTITION" \
  "gpu_list=$GPU_LIST_STR" \
  "global_batch=$GLOBAL_BATCH" \
  "pretrain_steps=$SMOKE_PRETRAIN_STEPS" \
  "train_steps=$SMOKE_TRAIN_STEPS" \
  "pretrain_burn_in=$SMOKE_PRETRAIN_BURN_IN" \
  "train_burn_in=$SMOKE_TRAIN_BURN_IN" \
  > "$SMOKE_ROOT/smoke_settings.txt"

read -r -a GPU_LIST_ARR <<< "$GPU_LIST_STR"

for g in "${GPU_LIST_ARR[@]}"; do
  if ! [[ "$g" =~ ^[0-9]+$ ]] || (( g < 1 )); then
    echo "Skipping invalid GPU count: $g" >&2
    continue
  fi

  cpus=$((CPUS_PER_GPU * g))
  mem_gb=$((MEM_PER_GPU_GB * g))
  case_dir="$SMOKE_ROOT/gpu${g}"
  run_dir="$case_dir/run"
  perf_dir="$case_dir/perf"
  mkdir -p "$run_dir" "$perf_dir"

  echo
  echo "============================================================"
  echo " Submit ${g} GPU smoke: global_batch=${GLOBAL_BATCH}, per_gpu=$(awk -v b="$GLOBAL_BATCH" -v g="$g" 'BEGIN{printf "%.3f", b/g}')"
  echo "============================================================"

  submit_out="$case_dir/sbatch_submit.txt"
  set +e
  sbatch_output=$(sbatch \
    --parsable \
    --wait \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --gres="gpu:${g}" \
    --cpus-per-task="$cpus" \
    --mem="${mem_gb}G" \
    --time="$TIME_LIMIT" \
    --job-name="jq-h24-${g}g" \
    --output="$SMOKE_ROOT/logs/gpu${g}-%j.out" \
    --error="$SMOKE_ROOT/logs/gpu${g}-%j.err" \
    "$JOB_SCRIPT" \
      --yml "$YML" \
      --save-path "$run_dir" \
      --perf-dir "$perf_dir" \
      --batch-size "$GLOBAL_BATCH" \
      --pretrain-steps "$SMOKE_PRETRAIN_STEPS" \
      --train-steps "$SMOKE_TRAIN_STEPS" \
      --pretrain-burn-in "$SMOKE_PRETRAIN_BURN_IN" \
      --train-burn-in "$SMOKE_TRAIN_BURN_IN" \
      --expected-gpus "$g" \
      --telemetry-interval "$TELEMETRY_INTERVAL" \
      --override pretrain.run.timing_warmup_steps=0 \
      --override train.run.timing_warmup_steps=0 \
      --override pretrain.run.save_step_interval=1000000 \
      --override train.run.save_step_interval=1000000 \
    2>&1)
  submit_rc=$?

  printf '%s\n' "$sbatch_output" | tee "$submit_out"
  # --parsable normally prints jobid[;cluster]. Use first numeric token.
  job_id=$(printf '%s\n' "$sbatch_output" | grep -Eo '^[0-9]+' | head -n 1 || true)
  printf '%s\n' "$job_id" > "$case_dir/job_id.txt"
  printf '%s\n' "$submit_rc" > "$case_dir/sbatch_wait_rc.txt"

  if [[ -n "$job_id" ]] && command -v sacct >/dev/null 2>&1; then
    sacct -j "$job_id" -X --parsable2 --noheader \
      --format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocTRES,ReqMem \
      > "$case_dir/sacct.txt" 2>/dev/null || true
  fi

  if [[ -f "$perf_dir/perf_summary.csv" ]]; then
    echo "GPU ${g}: performance summary produced."
  else
    echo "GPU ${g}: no perf_summary.csv (job may have failed before runner completion)." >&2
    # Create a failure stub so aggregation still contains this GPU count.
    "$PY" - "$g" "$job_id" "$submit_rc" "$perf_dir" "$run_dir" "$GLOBAL_BATCH" <<'PY'
import csv, sys
from pathlib import Path

g, job_id, rc, perf_dir, run_dir, global_batch = sys.argv[1:]
p = Path(perf_dir)
p.mkdir(parents=True, exist_ok=True)
row = {
    "job_id": job_id,
    "host": "",
    "exit_code": rc,
    "status": "FAIL_NO_SUMMARY",
    "jax_version": "",
    "backend": "",
    "gpu_count": g,
    "devices": "",
    "global_batch": global_batch,
    "per_gpu_batch": "",
    "process_wall_s": "",
    "train_start_s": "",
    "train_burn_complete_s": "",
    "train_start_to_step0_s": "",
    "burn_complete_to_step0_s": "",
    "first5_window_s": "",
    "steady_step_mean_s": "",
    "steady_step_median_s": "",
    "gpu_util_mean_pct": "",
    "gpu_util_peak_pct": "",
    "gpu_mem_peak_mib": "",
    "gpu_power_mean_w": "",
    "last_step": "",
    "last_total_energy_real": "",
    "last_total_energy_real_var": "",
    "last_pmove": "",
    "save_path": run_dir,
    "perf_dir": perf_dir,
    "step1_delta_s": "",
    "step2_delta_s": "",
    "step3_delta_s": "",
    "step4_delta_s": "",
}
with (p / "perf_summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row))
    w.writeheader(); w.writerow(row)
PY
  fi

done

# Aggregate all per-case summaries, add speedup and parallel efficiency.
"$PY" - "$SMOKE_ROOT" <<'PY'
import csv
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for p in sorted(root.glob("gpu*/perf/perf_summary.csv")):
    with p.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def gpu_num(row):
    try:
        return int(float(row.get("gpu_count", "")))
    except ValueError:
        return 10**9

rows.sort(key=gpu_num)
baseline = None
for row in rows:
    if gpu_num(row) == 1:
        baseline = num(row.get("steady_step_median_s"))
        break

for row in rows:
    g = gpu_num(row)
    t = num(row.get("steady_step_median_s"))
    if baseline and t and t > 0 and g < 10**9:
        speedup = baseline / t
        row["speedup_vs_1gpu"] = f"{speedup:.6f}"
        row["parallel_efficiency"] = f"{speedup/g:.6f}"
    else:
        row["speedup_vs_1gpu"] = ""
        row["parallel_efficiency"] = ""

if not rows:
    raise SystemExit("No per-GPU summaries found")

preferred = [
    "gpu_count", "job_id", "status", "exit_code", "global_batch", "per_gpu_batch",
    "process_wall_s", "train_start_to_step0_s", "burn_complete_to_step0_s",
    "step1_delta_s", "step2_delta_s", "step3_delta_s", "step4_delta_s",
    "steady_step_mean_s", "steady_step_median_s", "first5_window_s",
    "speedup_vs_1gpu", "parallel_efficiency",
    "gpu_util_mean_pct", "gpu_util_peak_pct", "gpu_mem_peak_mib", "gpu_power_mean_w",
    "last_step", "last_total_energy_real", "last_total_energy_real_var", "last_pmove",
    "host", "jax_version", "backend", "devices", "save_path", "perf_dir",
]
extras = []
seen = set(preferred)
for row in rows:
    for k in row:
        if k not in seen:
            extras.append(k); seen.add(k)
fields = preferred + extras

out = root / "smoke_summary.csv"
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)

print(f"\n===== Final summary: {out} =====")
with out.open() as f:
    print(f.read())
PY

echo "Smoke benchmark complete: $SMOKE_ROOT/smoke_summary.csv"
