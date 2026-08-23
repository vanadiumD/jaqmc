#!/usr/bin/env python3
"""Run a JaQMC solid hydrogen-chain job and collect lightweight performance data.

This wrapper deliberately runs JaQMC as a subprocess.  That keeps the wrapper
from creating a JAX GPU context of its own, while still letting it timestamp
console step events and poll nvidia-smi during the run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

TRAIN_START_RE = re.compile(r"\|\s*train\s*\|\s*Start\s+\d+\s+train steps", re.I)
TRAIN_STEP_RE = re.compile(r"\|\s*train\s*\|\s*step=(\d+)\b", re.I)
BURN_COMPLETE_RE = re.compile(r"\|\s*jaqmc\s*\|\s*Burn in .* complete", re.I)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--yml", type=Path, required=True)
    p.add_argument("--save-path", type=Path, required=True)
    p.add_argument("--perf-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--pretrain-steps", type=int, default=None)
    p.add_argument("--train-steps", type=int, default=None)
    p.add_argument("--pretrain-burn-in", type=int, default=None)
    p.add_argument("--train-burn-in", type=int, default=None)
    p.add_argument("--expected-gpus", type=int, default=None)
    p.add_argument("--telemetry-interval", type=float, default=0.25)
    p.add_argument("--jaqmc-bin", type=Path, default=None)
    p.add_argument("--python-bin", type=Path, default=None)
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional JaQMC key=value override; may be repeated.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def resolve_bin(args: argparse.Namespace, name: str) -> str:
    explicit = getattr(args, f"{name}_bin")
    if explicit:
        return str(explicit)
    env312 = Path(os.environ.get("ENV312", str(Path.home() / ".pyenv/versions/env312")))
    return str(env312 / "bin" / ("python" if name == "python" else "jaqmc"))


def probe_jax_devices(python_bin: str) -> dict[str, Any]:
    code = r'''
import json, jax
print(json.dumps({
    "jax_version": jax.__version__,
    "backend": jax.default_backend(),
    "local_device_count": jax.local_device_count(),
    "devices": [str(d) for d in jax.local_devices()],
}))
'''
    proc = subprocess.run(
        [python_bin, "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "JAX device probe failed:\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    # JAX/XLA may print diagnostics around the JSON. Pick the last JSON line.
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"Could not parse JAX probe output: {proc.stdout!r}")


def nvidia_query() -> list[dict[str, Any]]:
    fields = [
        "index",
        "uuid",
        "memory.used",
        "memory.total",
        "utilization.gpu",
        "utilization.memory",
        "power.draw",
    ]
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if proc.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    for raw in proc.stdout.splitlines():
        parts = [x.strip() for x in raw.split(",")]
        if len(parts) != len(fields):
            continue
        try:
            rows.append(
                {
                    "gpu_index": int(parts[0]),
                    "gpu_uuid": parts[1],
                    "memory_used_mib": float(parts[2]),
                    "memory_total_mib": float(parts[3]),
                    "gpu_util_pct": float(parts[4]),
                    "memory_util_pct": float(parts[5]),
                    "power_w": float(parts[6]),
                }
            )
        except ValueError:
            continue
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible not in {"NoDevFiles", "-1"}:
        tokens = {x.strip() for x in visible.split(",") if x.strip()}
        numeric = {int(x) for x in tokens if x.isdigit()}
        uuids = {x for x in tokens if x.startswith("GPU-")}
        if numeric:
            rows = [r for r in rows if int(r["gpu_index"]) in numeric]
        elif uuids:
            rows = [r for r in rows if str(r["gpu_uuid"]) in uuids]
    return rows


def telemetry_worker(
    stop: threading.Event,
    interval: float,
    t0: float,
    output: Path,
    sink: list[dict[str, Any]],
) -> None:
    fields = [
        "t_s",
        "wall_time",
        "gpu_index",
        "gpu_uuid",
        "memory_used_mib",
        "memory_total_mib",
        "gpu_util_pct",
        "memory_util_pct",
        "power_w",
    ]
    with output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        while not stop.is_set():
            now = time.monotonic()
            wall = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            for row in nvidia_query():
                sample: dict[str, Any] = {"t_s": now - t0, "wall_time": wall, **row}
                sink.append(sample)
                w.writerow(sample)
            f.flush()
            stop.wait(interval)


def as_override(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return f"{key}={'true' if value else 'false'}"
    return f"{key}={value}"


def read_last_train_stats(save_path: Path) -> dict[str, str]:
    path = save_path / "train_stats.csv"
    if not path.is_file():
        return {}
    last: dict[str, str] = {}
    with path.open(newline="") as f:
        for last in csv.DictReader(f):
            pass
    return last


def fnum(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    args = parse_args()
    args.save_path = args.save_path.resolve()
    args.perf_dir = args.perf_dir.resolve()
    args.yml = args.yml.resolve()
    args.save_path.mkdir(parents=True, exist_ok=True)
    args.perf_dir.mkdir(parents=True, exist_ok=True)

    python_bin = resolve_bin(args, "python")
    jaqmc_bin = resolve_bin(args, "jaqmc")

    probe = probe_jax_devices(python_bin)
    if probe.get("backend") != "gpu":
        raise RuntimeError(f"Expected JAX GPU backend, got {probe}")
    if args.expected_gpus is not None and probe.get("local_device_count") != args.expected_gpus:
        raise RuntimeError(
            f"Expected {args.expected_gpus} local GPUs, JAX found "
            f"{probe.get('local_device_count')}: {probe.get('devices')}"
        )

    overrides: list[str] = [
        as_override("workflow.save_path", args.save_path),
        as_override("workflow.restore_path", args.save_path),
        "logging.stream=stdout",
    ]
    if args.batch_size is not None:
        overrides.append(as_override("workflow.batch_size", args.batch_size))
    if args.pretrain_steps is not None:
        overrides.append(as_override("pretrain.run.iterations", args.pretrain_steps))
    if args.train_steps is not None:
        overrides.append(as_override("train.run.iterations", args.train_steps))
    if args.pretrain_burn_in is not None:
        overrides.append(as_override("pretrain.run.burn_in", args.pretrain_burn_in))
    if args.train_burn_in is not None:
        overrides.append(as_override("train.run.burn_in", args.train_burn_in))
    overrides.extend(args.override)

    cmd = [jaqmc_bin, "solid", "train", "--yml", str(args.yml)]
    if args.dry_run:
        cmd.append("--dry-run")
    cmd.extend(overrides)

    (args.perf_dir / "command.json").write_text(
        json.dumps({"cmd": cmd, "jax_probe": probe, "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            "SLURM_CPUS_PER_TASK": os.environ.get("SLURM_CPUS_PER_TASK"),
        }}, indent=2) + "\n"
    )

    print("===== JAX device probe =====", flush=True)
    print(json.dumps(probe, indent=2), flush=True)
    print("===== JaQMC command =====", flush=True)
    print(" ".join(cmd), flush=True)

    samples: list[dict[str, Any]] = []
    stop = threading.Event()
    run_t0 = time.monotonic()
    telemetry = threading.Thread(
        target=telemetry_worker,
        args=(stop, args.telemetry_interval, run_t0, args.perf_dir / "gpu_telemetry.csv", samples),
        daemon=True,
    )
    telemetry.start()

    train_start_t: float | None = None
    train_burn_complete_t: float | None = None
    step_times: dict[int, float] = {}
    combined_log = args.perf_dir / "jaqmc_combined.log"

    rc = 1
    try:
        with combined_log.open("w", buffering=1) as logf:
            proc = subprocess.Popen(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                env=os.environ.copy(),
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                now = time.monotonic()
                rel = now - run_t0
                sys.stdout.write(line)
                sys.stdout.flush()
                logf.write(f"[{rel:12.6f}s] {line}")
                if TRAIN_START_RE.search(line):
                    train_start_t = now
                if train_start_t is not None and BURN_COMPLETE_RE.search(line):
                    train_burn_complete_t = now
                m = TRAIN_STEP_RE.search(line)
                if m:
                    step_times[int(m.group(1))] = now
            rc = proc.wait()
    finally:
        stop.set()
        telemetry.join(timeout=max(1.0, args.telemetry_interval * 3))

    run_t1 = time.monotonic()

    step_rows: list[dict[str, Any]] = []
    previous_t: float | None = train_start_t
    for step in sorted(step_times):
        t = step_times[step]
        step_rows.append(
            {
                "step": step,
                "event_time_s": t - run_t0,
                "delta_from_previous_event_s": None if previous_t is None else t - previous_t,
            }
        )
        previous_t = t
    with (args.perf_dir / "train_step_events.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["step", "event_time_s", "delta_from_previous_event_s"],
        )
        w.writeheader()
        w.writerows(step_rows)

    # step0 is intentionally excluded from the steady estimate because it usually
    # absorbs compilation. Deltas step1..N are the useful short-smoke estimate.
    steady_deltas: list[float] = []
    ordered_steps = sorted(step_times)
    for prev, cur in zip(ordered_steps, ordered_steps[1:]):
        if prev >= 0:
            steady_deltas.append(step_times[cur] - step_times[prev])

    gpu_utils = [fnum(s.get("gpu_util_pct")) for s in samples]
    gpu_utils = [x for x in gpu_utils if x is not None]
    gpu_mems = [fnum(s.get("memory_used_mib")) for s in samples]
    gpu_mems = [x for x in gpu_mems if x is not None]
    powers = [fnum(s.get("power_w")) for s in samples]
    powers = [x for x in powers if x is not None]

    last = read_last_train_stats(args.save_path)
    ndev = int(probe.get("local_device_count", 0) or 0)
    global_batch = args.batch_size
    per_gpu_batch = None
    if global_batch is not None and ndev:
        per_gpu_batch = global_batch / ndev

    summary: dict[str, Any] = {
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "host": os.uname().nodename,
        "exit_code": rc,
        "status": "PASS" if rc == 0 else "FAIL",
        "jax_version": probe.get("jax_version"),
        "backend": probe.get("backend"),
        "gpu_count": ndev,
        "devices": ";".join(probe.get("devices", [])),
        "global_batch": global_batch,
        "per_gpu_batch": per_gpu_batch,
        "process_wall_s": run_t1 - run_t0,
        "train_start_s": None if train_start_t is None else train_start_t - run_t0,
        "train_burn_complete_s": None if train_burn_complete_t is None else train_burn_complete_t - run_t0,
        "train_start_to_step0_s": (
            None
            if train_start_t is None or 0 not in step_times
            else step_times[0] - train_start_t
        ),
        "burn_complete_to_step0_s": (
            None
            if train_burn_complete_t is None or 0 not in step_times
            else step_times[0] - train_burn_complete_t
        ),
        "first5_window_s": (
            None
            if train_start_t is None or 4 not in step_times
            else step_times[4] - train_start_t
        ),
        "steady_step_mean_s": statistics.mean(steady_deltas) if steady_deltas else None,
        "steady_step_median_s": statistics.median(steady_deltas) if steady_deltas else None,
        "gpu_util_mean_pct": statistics.mean(gpu_utils) if gpu_utils else None,
        "gpu_util_peak_pct": max(gpu_utils) if gpu_utils else None,
        "gpu_mem_peak_mib": max(gpu_mems) if gpu_mems else None,
        "gpu_power_mean_w": statistics.mean(powers) if powers else None,
        "last_step": last.get("step"),
        "last_total_energy_real": last.get("total_energy_real", last.get("energy")),
        "last_total_energy_real_var": last.get("total_energy_real_var", last.get("variance")),
        "last_pmove": last.get("pmove"),
        "save_path": str(args.save_path),
        "perf_dir": str(args.perf_dir),
    }
    for i in range(1, 5):
        if (i - 1) in step_times and i in step_times:
            summary[f"step{i}_delta_s"] = step_times[i] - step_times[i - 1]
        else:
            summary[f"step{i}_delta_s"] = None

    (args.perf_dir / "perf_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.perf_dir / "perf_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        w.writeheader()
        w.writerow(summary)

    print("===== Performance summary =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
