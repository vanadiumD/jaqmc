JaQMC H-chain Slurm bundle
==========================

Files
-----
1) jaqmc_hchain_repro.yml
   Generic H24/R=2 bohr reproduction template.
   IMPORTANT: vacuum_separation=20.0 is a placeholder until the original
   DeepSolid transverse cell size is recovered.

2) jaqmc_hchain_job.sbatch
   Generic single-node worker. Uses env312 directly and does not require
   `pyenv activate`. Default SBATCH resources are 1 V100 + 3 CPUs; command-line
   `sbatch` resource options override them.

3) run_jaqmc_hchain.py
   Launches `jaqmc solid train`, timestamps train-step console events, samples
   allocated GPUs with nvidia-smi, and writes performance CSV/JSON files.

4) smoke_benchmark_jaqmc_hchain.sh
   Login-node driver. Serially submits 1, 2, then 4 GPU jobs with `sbatch --wait`.
   Default global batch remains 2048 for all GPU counts. Each smoke does
   2 pretrain steps + 5 train steps and reduced burn-in=10 for fast probing.

Quick start
-----------
Copy all four files into one directory on the cluster, then:

  chmod +x jaqmc_hchain_job.sbatch run_jaqmc_hchain.py smoke_benchmark_jaqmc_hchain.sh

Run the serial 1/2/4 GPU benchmark:

  ./smoke_benchmark_jaqmc_hchain.sh

Useful overrides:

  GLOBAL_BATCH=2048 TIME_LIMIT=02:00:00 ./smoke_benchmark_jaqmc_hchain.sh

  GPU_LIST="1 2" ./smoke_benchmark_jaqmc_hchain.sh

  YML=/path/to/custom.yml ./smoke_benchmark_jaqmc_hchain.sh

Production example, 4 V100s on one node:

  sbatch --gres=gpu:4 --cpus-per-task=12 --mem=64G \
    jaqmc_hchain_job.sbatch \
      --yml jaqmc_hchain_repro.yml \
      --save-path "$HOME/jaqmc_runs/h24_r2_4gpu" \
      --perf-dir "$HOME/jaqmc_runs/h24_r2_4gpu/perf" \
      --batch-size 2048 \
      --pretrain-steps 200 \
      --train-steps 40000 \
      --pretrain-burn-in 100 \
      --train-burn-in 100 \
      --expected-gpus 4

Smoke outputs
-------------
<SMOKE_ROOT>/smoke_summary.csv
    Cross-GPU summary with 1-GPU baseline speedup and efficiency.

<SMOKE_ROOT>/gpuN/perf/perf_summary.csv
    One-row summary for that run.

<SMOKE_ROOT>/gpuN/perf/train_step_events.csv
    Step 0..4 event timestamps and inter-step deltas. Step 0 normally includes
    JIT/first-call overhead; step1..4 deltas are used for steady-step estimate.

<SMOKE_ROOT>/gpuN/perf/gpu_telemetry.csv
    nvidia-smi samples for allocated GPUs only.

<SMOKE_ROOT>/gpuN/perf/jaqmc_combined.log
    JaQMC stdout+stderr merged and prefixed with a monotonic elapsed timestamp.

<SMOKE_ROOT>/logs/gpuN-<jobid>.out/.err
    Normal Slurm logs.

Notes
-----
* JaQMC automatically uses all GPUs visible on one node. The benchmark keeps
  workflow.batch_size=2048 fixed and therefore tests strong scaling.
* The worker intentionally unsets LD_LIBRARY_PATH. This differs from the old
  diagnostic script because the repaired env312 has private runtime RPATHs and
  pip-installed jax[cuda12] should not be shadowed by site CUDA/FOSS libraries.
* Single-GPU H24/batch=2048/FP64 may OOM. That is useful benchmark information;
  the serial driver continues with 2 and 4 GPUs and records the failure.
* A 5-step test is a smoke benchmark, not a publication-grade timing study.
  For stable performance numbers, increase SMOKE_TRAIN_STEPS to 20-50.
