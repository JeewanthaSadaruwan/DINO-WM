# Adaptive CEM Early Stopping — PointMaze experiment

An inference-time extension to DINO-WM's CEM planner. It stops CEM optimisation once the
best predicted goal-distance cost has converged. **No retraining.** The pretrained
PointMaze checkpoint, the datasets and the earlier reproduction results are not modified.

Status: **full 50-task paired comparison completed (2026-11-25).** See `results/summary.md` and
`figures/`; the write-up is in `extension/latex/main.pdf`. Headline: total CEM iterations −51.3% and
planning time −50.9%. Excluding task 45, which the original planner failed after hitting the MPC cap,
the reductions are −20.6% and −20.3%. Success: 49/50 original vs 50/50 adaptive (McNemar p = 1.0).

## Algorithm

For each planning instance, at CEM iteration *k*:

- `C_k` = lowest predicted cost among the 300 candidates (DINO feature MSE to the goal, last frame)
- `C_k^best = min(C_{k-1}^best, C_k)`
- `r_k = (C_{k-1}^best − C_k^best) / max(|C_{k-1}^best|, δ)`, where `r_k ≥ 0`
- `s_k = s_{k-1} + 1` if `r_k < τ`, otherwise `0`
- **Stop the instance** when `k ≥ K_min` and `s_k ≥ P`. It is then frozen at the
  **lowest-cost sequence actually sampled** (not the refitted, never-evaluated CEM mean).

| key (`planner.sub_planner.adaptive_cem.*`) | default | this experiment |
|---|---|---|
| `enabled` | `false` | `true` (adaptive) / `false` (baseline) |
| `min_iterations` (K_min) | 3 | 3 |
| `relative_tolerance` (τ) | 0.01 | 0.01 |
| `patience` (P) | 2 | 2 |
| `delta` (δ) | 1e-8 | 1e-8 |
| `return_best` | `true` | `true` |
| `opt_steps` (K_max, existing) | 10 | 10 |

These are initial, **untuned** hyperparameters.

The rest of the planner behaves as before:

- The original **success-based stop is unchanged.** With `eval_every=1`, the current plan is executed in the
  simulator after every iteration, and CEM ends when *all* instances succeed. That stop returns the
  CEM mean, as before. If CEM reaches 10 iterations, it also returns the CEM mean, as before.
- **Batched tasks:** convergence is tracked **per instance**. A converged instance is skipped in later
  iterations and keeps its best sequence. The loop ends when all instances have converged, all have
  succeeded, or K_max is reached. The objective is unchanged. In this experiment `n_evals=1`, so
  every stop is per task.
- With `enabled=false` the code path is **bitwise identical** to the original. The RNG calls happen in the
  same order and the updates and returned actions are the same; this was verified against `git HEAD`, as described below.

## Files

Modified (in place, backward compatible):

| file | change |
|---|---|
| `planning/cem.py` | Optional `adaptive_cem` stopping; per-instance best-so-far tracking and best-sequence retention; optional per-call instrumentation (`stats_filename`) with GPU-synchronised timing. |
| `conf/plan_point_maze.yaml` | Adds `adaptive_cem` (disabled) and `stats_filename: null` under `planner.sub_planner`. Existing runs are unaffected. |

Not modified: `planning/mpc.py`, `planning/evaluator.py`, `plan.py`, models, checkpoints, datasets,
`plan_outputs/`, `evaluation_logs/`, `results/`.

New (this directory):

```
configs/common.overrides     overrides shared by both methods
configs/baseline.overrides   adaptive_cem.enabled=false
configs/adaptive.overrides   adaptive_cem.enabled=true + hyperparameters
configs/tasks/               task_01..50.pkl (plan_targets format) + manifest.json
scripts/env.sh               conda + MuJoCo env vars, MAX_MPC_STEPS, RUN_TIMEOUT_S
scripts/generate_tasks.py    builds the fixed task set (already run; see below)
scripts/run_one.sh           one method × one task
scripts/run_paired.sh        both methods over a task range (resumable)
scripts/run_full_experiment.sh  run_paired 1..50 + aggregate (the one command for tmux)
scripts/verify_baseline_equivalence.sh  original git-HEAD code vs baseline mode, real checkpoint
scripts/aggregate.py         metrics, reductions, paired statistics, figures, summary.md
logs/runs/<method>/task_XX/  hydra run dirs (logs.json, cem_calls.jsonl, stdout.log, status.json, videos)
results/                     baseline_metrics.csv, adaptive_metrics.csv (one row per navigation task),
                             per_call.csv (one row per CEM planning call), comparison_summary.csv/.json,
                             summary.md (written summary; wording follows the measured numbers)
figures/                     cem_iterations_comparison.png, planning_time_comparison.png,
                             per_task_iterations.png, convergence_example.png (only if a convergence stop occurred)
```

## Fair-comparison design

**Same tasks.** With `n_evals=1`, `plan.py` computes `eval_seed = seed*0 + 1 = 1` for every seed.
So the earlier 50 PointMaze runs (`run_50_evals.sh`) all used **one identical start/goal pair** and
differed only in CEM sampling noise. This was verified: 51 `plan_targets.pkl` files share the same
states. The earlier results are left untouched, but they cannot serve as a 50-task baseline.

For this experiment, `scripts/generate_tasks.py` creates 50 distinct tasks using the same procedure as
plan.py's `random_state` branch (`sample_random_init_goal_states` → `prepare`), with sampling seeds 1–50.

- Uniqueness is checked by hash.
- `task_01` reproduces the earlier task **bit for bit** (states and rendered images), which confirms
  the generator matches plan.py.
- 8 of the 50 tasks (1, 2, 22, 27, 34, 35, 39, 49) start within the 0.5 success radius. They come from
  the original sampling distribution and are kept, not filtered.

Both methods load the same `task_XX.pkl` through `goal_source=file`.

**Same everything else.** Both methods use:

- the same checkpoint (`point_maze`, `latest`), objective, 300/30/5/10 CEM settings, success criterion,
  GPU and fp32 precision
- `seed=<task index>` for both methods, so the torch/numpy RNG state is identical when planning
  starts. The CEM noise is therefore paired until the adaptive planner first stops early. After that,
  the trajectories legitimately diverge.

GPU kernel nondeterminism can still cause tiny differences. Method order alternates per task, and
`logs/environment.json` records the GPU, torch version and git commit.

**MPC cap.** The original config has `planner.max_iter: null`, meaning MPC replans until success. A task
that is never solved would therefore never finish. Both methods use the same cap, `MAX_MPC_STEPS=20`
(set in `env.sh`; override with `MAX_MPC_STEPS=30 scripts/run_paired.sh ...`). A per-run timeout,
`RUN_TIMEOUT_S`, is a second safety net. Tasks that hit the cap count as failures and are reported.

**Planning time** is GPU-synchronised wall time of goal encoding, world-model rollouts, cost and CEM
updates. It **excludes** the in-loop simulator success check, decoding, rendering, video writing and
model loading. Those are logged separately as `eval_time_s`, `wall_time_s` and `process_wall_time_s`.

**CEM calls vs tasks.** `cem_calls.jsonl` has one record per CEM optimisation call, i.e. one MPC step.
It contains:

- iterations, stop reason (`success` / `converged` / `max_iterations`), candidates evaluated
  (300 × iterations)
- per-iteration `cost`, `best_cost` and `rel_improvement`, and timings

`per_task.csv` aggregates calls into full MPC tasks and adds the final success and state distance from `logs.json`.

Adaptive stopping can change the **number of MPC steps**. Weaker plans may need more replanning,
so a task can cost more overall even when each call is cheaper. Per-task totals are therefore the
primary computation metric.

## Metrics

**Terminology.**
- A **navigation task** is one MPC episode.
- Each MPC replanning step runs one **CEM planning call**, and each call runs 1–10 **CEM iterations**.
- Every task total includes all MPC replanning calls. If adaptive stopping caused extra replanning, that cost counts against adaptive.

**Per CEM planning call** (`logs/runs/<method>/task_XX/cem_calls.jsonl`, also `results/per_call.csv`):
- iterations executed and stop reason (`success` / `converged` / `max_iterations`)
- candidates evaluated (300 × iterations)
- per-iteration `cost` (C_k), `best_cost` (C_k^best) and `rel_improvement` (r_k)
- `wm_rollout_calls`, one batched `wm.rollout` per iteration
- `wm_forward_counts`: batched DINO encoder and predictor forward calls and their sample counts, counted with
  forward hooks and paused during the simulator check. Each rollout makes 1 encoder pass (start frame × 300)
  and 5 predictor passes (horizon 5).
- `compute_time_s`: `time.perf_counter` with `torch.cuda.synchronize` around goal encoding, rollouts, cost
  and CEM updates
- `eval_time_s` (in-loop simulator check and rendering) and `wall_time_s`

**Per navigation task** (`results/baseline_metrics.csv`, `results/adaptive_metrics.csv`):
- task file, task hash, seed
- MPC planning calls, total CEM iterations, total candidates, WM forward calls
- planning time, success, final state distance
- stop-reason counts, and whether the task hit the MPC cap

**Reductions** (over all paired tasks, using totals):
- R_I = (I_base − I_adaptive)/I_base × 100 for CEM iterations
- R_T = (T_base − T_adaptive)/T_base × 100 for planning time
- R_C = (N_base − N_adaptive)/N_base × 100 for candidate sequences

Each reduction comes with a 95% bootstrap CI (10,000 task resamples). Success is reported as a
percentage-point change, with discordant pairs and an exact McNemar test.

`summary.md` only says computation was reduced when the CI excludes zero. A planning-time difference
is credited to adaptive stopping only if iteration counts also fell reliably.

**Pairing check.** For every task, `aggregate.py` checks that both methods used the same task file
(SHA-1), the same start/goal targets (hash of the `plan_targets.pkl` each run actually received), and the
same seed and CEM settings. Otherwise the task is excluded and listed.

## Verification

1. **CPU, toy world model, against `git HEAD` `planning/cem.py`:**
   - Baseline mode gives bitwise-identical actions for `n_evals ∈ {1,3,4}`, with and without the success stop.
   - `τ=0` reproduces the baseline.
   - Adaptive mode respects `K_min`/`P`, stops on convergence and returns the recorded best sequence.
   - The success stop still fires first.
   - Unknown keys raise an error.
2. **Real checkpoint, original code** (`scripts/verify_baseline_equivalence.sh`): the unmodified
   `git HEAD` code (extracted with `git archive`) and baseline mode were run on the same task and seed,
   and every logged metric was compared. **Task 5 (4 CEM iterations): bit-identical** (max abs diff
   0.0 on success, final state distance, visual/proprio distance and embedding divergence) — see
   `logs/verification/baseline_equivalence_task_05.json`. Task 3 (1 iteration) was also identical.
3. **5-task paired test on the real checkpoint** (`run_paired.sh 1 5`): all 10 runs completed. Task inputs were verified identical
   per pair. Counters are consistent: per call, encoder passes = 1 + iterations and predictor passes =
   5 × iterations. Each CEM iteration costs about 4.8 s of planning time on the RTX 5090 Laptop GPU.
   **Every planning call (both methods) ended on the success stop after 1–4 iterations in a single MPC
   step, so adaptive stopping never triggered and both methods did identical work** (13 iterations
   each, 5/5 success). Preliminary outputs are in `results/` and `figures/`; the full run overwrites them.

## Commands

From the repository root. The scripts source `scripts/env.sh` themselves.

```bash
cd ~/jeewan_space/dino_wm

# Full 50-task paired experiment + aggregation, in tmux (resumable; tasks 1-5 already done are skipped)
tmux new -s adaptive_cem 'experiments/adaptive_cem/scripts/run_full_experiment.sh 1 50 2>&1 | tee experiments/adaptive_cem/logs/full_run.log'
#   detach: Ctrl-b d      re-attach: tmux attach -t adaptive_cem
#   progress: grep -c "exit=0" experiments/adaptive_cem/logs/full_run.log   (100 runs in total)

# Results (written automatically at the end; re-run any time on whatever has finished)
source experiments/adaptive_cem/scripts/env.sh && python experiments/adaptive_cem/scripts/aggregate.py
cat experiments/adaptive_cem/results/summary.md

# Individual pieces
experiments/adaptive_cem/scripts/run_one.sh baseline 7          # one task, original CEM
experiments/adaptive_cem/scripts/run_one.sh adaptive 7          # same task, adaptive CEM
experiments/adaptive_cem/scripts/run_paired.sh 1 5              # small paired test
experiments/adaptive_cem/scripts/verify_baseline_equivalence.sh 5   # original code vs baseline mode
```

To run any other PointMaze config with adaptive CEM:
`python plan.py --config-name plan_point_maze.yaml ... planner.sub_planner.adaptive_cem.enabled=true planner.sub_planner.stats_filename=cem_calls.jsonl`.

## Limitations

- **Success-check stop.** The original planner (`eval_every=1`) runs the current plan in the
  *ground-truth simulator* after every CEM iteration and stops as soon as the task is solved. This stop is
  kept in both methods, as required. In the 5-task test it ended every planning call after 1–4 iterations,
  before adaptive stopping could fire (it needs at least 3 iterations plus 2 consecutive low-improvement
  iterations). Under this protocol, adaptive stopping can only save computation on calls that are not
  solved within the first few iterations.
- Convergence is measured on the *predicted* (world-model) cost, so world-model errors carry over.
- `τ` is relative to the scale of the feature-space cost, and the hyperparameters are not tuned.
- Final state distance is the full-state norm (position and velocity). Success uses position only.
- Stopping once an objective plateaus is a standard optimisation technique, so no novelty is claimed.
  This experiment only measures its effect inside DINO-WM.
