# PointMaze corrected evaluation (50 distinct tasks)

## Why this exists

The earlier PointMaze evaluation (`run_50_evals.sh`, logs in `evaluation_logs/eval_*.log`)
ran `plan.py` 50 times with `n_evals=1` and `seed=1..50`. `plan.py` derives evaluation seeds as

```python
eval_seed = [seed * n + 1 for n in range(n_evals)]
```

With `n_evals=1`, `n` is always 0, so every run used `eval_seed=1`. PointMaze samples its
start/goal with `np.random.RandomState(eval_seed)` (`env/pointmaze/point_maze_wrapper.py`), so all
50 runs evaluated **one task**. Their 50/50 result is 50 repeats of one task, not a 50-task benchmark.

## The fix

`plan.py` accepts an optional `+eval_seed_override=<int>`. When it is given, instance `n` uses
eval seed `override + n`. Without it, the original formula is used unchanged.
Checkpoint, world model, CEM, objective and success criterion are untouched.

Settings used (the original `conf/plan_point_maze.yaml`): MPC + CEM, 300 samples, 30 elites,
horizon 5, 10 CEM steps, `n_evals=1`. The one addition is `planner.max_iter=20`: the original
`max_iter: null` lets an unsolved task loop until it crashes, while a capped task ends with a normal
final evaluation and is counted as a failure.

## Layout

```
logs/seed_XX.log          stdout of each run; logs/progress.log one line per run
results/seed_XX/          Hydra run dir: logs.json, plan_targets.pkl, images, videos, run.json
results/_incomplete/      interrupted or crashed attempts (never counted)
results/per_run.csv       eval seed, status, distances, initial/goal state, output dir
results/summary.json      aggregate + previous-protocol numbers + paper number
results/comparison.md     previous vs corrected table
results/comparison_table.tex  same table for the LaTeX report
figures/task_layout.png, figures/final_state_dist_per_task.png
```

`run.json` is written after every run. A run is counted only if plan.py exited 0, wrote a
`final_eval` entry, and printed `eval_seed: [i]`.

## Commands

```bash
scripts/run_pointmaze_corrected.sh            # seeds 1..50, resumable
python scripts/eval_records.py precheck --env point_maze --first 1 --last 50   # no GPU
python scripts/eval_records.py verify    --exp-dir experiments/pointmaze_corrected
python scripts/eval_records.py aggregate --exp-dir experiments/pointmaze_corrected
```

## Verification done before the full run (2026-11-25)

- Precheck: eval seeds 1..50 give 50 distinct start/goal pairs. Seed 1 is the task the old protocol repeated.
- Smoke test (`experiments/pointmaze_corrected_smoketest/`): seeds 2, 3, 4 loaded the checkpoint,
  planned with CEM, succeeded, and are distinct. Peak GPU memory was about 19.5 GB for one `n_evals=1` run.
- Without the override, plan.py still prints `eval_seed: [1]` and reproduces the old task exactly.
- Resume: completed seeds are skipped, and a partial folder is moved to `_incomplete/` and re-run.

## Result (full run, 2026-11-25, 18:46 to 19:37)

50/50 runs complete, 50 distinct tasks (verify: PASS). **49/50 solved = 98%** (paper: 98%).
MPC iterations: 41 tasks needed 1, 6 needed 2, 2 needed 3, and 1 (seed 45, the longest
start-to-goal distance) hit the 20-iteration cap without success. See `results/comparison.md`.
These results are in `results/main.tex` (Section 3), and the figures are in
`results/figures/fig10`–`fig14b`.
