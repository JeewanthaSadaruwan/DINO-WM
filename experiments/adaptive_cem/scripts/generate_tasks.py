"""Generate a fixed set of PointMaze initial/goal tasks shared by both planners.

Each task is written in exactly the format plan.py dumps (`plan_targets.pkl`)
and loads with `goal_source=file`, so baseline and adaptive runs read the
*same* start/goal observations and states.

Why this exists: with `n_evals=1`, plan.py computes `eval_seed = seed*0 + 1 = 1`
for every `seed`, so the original `random_state` path produces the SAME task for
every seed (verified: the previous 50 PointMaze runs all share one start/goal
pair). Here each task gets its own sampling seed, mirroring plan.py's
`random_state` branch (sample_random_init_goal_states -> prepare), and
uniqueness is checked explicitly.

No world model or GPU is used; only the MuJoCo environment (for rendering).

Usage (from repo root, after `source experiments/adaptive_cem/scripts/env.sh`):
    python experiments/adaptive_cem/scripts/generate_tasks.py --n-tasks 50
"""
import argparse
import hashlib
import json
import os
import pickle
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO_ROOT)

import gym  # noqa: E402
import env  # noqa: E402,F401  (registers point_maze)
from env.venv import SubprocVectorEnv  # noqa: E402


def task_hash(state_0, state_g):
    arr = np.concatenate([np.asarray(state_0, np.float64).ravel(), np.asarray(state_g, np.float64).ravel()])
    return hashlib.sha1(np.round(arr, 8).tobytes()).hexdigest()[:16]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-tasks", type=int, default=50)
    p.add_argument("--first-seed", type=int, default=1, help="task i uses sampling seed first_seed + i - 1")
    p.add_argument("--goal-H", type=int, default=5)
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "experiments/adaptive_cem/configs/tasks"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.json")
    if os.path.exists(manifest_path) and not args.overwrite:
        sys.exit(f"{manifest_path} exists; refusing to regenerate tasks (use --overwrite).")

    venv = SubprocVectorEnv([lambda: gym.make("point_maze")])
    seen, tasks = {}, []
    sampling_seed = args.first_seed
    while len(tasks) < args.n_tasks:
        task_seed = [sampling_seed]
        sampling_seed += 1
        init_state, goal_state = venv.sample_random_init_goal_states(task_seed)
        h = task_hash(init_state, goal_state)
        if h in seen:
            print(f"seed {task_seed[0]} duplicates task {seen[h]}; skipping")
            continue
        obs_0, _ = venv.prepare(task_seed, init_state)
        obs_g, _ = venv.prepare(task_seed, goal_state)
        for k in obs_0.keys():  # add time dim, as in plan.py
            obs_0[k] = np.expand_dims(obs_0[k], axis=1)
            obs_g[k] = np.expand_dims(obs_g[k], axis=1)

        idx = len(tasks) + 1
        fname = f"task_{idx:02d}.pkl"
        with open(os.path.join(args.out, fname), "wb") as f:
            pickle.dump(
                {
                    "obs_0": obs_0,
                    "obs_g": obs_g,
                    "state_0": init_state,
                    "state_g": goal_state,
                    "gt_actions": None,
                    "goal_H": args.goal_H,
                },
                f,
            )
        start_goal_xy_dist = float(np.linalg.norm(goal_state[0, :2] - init_state[0, :2]))
        seen[h] = idx
        tasks.append(
            {
                "task": idx,
                "file": fname,
                "sampling_seed": task_seed[0],
                "hash": h,
                "state_0": init_state[0].tolist(),
                "state_g": goal_state[0].tolist(),
                "start_goal_xy_dist": start_goal_xy_dist,
                "trivially_solved_at_start": start_goal_xy_dist < 0.5,  # env success radius
            }
        )
        print(f"task {idx:02d}: seed={task_seed[0]} xy_dist={start_goal_xy_dist:.3f} hash={h}")
    venv.close()

    assert len({t["hash"] for t in tasks}) == len(tasks), "duplicate tasks"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "env": "point_maze",
                "generator": "PointMazeWrapper.sample_random_init_goal_states + prepare (mirrors plan.py random_state)",
                "n_tasks": len(tasks),
                "n_trivially_solved_at_start": sum(t["trivially_solved_at_start"] for t in tasks),
                "tasks": tasks,
            },
            f,
            indent=2,
        )
    print(f"Wrote {len(tasks)} unique tasks to {args.out}")


if __name__ == "__main__":
    main()
