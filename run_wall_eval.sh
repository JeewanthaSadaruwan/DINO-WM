#!/bin/bash
# Wall: 50 evaluations in ONE run (n_evals=50).
# eval seeds are seed*n+1 for n=0..49, so each eval gets a different start/goal.
# (Running n_evals=1 in a loop always uses eval_seed=1 -> the same task every time.)
set -u

cd "$HOME/jeewan_space/dino_wm"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate dino_wm

export DATASET_DIR="$PWD/data"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=disabled
export D4RL_SUPPRESS_IMPORT_ERROR=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p evaluation_logs/wall

test -d data/wall_single/obses || { echo "ERROR: Wall dataset missing"; exit 1; }
test -f checkpoints/outputs/wall_single/checkpoints/model_latest.pth || { echo "ERROR: Wall checkpoint missing"; exit 1; }

echo "===== Wall: 50 evaluations (n_evals=50, seed=99) ====="

python plan.py --config-name plan_wall.yaml \
    model_name=wall_single \
    ckpt_base_path="$PWD/checkpoints" \
    n_evals=50 \
    n_plot_samples=10 \
    seed=99 \
    planner.max_iter=30 \
    2>&1 | tee evaluation_logs/wall/eval_50.log

echo "===== Wall EVALUATION FINISHED (exit ${PIPESTATUS[0]}) ====="
