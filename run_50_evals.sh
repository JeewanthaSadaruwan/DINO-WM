#!/bin/bash

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate dino_wm

export DATASET_DIR="$HOME/jeewan_space/dino_wm/data"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=disabled
export D4RL_SUPPRESS_IMPORT_ERROR=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HYDRA_FULL_ERROR=1

mkdir -p evaluation_logs

for i in $(seq 1 50); do
    echo "========== Evaluation $i / 50 =========="

    python plan.py \
        --config-name plan_point_maze.yaml \
        model_name=point_maze \
        ckpt_base_path="$HOME/jeewan_space/dino_wm/checkpoints" \
        n_evals=1 \
        n_plot_samples=1 \
        seed="$i" \
        > "evaluation_logs/eval_${i}.log" 2>&1

    if [ $? -eq 0 ]; then
        echo "Evaluation $i completed successfully"
    else
        echo "Evaluation $i FAILED"
    fi
done

echo "All 50 evaluations attempted."
