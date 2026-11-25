#!/bin/bash
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

mkdir -p evaluation_logs/pusht

# Verify required files
test -d data/pusht_noise/train || {
    echo "ERROR: PushT training dataset missing"
    exit 1
}

test -f checkpoints/outputs/pusht/checkpoints/model_latest.pth || {
    echo "ERROR: PushT checkpoint missing"
    exit 1
}

echo "===== PushT: Initial Test ====="

python plan.py --config-name plan_pusht.yaml \
    model_name=pusht \
    ckpt_base_path="$PWD/checkpoints" \
    n_evals=1 \
    n_plot_samples=1 \
    seed=99 \
    > evaluation_logs/pusht/test.log 2>&1

if [ $? -ne 0 ]; then
    echo "Initial test FAILED. Check evaluation_logs/pusht/test.log"
    exit 1
fi

echo "Initial test completed. Starting 50 evaluations."

for i in $(seq 1 50); do
    echo "===== PushT Evaluation $i / 50 ====="

    python plan.py --config-name plan_pusht.yaml \
        model_name=pusht \
        ckpt_base_path="$PWD/checkpoints" \
        n_evals=1 \
        n_plot_samples=1 \
        seed="$i" \
        > "evaluation_logs/pusht/eval_${i}.log" 2>&1

    if [ $? -eq 0 ]; then
        echo "Evaluation $i completed"
    else
        echo "Evaluation $i FAILED"
    fi
done

echo "===== PushT EVALUATION FINISHED ====="
