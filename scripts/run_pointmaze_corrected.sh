#!/bin/bash
# Corrected PointMaze evaluation: 50 DISTINCT start/goal tasks, run sequentially with n_evals=1.
#
# Why: plan.py sets eval_seed = seed*n + 1, so with n_evals=1 every run used eval_seed=1
# (the same task). Here each run passes +eval_seed_override=<i>, so run i uses eval_seed=i.
#
# Usage:   scripts/run_pointmaze_corrected.sh [FIRST] [LAST]      (default 1 50)
# Resume:  just run it again -- seeds with a complete run.json are skipped; partial
#          run folders are moved to results/_incomplete/ and re-run.
#
# Reusable for Wall (not run yet):
#   CONFIG=plan_wall.yaml MODEL=wall_single ENV_ID=wall EXP_DIR=experiments/wall_corrected \
#     scripts/run_pointmaze_corrected.sh 1 50
set -u

FIRST="${1:-1}"
LAST="${2:-50}"

REPO="$HOME/jeewan_space/dino_wm"
CONFIG="${CONFIG:-plan_point_maze.yaml}"
MODEL="${MODEL:-point_maze}"
ENV_ID="${ENV_ID:-point_maze}"
EXP_DIR="${EXP_DIR:-experiments/pointmaze_corrected}"
# MPC iteration cap. The original config has max_iter: null (no cap), so an unsolved task
# loops until it crashes. A capped run ends with a normal final evaluation (success=False).
MAX_MPC_ITERS="${MAX_MPC_ITERS:-20}"
RUN_TIMEOUT_S="${RUN_TIMEOUT_S:-3600}"   # hard per-run safety timeout

cd "$REPO" || exit 1
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate dino_wm

export DATASET_DIR="$REPO/data"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=disabled
export D4RL_SUPPRESS_IMPORT_ERROR=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_DIR="$EXP_DIR/logs"
RES_DIR="$EXP_DIR/results"
mkdir -p "$LOG_DIR" "$RES_DIR" "$EXP_DIR/figures"

test -f "checkpoints/outputs/$MODEL/checkpoints/model_latest.pth" || { echo "ERROR: checkpoint for $MODEL missing"; exit 1; }

echo "===== $MODEL corrected evaluation: eval seeds $FIRST..$LAST (n_evals=1, max_iter=$MAX_MPC_ITERS) ====="
echo "started $(date -Is)" >> "$LOG_DIR/progress.log"

for i in $(seq "$FIRST" "$LAST"); do
    tag=$(printf "seed_%02d" "$i")
    run_dir="$RES_DIR/$tag"
    out_log="$LOG_DIR/$tag.log"

    if [ -f "$run_dir/run.json" ] && python -c "import json,sys; sys.exit(0 if json.load(open('$run_dir/run.json')).get('complete') else 1)"; then
        echo "[$tag] already complete -- skipping"
        continue
    fi
    if [ -e "$run_dir" ]; then   # interrupted or crashed earlier: keep it, but never count it
        mkdir -p "$RES_DIR/_incomplete"
        mv "$run_dir" "$RES_DIR/_incomplete/${tag}_$(date +%Y%m%d%H%M%S)"
        [ -f "$out_log" ] && mv "$out_log" "$RES_DIR/_incomplete/${tag}_$(date +%Y%m%d%H%M%S).log"
    fi

    echo "[$tag] running (eval_seed=$i)"
    start=$(date +%s.%N)
    timeout "$RUN_TIMEOUT_S" python plan.py --config-name "$CONFIG" \
        model_name="$MODEL" \
        ckpt_base_path="$REPO/checkpoints" \
        n_evals=1 \
        n_plot_samples=1 \
        seed="$i" \
        +eval_seed_override="$i" \
        planner.max_iter="$MAX_MPC_ITERS" \
        hydra.run.dir="$REPO/$run_dir" \
        > "$out_log" 2>&1
    code=$?
    end=$(date +%s.%N)
    timed_out=0; [ "$code" -eq 124 ] && timed_out=1

    mkdir -p "$run_dir"
    python scripts/eval_records.py record --run-dir "$run_dir" --eval-seed "$i" --global-seed "$i" \
        --exit-code "$code" --timed-out "$timed_out" --stdout "$out_log" \
        --max-mpc-iters "$MAX_MPC_ITERS" --wall-time "$(echo "$end - $start" | bc)" \
        | tee -a "$LOG_DIR/progress.log"
    [ "$code" -ne 0 ] && echo "[$tag] RUN FAILED (exit $code) -- see $out_log" | tee -a "$LOG_DIR/progress.log"
done

echo "===== runs finished; verifying task diversity and aggregating ====="
python scripts/eval_records.py verify --exp-dir "$EXP_DIR"
if [ "$ENV_ID" = "point_maze" ]; then
    python scripts/eval_records.py aggregate --exp-dir "$EXP_DIR" --env "$ENV_ID" --expected $((LAST - FIRST + 1))
else
    python scripts/eval_records.py aggregate --exp-dir "$EXP_DIR" --env "$ENV_ID" --expected $((LAST - FIRST + 1)) --previous-logs ""
fi
echo "finished $(date -Is)" >> "$LOG_DIR/progress.log"
