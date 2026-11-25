# Shared environment for the Adaptive CEM experiment (source, don't execute).
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate dino_wm

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_DIR="$REPO_ROOT/experiments/adaptive_cem"
export REPO_ROOT EXP_DIR

export DATASET_DIR="$REPO_ROOT/data"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=disabled
export D4RL_SUPPRESS_IMPORT_ERROR=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HYDRA_FULL_ERROR=1

CKPT_BASE_PATH="${CKPT_BASE_PATH:-$REPO_ROOT/checkpoints}"
# Safety cap on MPC replanning steps per task (original config: unlimited,
# i.e. loop until success). Applied identically to both methods.
MAX_MPC_STEPS="${MAX_MPC_STEPS:-20}"
# Hard wall-clock limit per run (seconds) as a second safety net.
RUN_TIMEOUT_S="${RUN_TIMEOUT_S:-7200}"
export CKPT_BASE_PATH MAX_MPC_STEPS RUN_TIMEOUT_S
