#!/bin/bash
# Run ONE PointMaze planning task with ONE method.
#   usage: scripts/run_one.sh <baseline|adaptive> <task_index 1..50> [extra hydra overrides...]
# Output: logs/runs/<method>/task_XX/  (hydra run dir: logs.json, cem_calls.jsonl,
#         .hydra/, videos, plan.log, stdout.log, status.json)
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

METHOD="${1:?method: baseline|adaptive}"; TASK="${2:?task index}"; shift 2
[[ "$METHOD" == baseline || "$METHOD" == adaptive ]] || { echo "bad method $METHOD"; exit 2; }
TASK_ID=$(printf "%02d" "$TASK")
TASK_FILE="$EXP_DIR/configs/tasks/task_${TASK_ID}.pkl"
[[ -f "$TASK_FILE" ]] || { echo "missing $TASK_FILE (run scripts/generate_tasks.py first)"; exit 2; }

RUN_DIR="$EXP_DIR/logs/runs/$METHOD/task_${TASK_ID}"
if [[ -f "$RUN_DIR/status.json" ]] && grep -q '"exit_code": 0' "$RUN_DIR/status.json"; then
  echo "[skip] $METHOD task $TASK_ID already completed"; exit 0
fi
rm -rf "$RUN_DIR"; mkdir -p "$RUN_DIR"

read_overrides() { grep -vE '^\s*(#|$)' "$1"; }
mapfile -t COMMON < <(read_overrides "$EXP_DIR/configs/common.overrides")
mapfile -t METHOD_OV < <(read_overrides "$EXP_DIR/configs/${METHOD}.overrides")

cd "$REPO_ROOT"
START=$(date +%s.%N)
# seed = task index for BOTH methods -> identical torch/numpy RNG state at the
# start of planning, so CEM sampling noise is paired until the methods diverge.
timeout "$RUN_TIMEOUT_S" python plan.py --config-name plan_point_maze.yaml \
  "${COMMON[@]}" "${METHOD_OV[@]}" \
  ckpt_base_path="$CKPT_BASE_PATH" \
  +goal_file_path="$TASK_FILE" \
  seed="$TASK" \
  planner.max_iter="$MAX_MPC_STEPS" \
  hydra.run.dir="$RUN_DIR" \
  "$@" > "$RUN_DIR/stdout.log" 2>&1
CODE=$?
END=$(date +%s.%N)
python - "$RUN_DIR/status.json" "$METHOD" "$TASK" "$CODE" "$START" "$END" "$MAX_MPC_STEPS" <<'PY'
import json, sys
path, method, task, code, start, end, cap = sys.argv[1:]
json.dump({"method": method, "task": int(task), "exit_code": int(code),
           "timed_out": int(code) == 124, "process_wall_time_s": float(end) - float(start),
           "max_mpc_steps": int(cap)}, open(path, "w"), indent=2)
PY
echo "[$METHOD task $TASK_ID] exit=$CODE  $(awk "BEGIN{printf \"%.1f\", $END-$START}")s"
exit $CODE
