#!/bin/bash
# Verify that baseline mode (adaptive_cem.enabled=false) reproduces the ORIGINAL
# DINO-WM planner on the real checkpoint: runs the unmodified code from a git
# revision (default HEAD = before this extension) on one task with the same seed,
# then compares every logged metric with logs/runs/baseline/task_XX/logs.json.
#   usage: scripts/verify_baseline_equivalence.sh [task=3] [git_rev=HEAD]
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
TASK="${1:-3}"; REV="${2:-HEAD}"; TASK_ID=$(printf "%02d" "$TASK")
OUT="$EXP_DIR/logs/verification/original_${REV}_task_${TASK_ID}"
SRC="$EXP_DIR/logs/verification/src_${REV}"

"$SCRIPT_DIR/run_one.sh" baseline "$TASK" || exit 1   # skipped if already done

rm -rf "$SRC" "$OUT"; mkdir -p "$SRC" "$OUT"
git -C "$REPO_ROOT" archive "$REV" | tar -x -C "$SRC"
grep -q adaptive "$SRC/planning/cem.py" && echo "WARNING: $REV already contains adaptive CEM code"

mapfile -t COMMON < <(grep -vE '^\s*(#|$)' "$EXP_DIR/configs/common.overrides" | grep -v stats_filename)
cd "$SRC"
timeout "$RUN_TIMEOUT_S" python plan.py --config-name plan_point_maze.yaml "${COMMON[@]}" \
  ckpt_base_path="$CKPT_BASE_PATH" +goal_file_path="$EXP_DIR/configs/tasks/task_${TASK_ID}.pkl" \
  seed="$TASK" planner.max_iter="$MAX_MPC_STEPS" hydra.run.dir="$OUT" > "$OUT/stdout.log" 2>&1 \
  || { echo "original run failed, see $OUT/stdout.log"; exit 1; }

python - "$OUT/logs.json" "$EXP_DIR/logs/runs/baseline/task_${TASK_ID}/logs.json" "$EXP_DIR/logs/verification/baseline_equivalence_task_${TASK_ID}.json" <<'PY'
import json, sys, math
a = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
b = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
max_abs, mism = 0.0, []
if len(a) != len(b):
    mism.append(f"different number of log entries: original={len(a)} baseline={len(b)}")
for i, (x, y) in enumerate(zip(a, b)):
    if set(x) != set(y):
        mism.append(f"entry {i}: different keys"); continue
    for k in x:
        if isinstance(x[k], (int, float)) and isinstance(y[k], (int, float)):
            d = abs(x[k] - y[k]); max_abs = max(max_abs, d)
            if not math.isclose(x[k], y[k], rel_tol=1e-4, abs_tol=1e-6):
                mism.append(f"entry {i} {k}: {x[k]} vs {y[k]}")
        elif x[k] != y[k]:
            mism.append(f"entry {i} {k}: {x[k]} vs {y[k]}")
res = {"original_entries": len(a), "baseline_entries": len(b), "max_abs_diff": max_abs,
       "identical": not mism and max_abs == 0.0, "equivalent_within_tol": not mism, "mismatches": mism[:20]}
json.dump(res, open(sys.argv[3], "w"), indent=2)
print(json.dumps(res, indent=2))
PY
