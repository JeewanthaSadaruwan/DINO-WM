#!/bin/bash
# Paired comparison: run baseline and adaptive on the SAME tasks.
#   usage: scripts/run_paired.sh <first_task> <last_task>
#   e.g.   scripts/run_paired.sh 1 5     # smoke test
#          scripts/run_paired.sh 1 50    # full comparison
# Method order alternates per task (odd: baseline first, even: adaptive first)
# to balance GPU warm-up/thermal effects. Completed runs are skipped, so the
# script can be re-launched after an interruption.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
FIRST="${1:?first task}"; LAST="${2:?last task}"

mkdir -p "$EXP_DIR/logs"
python - "$EXP_DIR/logs/environment.json" <<'PY'
import json, sys, subprocess, torch, platform
info = {"torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(), "precision": "float32 (default, unchanged)",
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())}
json.dump(info, open(sys.argv[1], "w"), indent=2); print(info)
PY

for i in $(seq "$FIRST" "$LAST"); do
  echo "================ task $i ================"
  if (( i % 2 )); then ORDER=(baseline adaptive); else ORDER=(adaptive baseline); fi
  for m in "${ORDER[@]}"; do
    "$SCRIPT_DIR/run_one.sh" "$m" "$i" || echo "!! $m task $i failed (see logs/runs/$m/task_$(printf %02d $i)/stdout.log)"
  done
done
echo "Done. Aggregate with: python experiments/adaptive_cem/scripts/aggregate.py"
