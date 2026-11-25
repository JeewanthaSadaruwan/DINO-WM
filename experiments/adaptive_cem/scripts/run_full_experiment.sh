#!/bin/bash
# Full paired experiment: Original vs Adaptive CEM on tasks FIRST..LAST (default 1..50),
# then aggregate metrics, tables and figures. Resumable: finished runs are skipped.
#   usage: scripts/run_full_experiment.sh [first=1] [last=50]
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
FIRST="${1:-1}"; LAST="${2:-50}"
echo "[$(date '+%F %T')] paired runs $FIRST..$LAST (MAX_MPC_STEPS=$MAX_MPC_STEPS)"
"$SCRIPT_DIR/run_paired.sh" "$FIRST" "$LAST"
echo "[$(date '+%F %T')] aggregating"
cd "$REPO_ROOT" && python "$SCRIPT_DIR/aggregate.py"
echo "[$(date '+%F %T')] done -> $EXP_DIR/results/summary.md"
