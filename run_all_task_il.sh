#!/usr/bin/env bash
# Task-IL core by default; full 48-entry matrix is explicit opt-in.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")"
suite="${TASK_IL_SUITE:-core}"
case "$suite" in
  core) ledger=experiments/task_il_core/job_ledger.jsonl ;;
  extended) ledger=experiments/task_il_48/job_ledger.jsonl ;;
  joint) ledger=experiments/task_il_joint/job_ledger.jsonl ;;
  *) echo "Unknown TASK_IL_SUITE: $suite" >&2; exit 2 ;;
esac
seed_args=()
for seed in ${TASK_IL_SEEDS:-42}; do seed_args+=(--seed "$seed"); done
python -m unittest discover -s tests -p 'test_task_il.py' -v
if [[ "$suite" == "joint" ]]; then
  python -m unittest discover -s tests -p 'test_task_il_joint.py' -v
fi
python scripts/build_task_il_ledger.py --suite "$suite" --output "$ledger" --split "${TASK_IL_SPLIT:-test}" "${seed_args[@]}"
python scripts/run_campaign.py --ledger "$ledger" --resume "$@"
if [[ "$suite" == "extended" ]]; then
  for arg in "$@"; do
    if [[ "$arg" == "--dry-run" || "$arg" == "--estimate-only" ]]; then exit 0; fi
  done
  python scripts/analyze_task_il.py --ledger "$ledger"
fi
