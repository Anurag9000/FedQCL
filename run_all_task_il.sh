#!/usr/bin/env bash
# Task-IL only. Original sync research campaign is never invoked.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")"
ledger=experiments/task_il_48/job_ledger.jsonl
seed_args=()
for seed in ${TASK_IL_SEEDS:-42}; do seed_args+=(--seed "$seed"); done
python -m unittest discover -s tests -p 'test_task_il.py' -v
python scripts/build_task_il_ledger.py --output "$ledger" --split "${TASK_IL_SPLIT:-test}" "${seed_args[@]}"
python scripts/run_campaign.py --ledger "$ledger" --resume "$@"
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" || "$arg" == "--estimate-only" ]]; then exit 0; fi
done
python scripts/analyze_task_il.py --ledger "$ledger"
