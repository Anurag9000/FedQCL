#!/usr/bin/env bash
set -uo pipefail
REPO=/home/rl-group4/Projects/FedQCL_Task
PY=/home/rl-group4/.venvs/fedqcl-sync-py312/bin/python
LEDGER=/tmp/fedqcl-attempt-4b555cc4f8587358-a2.jsonl
cd "$REPO" || exit 70
"$PY" /tmp/fedqcl-seed-attempt-dependencies.py "$LEDGER" > /tmp/fedqcl-4b555cc4f8587358-a2-seed.log 2>&1 || { rc=$?; echo "$rc" > /tmp/fedqcl-4b555cc4f8587358-a2.exit; exit "$rc"; }
"$PY" scripts/run_campaign.py --ledger "$LEDGER" --job-id 4b555cc4f8587358 --resume --device cuda --gpu-id 1 --workers 1 > /tmp/fedqcl-4b555cc4f8587358-a2.log 2>&1
rc=$?
echo "$rc" > /tmp/fedqcl-4b555cc4f8587358-a2.exit
exit "$rc"
