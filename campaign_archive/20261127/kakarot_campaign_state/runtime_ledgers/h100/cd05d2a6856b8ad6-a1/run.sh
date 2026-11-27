#!/usr/bin/env bash
set -uo pipefail
REPO=/home/rl-group4/Projects/FedQCL_Task
PY=/home/rl-group4/.venvs/fedqcl-sync-py312/bin/python
LEDGER=/tmp/fedqcl-attempts/cd05d2a6856b8ad6-a1/job_ledger.jsonl
STATUS="${LEDGER%.jsonl}.status.json"
SHARED=/tmp/fedqcl-dependencies-65ced6195777.status.json
cd "$REPO" || exit 70
if test -f "$SHARED"; then cp "$SHARED" "$STATUS"; else
  "$PY" /tmp/fedqcl-seed-attempt-dependencies.py "$LEDGER" > /tmp/fedqcl-cd05d2a6856b8ad6-a1-seed.log 2>&1 || { rc=$?; echo "$rc" > /tmp/fedqcl-cd05d2a6856b8ad6-a1.exit; exit "$rc"; }
  cp "$STATUS" "$SHARED" || exit 71
fi
"$PY" scripts/run_campaign.py --ledger "$LEDGER" --job-id cd05d2a6856b8ad6 --resume --device cuda --gpu-id 0 --workers 1 > /tmp/fedqcl-cd05d2a6856b8ad6-a1.log 2>&1
rc=$?
echo "$rc" > /tmp/fedqcl-cd05d2a6856b8ad6-a1.exit
exit "$rc"
