#!/usr/bin/env bash
set -uo pipefail
REPO=/home/anurag24092/Projects/FedQCL_Task
PY=/home/anurag24092/.venvs/fedqcl-sync-py312/bin/python
LEDGER=/tmp/fedqcl-attempts/2976287df5f52b5c-a1/job_ledger.jsonl
STATUS="${LEDGER%.jsonl}.status.json"
SHARED=/tmp/fedqcl-dependencies-65ced6195777.status.json
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH=/tmp/fedqcl-attempts/2976287df5f52b5c-a1${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO" || exit 70
if test -f "$SHARED"; then cp "$SHARED" "$STATUS"; else
  "$PY" /tmp/fedqcl-seed-attempt-dependencies.py "$LEDGER" > /tmp/fedqcl-2976287df5f52b5c-a1-seed.log 2>&1 || { rc=$?; echo "$rc" > /tmp/fedqcl-2976287df5f52b5c-a1.exit; exit "$rc"; }
  cp "$STATUS" "$SHARED" || exit 71
fi
"$PY" scripts/run_campaign.py --ledger "$LEDGER" --job-id 2976287df5f52b5c --resume --device cuda --gpu-id 0 --workers 1 > /tmp/fedqcl-2976287df5f52b5c-a1.log 2>&1
rc=$?
echo "$rc" > /tmp/fedqcl-2976287df5f52b5c-a1.exit
exit "$rc"
