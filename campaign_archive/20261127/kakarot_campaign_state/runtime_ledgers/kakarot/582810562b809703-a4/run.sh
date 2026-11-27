#!/usr/bin/env bash
set -uo pipefail
REPO=/home/anurag-basistha/.local/state/fedqcl-sync-campaign-65ced6195777/worktrees/fedqcl-65ced6195777-analysis-recovery
PY=/home/anurag-basistha/.venvs/fedqcl-sync-py312/bin/python
LEDGER=/tmp/fedqcl-attempts/582810562b809703-a4/job_ledger.jsonl
STATUS="${LEDGER%.jsonl}.status.json"
SHARED=/tmp/fedqcl-dependencies-65ced6195777.status.json
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH=/tmp/fedqcl-attempts/582810562b809703-a4${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO" || exit 70
if test -f "$SHARED"; then cp "$SHARED" "$STATUS"; else
  "$PY" /tmp/fedqcl-seed-attempt-dependencies.py "$LEDGER" > /tmp/fedqcl-582810562b809703-a4-seed.log 2>&1 || { rc=$?; echo "$rc" > /tmp/fedqcl-582810562b809703-a4.exit; exit "$rc"; }
  cp "$STATUS" "$SHARED" || exit 71
fi
"$PY" /tmp/fedqcl-attempts/582810562b809703-a4/restore_run_status.py "$STATUS" 582810562b809703 0 || exit 72
"$PY" scripts/run_campaign.py --ledger "$LEDGER" --job-id 582810562b809703 --resume --device cuda --gpu-id 0 --workers 1 > /tmp/fedqcl-582810562b809703-a4.log 2>&1
rc=$?
echo "$rc" > /tmp/fedqcl-582810562b809703-a4.exit
exit "$rc"
