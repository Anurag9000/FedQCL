#!/usr/bin/env bash
set -uo pipefail
REPO=/home/anurag-basistha/.local/state/fedqcl-sync-campaign-65ced6195777/worktrees/fedqcl-65ced6195777-analysis-recovery
LEDGER=/home/anurag-basistha/.local/state/fedqcl-sync-campaign-65ced6195777/runtime_ledgers/kakarot/f959e089c77ffb7b-a3/job_ledger.jsonl
cd "$REPO" || exit 70
/home/anurag-basistha/.venvs/fedqcl-sync-py312/bin/python scripts/run_campaign.py --ledger /home/anurag-basistha/.local/state/fedqcl-sync-campaign-65ced6195777/runtime_ledgers/kakarot/f959e089c77ffb7b-a3/job_ledger.jsonl --job-id f959e089c77ffb7b --resume --device cpu --workers 1 > /tmp/fedqcl-f959e089c77ffb7b-a3.log 2>&1
rc=$?
echo "$rc" > /tmp/fedqcl-f959e089c77ffb7b-a3.exit
exit "$rc"
