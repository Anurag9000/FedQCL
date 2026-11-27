#!/usr/bin/env bash
set -uo pipefail
LEDGER=/home/anurag-basistha/.local/state/fedqcl-sync-campaign-65ced6195777/runtime_ledgers/kakarot/5c00625c53888688-a1/job_ledger.jsonl
cd /home/anurag-basistha/Projects/Untapped/FedQCL_Task || exit 70
/home/anurag-basistha/.venvs/fedqcl-sync-py312/bin/python scripts/compact_campaign.py --ledger /home/anurag-basistha/.local/state/fedqcl-sync-campaign-65ced6195777/job_ledger.jsonl --output /home/anurag-basistha/Projects/Untapped/FedQCL_Task/research_results/compact-65ced6195777/shortlist.json > /home/anurag-basistha/.local/state/fedqcl-sync-campaign-65ced6195777/runtime_ledgers/kakarot/5c00625c53888688-a1/combined.log 2>&1
rc=$?
cp /home/anurag-basistha/.local/state/fedqcl-sync-campaign-65ced6195777/runtime_ledgers/kakarot/5c00625c53888688-a1/combined.log /tmp/fedqcl-5c00625c53888688-a1.log.copy
mv /tmp/fedqcl-5c00625c53888688-a1.log.copy /tmp/fedqcl-5c00625c53888688-a1.log
printf '%s\n' "$rc" > /tmp/fedqcl-5c00625c53888688-a1.exit
exit "$rc"
