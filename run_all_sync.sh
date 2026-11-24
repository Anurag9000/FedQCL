#!/usr/bin/env bash
# Generate the full ledger and run the selected stages. Default: documented pilot.
set -euo pipefail
cd "$(dirname "$0")"
python scripts/build_job_ledger.py
if [[ $# -eq 0 ]]; then
  python scripts/run_campaign.py --stage pilot --resume
else
  python scripts/run_campaign.py "$@"
fi
