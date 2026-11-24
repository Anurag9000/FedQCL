#!/usr/bin/env bash
# Generate the full ledger and run selected stages. Default: documented pilot.
set -euo pipefail
cd "$(dirname "$0")"
python -m unittest discover -s tests -v
python scripts/build_job_ledger.py
if [[ $# -eq 0 ]]; then
  python scripts/run_campaign.py --stage pilot --resume
elif [[ "$1" == "--all" ]]; then
  shift
  python scripts/run_campaign.py --resume "$@"
else
  python scripts/run_campaign.py "$@"
fi
