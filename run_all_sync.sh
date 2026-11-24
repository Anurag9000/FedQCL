#!/usr/bin/env bash
# Build the committed-source ledger, then run only the explicitly selected scope.
set -euo pipefail
cd "$(dirname "$0")"
python -m unittest discover -s tests -v
if [[ -n "${TINYIMAGENET_ROOT:-}" ]]; then
  python scripts/build_job_ledger.py --tiny-root "$TINYIMAGENET_ROOT"
else
  python scripts/build_job_ledger.py
fi
if [[ $# -eq 0 ]]; then
  python scripts/run_campaign.py --stage pilot --resume
elif [[ "$1" == "--all" ]]; then
  shift
  python scripts/run_campaign.py --resume "$@"
else
  python scripts/run_campaign.py "$@"
fi
