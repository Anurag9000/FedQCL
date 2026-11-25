#!/usr/bin/env bash
# Core is the default; the historical combinatorial suite requires explicit opt-in.
set -euo pipefail
cd "$(dirname "$0")"
python -m unittest discover -s tests -v
suite="${SYNC_SUITE:-core}"
case "$suite" in
  core) ledger=experiments/job_ledger.jsonl ;;
  extended) ledger=experiments/job_ledger_extended.jsonl ;;
  *) echo "Unknown SYNC_SUITE: $suite" >&2; exit 2 ;;
esac
args=(--suite "$suite" --output "$ledger")
if [[ -n "${TINYIMAGENET_ROOT:-}" ]]; then args+=(--tiny-root "$TINYIMAGENET_ROOT"); fi
python scripts/build_job_ledger.py "${args[@]}"
if [[ $# -eq 0 ]]; then
  python scripts/run_campaign.py --ledger "$ledger" --resume
elif [[ "$1" == "--all" ]]; then
  shift
  python scripts/run_campaign.py --ledger "$ledger" --resume "$@"
else
  python scripts/run_campaign.py --ledger "$ledger" "$@"
fi
