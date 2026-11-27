#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
PYTHON="${TASK_IL_PYTHON:?set TASK_IL_PYTHON to the verified campaign interpreter}"
SOURCE_LEDGER=experiments/task_il_48/job_ledger.jsonl
LEDGER=experiments/task_il_48/analysis_job_ledger.jsonl
OUT=research_results/task_il_48
mkdir -p "$OUT/verification"
"$PYTHON" "$OUT/coordination/build_taskil_analysis_ledger.py"
"$PYTHON" research_results/task_il_48/coordination/verify_task_il_results.py \
  --ledger "$LEDGER" --output "$OUT/verification/per_run_verification.jsonl"
"$PYTHON" scripts/analyze_task_il.py --ledger "$LEDGER" \
  --output "$OUT/analysis" --require-complete
"$PYTHON" - <<'PY'
import csv,json,pathlib
root=pathlib.Path('research_results/task_il_48')
receipt=json.loads((root/'verification/per_run_verification.summary.json').read_text())
assert receipt['expected']==receipt['verified']==48 and not receipt['failed'],receipt
rows=list(csv.DictReader((root/'analysis/experiment_results.csv').open()))
assert len(rows)==48,len(rows)
missing=list(csv.DictReader((root/'analysis/incomplete_jobs.csv').open()))
assert not missing,missing
print('FINAL_ACCEPTANCE: 48/48 artifacts verified; 48 analysis rows; no incomplete jobs')
PY
