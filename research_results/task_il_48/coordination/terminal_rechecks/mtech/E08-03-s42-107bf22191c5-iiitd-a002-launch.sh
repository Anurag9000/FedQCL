#!/usr/bin/env bash
set -uo pipefail
repo=/home/anurag/Projects/FedQCL_Task
out="$repo/research_results/task_il_48/E08-03-s42-107bf22191c5-iiitd-a002"
python=/home/anurag/.venvs/fedqcl-taskil-py312/bin/python
cd "$repo/system" || exit 90
python3 - "$out/attempt_started.json" <<'PY2'
import datetime,json,os,sys
p=sys.argv[1]
with open(p+'.tmp','w') as f: json.dump({'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'iiitd','attempt_id':'E08-03-s42-107bf22191c5-iiitd-a002'},f)
os.replace(p+'.tmp',p)
PY2
"$python" task_il_main.py --config "$out/config.json" --schedule "$out/schedule.json" --data "$repo/research_data/task_il_48/20f5e8923a8844c64666" --output "$out" --sync-resume -dev cuda -did 0 >> "$out/attempt_stdout.log" 2>> "$out/attempt_stderr.log"
rc=$?
python3 - "$rc" "$out/exit_status.json" <<'PY2'
import datetime,json,os,sys
p=sys.argv[2]
with open(p+'.tmp','w') as f: json.dump({'exit_code':int(sys.argv[1]),'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat()},f)
os.replace(p+'.tmp',p)
PY2
exit "$rc"
