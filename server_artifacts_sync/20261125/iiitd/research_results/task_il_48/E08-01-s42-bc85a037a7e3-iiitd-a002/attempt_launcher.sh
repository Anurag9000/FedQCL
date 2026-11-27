#!/usr/bin/env bash
set -uo pipefail
repo=/home/anurag/Projects/FedQCL_Task
out="$repo/research_results/task_il_48/E08-01-s42-bc85a037a7e3-iiitd-a002"
python=/home/anurag/.venvs/fedqcl-taskil-py312/bin/python
cd "$repo/system" || exit 90
python3 - "$out/attempt_started.json" <<'PY'
import datetime,json,sys
with open(sys.argv[1]+'.tmp','w') as f: json.dump({'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'iiitd'},f)
import os
os.replace(sys.argv[1]+'.tmp',sys.argv[1])
PY
"$python" task_il_main.py \
  --config "$out/config.json" \
  --schedule "$out/schedule.json" \
  --data "$repo/research_data/task_il_48/5b45a4a6762224aa8c6c" \
  --output "$out" \
  --sync-resume -dev cuda -did 0 \
  >> "$out/attempt_stdout.log" 2>> "$out/attempt_stderr.log"
rc=$?
python3 - "$rc" "$out/exit_status.json" <<'PY'
import datetime,json,os,sys
path=sys.argv[2]; tmp=path+'.tmp'
with open(tmp,'w') as f: json.dump({'exit_code':int(sys.argv[1]),'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat()},f)
os.replace(tmp,path)
PY
exit "$rc"
