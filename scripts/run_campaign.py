#!/usr/bin/env python3
"""Execute the canonical ledger, respecting dependencies and artifact integrity."""
import argparse,json,os,shlex,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def valid(job):
    if not all(Path(x).is_file() for x in job["artifacts"]):
        return False
    if job["kind"]=="train":
        out=Path(job["artifacts"][0]).parent
        try:
            summary=json.loads((out/"final_summary.json").read_text())
            status=json.loads((out/"status.json").read_text())
            events=[json.loads(l) for l in (out/"communication_events.jsonl").read_text().splitlines()]
            metrics=[json.loads(l) for l in (out/"server_metrics.jsonl").read_text().splitlines()]
            provenance=json.loads((out/"provenance.json").read_text())
            return (provenance["commit"]==job["source_commit"] and status["status"]=="completed" and summary["rounds"]==job["config"].get("rounds",50) and len(metrics)==(job["config"].get("rounds",50)//job["config"].get("tasks",5))*job["config"].get("tasks",5)*(job["config"].get("tasks",5)+1)//2
                    and sum(x["serialized_bytes"] for x in events)==summary["total_bytes"]
                    and (out/"rounds.jsonl").is_file())
        except (OSError,ValueError,KeyError):
            return False
    return True

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ledger",type=Path,default=ROOT/"experiments"/"job_ledger.jsonl")
    p.add_argument("--stage",action="append")
    p.add_argument("--job-id")
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--retry-failed",action="store_true")
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    p.add_argument("--gpu-id",default="0")
    a=p.parse_args()
    jobs=[json.loads(x) for x in a.ledger.read_text().splitlines()]
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    if any(j.get("source_commit")!=commit for j in jobs):
        raise ValueError("ledger source commit differs; rebuild ledger for this checkout")
    by_id={j["id"]:j for j in jobs}
    wanted=set(j["id"] for j in jobs if (not a.stage or j["stage"] in a.stage) and (not a.job_id or j["id"]==a.job_id))
    def include(j):
        for d in by_id[j]["dependencies"]:
            if d not in wanted:
                wanted.add(d);include(d)
    for j in list(wanted):
        include(j)
    state_path=a.ledger.with_suffix(".status.json")
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    def save():
        tmp=state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state,indent=2,sort_keys=True));os.replace(tmp,state_path)
    logs=a.ledger.parent/"logs";logs.mkdir(parents=True,exist_ok=True)
    failed=False
    for job in jobs:
        key=job["id"]
        if key not in wanted:continue
        command=list(job["command"])
        if job["kind"]=="train":
            command+=["-dev",a.device,"-did",a.gpu_id]
        cwd=ROOT/"system" if job["kind"]=="train" else ROOT
        if a.dry_run:
            print(key,job["stage"],shlex.join(command));continue
        if any(state.get(d,{}).get("status")!="completed" for d in job["dependencies"]):
            state[key]={"status":"blocked"};save();failed=True;continue
        old=state.get(key,{}).get("status")
        if old=="completed" and valid(job):
            continue
        if old in ("failed","running") and not (a.retry_failed or a.resume):
            failed=True;continue
        if job["kind"]=="generate" and valid(job):
            state[key]={"status":"completed"};save();continue
        if job["kind"]=="train" and (a.retry_failed or a.resume):
            out=Path(job["artifacts"][0]).parent
            if (out/"status.json").exists() and not valid(job):
                if a.resume and "--sync-checkpoint" in command and (out/"checkpoint.pt").exists():
                    command.append("--sync-resume")
                else:
                    # Preserve failed evidence before a fresh retry.
                    import time
                    out.rename(out.with_name(out.name+".failed-"+str(time.time_ns())))
        state[key]={"status":"running","command":command};save()
        with (logs/f"{key}.log").open("a") as log:
            rc=subprocess.run(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT).returncode
        ok=rc==0 and valid(job)
        state[key]={"status":"completed" if ok else "failed","exit_code":rc};save()
        failed|=not ok
    return int(failed)

if __name__=="__main__":
    sys.exit(main())
