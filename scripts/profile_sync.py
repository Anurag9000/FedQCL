#!/usr/bin/env python3
"""Run a production-configuration prefix in scratch storage, not the ledger."""
import argparse,json,subprocess,time
from pathlib import Path
from run_campaign import ROOT,valid

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ledger",type=Path,default=ROOT/"experiments/job_ledger.jsonl")
    p.add_argument("--experiment",required=True,action="append",help="Experiment ID, or all; serial isolated prefixes")
    p.add_argument("--stop-after-round",type=int,default=13,help="Zero-based round; must be before final round")
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    p.add_argument("--gpu-id",default="0")
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    jobs=[json.loads(l) for l in a.ledger.read_text().splitlines()]
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    if any(j["source_commit"]!=commit for j in jobs):raise ValueError("stale ledger")
    training=[j for j in jobs if j["kind"]=="train"]
    chosen=[j for j in training if "all" in a.experiment or j["config"].get("experiment_id") in a.experiment]
    if not chosen or ("all" not in a.experiment and set(a.experiment)!={j["config"]["experiment_id"] for j in chosen}):raise ValueError("unknown experiment")
    if not all(valid(j) for j in jobs if j["kind"] in ("generate","validate")):raise ValueError("run data stage first")
    if not 0<=a.stop_after_round<min(j["config"]["rounds"]-1 for j in chosen):raise ValueError("profiling must stop before final round")
    out=a.output.resolve()
    if out==ROOT/"research_results" or ROOT/"research_results" in out.parents:raise ValueError("use separate scratch output")
    out.mkdir(parents=True,exist_ok=False)
    results=[]
    for j in chosen:
        ident=j["config"]["experiment_id"];dest=out/ident
        cmd=list(j["command"]);cmd[cmd.index("--sync-output")+1]=str(dest)
        cmd += ["-dev",a.device,"-did",a.gpu_id,"--sync-profile","--sync-stop-after-round",str(a.stop_after_round)]
        begin=time.monotonic()
        with (out/(ident+".log")).open("w") as log:
            proc=subprocess.run(cmd,cwd=ROOT/"system",stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:raise RuntimeError(f"{ident} failed; inspect {out/(ident+'.log')}")
        status=json.loads((dest/"status.json").read_text())
        if status.get("status")!="paused" or status.get("last_committed_round")!=a.stop_after_round:raise RuntimeError("profile did not stop at requested round")
        rows=[json.loads(l) for l in (dest/"resource_profile.jsonl").read_text().splitlines()]
        if [x["round"] for x in rows]!=list(range(a.stop_after_round+1)):raise RuntimeError("incomplete profiling rows")
        peak={k:max(x[k] for x in rows) for k in ("process_peak_rss_bytes","cuda_peak_allocated_bytes","cuda_peak_reserved_bytes") if k in rows[0]}
        results.append({"experiment_id":ident,"wall_seconds":time.monotonic()-begin,"rounds":len(rows),**peak,
                        "ground_truth_hash":json.loads((dest/"config.json").read_text())["ground_truth_hash"],
                        "round_profiles":rows})
        (out/"profile_summary.json").write_text(json.dumps({"source_commit":commit,"device":a.device,"gpu_id":a.gpu_id,"results":results,
            "limitation":"Prefix only; torch allocator peaks exclude non-PyTorch allocations, other processes and unvisited later rounds. Not a safe concurrency limit."},indent=2))
    print(out/"profile_summary.json")

if __name__=="__main__":main()
