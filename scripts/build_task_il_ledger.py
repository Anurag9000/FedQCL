#!/usr/bin/env python3
"""Resolve the committed 48-entry portable job ledger for this checkout."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"system"))
from flcore.sync.task_il_schedule import read_entries, resolve, make_schedule, data_plan, digest

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    body=json.dumps(value,sort_keys=True,indent=2)+"\n"
    if path.exists() and path.read_text()!=body:
        raise ValueError("conflicting generated manifest: "+str(path))
    if not path.exists():
        tmp=path.with_suffix(".tmp");tmp.write_text(body);tmp.replace(path)

def build(output,seeds=(42,),split="test"):
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError("seeds must be distinct")
    output=Path(output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    # Refuse tracked uncommitted code: runtime hashes bind actual source.
    changed=subprocess.check_output(["git","diff","--name-only","-z","HEAD"],cwd=ROOT,text=True).split("\0")
    if any(path and not path.endswith(".pyc") for path in changed):
        raise ValueError("commit tracked source changes before generating the source-bound ledger")
    entries=read_entries();jobs=[];registered={}
    metadata=output.parent/"manifests"/commit[:12]
    results=ROOT/"research_results/task_il_48"
    def add(id,stage,kind,config,command,dependencies,artifacts):
        jobs.append(dict(id=id,stage=stage,kind=kind,config=config,command=command,dependencies=dependencies,
            artifacts=[str(p) for p in artifacts],source_commit=commit,schema=3))
    for row in entries:
        for seed in seeds:
            c=resolve(row,seed,split);s=make_schedule(c);plan=data_plan(c,s)
            key=digest([commit,plan])[:20]
            stage=f'experiment_{row["family"]:02d}'
            root=ROOT/"research_data/task_il_48"/key
            if key not in registered:
                planpath=metadata/(key+".data.json");write_json(planpath,plan)
                gen="data-"+key;val="validate-"+key
                common=["--plan",str(planpath),"--output",str(root)]
                add(gen,stage,"generate",dict(plan_id=digest(plan)),[sys.executable,"scripts/task_il_data.py","generate",*common],[],[root/"manifest.json"])
                add(val,stage,"validate",dict(plan_id=digest(plan)),[sys.executable,"scripts/task_il_data.py","validate",*common],[gen],[root/"validation_receipt.json"])
                registered[key]=val
            # Preserve all 48 executable entries, including the separately requested
            # E10 width-5 baseline. E02/E03's mandated reuse is in analysis metadata.
            jid=row["id"]+f"-s{seed}-"+digest([commit,c,row["id"]])[:12]
            cfgpath=metadata/(jid+".config.json");spath=metadata/(jid+".schedule.json")
            write_json(cfgpath,c);write_json(spath,s)
            out=results/jid
            cfg={**c,"campaign":"task_il_48","experiment_id":row["id"],"family":row["family"],
                 "baseline":row["baseline"],"partition_seed":seed,"training_seed":seed,
                 "evaluation_split":split,"rounds":s["rounds"],"tasks":len(s["tasks"]),
                 "k":c["clients"]-c["offline"],"schedule":str(spath),"schedule_sha256":digest(s),
                 "manifest_plan_id":digest(plan),"resolved_config_sha256":digest(c),"data_root":str(root),
                 "selected_draws":sum(map(len,s["selected"])),
                 "local_training_rounds":sum(map(len,s["active"]))}
            command=[sys.executable,"task_il_main.py","--config",str(cfgpath),"--schedule",str(spath),"--data",str(root),"--output",str(out)]
            names=("final_summary.json","status.json","communication_events.jsonl","server_metrics.jsonl","server.pt",
                   "provenance.json","rounds.jsonl","client_metrics.jsonl","schedule.json","config.json")
            add(jid,stage,"train",cfg,command,[registered[key]],[out/name for name in names])
    from run_campaign import dependency_order
    dependency_order(jobs)
    assert sum(j["kind"]=="train" for j in jobs)==48*len(seeds)
    tmp=output.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(j,sort_keys=True)+"\n" for j in jobs));tmp.replace(output)
    return jobs

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=ROOT/"experiments/task_il_48/job_ledger.jsonl")
    p.add_argument("--seed",type=int,action="append")
    p.add_argument("--split",choices=["test","validation"],default="test")
    a=p.parse_args();jobs=build(a.output,a.seed or [42],a.split)
    print(json.dumps(dict(jobs=len(jobs),training_jobs=sum(j["kind"]=="train" for j in jobs),ledger=str(a.output))))
if __name__=="__main__":main()
