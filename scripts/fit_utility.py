#!/usr/bin/env python3
import argparse,json,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"system"))
from flcore.sync.utility_calibration import records,fit
from run_campaign import valid,sha_file

def fit_jobs(jobs):
    train=[];cal=[];sources={}
    for j in jobs:
        c=j["config"]
        if c["mode"]!="server" or c["seed"]!=42 or c["name"] not in ("server","server-b40","server-b60") or c["setting_id"] not in ("S04","S14"):raise ValueError("unexpected utility calibration source")
        if not valid(j):raise ValueError("unverified utility development source")
        root=Path(j["artifacts"][0]).parent
        (train if c["setting_id"]=="S14" else cal).extend(records(root))
        sources[j["semantic_id"]]={name:sha_file(root/name) for name in ("config.json","decisions.jsonl","counterfactuals.jsonl")}
    if len(sources)!=6:raise ValueError("six development trajectories required")
    return {**fit(train,cal),"sources":sources,"fit_setting":"S14","calibration_setting":"S04","seed":42}

def main():
    p=argparse.ArgumentParser();p.add_argument("--ledger",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    jobs=[json.loads(x) for x in a.ledger.read_text().splitlines()]
    source=next(x for x in jobs if x["config"].get("scope")=="utility_calibration")
    by={j["id"]:j for j in jobs};model=fit_jobs([by[k] for k in source["dependencies"]])
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix(".tmp");tmp.write_text(json.dumps(model,sort_keys=True,indent=2,allow_nan=False));tmp.replace(a.output)
if __name__=="__main__":main()
