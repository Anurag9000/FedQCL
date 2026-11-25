#!/usr/bin/env python3
"""Verify every original CIFAR-100 sample and report each matched stream."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from generate_streams import load_sources
from run_campaign import valid,dependency_order,estimate,select_jobs

def check_stream(root,raw,test):
    manifest=json.loads((root/"manifest.json").read_text())
    receipt=json.loads((root/"validation_receipt.json").read_text())
    verified=0
    for name in manifest["shards"]:
        source=test if "/test/" in name else raw
        with np.load(root/name,allow_pickle=True) as z:d=z["data"].item()
        ids=d["ids"].astype(int)
        if not np.array_equal(np.asarray(source.targets)[ids],d["y"]):raise ValueError("source label mismatch "+name)
        x=np.rint((d["x"]*np.asarray(manifest["normalization"]["std"])[None,:,None,None]+np.asarray(manifest["normalization"]["mean"])[None,:,None,None])*255)
        if not np.array_equal(x.astype(np.uint8).transpose(0,2,3,1),source.data[ids]):raise ValueError("source pixels mismatch "+name)
        verified+=len(ids)
    if verified!=60000:raise ValueError("incomplete real source inventory")
    meta=json.loads((root/"ground_truth/specification.json").read_text())
    with np.load(root/"ground_truth/distributions.npz") as z:
        counts=z["counts"];work=counts.sum(-1);regimes={}
        for phase in meta["schedule"]:
            r=phase["round"];i,j=np.triu_indices(work.shape[1],k=1)
            row=regimes.setdefault(phase["regime"],{"rounds":0,"label_distances":[],"joint_distances":[]})
            row["rounds"]+=1
            row["label_distances"].extend((1-z["spatial_label"][r,i,j]).tolist())
            row["joint_distances"].extend((1-z["spatial_joint"][r,i,j]).tolist())
        for row in regimes.values():
            for key in ("label_distances","joint_distances"):
                v=row.pop(key);row[key]={"min":min(v),"mean":float(np.mean(v)),"max":max(v)}
        return {"status":"passed","dataset":"official CIFAR-100","setting_id":meta.get("setting_id","S14"),"partition_seed":manifest["seed"],
                "verified_source_examples":verified,"private_examples":int(counts.sum()),"public_probe":receipt["public_probe"],
                "validation_examples":receipt["train_and_validation"]-int(counts.sum()),"test_examples":receipt["test"],
                "client_round_samples":{"min":int(work.min()),"max":int(work.max())},"manifest_id":manifest["manifest_id"],
                "ground_truth_hash":receipt["ground_truth_hash"],"max_label_tv":meta["max_label_tv"],"max_joint_tv":meta["max_joint_tv"],"regimes":regimes}

def run(ledger,raw_root,settings=None,seeds=None):
    jobs=[json.loads(l) for l in Path(ledger).read_text().splitlines() if l.strip()]
    dependency_order(jobs)
    selected=select_jobs(jobs,["data"],None,settings,seeds)
    data=[j for j in selected if j["kind"]=="validate"]
    if not data or not all(j["config"].get("controlled_stream") for j in data):raise ValueError("not a controlled ledger")
    if not all(valid(j) for j in selected):raise ValueError("unverified generated data")
    raw,test,_,_,_=load_sources(SimpleNamespace(dataset="cifar100",raw_root=str(raw_root),download=False))
    results=[check_stream(Path(j["artifacts"][0]).parent,raw,test) for j in data]
    return {"status":"passed","verified_streams":len(results),"streams":results,"campaign":estimate(jobs),"source_commit":data[0]["source_commit"]}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--ledger",default="experiments/job_ledger.jsonl")
    p.add_argument("--raw-root",default="data/raw");p.add_argument("--setting",action="append");p.add_argument("--seed",type=int,action="append")
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args();result=run(a.ledger,a.raw_root,a.setting,a.seed);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2,allow_nan=False))
