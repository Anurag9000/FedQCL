#!/usr/bin/env python3
"""Canonical staged campaign ledger. Conditional extensions are not fake jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import itertools
ROOT=Path(__file__).resolve().parents[1]

def build(output):
    jobs=[]; seen=set()
    def add(stage,kind,config,command,deps=(),artifacts=()):
        key=hashlib.sha256(json.dumps([kind,config],sort_keys=True).encode()).hexdigest()[:16]
        if key in seen:
            return key
        seen.add(key)
        jobs.append({"id":key,"stage":stage,"kind":kind,"config":config,"command":command,
                     "dependencies":list(deps),"artifacts":list(artifacts)})
        return key
    data={}
    for n in [5]:
        for regime,seed in itertools.product(("near","separated","changing","clustered","dirichlet","conditional","drift"),(42,43,44)):
            folder=ROOT/"research_data"/f"{regime}-{seed}-n{n}"
            config={"regime":regime,"partition_seed":seed,"clients":n}
            stage="pilot" if regime in ("near","separated","changing") else "ablation"
            d=add(stage,"generate",config,[sys.executable,"scripts/generate_streams.py","--regime",regime,
                 "--seed",str(seed),"--clients",str(n),"--output",str(folder),"--download"],
                 artifacts=[str(folder/"manifest.json")])
            v=add(stage,"validate",config,[sys.executable,"scripts/validate_streams.py",str(folder)],deps=[d])
            data[regime,seed,n]=(folder,v)
    def run(stage,policy,regime,seed,k=3,flags=(),n=5):
        folder,dep=data[regime,seed,n]
        cfg={"policy":policy,"regime":regime,"partition_seed":seed,"training_seed":seed,
             "k":n if policy=="full" else k,"flags":list(flags),"clients":n}
        key=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:16]
        out=ROOT/"research_results"/key
        command=[sys.executable,"main.py","-data",str(folder),"-m","resnet18","-algo","FedAvg",
                 "-tasks","5","-nb","20","-nc",str(n),"-gr","49","-lbs","13","-lr","0.005","-ls","5",
                 "-V","200","-delta","1","-memory_size","500","-memory_batch_size","13","-seed",str(seed),
                 "--sync-policy",policy,"--sync-k",str(cfg["k"]),"--sync-output",str(out),*flags]
        return add(stage,"train",cfg,command,[dep],[str(out/"final_summary.json"),str(out/"status.json"),
            str(out/"communication_events.jsonl"),str(out/"server_metrics.jsonl")])
    policies=("full","random","oldest","own","server","peer")
    regimes=("near","separated","changing")
    for policy,regime,seed in itertools.product(policies,regimes,(42,43,44)):
        run("pilot",policy,regime,seed)
    for policy,regime,seed in itertools.product(policies,("clustered","dirichlet","conditional","drift"),(42,43,44)):
        run("ablation",policy,regime,seed,flags=("--sync-windowed",) if regime=="drift" else ())
    for policy,regime,seed,k in itertools.product(policies[1:],regimes,(42,43,44),(2,4,5)):
        run("budget",policy,regime,seed,k)
    for policy,regime,seed,tau in itertools.product(("own","server","peer"),regimes,(42,43,44),(.05,.1,.2,.35)):
        run("threshold",policy,regime,seed,flags=("--sync-mode","threshold","--sync-threshold",str(tau)))
    variants={
      "own":[("--sync-local-reference",x) for x in ("previous","window")],
      "server":[("--sync-bank-reference",x) for x in ("prior","mean","weighted_mean","envelope","history")]+
               [("--sync-delivery",x) for x in ("broadcast","telemetry")]+[("--sync-include-self",),("--sync-bank-transfer","delta")],
      "peer":[("--sync-peer-transport","relay"),("--sync-peer-solver","greedy"),("--sync-peer-objective","coverage")]}
    for policy,opts in variants.items():
        for flags,regime,seed in itertools.product(opts,regimes,(42,43,44)):
            run("ablation",policy,regime,seed,flags=flags)
    for policy,regime,seed in itertools.product(policies,regimes,(42,43,44)):
        for flags in (("--sync-refresh",),("--sync-local-capture",),("--sync-balanced-sampler",),
                      ("--sync-aggregation","complete"),("--sync-er",),("--sync-posttrain",),
                      ("--sync-max-skip","2"),("--sync-max-skip","8")):
            run("mechanism",policy,regime,seed,flags=flags)
    for policy,regime,seed in itertools.product(policies,regimes,(42,43,44)):
        run("counterfactual",policy,regime,seed,flags=("--sync-counterfactual",))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text("".join(json.dumps(job,sort_keys=True)+"\n" for job in jobs))
    counts={}
    for job in jobs:
        key=f'{job["stage"]}:{job["kind"]}'
        counts[key]=counts.get(key,0)+1
    print(json.dumps({"jobs":len(jobs),"counts":counts},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=ROOT/"experiments"/"job_ledger.jsonl")
    build(p.parse_args().output)
