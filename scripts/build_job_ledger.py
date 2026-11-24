#!/usr/bin/env python3
"""Canonical staged campaign ledger. Conditional extensions are not fake jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import itertools
import subprocess
ROOT=Path(__file__).resolve().parents[1]

def build(output):
    jobs=[]; seen=set()
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    generator_digest=hashlib.sha256((ROOT/"scripts/generate_streams.py").read_bytes()).hexdigest()[:12]
    def add(stage,kind,config,command,deps=(),artifacts=()):
        key=hashlib.sha256(json.dumps([kind,config,commit],sort_keys=True).encode()).hexdigest()[:16]
        if key in seen:
            return key
        seen.add(key)
        jobs.append({"id":key,"stage":stage,"kind":kind,"config":config,"command":command,"source_commit":commit,"schema":2,
                     "dependencies":list(deps),"artifacts":list(artifacts)})
        return key
    data={}
    for n in (5,10,20,50):
        regimes_for_n=("near","separated","changing","clustered","dirichlet","conditional","drift") if n==5 else ("near","clustered","separated")
        for regime,seed in itertools.product(regimes_for_n,(42,43,44)):
            folder=ROOT/"research_data"/f"{regime}-{seed}-n{n}-{generator_digest}"
            config={"regime":regime,"partition_seed":seed,"clients":n}
            stage=("pilot" if regime in ("near","separated","changing") else "ablation") if n==5 else "scale"
            d=add(stage,"generate",config,[sys.executable,"scripts/generate_streams.py","--regime",regime,
                 "--seed",str(seed),"--clients",str(n),"--output",str(folder),"--download"],
                 artifacts=[str(folder/"manifest.json")])
            v=add(stage,"validate",config,[sys.executable,"scripts/validate_streams.py",str(folder)],deps=[d])
            data[regime,seed,n]=(folder,v)
    def run(stage,policy,regime,seed,k=3,flags=(),n=5,extra_deps=()):
        folder,dep=data[regime,seed,n]
        split="test" if seed in (45,46,47) else "validation"
        cfg={"policy":policy,"regime":regime,"partition_seed":seed,"training_seed":seed,"evaluation_split":split,
             "k":n if policy=="full" else k,"flags":list(flags),"clients":n}
        key=hashlib.sha256(json.dumps([cfg,commit],sort_keys=True).encode()).hexdigest()[:16]
        out=ROOT/"research_results"/key
        command=[sys.executable,"main.py","-data",str(folder),"-m","resnet18","-algo","FedAvg",
                 "-tasks","5","-nb","20","-nc",str(n),"-gr","49","-lbs","13","-lr","0.005","-ls","5",
                 "-V","200","-delta","1","-memory_size","500","-memory_batch_size","13","-seed",str(seed),
                 "--sync-policy",policy,"--sync-k",str(cfg["k"]),"--sync-output",str(out),"--sync-checkpoint","--sync-eval-split",split,*flags]
        return add(stage,"train",cfg,command,[dep,*extra_deps],[str(out/"final_summary.json"),str(out/"status.json"),
            str(out/"communication_events.jsonl"),str(out/"server_metrics.jsonl"),str(out/"server.pt"),str(out/"provenance.json"),str(out/"rounds.jsonl")]+([str(out/"counterfactuals.jsonl")] if "--sync-counterfactual" in flags else []))
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
    for policy,regime,seed,signal in itertools.product(("own","server","peer"),regimes,(42,43,44),("replay","feature","gradient","update")):
        flags=("--sync-signal",signal)
        if signal=="update":flags+=("--sync-posttrain","--sync-native-space")
        if signal=="gradient":flags+=("--sync-native-space",)
        run("signals",policy,regime,seed,flags=flags)
    for policy,regime,seed in itertools.product(("cached","update_match","combined12","combined13","combined23","combined123"),regimes,(42,43,44)):
        flags=("--sync-signal","update","--sync-native-space") if policy=="update_match" else ()
        run("advanced",policy,regime,seed,flags=flags)
    for policy,regime,seed in itertools.product(("combined12","combined123"),regimes,(42,43,44)):
        run("advanced",policy,regime,seed,flags=("--sync-combine-rule","intersection"))
    impact_jobs=[]
    for regime,seed in itertools.product(regimes,(42,43,44)):
        impact_jobs.append(run("impact_data","cached",regime,seed,flags=("--sync-counterfactual","--sync-all-subsets")))
    model=ROOT/"research_results"/("impact_model-"+commit[:12]+".json")
    fit=add("impact_fit","fit",{"fit_seeds":[42],"cal_seeds":[43,44],"coverage":.5},[sys.executable,"scripts/fit_impact.py","--ledger",str(output.resolve()),"--output",str(model),"--fit-seeds","42","--cal-seeds","43","44","--coverage",".5"],impact_jobs,[str(model)])
    for regime,seed in itertools.product(regimes,(45,46,47)):
        folder=ROOT/"research_data"/f"{regime}-{seed}-n5-{generator_digest}"
        cfg={"regime":regime,"partition_seed":seed,"clients":5}
        d=add("adaptive", "generate",cfg,[sys.executable,"scripts/generate_streams.py","--regime",regime,"--seed",str(seed),"--clients","5","--output",str(folder),"--download"],artifacts=[str(folder/"manifest.json")])
        v=add("adaptive","validate",cfg,[sys.executable,"scripts/validate_streams.py",str(folder)],deps=[d]);data[regime,seed,5]=(folder,v)
        for policy in ("full","random","oldest","cached","peer"):
            run("adaptive",policy,regime,seed)
        for extra in ((),("--sync-soft-budget",)):
            run("adaptive","adaptive",regime,seed,flags=("--sync-impact-model",str(model),*extra),extra_deps=[fit])
    for n,policy,regime,seed in itertools.product((10,20,50),("full","random","oldest","cached","peer"),("near","clustered","separated"),(42,43,44)):
        run("scale",policy,regime,seed,k=max(1,int(.6*n)),n=n,flags=("--sync-peer-solver","greedy") if policy in ("cached","peer") else ())
    extended=("replay_histogram","replay_queue","prototypes","feature_covariance","spectral","parameter_norms",
              "normalization","predictions","loss","uncertainty","entropy","jacobian","fisher","gradient_variance")
    for policy,regime,seed,signal in itertools.product(("own","server","peer"),regimes,(42,43,44),extended):
        run("extended_signals",policy,regime,seed,flags=("--sync-signal",signal,"--sync-native-space"))
    for policy,regime,seed,metric in itertools.product(("own","server","peer"),regimes,(42,43,44),("js","hellinger")):
        run("extended_signals",policy,regime,seed,flags=("--sync-histogram-distance",metric))
    # Derive count schedules after actual threshold/adaptive runs. This is an
    # offline matched-rate baseline, not a deployable predictive scheduler.
    sources=[j for j in jobs if j["kind"]=="train" and j["stage"] in ("threshold","adaptive")
             and j["config"]["policy"] in ("own","server","peer","adaptive")]
    for job in sources:
        cfg=job["config"];out=Path(job["artifacts"][0]).parent
        schedule=ROOT/"research_results"/"count_schedules"/(job["id"]+".json")
        dep=add("rate_matched","schedule",{"source":job["id"]},[sys.executable,"scripts/extract_count_schedule.py",str(out/"rounds.jsonl"),"--output",str(schedule)],[job["id"]],[str(schedule)])
        run("rate_matched","random",cfg["regime"],cfg["partition_seed"],flags=("--sync-count-schedule",str(schedule)),extra_deps=[dep])
    for job in [j for j in jobs if j["kind"]=="train" and j["stage"] in ("counterfactual","impact_data")]:
        out=Path(job["artifacts"][0]).parent
        result=out/"counterfactual_analysis.json"
        add("counterfactual_analysis","analysis",{"source":job["id"]},[sys.executable,"scripts/analyze_counterfactuals.py",str(out/"counterfactuals.jsonl"),"--output",str(result)],[job["id"]],[str(result)])
    for regime,seed in itertools.product(regimes,(42,43,44)):
        for flags in (("--sync-peer-transport","sparse"),("--sync-peer-loss",".1"),("--sync-peer-loss",".3")):
            run("network","peer",regime,seed,flags=flags)
    for regime,seed in itertools.product(regimes,(42,43,44)):
        for horizon in (3,12):
            run("continuation","cached",regime,seed,flags=("--sync-counterfactual","--sync-counterfactual-horizon",str(horizon),"--sync-verify-branches"))
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
