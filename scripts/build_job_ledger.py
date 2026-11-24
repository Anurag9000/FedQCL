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

def build(output,tiny_root=None):
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
    dimensions={}
    def register(tag,regime,seed,n=5,dataset="cifar100",gen_flags=(),stage="data_extended"):
        tasks,classes=(10,20) if dataset=="tinyimagenet" else (5,2) if dataset=="cifar10" else (5,20)
        folder=ROOT/"research_data"/f"{tag}-{seed}-n{n}-{generator_digest}"
        cfg={"regime":regime,"tag":tag,"partition_seed":seed,"clients":n,"dataset":dataset,"generation_flags":list(gen_flags)}
        command=[sys.executable,"scripts/generate_streams.py","--regime",regime,"--dataset",dataset,"--seed",str(seed),"--clients",str(n),"--output",str(folder),*gen_flags]
        if dataset!="tinyimagenet":command.append("--download")
        d=add(stage,"generate",cfg,command,artifacts=[str(folder/"manifest.json")])
        receipt=folder/"validation_receipt.json"
        v=add(stage,"validate",cfg,[sys.executable,"scripts/validate_streams.py",str(folder),"--receipt",str(receipt)],deps=[d],artifacts=[str(receipt)])
        data[tag,seed,n]=(folder,v);dimensions[tag,seed,n]=(dataset,tasks,classes)
    for n in (5,10,20,50):
        regimes_for_n=("near","separated","changing","clustered","dirichlet","conditional","drift") if n==5 else ("near","clustered","separated")
        for regime,seed in itertools.product(regimes_for_n,(42,43,44)):
            folder=ROOT/"research_data"/f"{regime}-{seed}-n{n}-{generator_digest}"
            config={"regime":regime,"partition_seed":seed,"clients":n}
            stage=("pilot" if regime in ("near","separated","changing") else "ablation") if n==5 else "scale"
            d=add(stage,"generate",config,[sys.executable,"scripts/generate_streams.py","--regime",regime,
                 "--seed",str(seed),"--clients",str(n),"--output",str(folder),"--download"],
                 artifacts=[str(folder/"manifest.json")])
            receipt=folder/"validation_receipt.json"
            v=add(stage,"validate",config,[sys.executable,"scripts/validate_streams.py",str(folder),"--receipt",str(receipt)],deps=[d],artifacts=[str(receipt)])
            data[regime,seed,n]=(folder,v)
    def run(stage,policy,regime,seed,k=3,flags=(),n=5,extra_deps=(),eval_split=None,training_seed=None):
        folder,dep=data[regime,seed,n]
        split=eval_split or ("test" if seed in (45,46,47) else "validation")
        training_seed=seed if training_seed is None else training_seed
        dataset,tasks,classes=dimensions.get((regime,seed,n),("cifar100",5,20))
        total_rounds=tasks*10
        cfg={"policy":policy,"regime":regime,"partition_seed":seed,"training_seed":training_seed,"evaluation_split":split,"dataset":dataset,"tasks":tasks,"rounds":total_rounds,
             "k":n if policy=="full" else k,"flags":list(flags),"clients":n}
        key=hashlib.sha256(json.dumps([cfg,commit],sort_keys=True).encode()).hexdigest()[:16]
        out=ROOT/"research_results"/key
        command=[sys.executable,"main.py","-data",str(folder),"-m","resnet18","-algo","FedAvg",
                 "-tasks",str(tasks),"-nb",str(classes),"-nc",str(n),"-gr",str(total_rounds-1),"-lbs","13","-lr","0.005","-ls","5",
                 "-V",str(20 if dataset=="cifar10" else 220 if dataset=="tinyimagenet" else 200),"-delta","1","-memory_size","500","-memory_batch_size","13","-seed",str(training_seed),
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
        receipt=folder/"validation_receipt.json"
        v=add("adaptive","validate",cfg,[sys.executable,"scripts/validate_streams.py",str(folder),"--receipt",str(receipt)],deps=[d],artifacts=[str(receipt)]);data[regime,seed,5]=(folder,v)
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
    failures=ROOT/"configs/sync/robustness_failures.json"
    for policy,regime,seed in itertools.product(("full","random","own","server","peer"),regimes,(42,43,44)):
        flags=("--sync-failures",str(failures))
        if policy=="peer":flags+=("--sync-peer-transport","relay")
        run("robustness",policy,regime,seed,flags=flags)
        if policy=="peer":
            run("robustness",policy,regime,seed,flags=flags+("--sync-fallback",))
    # Focused implementation-extension matrices; no automatic training invocation.
    new_signals=("multichannel","gradient_conflict","information","temporal","bayesian_head",
                 "representation_anchor","replay_diagnostics","fisher_full","parameter_jacobian","head_hessian","conditional_moments")
    for policy,regime,seed,signal in itertools.product(("own","server","peer"),regimes,(42,43,44),new_signals):
        run("math_signals",policy,regime,seed,flags=("--sync-signal",signal,"--sync-native-space"))
    for regime,seed in itertools.product(regimes,(42,43,44)):
        for policy in ("cached","peer"):
            for weight in ("samples","replay","aggregation"):
                run("weighted",policy,regime,seed,flags=("--sync-coverage-weight",weight))
            run("weighted",policy,regime,seed,flags=("--sync-signal","multichannel","--sync-native-space","--sync-coverage-weight","replay"))
        run("weighted","update_match",regime,seed,flags=("--sync-signal","update","--sync-native-space","--sync-update-weighted"))
        run("weighted","cached",regime,seed,flags=("--sync-staleness-predict",))
        for policy in ("combined13","combined23","combined123"):
            run("weighted",policy,regime,seed,flags=("--sync-joint-hybrid",))
        for policy in ("own","server","combined12"):
            run("dynamic_threshold",policy,regime,seed,flags=("--sync-mode","threshold","--sync-dynamic-threshold"))
        for policy,tau in itertools.product(("combined12","combined13","combined23","combined123"),(.05,.1,.2,.35)):
            run("combined_threshold",policy,regime,seed,flags=("--sync-mode","threshold","--sync-threshold",str(tau)))
        for profile in ("uniform","balanced","rare"):
            for policy in ("full","cached","own"):
                run("replay_profiles",policy,regime,seed,flags=("--sync-memory-profile",profile))
        for policy in ("full","random","own","server","peer"):
            for variant in (("--sync-queue-mode","sample_mean"),("--sync-reference-recovery",),("--sync-v-normalized-lr",),
                            ("--sync-weighting","equal"),("--sync-weighting","current"),("--sync-weighting","processed")):
                run("correctness",policy,regime,seed,flags=variant)
        for policy in ("random","own","server","peer","cached"):
            for cap in (190000000,280000000,370000000):
                run("hard_bytes",policy,regime,seed,k=5,flags=("--sync-byte-cap",str(cap),"--sync-bootstrap-byte-cap","500000000"))
            run("hard_bytes",policy,regime,seed,k=3,flags=("--sync-campaign-byte-cap","14000000000","--sync-budget-reserve"))
        for policy in ("own","server","peer"):
            for fmt in ("json","fp16","int8"):
                run("encoding",policy,regime,seed,flags=("--sync-signal","feature","--sync-wire-format",fmt))
            for epsilon in (.5,2.):
                run("privacy",policy,regime,seed,flags=("--sync-histogram-rr-epsilon",str(epsilon)))
            run("encoding",policy,regime,seed,flags=("--sync-charge-control",))
        for fault in ("none","delay","duplicate","reorder","corrupt"):
            run("distributed", "peer",regime,seed,flags=("--sync-peer-transport","distributed_sparse","--sync-dynamic-topology","--sync-peer-loss",".1","--sync-gossip-fault",fault))
        for length in (2,4):
            run("comparators","block_random",regime,seed,flags=("--sync-block-length",str(length)))
        run("comparators","mass_stratified",regime,seed)
        run("comparators","divfl_sketch",regime,seed,flags=("--sync-signal","gradient","--sync-native-space"))
        for policy in ("oldest","random"):
            run("continuation_extended","cached",regime,seed,flags=("--sync-counterfactual","--sync-counterfactual-horizon","12","--sync-continuation-policy",policy,"--sync-oracle-past-weight","1"))
    # Controlled unequal workloads, asynchronous/recurring drift and conditional domains.
    for seed in (42,43,44):
        for tag,regime,gflags in (("unequal","separated",("--workload-skew","1.5")),
                 ("abrupt","drift",("--drift-pattern","abrupt")),("recurring","drift",("--drift-pattern","recurring")),
                 ("asynchronous","drift",("--drift-pattern","asynchronous")),
                 ("color","conditional",("--conditional-domain","color")),("contrast","conditional",("--conditional-domain","contrast"))):
            register(tag,regime,seed,gen_flags=gflags)
            for policy in policies:
                run("data_extended",policy,tag,seed,flags=("--sync-windowed",) if regime=="drift" else ())
        register("cifar10_near","near",seed,dataset="cifar10",stage="dataset_transfer")
        for policy in policies:run("dataset_transfer",policy,"cifar10_near",seed)
        if tiny_root is not None:
            register("tiny_near","near",seed,dataset="tinyimagenet",gen_flags=("--tiny-root",str(tiny_root)),stage="dataset_transfer")
            for policy in policies:run("dataset_transfer",policy,"tiny_near",seed)
    # Independent partition seeds and a common seed-level calibration unit.
    # Twenty calibration groups support a finite 95% marginal bound under exchangeability.
    fit_seeds=tuple(range(60,65));cal_seeds=tuple(range(65,85));test_seeds=tuple(range(85,90))
    strict_jobs=[]
    for seed in fit_seeds+cal_seeds:
        for regime in regimes:
            register("strict_"+regime,regime,seed,stage="impact_data_strict")
            strict_jobs.append(run("impact_data_strict","cached","strict_"+regime,seed,flags=("--sync-signal","multichannel","--sync-native-space","--sync-counterfactual","--sync-all-subsets")))
    strict_model=ROOT/"research_results"/("impact_strict-"+commit[:12]+".json")
    strict_fit=add("impact_fit_strict","fit",{"fit_seeds":list(fit_seeds),"cal_seeds":list(cal_seeds),"coverage":.95},
        [sys.executable,"scripts/fit_impact.py","--ledger",str(output.resolve()),"--output",str(strict_model),
         "--data-stage","impact_data_strict","--fit-seeds",*map(str,fit_seeds),"--cal-seeds",*map(str,cal_seeds),
         "--coverage",".95","--feature-map","quadratic","--ood-guard"],strict_jobs,[str(strict_model)])
    for seed,regime in itertools.product(test_seeds,regimes):
        tag="strict_"+regime
        register(tag,regime,seed,stage="confirmatory")
        for policy in ("full","random","oldest","cached","peer","own","server"):
            run("confirmatory",policy,tag,seed,eval_split="test")
        for extra in ((),("--sync-soft-budget",),("--sync-byte-cap","280000000","--sync-bootstrap-byte-cap","500000000")):
            run("confirmatory","adaptive",tag,seed,flags=("--sync-signal","multichannel","--sync-native-space","--sync-impact-model",str(strict_model),*extra),extra_deps=[strict_fit],eval_split="test")
        # Dedicated independent training seeds, not coupled to the partition seed.
        run("confirmatory","cached",tag,seed,eval_split="test",training_seed=seed+1000)
    # Byte-capped source-matched comparator: do not claim exact equality unless
    # actual serialized costs subsequently meet the declared tolerance.
    sources=[j for j in jobs if j["kind"]=="train" and j["stage"]=="threshold"]
    for job in sources:
        cfg=job["config"];out=Path(job["artifacts"][0]).parent
        schedule=ROOT/"research_results"/"byte_schedules"/(job["id"]+".json")
        dep=add("byte_matched","schedule",{"source":job["id"],"unit":"bytes"},[sys.executable,"scripts/extract_byte_schedule.py",str(out/"communication_events.jsonl"),"--output",str(schedule)],[job["id"]],[str(schedule)])
        run("byte_matched","random",cfg["regime"],cfg["partition_seed"],k=5,flags=("--sync-byte-schedule",str(schedule)),extra_deps=[dep])
    # Actual TCP loopback payload benchmark, no neural training.
    for size in (80,1024,65536,45000000):
        out=ROOT/"research_results"/"network_benchmarks"/f"{size}-{commit[:12]}.json"
        add("transport_benchmark","benchmark",{"size":size},[sys.executable,"scripts/benchmark_transport.py","--size",str(size),"--output",str(out)],artifacts=[str(out)])
    # Full postprocessing is dependency-gated; standalone scripts can still
    # analyze partial completed results without claiming campaign completion.
    training=[j["id"] for j in jobs if j["kind"]=="train"]
    results=ROOT/"research_results"/("analysis-"+commit[:12])
    csv_file=results/"aggregate.csv"
    aggregate=add("postprocess","analysis",{"scope":"complete_campaign"},[sys.executable,"scripts/aggregate_results.py","--ledger",str(output.resolve()),"--output",str(csv_file)],training,[str(csv_file),str(csv_file.with_suffix(".paired.json"))])
    matches=results/"budget_matches.json"
    add("postprocess","analysis",{"scope":"budget_matching"},[sys.executable,"scripts/match_budgets.py","--ledger",str(output.resolve()),"--output",str(matches)],training,[str(matches)])
    fig_receipt=results/"figures"/"receipt.json"
    add("postprocess","analysis",{"scope":"figures"},[sys.executable,"scripts/plot_results.py","--input",str(csv_file),"--output",str(fig_receipt.parent)], [aggregate],[str(fig_receipt)])
    # Verify unique IDs and topological dependency order before writing.
    ordered=set()
    for job in jobs:
        if job["id"] in ordered or not set(job["dependencies"])<=ordered:
            raise RuntimeError("invalid campaign dependency graph")
        ordered.add(job["id"])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text("".join(json.dumps(job,sort_keys=True)+"\n" for job in jobs))
    counts={}
    for job in jobs:
        key=f'{job["stage"]}:{job["kind"]}'
        counts[key]=counts.get(key,0)+1
    print(json.dumps({"jobs":len(jobs),"counts":counts},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=ROOT/"experiments"/"job_ledger.jsonl")
    p.add_argument("--tiny-root",type=Path,help="Opt in to extracted TinyImageNet transfer runs")
    a=p.parse_args();build(a.output,a.tiny_root.resolve() if a.tiny_root else None)
