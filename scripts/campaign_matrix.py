"""Immutable comprehensive controlled-stream experiment matrix."""
import hashlib,json,sys,subprocess
from pathlib import Path
from itertools import product
from core_suite import read_core_signal_entries,CORE_RUNTIME_FLAGS,PUBLIC_PROBE_PER_CLASS
ROOT=Path(__file__).resolve().parents[1]
CATALOG=ROOT/"configs/sync/campaign.json"
BASELINES={"B-RANDOM":"random","B-OLDEST":"oldest","B-FULL":"full"}
DETAILS=("regime_metrics","round_metrics","signal_comparisons","counterfactual_metrics","transition_response","optimization_comparisons","omission_comparisons")
OUTPUTS=("matrix.csv","matrix.json","matched_baseline_deltas.csv","seed_summary.csv","ground_truth_settings.csv","effectiveness.png","threshold_tradeoff.csv","threshold_tradeoff.png","learning_vs_bytes.jsonl","target_accuracy_bytes.jsonl","equal_byte_comparisons.jsonl","pareto_front.csv",*[x+".jsonl" for x in DETAILS])

def read_catalog():
    c=json.loads(CATALOG.read_text())
    if c["schema"]!=1 or [x["id"] for x in c["settings"]]!=[f"S{i:02d}" for i in range(1,25)] or c["seeds"]!=[42,43,44] or c["budgets"]!=[1,2,3,4] or c["baselines"]!=["random","oldest"] or not c["full_baseline"] or c["thresholds"]!=[.05,.1,.2,.35] or c["matched_controls"]!=["matched_count","matched_bytes"]:
        raise ValueError("comprehensive campaign catalog contract")
    compact=c.get("compact",{})
    if compact.get("seed")!=42 or compact.get("screening")!="S14" or compact.get("anchors")!=["S01","S03","S04","S10","S13"] or compact.get("oldest_anchor")!="S04" or compact.get("training_runs")!=101 or compact.get("counterfactual_all_subsets") is not True:
        raise ValueError("compact campaign catalog contract")
    for entry in c["settings"]:
        s=json.loads((ROOT/entry["path"]).read_text())
        if s["setting_id"]!=entry["id"] or s["setting"]!=entry["kind"]:raise ValueError("setting mismatch")
    return c

def variants():
    selectors=[(x["id"],x["signal"],x["method"],x["policy"],x["flags"]) for x in read_core_signal_entries()]
    return [*selectors,("B-RANDOM","none","baseline","random",[]),("B-OLDEST","none","baseline","oldest",[])]

def experiment_id(setting,seed,k,selector):
    return f"{setting}-seed{seed}-k{k}-{selector}"

def threshold_id(setting,seed,tau,selector):
    return f"{setting}-seed{seed}-t{round(tau*100):02d}-{selector}"

def primary_ids():
    c=read_catalog()
    return {experiment_id(s["id"],seed,k,x[0]) for s,seed,k,x in product(c["settings"],c["seeds"],c["budgets"],variants())}|{experiment_id(s["id"],seed,5,"B-FULL") for s,seed in product(c["settings"],c["seeds"])}|{threshold_id(s["id"],seed,tau,x[0]) for s,seed,tau,x in product(c["settings"],c["seeds"],c["thresholds"],variants()[:24])}

def expected_ids():
    base=primary_ids();selectors={x[0] for x in variants()[:24]}
    sources={x for x in base if any(x.endswith("-"+s) for s in selectors)}
    return base|{x+"-B-BYTES" for x in sources}|{x+"-B-COUNT" for x in sources if "-t" in x}

def build(output,compact=False,require_shortlist=False):
    from flcore.sync.backbone import BACKBONE
    from flcore.sync.bandwidth import ARTIFACTS as COMM_ARTIFACTS
    c=read_catalog();commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    generator=hashlib.sha256(b"".join((ROOT/p).read_bytes() for p in ("scripts/generate_streams.py","scripts/stream_design.py","scripts/validate_streams.py","scripts/build_ground_truth.py"))).hexdigest()[:12]
    jobs=[]
    suite="core" if compact else "exhaustive"
    seeds=[c["compact"]["seed"]] if compact else c["seeds"]
    result_root=ROOT/"research_results"/(("compact-" if compact else "comprehensive-")+commit[:12])
    def add(stage,kind,ident,cfg,cmd,deps,artifacts):
        identity={k:v for k,v in cfg.items() if k not in ("data_root","source_output")}
        key=hashlib.sha256(json.dumps([kind,ident,commit,identity],sort_keys=True).encode()).hexdigest()[:16]
        jobs.append({"id":key,"semantic_id":ident,"stage":stage,"kind":kind,"config":cfg,"command":cmd,"dependencies":list(deps),"artifacts":[str(p) for p in artifacts],"source_commit":commit,"schema":2})
        return key
    data={}
    settings=c["settings"]+([{"id":"S14N10","name":"S14 paired split, ten clients","kind":"split_dynamic","path":"configs/sync/settings/S14N10.json"}] if compact else [])
    for setting,seed in product(settings,seeds):
        sid=setting["id"];sp=ROOT/setting["path"];spec=json.loads(sp.read_text())
        spec_hash=hashlib.sha256(sp.read_bytes()).hexdigest()
        folder=ROOT/"research_data"/f"{sid}-seed{seed}-{generator}-{spec_hash[:12]}"
        cfg={"campaign":"controlled_comprehensive","setting_id":sid,"setting_name":setting["name"],"regime":spec["setting"],"partition_seed":seed,"clients":spec["clients"],"dataset":"cifar100","controlled_stream":True,"stream_spec_sha256":spec_hash,"data_root":str(folder)}
        ident=f"{sid}-seed{seed}"
        d=add("data","generate",ident+"-generate",cfg,[sys.executable,"scripts/generate_streams.py","--regime","clustered","--dataset","cifar100","--seed",str(seed),"--clients",str(spec["clients"]),"--output",str(folder),"--public-probe-per-class",str(PUBLIC_PROBE_PER_CLASS),"--controlled-stream",str(sp),"--download"],[],[folder/"manifest.json"])
        receipt=folder/"validation_receipt.json"
        v=add("data","validate",ident+"-validate",cfg,[sys.executable,"scripts/validate_streams.py",str(folder),"--receipt",str(receipt),"--verify-source"], [d],[receipt,*[folder/"ground_truth"/name for name in ("distributions.npz","specification.json","allocation_index.npz")]])
        data[sid,seed]=(folder,v,cfg,spec)
    training=[];sources=[]
    runtime=[*CORE_RUNTIME_FLAGS]+(["--sync-all-subsets"] if compact else [])
    shortlist_key=None;calibration_key=None;learning_rate=.005
    def add_case(sid,seed,mode,k,tau,selector,signal,mechanism,policy,options,stage=None,slot=None,calibration_lr=None):
            folder,dep,dcfg,spec=data[sid,seed]
            ident=experiment_id(sid,seed,k,selector) if mode=="fixed" else threshold_id(sid,seed,slot if slot is not None else tau,selector)
            n=spec["clients"];memory=2500//n
            if calibration_lr is not None:ident+=f"-LR{calibration_lr:g}"
            active_runtime=[x for x in runtime if x!="--sync-all-subsets" or (n==5 and calibration_lr is None)]
            flags=[*options,*active_runtime]
            cfg={**dcfg,"experiment_id":ident,"selector_id":selector,"signal":signal,"mechanism":mechanism,"policy":policy,"k":k,"selection_mode":mode,"threshold":tau,"max_skip":4,
                 "backbone":BACKBONE,"memory":memory,"replay_batch":13,"training_seed":seed,"learning_rate":learning_rate if calibration_lr is None else calibration_lr,"evaluation_split":"validation",
                 "tasks":5,"rounds":50,"flags":flags,"selection_timing":"pretrain","public_probe_per_class":PUBLIC_PROBE_PER_CLASS,
                 "counterfactual_rounds":spec["counterfactual_rounds"]}
            if compact:cfg.update(compact=True,all_subsets="--sync-all-subsets" in flags,threshold_slot=slot)
            out=result_root/ident
            cmd=[sys.executable,"main.py","-data",str(folder),"-m","resnet18","-algo","FedAvg","-tasks","5","-nb","20","-nc",str(n),"-gr","49","-lbs","13","-lr",str(cfg["learning_rate"]),"-ls","5","-memory_size",str(memory),"-memory_batch_size","13","-seed",str(seed),"--sync-er","--sync-memory-profile","uniform","--sync-aggregation","complete","--sync-weighting","current","--sync-local-capture","--sync-policy",policy,"--sync-k",str(k),"--sync-mode",mode,"--sync-threshold",str(tau),"--sync-max-skip","4","--sync-output",str(out),"--sync-checkpoint","--sync-eval-split","validation",*flags]
            if calibration_lr is not None:
                cfg.update(prefix_rounds=20)
                return add("calibration","calibration",ident,cfg,cmd+["--sync-stop-after-round","19"],[dep],[out/name for name in ("status.json","config.json","server_metrics.jsonl","rounds.jsonl","communication_events.jsonl","provenance.json")])
            artifacts=[out/name for name in ("final_summary.json","status.json","communication_events.jsonl","server_metrics.jsonl","server.pt","provenance.json","rounds.jsonl","config.json","decisions.jsonl","report_arrivals.jsonl","counterfactuals.jsonl","analysis/ground_truth_comparison.json",*COMM_ARTIFACTS,"heldout_test_metrics.json",*[f"ground_truth/{name}" for name in ("dataset_manifest.json","validation_receipt.json","distributions.npz","specification.json","allocation_index.npz")])]
            if mechanism!="baseline":artifacts += [out/"matched_byte_plan.json",out/"matched_byte_count_schedule.json"]
            key=add(stage or ("baselines" if mechanism=="baseline" else "signals"),"train",ident,cfg,cmd,[dep]+([calibration_key] if compact else [])+([shortlist_key] if compact and stage!="screening" else []),artifacts)
            training.append(key)
            if mechanism!="baseline":sources.append((key,ident,cfg,out,cmd,artifacts,dep))
    finalists=None
    calibration_ready=True
    if compact:
        from compact_calibration import calibration_valid
        calibration_jobs=[]
        for lr in (.005,.01,.02):
            key=add_case("S14",42,"fixed",5,.1,"B-FULL","none","baseline","full",[],calibration_lr=lr)
            calibration_jobs.append(jobs[-1])
        cal_receipt=result_root/"calibration.json"
        calibration_key=add("calibration","analysis","compact-calibration",{"scope":"compact_calibration","candidates":calibration_jobs},[sys.executable,"scripts/compact_calibration.py","--ledger",str(output.resolve()),"--output",str(cal_receipt)],[j["id"] for j in calibration_jobs],[cal_receipt])
        calibration_ready=calibration_valid(jobs[-1])
        if calibration_ready:learning_rate=json.loads(cal_receipt.read_text())["learning_rate"]
    if compact and calibration_ready:
        from compact_campaign import compact_cases,shortlist_valid,scale_cases
        for sid,mode,k,tau,x in compact_cases(None):
            add_case(sid,42,mode,k,tau,*x,stage="screening")
        screen=[j for j in jobs if j["kind"]=="train"]
        receipt=result_root/"shortlist.json"
        shortlist_key=add("shortlist","analysis","compact-shortlist",{"scope":"compact_shortlist","catalog_sha256":hashlib.sha256(CATALOG.read_bytes()).hexdigest(),"screening_jobs":screen},[sys.executable,"scripts/compact_campaign.py","--ledger",str(output.resolve()),"--output",str(receipt)],training.copy(),[receipt])
        if shortlist_valid(jobs[-1]):
            selection=json.loads(receipt.read_text());finalists=selection["finalists"]
            for sid,mode,k,tau,x in [*compact_cases(finalists)[27:],*scale_cases(finalists)]:
                actual=selection["thresholds"][x[0]][str(tau)] if mode=="threshold" else tau
                add_case(sid,42,mode,k,actual,*x,stage="scale" if sid=="S14N10" else "adaptive" if mode=="threshold" else "participation" if sid=="S14" else "generalization",slot=tau if mode=="threshold" else None)
        elif require_shortlist:
            raise ValueError("complete and verify the 27 screening jobs before expanding the compact ledger")
    elif not compact:
        for setting,seed in product(c["settings"],seeds):
            cases=[("fixed",k,.1,*x) for k,x in product(c["budgets"],variants())]+[("fixed",5,.1,"B-FULL","none","baseline","full",[])]+[("threshold",5,tau,*x) for tau,x in product(c["thresholds"],variants()[:24])]
            for case in cases:add_case(setting["id"],seed,*case)
    for key,source_id,source_cfg,source_out,source_cmd,source_artifacts,dep in sources:
        if compact and source_cfg["selection_mode"]!="threshold":continue
        modes=["matched_bytes"]+(["matched_count"] if source_cfg["selection_mode"]=="threshold" else [])
        for mode in modes:
            suffix="B-BYTES" if mode=="matched_bytes" else "B-COUNT"
            ident=source_id+"-"+suffix;out=source_out.with_name(ident)
            active_runtime=[x for x in runtime if x!="--sync-all-subsets" or source_cfg.get("all_subsets")]
            cfg={**source_cfg,"experiment_id":ident,"selector_id":suffix+"-"+source_cfg["selector_id"],"signal":"none","mechanism":"baseline","policy":"random","selection_mode":mode,"source_selection_mode":source_cfg["selection_mode"],"source_experiment_id":source_id,"source_selector_id":source_cfg["selector_id"],"source_output":str(source_out),"flags":list(active_runtime)}
            cmd=source_cmd[:source_cmd.index("--sync-eval-split")+2]+list(active_runtime)
            for flag,value in (("--sync-policy","random"),("--sync-mode","fixed"),("--sync-output",str(out))):cmd[cmd.index(flag)+1]=value
            cmd += ["--sync-count-schedule",str(source_out/("participation_schedule.json" if mode=="matched_count" else "matched_byte_count_schedule.json"))]
            artifacts=[out/p.relative_to(source_out) for p in source_artifacts if p.name not in ("matched_byte_plan.json","matched_byte_count_schedule.json")]
            training.append(add("matched" if compact else "baselines","train",ident,cfg,cmd,[dep,key],artifacts))
    result=result_root/"analysis"
    if compact and require_shortlist and not calibration_ready:raise ValueError("complete learning-rate calibration first")
    if not compact or finalists is not None:
        if compact:
            from compact_campaign import compact_ids,EXTRA_OUTPUTS
            expected=compact_ids(finalists,include_scale=True)
            deps=[*training,shortlist_key,calibration_key,*[v for _,v,_,_ in data.values()]]
            cfg={"scope":"controlled_compact","expected_training":123,"expected_ids":sorted(expected),"expected_groups":7,"calibration":str(cal_receipt),"catalog_sha256":hashlib.sha256(CATALOG.read_bytes()).hexdigest(),"shortlist":str(receipt)}
            outputs=(*OUTPUTS,*EXTRA_OUTPUTS)
        else:
            expected=expected_ids();deps=training;outputs=OUTPUTS
            cfg={"scope":"controlled_comprehensive","expected_training":35208,"catalog_sha256":hashlib.sha256(CATALOG.read_bytes()).hexdigest()}
        add("postprocess","analysis","compact-analysis" if compact else "comprehensive-analysis",cfg,[sys.executable,"scripts/analyze_controlled_campaign.py","--ledger",str(output.resolve()),"--output",str(result)],deps,[result/name for name in outputs])
        assert len(training)==len(expected)==(123 if compact else 35208) and {j["semantic_id"] for j in jobs if j["kind"]=="train"}==expected
    assert len(jobs)==((179 if finalists is not None else 82 if calibration_ready else 54) if compact else 35353)
    assert len({j["id"] for j in jobs})==len(jobs)
    from run_campaign import dependency_order
    dependency_order(jobs)
    output.parent.mkdir(parents=True,exist_ok=True);tmp=output.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(j,sort_keys=True)+"\n" for j in jobs));tmp.replace(output)
    print(json.dumps({"suite":suite,"planned_training_jobs":123 if compact else 35208,"planned_jobs":179 if compact else 35353,"calibration_prefixes":3 if compact else 0,"expanded":not compact or finalists is not None,"settings":len(settings),"seeds":seeds,"budgets":c["budgets"],"jobs":len(jobs),"training_jobs":len(training),"baseline_runs":sum(j["kind"]=="train" and j["config"]["mechanism"]=="baseline" for j in jobs)},indent=2))
    return jobs
