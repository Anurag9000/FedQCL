#!/usr/bin/env python3
"""Follow-up to archived compact-65ced6195777; independent, immutable ledger."""
import argparse,csv,hashlib,json,sys,subprocess,math
from pathlib import Path
from collections import defaultdict
from statistics import mean,stdev
from itertools import product
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"system"))
from core_suite import CORE_RUNTIME_FLAGS,PUBLIC_PROBE_PER_CLASS,read_core_signal_entries
from flcore.sync.bandwidth import ARTIFACTS
from flcore.sync.backbone import BACKBONE
ARCHIVE=ROOT/"research_results/compact-65ced6195777/analysis"
EXPECTED_LR=.01
FINALISTS={"M1":"A03-M1","M2":"A03-M2","M3":"A05-M3"}
MODES={"local":"own","server":"server","peer":"peer","loss":"own","balanced":"random","random":"random"}
OUTPUTS=("matrix.json","matrix.csv","paired_comparisons.csv","oracle_diagnostics.csv","learning_vs_bytes.jsonl","equal_bytes.jsonl","report.md")

def cases():
    rows=[]
    def add(stage,sid,seed,name,mode="off",policy=None,k=3,budget=0.,fresh=True,options=(),source=None):
        n=10 if sid=="S14N10" else 5
        rows.append(dict(stage=stage,setting_id=sid,seed=seed,name=name,mode=mode,policy=policy or MODES[mode],k=k,budget=budget,fresh=fresh,options=list(options),source=source,clients=n))
    for sid in ("S04","S14"):
        for mode in ("random","balanced","loss","local","server","peer"):
            add("development",sid,42,mode,mode)
        add("development",sid,42,"full",policy="full",k=5)
        add("aggregation",sid,42,"stale-random",policy="random",fresh=False)
        add("aggregation",sid,42,"delta-random",policy="random",fresh=False,options=["--sync-delta-aggregation"])
        for mech,selector in FINALISTS.items():
            entry=next(x for x in read_core_signal_entries() if x["id"]==selector)
            add("aggregation",sid,42,"fresh-"+selector,policy=entry["policy"],options=entry["flags"])
            # The archive is a historical reference, but rerun matched stale controls
            # to avoid relying on different source revisions or incomplete artifacts.
            if mech=="M2":
                add("aggregation",sid,42,"stale-"+selector,policy=entry["policy"],fresh=False,options=entry["flags"])
                add("aggregation",sid,42,"delta-"+selector,policy=entry["policy"],fresh=False,options=[*entry["flags"],"--sync-delta-aggregation"])
        for budget in (.4,.6):
            for mode in ("random","server","local","peer"):
                name=f"{mode}-b{int(budget*100)}"
                add("budget",sid,42,name,mode,k=5,budget=budget)
                if mode!="random":add("matched",sid,42,name+"-count",policy="random",k=5,source=name)
    for label,options in (
        ("no-mass",["--sync-utility-mass","0"]),("no-match",["--sync-utility-match","0"]),
        ("no-fair",["--sync-utility-fair","0"]),("no-risk",["--sync-utility-risk","0","--sync-utility-gate","0"]),
        ("no-retention",["--sync-utility-past","0","--sync-utility-retention","100"]),
        ("fp16",["--sync-utility-precision","fp16"]),("int8",["--sync-utility-precision","int8"]),
        ("small-sketch",["--sync-sketch-dim","256"]),("large-sketch",["--sync-sketch-dim","2048"]),
        ("skip2",["--sync-max-skip","2"]),("skip8",["--sync-max-skip","8"]),
        ("horizon3",["--sync-counterfactual-horizon","3"]),("proxy",["--sync-utility-proxy"]),("adaptive-precision",["--sync-utility-adaptive-precision"])):
        add("ablation","S14",42,"server-"+label,"server",options=options)
    # Same predefined method; confirmation is not winner-picked on test accuracy.
    for sid,seed in (("S01",43),("S08",43),("S13",43),("S18",43),("S04",43),("S14",43)):
        for mode in ("random","server"):add("confirmation",sid,seed,mode,mode)
        add("learned",sid,seed,"server-learned","server")
    for mode in ("random","server","peer"):
        add("scale","S14N10",42,mode,mode,k=6)
    return rows

def build(output):
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    generator=hashlib.sha256(b"".join((ROOT/p).read_bytes() for p in ("scripts/generate_streams.py","scripts/stream_design.py","scripts/validate_streams.py","scripts/build_ground_truth.py"))).hexdigest()[:12]
    lr=json.loads((ARCHIVE/"calibration.json").read_text())
    shortlist=json.loads((ARCHIVE/"shortlist.json").read_text())
    if lr["learning_rate"]!=EXPECTED_LR or shortlist["finalists"]!=FINALISTS:raise ValueError("archive reference contract")
    jobs=[];root=ROOT/"research_results"/("utility-"+commit[:12])
    def add(stage,kind,ident,cfg,cmd,deps,artifacts):
        identity={k:v for k,v in cfg.items() if k not in ("data_root","source_output","expected_flags","source_jobs")}
        key=hashlib.sha256(json.dumps([kind,ident,commit,identity],sort_keys=True).encode()).hexdigest()[:16]
        jobs.append(dict(id=key,semantic_id=ident,stage=stage,kind=kind,config=cfg,command=cmd,dependencies=list(deps),artifacts=list(map(str,artifacts)),source_commit=commit,schema=2))
        return key
    data={};source_jobs={}
    for sid,seed in sorted({(x["setting_id"],x["seed"]) for x in cases()}):
        sp=ROOT/"configs/sync/settings"/(sid+".json");spec=json.loads(sp.read_text())
        h=hashlib.sha256(sp.read_bytes()).hexdigest()
        folder=ROOT/"research_data"/f"{sid}-seed{seed}-{generator}-{h[:12]}"
        cfg=dict(campaign="utility",setting_id=sid,partition_seed=seed,clients=spec["clients"],dataset="cifar100",controlled_stream=True,stream_spec_sha256=h,data_root=str(folder))
        ident=f"{sid}-seed{seed}"
        d=add("data","generate",ident+"-generate",cfg,[sys.executable,"scripts/generate_streams.py","--regime","clustered","--dataset","cifar100","--seed",str(seed),"--clients",str(spec["clients"]),"--output",str(folder),"--public-probe-per-class",str(PUBLIC_PROBE_PER_CLASS),"--controlled-stream",str(sp),"--download"],[],[folder/"manifest.json"])
        v=add("data","validate",ident+"-validate",cfg,[sys.executable,"scripts/validate_streams.py",str(folder),"--receipt",str(folder/"validation_receipt.json"),"--verify-source"],[d],[folder/"validation_receipt.json",*[folder/"ground_truth"/x for x in ("distributions.npz","specification.json","allocation_index.npz")]])
        data[sid,seed]=(folder,v,cfg,spec)
    training=[];calibration_key=None;model_path=root/"analysis"/"utility_model.json"
    for case in cases():
        sid,seed,name=case["setting_id"],case["seed"],case["name"]
        folder,dep,dcfg,spec=data[sid,seed];n=case["clients"];k=case["k"]
        ident=f"{sid}-seed{seed}-{name}";out=root/ident
        mode=case["mode"];budget=case["budget"]
        flags=[*CORE_RUNTIME_FLAGS,"--sync-utility",mode,"--sync-sketch-dim","1024","--sync-observation-size","64","--sync-min-observation","1","--sync-utility-budget",str(budget)]
        if n==5:flags+=["--sync-all-subsets"]
        if case["fresh"]:flags+=["--sync-fresh-start"]
        if case["policy"]=="server":flags+=["--sync-bank-reference","aggregate"]
        flags+=case["options"]
        cmd=[sys.executable,"main.py","-data",str(folder),"-m","resnet18","-algo","FedAvg","-tasks","5","-nb","20","-nc",str(n),"-gr","49","-lbs","13","-lr",str(EXPECTED_LR),"-ls","5","-memory_size",str(2500//n),"-memory_batch_size","13","-seed",str(seed),"--sync-er","--sync-memory-profile","uniform","--sync-aggregation","complete","--sync-weighting","current","--sync-local-capture","--sync-policy",case["policy"],"--sync-k",str(k),"--sync-mode","fixed","--sync-threshold",".1","--sync-max-skip","4","--sync-output",str(out),"--sync-checkpoint","--sync-eval-split","validation",*flags]
        dependencies=[dep];source_out=None
        if case["stage"]=="learned":
            if calibration_key is None:
                selected=[source_jobs[s,42,n][0] for s in ("S14","S04") for n in ("server","server-b40","server-b60")]
                calibration_key=add("calibration","analysis","utility-calibration",{"scope":"utility_calibration","sources":selected,"source_jobs":[j for j in jobs if j["id"] in selected]},[sys.executable,"scripts/fit_utility.py","--ledger",str(output.resolve()),"--output",str(model_path)],selected,[model_path])
            dependencies.append(calibration_key);cmd+=["--sync-utility-model",str(model_path)]
        if case["source"]:
            src,source_out=source_jobs[sid,seed,case["source"]]
            dependencies.append(src);cmd+=["--sync-count-schedule",str(source_out/"participation_schedule.json")]
        # Record the actual CLI values for validation, including override flags.
        resolved={}
        bools={x[2:].replace("-","_") for x in CORE_RUNTIME_FLAGS}|{"sync_fresh_start","sync_all_subsets","sync_native_space","sync_er","sync_local_capture","sync_checkpoint","sync_delta_aggregation","sync_utility_proxy","sync_utility_adaptive_precision"}
        for i,arg in enumerate(cmd):
            if arg.startswith("--sync-"):
                field=arg[2:].replace("-","_")
                if field in bools:resolved[field]=True
                else:resolved[field]=cmd[i+1]
        cfg={**dcfg,**case,"experiment_id":ident,"selector_id":name,"training_seed":seed,"partition_seed":seed,"memory":2500//n,"replay_batch":13,"rounds":50,"tasks":5,"counterfactual_rounds":spec["counterfactual_rounds"],"all_subsets":n==5,"selection_mode":"matched_count" if source_out else "threshold" if budget else "fixed","source_output":str(source_out) if source_out else None,"expected_flags":resolved,"learning_rate":EXPECTED_LR,"backbone":BACKBONE}
        artifacts=[out/x for x in ("final_summary.json","status.json","config.json","provenance.json","rounds.jsonl","decisions.jsonl","communication_events.jsonl","server_metrics.jsonl","report_arrivals.jsonl","server.pt","counterfactuals.jsonl","heldout_test_metrics.json","analysis/ground_truth_comparison.json",*ARTIFACTS,*["ground_truth/"+x for x in ("dataset_manifest.json","validation_receipt.json","distributions.npz","specification.json","allocation_index.npz")])]
        key=add(case["stage"],"train",ident,cfg,cmd,dependencies,artifacts);training.append(key);source_jobs[sid,seed,name]=(key,out)
    add("postprocess","analysis","utility-analysis",{"scope":"utility_analysis","expected_training":len(training)},[sys.executable,"scripts/utility_campaign.py","--analyze","--ledger",str(output.resolve()),"--output",str(root/"analysis")],training,[root/"analysis"/x for x in OUTPUTS])
    from run_campaign import dependency_order
    dependency_order(jobs)
    if len(training)!=len(cases()):raise ValueError("utility inventory")
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_suffix(".tmp");tmp.write_text("".join(json.dumps(j,sort_keys=True)+"\n" for j in jobs));tmp.replace(output)
    print(json.dumps({"suite":"utility","training_jobs":len(training),"data_jobs":2*len(data),"jobs":len(jobs)},indent=2))
    return jobs

def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def verify(job,root,cfg,summary,rounds,metrics,events):
    from flcore.sync.bandwidth import verify_artifacts
    from flcore.sync.decision_audit import verify_decisions
    from flcore.sync.ground_truth import verify as verify_truth
    from run_campaign import sha_file
    c=job["config"];n=c["clients"]
    for key,expected in c["expected_flags"].items():
        actual=cfg.get(key)
        if isinstance(actual,bool):
            if actual!=expected:return False
        elif isinstance(actual,(float,int)):
            if float(expected)!=actual:return False
        elif str(actual)!=str(expected):return False
    if any(cfg.get(k)!=v for k,v in dict(seed=c["seed"],num_clients=n,tasks=5,num_classes=20,global_rounds=49,local_epochs=5,batch_size=13,memory_size=c["memory"],memory_batch_size=13,local_learning_rate=EXPECTED_LR).items()):return False
    if cfg["source_commit"]!=job["source_commit"]:return False
    decisions=rows(root/"decisions.jsonl")
    verify_artifacts(root,events,rounds,n,summary,c["policy"])
    verify_decisions(cfg,rounds,decisions,events)
    truth=root/"ground_truth";receipt=json.loads((truth/"validation_receipt.json").read_text())
    if not receipt.get("source_verified") or receipt["verified_source_examples"]!=60000:return False
    verify_truth(truth,receipt)
    if receipt!=json.loads((Path(c["data_root"])/"validation_receipt.json").read_text()) or summary["ground_truth_hash"]!=receipt["ground_truth_hash"] or cfg["ground_truth_hash"]!=receipt["ground_truth_hash"]:return False
    manifest=json.loads((truth/"dataset_manifest.json").read_text())
    if manifest["controlled_stream_sha256"]!=c["stream_spec_sha256"] or manifest["manifest_id"]!=cfg["dataset_manifest_id"]:return False
    if c["stage"]=="learned" and cfg["sync_utility_model_sha256"]!=sha_file(cfg["sync_utility_model"]):return False
    if c["source_output"]:
        src=Path(c["source_output"]);path=src/"participation_schedule.json"
        if cfg["sync_count_schedule_sha256"]!=sha_file(path) or json.loads(path.read_text())!=[len(x["selected"]) for x in rounds]:return False
    elif not c["budget"] and any(len(x["selected"])!=(n if i==0 else c["k"]) for i,x in enumerate(rounds)):return False
    if any(x["selected"]!=x["trained"] or x["accepted"]!=x["downloaded"] or x["selected"]!=x["accepted"] for x in rounds):return False
    if c["budget"] and any(d["deadline_cap_violations"] for d in decisions):return False
    cf=rows(root/"counterfactuals.jsonl")
    if set(x["round"] for x in cf)!=set(c["counterfactual_rounds"]):return False
    if c["all_subsets"]:
        from itertools import combinations
        expected={s for k in range(n+1) for s in combinations(range(n),k)}
        if len(cf)!=len(c["counterfactual_rounds"])*2**n or any({tuple(x["subset"]) for x in cf if x["round"]==r}!=expected for r in c["counterfactual_rounds"]):return False
    audit=json.loads((root/"analysis/ground_truth_comparison.json").read_text())
    if audit["ground_truth_hash"]!=receipt["ground_truth_hash"] or len(audit["rounds"])!=c["rounds"] or set(x["round"] for x in audit["counterfactuals"])!=set(c["counterfactual_rounds"]):return False
    held=json.loads((root/"heldout_test_metrics.json").read_text())
    if held["evaluation_split"]!="test" or held["ground_truth_hash"]!=receipt["ground_truth_hash"] or sum(x["count"] for x in held["rows"])!=10000 or len(held["rows"])!=5:return False
    if summary["test_final_accuracy"]!=mean(x["accuracy"] for x in held["rows"]) and not math.isclose(summary["test_final_accuracy"],mean(x["accuracy"] for x in held["rows"]),abs_tol=1e-12):return False
    return True

def write_csv(path,items):
    keys=sorted(set().union(*(x.keys() for x in items))) if items else ["empty"]
    with path.open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=keys);w.writeheader();w.writerows(items)

def analyze(ledger,out):
    from run_campaign import valid
    from communication_efficiency import learning_curve,compare_equal_bytes
    jobs=[x for x in rows(ledger) if x["kind"]=="train"]
    if len(jobs)!=len(cases()):raise ValueError("incomplete utility ledger")
    matrix=[];oracles=[];curves={};by_id={}
    out.mkdir(parents=True,exist_ok=True)
    for j in jobs:
        if not valid(j):raise ValueError("unverified "+j["semantic_id"])
        root=Path(j["artifacts"][0]).parent;c=j["config"];s=json.loads((root/"final_summary.json").read_text())
        row={**{k:c[k] for k in ("setting_id","seed","name","stage","mode","policy","k","budget","fresh","clients","source","experiment_id")},**{k:s.get(k) for k in ("final_accuracy","test_final_accuracy","forgetting","worst_task_accuracy","test_worst_task_accuracy","total_bytes","model_bytes","statistics_bytes","control_bytes","optimizer_steps","processed_samples","replay_samples","mean_participants","deadline_cap_violations","ground_truth_hash")}}
        matrix.append(row);by_id[c["setting_id"],c["seed"],c["name"]]=row
        curves[c["experiment_id"]]=learning_curve(rows(root/"server_metrics.jsonl"),rows(root/"communication_rounds.jsonl"),5,50)
        decisions=rows(root/"decisions.jsonl");cf=rows(root/"counterfactuals.jsonl")
        for r in c["counterfactual_rounds"]:
            group=[x for x in cf if x["round"]==r];a=next(x for x in group if x["actual"]);matched=[x for x in group if x["size"]==a["size"]]
            obj=lambda x:x["current_loss"]+(x["past_loss"] or 0.)
            oracles.append({"experiment_id":c["experiment_id"],"round":r,"k":a["size"],"actual_loss":obj(a),"random_subset_mean_loss":mean(map(obj,matched)),"oracle_loss":min(map(obj,matched)),"regret":obj(a)-min(map(obj,matched)),"gain_over_uniform":mean(map(obj,matched))-obj(a),"decision_reason":decisions[r].get("utility",{}).get("reason"),"continuation_accuracy":a.get("continuation",{}).get("final_accuracy")})
    pairs=[]
    with (out/"learning_vs_bytes.jsonl").open("w") as h,(out/"equal_bytes.jsonl").open("w") as e:
        for x in matrix:
            ident=x["experiment_id"]
            for point in curves[ident]:h.write(json.dumps({"experiment_id":ident,**point})+"\n")
            key=(x["setting_id"],x["seed"])
            refs=[]
            if x["mode"] not in ("random","off") or x["name"].startswith(("fresh-","stale-")):
                ref="random-b"+str(int(x["budget"]*100)) if x["budget"] else "random" if x["fresh"] else "stale-random"
                refs.append(ref)
            if x["name"]=="server-learned":refs.append("server")
            if x["name"].startswith("delta-"):
                refs.extend(["stale-"+x["name"][6:],"random" if x["name"]=="delta-random" else "fresh-"+x["name"][6:]])
            if x["name"].startswith("fresh-") and (*key,x["name"].replace("fresh-","stale-")) in by_id:refs.append(x["name"].replace("fresh-","stale-"))
            if x["name"]=="random" and (*key,"stale-random") in by_id:refs.append("stale-random")
            if x["budget"] and x["mode"]!="random":refs.append(x["name"]+"-count")
            if (*key,"full") in by_id and x["name"]!="full":refs.append("full")
            for ref in dict.fromkeys(refs):
                b=by_id.get((*key,ref))
                if not b:continue
                if b["ground_truth_hash"]!=x["ground_truth_hash"]:raise ValueError("paired stream mismatch")
                p={"experiment_id":ident,"reference":b["experiment_id"],"setting_id":key[0],"seed":key[1]}
                for metric in ("final_accuracy","test_final_accuracy","forgetting","total_bytes","statistics_bytes","optimizer_steps","processed_samples"):p[metric+"_delta"]=x[metric]-b[metric]
                pairs.append(p)
                for point in compare_equal_bytes(curves[ident],curves[b["experiment_id"]]):e.write(json.dumps({**p,**point})+"\n")
    write_csv(out/"matrix.csv",matrix);write_csv(out/"paired_comparisons.csv",pairs);write_csv(out/"oracle_diagnostics.csv",oracles)
    (out/"matrix.json").write_text(json.dumps({"training_runs":len(matrix),"results":matrix},indent=2,allow_nan=False))
    report=["# Utility follow-up: measured results","",f"Verified trajectories: {len(matrix)}.","","Validation drives method development; final test metrics are descriptive, not used to choose methods.","","## Paired comparisons","","| Method | Reference | Validation accuracy delta | Bytes delta | Forgetting delta |","|---|---|---:|---:|---:|"]
    report.extend(f'| {p["experiment_id"]} | {p["reference"]} | {p["final_accuracy_delta"]:+.4f} | {p["total_bytes_delta"]:+,} | {p["forgetting_delta"]:+.4f} |' for p in pairs)
    report+=["","Analytical utility is a proxy; no calibrated uncertainty guarantee. Logical bytes are not physical throughput. Same participant counts need not match samples. Archive data informed the design; confirmation settings and seed 43 are prespecified."]
    (out/"report.md").write_text("\n".join(report)+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--analyze",action="store_true");p.add_argument("--ledger",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    analyze(a.ledger,a.output) if a.analyze else build(a.ledger)
