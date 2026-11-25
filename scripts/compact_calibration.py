"""Validation-only prefix calibration and fixed-state threshold calibration."""
import argparse,json,math
from pathlib import Path
from itertools import combinations
from statistics import mean
def read_rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]
def prefix_valid(job):
    from run_campaign import verify_controlled_source
    try:
        root=Path(job["artifacts"][0]).parent;c=job["config"]
        status=json.loads((root/"status.json").read_text());cfg=json.loads((root/"config.json").read_text())
        rows=read_rows(root/"rounds.jsonl");metrics=read_rows(root/"server_metrics.jsonl")
        receipt=json.loads((Path(c["data_root"])/"validation_receipt.json").read_text())
        return (status=={"status":"paused","last_committed_round":19} and cfg["source_commit"]==job["source_commit"]
            and cfg["local_learning_rate"]==c["learning_rate"] and cfg["sync_policy"]=="full" and cfg["sync_eval_split"]=="validation"
            and cfg["seed"]==42 and cfg["num_clients"]==5 and cfg["local_epochs"]==5 and cfg["memory_size"]==500 and cfg["batch_size"]==13 and cfg["memory_batch_size"]==13 and cfg["ground_truth_hash"]==receipt["ground_truth_hash"] and receipt["source_verified"]
            and verify_controlled_source(c["data_root"],c) and [x["round"] for x in rows]==list(range(20))
            and all(x["selected"]==list(range(5)) and x["accepted"]==x["downloaded"]==x["selected"] for x in rows)
            and len(metrics)==30 and {(x["round"],x["task"]) for x in metrics}=={(r,t) for r in range(20) for t in range(r//10+1)}
            and all(math.isfinite(x["accuracy"]) and math.isfinite(x["loss"]) for x in metrics))
    except (OSError,KeyError,ValueError,TypeError,IndexError):return False
def choose_lr(jobs):
    from run_campaign import valid,sha_file
    if len(jobs)!=3 or {j["config"]["learning_rate"] for j in jobs}!={.005,.01,.02}:raise ValueError("calibration candidate inventory")
    candidates=[]
    for j in jobs:
        if not valid(j):raise ValueError("unverified calibration prefix")
        root=Path(j["artifacts"][0]).parent;rows=read_rows(root/"server_metrics.jsonl")
        current=mean(x["accuracy"] for x in rows if x["task"]==x["round"]//10)
        past=mean(x["accuracy"] for x in rows if x["task"]<x["round"]//10)
        forgetting=max(x["accuracy"] for x in rows if x["task"]==0)-next(x["accuracy"] for x in rows if x["round"]==19 and x["task"]==0)
        candidates.append({"learning_rate":j["config"]["learning_rate"],"current_accuracy":current,"past_accuracy":past,"objective":(current+past)/2,"forgetting":forgetting,"validation_sha256":sha_file(root/"server_metrics.jsonl")})
    winner=min(candidates,key=lambda x:(-x["objective"],x["forgetting"],x["learning_rate"]))
    return {"schema":1,"source_commit":jobs[0]["source_commit"],"prefix_rounds":20,"learning_rate":winner["learning_rate"],"candidates":sorted(candidates,key=lambda x:x["learning_rate"]),"scope":"validation-only full participation, first two tasks, no test selection"}
def calibration_valid(job):
    try:return json.loads(Path(job["artifacts"][0]).read_text())==choose_lr(job["config"]["candidates"])
    except (OSError,ValueError,KeyError,TypeError,IndexError):return False
def calibrated_thresholds(root):
    import sys
    system=str(Path(__file__).resolve().parents[1]/"system")
    if system not in sys.path:sys.path.insert(0,system)
    from flcore.sync.decision_audit import report
    from flcore.sync.policies import novelty_admit,peer_select,threshold_value
    cfg=json.loads((root/"config.json").read_text());rounds=read_rows(root/"rounds.jsonl");decisions=read_rows(root/"decisions.jsonl")
    ids=tuple(range(cfg["num_clients"]));states=[]
    for r in range(1,len(rounds)):
        row=rounds[r];d=decisions[r];last={i:rounds[r-1]["last_complete_sync"][str(i)] for i in ids}
        reports={int(i):report(x) for i,x in row["sampled_reports"].items()}
        reports={i:x if x is not None and x.client==i and x.round==r and x.task==row["task"] and x.valid() else None for i,x in reports.items()}
        states.append((last,tuple(d["due"]),reports,{int(i):v for i,v in d["scores"].items()}))
    def expected(t):
        vals=[]
        for last,due,reports,scores in states:
            if cfg["sync_policy"]=="peer":s=peer_select(ids,last,due,reports,threshold_value(t),None,cfg["sync_peer_solver"],cfg["sync_peer_objective"],cfg["sync_coverage_weight"]!="equal")
            else:s=novelty_admit(ids,last,due,scores,threshold_value(t),None)
            vals.append(len(s))
        return mean(vals)
    nominal=(.05,.1,.2,.35);grid=sorted(set([i/100 for i in range(101)]+list(nominal)))
    means={t:expected(t) for t in grid}
    if all(means[a]-means[b]>=.25 for a,b in zip(nominal,nominal[1:])):chosen=nominal
    else:
        # Greedy target matching over increasing grid indices, with four
        # distinct thresholds. Attainable participation counts may coincide.
        chosen=[];start=0
        for index,target in enumerate((4,3,2,1)):
            stop=len(grid)-(3-index)
            choices=list(range(start,stop))
            if chosen:
                separated=[j for j in choices if means[chosen[-1]]-means[grid[j]]>=.25]
                if separated:choices=separated
            j=min(choices,key=lambda j:(abs(means[grid[j]]-target),abs(grid[j]-nominal[index]),j))
            chosen.append(grid[j]);start=j+1
    return {str(t):v for t,v in zip(nominal,chosen)},{"nominal":list(nominal),"chosen":list(chosen),"predicted_mean_counts":[means[t] for t in chosen],"distinct_counts":len(set(means[t] for t in chosen)),"grid":[{"threshold":t,"mean_count":means[t]} for t in grid],"scope":"fixed actual screening states, excludes bootstrap; not adaptive trajectory"}
def scale_comparisons(rows,deltas):
    by={(r["setting_id"],r["k"],r["selector_id"]):r for r in rows if r.get("selection_mode")=="fixed"}
    delta={r["experiment_id"]:r for r in deltas};out=[]
    for r in rows:
        if r["setting_id"]!="S14N10" or r.get("selection_mode")!="fixed" or r["k"]==10:continue
        for mode,k5 in (("same_fraction",{6:3,8:4}.get(r["k"])),("same_omitted_count",{9:4,8:3,6:1}.get(r["k"]))):
            if k5 is None:continue
            b=by.get(("S14",k5,r["selector_id"]))
            if b is None:continue
            a1=delta[r["experiment_id"]];b1=delta[b["experiment_id"]]
            item={"comparison":mode,"selector_id":r["selector_id"],"five_client_id":b["experiment_id"],"ten_client_id":r["experiment_id"],"five_participation":k5/5,"ten_participation":r["k"]/10,"five_omitted":5-k5,"ten_omitted":10-r["k"]}
            for key in ("final_accuracy_delta_vs_full","forgetting_delta_vs_full","byte_savings_fraction_vs_full","total_bytes_delta_vs_full"):
                item["five_"+key]=b1.get(key);item["ten_"+key]=a1.get(key)
            out.append(item)
    for r in rows:
        if r["setting_id"]!="S14N10" or r.get("selection_mode")!="threshold":continue
        b=next((x for x in rows if x["setting_id"]=="S14" and x.get("selection_mode")=="threshold" and x.get("threshold_slot")==r.get("threshold_slot") and x["selector_id"]==r["selector_id"]),None)
        if b is None:raise ValueError("missing paired adaptive threshold")
        a1=delta[r["experiment_id"]];b1=delta[b["experiment_id"]]
        item={"comparison":"same_threshold","selector_id":r["selector_id"],"five_client_id":b["experiment_id"],"ten_client_id":r["experiment_id"],"five_participation":b["participation_fraction"],"ten_participation":r["participation_fraction"],"threshold":r["threshold"]}
        for key in ("final_accuracy_delta_vs_full","forgetting_delta_vs_full","byte_savings_fraction_vs_full","total_bytes_delta_vs_full","final_accuracy_delta_vs_matched_byte_random","final_accuracy_delta_vs_matched_count_random"):
            item["five_"+key]=b1.get(key);item["ten_"+key]=a1.get(key)
        out.append(item)
    return out
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--ledger",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    receipt=choose_lr([j for j in read_rows(a.ledger) if j["kind"]=="calibration"])
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix(".tmp");tmp.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False));tmp.replace(a.output)
