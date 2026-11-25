"""Research-only diagnostics; fixed recorded states, never an alternate trajectory."""
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from run_campaign import valid
from communication_efficiency import read_rows

def subsets(n):
    return [s for k in range(n+1) for s in combinations(range(n),k)]

def subset_distribution(counts,dist,s):
    if not s:return {"coverage_error":None,"mass_error":None}
    p=counts.sum(0);q=counts[list(s)].sum(0)
    return {"coverage_error":float(np.min(dist[:,s],axis=1).mean()),"mass_error":float(.5*np.abs(p/p.sum()-q/q.sum()).sum())}

def distribution_analysis(jobs,out,truths):
    from analyze_controlled_campaign import truth_stats
    data=[j for j in jobs if j["kind"]=="validate"]
    if len(data)!=24 or {j["config"]["partition_seed"] for j in data}!={42}:raise ValueError("compact dataset inventory")
    trained={j["config"]["setting_id"] for j in jobs if j["kind"]=="train"}
    with (out/"distribution_subset_metrics.jsonl").open("w") as handle:
        for job in data:
            if not valid(job):raise ValueError("unverified dataset "+job["semantic_id"])
            root=Path(job["artifacts"][0]).parent;receipt=json.loads(Path(job["artifacts"][0]).read_text());sid=job["config"]["setting_id"]
            truths[sid,42]={"setting_id":sid,"partition_seed":42,"ground_truth_hash":receipt["ground_truth_hash"],"neural_trajectory":sid in trained,**truth_stats(root/"ground_truth")}
            with np.load(root/"ground_truth/distributions.npz") as z:
                counts=z["counts"];joint=z["joint_counts"].reshape(*counts.shape[:2],-1)
                for r in range(len(counts)):
                    for s in subsets(counts.shape[1]):
                        row={"setting_id":sid,"seed":42,"round":r,"subset":s,"size":len(s),"label":subset_distribution(counts[r],1-z["spatial_label"][r],s),"joint":subset_distribution(joint[r],1-z["spatial_joint"][r],s)}
                        handle.write(json.dumps(row,allow_nan=False)+"\n")

def run_diagnostics(root,c,common,subset_file,sensitivity_file):
    import sys
    system=str(Path(__file__).resolve().parents[1]/"system")
    if system not in sys.path:sys.path.insert(0,system)
    from flcore.sync.decision_audit import report
    from flcore.sync.policies import novelty_admit,peer_select,threshold_value,distance
    raw=read_rows(root/"counterfactuals.jsonl");cf={(x["round"],tuple(x["subset"])):x for x in raw}
    cfg=json.loads((root/"config.json").read_text())
    schema=json.loads((root/"model_wire_schema.json").read_text())
    with np.load(root/"ground_truth/distributions.npz") as z:
        counts=z["counts"];joint=z["joint_counts"].reshape(*counts.shape[:2],-1)
        label_dist=1-z["spatial_label"];joint_dist=1-z["spatial_joint"]
        for x in raw:
            r=x["round"];s=tuple(x["subset"])
            subset_file.write(json.dumps({**common,**x,"label_distribution":subset_distribution(counts[r],label_dist[r],s),"joint_distribution":subset_distribution(joint[r],joint_dist[r],s),"hypothetical_model_tensor_bytes":2*x["size"]*schema["tensor_bytes"],"byte_scope":"model tensors only; actual total logical bytes are in communication logs"},allow_nan=False)+"\n")
    if c["mechanism"]=="baseline":return
    rounds=read_rows(root/"rounds.jsonl");decisions=read_rows(root/"decisions.jsonl")
    ids=tuple(range(c["clients"]))
    for r,(row,d) in enumerate(zip(rounds,decisions)):
        last={i:-1 if r==0 else rounds[r-1]["last_complete_sync"][str(i)] for i in ids}
        due=tuple(d["due"]);reports={int(i):report(x) for i,x in row["sampled_reports"].items()}
        reports={i:x if x is not None and x.client==i and x.round==r and x.task==row["task"] and x.valid() else None for i,x in reports.items()}
        scores={int(i):v for i,v in d["scores"].items()}
        # Tier-A distances are bounded. Evaluate exact breakpoints as well as the four campaign thresholds.
        ds=list(scores.values()) if c["policy"]!="peer" else [distance(reports[i],reports[j]) for i,j in combinations(ids,2)]
        levels=sorted({0.,1.,.05,.1,.2,.35,*[float(v) for v in ds if v is not None]})
        def choose(t,k):
            if r==0:return ids
            if c["policy"]=="peer":return peer_select(ids,last,due,reports,threshold_value(t),k,cfg["sync_peer_solver"],cfg["sync_peer_objective"],cfg["sync_coverage_weight"]!="equal")
            return novelty_admit(ids,last,due,scores,threshold_value(t),k)
        for mode,level in [("threshold",t) for t in levels]+[("fixed",k) for k in range(1,len(ids)+1)]:
            s=choose(level,None) if mode=="threshold" else choose(cfg["sync_threshold"],level)
            x=cf.get((r,s));item={**common,"round":r,"sweep_mode":mode,"level":level,"subset":s,"participants":len(s),"deadline_violations":[i for i in due if i not in s],"hypothetical_model_tensor_bytes":2*len(s)*schema["tensor_bytes"],"current_loss":x["current_loss"] if x else None,"past_loss":x["past_loss"] if x else None,"accuracy":x["accuracy"] if x else None,"scope":"one-step on actual historical state; not alternative trajectory or full hypothetical wire cost"}
            sensitivity_file.write(json.dumps(item,allow_nan=False)+"\n")
