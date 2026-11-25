"""Researcher-side bundle verification and offline evaluation only."""
import hashlib,json,shutil,math
from pathlib import Path
from itertools import combinations
import numpy as np

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False));tmp.replace(path)
def verify(root,receipt):
    files=receipt["ground_truth_files"]
    if set(files)!={"distributions.npz","specification.json"}:raise ValueError("ground truth inventory")
    if not all(sha(root/name)==v for name,v in files.items()):raise ValueError("ground truth hash mismatch")
    if hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()!=receipt["ground_truth_hash"]:raise ValueError("bundle hash mismatch")
    return True
def install_bundle(dataset,out,manifest):
    if not manifest or not manifest.get("controlled_stream"):raise ValueError("controlled manifest required")
    receipt=json.loads((dataset/"validation_receipt.json").read_text())
    if not receipt["valid"] or receipt["manifest_id"]!=manifest["manifest_id"]:raise ValueError("unvalidated dataset")
    verify(dataset/"ground_truth",receipt)
    dest=out/"ground_truth";dest.mkdir(exist_ok=True)
    for name in receipt["ground_truth_files"]:
        shutil.copyfile(dataset/"ground_truth"/name,dest/name)
    write(dest/"dataset_manifest.json",manifest);write(dest/"validation_receipt.json",receipt)
    verify(dest,receipt)
    return receipt
def tv(x,y):return float(.5*np.abs(x-y).sum())
def source_round(report):
    if report["kind"]=="tier_a_cached_update":return int(report["window"].split(":")[1])
    return report["round"]
def coverage_error(dist,subset):
    return float(np.min(dist[:,subset],axis=1).mean())
def subset_metrics(p,dist,selected,weights):
    ids=list(range(len(p)));chosen=list(selected)
    if not chosen:return None
    w=np.array(weights,float);w/=w.sum()
    pooled=(p*w[:,None]).sum(0)
    def mass(s):
        v=w[list(s)];v=v/v.sum()
        return tv(pooled,(p[list(s)]*v[:,None]).sum(0))
    allsets=list(combinations(ids,len(chosen)))
    actual=coverage_error(dist,chosen);actual_mass=mass(chosen)
    oracle=min(coverage_error(dist,list(s)) for s in allsets)
    best_mass=min(mass(s) for s in allsets)
    return {"coverage":actual,"coverage_oracle":oracle,"coverage_regret":max(0.,actual-oracle),
            "mass_error":actual_mass,"mass_oracle":best_mass,"mass_regret":max(0.,actual_mass-best_mass)}
def corr(rows):
    if len(rows)<2:return {"n":len(rows),"spearman":None,"kendall":None}
    from scipy.stats import spearmanr,kendalltau
    x,y=np.asarray(rows).T
    if len(set(x))<2 or len(set(y))<2:return {"n":len(rows),"spearman":None,"kendall":None}
    return {"n":len(rows),"spearman":float(spearmanr(x,y).statistic),"kendall":float(kendalltau(x,y).statistic)}
def analyze_run(out,rounds,decisions,metrics,counterfactuals):
    root=Path(out)/"ground_truth";receipt=json.loads((root/"validation_receipt.json").read_text());verify(root,receipt)
    meta=json.loads((root/"specification.json").read_text())
    with np.load(root/"distributions.npz") as z:counts=z["counts"];p=z["realized_labels"];q=z["realized_joint"].reshape(*p.shape[:2],-1);joint=z["spatial_joint"];label=z["spatial_label"]
    if len(rounds)!=len(p) or len(decisions)!=len(p):raise ValueError("incomplete ground truth trajectory")
    index={report["id"]:report for row in rounds for report in row["sampled_reports"].values() if report is not None}
    by_round={};observations=[];comparisons=[];regimes={}
    for row,decision,phase in zip(rounds,decisions,meta["schedule"]):
        r=row["round"];task=row["task"];n=len(p[r]);c=counts.shape[-1]//(len(p)//meta["specification"]["windows"])
        for i,h in row["population_histograms"].items():
            if not np.array_equal(h,counts[r,int(i),task*c:(task+1)*c]):raise ValueError(f"actual supplied distribution mismatch {r}:{i}")
        weights=counts[r].sum(-1);selected=row["accepted"]
        item={"round":r,"task":task,"regime":phase["regime"],"selected":selected,
              "label":subset_metrics(p[r],1-label[r],selected,weights),"joint":subset_metrics(q[r],1-joint[r],selected,weights),
              "valid_reports":sum(v is not None for v in row["sampled_reports"].values()),"available":len(row["available"]),
              "counterfactual_seconds":row["counterfactual_seconds"]}
        by_round[r]=item;regimes.setdefault(phase["regime"],[]).append(item)
        def add(i,j,estimate,pr,qr,reference_p,reference_q,kind,source):
            if estimate is None:return
            comparisons.append({"round":r,"regime":phase["regime"],"client":i,"reference":j,"kind":kind,
                                "estimated_distance":float(estimate),"label_tv":tv(pr,reference_p),"joint_tv":tv(qr,reference_q),"source":source})
        for i,report in row["sampled_reports"].items():
            i=int(i)
            if report is None:continue
            sr=source_round(report)
            if not 0<=sr<=r:raise ValueError("future report")
            observations.append({"round":r,"client":i,"source_round":sr,"model_version":report["model_version"],"report_id":report["id"],"observation_window":row["observation_window"]})
            score=decision["scores"].get(str(i),decision["scores"].get(i))
            ref=decision["references"].get(str(i),decision["references"].get(i))
            members=decision.get("reference_sources",{}).get(str(i),decision.get("reference_sources",{}).get(i,[]))
            if score is not None and isinstance(ref,str) and ref in index:
                other=index[ref];u=source_round(other);j=other["client"]
                add(i,j,score,p[sr,i],q[sr,i],p[u,j],q[u,j],"local",ref)
            elif score is not None and members:
                w=np.array([m["sample_count"] for m in members],float);w/=w.sum()
                mp=sum(a*p[m["source_round"],m["client"]] for a,m in zip(w,members))
                mq=sum(a*q[m["source_round"],m["client"]] for a,m in zip(w,members))
                add(i,-1,score,p[sr,i],q[sr,i],mp,mq,"server_sample_weight_analogue",members)
        for pair in decision.get("peer_distances",[]):
            i,j=pair["clients"]
            a=row["sampled_reports"].get(str(i),row["sampled_reports"].get(i));b=row["sampled_reports"].get(str(j),row["sampled_reports"].get(j))
            if a is not None and b is not None:
                u,v=source_round(a),source_round(b)
                add(i,j,pair["distance"],p[u,i],q[u,i],p[v,j],q[v,j],"peer",[u,v])
    cf=[]
    for r in meta["specification"]["counterfactual_rounds"]:
        items=[x for x in counterfactuals if x["round"]==r]
        actual=[x for x in items if x["actual"]];full=[x for x in items if x["size"]==len(p[r])]
        k=len(by_round[r]["selected"]);matched=[x for x in items if x["size"]==k]
        if len(actual)!=1 or len(full)!=1 or len(matched)!=math.comb(len(p[r]),k):raise ValueError("incomplete counterfactual subsets")
        a=actual[0]
        cf.append({"round":r,"regime":by_round[r]["regime"],"current_harm":a["current_harm"],"past_harm":a["past_harm"],
                   "current_regret":a["current_loss"]-min(x["current_loss"] for x in matched),
                   "past_regret":None if a["past_loss"] is None else a["past_loss"]-min(x["past_loss"] for x in matched),
                   "current_rank":1+sum(x["current_loss"]<a["current_loss"] for x in matched),
                   "past_rank":None if a["past_loss"] is None else 1+sum(x["past_loss"]<a["past_loss"] for x in matched)})
    def mean(x):return float(np.mean(x)) if x else None
    def summarize(items):
        rr={x["round"] for x in items};comp=[x for x in comparisons if x["round"] in rr]
        return {"rounds":len(items),"mean_label_coverage_regret":mean([x["label"]["coverage_regret"] for x in items]),
                "mean_joint_coverage_regret":mean([x["joint"]["coverage_regret"] for x in items]),
                "mean_joint_mass_regret":mean([x["joint"]["mass_regret"] for x in items]),
                "missing_report_fraction":1-sum(x["valid_reports"] for x in items)/sum(x["available"] for x in items),
                "mean_current_accuracy":mean([x["accuracy"] for x in metrics if x["round"] in rr and x["task"]==rounds[x["round"]]["task"]]),
                "label_rank_agreement":corr([(x["estimated_distance"],x["label_tv"]) for x in comp]),
                "joint_rank_agreement":corr([(x["estimated_distance"],x["joint_tv"]) for x in comp])}
    summary=summarize(list(by_round.values()))
    analysis=Path(out)/"analysis";analysis.mkdir(exist_ok=True)
    write(analysis/"ground_truth_comparison.json",{"ground_truth_hash":receipt["ground_truth_hash"],"summary":summary,
        "regimes":{k:summarize(v) for k,v in regimes.items()},"rounds":list(by_round.values()),"comparisons":comparisons,"observations":observations,"counterfactuals":cf})
    return {"ground_truth_hash":receipt["ground_truth_hash"],"ground_truth_audit":summary,
            "counterfactual_checkpoints":len(cf),"counterfactual_seconds":sum(x["counterfactual_seconds"] for x in by_round.values())}
