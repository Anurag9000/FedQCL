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
    if set(files)!={"distributions.npz","specification.json","allocation_index.npz"}:raise ValueError("ground truth inventory")
    if not all(sha(root/name)==v for name,v in files.items()):raise ValueError("ground truth hash mismatch")
    if hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()!=receipt["ground_truth_hash"]:raise ValueError("bundle hash mismatch")
    return True
def install_bundle(dataset,out,manifest):
    if not manifest or not manifest.get("controlled_stream"):raise ValueError("controlled manifest required")
    receipt=json.loads((dataset/"validation_receipt.json").read_text())
    if not receipt["valid"] or receipt["manifest_id"]!=manifest["manifest_id"]:raise ValueError("unvalidated dataset")
    verify(dataset/"ground_truth",receipt)
    dest=out/"ground_truth";dest.mkdir(exist_ok=True,mode=0o700);dest.chmod(0o700)
    for name in receipt["ground_truth_files"]:
        shutil.copyfile(dataset/"ground_truth"/name,dest/name)
    write(dest/"dataset_manifest.json",manifest);write(dest/"validation_receipt.json",receipt)
    for path in dest.iterdir():path.chmod(0o600)
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
def analyze_run(out,rounds,decisions,metrics,counterfactuals,events=()):
    root=Path(out)/"ground_truth";receipt=json.loads((root/"validation_receipt.json").read_text());verify(root,receipt)
    meta=json.loads((root/"specification.json").read_text())
    with np.load(root/"distributions.npz") as z:
        counts=z["counts"];p=z["realized_labels"];q=z["realized_joint"].reshape(*p.shape[:2],-1);joint=z["spatial_joint"];label=z["spatial_label"];conditional=z["spatial_conditional"]
    from .statistics import js
    from .tier_a_reports import blocks
    from .policies import Report
    if len(rounds)!=len(p) or len(decisions)!=len(p):raise ValueError("incomplete ground truth trajectory")
    index={report["id"]:report for row in rounds for report in row["sampled_reports"].values() if report is not None}
    by_round={};observations=[];comparisons=[];regimes={};replay={};transitions=[]
    def vector(x):
        x=np.asarray(x,dtype=float);return x/x.sum() if x.sum() else None
    def mapping(d,k,default=None):return d.get(k,d.get(str(k),default))
    c=counts.shape[-1]//meta["specification"]["tasks"]
    for row in rounds:
        for i in range(p.shape[1]):
            v=np.zeros(counts.shape[-1]);history=mapping(row["pre_memory_class_counts"],i,{})
            for k,h in history.items():v[int(k)*c:(int(k)+1)*c]=h
            replay[row["round"],i]=v
    event_bytes={}
    for e in events:event_bytes[e["round"]]=event_bytes.get(e["round"],0)+e["serialized_bytes"]
    for row,decision,phase in zip(rounds,decisions,meta["schedule"]):
        r=row["round"];task=row["task"];n=len(p[r]);c=counts.shape[-1]//(len(p)//meta["specification"]["windows"])
        for i,h in row["population_histograms"].items():
            if not np.array_equal(h,counts[r,int(i),task*c:(task+1)*c]):raise ValueError(f"actual supplied distribution mismatch {r}:{i}")
        weights=counts[r].sum(-1);selected=row["accepted"]
        if len(row["available"])!=n or len(row["population_histograms"])!=n:raise ValueError("controlled run missing clients")
        for o in row["outcomes"]:
            i=o["client"];raw=counts[r,i,task*c:(task+1)*c]
            if o["samples"]<=0 or o["samples"]%weights[i] or not np.array_equal(np.asarray(o["training_class_counts"]),raw*(o["samples"]//weights[i])):raise ValueError("trained class counts mismatch")
        rp=[vector(replay[r,i]) for i in range(n)]
        hist_cov=None
        if all(x is not None for x in rp):
            rv=np.asarray(rp);rd=.5*np.abs(rv[:,None]-rv[None,:]).sum(-1)
            hist_cov=subset_metrics(rv,rd,selected,[replay[r,i].sum() for i in range(n)])
        cond=1-conditional[r]
        # Undefined conditional comparisons are excluded, never fabricated.
        cond_cov=None
        if np.isfinite(cond).all():
            actual=coverage_error(cond,selected);best=min(coverage_error(cond,list(s)) for s in combinations(range(n),len(selected)))
            cond_cov={"coverage":actual,"coverage_oracle":best,"coverage_regret":max(0.,actual-best)}
        item={"round":r,"task":task,"regime":phase["regime"],"selected":selected,
              "label":subset_metrics(p[r],1-label[r],selected,weights),"joint":subset_metrics(q[r],1-joint[r],selected,weights),
              "conditional":cond_cov,"replay_label":hist_cov,
              "bytes":event_bytes.get(r,0),"optimizer_steps":sum(o["optimizer_steps"] for o in row["outcomes"]),
              "processed_samples":sum(o["samples"] for o in row["outcomes"]),"replay_samples":sum(o["replay_samples"] for o in row["outcomes"]),
              "valid_reports":sum(v is not None for v in row["sampled_reports"].values()),"available":len(row["available"]),
              "counterfactual_seconds":row["counterfactual_seconds"]}
        by_round[r]=item;regimes.setdefault(phase["regime"],[]).append(item)
        if r and task==rounds[r-1]["task"]:
            for i in range(n):
                change=tv(p[r,i],p[r-1,i]);joint_change=tv(q[r,i],q[r-1,i])
                transitions.append({"round":r,"client":i,"regime":phase["regime"],"previous_regime":meta["schedule"][r-1]["regime"],"label_change":change,"joint_change":joint_change,"selected":i in selected,"score":mapping(decision["scores"],i),"reason":mapping(decision["reasons"],i),"explicit_reassignment":phase["transition"]})
        def add(i,j,estimate,pr,qr,reference_p,reference_q,kind,source,old_a=None,old_b=None):
            if estimate is None:return
            comparisons.append({"round":r,"regime":phase["regime"],"client":i,"reference":j,"kind":kind,
                                "estimated_distance":float(estimate),"label_tv":tv(pr,reference_p),"label_js":js(pr,reference_p),"joint_tv":tv(qr,reference_q),
                                "replay_label_tv":tv(old_a,old_b) if old_a is not None and old_b is not None else None,"source":source})
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
                add(i,j,score,p[sr,i],q[sr,i],p[u,j],q[u,j],"local",ref,vector(replay[sr,i]),vector(replay[u,j]))
            elif score is not None and members:
                w=np.array([m["sample_count"] for m in members],float)
                # For A06, use the exact current-channel support and past-channel
                # support of the delivered report, instead of total report mass.
                past_mix=None
                if report["kind"]=="tier_a_distributions":
                    w=[];old=[];oldw=[]
                    for m in members:
                        source_report=index[m["id"]]
                        br=Report(**{k:source_report[k] for k in Report.__dataclass_fields__})
                        b={x[0]:x for x in blocks(br)}
                        w.append(b["current_labels"][-2]);weight=b["historical_labels"][-2]
                        v=vector(replay[m["source_round"],m["client"]])
                        if weight and v is not None:old.append(v);oldw.append(weight)
                    if old:past_mix=np.average(old,axis=0,weights=oldw)
                w=np.asarray(w,float);w/=w.sum()
                mp=sum(a*p[m["source_round"],m["client"]] for a,m in zip(w,members))
                mq=sum(a*q[m["source_round"],m["client"]] for a,m in zip(w,members))
                add(i,-1,score,p[sr,i],q[sr,i],mp,mq,"server_current_support" if report["kind"]=="tier_a_distributions" else "server_sample_weight_analogue",members,vector(replay[sr,i]),past_mix)
        for pair in decision.get("peer_distances",[]):
            i,j=pair["clients"]
            a=row["sampled_reports"].get(str(i),row["sampled_reports"].get(i));b=row["sampled_reports"].get(str(j),row["sampled_reports"].get(j))
            if a is not None and b is not None:
                u,v=source_round(a),source_round(b)
                add(i,j,pair["distance"],p[u,i],q[u,i],p[v,j],q[v,j],"peer",[u,v],vector(replay[u,i]),vector(replay[v,j]))
    cf=[]
    for r in meta["specification"]["counterfactual_rounds"]:
        items=[x for x in counterfactuals if x["round"]==r]
        actual=[x for x in items if x["actual"]];full=[x for x in items if x["size"]==len(p[r])]
        k=len(by_round[r]["selected"]);matched=[x for x in items if x["size"]==k]
        if len(actual)!=1 or len(full)!=1 or len(matched)!=math.comb(len(p[r]),k):raise ValueError("incomplete counterfactual subsets")
        a=actual[0]
        for expected in a["task_metrics"]:
            observed=next(x for x in metrics if x["round"]==r and x["task"]==expected["task"])
            if not math.isclose(observed["loss"],expected["loss"],rel_tol=1e-5,abs_tol=1e-6) or not math.isclose(observed["accuracy"],expected["accuracy"],abs_tol=1e-6):raise ValueError("actual subset counterfactual differs from real trajectory")
        cf.append({"round":r,"regime":by_round[r]["regime"],"current_harm":a["current_harm"],"past_harm":a["past_harm"],
                   "current_regret":a["current_loss"]-min(x["current_loss"] for x in matched),
                   "past_regret":None if a["past_loss"] is None else a["past_loss"]-min(x["past_loss"] for x in matched),
                   "update_discrepancy":a["update_discrepancy"],"candidate_update_geometry":full[0]["candidate_update_geometry"],
                   "current_rank":1+sum(x["current_loss"]<a["current_loss"] for x in matched),
                   "past_rank":None if a["past_loss"] is None else 1+sum(x["past_loss"]<a["past_loss"] for x in matched)})
    def mean(x):return float(np.mean(x)) if x else None
    def summarize(items):
        rr={x["round"] for x in items};comp=[x for x in comparisons if x["round"] in rr]
        return {"rounds":len(items),"mean_label_coverage_regret":mean([x["label"]["coverage_regret"] for x in items]),
                "mean_joint_coverage_regret":mean([x["joint"]["coverage_regret"] for x in items]),
                "mean_joint_mass_regret":mean([x["joint"]["mass_regret"] for x in items]),
                "mean_conditional_coverage_regret":mean([x["conditional"]["coverage_regret"] for x in items if x["conditional"] is not None]),
                "mean_replay_coverage_regret":mean([x["replay_label"]["coverage_regret"] for x in items if x["replay_label"] is not None]),
                "communication_bytes":sum(x["bytes"] for x in items),"optimizer_steps":sum(x["optimizer_steps"] for x in items),
                "processed_samples":sum(x["processed_samples"] for x in items),"replay_samples":sum(x["replay_samples"] for x in items),
                "mean_past_accuracy":mean([x["accuracy"] for x in metrics if x["round"] in rr and x["task"]<rounds[x["round"]]["task"]]),
                "missing_report_fraction":1-sum(x["valid_reports"] for x in items)/sum(x["available"] for x in items),
                "mean_current_accuracy":mean([x["accuracy"] for x in metrics if x["round"] in rr and x["task"]==rounds[x["round"]]["task"]]),
                "label_rank_agreement":corr([(x["estimated_distance"],x["label_tv"]) for x in comp]),
                "joint_rank_agreement":corr([(x["estimated_distance"],x["joint_tv"]) for x in comp]),
                "replay_rank_agreement":corr([(x["estimated_distance"],x["replay_label_tv"]) for x in comp if x["replay_label_tv"] is not None])}
    summary=summarize(list(by_round.values()))
    analysis=Path(out)/"analysis";analysis.mkdir(exist_ok=True)
    write(analysis/"ground_truth_comparison.json",{"ground_truth_hash":receipt["ground_truth_hash"],"summary":summary,
        "regimes":{k:summarize(v) for k,v in regimes.items()},"rounds":list(by_round.values()),"comparisons":comparisons,"observations":observations,"counterfactuals":cf,"transitions":transitions})
    return {"ground_truth_hash":receipt["ground_truth_hash"],"ground_truth_audit":summary,
            "counterfactual_checkpoints":len(cf),"counterfactual_seconds":sum(x["counterfactual_seconds"] for x in by_round.values())}
