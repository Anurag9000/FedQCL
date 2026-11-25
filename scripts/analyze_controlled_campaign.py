#!/usr/bin/env python3
"""Strict matched-stream analysis; one observation per trajectory for seed inference."""
import argparse,csv,json,hashlib,statistics
from pathlib import Path
from collections import defaultdict
from contextlib import ExitStack
import numpy as np
from campaign_matrix import expected_ids,read_catalog,CATALOG,DETAILS
from run_campaign import valid

MEASURES=("final_accuracy","forgetting","worst_task_accuracy","worst_class_accuracy","backward_transfer","retention_auc_mean","acquisition_auc_mean","total_bytes","optimizer_steps","processed_samples","replay_samples","complete_syncs","mean_model_age","jain_sync_fairness","deadline_cap_violations","counterfactual_seconds","model_bytes","statistics_bytes","control_bytes","upload_bytes","download_bytes","peer_bytes","model_tensor_bytes","statistic_vector_bytes","full_model_payload_bytes","model_uploads","model_downloads","initial_downloads","model_refreshes","skipped_client_rounds","no_aggregation_rounds","mean_participants","participation_fraction")
AUDIT=("mean_label_coverage_regret","mean_joint_coverage_regret","mean_joint_mass_regret","mean_conditional_coverage_regret","mean_replay_coverage_regret","mean_selection_delay","shift_precision","shift_recall","mean_past_accuracy","mean_current_accuracy","missing_report_fraction")
def csv_rows(path,rows):
    rows=list(rows)
    with path.open("w",newline="") as handle:
        fields=sorted(set().union(*(x.keys() for x in rows))) if rows else ["empty"]
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def truth_stats(root):
    with np.load(root/"distributions.npz") as z:
        p=z["realized_labels"];q=z["realized_joint"].reshape(*p.shape[:2],-1);n=p.shape[1];i,j=np.triu_indices(n,1)
        h=1-z["spatial_label"][:,i,j];joint=1-z["spatial_joint"][:,i,j]
        spec=json.loads((root/"specification.json").read_text())["specification"];w=spec["windows"];work=z["counts"].sum(-1)
        within=np.array([r for r in range(1,len(p)) if r%w])
        hist=z["counts"][:-w].sum(0);hist=hist/hist.sum(1,keepdims=True)
        old=.5*np.abs(hist[i]-hist[j]).sum(-1)
        return {"spatial_label_tv":float(h.mean()),"spatial_joint_tv":float(joint.mean()),
                "temporal_label_tv":float((.5*np.abs(p[within]-p[within-1]).sum(-1)).mean()),
                "temporal_joint_tv":float((.5*np.abs(q[within]-q[within-1]).sum(-1)).mean()),
                "historical_label_tv_before_last_task":float(old.mean()),"last_task_spatial_label_tv":float(h[-w:].mean()),
                "workload_min":int(work.min()),"workload_max":int(work.max()),"workload_ratio":float(work.max()/work.min()),
                "max_label_error":float(z["label_error"].max()),"max_joint_error":float(z["joint_error"].max()),
                "alpha":spec["alpha"] if spec["setting"]=="alpha" else None}

def flatten(job,summary,audit,truth):
    c=job["config"]
    row={k:c[k] for k in ("experiment_id","selector_id","setting_id","setting_name","partition_seed","training_seed","k","signal","mechanism","policy","stream_spec_sha256")}
    row.update(selection_mode=c.get("selection_mode","fixed"),threshold=c.get("threshold",.1),job_id=job["id"],ground_truth_hash=summary["ground_truth_hash"],**truth)
    for key in MEASURES:row[key]=summary.get(key)
    for key in AUDIT:row[key]=audit["summary"].get(key)
    for key in ("label_rank_agreement","joint_rank_agreement","replay_rank_agreement","update_geometry_rank_agreement","novelty_current_marginal_rank_agreement","novelty_past_marginal_rank_agreement"):
        for field in ("n","spearman","kendall"):row[key+"_"+field]=audit["summary"][key][field]
    for name in ("current_harm","past_harm","current_regret","past_regret"):
        vals=[v[name] for v in audit["counterfactuals"] if v[name] is not None]
        row["counterfactual_mean_"+name]=statistics.mean(vals) if vals else None
    return row

def baseline_deltas(rows):
    fixed=[x for x in rows if x.get("selection_mode","fixed")=="fixed"]
    idx={(x["setting_id"],x["partition_seed"],x["k"],x["selector_id"]):x for x in fixed}
    if len(idx)!=len(fixed):raise ValueError("duplicate trajectory identity")
    result=[]
    for row in rows:
        if row["selector_id"]=="B-FULL":continue
        refs=[("full",idx[row["setting_id"],row["partition_seed"],5,"B-FULL"])]
        if row.get("selection_mode","fixed")=="fixed":refs += [("random",idx[row["setting_id"],row["partition_seed"],row["k"],"B-RANDOM"]),("oldest",idx[row["setting_id"],row["partition_seed"],row["k"],"B-OLDEST"])]
        v={k:row[k] for k in ("experiment_id","selector_id","setting_id","partition_seed","k")}
        v.update(selection_mode=row.get("selection_mode","fixed"),threshold=row.get("threshold",.1))
        for name,base in refs:
            if row["ground_truth_hash"]!=base["ground_truth_hash"]:raise ValueError("baseline stream mismatch")
            for key in ("final_accuracy","forgetting","worst_class_accuracy","backward_transfer","total_bytes","optimizer_steps","mean_joint_coverage_regret","model_bytes","statistics_bytes","control_bytes","upload_bytes","download_bytes","peer_bytes","participation_fraction"):
                a,b=row.get(key),base.get(key)
                v[key+"_delta_vs_"+name]=a-b if a is not None and b is not None else None
        full=refs[0][1]
        v["net_bytes_saved_vs_full"]=full["total_bytes"]-row["total_bytes"]
        parts=("model_bytes","statistics_bytes","control_bytes")
        if all(x.get(key) is not None for x in (full,row) for key in parts):
            v["model_bytes_avoided_vs_full"]=full["model_bytes"]-row["model_bytes"]
            v["extra_statistics_control_bytes_vs_full"]=sum(row[key]-full[key] for key in parts[1:])
            if v["net_bytes_saved_vs_full"]!=v["model_bytes_avoided_vs_full"]-v["extra_statistics_control_bytes_vs_full"]:raise ValueError("net byte savings reconciliation failed")
        v["byte_savings_fraction_vs_full"]=1-row["total_bytes"]/refs[0][1]["total_bytes"] if refs[0][1]["total_bytes"] else None
        result.append(v)
    return result

def seed_summary(rows,deltas):
    grouped=defaultdict(list);paired={r["experiment_id"]:r for r in deltas}
    for r in rows:grouped[r["setting_id"],r.get("selection_mode","fixed"),r.get("threshold",.1),r["k"],r["selector_id"]].append({**r,**paired.get(r["experiment_id"],{})})
    result=[]
    for (setting,mode,tau,k,selector),items in sorted(grouped.items()):
        if {x["partition_seed"] for x in items}!={42,43,44} or len(items)!=3:raise ValueError("incomplete paired seeds")
        row={"setting_id":setting,"selection_mode":mode,"threshold":tau,"k":k,"selector_id":selector,"n_seeds":3}
        keys=[key for key in items[0] if key in MEASURES or key in AUDIT or "_delta_vs_" in key or key in ("byte_savings_fraction_vs_full","net_bytes_saved_vs_full","model_bytes_avoided_vs_full","extra_statistics_control_bytes_vs_full") or key.endswith(("_spearman","_kendall")) or key.startswith("counterfactual_mean_")]
        for key in keys:
            vals=[v[key] for v in items if v.get(key) is not None]
            row[key+"_mean"]=statistics.mean(vals) if vals else None
            row[key+"_sd"]=statistics.stdev(vals) if len(vals)>1 else None
            row[key+"_n"]=len(vals)
        result.append(row)
    return result

def plot(out,summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    settings=[x["id"] for x in read_catalog()["settings"]];selectors=sorted({r["selector_id"] for r in summary if r["selector_id"].startswith("A")})
    idx={(r["setting_id"],r["k"],r["selector_id"]):r["final_accuracy_delta_vs_random_mean"] for r in summary if r["selector_id"].startswith("A") and r["selection_mode"]=="fixed"}
    values=[np.array([[idx[s,k,a] for s in settings] for a in selectors]) for k in (1,2,3,4)]
    scale=max(1e-9,max(float(np.max(np.abs(x))) for x in values))
    fig,axes=plt.subplots(2,2,figsize=(19,16),layout="constrained")
    for k,ax,m in zip((1,2,3,4),axes.ravel(),values):
        im=ax.imshow(m,vmin=-scale,vmax=scale,cmap="RdBu",aspect="auto");ax.set(title=f"k={k}/5: mean accuracy difference vs matched random")
        ax.set_xticks(range(len(settings)),settings,rotation=90);ax.set_yticks(range(len(selectors)),selectors)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7);fig.savefig(out/"effectiveness.png",dpi=150);plt.close(fig)

def plot_threshold(out,rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout="constrained")
    for ax,method in zip(axes,("M1","M2","M3")):
        for tau in (.05,.1,.2,.35):
            points=[x for x in rows if x["threshold"]==tau and x["selector_id"].endswith(method)]
            ax.scatter([x["byte_savings_fraction_vs_full_mean"] for x in points],[x["final_accuracy_delta_vs_full_mean"] for x in points],s=9,alpha=.4,label=str(tau))
        ax.set(title=method,xlabel="Fraction of full-baseline bytes saved",ylabel="Accuracy difference vs full");ax.legend(title="Threshold")
    fig.savefig(out/"threshold_tradeoff.png",dpi=150);plt.close(fig)

def analyze(ledger,out):
    jobs=[json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    train=[j for j in jobs if j["kind"]=="train"]
    expected=expected_ids()
    if len(train)!=len(expected) or {j["config"]["experiment_id"] for j in train}!=expected:raise ValueError("incomplete comprehensive ledger")
    out.mkdir(parents=True,exist_ok=True)
    fields=("regimes","rounds","comparisons","counterfactuals","transitions","geometry_comparisons","omission_comparisons")
    rows=[];truths={};hashes={}
    with ExitStack() as stack:
        files=[stack.enter_context((out/(name+".jsonl")).open("w")) for name in DETAILS]
        for job in train:
            if not valid(job):raise ValueError("unverified run "+job["config"]["experiment_id"])
            root=Path(job["artifacts"][0]).parent;c=job["config"];group=(c["setting_id"],c["partition_seed"])
            summary=json.loads((root/"final_summary.json").read_text());audit=json.loads((root/"analysis/ground_truth_comparison.json").read_text())
            h=summary["ground_truth_hash"]
            if group in hashes and hashes[group]!=h:raise ValueError("mismatched ground truth in "+str(group))
            if group not in hashes:hashes[group]=h;truths[group]={"setting_id":group[0],"partition_seed":group[1],"ground_truth_hash":h,**truth_stats(root/"ground_truth")}
            row=flatten(job,summary,audit,{k:v for k,v in truths[group].items() if k not in ("setting_id","partition_seed","ground_truth_hash")})
            rows.append(row)
            common={k:row[k] for k in ("experiment_id","selector_id","setting_id","partition_seed","k","selection_mode","threshold","signal","mechanism","ground_truth_hash")}
            for field,handle in zip(fields,files):
                values=[{"regime":k,**v} for k,v in audit[field].items()] if field=="regimes" else audit[field]
                for value in values:handle.write(json.dumps({**common,**value},allow_nan=False)+"\n")
    if len(hashes)!=72:raise ValueError("missing matched stream groups")
    deltas=baseline_deltas(rows);summary=seed_summary(rows,deltas)
    csv_rows(out/"matrix.csv",rows);csv_rows(out/"matched_baseline_deltas.csv",deltas);csv_rows(out/"seed_summary.csv",summary);csv_rows(out/"ground_truth_settings.csv",truths.values());plot(out,summary)
    threshold_rows=[x for x in summary if x["selection_mode"]=="threshold"]
    csv_rows(out/"threshold_tradeoff.csv",threshold_rows);plot_threshold(out,threshold_rows)
    (out/"matrix.json").write_text(json.dumps({"training_runs":len(rows),"matched_groups":len(hashes),"catalog_sha256":hashlib.sha256(CATALOG.read_bytes()).hexdigest(),"design":"fixed budget and threshold selection, three paired seeds; seed-level descriptive means/SD; one-round counterfactuals","results":rows},indent=2,allow_nan=False))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--ledger",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();analyze(a.ledger,a.output)
