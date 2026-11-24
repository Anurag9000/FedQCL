#!/usr/bin/env python3
"""Derive Task-IL results and plots only from validated completed ledger jobs."""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import statistics
from run_campaign import valid

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def write_csv(path,rows):
    keys=sorted(set().union(*(r.keys() for r in rows))) if rows else ["experiment_id"]
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})

def run(ledger,output,require_complete=False):
    jobs=[j for j in read_jsonl(ledger) if j["kind"]=="train"]
    output.mkdir(parents=True,exist_ok=True)
    summaries=[];round_rows=[];task_rows=[];class_rows=[];diagnostics=[];missing=[]
    for job in jobs:
        cfg=job["config"]
        if not valid(job):
            missing.append(dict(experiment_id=cfg["experiment_id"],seed=cfg["seed"],job_id=job["id"]))
            continue
        out=Path(job["artifacts"][0]).parent
        prefix=dict(experiment_id=cfg["experiment_id"],seed=cfg["seed"],job_id=job["id"],family=cfg["family"],evaluation_split=cfg["evaluation_split"])
        summary=json.loads((out/"final_summary.json").read_text())
        summaries.append({**cfg,**prefix,**summary})
        for r in read_jsonl(out/"rounds.jsonl"):
            round_rows.append({**prefix,**r})
        for r in read_jsonl(out/"server_metrics.jsonl"):
            task_rows.append({**prefix,**r})
            for label,count,correct in zip(r["class_ids"],r["class_counts"],r["class_correct"]):
                class_rows.append({**prefix,"round":r["round"],"task":r["task"],"class_id":label,
                                   "count":count,"correct":correct,"accuracy":correct/count if count else None})
        for r in read_jsonl(out/"client_metrics.jsonl"):
            diagnostics.append({**prefix,**r})
    if require_complete and missing:raise ValueError("Task-IL analysis requires all selected jobs to pass verification")
    by_id={(x["experiment_id"],x["seed"],x["evaluation_split"]):x for x in summaries}
    contrasts=[]
    for x in summaries:
        b=by_id.get((x["baseline"],x["seed"],x["evaluation_split"]))
        if b is None or b["experiment_id"]==x["experiment_id"]:continue
        f=x["forgetting"];bf=b["forgetting"]
        contrasts.append(dict(experiment_id=x["experiment_id"],baseline_id=b["experiment_id"],seed=x["seed"],evaluation_split=x["evaluation_split"],
            accuracy_delta=x["final_accuracy"]-b["final_accuracy"],forgetting_delta=f-bf if f is not None and bf is not None else None,
            byte_savings=1-x["total_bytes"]/b["total_bytes"],processed_samples_delta=x["processed_samples"]-b["processed_samples"],
            optimizer_steps_delta=x["optimizer_steps"]-b["optimizer_steps"],
            comparison="matched curriculum and participation contrast" if x["family"] in (1,5,7) else "descriptive; check schedule and exposure differences"))
    # Reuse mandated 5-sync / 5-client baselines without executing extra jobs.
    study_rows=[dict(x,study_family=x["family"],reused=False) for x in summaries]
    for x in summaries:
        if x["experiment_id"]=="E01-05":
            study_rows.extend(dict(x,study_family=f,reused=True) for f in (2,3))
        elif x["experiment_id"]=="E02-01":
            study_rows.append(dict(x,study_family=3,reused=True))
        elif x["experiment_id"]=="E05-05" and not any(y["experiment_id"]=="E10-02" and y["seed"]==x["seed"] and y["evaluation_split"]==x["evaluation_split"] for y in summaries):
            study_rows.append(dict(x,study_family=10,reused=True))
    diagnostic_classes=[];recurrence_gains=[]
    grouped=defaultdict(dict)
    for r in diagnostics:
        for label,n,correct in zip(r["class_ids"],r["class_counts"],r["class_correct"]):
            diagnostic_classes.append({k:r[k] for k in ("job_id","experiment_id","family","seed","evaluation_split","client","episode","round","phase","task")}
                                      |dict(class_id=label,count=n,correct=correct,accuracy=correct/n if n else None))
        if r["episode"] is not None:
            grouped[(r["job_id"],r["client"],r["episode"],r["task"])][r["phase"]]=r
    for (jid,client,episode,task),phases in grouped.items():
        for before,after,kind in (("before_exposure_server","after_exposure_server","server"),
                                  ("before_exposure_client","after_exposure_before_sync","local_pre_download")):
            if before not in phases or after not in phases:continue
            a,b=phases[before],phases[after]
            for label,n,c0,c1 in zip(a["class_ids"],a["class_counts"],a["class_correct"],b["class_correct"]):
                recurrence_gains.append(dict(job_id=jid,experiment_id=a["experiment_id"],seed=a["seed"],client=client,
                    episode=episode,task=task,kind=kind,class_id=label,round_before=a["round"],round_after=b["round"],
                    accuracy_before=c0/n,accuracy_after=c1/n,accuracy_gain=(c1-c0)/n))
    overlap_rows=[]
    for x in summaries:
        if x["family"]!=8:continue
        out=Path(next(j for j in jobs if j["id"]==x["job_id"])["artifacts"][0]).parent
        s=json.loads((out/"schedule.json").read_text())
        rows={(r["round"],r["task"]):r for r in task_rows if r["job_id"]==x["job_id"]}
        for i in range(s["clients"]):
            es=sorted((e for e in s["episodes"] if e["client"]==i),key=lambda e:e["start"])
            for a,b in zip(es,es[1:]):
                old=rows[(a["end"],a["task"])];new=rows[(b["end"],a["task"])]
                shared=set(s["tasks"][a["task"]])&set(s["tasks"][b["task"]])
                for label,n,c0,c1 in zip(old["class_ids"],old["class_counts"],old["class_correct"],new["class_correct"]):
                    overlap_rows.append(dict(job_id=x["job_id"],experiment_id=x["experiment_id"],seed=x["seed"],client=i,
                        old_task=a["task"],next_task=b["task"],class_id=label,shared=label in shared,accuracy_before=c0/n,accuracy_after=c1/n,change=(c1-c0)/n))
    for name,rows in (("experiment_results",summaries),("study_results",study_rows),("round_metrics",round_rows),
        ("task_metrics",task_rows),("class_metrics",class_rows),("client_diagnostics",diagnostics),
        ("diagnostic_class_metrics",diagnostic_classes),("recurrence_gains",recurrence_gains),
        ("baseline_contrasts",contrasts),("overlap_retention",overlap_rows),("incomplete_jobs",missing)):
        write_csv(output/(name+".csv"),rows)
    # Every seed stays separately visible; figures use mean over available seeds,
    # never rounds or clients as independent replicates.
    if study_rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plot=output/"plots";plot.mkdir(exist_ok=True)
        xkeys={1:"offline",2:"syncs",3:"clients",4:"sharing",5:"offline",6:"recurrence",7:"outage",8:"overlap",9:"delay",10:"width"}
        for family,key in xkeys.items():
            data=[r for r in study_rows if r["study_family"]==family]
            if not data:continue
            groupkey="curriculum" if family==1 else "syncs" if family==3 else "width" if family==5 else None
            for metric in ("final_accuracy","forgetting","final_current_task_accuracy","final_current_task_loss","final_previous_task_accuracy","total_bytes","processed_samples","chance_adjusted_accuracy"):
                groups=defaultdict(list)
                for row in data:
                    if row.get(metric) is not None:
                        groups[str(row[groupkey]) if groupkey else "all"].append(row)
                if not groups:continue
                fig,ax=plt.subplots()
                for label,rr in sorted(groups.items()):
                    points=defaultdict(list)
                    for row in rr:points[row[key]].append(row[metric])
                    xx=sorted(points);ax.plot(xx,[statistics.mean(points[v]) for v in xx],marker="o",label=label)
                ax.set(xlabel=key,ylabel=metric,title=f"Task-IL study {family:02d} ({data[0]['evaluation_split']})")
                if groupkey:ax.legend(title=groupkey)
                fig.tight_layout();fig.savefig(plot/f"E{family:02d}_{metric}.png",dpi=150);plt.close(fig)
        for family in range(1,11):
            data=[r for r in round_rows if r["family"]==family]
            if not data:continue
            for xkey in ("round","cumulative_bytes","cumulative_samples","cumulative_optimizer_steps"):
                for metric in ("global_accuracy","current_task_accuracy","previous_task_accuracy","forgetting","current_task_loss"):
                    fig,ax=plt.subplots()
                    groups=defaultdict(list)
                    for row in data:
                        if row.get(metric) is not None:groups[row["job_id"]].append(row)
                    for label,rr in sorted(groups.items()):
                        rr.sort(key=lambda r:r["round"]);ax.plot([r[xkey] for r in rr],[r[metric] for r in rr],label=label.split("-s")[0],alpha=.7)
                    ax.set(xlabel=xkey,ylabel=metric,title=f"Study {family:02d}")
                    if len(groups)<=12:ax.legend(fontsize=6)
                    fig.tight_layout();fig.savefig(plot/f"E{family:02d}_{metric}_vs_{xkey}.png",dpi=150);plt.close(fig)
        for family in (6,7,9):
            data=[r for r in diagnostics if r["family"]==family]
            if not data:continue
            fig,ax=plt.subplots()
            groups=defaultdict(list)
            for r in data:groups[(r["job_id"],r["phase"],r["round"])].append(r["accuracy"])
            series=defaultdict(list)
            for (jid,phase,r),values in groups.items():series[jid,phase].append((r,statistics.mean(values)))
            for (jid,phase),values in sorted(series.items()):
                values.sort();ax.plot([p[0] for p in values],[p[1] for p in values],marker=".",label=jid.split("-s")[0]+" "+phase)
            ax.set(xlabel="round",ylabel="task-macro accuracy",title=f"Study {family:02d} local / exposure diagnostics")
            ax.legend(fontsize=5);fig.tight_layout();fig.savefig(plot/f"E{family:02d}_diagnostics.png",dpi=150);plt.close(fig)
    receipt=dict(completed=len(summaries),expected=len(jobs),incomplete=len(missing),output=str(output))
    (output/"receipt.json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))

def main():
    p=argparse.ArgumentParser();p.add_argument("--ledger",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("research_results/task_il_48/analysis"))
    p.add_argument("--require-complete",action="store_true")
    a=p.parse_args();run(a.ledger,a.output,a.require_complete)
if __name__=="__main__":main()
