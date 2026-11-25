"""Matched-stage, measured-byte comparisons; no interpolation across tasks."""
import json,math
from collections import defaultdict
from pathlib import Path

def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]

def learning_curve(metrics,communication,tasks,rounds):
    if rounds%tasks:raise ValueError("unequal task lengths")
    rpt=rounds//tasks
    if [x["round"] for x in communication]!=list(range(-1,rounds)):raise ValueError("invalid communication timeline")
    by_round=defaultdict(dict)
    for m in metrics:
        if m["task"] in by_round[m["round"]]:raise ValueError("duplicate evaluation")
        by_round[m["round"]][m["task"]]=m
    result=[]
    for r in range(rounds):
        t=r//rpt;current=by_round[r];b=communication[r+1]
        if set(current)!=set(range(t+1)):raise ValueError("evaluation tasks mismatch")
        acc=[current[j]["accuracy"] for j in range(t+1)]
        if any(not 0<=x<=1 for x in acc):raise ValueError("invalid accuracy")
        base=communication[t*rpt]["cumulative_bytes"]
        result.append({"round":r,"task":t,"cumulative_bytes":b["cumulative_bytes"],
                       "task_bytes":b["cumulative_bytes"]-base,"current_accuracy":acc[-1],
                       "mean_seen_accuracy":sum(acc)/len(acc),
                       "mean_past_accuracy":sum(acc[:-1])/t if t else None,
                       "participants":b["participants"],"model_bytes":b["model_bytes"],
                       "statistics_bytes":b["statistics_bytes"],"control_bytes":b["control_bytes"]})
    return result

def bytes_to_targets(curve,targets=(.5,.7,.9)):
    result=[]
    for t in sorted({x["task"] for x in curve}):
        rows=[x for x in curve if x["task"]==t]
        for target in targets:
            found=next((x for x in rows if x["current_accuracy"]>=target),None)
            result.append({"task":t,"target_accuracy":target,"reached":found is not None,
                           "round":found["round"] if found else None,
                           "task_bytes":found["task_bytes"] if found else None,
                           "cumulative_bytes":found["cumulative_bytes"] if found else None,
                           "observed_task_bytes":rows[-1]["task_bytes"]})
    return result

def compare_equal_bytes(method,reference,points=(.25,.5,.75,1.0)):
    """Right-continuous observed checkpoints, no extrapolation; same current task.

    Use a common within-task byte horizon that both complete trajectories
    reach. Report an absent estimate when a budget precedes the first
    evaluation. No cross-task averaging or assumed interpolation.
    """
    result=[]
    for t in sorted({x["task"] for x in method}):
        a=[x for x in method if x["task"]==t];b=[x for x in reference if x["task"]==t]
        if not b:raise ValueError("reference missing task")
        common=min(a[-1]["task_bytes"],b[-1]["task_bytes"])
        for fraction in points:
            budget=math.floor(common*fraction)
            x=next((v for v in reversed(a) if v["task_bytes"]<=budget),None)
            y=next((v for v in reversed(b) if v["task_bytes"]<=budget),None)
            result.append({"task":t,"fraction_of_common_task_budget":fraction,"byte_budget":budget,
                           "method_round":x["round"] if x else None,"baseline_round":y["round"] if y else None,
                           "method_actual_bytes":x["task_bytes"] if x else None,"baseline_actual_bytes":y["task_bytes"] if y else None,
                           "current_accuracy_delta":x["current_accuracy"]-y["current_accuracy"] if x and y else None,
                           "mean_seen_accuracy_delta":x["mean_seen_accuracy"]-y["mean_seen_accuracy"] if x and y else None})
    return result

def pareto_front(rows):
    """Nondominated seed means on validation accuracy, forgetting, bytes.

    Ties are retained; test data are not used to pick the frontier.
    """
    groups=defaultdict(list)
    for row in rows:
        vals=(row.get("total_bytes_mean"),row.get("forgetting_mean"),row.get("final_accuracy_mean"))
        if any(v is None or not math.isfinite(v) for v in vals):raise ValueError("missing Pareto objective")
        groups[row["setting_id"]].append((row,(vals[0],vals[1],-vals[2])))
    result=[]
    for setting,items in sorted(groups.items()):
        for row,a in items:
            dominated=any(all(x<=y for x,y in zip(b,a)) and any(x<y for x,y in zip(b,a)) for _,b in items)
            result.append({**{k:row[k] for k in ("setting_id","selector_id","selection_mode","threshold","k","n_seeds",
                                                  "total_bytes_mean","forgetting_mean","final_accuracy_mean")},
                           "nondominated":not dominated})
    return result
