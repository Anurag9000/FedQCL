"""Portable calibrated linear harm prediction, trained offline on development branches.
The deployed selector uses only previously delivered server reports and sync ages.
"""
from itertools import combinations
import math
from .policies import nearest,priority

FEATURES=("intercept","fraction_selected","mean_omitted_distance","unknown_fraction",
          "mean_omitted_sync_age","max_omitted_sync_age","mean_report_age","task","task_progress")

def subset_features(ids,subset,reports,last,r,task,rpt,max_age):
    ids=tuple(ids); chosen=set(subset); omitted=[i for i in ids if i not in chosen]
    distances=[];unknown=0;ages=[];report_ages=[]
    for i in omitted:
        d,_=nearest(i,subset,reports)
        distances.append(1. if d is None else float(d))
        unknown+=d is None
        ages.append(max(0,r-last.get(i,-1))/max(1,max_age+1))
        x=reports.get(i)
        report_ages.append(1. if x is None else max(0,r-x.round)/max(1,max_age+1))
    n=max(1,len(omitted))
    return [1.,len(chosen)/max(1,len(ids)),sum(distances)/n,unknown/n,
            sum(ages)/n,max(ages,default=0.),sum(report_ages)/n,float(task),(r%rpt)/max(1,rpt-1)]

def predict(model,features):
    if model["features"]!=list(FEATURES):
        raise ValueError("incompatible impact feature schema")
    return {kind:sum(x*w for x,w in zip(features,params["coef"]))+params["upper_residual"]
            for kind,params in model["targets"].items()}

def adaptive_select(ids,last,due,reports,r,task,rpt,max_age,model,k,current_tol,past_tol,soft=False):
    ids=tuple(sorted(ids));k=min(k,len(ids)); mandatory=tuple(due)
    # Hard capacity may prevent visiting all due clients; soft mode expands.
    maximum=len(ids) if soft else k
    required=mandatory[:maximum]
    others=tuple(i for i in ids if i not in required)
    feasible=[];fallback=[]
    for size in range(max(1,len(required)),maximum+1):
        for ext in combinations(others,size-len(required)):
            s=tuple(sorted(required+ext))
            pred=predict(model,subset_features(ids,s,reports,last,r,task,rpt,max_age))
            vc=max(0.,pred["current"]-current_tol)
            vp=max(0.,pred["past"]-past_tol) if task else 0.
            # Unknown omitted evidence is not grounds for a safe prediction.
            unknown=sum(reports.get(i) is None for i in ids if i not in s)
            if size==len(ids):
                # Full cohort is the reference: its harm is exactly zero.
                vc=vp=0.;unknown=0;pred={"current":0.,"past":0.}
            record={"upper_harm":pred,"unknown_omitted":unknown,
                    "tolerance_met":vc==vp==0. and unknown==0}
            tie=tuple(sorted(priority(i,last) for i in s))
            if record["tolerance_met"]:
                feasible.append((size,tie,s,record))
            fallback.append((unknown,vc+vp,size,tie,s,record))
        if feasible:
            return min(feasible,key=lambda x:x[:2])[2:]
    best=min(fallback,key=lambda x:x[:4])
    return best[4],best[5]
