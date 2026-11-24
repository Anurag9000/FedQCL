"""Portable calibrated linear harm prediction, trained offline on development branches.
The deployed selector uses only previously delivered server reports and sync ages.
"""
from itertools import combinations
import math
from .policies import nearest,priority

FEATURES=("intercept","fraction_selected","mean_omitted_distance","unknown_fraction",
          "mean_omitted_sync_age","max_omitted_sync_age","mean_report_age","task","task_progress",
          "max_omitted_distance","distance_variance","weighted_omitted_fraction","incompatible_fraction",
          "mean_report_magnitude","mean_channel_missing_fraction")

def subset_features(ids,subset,reports,last,r,task,rpt,max_age):
    ids=tuple(ids); chosen=set(subset); omitted=[i for i in ids if i not in chosen]
    distances=[];unknown=0;ages=[];report_ages=[];magnitudes=[];missing_channels=[];incompatible=0
    for i in omitted:
        d,_=nearest(i,subset,reports)
        distances.append(1. if d is None else float(d))
        unknown+=d is None
        ages.append(max(0,r-last.get(i,-1))/max(1,max_age+1))
        x=reports.get(i)
        report_ages.append(1. if x is None else max(0,r-x.round)/max(1,max_age+1))
        if x is not None:
            incompatible+=d is None
            magnitudes.append(math.log1p(math.sqrt(sum(v*v for v in x.counts)/max(1,len(x.counts)))))
            if x.metric=="channels":
                from .multichannel import layout
                pos=0;missing=0;blocks=layout(x.space)
                for _,length,_,_ in blocks:
                    missing+=x.counts[pos]<=.5;pos+=length+1
                missing_channels.append(missing/len(blocks))
    n=max(1,len(omitted))
    weights={i:reports[i].importance if reports.get(i) is not None else 1. for i in ids}
    mean=sum(distances)/n
    return [1.,len(chosen)/max(1,len(ids)),mean,unknown/n,
            sum(ages)/n,max(ages,default=0.),sum(report_ages)/n,float(task),(r%rpt)/max(1,rpt-1),
            max(distances,default=0.),sum((d-mean)**2 for d in distances)/n,
            sum(weights[i] for i in omitted)/max(1e-12,sum(weights.values())),incompatible/n,
            sum(magnitudes)/max(1,len(magnitudes)),sum(missing_channels)/max(1,len(missing_channels))]

def feature_map(features,mode):
    if mode=="linear":return list(features)
    if mode=="quadratic":return list(features)+[features[i]*features[j] for i in range(1,len(features)) for j in range(i,len(features))]
    raise ValueError("unknown feature map")

def out_of_distribution(model,features):
    bounds=model.get("fit_bounds")
    if not bounds:return False
    return any(x<lo-.1*max(1.,hi-lo) or x>hi+.1*max(1.,hi-lo) for x,(lo,hi) in zip(features,bounds))

def predict(model,features):
    if model["features"]!=list(FEATURES):
        raise ValueError("incompatible impact feature schema")
    mapped=feature_map(features,model.get("feature_map","linear"))
    if any(len(params["coef"])!=len(mapped) for params in model["targets"].values()):raise ValueError("impact coefficient dimension mismatch")
    return {kind:sum(x*w for x,w in zip(mapped,params["coef"]))+params["upper_residual"]
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
            features=subset_features(ids,s,reports,last,r,task,rpt,max_age)
            pred=predict(model,features)
            ood=out_of_distribution(model,features)
            vc=max(0.,pred["current"]-current_tol)
            vp=max(0.,pred["past"]-past_tol) if task else 0.
            # Unknown omitted evidence is not grounds for a safe prediction.
            unknown=sum(nearest(i,s,reports)[0] is None for i in ids if i not in s)
            if size==len(ids):
                # Full cohort is the reference: its harm is exactly zero.
                vc=vp=0.;unknown=0;ood=False;pred={"current":0.,"past":0.}
            record={"upper_harm":pred,"unknown_omitted":unknown,
                    "out_of_distribution":ood,"tolerance_met":vc==vp==0. and unknown==0 and not ood}
            tie=tuple(sorted(priority(i,last) for i in s))
            if record["tolerance_met"]:
                feasible.append((size,tie,s,record))
            fallback.append((unknown,int(ood),vc+vp,size,tie,s,record))
        if feasible:
            return min(feasible,key=lambda x:x[:2])[2:]
    best=min(fallback,key=lambda x:x[:5])
    return best[5],best[6]
