"""Selection on delivered evidence only; no access to neural models or evaluation.
Cached reports may be stale. Unknown evidence receives lexicographic priority.
"""
from itertools import combinations
import math
from .policies import distance, priority, nearest

def cached_views(records, ids, task, round_id, max_age):
    out = dict.fromkeys(ids)
    for x in sorted(records, key=lambda x:(x.round,x.id)):
        if x.client in out and x.task==task and 0<=round_id-x.round<=max_age and x.valid():
            out[x.client]=x
    return out

def select_cached(ids,last,due,reports,k,round_id,max_age,solver="exact",weights=None):
    """Age-aware facility location. Stale evidence discounts similarity.
    Unknown omitted clients have priority over total distance.
    """
    ids=tuple(sorted(ids)); k=min(k,len(ids)); mandatory=tuple(due[:k])
    others=tuple(i for i in ids if i not in mandatory)
    weights=weights or dict.fromkeys(ids,1.)
    def objective(s):
        missing=0; loss=0.
        for i in ids:
            if i in s: continue
            x=reports.get(i)
            if x is None or not x.valid():
                missing+=1;continue
            costs=[]
            for j in s:
                y=reports.get(j); d=distance(x,y)
                if d is not None:
                    age=max(round_id-x.round,round_id-y.round)
                    uncertainty=min(1.,age/max(1,max_age+1))
                    costs.append(float(d)+(1-float(d))*uncertainty)
            loss+=weights[i]*min(costs,default=1.)
        return (missing,loss,tuple(sorted(priority(i,last) for i in s)))
    if solver=="greedy":
        s=list(mandatory)
        while len(s)<k:
            s.append(min((i for i in others if i not in s),key=lambda i:objective(s+[i])))
        return tuple(sorted(s))
    return tuple(sorted(min((mandatory+s for s in combinations(others,k-len(mandatory))),key=objective)))

def update_match(ids,last,due,reports,k,round_id,max_age,solver="exact"):
    """Match the cached equal-client-weight mean update sketch, not an oracle
    current update. Missing or incompatible records are prioritized for refresh.
    """
    ids=tuple(sorted(ids)); k=min(k,len(ids)); mandatory=tuple(due[:k])
    others=tuple(i for i in ids if i not in mandatory)
    known={i:x for i,x in reports.items() if x is not None and x.valid() and x.kind=="update"}
    if known:
        base=max(known.values(),key=lambda x:(x.round,-x.client))
        known={i:x for i,x in known.items() if distance(base,x) is not None}
    def mean(keys):
        if not keys:return None
        return [sum(known[i].counts[j] for i in keys)/len(keys) for j in range(len(next(iter(known.values())).counts))]
    target=mean(list(known))
    norm=sum(x*x for x in target) if target is not None else 0.
    def cost(s):
        unknown=sum(i not in s and i not in known for i in ids)
        chosen=[i for i in s if i in known]
        estimate=mean(chosen)
        error=(sum((a-b)**2 for a,b in zip(target,estimate))/(norm+1e-12)) if estimate is not None else (1. if target else 0.)
        age=sum(round_id-known[i].round for i in chosen)/max(1,len(chosen)*max(1,max_age+1))
        return unknown,error+age,tuple(sorted(priority(i,last) for i in s))
    if solver=="greedy":
        s=list(mandatory)
        while len(s)<k:
            s.append(min((i for i in others if i not in s),key=lambda i:cost(s+[i])))
        return tuple(sorted(s))
    return tuple(sorted(min((mandatory+s for s in combinations(others,k-len(mandatory))),key=cost)))

def combine_scores(local,server,rule="union"):
    # Unknown is never evidence for suppression, even in intersection mode.
    if local is None or server is None:return None
    return max(local,server) if rule=="union" else min(local,server)

def peer_filter(ids,last,due,reports,scores,threshold,k,solver,objective):
    """Sequential hybrid: request/unknown/due first; peer coverage resolves
    redundant requests. Fixed count fills unused slots by original novelty."""
    from .policies import peer_select
    ids=tuple(sorted(ids))
    mandatory=tuple(due)+tuple(i for i in sorted(ids,key=lambda x:priority(x,last))
                               if i not in due and scores.get(i) is None)
    requests=tuple(i for i in ids if i in mandatory or scores.get(i) is None or scores[i]>threshold)
    if k is None:
        return peer_select(requests,last,mandatory,reports,threshold,None,solver,objective) if requests else ()
    selected=peer_select(requests,last,mandatory,reports,threshold,min(k,len(requests)),solver,objective) if requests else ()
    remaining=sorted((i for i in ids if i not in selected),key=lambda i:(scores.get(i) is not None,-(scores.get(i) or 0),priority(i,last)))
    return tuple(sorted(tuple(selected)+tuple(remaining[:max(0,k-len(selected))])))
