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

def select_cached(ids,last,due,reports,k,round_id,max_age,solver="exact",weights=None,trends=None):
    """Age-aware facility location. Stale evidence discounts similarity.
    Unknown omitted clients have priority over total distance.
    """
    ids=tuple(sorted(ids)); k=min(k,len(ids)); mandatory=tuple(due[:k])
    others=tuple(i for i in ids if i not in mandatory)
    weights=weights or dict.fromkeys(ids,1.)
    trends=trends or {}
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
                    uncertainty=min(1.,age/max(1,max_age+1)+age*(trends.get(i,0.)+trends.get(j,0.)))
                    costs.append(float(d)+(1-float(d))*uncertainty)
            loss+=weights[i]*min(costs,default=1.)
        return (missing,loss,tuple(sorted(priority(i,last) for i in s)))
    if solver=="greedy":
        s=list(mandatory)
        while len(s)<k:
            s.append(min((i for i in others if i not in s),key=lambda i:objective(s+[i])))
        return tuple(sorted(s))
    return tuple(sorted(min((mandatory+s for s in combinations(others,k-len(mandatory))),key=objective)))

def update_match(ids,last,due,reports,k,round_id,max_age,solver="exact",weighted=False):
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
        w={i:known[i].aggregation_weight if weighted else 1. for i in keys}
        total=sum(w.values())
        return [sum(w[i]*known[i].counts[j] for i in keys)/total for j in range(len(next(iter(known.values())).counts))]
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
    from .policies import above_threshold
    """Sequential hybrid: request/unknown/due first; peer coverage resolves
    redundant requests. Fixed count fills unused slots by original novelty."""
    from .policies import peer_select
    ids=tuple(sorted(ids))
    mandatory=tuple(due)+tuple(i for i in sorted(ids,key=lambda x:priority(x,last))
                               if i not in due and scores.get(i) is None)
    requests=tuple(i for i in ids if i in mandatory or scores.get(i) is None or above_threshold(scores[i],threshold))
    if k is None:
        return peer_select(requests,last,mandatory,reports,threshold,None,solver,objective) if requests else ()
    selected=peer_select(requests,last,mandatory,reports,threshold,min(k,len(requests)),solver,objective) if requests else ()
    remaining=sorted((i for i in ids if i not in selected),key=lambda i:(scores.get(i) is not None,-(scores.get(i) or 0),priority(i,last)))
    return tuple(sorted(tuple(selected)+tuple(remaining[:max(0,k-len(selected))])))

def observed_trends(history,ids,task,r):
    out={}
    for i in ids:
        rr=sorted((x for x in history if x.client==i and x.task==task and x.round<r),key=lambda x:x.round)
        if len(rr)>=2:
            a,b=rr[-2:];d=distance(a,b)
            if d is not None and b.round>a.round:out[i]=float(d)/(b.round-a.round)
    return out

def joint_peer_select(ids,last,due,reports,scores,threshold,k,solver="exact",novelty_weight=1.):
    ids=tuple(sorted(ids));k=min(k,len(ids));mandatory=tuple(due[:k])
    candidates=[i for i in ids if i not in mandatory]
    def cost(selected):
        unknown=0;value=0.
        for i in ids:
            if i in selected:continue
            d,_=nearest(i,selected,reports);score=scores.get(i)
            unknown+=score is None or d is None
            value+=(1. if d is None else float(d))+novelty_weight*(1. if score is None else float(score))
        return unknown,value,tuple(sorted(priority(i,last) for i in selected))
    if solver=="greedy":
        selected=list(mandatory)
        while len(selected)<k:selected.append(min((i for i in candidates if i not in selected),key=lambda i:cost(selected+[i])))
        return tuple(sorted(selected))
    return tuple(sorted(min((mandatory+x for x in combinations(candidates,k-len(mandatory))),key=cost)))
