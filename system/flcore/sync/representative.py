"""Causal represented-mass correction from delivered collective reports."""
import math
import numpy as np

def discrepancy(a,b,common_proxy=False):
    if not a or not b or a["task"]!=b["task"] or a["round"]!=b["round"]:
        return None
    if not common_proxy and a["version"]!=b["version"]:
        return None
    ga,gb=a["gradient"],b["gradient"]
    t=str(a["task"])
    if t not in ga or t not in gb or set(ga)!=set(gb):
        return None
    scores=[.5*float(np.abs(np.asarray(a["labels_hist"])-np.asarray(b["labels_hist"])).sum())]
    for k in sorted(ga):
        if min(a["support"].get(k,0),b["support"].get(k,0))<1:return None
        x,y=np.asarray(ga[k]["counts"],float),np.asarray(gb[k]["counts"],float)
        if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():return None
        den=float(np.linalg.norm(x)+np.linalg.norm(y))
        scores.append(float(np.linalg.norm(x-y))/den if den else 0.)
    d=max(scores)
    return d if math.isfinite(d) else None

def plan(view,selected,args):
    selected=tuple(sorted(selected))
    beta=getattr(args,"sync_representative_beta",0.)
    limit=getattr(args,"sync_representative_distance",.35)
    transfers={};unrepresented=[]
    if not beta:return {"transfers":transfers,"unrepresented":unrepresented}
    for i,x in sorted(view.items()):
        if i in selected:continue
        options=[]
        for j in selected:
            d=discrepancy(x,view.get(j),getattr(args,"sync_utility_proxy",False))
            if d is not None and d<=limit:options.append((d,j))
        if not options or not x or not math.isfinite(x["mass"]) or x["mass"]<=0:
            unrepresented.append(i);continue
        d,j=min(options)
        transfers[str(i)]={"representative":j,"mass":float(beta*x["mass"]),"distance":d}
    return {"transfers":transfers,"unrepresented":unrepresented}

def weights(selected,own,representation):
    selected=tuple(selected)
    w={i:float(own[i]) for i in selected}
    for i,item in representation["transfers"].items():
        j=item["representative"];m=item["mass"]
        if int(i) in w or j not in w or not math.isfinite(m) or m<0:
            raise ValueError("invalid represented mass transfer")
        w[j]+=m
    return [w[i] for i in selected]
