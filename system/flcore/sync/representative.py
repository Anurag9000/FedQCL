"""Causal represented-mass correction from delivered collective reports."""
import math
import numpy as np

def discrepancy(a,b,common_proxy=False):
    if not a or not b:return None
    try:
        if a["task"]!=b["task"] or a["round"]!=b["round"]:
            return None
        if not common_proxy and a["version"]!=b["version"]:
            return None
        ga,gb=a["gradient"],b["gradient"]
        t=str(a["task"])
        if t not in ga or t not in gb or set(ga)!=set(gb):
            return None
        ha,hb=np.asarray(a["labels_hist"],float),np.asarray(b["labels_hist"],float)
        if (ha.ndim!=1 or hb.ndim!=1 or ha.size==0 or ha.shape!=hb.shape
                or not np.isfinite(ha).all() or not np.isfinite(hb).all()
                or np.any(ha<0) or np.any(hb<0)
                or not np.isclose(ha.sum(),1.,rtol=0.,atol=1e-6)
                or not np.isclose(hb.sum(),1.,rtol=0.,atol=1e-6)):
            return None
        scores=[.5*float(np.abs(ha-hb).sum())]
        for k in sorted(ga):
            sa,sb=float(a["support"].get(k,0)),float(b["support"].get(k,0))
            if not math.isfinite(sa) or not math.isfinite(sb) or min(sa,sb)<1:return None
            x,y=np.asarray(ga[k]["counts"],float),np.asarray(gb[k]["counts"],float)
            if x.ndim!=1 or y.ndim!=1 or x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():return None
            den=float(np.linalg.norm(x)+np.linalg.norm(y))
            # Identical zero sketches contain no evidence of representation.
            if not math.isfinite(den) or den<=1e-12:return None
            scores.append(float(np.linalg.norm(x-y))/den)
    except (KeyError,TypeError,ValueError,OverflowError):
        return None
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
        try:mass=float(x["mass"]) if x else 0.
        except (KeyError,TypeError,ValueError,OverflowError):mass=0.
        if not options or not math.isfinite(mass) or mass<=0:
            unrepresented.append(i);continue
        d,j=min(options)
        transfers[str(i)]={"representative":j,"mass":float(beta*mass),"distance":d}
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
