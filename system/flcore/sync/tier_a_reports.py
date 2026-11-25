"""Poolable, support-aware reports. No raw examples enter the wire payload."""
import json,math
from dataclasses import replace

def make_layout(blocks,coordinate):
    return "tier_a:"+json.dumps({"blocks":blocks,"coordinate":coordinate},sort_keys=True,separators=(",",":"))

def blocks(report):
    offset=0
    for name,group,n,metric in json.loads(report.space.split(":",1)[1])["blocks"]:
        support=report.counts[offset]
        yield name,group,n,metric,support,report.counts[offset+1:offset+1+n]
        offset+=n+1
    if offset!=len(report.counts):raise ValueError("report length")

def valid_layout(report):
    try:
        if not report.space.startswith("tier_a:"):return False
        names=set();available=False
        for name,group,n,metric,support,v in blocks(report):
            if name in names or group not in ("current","past") or type(n)!=int or n<1 or len(v)!=n or support<0 or not math.isfinite(support):return False
            names.add(name)
            if metric not in ("relative","moments","js","unit_l1") or (metric=="moments" and n%2):return False
            if support==0 and any(v):return False
            if support and metric=="unit_l1" and any(x<0 or x>1 for x in v):return False
            if support and metric=="js" and (min(v)<0 or sum(v)<=0):return False
            available|=support>0
        return available
    except (ValueError,TypeError,KeyError,IndexError):return False

def relative(x,y):
    norm=lambda v:math.sqrt(sum(a*a for a in v))
    den=norm(x)+norm(y)
    return norm([a-b for a,b in zip(x,y)])/den if den else 0.

def report_distance(a,b):
    groups={}
    for aa,bb in zip(blocks(a),blocks(b)):
        _,group,n,metric,sa,x=aa
        sb,y=bb[-2:]
        if not sa and not sb:continue
        if not sa or not sb:d=1.
        elif metric=="relative":d=relative(x,y)
        elif metric=="unit_l1":d=sum(abs(u-v) for u,v in zip(x,y))/n
        elif metric=="js":
            from .statistics import js
            d=js(x,y) # equal-mass ordered probability rows => mean row JS
        else:
            k=n//2
            vx=[math.sqrt(max(0.,q-m*m)) for m,q in zip(x[:k],x[k:])]
            vy=[math.sqrt(max(0.,q-m*m)) for m,q in zip(y[:k],y[k:])]
            d=(relative(x[:k],y[:k])+relative(vx,vy))/2
        groups.setdefault(group,[]).append(max(0.,min(1.,float(d))))
    return sum(sum(v)/len(v) for v in groups.values())/len(groups) if groups else None

def pooled_report(records,template,weighted):
    if any((r.space,r.kind,r.task)!=(template.space,template.kind,template.task) for r in records):
        raise ValueError("incompatible Tier-A reports")
    result=[]
    for group in zip(*(blocks(r) for r in records)):
        n=group[0][2]
        active=[(b[-2] if weighted else 1.,b[-1]) for b in group if b[-2]>0]
        count=sum(w for w,_ in active)
        result.extend([float(count),*([sum(w*v[j] for w,v in active)/count for j in range(n)] if count else [0.]*n)])
    return replace(template,client=-1,round=max(r.round for r in records),window="pooled",
                   counts=tuple(result),sample_count=sum(r.sample_count for r in records))
