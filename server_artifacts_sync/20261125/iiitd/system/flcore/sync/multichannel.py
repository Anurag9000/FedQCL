"""Blockwise data/replay/feature/current-gradient/replay-gradient evidence.
Each channel is explicit, masked when unavailable and compared independently.
The compact report uses a fixed layout, not an unscaled concatenated Euclidean metric.
"""
import math
import json
import time
from .policies import Report

def layout(space):
    return json.loads(space.split(":",1)[1])["blocks"]

def valid_layout(report):
    try:
        blocks=layout(report.space);offset=0;names=set()
        if not blocks:return False
        for name,n,w,metric in blocks:
            if name in names or type(n) is not int or n<=0 or not math.isfinite(w) or w<=0 or metric not in ("tv","cosine","l2"):return False
            names.add(name)
            mask=report.counts[offset];values=report.counts[offset+1:offset+1+n]
            if len(values)!=n or mask not in (0,1):return False
            if not mask and any(values):return False
            if mask and metric=="tv" and (min(values)<0 or sum(values)<=0):return False
            offset+=n+1
        return offset==len(report.counts)
    except (ValueError,TypeError,KeyError,IndexError):return False

def channel_distance(a,b):
    start=0;total=0.;weight=0.;shared=0
    for name,n,w,metric in layout(a.space):
        present_a=a.counts[start]>0.5;present_b=b.counts[start]>0.5
        x=a.counts[start+1:start+n+1];y=b.counts[start+1:start+n+1]
        start+=n+1
        if not present_a and not present_b:continue
        weight+=w
        if not (present_a and present_b):
            total+=w
            continue
        shared+=1
        if metric=="tv":
            sx=sum(x);sy=sum(y)
            d=.5*sum(abs(u/sx-v/sy) for u,v in zip(x,y)) if sx>0 and sy>0 else float(sx!=sy)
        elif metric=="cosine":
            nx=math.sqrt(sum(u*u for u in x));ny=math.sqrt(sum(v*v for v in y))
            d=(1-sum(u*v for u,v in zip(x,y))/(nx*ny))/2 if nx and ny else float(nx!=ny)
        else:
            norm=math.sqrt(sum((u-v)**2 for u,v in zip(x,y)))
            d=norm/(1+norm)
        total+=w*max(0.,min(1.,d))
    return total/weight if weight and shared else None

def gradient(ex,c,model,x,y,task):
    import torch
    model.zero_grad(set_to_none=True)
    for xx,yy in zip(x.split(c.batch_size),y.split(c.batch_size)):
        (torch.nn.functional.cross_entropy(model(xx,task),yy,reduction="sum")/len(x)).backward()
    return ex.sketch((name,p.grad) for name,p in model.named_parameters() if p.grad is not None)

def extract(ex,c,data,hist,t,r,version,kind):
    import copy
    import torch
    from torch.utils.data import default_collate
    start=time.monotonic();probe=ex._probe(c,data,t,r)
    cost={"forward_examples":0,"backward_examples":0,"seconds":0.}
    if probe is None:return None,cost
    x,y=probe;model=copy.deepcopy(c.model).eval()
    current=gradient(ex,c,model,x,y,t)
    cost["forward_examples"]+=len(x);cost["backward_examples"]+=len(x)
    old=[]
    # A deterministic bounded replay probe, without advancing the training cursor.
    import random
    from .signals import seed
    candidates=[(k,item) for k in range(t) for item in c.task_memories.get(k,[])]
    rng=random.Random(seed(ex.args.seed,"replay_probe",c.id,t,r))
    chosen=rng.sample(candidates,min(len(candidates),ex.args.sync_observation_size))
    for k in range(t):
        items=[item for task,item in chosen if task==k]
        if not items:continue
        xx,yy=default_collate(items);xx=xx.to(c.device);yy=yy.to(c.device)%c.classes_per_task
        old.append((len(items),gradient(ex,c,model,xx,yy,k)))
        cost["forward_examples"]+=len(items);cost["backward_examples"]+=len(items)
    replay=[sum(n*v[j] for n,v in old)/sum(n for n,_ in old) for j in range(len(current))] if old else None
    if kind=="gradient_conflict":
        if replay is None:
            cost["seconds"]=time.monotonic()-start
            return None,cost
        norm=lambda z:math.sqrt(sum(v*v for v in z))
        nc,nr=norm(current),norm(replay)
        cosine=sum(a*b for a,b in zip(current,replay))/(nc*nr) if nc and nr else 0.
        agreement=sum((a>0)==(b>0) for a,b in zip(current,replay))/len(current)
        values=[cosine,agreement,math.log1p(nc),math.log1p(nr)]
        report=ex._vector(c,t,r,version,kind,"native_sketch_conflict" if ex.args.sync_native_space else f"server_version:{version}",values,len(x),"bounded_l2")
    else:
        dims=ex.args.sync_sketch_dim
        features=[]
        with torch.no_grad():
            for xx in x.split(c.batch_size):
                m=ex.anchor
                z=torch.relu(m.bn1(m.conv1(xx)));z=m.layer4(m.layer3(m.layer2(m.layer1(z))))
                features.append(m.avgpool(z).flatten(1).cpu())
        z=torch.cat(features)
        feature=ex.sketch([("multichannel_feature",torch.cat((z.mean(0),z.var(0,unbiased=False))))])
        cost["forward_examples"]+=len(x)
        memory=[len(c.task_memories.get(k,[])) for k in range(c.num_tasks)]
        blocks=[("histogram",c.classes_per_task,1.,"tv"),("memory",c.num_tasks,1.,"tv"),
                ("feature",dims,1.,"l2"),("current_gradient",dims,1.,"cosine"),("replay_gradient",dims,1.,"cosine")]
        vectors=[list(hist.counts) if hist else None,memory if sum(memory) else None,feature,current,replay]
        values=[]
        for (_,n,_,_),v in zip(blocks,vectors):
            values.extend([1. if v is not None else 0.,*(v if v is not None else [0.]*n)])
        space="channels:"+json.dumps({"blocks":blocks,"seed":ex.args.seed,
                 "gradient_space":"native" if ex.args.sync_native_space else version},sort_keys=True,separators=(",",":"))
        report=Report(c.id,t,r,tuple(range(len(values))),tuple(float(v) for v in values),kind=kind,space=space,
                      sample_count=len(x),model_version=version,metric="channels")
    cost["seconds"]=time.monotonic()-start
    return report,cost
