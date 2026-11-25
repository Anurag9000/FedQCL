"""Merged Tier-A extractors; private replay never leaves client observation."""
import copy,math,random,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from .backbone import sample_replay,replay_ce
from .signals import seed
from .tier_a_reports import make_layout

def replay_groups(ex,c,t,r):
    selected=sample_replay({k:v for k,v in c.task_memories.items() if k<t},
                           ex.args.sync_observation_size,random.Random(seed(ex.args.seed,"tier_a_replay",c.id,t,r)))
    grouped={}
    for k,item in selected:grouped.setdefault(k,[]).append(item)
    out={}
    for k,items in sorted(grouped.items()):
        x,y=default_collate(items);out[k]=(x.to(c.device),y.to(c.device)%c.classes_per_task)
    return out

def replay_diagnostic(ex,c,model,groups,cost,t):
    # Compare the same fixed private memory examples within a task (memory may
    # be rebalanced at boundaries). Log loss deltas, never use as a redundancy metric.
    values={};changes={}
    for k in sorted(groups):
        items=c.task_memories[k]
        rng=random.Random(seed(ex.args.seed,"tier_a_retention",c.id,t,k))
        chosen=rng.sample(items,min(32,len(items)))
        x,y=default_collate(chosen);x=x.to(c.device);y=y.to(c.device)%c.classes_per_task
        with torch.no_grad():
            loss=sum(torch.nn.functional.cross_entropy(model(xx,k),yy,reduction="sum").item()
                     for xx,yy in zip(x.split(c.batch_size),y.split(c.batch_size)))/len(x)
        key=f"tier_a_retention:{t}:{k}"
        baseline=c.signal_history.setdefault(key,loss)
        values[k]=loss;changes[k]=loss-baseline
        cost["forward_examples"]+=len(x)
    cost["replay_loss_by_task"]=values;cost["replay_loss_change_by_task"]=changes

def public_inputs(ex,c,k):
    cache=getattr(ex,"tier_a_public_cache",None)
    if cache is None:cache={};ex.tier_a_public_cache=cache
    if k not in cache:
        a=ex.args;manifest=getattr(a,"dataset_manifest",None)
        if not manifest or not manifest.get("public_probe_per_class"):raise ValueError("functional signal requires reserved common public probes")
        root=Path(a.dataset) if Path(a.dataset).is_absolute() else Path("../dataset")/a.dataset
        rel=f"task_{k}/public_probe/0.npz";path=root/rel
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest["shards"].get(rel):
            raise ValueError("public probe fingerprint mismatch")
        with np.load(path,allow_pickle=True) as z:data=z["data"].item()
        n=manifest["public_probe_per_class"]
        if not np.array_equal(data["y"],np.repeat(np.arange(k*c.classes_per_task,(k+1)*c.classes_per_task),n)):
            raise ValueError("public probe alignment mismatch")
        cache[k]=torch.from_numpy(data["x"])
    return cache[k].to(c.device)

def extract(ex,c,data,t,r,version,kind):
    from .extra_signals import features
    start=time.monotonic()
    cost={"forward_examples":0,"backward_examples":0,"seconds":0.,"replay_loss_by_task":{},"replay_loss_change_by_task":{}}
    model=copy.deepcopy(c.model).eval()
    groups=replay_groups(ex,c,t,r)
    # diagnostics use all nonempty past memories, regardless of sampled gradient groups
    replay_diagnostic(ex,c,model,{k:None for k in c.task_memories if k<t and c.task_memories[k]},cost,t)
    layout=[];values=[];support_total=0
    def put(name,group,n,metric,support,v):
        nonlocal support_total
        layout.append((name,group,n,metric));values.extend([float(support),*(list(map(float,v)) if support else [0.]*n)])
        support_total+=support
    def finish(coordinate):
        cost["seconds"]=time.monotonic()-start
        if not support_total:return None,cost
        return ex._vector(c,t,r,version,kind,make_layout(layout,coordinate),values,support_total,"tier_a"),cost
    if kind=="tier_a_functional":
        for k in range(t+1):
            x=public_inputs(ex,c,k)
            with torch.no_grad():
                prob=torch.cat([model(xx,k).softmax(-1) for xx in x.split(c.batch_size)])
            put(f"task_{k}","current" if k==t else "past",prob.numel(),"js",len(x),prob.flatten().cpu().tolist())
            cost["forward_examples"]+=len(x)
        return finish(f"public_probe:{ex.args.dataset_manifest['manifest_id']}")
    probe=ex._probe(c,data,t,r)
    if probe is None:return finish("unavailable")
    x,y=probe
    if kind=="tier_a_gradients":
        model.zero_grad(set_to_none=True)
        for xx,yy in zip(x.split(c.batch_size),y.split(c.batch_size)):
            (torch.nn.functional.cross_entropy(model(xx,t),yy,reduction="sum")/len(x)).backward()
        current=ex.sketch((n,p.grad) for n,p in model.named_parameters() if p.grad is not None)
        current_shared=ex.sketch((n,p.grad) for n,p in model.named_parameters() if p.grad is not None and not n.startswith("heads."))
        put("current_gradient","current",len(current),"relative",len(x),current)
        cost["forward_examples"]+=len(x);cost["backward_examples"]+=len(x)
        model.zero_grad(set_to_none=True)
        count=sum(len(yy) for _,yy in groups.values())
        if count:
            for k,(xx,yy) in groups.items():
                for bx,by in zip(xx.split(c.batch_size),yy.split(c.batch_size)):
                    (torch.nn.functional.cross_entropy(model(bx,k),by,reduction="sum")/count).backward()
            replay=ex.sketch((n,p.grad) for n,p in model.named_parameters() if p.grad is not None)
            replay_shared=ex.sketch((n,p.grad) for n,p in model.named_parameters() if p.grad is not None and not n.startswith("heads."))
            den=math.sqrt(sum(v*v for v in current_shared)*sum(v*v for v in replay_shared))
            cost["gradient_interference_cosine"]=max(-1.,min(1.,sum(a*b for a,b in zip(current_shared,replay_shared))/den)) if den else None
            cost["forward_examples"]+=count;cost["backward_examples"]+=count
        else:replay=[];cost["gradient_interference_cosine"]=None
        put("replay_gradient","past",len(current),"relative",count,replay)
        return finish("shared_native_parameter_coordinates")
    if kind!="tier_a_features":raise ValueError("unknown Tier-A signal")
    # Every client already receives the same bootstrap global model at round 0.
    # Freeze its backbone as a learned, common coordinate system for later probes.
    if ex.anchor is None or not getattr(ex,"tier_a_anchor_ready",False):
        return finish("bootstrap_pending")
    dim=min(32,ex.args.sync_sketch_dim)
    for k in range(t+1):
        samples=(x,y) if k==t else groups.get(k)
        if samples is None:z=None;labels=None
        else:
            xx,labels=samples
            raw=features(ex.anchor,xx,c.batch_size).cpu()
            g=torch.Generator().manual_seed(seed(ex.args.seed,"tier_a_projection",raw.shape[1],dim))
            proj=torch.randn((raw.shape[1],dim),generator=g)/math.sqrt(dim)
            z=raw@proj
            z=torch.nn.functional.normalize(z,dim=1)
            labels=labels.cpu();cost["forward_examples"]+=len(xx)
        for label in range(c.classes_per_task):
            q=z[labels==label] if z is not None else None
            n=len(q) if q is not None else 0
            v=torch.cat((q.mean(0),q.square().mean(0))).tolist() if n else []
            put(f"task_{k}_class_{label}","current" if k==t else "past",2*dim,"moments",n,v)
    return finish(f"frozen_bootstrap_encoder:{ex.args.seed}:{dim}")
