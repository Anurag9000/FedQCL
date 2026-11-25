"""Eight high-priority extractors; private replay never leaves client observation."""
import copy,math,random,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from .backbone import sample_replay
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
        fingerprint=hashlib.sha256()
        for xx,yy in chosen:
            fingerprint.update(xx.detach().cpu().contiguous().numpy().tobytes());fingerprint.update(torch.as_tensor(yy).cpu().numpy().tobytes())
        identity=fingerprint.hexdigest()
        ref=c.signal_history.get(key)
        if not isinstance(ref,dict) or ref.get("probe")!=identity:
            ref={"probe":identity,"loss":loss,"reference_round":cost.get("observation_round")};c.signal_history[key]=ref
        values[k]=loss;changes[k]=loss-ref["loss"]
        cost.setdefault("replay_reference_round_by_task",{})[k]=ref["reference_round"]
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
    cost={"observation_round":r,"forward_examples":0,"backward_examples":0,"seconds":0.,"replay_loss_by_task":{},"replay_loss_change_by_task":{}}
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
    if kind=="tier_a_cached_update":
        cache=c.signal_history.get("tier_a_update_cache")
        age=r-cache["round"] if cache is not None else None
        usable=cache is not None and cache["task"]==t and 1<=age<=ex.args.sync_bank_age
        cost["cached_update_age"]=age if usable else None
        cost["cached_update_source_round"]=cache["round"] if usable else None
        if usable:
            put("completed_update","current",len(cache["values"]),"relative",cache["samples"],cache["values"])
            from dataclasses import replace
            return replace(finish("cached_native_parameter_update")[0],window=f"cached_update:{cache['round']}",model_version=cache["start_version"]),cost
        return finish("cached_native_parameter_update")
    if kind=="tier_a_replay_loss":
        for k in range(t):
            loss=cost["replay_loss_by_task"].get(k)
            change=cost["replay_loss_change_by_task"].get(k,0.)
            v=[loss/(1+loss),.5*(1+change/(1+abs(change)))] if loss is not None else []
            put(f"past_loss_{k}","past",2,"unit_l1",min(32,len(c.task_memories.get(k,[]))) if loss is not None else 0,v)
        return finish("fixed_within_task_replay_loss_v1")
    if kind=="tier_a_distributions":
        current=[0]*c.classes_per_task
        for _,label in data:current[int(label)%c.classes_per_task]+=1
        put("current_labels","current",c.classes_per_task,"js",sum(current),
            [v/sum(current) for v in current] if sum(current) else [])
        past=[0]*(c.num_tasks*c.classes_per_task)
        for k,items in c.task_memories.items():
            if k<t:
                for _,label in items:past[k*c.classes_per_task+int(label)%c.classes_per_task]+=1
        put("historical_labels","past",len(past),"js",sum(past),[v/sum(past) for v in past] if sum(past) else [])
        return finish("current_and_pooled_replay_label_distributions_v1")
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
    if kind in ("tier_a_gradients","tier_a_interference"):
        model.zero_grad(set_to_none=True)
        for xx,yy in zip(x.split(c.batch_size),y.split(c.batch_size)):
            (torch.nn.functional.cross_entropy(model(xx,t),yy,reduction="sum")/len(x)).backward()
        current=ex.sketch((n,p.grad) for n,p in model.named_parameters() if p.grad is not None) if kind=="tier_a_gradients" else None
        current_shared=ex.sketch((n,p.grad) for n,p in model.named_parameters() if p.grad is not None and not n.startswith("heads."))
        if kind=="tier_a_gradients":put("current_gradient","current",len(current),"relative",len(x),current)
        cost["forward_examples"]+=len(x);cost["backward_examples"]+=len(x)
        if kind=="tier_a_interference":
            details={}
            for k in range(t):
                if k not in groups:
                    put(f"interference_{k}","past",2,"unit_l1",0,[])
                    continue
                xx,yy=groups[k]
                model.zero_grad(set_to_none=True)
                for bx,by in zip(xx.split(c.batch_size),yy.split(c.batch_size)):
                    (torch.nn.functional.cross_entropy(model(bx,k),by,reduction="sum")/len(xx)).backward()
                old=ex.sketch((n,p.grad) for n,p in model.named_parameters() if p.grad is not None and not n.startswith("heads."))
                dot=sum(a*b for a,b in zip(current_shared,old))
                den=math.sqrt(sum(v*v for v in current_shared)*sum(v*v for v in old))
                cosine=max(-1.,min(1.,dot/den)) if den else 0.
                risk=max(0.,-dot)
                details[k]={"cosine":cosine,"negative_inner_product":risk}
                put(f"interference_{k}","past",2,"unit_l1",len(xx),[(1-cosine)/2,risk/(1+risk)])
                cost["forward_examples"]+=len(xx);cost["backward_examples"]+=len(xx)
            cost["gradient_interference_by_task"]=details
            return finish("per_past_task_shared_gradient_interference_v1")
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
    if kind not in ("tier_a_features","tier_a_prototypes"):raise ValueError("unknown high-priority signal")
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
            moments=kind=="tier_a_features"
            v=(torch.cat((q.mean(0),q.square().mean(0))) if moments else q.mean(0)).tolist() if n else []
            put(f"task_{k}_class_{label}","current" if k==t else "past",2*dim if moments else dim,"moments" if moments else "relative",n,v)
    return finish(f"frozen_bootstrap_encoder:{ex.args.seed}:{dim}")

def cache_completed_update(ex,c,start,t,r,version):
    """Cache actual completed local parameter delta BEFORE model download."""
    delta=ex.after(c,start,t,r,version)[0]
    if not delta.valid():raise ValueError("invalid completed local update")
    c.signal_history["tier_a_update_cache"]={"round":r,"task":t,"start_version":version,
        "samples":c.current_samples,"values":delta.counts}
