"""Shared research learning contract: plain FedAvg and pooled vanilla ER."""
import math
BACKBONE="fedavg_uniform_er_v1"

def sample_replay(memory,batch_size,rng):
    if batch_size<1:raise ValueError("positive replay batch required")
    pool=[(task,item) for task,items in sorted(memory.items()) for item in items]
    return rng.sample(pool,min(batch_size,len(pool)))

def replay_ce(model,groups,label_map):
    """One sample-mean CE over a pooled replay batch, across task heads."""
    import torch.nn.functional as F
    total=sum(len(y) for _,y in groups.values())
    if not total:return None
    return sum(F.cross_entropy(model(x,k),label_map(y,k),reduction="sum")
               for k,(x,y) in sorted(groups.items()))/total

def fedavg(server,clients,weights):
    """Direct weighted model mean, without stale-delta or queue correction."""
    import torch
    if len(clients)!=len(weights) or any(not math.isfinite(w) or w<=0 for w in weights):
        raise ValueError("invalid FedAvg weights")
    if not clients:return server
    total=sum(weights);states=[c.model.state_dict() for c in clients]
    with torch.no_grad():
        target={}
        for k,v in server.state_dict().items():
            if v.is_floating_point() or v.is_complex():
                x=torch.zeros_like(v)
                for w,s in zip(weights,states):x.add_(s[k].to(v.device),alpha=w/total)
                if not torch.isfinite(x).all():raise FloatingPointError("nonfinite FedAvg state")
            else:
                x=torch.stack([s[k].to(v.device) for s in states]).amax(0)
            target[k]=x
        server.load_state_dict(target)
    return server
