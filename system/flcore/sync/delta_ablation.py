"""Stale-base delta ablation; not ordinary FedAvg and never the primary mode."""
import copy,math
def aggregate_delta(server,clients,weights):
    import torch
    if len(clients)!=len(weights) or any(not math.isfinite(w) or w<=0 for w in weights):raise ValueError("invalid delta weights")
    result=copy.deepcopy(server);total=sum(weights)
    with torch.no_grad():
        state={}
        for key,base in server.state_dict().items():
            if base.is_floating_point() or base.is_complex():
                out=base.clone()
                for c,w in zip(clients,weights):
                    out.add_(c.model.state_dict()[key].to(base.device)-c.delta_base[key].to(base.device),alpha=w/total)
                if key.endswith("running_var"):out.clamp_min_(0.)
                if not torch.isfinite(out).all():raise ValueError("nonfinite delta aggregation")
            else:
                out=torch.stack([base,*[c.model.state_dict()[key].to(base.device) for c in clients]]).amax(0)
            state[key]=out
        result.load_state_dict(state)
    return result
