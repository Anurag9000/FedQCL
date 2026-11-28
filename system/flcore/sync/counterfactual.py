"""Isolated single-round omission branches using validation data only."""
import copy
from itertools import combinations
import random
import numpy as np
import torch

def update_geometry(server,clients,candidates,ctl):
    """Full parameter deltas, not the admission sketches; diagnostic only."""
    n=len(clients);gram=torch.zeros((n,n),dtype=torch.float64)
    before=[dict(c.model.named_parameters()) for c in clients]
    after=[dict(c.model.named_parameters()) for c in candidates]
    common=dict(server.named_parameters())
    with torch.no_grad():
        for key in common:
            # CPU chunks bound transient memory and avoid changing GPU precision flags.
            for start in range(0,common[key].numel(),65536):
                block=[]
                for i in range(n):
                    base=common[key] if (getattr(ctl.args,'sync_fresh_start',False) or (ctl.args.sync_refresh and ctl.local[i].model_version!=ctl.model_version)) else before[i][key]
                    a=after[i][key].detach().reshape(-1)[start:start+65536].cpu().double()
                    b=base.detach().reshape(-1)[start:start+65536].cpu().double()
                    block.append(a-b)
                x=torch.stack(block);gram+=x@x.T
        norms=gram.diag().clamp_min(0).sqrt()
        dist=(norms.square()[:,None]+norms.square()[None,:]-2*gram).clamp_min(0).sqrt()
        denominator=norms[:,None]+norms[None,:]
        relative=torch.where(denominator>0,dist/denominator,0.).clamp(0,1)
        relative.fill_diagonal_(0.)
        cosine=[]
        for i in range(n):
            cosine.append([float((gram[i,j]/(norms[i]*norms[j])).clamp(-1,1)) if norms[i]*norms[j]>0 else None for j in range(n)])
    return {"relative_l2":relative.tolist(),"cosine":cosine,"norms":norms.tolist()}

def parameter_discrepancy(model,full,server):
    with torch.no_grad():
        f=dict(full.named_parameters());base=dict(server.named_parameters())
        error=total=0.
        for name,p in model.named_parameters():
            x=f[name].detach();error+=float((p.detach()-x).double().square().sum())
            total+=float((x-base[name].detach()).double().square().sum())
    return {"l2":error**.5,"relative_to_full_update":(error/total)**.5 if total else None}

def analyze(server,clients,datasets,task,r,ctl,actual,loaders,complete,device,classes):
    from .engine import aggregate,evaluate,download,aggregation_weights
    from .advanced import cached_views
    from .impact import subset_features
    # Candidate training is isolated and reused only within this single round.
    # Under the sequential native client adapter, training uses its own starting
    # model, state, dataset, and deterministic per-client/round seeds.
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),
         torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    candidates=[]
    ids=tuple(range(len(clients)))
    # Capture only the pre-decision delivered evidence. For telemetry policies
    # the observed bank is not equivalent to accepted-upload evidence.
    prior_records=[x for x in ctl.bank.values() if x.round<r]
    cached=cached_views(prior_records,ids,task,r,ctl.args.sync_bank_age)
    last={i:ctl.local[i].last_sync for i in ids}
    rpt=(ctl.args.global_rounds+1)//ctl.args.tasks
    try:
        for original in clients:
            c=copy.deepcopy(original)
            i=c.id
            if (getattr(ctl.args,'sync_fresh_start',False) or (ctl.args.sync_refresh and ctl.local[i].model_version!=ctl.model_version)):
                download(c,server,complete)
            c.train_round(r,task,datasets[i])
            candidates.append(c)
        ids=tuple(range(len(clients)))
        k=len(actual) if getattr(ctl.args,"sync_controlled_stream",False) else min(3,len(ids))
        subsets=set(combinations(ids,k))
        if getattr(ctl.args,"sync_all_subsets",False):
            if len(ids)>10:raise ValueError("exhaustive counterfactual limit is 10 clients")
            subsets.update(s for k in range(len(ids)+1) for s in combinations(ids,k))
        subsets.add(ids)
        subsets.add(tuple(sorted(actual)))
        if not getattr(ctl.args,"sync_controlled_stream",False):
            subsets.update(tuple(j for j in ids if j!=i) for i in ids)
            subsets.update(tuple(j for j in ids if j not in pair) for pair in combinations(ids,2))
        controlled=getattr(ctl.args,"sync_controlled_stream",False)
        full_model=aggregate(server,candidates,aggregation_weights(ctl.args,candidates,datasets,ids,complete),complete) if controlled else None
        geometry=update_geometry(server,clients,candidates,ctl) if controlled else None
        outcomes=[]
        for s in sorted(subsets,key=lambda x:(len(x),x)):
            weights=aggregation_weights(ctl.args,candidates,datasets,s,complete)
            m=aggregate(server,[candidates[i] for i in s],weights,complete) if s else copy.deepcopy(server)
            if getattr(ctl.args,"sync_verify_branches",False) and s in (ids,tuple(sorted(actual))):
                explicit=[]
                for i in s:
                    c=copy.deepcopy(clients[i])
                    if (getattr(ctl.args,'sync_fresh_start',False) or (ctl.args.sync_refresh and ctl.local[i].model_version!=ctl.model_version)):download(c,server,complete)
                    c.train_round(r,task,datasets[i]);explicit.append(c)
                alternative=aggregate(server,explicit,weights,complete) if s else copy.deepcopy(server)
                if not all(torch.equal(v,alternative.state_dict()[k]) for k,v in m.state_dict().items()):
                    raise RuntimeError("candidate reuse differs from explicit independent branch")
                del alternative,explicit
            rows=evaluate(m,loaders,task,r,ctl.model_version+bool(s),device,classes)
            cur=rows[-1]["loss"]
            past=sum(x["loss"] for x in rows[:-1])/task if task else None
            outcomes.append({"round":r,"task":task,"subset":list(s),"size":len(s),
                             "actual":s==tuple(sorted(actual)),"current_loss":cur,"past_loss":past,
                             "features":subset_features(ids,s,cached,last,r,task,rpt,ctl.args.sync_bank_age),
                             "accuracy":sum(x["accuracy"] for x in rows)/len(rows),
                             "task_metrics":[{"task":x["task"],"loss":x["loss"],"accuracy":x["accuracy"]} for x in rows]})
            if controlled:
                outcomes[-1]["update_discrepancy"]=parameter_discrepancy(m,full_model,server)
                if s==ids:outcomes[-1]["candidate_update_geometry"]=geometry
            del m
        del full_model
        full=next(x for x in outcomes if x["subset"]==list(ids))
        for row in outcomes:
            row["current_harm"]=row["current_loss"]-full["current_loss"]
            row["past_harm"]=None if task==0 else row["past_loss"]-full["past_loss"]
        horizon=getattr(ctl.args,"sync_counterfactual_horizon",1)
        if horizon>1 and r<ctl.args.global_rounds:
            from .continuation import continue_branch
            # Prespecified actual/full and validation-oracle best/worst matched subsets.
            matched=[x for x in outcomes if x["size"]==len(actual)]
            choices={ids,tuple(sorted(actual))}
            if matched:
                objective=lambda x:x["current_loss"]+getattr(ctl.args,"sync_oracle_past_weight",0.)*(x["past_loss"] or 0.)
                choices.add(tuple(min(matched,key=objective)["subset"]))
                choices.add(tuple(max(matched,key=objective)["subset"]))
            for row in outcomes:
                if tuple(row["subset"]) in choices:
                    row["continuation"]=continue_branch(server,clients,candidates,datasets,ctl,tuple(row["subset"]),r,task,horizon,complete,device,classes)
            baseline=next(x["continuation"] for x in outcomes if tuple(x["subset"])==ids)
            for row in outcomes:
                if "continuation" in row:
                    row["continuation"]["accuracy_delta_vs_full"]=row["continuation"]["final_accuracy"]-baseline["final_accuracy"]
                    row["continuation"]["forgetting_delta_vs_full"]=row["continuation"]["within_horizon_forgetting"]-baseline["within_horizon_forgetting"]
        return outcomes
    finally:
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
        if rng[3]:torch.cuda.set_rng_state_all(rng[3])
