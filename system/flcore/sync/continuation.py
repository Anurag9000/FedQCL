"""Offline continuation using a common full-participation policy after omission.
Future windows are revealed only as the isolated branch clock advances. No branch
results or held-out labels enter the live selector.
"""
import copy
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

def continue_branch(server,originals,candidates,datasets,controller,subset,r,task,horizon,complete,device,classes):
    from .engine import aggregate,download,evaluate,aggregation_weights
    args=controller.args
    selected=set(subset)
    clients=[copy.deepcopy(candidates[i] if i in selected else c) for i,c in enumerate(originals)]
    weights=aggregation_weights(args,clients,datasets,subset,complete)
    model=aggregate(server,[clients[i] for i in subset],weights,complete) if subset else copy.deepcopy(server)
    version=controller.model_version+bool(subset)
    held={i:controller.local[i].model_version for i in range(len(clients))}
    last={i:controller.local[i].last_sync for i in range(len(clients))}
    for i in subset:download(clients[i],model,complete);held[i]=version;last[i]=r
    rpt=(args.global_rounds+1)//args.tasks
    if args.sync_local_capture and (r+1)%rpt==0:
        for i in set(range(len(clients)))-selected:clients[i].capture(task,r,datasets[i])
    data_root=Path("../dataset")/args.dataset
    validation={}
    trace=[]
    def evaluate_at(round_id,t):
        for u in range(t+1):
            if u not in validation:
                validation[u]=[]
                for i in range(len(clients)):
                    with np.load(data_root/f"task_{u}"/"validation"/f"{i}.npz",allow_pickle=True) as z:
                        d=z["data"].item()
                    validation[u].append(DataLoader(list(zip(torch.from_numpy(d["x"]),torch.from_numpy(d["y"]))),batch_size=args.batch_size,shuffle=False))
        trace.extend(evaluate(model,validation,t,round_id,version,device,classes))
    evaluate_at(r,task)
    end=min(r+horizon-1,args.global_rounds)
    for rr in range(r+1,end+1):
        tt=rr//rpt
        available={c.id:c.reveal(tt,rr) for c in clients}
        from .policies import baseline_select,due_clients
        ids=tuple(c.id for c in clients);policy=getattr(args,"sync_continuation_policy","full")
        roster=baseline_select(policy,ids,due_clients(ids,last,rr,args.sync_max_skip),last,min(args.sync_k,len(ids)),args.seed,rr)
        for i in roster:
            c=clients[i]
            if args.sync_refresh and held[c.id]!=version:
                download(c,model,complete);held[c.id]=version
            c.train_round(rr,tt,available[c.id])
        weights=aggregation_weights(args,clients,available,roster,complete)
        model=aggregate(model,[clients[i] for i in roster],weights,complete);version+=bool(roster)
        for i in roster:download(clients[i],model,complete);held[i]=version;last[i]=rr
        if args.sync_local_capture and (rr+1)%rpt==0:
            for i in set(ids)-set(roster):clients[i].capture(tt,rr,available[i])
        evaluate_at(rr,tt)
    final={x["task"]:x["accuracy"] for x in trace if x["round"]==end}
    forgetting=[max(x["accuracy"] for x in trace if x["task"]==t)-final[t] for t in final if t<end//rpt]
    return {"start_round":r,"end_round":end,"horizon":end-r+1,"policy":getattr(args,"sync_continuation_policy","full")+"_after_initial_subset",
            "final_accuracy":sum(final.values())/len(final),
            "within_horizon_forgetting":sum(forgetting)/len(forgetting) if forgetting else 0.,
            "task_round_metrics":trace}
