"""Isolated single-round omission branches using validation data only."""
import copy
from itertools import combinations
import random
import numpy as np
import torch

def analyze(server,clients,datasets,task,r,ctl,actual,loaders,complete,device,classes):
    from .engine import aggregate,evaluate,download
    # Candidate training is isolated and reused only within this single round.
    # Under the sequential native client adapter, training uses its own starting
    # model, state, dataset, and deterministic per-client/round seeds.
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),
         torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    candidates=[]
    try:
        for original in clients:
            c=copy.deepcopy(original)
            i=c.id
            if ctl.args.sync_refresh and ctl.local[i].model_version!=ctl.model_version:
                download(c,server,complete)
            c.train_round(r,task,datasets[i])
            candidates.append(c)
        ids=tuple(range(len(clients)))
        k=min(3,len(ids))
        subsets=set(combinations(ids,k))
        subsets.add(ids)
        subsets.add(tuple(sorted(actual)))
        subsets.update(tuple(j for j in ids if j!=i) for i in ids)
        subsets.update(tuple(j for j in ids if j not in pair) for pair in combinations(ids,2))
        outcomes=[]
        for s in sorted(subsets,key=lambda x:(len(x),x)):
            weights=[len(datasets[i]) if complete else candidates[i].train_samples for i in s]
            m=aggregate(server,[candidates[i] for i in s],weights,complete) if s else copy.deepcopy(server)
            rows=evaluate(m,loaders,task,r,ctl.model_version+bool(s),device,classes)
            cur=rows[-1]["loss"]
            past=sum(x["loss"] for x in rows[:-1])/task if task else None
            outcomes.append({"round":r,"task":task,"subset":list(s),"size":len(s),
                             "actual":s==tuple(sorted(actual)),"current_loss":cur,"past_loss":past,
                             "accuracy":sum(x["accuracy"] for x in rows)/len(rows)})
            del m
        full=next(x for x in outcomes if x["subset"]==list(ids))
        for row in outcomes:
            row["current_harm"]=row["current_loss"]-full["current_loss"]
            row["past_harm"]=None if task==0 else row["past_loss"]-full["past_loss"]
        return outcomes
    finally:
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
        if rng[3]:torch.cuda.set_rng_state_all(rng[3])
