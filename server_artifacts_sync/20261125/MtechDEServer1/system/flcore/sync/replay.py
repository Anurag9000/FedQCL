"""Bounded replay selection from already revealed examples only."""
from collections import defaultdict
import math

def select_memory(pool,budget,profile,rng):
    if budget<0:raise ValueError("negative budget")
    if profile not in ("uniform","balanced","rare"):raise ValueError("unknown memory profile")
    capacities={k:len(v) for k,v in pool.items() if len(v)}
    quotas={k:0 for k in capacities}
    remaining=min(budget,sum(capacities.values()))
    while remaining:
        for k in sorted(quotas):
            if remaining and quotas[k]<capacities[k]:
                quotas[k]+=1;remaining-=1
    result={}
    for k,q in quotas.items():
        if not q:continue
        items=pool[k]
        if profile=="uniform":indices=rng.sample(range(len(items)),q)
        else:
            groups=defaultdict(list)
            for i,(_,y) in enumerate(items):groups[int(y)].append(i)
            if profile=="balanced":
                for g in groups.values():rng.shuffle(g)
                indices=[]
                while len(indices)<q:
                    for label in sorted(groups):
                        if groups[label] and len(indices)<q:indices.append(groups[label].pop())
            else:
                keys=[(-math.log(max(rng.random(),1e-300))*len(groups[int(y)]),i)
                      for i,(_,y) in enumerate(items)]
                indices=[i for _,i in sorted(keys)[:q]]
        result[k]=[items[i] for i in indices]
    return result
