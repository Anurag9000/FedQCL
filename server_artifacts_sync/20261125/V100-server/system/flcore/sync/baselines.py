"""Literature-inspired comparators, not official reproductions.
Stratified sample-mass draws use multiplicities. Gradient facility selection
is a sketch-based adaptation of DivFL.
"""
from collections import Counter
from itertools import accumulate
import random,math
from .policies import priority,distance

def mass_stratified(ids,weights,k,seed,r):
    ids=tuple(sorted(ids,key=lambda i:(weights[i],i)))
    total=sum(weights[i] for i in ids)
    if not ids:return (),{}
    if total<=0 or k<1:raise ValueError("invalid mass sampling")
    cumulative=list(accumulate(weights[i]/total for i in ids));cumulative[-1]=1.
    rng=random.Random(f"{seed}:mass_stratified:{r}");multiplicity=Counter()
    for j in range(k):
        point=(j+rng.random())/k
        idx=next(i for i,bound in enumerate(cumulative) if point<bound)
        multiplicity[ids[idx]]+=1
    return tuple(sorted(multiplicity)),dict(multiplicity)

def gradient_facility(ids,last,due,reports,k):
    ids=tuple(sorted(ids));k=min(k,len(ids));selected=list(due[:k])
    def score(s):
        unknown=0;cost=0.
        for i in ids:
            if i in s:continue
            x=reports.get(i);candidates=[]
            if x is not None:
                for j in s:
                    y=reports.get(j)
                    if distance(x,y) is not None:
                        candidates.append(math.sqrt(sum((a-b)**2 for a,b in zip(x.counts,y.counts))))
            if not candidates:unknown+=1
            else:cost+=min(candidates)
        return unknown,cost,tuple(sorted(priority(i,last) for i in s))
    while len(selected)<k:selected.append(min((i for i in ids if i not in selected),key=lambda i:score(selected+[i])))
    return tuple(sorted(selected))
