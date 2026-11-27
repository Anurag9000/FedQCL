"""Sparse ring peer graph; edge distances delivered to central arbiter.
Only direct observed edges can cover clients. This is a distinct decision-placement
variant, not fully distributed election.
"""
from dataclasses import asdict
from itertools import combinations
from .policies import Report,distance,priority

def sparse_election(transport,ids,last,due,reports,r,threshold,k,degree,solver,objective):
    ids=tuple(sorted(ids));n=len(ids)
    edges=set()
    for j,i in enumerate(ids):
        for offset in range(1,min(degree,n-1)+1):
            edges.add(tuple(sorted((i,ids[(j+offset)%n]))))
    known={}
    for i,j in sorted(edges):
        # One report per undirected edge is enough to compute a symmetric metric.
        msg=transport.send(r,"observe",i,j,"SPARSE_PEER_REPORT",asdict(reports[i]) if reports[i] else None)
        x=Report(**{**msg,"labels":tuple(msg["labels"]),"counts":tuple(msg["counts"])}) if msg else None
        d=distance(x,reports[j])
        data=transport.send(r,"observe",j,"server","SPARSE_EDGE_DISTANCE",{"i":i,"j":j,"distance":None if d is None else float(d)})
        known[i,j]=data["distance"]
    # Validity flags are charged separately, not full statistics.
    valid={}
    for i in ids:
        payload=transport.send(r,"observe",i,"server","REPORT_VALIDITY",{"valid":reports[i] is not None})
        valid[i]=payload["valid"]
    def nearest(i,s):
        candidates=[(known.get(tuple(sorted((i,j)))),j) for j in s]
        candidates=[(d,j) for d,j in candidates if d is not None]
        return min(candidates) if candidates else (None,None)
    if k is None:
        selected=list(due)
        for i in sorted(ids,key=lambda x:priority(x,last)):
            if i in selected:continue
            d,_=nearest(i,selected)
            if d is None or d>float(threshold):selected.append(i)
    else:
        k=min(k,n);mandatory=tuple(due[:k]);others=[i for i in ids if i not in mandatory]
        def cost(s):
            unknown=sum(not valid[i] for i in ids if i not in s)
            distances=[nearest(i,s)[0] for i in ids if i not in s]
            value=sum((1. if d is None else float(d>float(threshold)) if objective=="coverage" else d) for d in distances)
            return unknown,value,tuple(sorted(priority(i,last) for i in s))
        if solver=="greedy":
            selected=list(mandatory)
            while len(selected)<k:
                selected.append(min((i for i in others if i not in selected),key=lambda i:cost(selected+[i])))
        else:
            selected=list(min((mandatory+s for s in combinations(others,k-len(mandatory))),key=cost))
    if r==0:selected=list(ids)
    assignments={}
    for i in ids:
        if i not in selected:
            d,j=nearest(i,selected)
            assignments[i]={"representative":j,"distance":d,"covered":d is not None and d<=float(threshold)}
    for i in selected:
        transport.send(r,"admission","server",i,"ADMISSION_RESPONSE",{"admitted":True})
    return tuple(sorted(selected)),assignments
