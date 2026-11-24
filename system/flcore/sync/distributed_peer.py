"""Decentralized ring election using direct edge measurements plus bounded flooding.
No central edge arbiter. All participants compute the roster from the same
authenticated-by-simulator (not cryptographically signed) edge map. Reliable
consensus requires a connected graph and eventual delivery within the round.
"""
from dataclasses import asdict
from itertools import combinations
import hashlib,random
from .policies import Report,distance,priority
from .message_integrity import Inbox,packet,canonical

def distributed_election(transport,ids,last,due,reports,r,threshold,k,degree,solver,objective,seed,
                         dynamic=False,loss=0.,fault_mode="none"):
    ids=tuple(sorted(ids));n=len(ids)
    if not n:return (),{}
    order=list(ids)
    if dynamic:random.Random(f"{seed}:topology:{r}").shuffle(order)
    edges=set()
    for pos,i in enumerate(order):
        for offset in range(1,min(degree,n-1)+1):
            edges.add(tuple(sorted((i,order[(pos+offset)%n]))))
    neighbors={i:set() for i in ids}
    for i,j in edges:neighbors[i].add(j);neighbors[j].add(i)
    local={i:{f"v:{i}":reports[i] is not None} for i in ids}
    for i,j in sorted(edges):
        msg=transport.send(r,"observe",i,j,"SPARSE_DIRECT_REPORT",asdict(reports[i]) if reports[i] else None)
        x=Report(**{**msg,"labels":tuple(msg["labels"]),"counts":tuple(msg["counts"])}) if msg else None
        d=distance(x,reports[j])
        local[j][f"e:{i}:{j}"]=None if d is None else float(d)
    # Flood only new facts. Each sender sequence is unique for this round.
    frontier={i:dict(v) for i,v in local.items()};seq={i:0 for i in ids};inbox={i:Inbox(r) for i in ids}
    rng=random.Random(f"{seed}:gossip:{r}")
    delayed=[]
    for step in range(max(1,2*n)):
        next_frontier={i:{} for i in ids}
        pending=list(delayed);delayed=[]
        for i in ids:
            if not frontier[i]:continue
            for j in sorted(neighbors[i]):
                msg=packet(r,i,seq[i],frontier[i]);seq[i]+=1
                # First attempt may be lost; reliable charged retry follows.
                delivered=rng.random()>=loss
                received=transport.send(r,"observe",i,j,"GOSSIP_FACTS",msg,delivered=delivered)
                if received is None:received=transport.send(r,"observe",i,j,"GOSSIP_RETRY",msg)
                if fault_mode=="corrupt" and step==0:
                    bad={**received,"digest":"0"*64}
                    pending.append((j,bad))
                    received=transport.send(r,"observe",i,j,"GOSSIP_RECOVERY",msg)
                if fault_mode=="delay" and step==0:
                    delayed.append((j,received));continue
                pending.append((j,received))
                if fault_mode=="duplicate" and step==0:
                    pending.append((j,transport.send(r,"observe",i,j,"GOSSIP_DUPLICATE",msg)))
        for j,msg in reversed(pending) if fault_mode=="reorder" else pending:
            status,facts=inbox[j].accept(msg)
            if status!="accepted":continue
            for key,value in facts.items():
                if key in local[j] and local[j][key]!=value:raise RuntimeError("conflicting gossip facts")
                if key not in local[j]:
                    local[j][key]=value;next_frontier[j][key]=value
        frontier=next_frontier
        if not delayed and not any(frontier.values()):break
    expected={f"v:{i}" for i in ids}|{f"e:{i}:{j}" for i,j in edges}
    if any(set(local[i])!=expected for i in ids):raise RuntimeError("gossip deadline without consensus")
    def elect(facts):
        def nearest(i,s):
            rr=[(facts.get(f"e:{min(i,j)}:{max(i,j)}"),j) for j in s]
            rr=[(d,j) for d,j in rr if d is not None]
            return min(rr) if rr else (None,None)
        if r==0:return ids,{}
        if k is None:
            selected=list(due)
            for i in sorted(ids,key=lambda x:priority(x,last)):
                if i in selected:continue
                d,_=nearest(i,selected)
                if d is None or d>float(threshold):selected.append(i)
        else:
            size=min(k,n);mandatory=tuple(due[:size]);others=[i for i in ids if i not in mandatory]
            def cost(s):
                unknown=sum(not facts[f"v:{i}"] for i in ids if i not in s)
                values=[nearest(i,s)[0] for i in ids if i not in s]
                return unknown,sum(1. if d is None else float(d>float(threshold)) if objective=="coverage" else d for d in values),tuple(sorted(priority(i,last) for i in s))
            if solver=="greedy":
                selected=list(mandatory)
                while len(selected)<size:selected.append(min((i for i in others if i not in selected),key=lambda j:cost(selected+[j])))
            else:selected=list(min((mandatory+s for s in combinations(others,size-len(mandatory))),key=cost))
        assignment={}
        for i in ids:
            if i not in selected:
                d,j=nearest(i,selected);assignment[i]={"representative":j,"distance":d,"covered":d is not None and d<=float(threshold)}
        return tuple(sorted(selected)),assignment
    outcomes=[elect(local[i]) for i in ids]
    if any(x!=outcomes[0] for x in outcomes):raise RuntimeError("distributed election divergence")
    for i,(chosen,_) in zip(ids,outcomes):
        transport.send(r,"admission",i,"server","DISTRIBUTED_ELECTION",{"selected":chosen,"digest":hashlib.sha256(canonical(local[i])).hexdigest()})
    return outcomes[0]
