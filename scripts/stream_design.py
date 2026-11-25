"""Prespecified profiles and joint exact client-window allocation."""
import numpy as np

def validate_spec(s,n,tasks,windows):
    if s["schema"]!=1 or (s["clients"],s["tasks"],s["windows"])!=(n,tasks,windows):
        raise ValueError("controlled stream dimensions")
    if s["domain"]!="brightness" or not 2<=s["domain_bins"]<=10:
        raise ValueError("unsupported domain")
    if s["mode"] not in ("balanced_dynamic","dirichlet"):raise ValueError("stream mode")
    if not np.isfinite(s["alpha"]) or s["alpha"]<=0 or not 0<=s["temporal_persistence"]<=1:
        raise ValueError("alpha/persistence")
    if any(not 0<=s[k]<=1 for k in ("label_specialization","domain_specialization")):
        raise ValueError("specialization")
    if s["mode"]=="balanced_dynamic" and (n!=5 or windows!=10 or s["domain_bins"]!=5 or s["regimes"]!=["similar","clustered","separated","reassigned","conditional"]):
        raise ValueError("balanced schedule contract")
    if len(set(s["counterfactual_rounds"]))!=len(s["counterfactual_rounds"]) or any(not 0<=r<tasks*windows for r in s["counterfactual_rounds"]):
        raise ValueError("counterfactual schedule")

def domain_ids(ids,pixels,bins):
    # Stable within-class equal-frequency rank using exact integer brightness.
    scores=pixels.astype(np.int64).sum(axis=(1,2,3))
    order=np.lexsort((ids,scores))
    return [ids[part] for part in np.array_split(order,bins)]

def target_profiles(spec,task,c,rng):
    n,w,d=spec["clients"],spec["windows"],spec["domain_bins"]
    uniform=np.full((n,c),1/c)
    result=[];schedule=[];previous=None
    if spec["mode"]=="balanced_dynamic" and c%n:raise ValueError("classes must divide into five groups")
    for window in range(w):
        regime=spec["regimes"][(window//2+task)%5] if spec["mode"]=="balanced_dynamic" else "dirichlet"
        groups=list(range(n))
        p=uniform.copy();q=np.full((n,c,d),1/d)
        if regime=="dirichlet":
            fresh=rng.dirichlet(np.full(c,spec["alpha"]),n)
            rho=spec["temporal_persistence"]
            p=fresh if previous is None else rho*previous+(1-rho)*fresh
            previous=p.copy()
        elif regime in ("clustered","reassigned","separated"):
            groups=[0,0,0,1,1] if regime!="separated" else list(range(n))
            if regime=="reassigned" and window%2:
                groups[0],groups[3]=groups[3],groups[0]
            start=0
            for g in sorted(set(groups)):
                members=[i for i in range(n) if groups[i]==g]
                size=len(members)*c//n
                for i in members:
                    p[i]=(1-spec["label_specialization"])/c
                    p[i,start:start+size]+=spec["label_specialization"]/size
                start+=size
        elif regime=="conditional":
            for i in range(n):
                q[i]=(1-spec["domain_specialization"])/d
                q[i,:,i]+=spec["domain_specialization"]
        result.append(p[:,:,None]*q)
        schedule.append({"window":window,"regime":regime,"groups":groups,
                         "transition":window%2==1 and regime=="reassigned"})
    return np.asarray(result),schedule

def joint_allocate(spec,task,train_ids,raw_data,rng,allocate,client_rows,read_pixels):
    n,w,d=spec["clients"],spec["windows"],spec["domain_bins"]
    c=len(train_ids)
    sources=[domain_ids(ids,read_pixels(raw_data,ids),d) for ids in train_ids]
    cols=np.asarray([len(ids) for group in sources for ids in group])
    rows=client_rows(int(cols.sum()),w*n,0)
    targets,schedule=target_profiles(spec,task,c,rng)
    counts=allocate(rows,cols,targets.reshape(w*n,c*d),0).reshape(w,n,c,d)
    assigned=[[] for _ in range(n)];windows=[[] for _ in range(n)]
    for j,group in enumerate(sources):
        for k,ids in enumerate(group):
            ids=rng.permutation(ids);offset=0
            for a in range(w):
                for i in range(n):
                    m=int(counts[a,i,j,k])
                    assigned[i].extend(ids[offset:offset+m].tolist())
                    windows[i].extend([a]*m);offset+=m
            if offset!=len(ids):raise AssertionError("joint inventory")
    return assigned,windows,counts,targets,schedule
