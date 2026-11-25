"""Prespecified profiles and joint exact client-window allocation."""
import numpy as np

def validate_spec(s,n,tasks,windows):
    if s["schema"]!=1 or (s["clients"],s["tasks"],s["windows"])!=(n,tasks,windows):
        raise ValueError("controlled stream dimensions")
    if s["domain"]!="brightness" or not 2<=s["domain_bins"]<=10:
        raise ValueError("unsupported domain")
    if s["mode"] not in ("balanced_dynamic","dirichlet","custom","named","split_dynamic"):raise ValueError("stream mode")
    if not np.isfinite(s["alpha"]) or s["alpha"]<=0 or not 0<=s["temporal_persistence"]<=1:
        raise ValueError("alpha/persistence")
    if any(not 0<=s[k]<=1 for k in ("label_specialization","domain_specialization")):
        raise ValueError("specialization")
    if s["mode"]=="balanced_dynamic" and (n!=5 or windows!=10 or s["domain_bins"]!=5 or s["regimes"]!=["similar","clustered","separated","reassigned","conditional"]):
        raise ValueError("balanced schedule contract")
    if "alpha_schedule" in s and (len(s["alpha_schedule"]) not in (windows,tasks*windows) or any(not np.isfinite(x) or x<=0 for x in s["alpha_schedule"])):raise ValueError("alpha schedule")
    if s["mode"]=="split_dynamic" and (n,windows,s["domain_bins"])!=(10,10,5):raise ValueError("paired scale requires ten clients, ten windows, five domains")
    if s["mode"]=="named" and (n,windows,s["domain_bins"])!=(5,10,5):raise ValueError("named settings require five clients, ten windows, five domains")
    weights=np.asarray(s.get("workload_weights",[1]*n),float)
    if weights.shape!=(n,) or not np.isfinite(weights).all() or np.any(weights<=0):raise ValueError("workload weights")
    if len(set(s["counterfactual_rounds"]))!=len(s["counterfactual_rounds"]) or any(not 0<=r<tasks*windows for r in s["counterfactual_rounds"]):
        raise ValueError("counterfactual schedule")

def domain_ids(ids,pixels,bins):
    # Stable within-class equal-frequency rank using exact integer brightness.
    scores=pixels.astype(np.int64).sum(axis=(1,2,3))
    order=np.lexsort((ids,scores))
    return [ids[part] for part in np.array_split(order,bins)]

def window_workloads(spec,total):
    n,w=spec["clients"],spec["windows"]
    if spec["mode"]=="split_dynamic":
        parent=window_workloads({**spec,"mode":"balanced_dynamic","clients":5},total)
        return np.column_stack(((parent+1)//2,parent//2)).ravel()
    weights=np.asarray(spec.get("workload_weights",[1]*n),dtype=float)
    ideal=np.tile(weights/weights.sum(),w)*total/w
    rows=np.floor(ideal).astype(int)
    rows[np.argsort(-(ideal-rows),kind="stable")[:total-rows.sum()]]+=1
    return rows

def group_profiles(groups,c,strength):
    n=len(groups);p=np.full((n,c),(1-strength)/c);offset=0
    for g in sorted(set(groups)):
        ids=[i for i in range(n) if groups[i]==g];size=len(ids)*c//n
        if not size:raise ValueError("empty class group")
        p[ids,offset:offset+size]+=strength/size;offset+=size
    if offset!=c:raise ValueError("class groups not divisible")
    return p

def balanced_dirichlet(rng,n,c,alpha):
    # Fixed task inventory and equal client mass: project the Dirichlet
    # draw onto both margins. Alpha refers to the seed draw, not final TV.
    x=np.maximum(rng.dirichlet(np.full(c,alpha),n),1e-12)
    for _ in range(10000):
        x/=x.sum(1,keepdims=True);x*= (n/c)/x.sum(0,keepdims=True)
        if np.max(np.abs(x.sum(1)-1))<1e-10:return x/x.sum(1,keepdims=True)
    raise ValueError("Dirichlet balancing did not converge")

def named_profiles(spec,task,c,rng):
    n,w,d=spec["clients"],spec["windows"],spec["domain_bins"]
    kind=spec["setting"];strength=spec["label_specialization"]
    uniform=np.full((n,c),1/c);groups=[0,0,0,1,1];swapped=[1,0,0,0,1]
    base=group_profiles(groups,c,strength);other=group_profiles(swapped,c,strength)
    sep=group_profiles(list(range(n)),c,strength)
    alpha=balanced_dirichlet(rng,n,c,spec["alpha"]) if kind=="alpha" else None
    result=[];schedule=[]
    for r in range(w):
        p=uniform.copy();q=np.full((n,c,d),1/d);g=list(range(n));phase=kind
        if kind=="similar":phase="similar"
        elif kind=="moderate":p=group_profiles(g,c,.54)
        elif kind=="separated":p=sep
        elif kind in ("clustered","unequal"):p=base;g=groups;phase="clustered"
        elif kind=="four_groups":g=[0,0,1,2,3];p=group_profiles(g,c,strength)
        elif kind in ("abrupt","recurring","gradual_recurring","gradual"):
            mix={"abrupt":[0]*5+[1]*5,"recurring":[0,0,1,1,0,0,1,1,0,0],
                 "gradual_recurring":[0,.25,.5,.75,1,1,.75,.5,.25,0],
                 "gradual":np.linspace(0,1,w)}[kind][r]
            p=(1-mix)*base+mix*other;g=groups if mix<.5 else swapped
        elif kind in ("asynchronous","common_drift"):
            # Each client occupies opposite class halves for five windows,
            # conserving the global inventory even with asynchronous phases.
            for i in range(n):
                shift=i*2 if kind=="asynchronous" else 0
                side=int((r+shift)%w>=w//2)
                p[i]=(1-strength)/c
                p[i,side*(c//2):(side+1)*(c//2)]+=strength/(c//2)
        elif kind in ("conditional","conditional_dynamic"):
            phase="conditional"
            for i in range(n):
                j=(i+(1 if kind=="conditional_dynamic" and r>=5 else 0))%d
                q[i]=(1-spec["domain_specialization"])/d;q[i,:,j]+=spec["domain_specialization"]
        elif kind=="convergence":p=(1-r/(w-1))*sep+r/(w-1)*uniform
        elif kind=="divergence":p=r/(w-1)*sep+(1-r/(w-1))*uniform
        elif kind=="history_different_current_similar":
            p=sep if task<spec["tasks"]-1 else uniform;phase="separated" if task<spec["tasks"]-1 else "similar"
        elif kind=="history_similar_current_different":
            p=uniform if task<spec["tasks"]-1 else sep;phase="similar" if task<spec["tasks"]-1 else "separated"
        elif kind=="alpha":p=alpha
        else:raise ValueError("unknown named setting "+kind)
        x=p[:,:,None]*q
        transition=bool(r and np.max(np.abs(x-result[-1]))>1e-12)
        result.append(x);schedule.append({"window":r,"regime":phase,"groups":g,"transition":transition})
    return np.asarray(result),schedule

def target_profiles(spec,task,c,rng):
    if spec["mode"]=="named":return named_profiles(spec,task,c,rng)
    if spec["mode"]=="split_dynamic":
        targets,schedule=target_profiles({**spec,"mode":"balanced_dynamic","clients":5},task,c,rng)
        return np.repeat(targets,2,axis=1),[{**v,"groups":[g for g in v["groups"] for _ in range(2)]} for v in schedule]
    n,w,d=spec["clients"],spec["windows"],spec["domain_bins"]
    uniform=np.full((n,c),1/c)
    result=[];schedule=[];previous=None
    if spec["mode"]=="balanced_dynamic" and c%n:raise ValueError("classes must divide into five groups")
    for window in range(w):
        regime=spec["regimes"][(window//2+task)%5] if spec["mode"]=="balanced_dynamic" else spec["mode"]
        groups=list(range(n))
        p=uniform.copy();q=np.full((n,c,d),1/d)
        if regime=="dirichlet":
            alpha=spec.get("alpha_schedule",[spec["alpha"]]*w)
            fresh=rng.dirichlet(np.full(c,alpha[(task*w+window)%len(alpha)]),n)
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
        if regime=="custom":
            target=np.asarray(spec["target_joint"],dtype=float)[task,window]
            if target.shape!=(n,c,d) or not np.isfinite(target).all() or np.any(target<0) or not np.allclose(target.sum((1,2)),1):raise ValueError("invalid custom joint profiles")
            result.append(target)
        else:result.append(p[:,:,None]*q)
        schedule.append({"window":window,"regime":regime,"groups":groups,
                         "transition":window%2==1 and regime=="reassigned"})
    return np.asarray(result),schedule

def allocate_joint(rows,cols,target,allocate):
    """Exact integer allocation with primary label and secondary domain margins.

    Conserve the same joint inventory but first minimize label-profile error.
    Then independently realize the domain profile conditional on each exact
    allocated class count. This prevents the joint L1 objective sacrificing
    class balance to improve individual domain bins.
    """
    target=np.asarray(target,float);n,c,d=target.shape
    cols=np.asarray(cols,dtype=int).reshape(c,d)
    label_target=target.sum(-1)
    label_counts=allocate(rows,cols.sum(-1),label_target,0)
    counts=np.empty((n,c,d),dtype=int)
    for j in range(c):
        mass=label_target[:,j,None]
        conditional=np.divide(target[:,j,:],mass,out=np.full((n,d),1/d),where=mass>0)
        counts[:,j,:]=allocate(label_counts[:,j],cols[j],conditional,0)
    if not np.array_equal(counts.sum(-1),label_counts) or not np.array_equal(counts.sum(0),cols):raise ValueError("joint margins")
    return counts

def joint_allocate(spec,task,train_ids,raw_data,rng,allocate,client_rows,read_pixels,target_rng=None):
    n,w,d=spec["clients"],spec["windows"],spec["domain_bins"]
    c=len(train_ids)
    sources=[domain_ids(ids,read_pixels(raw_data,ids),d) for ids in train_ids]
    cols=np.asarray([len(ids) for group in sources for ids in group])
    rows=window_workloads(spec,int(cols.sum()))
    targets,schedule=target_profiles(spec,task,c,rng if target_rng is None else target_rng)
    if spec["mode"]=="split_dynamic":
        # Identical parent-cell allocation and permutation order to S14, then
        # split each parent cell into two disjoint integer-balanced children.
        parent=allocate_joint(rows.reshape(-1,2).sum(1),cols,targets[:,::2].reshape(w*5,c,d),allocate).reshape(w,5,c,d)
        counts=np.zeros((w,n,c,d),dtype=int)
        for a in range(w):
            for i in range(5):
                flat=parent[a,i].ravel();first=flat//2
                odd=np.flatnonzero(flat%2);first[odd[:(int(flat.sum())+1)//2-int(first.sum())]]+=1
                counts[a,2*i]=first.reshape(c,d);counts[a,2*i+1]=parent[a,i]-counts[a,2*i]
        if not np.array_equal(counts.sum((2,3)).ravel(),rows):raise ValueError("paired scale workload")
    else:counts=allocate_joint(rows,cols,targets.reshape(w*n,c,d),allocate).reshape(w,n,c,d)
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
