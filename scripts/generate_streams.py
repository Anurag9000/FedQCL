#!/usr/bin/env python3
"""Generate disjoint CIFAR-100 Task-IL shards and explicit causal windows."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def allocate(rows, cols, target, floor=0):
    """Integer transportation with fixed row/column totals. Greedy residual
    allocation minimizes local deviation, not a claim of global optimality."""
    rows=np.asarray(rows,dtype=int); cols=np.asarray(cols,dtype=int)
    n,c=len(rows),len(cols)
    if rows.sum()!=cols.sum() or np.any(rows<c*floor) or np.any(cols<n*floor):
        raise ValueError("infeasible inventory or coverage floor")
    x=np.full((n,c),floor,dtype=int)
    rr=rows-x.sum(1); cc=cols-x.sum(0)
    ideal=np.asarray(target)*rows[:,None]
    if ideal.shape!=x.shape or not np.all(np.isfinite(ideal)) or np.any(ideal<0):
        raise ValueError("invalid profiles")
    # Allocate large feasible chunks toward the ideal first, then exact residuals.
    for i,j in sorted(np.ndindex(x.shape),key=lambda ij:ideal[ij]-x[ij],reverse=True):
        q=min(rr[i],cc[j],max(0,int(ideal[i,j])-x[i,j]))
        x[i,j]+=q; rr[i]-=q; cc[j]-=q
    while rr.sum():
        i,j=max(((i,j) for i in range(n) for j in range(c) if rr[i] and cc[j]),
                key=lambda ij:(ideal[ij]-x[ij],-ij[0],-ij[1]))
        q=min(rr[i],cc[j])
        x[i,j]+=q; rr[i]-=q; cc[j]-=q
    assert np.array_equal(x.sum(0),cols) and np.array_equal(x.sum(1),rows)
    return x

def profiles(regime,n,c,t,rng):
    p=np.ones((n,c))
    if regime=="near":
        p+=rng.uniform(0,.15,p.shape)
    elif regime in ("clustered","changing"):
        for i in range(n):
            group=(i+(t if regime=="changing" else 0))%2
            p[i,np.arange(c)%2==group]=8
    elif regime=="separated":
        for i in range(n):
            p[i,np.arange(c)%n==i]=15
    elif regime=="dirichlet":
        return rng.dirichlet(np.ones(c)*10,n)
    elif regime not in ("equal","conditional","drift"):
        raise ValueError(regime)
    return p/p.sum(1,keepdims=True)

def similarity(p):
    return 1-.5*np.abs(p[:,None,:]-p[None,:,:]).sum(2)

def desired_profiles(matrix,rows,cols,seed,floor=0):
    from scipy.optimize import minimize
    n,c=len(rows),len(cols)
    matrix=np.asarray(matrix,float)
    if matrix.shape!=(n,n) or not np.allclose(matrix,matrix.T) or not np.allclose(np.diag(matrix),1) or np.any((matrix<0)|(matrix>1)):
        raise ValueError("invalid similarity matrix")
    d=1-matrix
    if any(d[i,j]>d[i,k]+d[k,j]+1e-10 for i in range(n) for j in range(n) for k in range(n)):
        raise ValueError("target violates triangle inequality")
    rows=np.asarray(rows); cols=np.asarray(cols)
    def objective(z):
        return float(((similarity(z.reshape(n,c))-matrix)**2).sum())
    def constraints(z):
        p=z.reshape(n,c)
        return np.r_[p.sum(1)-1,(rows@p-cols)[:-1]/rows.sum()]
    rng=np.random.default_rng(seed)
    best=None
    for _ in range(5):
        p=rng.dirichlet(np.ones(c),n)
        result=minimize(objective,p.ravel(),method="SLSQP",
                        bounds=[(floor/rows[i],1) for i in range(n) for j in range(c)],
                        constraints={"type":"eq","fun":constraints},
                        options={"maxiter":500,"ftol":1e-10})
        if np.max(np.abs(constraints(result.x)))<1e-6 and (best is None or result.fun<best.fun):
            best=result
    if best is None:
        raise RuntimeError("solver did not find feasible profiles; not proof of infeasibility")
    return best.x.reshape(n,c)

def write_shard(root,task,split,i,x,y,ids,windows):
    path=root/f"task_{task}"/split
    path.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path/f"{i}.npz",data={"x":x,"y":y,"ids":ids,"window":windows})

def generate(args):
    from torchvision.datasets import CIFAR100
    raw=CIFAR100(args.raw_root,train=True,download=args.download)
    test=CIFAR100(args.raw_root,train=False,download=args.download)
    root=Path(args.output).resolve()
    if (root/"manifest.json").exists():
        raise FileExistsError("dataset already generated")
    root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    n=args.clients; c=20
    if n<1 or args.windows<1 or args.validation_per_class<1 or args.validation_per_class>=500:
        raise ValueError("invalid dimensions")
    train_y=np.asarray(raw.targets); test_y=np.asarray(test.targets)
    # Match CIFAR-100 normalization used in standard preprocessing; stored float32 CHW.
    def images(data,ids):
        x=data[ids].astype(np.float32).transpose(0,3,1,2)/255
        return (x-np.array([.5071,.4867,.4408],np.float32)[None,:,None,None])/np.array([.2675,.2565,.2761],np.float32)[None,:,None,None]
    manifest={"seed":args.seed,"regime":args.regime,"clients":n,"tasks":5,
              "windows":args.windows,"validation_per_class":args.validation_per_class,
              "normalization":{"mean":[.5071,.4867,.4408],"std":[.2675,.2565,.2761]},
              "tasks_data":[]}
    seen=set()
    for t in range(5):
        labels=range(t*c,(t+1)*c)
        train_ids=[]; dev_ids=[]
        for y in labels:
            perm=rng.permutation(np.flatnonzero(train_y==y))
            dev_ids.append(perm[:args.validation_per_class])
            train_ids.append(perm[args.validation_per_class:])
        cols=np.array([len(x) for x in train_ids])
        rows=np.full(n,cols.sum()//n); rows[:cols.sum()%n]+=1
        target=profiles(args.regime,n,c,t,rng)
        requested_similarity=similarity(target)
        if args.similarity:
            requested_similarity=np.asarray(json.loads(Path(args.similarity).read_text()))
            target=desired_profiles(requested_similarity,rows,cols,args.seed+t,args.floor)
        if args.counts:
            counts=np.asarray(json.loads(Path(args.counts).read_text()),dtype=int)
            if counts.shape!=(n,c) or np.any(counts<args.floor) or not np.array_equal(counts.sum(0),cols) or not np.array_equal(counts.sum(1),rows):
                raise ValueError("invalid exact counts")
        else:
            counts=allocate(rows,cols,target,args.floor)
        assigned=[[] for _ in range(n)]
        for j,ids in enumerate(train_ids):
            if args.regime=="conditional":
                # Rank within-class samples by brightness before assigning disjoint domains.
                ids=ids[np.argsort(raw.data[ids].mean((1,2,3)),kind="stable")]
            cursor=0
            for i in range(n):
                assigned[i].extend(ids[cursor:cursor+counts[i,j]].tolist())
                cursor+=counts[i,j]
        for i,items in enumerate(assigned):
            ids=np.array(items,dtype=int)
            if seen.intersection(ids.tolist()) or len(set(ids.tolist()))!=len(ids):
                raise AssertionError("duplicate train allocation")
            seen.update(ids.tolist())
            if args.regime=="drift":
                # Deterministic gradual class-order change across disjoint windows.
                ids=ids[np.argsort((train_y[ids]-t*c)+rng.uniform(0,4,len(ids)),kind="stable")]
            else:
                ids=rng.permutation(ids)
            windows=np.empty(len(ids),dtype=int)
            for w,index in enumerate(np.array_split(np.arange(len(ids)),args.windows)):
                windows[index]=w
            write_shard(root,t,"train",i,images(raw.data,ids),train_y[ids],ids,windows)
        for split,source,yarray,groups in [
            ("validation",raw.data,train_y,dev_ids),
            ("test",test.data,test_y,[rng.permutation(np.flatnonzero(test_y==y)) for y in labels])]:
            buckets=[[] for _ in range(n)]
            for ids in groups:
                for i,piece in enumerate(np.array_split(ids,n)):
                    buckets[i].extend(piece.tolist())
            for i,items in enumerate(buckets):
                ids=np.asarray(items,dtype=int)
                if split=="validation":
                    if seen.intersection(items):
                        raise AssertionError("train/validation overlap")
                    seen.update(items)
                write_shard(root,t,split,i,images(source,ids),yarray[ids],ids,np.zeros(len(ids),dtype=int))
        realized=counts/rows[:,None]
        manifest["tasks_data"].append({"task":t,"counts":counts.tolist(),"requested_profiles":target.tolist(),
            "realized_profiles":realized.tolist(),"requested_similarity":requested_similarity.tolist(),
            "realized_similarity":similarity(realized).tolist(),
            "similarity_max_abs_error":float(np.max(np.abs(similarity(realized)-requested_similarity)))})
    if len(seen)!=len(raw.targets):
        raise AssertionError("incomplete train inventory")
    manifest["manifest_id"]=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(manifest["manifest_id"])

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--raw-root",default="data/raw")
    p.add_argument("--download",action="store_true")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--clients",type=int,default=5)
    p.add_argument("--windows",type=int,default=10)
    p.add_argument("--floor",type=int,default=1)
    p.add_argument("--validation-per-class",type=int,default=50)
    p.add_argument("--regime",choices=["near","clustered","separated","changing","dirichlet","equal","conditional","drift"],default="near")
    p.add_argument("--similarity")
    p.add_argument("--counts")
    generate(p.parse_args())
