#!/usr/bin/env python3
"""Generate disjoint CIFAR-100 Task-IL shards and explicit causal windows."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def allocate(rows, cols, target, floor=0):
    """Exact integer transportation for minimum absolute count error.

    Each cell has three integral-capacity cost segments for the discrete absolute
    deviation objective. The transportation constraint matrix is totally unimodular;
    solve the continuous min-cost flow and verify integrality and inventory explicitly.
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    rows=np.asarray(rows,dtype=int);cols=np.asarray(cols,dtype=int)
    n,c=len(rows),len(cols);target=np.asarray(target,dtype=float)
    if floor<0 or rows.sum()!=cols.sum() or np.any(rows<c*floor) or np.any(cols<n*floor):
        raise ValueError("infeasible inventory or coverage floor")
    if target.shape!=(n,c) or not np.all(np.isfinite(target)) or np.any(target<0) or not np.allclose(target.sum(1),1):
        raise ValueError("invalid target profiles")
    rr=rows-c*floor;cc=cols-n*floor
    ideal=target*rows[:,None]-floor
    cells=[];costs=[];bounds=[]
    for i,j in np.ndindex((n,c)):
        cap=int(min(rr[i],cc[j]))
        desired=max(0.,ideal[i,j]);low=min(cap,int(np.floor(desired)))
        middle=int(low<cap and desired>low)
        for capacity,cost in ((low,-1.),(middle,1.-2*(desired-low)),(cap-low-middle,1.)):
            if capacity:
                cells.append((i,j));costs.append(cost);bounds.append((0,capacity))
    x=np.full((n,c),floor,dtype=int)
    if not cells:
        if rr.sum():raise ValueError("no feasible residual flow")
        return x
    ri=[];ci=[]
    for k,(i,j) in enumerate(cells):
        ri.extend((i,n+j));ci.extend((k,k))
    matrix=coo_matrix((np.ones(len(ri)),(ri,ci)),shape=(n+c,len(cells))).tocsr()
    solution=linprog(costs,A_eq=matrix,b_eq=np.r_[rr,cc],bounds=bounds,method="highs")
    if not solution.success:raise RuntimeError("allocation solver failed: "+solution.message)
    rounded=np.rint(solution.x)
    if np.max(np.abs(solution.x-rounded))>1e-6:raise RuntimeError("nonintegral flow solution")
    for (i,j),value in zip(cells,rounded.astype(int)):x[i,j]+=value
    if not np.array_equal(x.sum(0),cols) or not np.array_equal(x.sum(1),rows) or np.any(x<floor):
        raise RuntimeError("allocation violated inventory")
    return x

def profiles(regime,n,c,t,rng,alpha=10.):
    p=np.ones((n,c))
    if regime=="near":
        p+=rng.uniform(0,.15,p.shape)
    elif regime in ("clustered","changing"):
        for i in range(n):
            group=(i%2 if regime=="clustered" else (i%2)^int((i+t)%max(2,n)==0))
            p[i,np.arange(c)%2==group]=8
    elif regime=="separated":
        for i in range(n):
            p[i,np.arange(c)%min(c,n)==i%min(c,n)]=15
    elif regime=="dirichlet":
        return rng.dirichlet(np.ones(c)*alpha,n)
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

def sha_file(path):
    digest=hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda:file.read(1048576),b""):digest.update(block)
    return digest.hexdigest()

def load_sources(args):
    from types import SimpleNamespace
    if args.dataset in ("cifar10","cifar100"):
        from torchvision.datasets import CIFAR10,CIFAR100
        cls=CIFAR10 if args.dataset=="cifar10" else CIFAR100
        normalization=([.4914,.4822,.4465],[.2470,.2435,.2616]) if args.dataset=="cifar10" else ([.5071,.4867,.4408],[.2675,.2565,.2761])
        return cls(args.raw_root,train=True,download=args.download),cls(args.raw_root,train=False,download=args.download),5,2 if args.dataset=="cifar10" else 20,normalization
    root=Path(args.tiny_root or args.raw_root)
    train=root/"train";val=root/"val"
    if not train.is_dir() or not (val/"val_annotations.txt").is_file():
        raise FileNotFoundError("TinyImageNet requires an extracted official tiny-imagenet-200 directory")
    classes=sorted(p.name for p in train.iterdir() if p.is_dir())
    if len(classes)!=200:raise ValueError("expected 200 TinyImageNet classes")
    index={name:i for i,name in enumerate(classes)}
    tx=[];ty=[]
    for name in classes:
        for image in sorted((train/name/"images").glob("*.JPEG")):
            tx.append(image);ty.append(index[name])
    vx=[];vy=[]
    for line in sorted((val/"val_annotations.txt").read_text().splitlines()):
        file,label,*_=line.split()
        vx.append(val/"images"/file);vy.append(index[label])
    if len(tx)!=100000 or len(vx)!=10000:raise ValueError("TinyImageNet inventory mismatch")
    return SimpleNamespace(data=tx,targets=ty),SimpleNamespace(data=vx,targets=vy),10,20,([.4802,.4481,.3975],[.2770,.2691,.2821])

def read_pixels(source,ids):
    if isinstance(source,np.ndarray):return source[ids]
    from PIL import Image
    out=[]
    for i in ids:
        with Image.open(source[int(i)]) as image:
            out.append(np.asarray(image.convert("RGB").resize((64,64)),dtype=np.uint8))
    return np.stack(out) if out else np.empty((0,64,64,3),np.uint8)

def client_rows(total,n,skew):
    if skew<0 or not np.isfinite(skew):raise ValueError("invalid workload skew")
    p=np.exp(np.linspace(-skew,0,n));p/=p.sum()
    ideal=total*p;rows=np.floor(ideal).astype(int)
    for i in np.argsort(-(ideal-rows),kind="stable")[:total-rows.sum()]:rows[i]+=1
    return rows

def window_assignment(labels,classes,windows,pattern,i,rng,custom=None):
    labels=np.asarray(labels,dtype=int)
    cols=np.bincount(labels,minlength=classes)
    rows=client_rows(len(labels),windows,0)
    if not len(labels):return np.zeros(0,dtype=int),np.zeros((windows,classes),dtype=int)
    if custom is not None:target=np.asarray(custom,dtype=float)
    elif pattern=="static":
        target=np.repeat((cols/cols.sum())[None,:],windows,axis=0)
    else:
        base=np.linspace(-1,1,classes);target=[]
        for w in range(windows):
            x=w/max(1,windows-1)
            phase=(1 if x>=.5 else -1) if pattern=="abrupt" else (2*x-1) if pattern=="gradual" else np.cos(2*np.pi*x) if pattern=="recurring" else (1 if x>=(i%5+1)/6 else -1)
            p=np.exp(3*phase*base)*(cols>0);target.append(p/p.sum())
        target=np.asarray(target)
    counts=allocate(rows,cols,target,0)
    out=np.empty(len(labels),dtype=int)
    for j in range(classes):
        indices=rng.permutation(np.flatnonzero(labels==j));offset=0
        for w in range(windows):
            out[indices[offset:offset+counts[w,j]]]=w;offset+=counts[w,j]
    return out,counts

def write_shard(root,task,split,i,x,y,ids,windows):
    path=root/f"task_{task}"/split
    path.mkdir(parents=True,exist_ok=True)
    file=path/f"{i}.npz"
    np.savez_compressed(file,data={"x":x,"y":y,"ids":ids,"window":windows})
    return str(file.relative_to(root)),sha_file(file)

def generate(args):
    raw,test,tasks,c,normalization=load_sources(args)
    mean,std=normalization
    root=Path(args.output).resolve()
    if (root/"manifest.json").exists():
        raise FileExistsError("dataset already generated")
    root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    n=args.clients
    if n<1 or args.windows<1 or args.validation_per_class<1 or args.validation_per_class>=min(np.bincount(raw.targets)):
        raise ValueError("invalid dimensions")
    public_n=getattr(args,"public_probe_per_class",0)
    if public_n<0 or args.validation_per_class+public_n>=min(np.bincount(raw.targets)):
        raise ValueError("invalid disjoint public probe size")
    train_y=np.asarray(raw.targets); test_y=np.asarray(test.targets)
    # Match CIFAR-100 normalization used in standard preprocessing; stored float32 CHW.
    def images(data,ids):
        x=read_pixels(data,ids).astype(np.float32).transpose(0,3,1,2)/255
        return (x-np.array(mean,np.float32)[None,:,None,None])/np.array(std,np.float32)[None,:,None,None]
    manifest={"schema":2,"dataset":args.dataset,"seed":args.seed,"regime":args.regime,"clients":n,"tasks":tasks,"classes_per_task":c,
              "train_inventory":len(raw.targets),"test_inventory":len(test.targets),
              "windows":args.windows,"validation_per_class":args.validation_per_class,"workload_skew":args.workload_skew,
              "drift_pattern":args.drift_pattern,"conditional_domain":args.conditional_domain,
              "normalization":{"mean":mean,"std":std},"shards":{}, "tasks_data":[]}
    if public_n:manifest["public_probe_per_class"]=public_n
    spec_path=getattr(args,"controlled_stream",None)
    spec=json.loads(Path(spec_path).read_text()) if spec_path else None
    if spec:
        from stream_design import validate_spec
        validate_spec(spec,n,tasks,args.windows)
        if any(getattr(args,k,None) for k in ("profiles","similarity","counts","window_profiles","client_counts","drift_pattern","workload_skew")):
            raise ValueError("controlled stream cannot mix legacy allocation overrides")
        manifest["controlled_stream"]=spec
    elif not np.isfinite(getattr(args,"dirichlet_alpha",10.)) or getattr(args,"dirichlet_alpha",10.)<=0:
        raise ValueError("Dirichlet alpha must be positive")
    if args.regime=="dirichlet":manifest["dirichlet_alpha"]=getattr(args,"dirichlet_alpha",10.)
    seen=set()
    for t in range(tasks):
        labels=range(t*c,(t+1)*c)
        train_ids=[]; dev_ids=[]; public_ids=[]
        for y in labels:
            perm=np.random.default_rng(np.random.SeedSequence([args.seed,y,1])).permutation(np.flatnonzero(train_y==y))
            dev_ids.append(perm[:args.validation_per_class])
            public_ids.extend(perm[args.validation_per_class:args.validation_per_class+public_n].tolist())
            train_ids.append(perm[args.validation_per_class+public_n:])
        if public_n:
            ids=np.asarray(public_ids,dtype=int)
            if seen.intersection(ids.tolist()):raise AssertionError("public probe overlap")
            seen.update(ids.tolist())
            file,digest=write_shard(root,t,"public_probe",0,images(raw.data,ids),train_y[ids],ids,np.zeros(len(ids),dtype=int))
            manifest["shards"][file]=digest
        cols=np.array([len(x) for x in train_ids])
        rows=client_rows(int(cols.sum()),n,args.workload_skew)
        if args.client_counts:
            custom=np.asarray(json.loads(Path(args.client_counts).read_text()))
            rows=custom[t] if custom.ndim==2 else custom
            if rows.shape!=(n,) or not np.issubdtype(rows.dtype,np.integer) or rows.sum()!=cols.sum() or np.any(rows<c*args.floor):
                raise ValueError("invalid client workload totals")
        if spec:
            from stream_design import joint_allocate
            assigned,window_ids,joint,target_joint,schedule=joint_allocate(spec,t,train_ids,raw.data,rng,allocate,client_rows,read_pixels)
            counts=joint.sum((0,3));rows=counts.sum(1)
            target=(target_joint.sum(3)*joint.sum((2,3))[:,:,None]).sum(0)/rows[:,None]
            requested_similarity=similarity(target)
            window_counts=joint.sum(3).transpose(1,0,2).tolist()
            for i,items in enumerate(assigned):
                ids=np.asarray(items,dtype=int);windows=np.asarray(window_ids[i],dtype=int)
                order=rng.permutation(len(ids));ids=ids[order];windows=windows[order]
                if seen.intersection(ids.tolist()) or len(set(ids.tolist()))!=len(ids):raise ValueError("duplicate joint allocation")
                seen.update(ids.tolist())
                file,digest=write_shard(root,t,"train",i,images(raw.data,ids),train_y[ids],ids,windows)
                manifest["shards"][file]=digest
        else:
            target=profiles(args.regime,n,c,t,rng,getattr(args,'dirichlet_alpha',10.))
            if args.profiles:
                requested=np.asarray(json.loads(Path(args.profiles).read_text()),dtype=float)
                target=requested[t] if requested.ndim==3 else requested
            requested_similarity=similarity(target)
            if args.similarity:
                requested_similarity=np.asarray(json.loads(Path(args.similarity).read_text()))
                if requested_similarity.ndim==3:requested_similarity=requested_similarity[t]
                target=desired_profiles(requested_similarity,rows,cols,args.seed+t,args.floor)
            if args.counts:
                counts=np.asarray(json.loads(Path(args.counts).read_text()))
                if counts.ndim==3:counts=counts[t]
                if not np.issubdtype(counts.dtype,np.integer):raise ValueError("class counts must be integer")
                if counts.shape!=(n,c) or np.any(counts<args.floor) or not np.array_equal(counts.sum(0),cols) or not np.array_equal(counts.sum(1),rows):
                    raise ValueError("invalid exact counts")
            else:
                counts=allocate(rows,cols,target,args.floor)
            assigned=[[] for _ in range(n)]
            for j,ids in enumerate(train_ids):
                if args.regime=="conditional":
                    # Rank within-class samples by brightness before assigning disjoint domains.
                    pixels=read_pixels(raw.data,ids).astype(np.float32)
                    score=pixels.mean((1,2,3)) if args.conditional_domain=="brightness" else (pixels[:,:,:,0]-pixels[:,:,:,2]).mean((1,2)) if args.conditional_domain=="color" else pixels.std((1,2,3))
                    ids=ids[np.argsort(score,kind="stable")]
                cursor=0
                for i in range(n):
                    assigned[i].extend(ids[cursor:cursor+counts[i,j]].tolist())
                    cursor+=counts[i,j]
            window_counts=[]
            for i,items in enumerate(assigned):
                ids=np.array(items,dtype=int)
                if seen.intersection(ids.tolist()) or len(set(ids.tolist()))!=len(ids):
                    raise AssertionError("duplicate train allocation")
                seen.update(ids.tolist())
                ids=rng.permutation(ids)
                pattern=args.drift_pattern or ("gradual" if args.regime=="drift" else "static")
                custom=None
                if args.window_profiles:
                    requested=np.asarray(json.loads(Path(args.window_profiles).read_text()))
                    custom=requested[t,i] if requested.ndim==4 else requested[i] if requested.ndim==3 else requested
                windows,wcounts=window_assignment(train_y[ids]-t*c,c,args.windows,pattern,i,rng,custom)
                window_counts.append(wcounts.tolist())
                file,digest=write_shard(root,t,"train",i,images(raw.data,ids),train_y[ids],ids,windows)
                manifest["shards"][file]=digest
        for split,source,yarray,groups in [
            ("validation",raw.data,train_y,dev_ids),
            ("test",test.data,test_y,[np.random.default_rng(np.random.SeedSequence([args.seed,y,2])).permutation(np.flatnonzero(test_y==y)) for y in labels])]:
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
                file,digest=write_shard(root,t,split,i,images(source,ids),yarray[ids],ids,np.zeros(len(ids),dtype=int))
                manifest["shards"][file]=digest
        realized=counts/rows[:,None]
        residual=float(np.max(np.abs(similarity(realized)-requested_similarity)))
        if args.similarity_tolerance is not None and residual>args.similarity_tolerance:
            raise RuntimeError(f"task {t}: realized similarity residual {residual} exceeds tolerance; infeasibility not established")
        manifest["tasks_data"].append({"task":t,"counts":counts.tolist(),"window_counts":window_counts,"requested_profiles":target.tolist(),
            "realized_profiles":realized.tolist(),"requested_similarity":requested_similarity.tolist(),
            "realized_similarity":similarity(realized).tolist(),
            "similarity_max_abs_error":float(np.max(np.abs(similarity(realized)-requested_similarity)))})
        if spec:manifest["tasks_data"][-1].update(requested_joint=target_joint.tolist(),joint_counts=joint.tolist(),schedule=schedule)
    if len(seen)!=len(raw.targets):
        raise AssertionError("incomplete train inventory")
    manifest["manifest_id"]=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    temporary=root/"manifest.json.tmp";temporary.write_text(json.dumps(manifest,indent=2));temporary.replace(root/"manifest.json")
    print(manifest["manifest_id"])

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--dataset",choices=["cifar10","cifar100","tinyimagenet"],default="cifar100")
    p.add_argument("--tiny-root")
    p.add_argument("--workload-skew",type=float,default=0.)
    p.add_argument("--client-counts",help="Exact client row totals or per-task row totals")
    p.add_argument("--drift-pattern",choices=["static","abrupt","gradual","recurring","asynchronous"])
    p.add_argument("--window-profiles",help="Window class profiles [task,client,window,class] or broadcast variant")
    p.add_argument("--conditional-domain",choices=["brightness","color","contrast"],default="brightness")
    p.add_argument("--raw-root",default="data/raw")
    p.add_argument("--download",action="store_true")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--clients",type=int,default=5)
    p.add_argument("--windows",type=int,default=10)
    p.add_argument("--floor",type=int,default=1)
    p.add_argument("--validation-per-class",type=int,default=50)
    p.add_argument("--public-probe-per-class",type=int,default=0,help="Reserve public training-source examples before private allocation; never train or evaluate on them")
    p.add_argument("--regime",choices=["near","clustered","separated","changing","dirichlet","equal","conditional","drift"],default="near")
    p.add_argument("--controlled-stream",help="Immutable joint client-window stream specification JSON")
    p.add_argument("--dirichlet-alpha",type=float,default=10.)
    p.add_argument("--profiles",help="JSON target profile matrix or per-task matrices")
    p.add_argument("--similarity-tolerance",type=float)
    p.add_argument("--similarity")
    p.add_argument("--counts")
    generate(p.parse_args())
