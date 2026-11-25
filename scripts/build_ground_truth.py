"""Independent reconstruction of the empirical controlled stream from shards."""
import hashlib,json
from pathlib import Path
import numpy as np

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def normalize(x):
    x=np.asarray(x,dtype=float);s=x.sum(axis=-1,keepdims=True)
    return np.divide(x,s,out=np.zeros_like(x),where=s>0)
def tv(p,q):return .5*np.abs(p-q).sum(axis=-1)
def pairwise(p):
    from scipy.spatial.distance import cdist
    return .5*cdist(p,p,metric="cityblock")
def js(p,q):
    p,q=np.broadcast_arrays(p,q)
    m=(p+q)/2
    def kl(a):
        ratio=np.divide(a,m,out=np.ones_like(a),where=(a>0)&(m>0))
        return (a*np.log2(ratio,where=ratio>0,out=np.zeros_like(a))).sum(-1)
    return (kl(p)+kl(q))/2
def conditional_distance(a,b):
    pa=a.sum(-1);pb=b.sum(-1)
    support=np.minimum(pa,pb);mass=support.sum()
    if not mass:return np.nan
    return float((support*tv(normalize(a),normalize(b))).sum()/mass)
def require(ok,msg):
    if not ok:raise ValueError(msg)

def reconstruct(root,manifest):
    spec=manifest["controlled_stream"]
    n,t,c,w,d=manifest["clients"],manifest["tasks"],manifest["classes_per_task"],manifest["windows"],spec["domain_bins"]
    joint=np.zeros((t*w,n,t*c,d),dtype=np.int64);requested=np.zeros_like(joint,dtype=float)
    schedule=[];allocation=[];tol=spec["tolerances"]
    for task in range(t):
        chunks=[]
        for i in range(n):
            with np.load(root/f"task_{task}/train/{i}.npz",allow_pickle=True) as z:data=z["data"].item()
            # Recover exact uint8 pixels from normalized values; rank independently.
            x=np.rint((data["x"]*np.asarray(manifest["normalization"]["std"])[None,:,None,None]+np.asarray(manifest["normalization"]["mean"])[None,:,None,None])*255)
            require(np.all((x>=0)&(x<=255)),"pixel range")
            chunks.extend(zip(data["ids"],data["y"],data["window"],np.full(len(x),i),x.astype(np.int64).sum((1,2,3))))
        arr=np.asarray(chunks,dtype=np.int64)
        for label in range(task*c,(task+1)*c):
            a=arr[arr[:,1]==label];order=np.lexsort((a[:,0],a[:,4]))
            for k,part in enumerate(np.array_split(order,d)):
                for item in a[part]:
                    joint[task*w+item[2],item[3],label,k]+=1
                    allocation.append([int(item[0]),task*w+int(item[2]),int(item[3]),label,k])
        m=manifest["tasks_data"][task];sl=slice(task*w,(task+1)*w);cs=slice(task*c,(task+1)*c)
        require(np.array_equal(joint[sl,:,cs,:],m["joint_counts"]),"joint count mismatch")
        from stream_design import target_profiles
        expected,expected_schedule=target_profiles(spec,task,c,np.random.default_rng(np.random.SeedSequence([manifest["seed"],task,3])))
        require(np.allclose(expected,m["requested_joint"]) and expected_schedule==m["schedule"],"requested profile/schedule differs from prespecified design")
        requested[sl,:,cs,:]=expected
        for entry in m["schedule"]:schedule.append(dict(entry,round=task*w+entry["window"],task=task))
    counts=joint.sum(-1);rows=counts.sum(-1);require(np.all(rows>0),"empty window")
    from stream_design import window_workloads
    for task in range(t):
        actual=rows[task*w:(task+1)*w].ravel()
        require(np.array_equal(actual,window_workloads(spec,int(actual.sum()))),"client-window workload mismatch")
    p=normalize(counts);q=normalize(joint.reshape(t*w,n,-1))
    target=requested.sum(-1);targetq=requested.reshape(t*w,n,-1)
    require(np.allclose(target.sum(-1),1),"target normalization")
    label_error=tv(p,target);joint_error=tv(q,targetq)
    require(label_error.max()<=tol["label_tv"] and joint_error.max()<=tol["joint_tv"],"target realization tolerance exceeded")
    spatial=np.asarray([1-pairwise(x) for x in p]);joint_spatial=np.asarray([1-pairwise(x) for x in q])
    for r,entry in enumerate(schedule):
        regime=entry["regime"];groups=entry["groups"]
        for i in range(n):
            for j in range(i):
                dist=1-spatial[r,i,j]
                if regime in ("similar","conditional"):require(dist<=tol["near_pair_tv"],f"near label regime {r}")
                if regime in ("clustered","reassigned"):
                    require(dist<=tol["within_cluster_tv"] if groups[i]==groups[j] else dist>=tol["between_cluster_tv"],f"cluster regime {r}")
                if regime in ("separated","four_groups"):
                    require(dist<=tol["within_cluster_tv"] if groups[i]==groups[j] else dist>=tol["separated_tv"],f"separated regime {r}")
                if regime=="conditional" and (spec["mode"]!="split_dynamic" or i//2!=j//2):require(1-joint_spatial[r,i,j]>=tol["conditional_joint_tv"],f"conditional regime {r}")
    # Validate realized pairwise structure relative to requested pairwise structure.
    target_dist=np.asarray([pairwise(x) for x in target]);realized_dist=1-spatial
    require(np.max(np.abs(target_dist-realized_dist))<=2*tol["label_tv"]+1e-12,"pairwise target mismatch")
    joint_prob=q.reshape(t*w,n,t*c,d)
    conditional=np.asarray([[[conditional_distance(a,b) for b in x] for a in x] for x in joint_prob])
    flat=p.reshape(t*w*n,-1);flatq=q.reshape(t*w*n,-1)
    temporal=np.asarray([1-pairwise(p[:,i]) for i in range(n)])
    federation=normalize(counts.sum(1))
    cumulative=np.cumsum(counts,axis=0)
    arrays=dict(counts=counts,joint_counts=joint,requested_joint=requested,requested_labels=target,realized_labels=p,
        realized_joint=joint_prob,spatial_label=spatial,spatial_joint=joint_spatial,spatial_conditional=1-conditional,
        spatial_js=np.asarray([js(x[:,None,:],x[None,:,:]) for x in p]),
        spatiotemporal_label=1-pairwise(flat),spatiotemporal_joint=1-pairwise(flatq),temporal_label=temporal,
        federation_label=federation,federation_distance=tv(p,federation[:,None,:]),
        cumulative_counts=cumulative,cumulative_label=normalize(cumulative),
        requested_spatial_label=np.asarray([1-pairwise(x) for x in target]),requested_spatial_joint=np.asarray([1-pairwise(x) for x in targetq]),
        requested_spatiotemporal_label=1-pairwise(target.reshape(t*w*n,-1)),requested_spatiotemporal_joint=1-pairwise(targetq.reshape(t*w*n,-1)),
        temporal_joint=np.asarray([1-pairwise(q[:,i]) for i in range(n)]),
        temporal_change=tv(p[1:],p[:-1]),joint_temporal_change=tv(q[1:],q[:-1]),label_error=label_error,joint_error=joint_error)
    meta={"schema":1,"manifest_id":manifest["manifest_id"],"specification":spec,"schedule":schedule,
          "allocation_index_columns":["source_sample_id","round","client","global_class","brightness_rank_bin"],
          "stream_spec_sha256":manifest["controlled_stream_sha256"],
          "axes":{"counts":["round","client","global_class"],"joint_counts":["round","client","global_class","brightness_rank_bin"],"spatiotemporal":"round-major, client-minor"},
          "metrics":{"similarity":"1-total variation","spatial_js":"base-2 JS divergence (distance)","conditional":"overlap-weighted within-class TV; NaN if no shared classes"},
          "setting_id":spec.get("setting_id","S14"),"max_pairwise_label_error":float(np.max(np.abs(target_dist-realized_dist))),
          "max_label_tv":float(label_error.max()),"max_joint_tv":float(joint_error.max())}
    return arrays,meta,np.asarray(sorted(allocation),dtype=np.int64)

def build(root,manifest):
    root=Path(root);arrays,meta,allocation=reconstruct(root,manifest)
    dest=root/"ground_truth";dest.mkdir(exist_ok=True,mode=0o700);dest.chmod(0o700)
    np.savez_compressed(dest/"distributions.npz",**arrays)
    (dest/"specification.json").write_text(json.dumps(meta,sort_keys=True,indent=2,allow_nan=False))
    np.savez_compressed(dest/"allocation_index.npz",allocation=allocation)
    hashes={name:digest(dest/name) for name in ("distributions.npz","specification.json","allocation_index.npz")}
    for name in hashes:(dest/name).chmod(0o600)
    return {"ground_truth_hash":hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),"ground_truth_files":hashes}

def verify_bundle(root,receipt):
    hashes=receipt["ground_truth_files"]
    return all(digest(Path(root)/"ground_truth"/name)==value for name,value in hashes.items()) and hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()==receipt["ground_truth_hash"]
