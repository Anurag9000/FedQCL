#!/usr/bin/env python3
"""Build and verify private Task-IL episode shards from the existing CIFAR loader."""
import argparse
from collections import defaultdict, Counter
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"system"))
from flcore.sync.task_il_schedule import digest
from generate_streams import load_sources, read_pixels, sha_file

def atomic_json(path,obj):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False))
    os.replace(tmp,path)

def episode_path(e):
    return f'private/{e["client"]}/{e["id"]}.npz'

def generate_from_sources(plan,root,raw,test,normalization):
    root=Path(root)
    if (root/"manifest.json").exists():
        raise FileExistsError("dataset already generated")
    root.mkdir(parents=True,exist_ok=True)
    y=np.asarray(raw.targets,dtype=np.int64)
    ty=np.asarray(test.targets,dtype=np.int64)
    if set(y)!=set(range(100)) or set(ty)!=set(range(100)):
        raise ValueError("expected CIFAR-100 class inventory")
    mean,std=normalization
    def pixels(source,ids):
        x=read_pixels(source,ids).astype(np.float32).transpose(0,3,1,2)/255
        return (x-np.asarray(mean,np.float32)[None,:,None,None])/np.asarray(std,np.float32)[None,:,None,None]
    tasks=plan["tasks"]
    occurrences=defaultdict(list)
    allocated={e["id"]:[] for e in plan["episodes"]}
    for e in plan["episodes"]:
        for label in tasks[e["task"]]:
            occurrences[label].append(e)
    validation=[]
    for label in range(100):
        ids=np.random.default_rng(np.random.SeedSequence([plan["seed"],label,1])).permutation(np.flatnonzero(y==label))
        reserve=plan["validation_per_class"]
        if not 0<reserve<len(ids):
            raise ValueError("invalid validation reservation")
        validation.extend(ids[:reserve].tolist())
        es=sorted(occurrences[label],key=lambda e:(e["client"],e["id"]))
        if not es or len(ids)-reserve<len(es):
            raise ValueError("insufficient private examples for all occurrences")
        for e,chunk in zip(es,np.array_split(ids[reserve:],len(es))):
            allocated[e["id"]].extend(chunk.tolist())
    manifest=dict(schema=1,plan=plan,plan_id=digest(plan),train_inventory=len(y),
                  test_inventory=len(ty),source_train_counts=np.bincount(y,minlength=100).tolist(),
                  source_test_counts=np.bincount(ty,minlength=100).tolist(),shards={},
                  normalization=dict(mean=list(mean),std=list(std)))
    def write(rel,source,labels,ids):
        ids=np.asarray(ids,dtype=np.int64)
        path=root/rel;path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,x=pixels(source,ids),y=labels[ids],ids=ids)
        manifest["shards"][rel]=dict(sha256=sha_file(path),count=len(ids),
            class_counts=np.bincount(labels[ids],minlength=100).tolist())
    for e in plan["episodes"]:
        ids=np.random.default_rng(np.random.SeedSequence([plan["seed"],e["client"],e["id"],3])).permutation(allocated[e["id"]])
        write(episode_path(e),raw.data,y,ids)
    write("eval/validation.npz",raw.data,y,sorted(validation))
    write("eval/test.npz",test.data,ty,np.arange(len(ty)))
    manifest["manifest_id"]=digest(manifest)
    atomic_json(root/"manifest.json",manifest)
    return manifest

def verify_manifest(root,plan=None):
    root=Path(root)
    m=json.loads((root/"manifest.json").read_text())
    if digest({k:v for k,v in m.items() if k!="manifest_id"})!=m["manifest_id"]:
        raise ValueError("manifest fingerprint mismatch")
    if digest(m["plan"])!=m["plan_id"] or (plan is not None and plan!=m["plan"]):
        raise ValueError("data plan mismatch")
    expected={episode_path(e) for e in m["plan"]["episodes"]}|{"eval/validation.npz","eval/test.npz"}
    if set(m["shards"])!=expected:
        raise ValueError("unexpected shard inventory")
    return m

def read_verified(root,m,rel):
    path=Path(root)/rel
    metadata=m["shards"][rel]
    if sha_file(path)!=metadata["sha256"]:
        raise ValueError("corrupt shard: "+rel)
    with np.load(path,allow_pickle=False) as z:
        x,y,ids=z["x"],z["y"],z["ids"]
    if x.ndim!=4 or x.shape[1]!=3 or y.ndim!=1 or ids.ndim!=1 or len(x)!=len(y) or len(y)!=len(ids) or len(y)!=metadata["count"]:
        raise ValueError("invalid shard dimensions")
    if not np.issubdtype(y.dtype,np.integer) or not np.issubdtype(ids.dtype,np.integer) or np.any(y<0) or np.any(y>=100) or not np.isfinite(x).all():
        raise ValueError("invalid shard values")
    if np.bincount(y,minlength=100).tolist()!=metadata["class_counts"]:
        raise ValueError("shard label counts mismatch")
    return x,y,ids

def validate(root,plan=None):
    m=verify_manifest(root,plan);p=m["plan"]
    seen=set();owners={};counts=np.zeros(100,dtype=np.int64)
    for e in p["episodes"]:
        _,y,ids=read_verified(root,m,episode_path(e))
        if set(y.tolist())!=set(p["tasks"][e["task"]]):
            raise ValueError("episode class membership mismatch")
        values=list(map(int,ids))
        if len(set(values))!=len(values) or seen.intersection(values):
            raise ValueError("duplicate source training example")
        seen.update(values)
        for v in values:owners[v]=e["client"]
        counts+=np.bincount(y,minlength=100)
    _,y,ids=read_verified(root,m,"eval/validation.npz")
    val=set(map(int,ids))
    if len(val)!=len(ids) or val&seen or seen|val!=set(range(m["train_inventory"])):
        raise ValueError("training/validation ownership or inventory violation")
    vc=np.bincount(y,minlength=100)
    if not np.all(vc==p["validation_per_class"]) or (counts+vc).tolist()!=m["source_train_counts"]:
        raise ValueError("train/validation class inventory mismatch")
    _,y,ids=read_verified(root,m,"eval/test.npz")
    if len(set(map(int,ids)))!=len(ids) or set(map(int,ids))!=set(range(m["test_inventory"])) or np.bincount(y,minlength=100).tolist()!=m["source_test_counts"]:
        raise ValueError("test inventory mismatch")
    return dict(valid=True,manifest_id=m["manifest_id"],plan_id=m["plan_id"],
                private_examples=len(owners),validation_examples=len(val),test_examples=len(ids),
                ownership_sha256=digest(sorted(owners.items())),cross_client_overlap=0)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["generate","validate"])
    p.add_argument("--plan",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--raw-root",default=str(ROOT/"data/raw"))
    a=p.parse_args();plan=json.loads(a.plan.read_text())
    if a.action=="generate":
        raw,test,_,_,norm=load_sources(SimpleNamespace(dataset="cifar100",raw_root=a.raw_root,download=True))
        print(generate_from_sources(plan,a.output,raw,test,norm)["manifest_id"])
    else:
        receipt=validate(a.output,plan)
        atomic_json(a.output/"validation_receipt.json",receipt)
        print(json.dumps(receipt))
if __name__=="__main__":main()
