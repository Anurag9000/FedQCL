#!/usr/bin/env python3
"""Validate manifest, shard fingerprints, exact inventories, class and window counts."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from generate_streams import sha_file

def require(condition,message):
    if not condition:raise ValueError(message)

def validate(root):
    manifest=json.loads((root/"manifest.json").read_text())
    payload={k:v for k,v in manifest.items() if k!="manifest_id"}
    require(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==manifest["manifest_id"],"manifest hash mismatch")
    c=manifest.get("classes_per_task",20);train_n=manifest.get("train_inventory",50000);test_n=manifest.get("test_inventory",10000)
    train_seen=set();test_seen=set();expected=set()
    for t in range(manifest["tasks"]):
        public_n=manifest.get("public_probe_per_class",0)
        if public_n:
            rel=f"task_{t}/public_probe/0.npz";expected.add(rel);path=root/rel
            require(sha_file(path)==manifest["shards"][rel],f"public probe hash mismatch: {rel}")
            with np.load(path,allow_pickle=True) as z:data=z["data"].item()
            ids=list(map(int,data["ids"]))
            require(len(ids)==len(set(ids))==len(data["x"])==len(data["y"])==c*public_n and not train_seen.intersection(ids),f"public probe overlap or length: {rel}")
            require(np.array_equal(data["y"],np.repeat(np.arange(t*c,(t+1)*c),public_n)),f"public probe order: {rel}")
            require(np.isfinite(data["x"]).all(),f"nonfinite public probe: {rel}")
            train_seen.update(ids)
        for split in ("train","validation","test"):
            for i in range(manifest["clients"]):
                rel=f"task_{t}/{split}/{i}.npz";expected.add(rel);path=root/rel
                if manifest.get("schema",1)>=2:require(sha_file(path)==manifest["shards"][rel],f"shard hash mismatch: {rel}")
                with np.load(path,allow_pickle=True) as z:data=z["data"].item()
                ids=list(map(int,data["ids"]));seen=test_seen if split=="test" else train_seen
                require(len(ids)==len(set(ids)) and not seen.intersection(ids),f"duplicate IDs: {rel}")
                seen.update(ids)
                require(len(ids)==len(data["x"])==len(data["y"])==len(data["window"]),f"length mismatch: {rel}")
                require(np.all((data["y"]>=c*t)&(data["y"]<c*(t+1))),f"label range: {rel}")
                require(np.isfinite(data["x"]).all(),f"nonfinite images: {rel}")
                if split=="train":
                    counts=np.bincount(data["y"]-c*t,minlength=c)
                    require(counts.tolist()==manifest["tasks_data"][t]["counts"][i],f"class counts: {rel}")
                    require(np.all((data["window"]>=0)&(data["window"]<manifest["windows"])),f"windows: {rel}")
                    if "window_counts" in manifest["tasks_data"][t]:
                        actual=[np.bincount(data["y"][data["window"]==w]-c*t,minlength=c).tolist() for w in range(manifest["windows"])]
                        require(actual==manifest["tasks_data"][t]["window_counts"][i],f"window counts: {rel}")
    require(train_seen==set(range(train_n)) and test_seen==set(range(test_n)),"incomplete inventory")
    if manifest.get("schema",1)>=2:require(set(manifest["shards"])==expected,"unexpected manifest shard list")
    return {"manifest_id":manifest["manifest_id"],"train_and_validation":len(train_seen)-manifest.get("public_probe_per_class",0)*c*manifest["tasks"],"test":len(test_seen),"valid":True,**({"public_probe":manifest["public_probe_per_class"]*c*manifest["tasks"]} if manifest.get("public_probe_per_class") else {})}
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("--receipt",type=Path);a=p.parse_args()
    result=validate(a.root)
    if a.receipt:
        tmp=a.receipt.with_suffix(".tmp");tmp.write_text(json.dumps(result,indent=2));tmp.replace(a.receipt)
    print(json.dumps(result,indent=2))
