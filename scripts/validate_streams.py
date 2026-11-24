#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np

def validate(root):
    manifest=json.loads((root/"manifest.json").read_text())
    train_seen=set();test_seen=set()
    for t in range(manifest["tasks"]):
        for split in ("train","validation","test"):
            for i in range(manifest["clients"]):
                with np.load(root/f"task_{t}"/split/f"{i}.npz",allow_pickle=True) as z:
                    data=z["data"].item()
                ids=list(map(int,data["ids"]))
                seen=test_seen if split=="test" else train_seen
                assert len(ids)==len(set(ids)) and not seen.intersection(ids)
                seen.update(ids)
                assert len(ids)==len(data["x"])==len(data["y"])==len(data["window"])
                assert np.all((data["y"]>=20*t)&(data["y"]<20*(t+1)))
                if split=="train":
                    counts=np.bincount(data["y"]-20*t,minlength=20)
                    assert counts.tolist()==manifest["tasks_data"][t]["counts"][i]
                    assert np.all((data["window"]>=0)&(data["window"]<manifest["windows"]))
    assert len(train_seen)==50000 and len(test_seen)==10000
    return {"manifest_id":manifest["manifest_id"],"train_and_validation":len(train_seen),"test":len(test_seen),"valid":True}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);a=p.parse_args()
    print(json.dumps(validate(a.root),indent=2))
