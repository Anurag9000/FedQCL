"""Bounded report extractors; gradient probes never mutate the training model."""
import copy
import hashlib
import random
import time
import numpy as np
import torch
from torch.utils.data import default_collate
from .policies import Report

def seed(seed,*keys):
    return int.from_bytes(hashlib.sha256(repr((seed,keys)).encode()).digest()[:4],"little")

class SignalExtractor:
    def __init__(self,args,anchor=None):
        self.args=args
        self.anchor=anchor
        if anchor is not None:
            self.anchor.to(args.device).eval()
        self.projection_cache={}
    def _probe(self,c,data,t,r):
        n=min(len(data),self.args.sync_observation_size)
        if n<self.args.sync_min_observation:return None
        idx=random.Random(seed(self.args.seed,"probe",c.id,t,r)).sample(range(len(data)),n)
        x,y=default_collate([data[i] for i in idx])
        return x.to(c.device),y.to(c.device)%c.classes_per_task
    def _vector(self,c,t,r,version,kind,space,values,n,metric="cosine"):
        return Report(c.id,t,r,tuple(range(len(values))),tuple(float(x) for x in values),
                      model_version=version,kind=kind,space=space,sample_count=n,metric=metric)
    def sketch(self,tensors):
        # Seeded per-tensor CountSketch, cached CPU projection coordinates.
        d=self.args.sync_sketch_dim
        result=torch.zeros(d,dtype=torch.float64)
        for name,tensor in tensors:
            key=(name,tensor.numel(),d)
            flat=tensor.detach().reshape(-1).cpu().double()
            # Chunked hashing avoids storing a dense projection matrix.
            gen=torch.Generator().manual_seed(seed(self.args.seed,"projection",name,d))
            for start in range(0,len(flat),65536):
                v=flat[start:start+65536]
                index=torch.randint(d,(len(v),),generator=gen)
                sign=torch.randint(2,(len(v),),generator=gen,dtype=torch.int64)*2-1
                result.scatter_add_(0,index,v*sign)
        return result.tolist()
    def before(self,c,data,hist,t,r,version):
        kind=self.args.sync_signal
        start=time.monotonic()
        cost={"forward_examples":0,"backward_examples":0,"seconds":0.}
        if kind=="histogram":return hist,cost
        if kind=="update":return None,cost
        if kind=="replay":
            # Current histogram and raw per-task memory counts are separate
            # studies; replay profile explicitly encodes missing memory as zero.
            vals=tuple(len(c.task_memories.get(k,[])) for k in range(c.num_tasks))
            report=Report(c.id,t,r,tuple(range(len(vals))),vals,kind="replay",
                          space=f"replay_tasks:{c.num_tasks}",model_version=version)
            cost["seconds"]=time.monotonic()-start
            return (report if report.valid() else None),cost
        probe=self._probe(c,data,t,r)
        if probe is None:return None,cost
        x,y=probe
        if kind=="feature":
            values=[]
            with torch.no_grad():
                model=self.anchor
                for chunk in x.split(c.batch_size):
                    z=torch.relu(model.bn1(model.conv1(chunk)))
                    z=model.layer4(model.layer3(model.layer2(model.layer1(z))))
                    values.append(model.avgpool(z).flatten(1).cpu())
            z=torch.cat(values)
            vector=torch.cat((z.mean(0),z.var(0,unbiased=False)))
            # Fixed projection on mean/variance coordinates.
            v=self.sketch([("feature_moments",vector)])
            report=self._vector(c,t,r,version,kind,f"fixed_init:{self.args.seed}",v,len(x),"bounded_l2")
        else:
            model=copy.deepcopy(c.model).eval()
            model.zero_grad(set_to_none=True)
            for xx,yy in zip(x.split(c.batch_size),y.split(c.batch_size)):
                loss=torch.nn.functional.cross_entropy(model(xx,t),yy,reduction="sum")/len(x)
                loss.backward()
            v=self.sketch((name,p.grad) for name,p in model.named_parameters() if p.grad is not None)
            space="native_parameter_coordinates" if self.args.sync_native_space else f"server_version:{version}"
            report=self._vector(c,t,r,version,kind,space,v,len(x))
            cost["backward_examples"]=len(x)
        cost["forward_examples"]=len(x)
        cost["seconds"]=time.monotonic()-start
        return report,cost
    def snapshot(self,c):
        return {name:p.detach().cpu().clone() for name,p in c.model.named_parameters()}
    def after(self,c,start,t,r,version):
        began=time.monotonic()
        v=self.sketch((name,p.detach().cpu()-start[name]) for name,p in c.model.named_parameters())
        space="native_parameter_coordinates" if self.args.sync_native_space else f"server_version:{version}"
        return self._vector(c,t,r,version,"update",space,v,c.current_samples),{"seconds":time.monotonic()-began,"forward_examples":0,"backward_examples":0}
