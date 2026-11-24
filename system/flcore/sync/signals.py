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
        if kind in ("replay","replay_histogram","replay_queue"):
            # Current histogram and raw per-task memory counts are separate
            # studies; replay profile explicitly encodes missing memory as zero.
            vals=tuple(len(c.task_memories.get(k,[])) for k in range(c.num_tasks))
            if kind=="replay_histogram":
                if hist is None:return None,cost
                h=np.array(hist.counts,dtype=float);h/=h.sum()
                m=np.array(vals,dtype=float);m=m/m.sum() if m.sum() else m
                vector=np.r_[h,m,float(m.sum()==0)]
                return self._vector(c,t,r,version,kind,"task_labels_replay_tasks",vector,sum(hist.counts),"bounded_l2"),cost
            if kind=="replay_queue":
                vector=[*vals,*[float(v) for v in c.Q],*[float(k in c.task_memories) for k in range(c.num_tasks)]]
                return self._vector(c,t,r,version,kind,"replay_queue_tasks",np.log1p(vector),max(1,len(data)),"bounded_l2"),cost
            report=Report(c.id,t,r,tuple(range(len(vals))),vals,kind="replay",
                          space=f"replay_tasks:{c.num_tasks}",model_version=version)
            cost["seconds"]=time.monotonic()-start
            return (report if report.valid() else None),cost
        if kind in ("parameter_norms","normalization"):
            if kind=="parameter_norms":
                v=[]
                for p in c.model.parameters():
                    z=p.detach().float()
                    v.extend([z.abs().mean().item(),z.square().mean().sqrt().item(),z.abs().max().item()])
            else:
                v=[]
                for name,b in c.model.named_buffers():
                    if "running_mean" in name or "running_var" in name:
                        z=b.detach().float()
                        v.extend([z.mean().item(),z.var(unbiased=False).item()])
            cost["seconds"]=time.monotonic()-start
            return self._vector(c,t,r,version,kind,"native_parameter_coordinates",v,max(1,len(data)),"bounded_l2"),cost
        probe=self._probe(c,data,t,r)
        if probe is None:return None,cost
        x,y=probe
        if kind in ("feature","prototypes","feature_covariance","spectral"):
            values=[]
            with torch.no_grad():
                model=self.anchor
                for chunk in x.split(c.batch_size):
                    z=torch.relu(model.bn1(model.conv1(chunk)))
                    z=model.layer4(model.layer3(model.layer2(model.layer1(z))))
                    values.append(model.avgpool(z).flatten(1).cpu())
            z=torch.cat(values)
            if kind=="prototypes":
                yc=y.cpu(); parts=[]
                for label in range(c.classes_per_task):
                    group=z[yc==label]
                    parts.extend([torch.tensor([float(len(group)>0)]),group.mean(0) if len(group) else torch.zeros(z.shape[1])])
                vector=torch.cat(parts)
            elif kind=="feature_covariance":
                centered=z-z.mean(0)
                vector=(centered.T@centered/max(1,len(z))).flatten()
            elif kind=="spectral":
                from .statistics import spectral_summary
                vector=torch.tensor(spectral_summary((z-z.mean(0)).numpy()))
            else:
                vector=torch.cat((z.mean(0),z.var(0,unbiased=False)))
            v=self.sketch([(kind,vector)]) if kind!="spectral" else vector.tolist()
            report=self._vector(c,t,r,version,kind,f"fixed_init:{self.args.seed}",v,len(x),"bounded_l2")
        elif kind in ("predictions","loss","uncertainty","entropy"):
            model=c.model
            previous=model.training;model.eval()
            try:
                with torch.no_grad():
                    logits=torch.cat([model(xx,t) for xx in x.split(c.batch_size)])
                    probs=logits.softmax(-1)
                    losses=torch.nn.functional.cross_entropy(logits,y,reduction="none")
                    ent=-(probs*probs.clamp_min(1e-12).log()).sum(-1)
                    top=probs.topk(min(2,probs.shape[1]),dim=-1).values
                    margin=top[:,0]-top[:,-1]
                    if kind=="predictions":
                        vector=torch.cat((probs.mean(0),probs.var(0,unbiased=False),logits.mean(0)))
                    elif kind=="loss":
                        vector=torch.cat((losses.mean().view(1),losses.var(unbiased=False).view(1),torch.quantile(losses,torch.tensor([.1,.5,.9],device=losses.device))))
                    elif kind=="entropy":
                        vector=torch.stack((ent.mean(),ent.var(unbiased=False),-(probs.mean(0)*probs.mean(0).clamp_min(1e-12).log()).sum()))
                    else:
                        vector=torch.stack((ent.mean(),ent.var(unbiased=False),top[:,0].mean(),margin.mean(),margin.var(unbiased=False)))
                space="native_output_coordinates" if self.args.sync_native_space else f"server_version:{version}"
                report=self._vector(c,t,r,version,kind,space,vector.cpu().tolist(),len(x),"bounded_l2")
            finally:model.train(previous)
        elif kind in ("jacobian","fisher"):
            # Jacobian: input gradient of a fixed projected logit per example.
            # Fisher: empirical diagonal of per-example gradients of the current head.
            model=copy.deepcopy(c.model).eval()
            if kind=="jacobian":
                grads=[]
                for xx in x.split(c.batch_size):
                    xx=xx.detach().requires_grad_(True)
                    outputs=model(xx,t)
                    direction=torch.arange(1,outputs.shape[1]+1,device=outputs.device,dtype=outputs.dtype)
                    direction=direction/direction.norm()
                    grad=torch.autograd.grad((outputs*direction).sum(),xx)[0]
                    grads.append(grad.detach().flatten(1).cpu())
                z=torch.cat(grads)
                vector=self.sketch([("jacobian",torch.cat((z.mean(0),z.square().mean(0))))])
            else:
                head=list(model.heads[t].parameters())
                sums=[torch.zeros_like(p) for p in head]
                for xx,yy in zip(x,y):
                    loss=torch.nn.functional.cross_entropy(model(xx.unsqueeze(0),t),yy.unsqueeze(0))
                    grads=torch.autograd.grad(loss,head)
                    for acc,g in zip(sums,grads):acc.add_(g.detach().square())
                vector=self.sketch([(f"head_{i}",g/len(x)) for i,g in enumerate(sums)])
            space="native_parameter_coordinates" if self.args.sync_native_space else f"server_version:{version}"
            report=self._vector(c,t,r,version,kind,space,vector,len(x),"bounded_l2")
            cost["backward_examples"]=len(x)
        else:
            model=copy.deepcopy(c.model).eval()
            model.zero_grad(set_to_none=True)
            if kind=="gradient_variance":
                sketches=[]
                for xx,yy in zip(x.split(c.batch_size),y.split(c.batch_size)):
                    model.zero_grad(set_to_none=True)
                    torch.nn.functional.cross_entropy(model(xx,t),yy).backward()
                    sketches.append(self.sketch((name,p.grad) for name,p in model.named_parameters() if p.grad is not None))
                z=np.asarray(sketches);v=np.r_[z.mean(0),z.var(0)].tolist()
            else:
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
