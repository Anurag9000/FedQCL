"""Additional bounded local probe estimators. No held-out labels or future windows.
Bayesian output is a diagonal empirical-Fisher/Laplace approximation of the head,
not exact posterior inference. Paired probes use generated common synthetic inputs.
"""
import copy,math,time,random
import numpy as np
import torch
from torch.utils.data import default_collate
from .signals import seed

KINDS=("information","mutual_information","temporal","bayesian_head","representation_anchor","replay_diagnostics",
       "fisher_full","parameter_jacobian","head_hessian","conditional_moments")

def features(model,x,batch):
    values=[]
    with torch.no_grad():
        for xx in x.split(batch):
            z=torch.relu(model.bn1(model.conv1(xx)));z=model.layer4(model.layer3(model.layer2(model.layer1(z))))
            values.append(model.avgpool(z).flatten(1))
    return torch.cat(values)

def extra(ex,c,data,hist,t,r,version,kind):
    start=time.monotonic();cost={"forward_examples":0,"backward_examples":0,"seconds":0.}
    probe=ex._probe(c,data,t,r)
    def finish(values,n,space,metric="bounded_l2"):
        cost["seconds"]=time.monotonic()-start
        return ex._vector(c,t,r,version,kind,space,values,n,metric),cost
    if probe is None:return None,cost
    x,y=probe;n=len(x)
    if kind=="temporal":
        from .information import bin_unit,mutual_information,transfer_entropy,conditional_mi
        from .statistics import entropy,js
        if hist is None:return None,cost
        history=c.signal_history.setdefault("temporal",[])
        label_entropy=entropy(hist.counts)/max(1e-12,math.log(len(hist.counts))) if len(hist.counts)>1 else 0.
        replay_fraction=sum(len(v) for v in c.task_memories.values())/max(1,c.memory_size)
        history.append((t,r,min(1.,max(0.,label_entropy)),min(1.,replay_fraction),tuple(hist.counts)))
        history[:]=history[-max(5,ex.args.sync_history):]
        if len(history)<5:return None,cost
        a=bin_unit([v[2] for v in history]);b=bin_unit([v[3] for v in history])
        prev=history[-2]
        drift=js(prev[4],hist.counts) if prev[0]==t else 1.
        return finish([mutual_information(a,b),transfer_entropy(a,b),transfer_entropy(b,a),
                       mutual_information(a[:-1],a[1:]),drift],len(history),"temporal_plugin_v1")
    if kind=="replay_diagnostics":
        model=copy.deepcopy(c.model).eval()
        v=[]
        for k in range(c.num_tasks):
            memory=c.task_memories.get(k,[]) if k<t else []
            if not memory:v.extend([0.]*6);continue
            chosen=random.Random(seed(ex.args.seed,"replay_diagnostic",c.id,r,k)).sample(memory,min(len(memory),ex.args.sync_observation_size))
            xx,yy=default_collate(chosen);xx=xx.to(c.device);yy=yy.to(c.device)%c.classes_per_task
            with torch.no_grad():
                losses=torch.nn.functional.cross_entropy(model(xx,k),yy,reduction="none")
            counts=torch.bincount(yy,minlength=c.classes_per_task).double();p=counts/counts.sum()
            ent=-(p*p.clamp_min(1e-12).log()).sum().item()
            current_loss=losses.mean().item()
            key=f"replay_loss:{k}";prev=c.signal_history.get(key,current_loss);c.signal_history[key]=current_loss
            age=max(0,r-c.memory_rounds.get(k,r))
            v.extend([1.,math.log1p(len(memory)),math.log1p(age),ent,current_loss,current_loss-prev])
            cost["forward_examples"]+=len(xx)
        return finish(v,n,"replay_age_loss_v1")
    model=copy.deepcopy(c.model).eval()
    native="native_coordinates" if ex.args.sync_native_space else f"server_version:{version}"
    if kind=="representation_anchor":
        # Every client generates the same nonprivate probe from a public seed.
        g=torch.Generator().manual_seed(seed(ex.args.seed,"public_synthetic_probe",tuple(x.shape[1:])))
        count=max(8,min(32,ex.args.sync_observation_size))
        public=torch.randn((count,*x.shape[1:]),generator=g).to(c.device)
        a=features(model,public,c.batch_size).cpu().numpy();b=features(ex.anchor,public,c.batch_size).cpu().numpy()
        from .representation import measures
        v=measures(a,b,rank=8)
        cost["forward_examples"]=2*count
        return finish(v,count,f"public_synthetic_anchor:{ex.args.seed}") if v is not None else (None,cost)
    if kind=="conditional_moments":
        z=features(ex.anchor,x,c.batch_size).cpu();yy=y.cpu();v=[]
        for k in range(c.classes_per_task):
            q=z[yy==k]
            if not len(q):v.extend([0.,*([0.]*(3*ex.args.sync_sketch_dim))]);continue
            m=q.mean(0);delta=q-m
            moments=[m,delta.square().mean(0),delta.pow(3).mean(0)]
            v.extend([1.,*sum((ex.sketch([(f"conditional_{k}_{j}",a)]) for j,a in enumerate(moments)),[])])
        cost["forward_examples"]=n
        return finish(v,n,f"conditional_anchor:{ex.args.seed}")
    if kind in ("information","mutual_information"):
        from .information import joint_entropy,mutual_information,conditional_mi,total_correlation,interaction_information
        with torch.no_grad():
            prob=torch.cat([model(xx,t) for xx in x.split(c.batch_size)]).softmax(-1)
        pred=prob.argmax(-1).cpu().tolist();label=y.cpu().tolist()
        confidence=(prob.max(-1).values*4).long().clamp(0,3).cpu().tolist()
        cost["forward_examples"]=n
        if kind=="mutual_information":return finish([mutual_information(label,pred)],n,native)
        return finish([joint_entropy(label,pred),mutual_information(label,pred),
                       conditional_mi(label,pred,confidence),total_correlation(label,pred,confidence),
                       interaction_information(label,pred,confidence)],n,native)
    if kind in ("bayesian_head","head_hessian","parameter_jacobian"):
        z=features(model,x,c.batch_size);cost["forward_examples"]=n
        head=model.heads[t];logits=head(z).detach();prob=logits.softmax(-1)
        # Head logit Jacobian is exact for its linear parameters.
        design=torch.cat((z,torch.ones(len(z),1,device=z.device)),1).detach()
        if kind=="parameter_jacobian":
            # Analytic output-head Jacobian summary, not whole-network Jacobian.
            norm=design.norm(dim=1);v=[norm.mean().item(),norm.var(unbiased=False).item(),norm.max().item()]
            return finish(v,n,native)
        if kind=="head_hessian":
            # Trace and power iteration on the positive semidefinite softmax CE Hessian.
            trace=((prob*(1-prob)).sum(1)*design.square().sum(1)).mean()
            g=torch.Generator().manual_seed(seed(ex.args.seed,"head_hessian",c.id,t,r))
            v=torch.randn((prob.shape[1],design.shape[1]),generator=g).to(design)
            v=v/v.norm().clamp_min(1e-12)
            def hv(v):
                u=design@v.T
                z1=prob*(u-(prob*u).sum(1,keepdim=True))
                return z1.T@design/n
            for _ in range(8):
                w=hv(v);v=w/w.norm().clamp_min(1e-12)
            top=(v*hv(v)).sum().clamp_min(0)
            return finish([math.log1p(trace.item()),math.log1p(top.item()),float(top/trace.clamp_min(1e-12))],n,native)
        # Per-example exact head gradients: empirical Fisher precision plus unit prior.
        residual=prob-torch.nn.functional.one_hot(y,num_classes=prob.shape[1])
        fisher=(residual.square().T@design.square())
        variance=1/(1+fisher)
        mean=torch.cat((head.weight.detach(),head.bias.detach()[:,None]),1)
        g=torch.Generator().manual_seed(seed(ex.args.seed,"laplace_mc",c.id,t,r))
        probabilities=[];ll=[]
        for _ in range(8):
            eps=torch.randn(mean.shape,generator=g).to(mean)
            weight=mean+variance.sqrt()*eps
            logit=design@weight.T
            probabilities.append(logit.softmax(-1))
            ll.append(-torch.nn.functional.cross_entropy(logit,y,reduction="sum"))
        pp=torch.stack(probabilities);avg=pp.mean(0)
        entropy=lambda p:-(p*p.clamp_min(1e-12).log()).sum(-1)
        total=entropy(avg).mean();aleatoric=entropy(pp).mean()
        kl=.5*(variance+mean.square()-1-variance.log()).sum()
        elbo=(torch.stack(ll).mean()-kl)/n
        # Expected information gain proxy (BALD) in nats, not exact posterior contraction.
        return finish([total.item(),aleatoric.item(),max(0.,(total-aleatoric).item()),
                       variance.mean().item(),(-.5*variance.log()).mean().item(),
                       kl.item()/mean.numel(),elbo.item()],n,native)
    if kind=="fisher_full":
        # A bounded empirical diagonal Fisher sketch on at most eight probe examples.
        accum={};count=min(n,8)
        for xx,yy in zip(x[:count],y[:count]):
            model.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(model(xx[None],t),yy[None]).backward()
            for name,p in model.named_parameters():
                if p.grad is not None:
                    value=p.grad.detach().cpu().square()
                    if name in accum:accum[name]+=value
                    else:accum[name]=value
        cost["forward_examples"]=count;cost["backward_examples"]=count
        return finish(ex.sketch([(name,v/count) for name,v in accum.items()]),count,native)
    raise ValueError("unknown extra signal")
