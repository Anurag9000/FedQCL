"""Causal utility selection. All probes use cached local models, never server weights.
Analytical estimates are hypotheses, not calibrated confidence intervals.
Only decoded telemetry participates in decisions; offline labels remain isolated.
"""
import math,random,time,json,base64
from itertools import combinations
from types import SimpleNamespace
import numpy as np
from .policies import baseline_select,due_clients

def encode_vector(v,precision):
    # The nested wire descriptor is real binary data (not JSON float estimates).
    x=np.asarray(v,dtype=float)
    if precision=="int8":
        scale=float(max(np.max(np.abs(x)),1e-12)/127)
        return {"counts":(np.rint(x/scale).clip(-127,127)*scale).tolist(),"labels":[],"schema":1,"metric":"utility","quant_scale":scale,"utility_precision":"int8"}
    dtype={"fp64":np.float64,"fp32":np.float32,"fp16":np.float16}[precision]
    return {"counts":x.astype(dtype).astype(float).tolist(),"labels":[],"schema":1,"metric":"utility","utility_precision":precision}

def precision_for(a,prior):
    # Only previous decisions influence the next requested representation.
    reason=prior[-1].get("utility",{}).get("reason") if prior else None
    return ("int8",min(256,a.sync_sketch_dim)) if reason=="utility" else (a.sync_utility_precision,a.sync_sketch_dim)

def proxy_gradient(x,y,task,classes,total,ex):
    """Common zero-logit classifier head on deterministic pooled input features."""
    import torch
    z=torch.nn.functional.adaptive_avg_pool2d(x,(4,4)).flatten(1)
    z=torch.cat((z,torch.ones((len(z),1),device=z.device)),dim=1)
    z=z/z.norm(dim=1,keepdim=True).clamp_min(1.)
    error=torch.full((len(z),total),1/total,device=z.device)
    error[torch.arange(len(z),device=z.device),y+task*classes]-=1
    grad=error.T@z/len(z)
    return np.asarray(ex.sketch([("common_fixed_head",grad)]),float)

def probe(ex,c,data,t,r,version,mode):
    start=time.monotonic()
    cost={"forward_examples":0,"backward_examples":0,"seconds":0.,"quantization_l2":0.,"replay_loss_by_task":{},"replay_loss_change_by_task":{}}
    if mode in ("random","balanced"):return None,cost
    import copy,torch
    from torch.utils.data import default_collate
    from .signals import seed
    from .tier_a_signals import replay_diagnostic
    m=copy.deepcopy(c.model).eval();groups={}
    current=ex._probe(c,data,t,r)
    if current is not None:groups[t]=current
    past=[k for k,v in sorted(c.task_memories.items()) if k<t and v]
    for k in past:
        items=c.task_memories[k]
        rng=random.Random(seed(ex.args.seed,"utility-past",c.id,t,r,k))
        chosen=rng.sample(items,min(len(items),max(1,ex.args.sync_observation_size//len(past))))
        x,y=default_collate(chosen);groups[k]=(x.to(c.device),y.to(c.device)%c.classes_per_task)
    cost["observation_round"]=r
    replay_diagnostic(ex,c,m,{k:None for k in past},cost,t)
    gradients={};losses={};support={}
    for k,(x,y) in groups.items():
        m.zero_grad(set_to_none=True);loss=0.
        for bx,by in zip(x.split(c.batch_size),y.split(c.batch_size)):
            value=torch.nn.functional.cross_entropy(m(bx,k),by,reduction="sum")/len(x)
            loss+=float(value.detach());cost["forward_examples"]+=len(bx)
            if mode!="loss" and not ex.args.sync_utility_proxy:value.backward();cost["backward_examples"]+=len(bx)
        losses[str(k)]=loss;support[str(k)]=len(x)
        if mode!="loss":
            if ex.args.sync_utility_proxy:
                v=proxy_gradient(x,y,k,c.classes_per_task,c.num_tasks*c.classes_per_task,ex)
                cost["proxy_examples"]=cost.get("proxy_examples",0)+len(x)
            else:v=np.asarray(ex.sketch((n,p.grad) for n,p in m.named_parameters() if p.grad is not None),float)
            # Clip sketch norm before encoding, without altering training.
            v*=min(1.,10/max(1e-12,np.linalg.norm(v)))
            gradients[str(k)]=encode_vector(v,ex.args.sync_utility_precision)
    counts=np.bincount([int(y)%c.classes_per_task for _,y in data],minlength=c.classes_per_task).astype(float)
    counts/=max(1.,counts.sum())
    cost["seconds"]=time.monotonic()-start
    return {"client":c.id,"round":r,"task":t,"version":version,"mass":len(data),"support":support,"loss":losses,
            "deterioration":{str(k):v for k,v in cost["replay_loss_change_by_task"].items()},"gradient":gradients,"labels_hist":counts.tolist()},cost

def local_score(x,a,version):
    if x is None:return None
    t=str(x["task"]);loss=x["loss"].get(t)
    if loss is None:return None
    old=[v for k,v in x["loss"].items() if k!=t]
    urgency=loss/(1+loss)+a.sync_utility_past*(sum(max(0,v)/(1+abs(v)) for v in x["deterioration"].values())/max(1,len(old)))
    gs={k:np.array(v["counts"]) for k,v in x["gradient"].items()}
    if t in gs:
        past=[v for k,v in gs.items() if k!=t];h=np.mean(past,axis=0) if past else np.zeros_like(gs[t])
        d=gs[t]+h;den=max(1.,np.linalg.norm(d)*(np.linalg.norm(gs[t])+a.sync_utility_past*np.linalg.norm(h)))
        urgency=float((gs[t]+a.sync_utility_past*h)@d/den)+.2*urgency
    risk=1-1/(1+max(0,version-x["version"]))+1/math.sqrt(max(1,x["support"].get(t,1)))
    return {"score":float(urgency),"risk":float(risk)}

def objective(view,s,a,version,visits):
    if not s:return 0.,0.,True
    t=str(next(x["task"] for x in view.values() if x is not None))
    if any(view[i] is None or t not in view[i]["gradient"] for i in s):return None,1.,False
    valid={i:x for i,x in view.items() if x is not None and t in x["gradient"]}
    if not valid:return None,1.,False
    def g(x,k):return np.asarray(x["gradient"][k]["counts"],float)
    def update(x):
        past=[g(x,k) for k in x["gradient"] if k!=t]
        return g(x,t)+(np.mean(past,axis=0) if past else 0)
    def mean(ids,fn):return np.average([fn(valid[i]) for i in ids],axis=0,weights=[valid[i]["mass"] for i in ids])
    full=mean(list(valid),update);d=mean(s,update);current=mean(list(valid),lambda x:g(x,t))
    scale=max(1.,float(np.linalg.norm(full)**2+np.linalg.norm(current)**2))
    gain=float(current@d)/scale;past_gains=[]
    for k in sorted(set().union(*(x["gradient"] for x in valid.values()))-{t}):
        group=[i for i,x in valid.items() if k in x["gradient"]]
        h=mean(group,lambda x:g(x,k));past_gains.append(float(h@d)/scale)
    gain+=a.sync_utility_past*(sum(past_gains)/max(1,len(past_gains)))
    gain-=.05*float(d@d)/scale
    gain-=a.sync_utility_match*float((d-full)@(d-full))/scale
    gain-=a.sync_utility_mass*.5*float(np.abs(mean(s,lambda x:x["labels_hist"])-mean(list(valid),lambda x:x["labels_hist"])).sum())
    risk=float(np.mean([local_score(valid[i],a,version)["risk"] for i in s]))+(1-len(valid)/len(view))
    fair=sum(max(visits.values())-visits[i] for i in s)/max(1,sum(visits.values()))
    return gain+a.sync_utility_fair*fair-a.sync_utility_risk*risk,risk,all(v>=-a.sync_utility_retention for v in past_gains)

# Upper bounds are deliberately conservative; audit rejects any violation.
OVERHEAD=2_000_000
MODEL_ENVELOPE=4096
def ceiling(a,schema):
    return math.floor(a.sync_utility_budget*(a.num_clients*schema["tensor_bytes"]+2*a.num_clients*(a.global_rounds+1)*schema["tensor_bytes"])) if a.sync_utility_budget else None

def feasible(a,ids,last,r,used,schema,s):
    cap=ceiling(a,schema)
    if cap is None:return True
    per=2*schema["tensor_bytes"]+MODEL_ENVELOPE
    end=a.global_rounds
    # Reserve a constructive per-client deadline schedule with unlimited
    # simultaneous admissions, and conservative overhead each future round.
    mandatory=sum(max(0,(end-(r if i in s else last[i]))//(a.sync_max_skip+1)) for i in ids)
    return used+per*len(s)+OVERHEAD+per*mandatory+OVERHEAD*(end-r)<=cap

def choose(a,ids,last,due,r,version,visits,view,used,schema):
    mode=a.sync_utility
    sizes=[len(ids)] if r==0 else range(len(ids)+1) if a.sync_utility_budget or a.sync_mode=="threshold" else [a.sync_k]
    pool=[s for k in sizes for s in combinations(ids,k) if set(due)<=set(s) and feasible(a,ids,last,r,used,schema,s)]
    if not pool and not a.sync_utility_budget:
        # Explicit fixed-k deadline conflict: same age priority as old suite.
        k=len(ids) if r==0 else a.sync_k
        forced=set(due[:k]);pool=[s for s in combinations(ids,k) if forced<=set(s)]
    if not pool:raise ValueError("utility communication ceiling/deadlines infeasible")
    if r==0:return ids,{"reason":"bootstrap","candidates":[]}
    rng=random.Random(f"{a.seed}:utility:{r}")
    if mode in ("random","balanced"):
        if a.sync_utility_budget:
            remain=ceiling(a,schema)-used
            desired=max(len(due),int(remain/max(1,a.global_rounds-r+1)/(2*schema["tensor_bytes"]+MODEL_ENVELOPE)))
            distance=min(abs(len(s)-desired) for s in pool);pool=[s for s in pool if abs(len(s)-desired)==distance]
        if mode=="balanced":return min(pool,key=lambda s:(sum(visits[i] for i in s),s)),{"reason":"balanced","candidates":[]}
        k=len(pool[0])
        chosen=baseline_select("random",ids,due,last,k,a.seed,r)
        if chosen not in pool:chosen=rng.choice(pool)
        return chosen,{"reason":"random","candidates":[]}
    if not any(v is not None for v in view.values()):
        return rng.choice(pool),{"reason":"missing_evidence","candidates":[]}
    ranked=[]
    for s in pool:
        if mode in ("local","loss"):
            known=[view[i] for i in s if view[i] is not None]
            score=sum(v["score"]-a.sync_utility_risk*v["risk"] for v in known)/max(1,len(s))
            risk=sum(v["risk"] for v in known)/max(1,len(s))+1-len(known)/max(1,len(s)) if s else 0.
            ok=len(known)==len(s)
            score+=a.sync_utility_fair*sum(max(visits.values())-visits[i] for i in s)/max(1,sum(visits.values()))
        else:score,risk,ok=objective(view,s,a,version,visits) if any(v is not None for v in view.values()) else (None,1.,False)
        if score is not None:
            # Price per complete exchange; for fixed k this is constant.
            score-=a.sync_utility_price*len(s)
            ranked.append({"subset":list(s),"score":score,"risk":risk,"retention_ok":ok,"features":[1.,score,risk,float(ok),len(s)/len(ids),r/max(1,a.global_rounds)]})
    model=getattr(a,"utility_model",None)
    if model and model["qualified"]:
        for x in ranked:
            x["analytical_score"]=x["score"]
            x["score"]=float(np.dot(model["coef"],x["features"]))
            x["risk"]+=model["calibration_error"]
    safe=[x for x in ranked if x["retention_ok"]];candidates=safe or ranked
    if not candidates:return rng.choice(pool),{"reason":"missing_evidence","candidates":ranked}
    best=max(candidates,key=lambda x:(x["score"],tuple(-i for i in x["subset"])))
    matched=[x for x in candidates if len(x["subset"])==len(best["subset"])]
    advantage=best["score"]-sum(x["score"] for x in matched)/len(matched)
    if (model and not model["qualified"]) or advantage<=a.sync_utility_gate*(1+best["risk"])+(model["calibration_error"] if model and model["qualified"] else 0.):
        chosen=rng.choice([tuple(x["subset"]) for x in matched]);reason="uncertain_random"
    else:chosen=tuple(best["subset"]);reason="utility" if safe else "least_harm_relaxation"
    return chosen,{"reason":reason,"candidates":ranked,"advantage":advantage,"best":best["subset"]}

def decide(ctl,r,ids,last,due):
    a=ctl.args;mode=a.sync_utility;reports=ctl.utility_reports;tr=ctl.transport
    views={i:{} for i in ids};view={}
    if mode in ("local","loss"):
        for i in ids:view[i]=tr.send(r,"observe",i,"server","UTILITY_SCALAR",local_score(reports[i],a,ctl.model_version))
    elif mode=="server":
        for i in ids:view[i]=tr.send(r,"observe",i,"server","UTILITY_REPORT",reports[i])
    elif mode=="peer":
        for i in ids:
            views[i][i]=reports[i]
            for j in ids:
                if j!=i:views[j][i]=tr.send(r,"observe",i,j,"UTILITY_PEER_REPORT",reports[i])
        view=views[ids[0]]
        if any(v!=view for v in views.values()):raise ValueError("inconsistent utility peer view")
    used=sum(e["serialized_bytes"] for e in tr.events if e["round"]<r)
    visits={i:sum(i in x["complete"] for x in ctl.decisions) for i in ids}
    if mode=="peer":
        context={"used":used,"visits":{str(i):visits[i] for i in ids},"last":{str(i):last[i] for i in ids},"version":ctl.model_version,"tensor_bytes":ctl.utility_schema["tensor_bytes"],"cap":ceiling(a,ctl.utility_schema)}
        for i in ids:
            if tr.send(r,"observe","server",i,"UTILITY_CONTEXT",context)!=context:raise ValueError("peer context disagreement")
    observed=sum(e["serialized_bytes"] for e in tr.events if e["round"]==r)
    if observed>OVERHEAD:raise ValueError("utility telemetry overhead bound exceeded")
    selected,info=choose(a,ids,last,due,r,ctl.model_version,visits,view,used,ctl.utility_schema)
    if mode=="peer":
        for i in ids:
            election,_=choose(a,ids,last,due,r,ctl.model_version,visits,views[i],used,ctl.utility_schema)
            if election!=selected:raise ValueError("utility peer election disagreement")
            tr.send(r,"admission",i,"server","ELECTION_RESULT",{"selected":election})
    for i in selected:tr.send(r,"admission","server",i,"ADMISSION_RESPONSE",{"admitted":True})
    ctl.decisions.append({"round":r,"available":ids,"due":due,"selected":selected,"multiplicity":None,"threshold":float(a.sync_threshold),
        "deadline_cap_violations":[i for i in due if i not in selected],"scores":{},"references":{},"bank_versions":{i:0 for i in ids},"coverage":{},"impact":None,
        "reasons":{i:"bootstrap" if r==0 else "mandatory" if i in due and i in selected else info["reason"] if i in selected else "deferred" for i in ids},
        "utility":info,"utility_used_before_admission":used+observed,"utility_cap":ceiling(a,ctl.utility_schema)})
    return selected

def check_budget(ctl,r):
    events=[e for e in ctl.transport.events if e["round"]==r]
    if sum(e["serialized_bytes"]-e["tensor_bytes"] for e in events)>OVERHEAD:raise ValueError("utility overhead bound exceeded")
    cap=ceiling(ctl.args,ctl.utility_schema)
    if cap is not None and sum(e["serialized_bytes"] for e in ctl.transport.events)>cap:raise ValueError("utility budget exceeded")

def verify_decisions(config,rounds,decisions,events):
    from .wire import unpack
    a=SimpleNamespace(**config)
    if config.get("sync_utility_model"):
        from pathlib import Path
        import hashlib
        raw=Path(config["sync_utility_model"]).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=config["sync_utility_model_sha256"]:raise ValueError("utility model fingerprint")
        a.utility_model=json.loads(raw)
    ids=tuple(range(a.num_clients));last={i:-1 for i in ids};visits={i:0 for i in ids};version=0
    # Use schema measured in model transfers; identity is not needed by feasibility.
    schema={"tensor_bytes":next(e["tensor_bytes"] for e in events if e["kind"]=="MODEL_INITIAL")}
    consumed=sum(e["serialized_bytes"] for e in events if e["round"]==-1)
    if len(rounds)!=len(decisions):raise ValueError("utility decision inventory")
    for r,(row,dec) in enumerate(zip(rounds,decisions)):
        if row["round"]!=r or dec["round"]!=r or tuple(dec["available"])!=ids:raise ValueError("utility round inventory")
        view={};re= [e for e in events if e["round"]==r];before=0;reports={int(i):v for i,v in row["utility_reports"].items()}
        for e in re:
            if e["phase"]!="observe":continue
            before+=e["serialized_bytes"]
            if e["kind"] in ("UTILITY_SCALAR","UTILITY_REPORT","UTILITY_PEER_REPORT"):
                payload=unpack(base64.b64decode(e["encoded_envelope_b64"]))["payload"];i=e["sender"]
                expected=local_score(reports[i],a,version) if e["kind"]=="UTILITY_SCALAR" else reports[i]
                if payload!=expected:raise ValueError("utility report mismatch")
                if i in view and view[i]!=payload:raise ValueError("peer view mismatch")
                view[i]=payload
        kind={"local":"UTILITY_SCALAR","loss":"UTILITY_SCALAR","server":"UTILITY_REPORT","peer":"UTILITY_PEER_REPORT"}.get(a.sync_utility)
        expected=[(i,j) for i in ids for j in ids if i!=j] if a.sync_utility=="peer" else [(i,"server") for i in ids] if kind else []
        actual=[(e["sender"],e["receiver"]) for e in re if e["kind"]==kind]
        if sorted(actual)!=sorted(expected) or any(e["kind"] in ("UTILITY_SCALAR","UTILITY_REPORT","UTILITY_PEER_REPORT") and (e["kind"]!=kind or e["phase"]!="observe") for e in re):raise ValueError("utility telemetry transaction roster")
        requests=[e for e in re if e["kind"]=="UTILITY_PRECISION"]
        if getattr(a,"sync_utility_adaptive_precision",False):
            precision,dim=precision_for(a,decisions[:r]);expected_request={"precision":precision,"dimension":dim}
            if sorted(e["receiver"] for e in requests)!=list(ids) or any(e["sender"]!="server" or e["phase"]!="observe" or unpack(base64.b64decode(e["encoded_envelope_b64"]))["payload"]!=expected_request for e in requests):raise ValueError("adaptive precision request")
            if any(v["utility_precision"]!=precision or len(v["counts"])!=dim for x in reports.values() if x for v in x["gradient"].values()):raise ValueError("adaptive precision payload")
        elif requests:raise ValueError("unexpected precision control")
        if kind and set(view)!=set(ids):raise ValueError("utility telemetry roster")
        contexts=[e for e in re if e["kind"]=="UTILITY_CONTEXT"]
        if a.sync_utility=="peer":
            context={"used":consumed,"visits":{str(i):visits[i] for i in ids},"last":{str(i):last[i] for i in ids},"version":version,"tensor_bytes":schema["tensor_bytes"],"cap":ceiling(a,schema)}
            if sorted(e["receiver"] for e in contexts)!=list(ids) or any(e["sender"]!="server" or e["phase"]!="observe" or unpack(base64.b64decode(e["encoded_envelope_b64"]))["payload"]!=context for e in contexts):raise ValueError("utility peer context mismatch")
            elections=[e for e in re if e["kind"]=="ELECTION_RESULT"]
            if sorted(e["sender"] for e in elections)!=list(ids) or any(e["receiver"]!="server" or unpack(base64.b64decode(e["encoded_envelope_b64"]))["payload"]["selected"]!=list(row["selected"]) for e in elections):raise ValueError("utility peer election roster")
        if a.sync_utility!="peer" and contexts:raise ValueError("unexpected peer context")
        due=due_clients(ids,last,r,a.sync_max_skip)
        selected,info=choose(a,ids,last,due,r,version,visits,view,consumed,schema)
        if list(due)!=list(dec["due"]) or [i for i in due if i not in selected]!=dec["deadline_cap_violations"]:raise ValueError("utility deadline mismatch")
        if list(selected)!=list(row["selected"]) or list(selected)!=list(dec["selected"]) or json.dumps(info,sort_keys=True)!=json.dumps(dec["utility"],sort_keys=True):raise ValueError("utility decision mismatch")
        if dec["utility_used_before_admission"]!=consumed+before or dec["utility_cap"]!=ceiling(a,schema):raise ValueError("utility budget evidence")
        consumed+=sum(e["serialized_bytes"] for e in re)
        if ceiling(a,schema) is not None and consumed>ceiling(a,schema):raise ValueError("utility cumulative cap")
        for i in selected:last[i]=r;visits[i]+=1
        version+=bool(selected)
        if row["version"]!=version or list(dec["complete"])!=list(selected) or {int(i):v for i,v in row["last_complete_sync"].items()}!=last:raise ValueError("utility completion")
    return True
