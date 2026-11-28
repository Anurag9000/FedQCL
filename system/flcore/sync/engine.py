"""Independent research engine for plain FedAvg + vanilla experience replay.
Sequential local workers deliberately avoid the legacy shared-RNG thread race.
"""
import copy
from dataclasses import replace,asdict
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate
from .backbone import BACKBONE, fedavg, sample_replay, replay_ce
from flcore.trainmodel.resnet import ResNet18_dpp
from utils.data_utils import read_client_data
from .protocol import Controller,observe_local

def seed_for(seed,*keys):
    return int.from_bytes(hashlib.sha256(repr((seed,keys)).encode()).digest()[:4],"little")

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def verify_shard(args,task,split,i):
    manifest=getattr(args,"dataset_manifest",None)
    if not manifest or manifest.get("schema",1)<2:return
    key=f"task_{task}/{split}/{i}.npz"
    path=Path("../dataset")/args.dataset/key
    stat=path.stat();identity=(stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    cache=getattr(args,"_verified_shards",{})
    if cache.get(key)==identity:return
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1048576),b""):digest.update(chunk)
    if digest.hexdigest()!=manifest["shards"].get(key):raise ValueError("shard fingerprint mismatch: "+key)
    cache[key]=identity;args._verified_shards=cache

class ResearchClient:
    """Independent of the legacy FEDQCL client, with no queue or reference state."""
    def __init__(self,args,**kwargs):
        self.research_seed=args.seed;self.research_args=args
        self.id=kwargs["id"];self.train_samples=kwargs["train_samples"]
        self.device=torch.device(args.device);self.dataset=args.dataset;self.few_shot=args.few_shot
        self.num_tasks=args.tasks;self.classes_per_task=args.num_classes
        self.round_per_task=(args.global_rounds+1)//args.tasks
        self.batch_size=args.batch_size;self.epochs=args.local_epochs
        self.memory_size=args.memory_size;self.memory_batch_size=args.memory_batch_size
        self.model=copy.deepcopy(args.model).to(self.device)
        self.optimizer=torch.optim.SGD(self.model.parameters(),lr=args.local_learning_rate)
        self.loss=torch.nn.CrossEntropyLoss()
        self.task_trainloaders=[None]*self.num_tasks
        self.task_memories={};self.observed_windows={}
        self.replay_samples=0;self.replay_by_task={}
        self.optimizer_steps=0;self.current_samples=0;self.training_counts=[]
        self.signal_history={};self.memory_rounds={};self.active_round=0

    def window_samples(self,task,windows):
        # Read the private shard into a transient local scope; expose only
        # the permitted windows, never retain a full future-task loader.
        path=Path("../dataset")/self.dataset/f"task_{task}"/"train"/f"{self.id}.npz"
        with np.load(path,allow_pickle=True) as z:
            data=z["data"].item()
        if "window" not in data:raise ValueError("missing window IDs")
        mask=np.isin(data["window"],sorted(windows))
        x=torch.from_numpy(data["x"][mask]).float()
        y=torch.from_numpy(data["y"][mask]).long()
        return list(zip(x,y))

    def reveal(self,task,r):
        a=self.research_args
        verify_shard(a,task,"train",self.id)
        window=r%self.round_per_task
        self.observed_windows.setdefault(task,set()).add(window)
        if a.sync_windowed:
            return self.window_samples(task,{window})
        if self.task_trainloaders[task] is None:
            path=os.path.join(self.dataset,f"task_{task}")
            self.task_trainloaders[task]=read_client_data(path,self.id,True,self.few_shot)
        current=self.task_trainloaders[task]
        return current.dataset if isinstance(current,DataLoader) else current

    def prepare_training(self,task,r,dataset):
        g=torch.Generator().manual_seed(seed_for(self.research_seed,"train",self.id,task,r))
        if not len(dataset):raise ValueError("empty training workload")
        return DataLoader(dataset,batch_size=self.batch_size,shuffle=True,drop_last=False,generator=g)

    def save_task_memory_from_loader(self,client_id,task_id,trainloader,memory_size):
        from .replay import select_memory
        rng=random.Random(seed_for(self.research_seed,"memory_profile",self.id,task_id))
        pool={**self.task_memories,task_id:list(trainloader.dataset)}
        selected=select_memory(pool,memory_size,"uniform",rng)
        self.task_memories={k:[(x.detach().cpu().clone(),torch.as_tensor(y).detach().cpu().clone()) for x,y in items]
                            for k,items in selected.items()}
        self.memory_rounds={k:v for k,v in self.memory_rounds.items() if k in self.task_memories}
        if self.task_memories.get(task_id):self.memory_rounds[task_id]=self.active_round
        if sum(map(len,self.task_memories.values()))>memory_size:raise RuntimeError("replay capacity exceeded")

    def train_round(self,r,task,dataset):
        self.active_round=r
        if getattr(self.research_args,"sync_delta_aggregation",False):
            self.delta_base={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
        loader=self.prepare_training(task,r,dataset)
        self.replay_samples=0;self.replay_by_task={}
        self.optimizer_steps=0;self.current_samples=0
        self.training_counts=[0]*self.classes_per_task
        seed_all(seed_for(self.research_seed,"er",self.id,r))
        self.model.train()
        for epoch in range(self.epochs):
            for step,(x,y) in enumerate(loader):
                x,y=x.to(self.device),y.to(self.device)%self.classes_per_task
                self.record_training_labels(y)
                self.optimizer.zero_grad()
                loss=self.loss(self.model(x,task),y)
                memory={k:v for k,v in self.task_memories.items() if k<task}
                rng=random.Random(seed_for(self.research_seed,"pooled_replay",self.id,r,epoch,step))
                chosen=sample_replay(memory,self.memory_batch_size,rng)
                grouped={}
                for k,item in chosen:grouped.setdefault(k,[]).append(item)
                groups={}
                for k,items in grouped.items():
                    mx,my=default_collate(items);groups[k]=(mx.to(self.device),my.to(self.device))
                    self.replay_by_task[k]=self.replay_by_task.get(k,0)+len(items)
                past=replay_ce(self.model,groups,lambda labels,k:labels%self.classes_per_task)
                if past is not None:loss=loss+past
                if not torch.isfinite(loss):raise FloatingPointError("nonfinite local ER objective")
                loss.backward();self.optimizer.step()
                self.replay_samples+=len(chosen);self.optimizer_steps+=1;self.current_samples+=len(y)
        if (r+1)%self.round_per_task==0:self.capture(task,r)
        if self.research_args.sync_windowed:self.task_trainloaders[task]=None

    def record_training_labels(self,y):
        counts=torch.bincount(y.detach().cpu(),minlength=self.classes_per_task).tolist()
        self.training_counts=[a+b for a,b in zip(self.training_counts,counts)]

    def capture(self,task,r,dataset=None):
        self.active_round=r
        observed=self.observed_windows.get(task,set())
        if not observed:return
        a=self.research_args
        verify_shard(a,task,"train",self.id)
        if a.sync_windowed:raw=self.window_samples(task,observed)
        else:
            raw=self.task_trainloaders[task]
            if raw is None:raw=read_client_data(os.path.join(self.dataset,f"task_{task}"),self.id,True,self.few_shot)
        self.save_task_memory_from_loader(self.id,task,DataLoader(raw,batch_size=self.batch_size),self.memory_size)
        self.task_trainloaders[task]=None
        self.observed_windows.pop(task,None)

def aggregation_weights(args,clients,datasets,ids,complete):
    if not complete:raise ValueError("research requires complete-state FedAvg")
    return [len(datasets[i]) for i in ids]

def model_bytes(model,complete):
    tensors=model.state_dict().values() if complete else model.parameters()
    return sum(t.numel()*t.element_size() for t in tensors)

def aggregate(global_model,clients,weights,complete):
    if not complete:raise ValueError("research requires complete-state FedAvg")
    if clients and getattr(getattr(clients[0],"research_args",None),"sync_delta_aggregation",False):
        from .delta_ablation import aggregate_delta
        return aggregate_delta(global_model,clients,weights)
    return fedavg(copy.deepcopy(global_model),clients,weights) if clients else global_model

def download(client,model,complete):
    if not complete:raise ValueError("research requires complete-state FedAvg")
    client.model.load_state_dict(model.state_dict())

def evaluate(model,loaders,task,r,version,device,classes):
    model.eval()
    rows=[]
    with torch.no_grad():
        for t in range(task+1):
            counts=[0]*classes
            correct=[0]*classes
            loss=0.;brier=0.;bin_n=[0]*15;bin_conf=[0.]*15;bin_correct=[0]*15
            for loader in loaders[t]:
                for x,y in loader:
                    x,y=x.to(device),y.to(device)%classes
                    logits=model(x,t)
                    loss+=torch.nn.functional.cross_entropy(logits,y,reduction="sum").item()
                    prob=logits.softmax(-1);conf,pred=prob.max(-1)
                    target=torch.nn.functional.one_hot(y,num_classes=classes)
                    brier+=(prob-target).square().sum().item()
                    for k in range(15):
                        mask=(conf>=k/15)&((conf<(k+1)/15) if k<14 else (conf<=1))
                        bin_n[k]+=int(mask.sum());bin_conf[k]+=conf[mask].sum().item();bin_correct[k]+=int((pred[mask]==y[mask]).sum())
                    n=torch.bincount(y,minlength=classes).cpu().tolist()
                    c=torch.bincount(y[logits.argmax(1)==y],minlength=classes).cpu().tolist()
                    counts=[a+b for a,b in zip(counts,n)]
                    correct=[a+b for a,b in zip(correct,c)]
            if not sum(counts):
                raise ValueError("empty evaluation task")
            rows.append({"round":r,"task":t,"version":version,"accuracy":sum(correct)/sum(counts),
                         "loss":loss/sum(counts),"brier":brier/sum(counts),
                         "ece":sum(abs(a-b) for a,b in zip(bin_conf,bin_correct))/sum(counts),
                         "count":sum(counts),"class_counts":counts,
                         "class_correct":correct})
    return rows

def atomic_json(path,obj):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False))
    os.replace(tmp,path)

def write_jsonl(path,rows):
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
    os.replace(tmp,path)

def checkpoint_client(c):
    return {"model":c.model.state_dict(),"optimizer":c.optimizer.state_dict(),
            "backbone":BACKBONE,"memories":c.task_memories,"observed_windows":c.observed_windows,
            "signal_history":c.signal_history,"memory_rounds":c.memory_rounds}

def restore_client(c,state):
    if state.get("backbone")!=BACKBONE:raise ValueError("incompatible local checkpoint backbone")
    c.model.load_state_dict(state["model"]);c.optimizer.load_state_dict(state["optimizer"])
    c.task_memories=state["memories"];c.observed_windows=state["observed_windows"]
    c.signal_history=state["signal_history"];c.memory_rounds=state["memory_rounds"]

def save_checkpoint(path,r,model,clients,controller,metrics,rounds,signature,counterfactual_rows,tier_a_anchor=None):
    tmp=path.with_suffix(".tmp")
    torch.save({"round":r,"server":model.state_dict(),"clients":[checkpoint_client(c) for c in clients],
                "controller":controller,"metrics":metrics,"rounds":rounds,"signature":signature,
                "counterfactual_rows":counterfactual_rows,"tier_a_anchor":tier_a_anchor,
                "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),
                "torch_rng":torch.get_rng_state(),
                "cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},tmp)
    os.chmod(tmp,0o600)
    os.replace(tmp,path)

def run_research(args):
    from .config import validate
    a=args
    validate(a)
    if a.sync_policy=="legacy":raise ValueError("legacy is not a research policy")
    out=Path(a.sync_output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if getattr(a,"sync_bandwidth_audit",False):out.chmod(0o700)
    config={k:v for k,v in vars(a).items() if k not in ("sync_resume","sync_stop_after_round","V","delta")}
    config["backbone"]=BACKBONE
    dataset_root=Path(a.dataset).resolve() if Path(a.dataset).is_absolute() else (Path('../dataset')/a.dataset).resolve()
    manifest_path=dataset_root/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest and manifest.get("schema",1)>=2:
        fingerprint=hashlib.sha256(json.dumps({k:v for k,v in manifest.items() if k!="manifest_id"},sort_keys=True).encode()).hexdigest()
        if fingerprint!=manifest["manifest_id"]:raise ValueError("dataset manifest fingerprint mismatch")
    if manifest and (manifest['clients']!=a.num_clients or manifest['tasks']!=a.tasks or manifest.get('classes_per_task',a.num_classes)!=a.num_classes or (a.sync_windowed and manifest['windows']!=(a.global_rounds+1)//a.tasks)):
        raise ValueError('dataset manifest does not match workload')
    if a.sync_windowed and not manifest:
        raise ValueError('windowed runs require dataset manifest')
    if a.sync_counterfactual and not manifest:
        raise ValueError('counterfactual validation requires controlled dataset manifest')
    config['dataset_manifest_id']=manifest['manifest_id'] if manifest else None
    counterfactual_schedule=None
    if a.sync_controlled_stream:
        from .ground_truth import install_bundle
        receipt=install_bundle(dataset_root,out,manifest)
        config["ground_truth_hash"]=receipt["ground_truth_hash"]
        counterfactual_schedule=set(manifest["controlled_stream"]["counterfactual_rounds"])
    integrity_manifest={k:manifest[k] for k in ("schema","manifest_id","public_probe_per_class","shards") if k in manifest} if manifest else None
    for field in ('sync_impact_model','sync_count_schedule','sync_failures','sync_byte_schedule','sync_utility_model'):
        path=getattr(a,field,None)
        if path:config[field+'_sha256']=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    try:
        config['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):config['source_commit']='unknown'
    if a.sync_eval_split=='validation' and not manifest:raise ValueError('validation evaluation requires controlled dataset')
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if not a.sync_resume and (out/"status.json").exists():
        raise FileExistsError("research output already exists; use --sync-resume or a fresh directory")
    atomic_json(out/"config.json",config)
    try:
        commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        commit="unknown"
    atomic_json(out/"provenance.json",{"commit":commit,"torch":torch.__version__,"signature":signature,"backbone":BACKBONE,**({"ground_truth_hash":config["ground_truth_hash"],"dataset_manifest_id":config["dataset_manifest_id"]} if a.sync_controlled_stream else {})})
    atomic_json(out/"status.json",{"status":"running"})
    a.dataset_manifest=integrity_manifest
    a._verified_shards={}
    seed_all(a.seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    a.encoder=None
    a.model=ResNet18_dpp(num_tasks=a.tasks,num_classes_per_task=a.num_classes)
    device=torch.device(a.device)
    model=copy.deepcopy(a.model).to(device)
    from .signals import SignalExtractor
    observation_args=copy.copy(a)
    if integrity_manifest:
        observation_args.dataset_manifest={**integrity_manifest,"shards":{k:v for k,v in integrity_manifest.get("shards",{}).items() if "/public_probe/" in k}}
    extractor=SignalExtractor(observation_args,copy.deepcopy(a.model) if a.sync_signal in ("feature","prototypes","feature_covariance","spectral","multichannel","representation_anchor","conditional_moments","tier_a_features","tier_a_prototypes") else None)
    clients=[]
    for i in range(a.num_clients):
        task0=os.path.join(a.dataset,"task_0")
        verify_shard(a,0,"train",i)
        n=0 if a.sync_windowed else len(read_client_data(task0,i,True))
        client_args=copy.copy(a)
        if integrity_manifest:
            client_args.dataset_manifest={**integrity_manifest,"shards":{k:v for k,v in integrity_manifest.get("shards",{}).items() if k.endswith(f"/train/{i}.npz") or "/public_probe/" in k}}
            client_args._verified_shards={}
        clients.append(ResearchClient(client_args,id=i,train_samples=n,test_samples=0,train_slow=False,send_slow=False))
    ctl=Controller(a,range(a.num_clients))
    byte_schedule=None
    if a.sync_byte_schedule:
        byte_schedule=json.loads(Path(a.sync_byte_schedule).read_text())
        if len(byte_schedule)!=a.global_rounds+1 or any(type(v) is not int or v<0 for v in byte_schedule):
            raise ValueError("invalid byte schedule")
    if a.sync_count_schedule:
        counts=json.loads(Path(a.sync_count_schedule).read_text())
        if len(counts)!=a.global_rounds+1 or counts[0]!=a.num_clients or any(type(k) is not int or not 0<=k<=a.num_clients for k in counts):
            raise ValueError("invalid matched-count schedule")
        ctl.count_schedule=counts
    ctl.args=copy.copy(a)
    ctl.args.model=None
    ctl.args.encoder=None
    ctl.args.dataset_manifest=None
    ctl.args._verified_shards={}
    if getattr(a,"sync_utility_model",None):
        from .utility_calibration import load_model
        ctl.args.utility_model=load_model(a.sync_utility_model)
    # Research-only allocations, manifest and oracle outputs are excluded.
    metrics=[]; rounds=[]; start=0
    ckpt=out/"checkpoint.pt"
    if a.sync_resume:
        state=torch.load(ckpt,map_location="cpu",weights_only=False)
        if state["signature"]!=signature:
            raise ValueError("checkpoint configuration mismatch")
        model.load_state_dict(state["server"])
        if a.sync_signal in ("tier_a_features","tier_a_prototypes"):
            if state.get("tier_a_anchor") is None:raise ValueError("missing learned feature anchor in checkpoint")
            extractor.anchor.load_state_dict(state["tier_a_anchor"]);extractor.tier_a_anchor_ready=True
        for c,s in zip(clients,state["clients"]):
            restore_client(c,s)
        ctl=state["controller"]; metrics=state["metrics"]; rounds=state["rounds"]
        start=state["round"]+1
        random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    loaders={}
    validation_loaders={}
    counterfactual_rows=[]
    if a.sync_resume:
        counterfactual_rows=state.get('counterfactual_rows',[])
    rpt=(a.global_rounds+1)//a.tasks
    complete_state=True
    payload=model_bytes(model,complete_state)
    if getattr(a,"sync_bandwidth_audit",False):
        from .bandwidth import model_schema
        schema=model_schema(model.state_dict())
        if schema["tensor_bytes"]!=payload:raise ValueError("model schema size mismatch")
        atomic_json(out/"model_wire_schema.json",schema)
        if start==0:
            for i in ctl.ids:
                ctl.transport.send(-1,"initialization","server",i,"MODEL_INITIAL",{"version":0,"schema":schema["id"]},payload)
                ctl.transport.send(-1,"initialization",i,"server","MODEL_ACK",{"version":0})
    failures=json.loads(Path(a.sync_failures).read_text()) if a.sync_failures else {}
    def fails(r,kind,i):
        return i in failures.get(str(r),{}).get(kind,[])
    try:
        for r in range(start,a.global_rounds+1):
            t=r//rpt
            tick=time.monotonic()
            if getattr(a,"sync_profile",False) and device.type=="cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            available=tuple(c.id for c in clients if not fails(r,"unavailable",c.id))
            if r==0 and len(available)!=a.num_clients:raise RuntimeError("bootstrap availability incomplete")
            datasets={i:clients[i].reveal(t,r) for i in available}
            pre_memory={c.id:{k:np.bincount(np.asarray([int(y)%a.num_classes for _,y in v],dtype=int),minlength=a.num_classes).tolist() for k,v in c.task_memories.items()} for c in clients} if a.sync_controlled_stream else {}
            reports={}; population={}; probe_cost={}; update_starts={}; utility_reports={}
            utility=getattr(a,"sync_utility","off")
            if getattr(a,"sync_utility_adaptive_precision",False):
                from .utility import precision_for
                precision,dim=precision_for(a,ctl.decisions)
                extractor.args.sync_utility_precision=precision;extractor.args.sync_sketch_dim=dim
                for i in available:ctl.transport.send(r,"observe","server",i,"UTILITY_PRECISION",{"precision":precision,"dimension":dim})
            for i in available:
                c=clients[i]
                if utility!="off":
                    from .utility import probe
                    population[i]=tuple(np.bincount([int(y)-t*a.num_classes for _,y in datasets[i]],minlength=a.num_classes).tolist())
                    reports[i]=None
                    utility_reports[i],probe_cost[i]=probe(extractor,c,datasets[i],t,r,ctl.local[i].model_version,utility)
                    continue
                if a.sync_controlled_stream and a.sync_policy in ("full","random","oldest"):
                    # Research-only counts, not a baseline admission signal.
                    population[i]=tuple(np.bincount([int(y)-t*a.num_classes for _,y in datasets[i]],minlength=a.num_classes).tolist())
                    reports[i]=None;probe_cost[i]={"forward_examples":0,"backward_examples":0,"seconds":0.,"quantization_l2":0.}
                    continue
                histogram,population[c.id]=observe_local(c.research_args,c.id,t,r,datasets[c.id],ctl.local[c.id].model_version)
                if histogram is not None and a.sync_windowed:
                    histogram=replace(histogram,window=f"task{t}:window{r%rpt}")
                reports[c.id],probe_cost[c.id]=extractor.before(c,datasets[c.id],histogram,t,r,ctl.local[c.id].model_version)
                if reports[c.id] is not None:
                    mode=a.sync_coverage_weight
                    importance=(len(datasets[c.id]) if mode=="samples" else
                        1+sum(len(v) for v in c.task_memories.values()) if mode=="replay" else
                        aggregation_weights(a,clients,datasets,[c.id],complete_state)[0] if mode=="aggregation" and a.sync_weighting!="processed" else 1.)
                    reports[c.id]=replace(reports[c.id],importance=max(1.,float(importance)))
                from .wire import quantize_report
                reports[c.id],probe_cost[c.id]["quantization_l2"]=quantize_report(reports[c.id],a.sync_wire_format)
                if reports[c.id] is not None and a.sync_windowed and reports[c.id].kind!="tier_a_cached_update":
                    reports[c.id]=replace(reports[c.id],window=f"task{t}:window{r%rpt}")
            if utility!="off":
                ctl.utility_reports=utility_reports;ctl.utility_schema=schema
            counterfactual_seconds=0.
            budget_record=None
            if a.sync_byte_cap or a.sync_campaign_byte_cap or byte_schedule is not None:
                from .budgets import choose_budgeted
                spent=sum(x["serialized_bytes"] for x in ctl.transport.events)
                remaining=a.sync_campaign_byte_cap-spent if a.sync_campaign_byte_cap else None
                cap=(a.sync_bootstrap_byte_cap or a.sync_byte_cap) if r==0 else a.sync_byte_cap
                limit=byte_schedule[r] if byte_schedule is not None else cap or remaining
                if remaining is not None:
                    limit=min(limit,remaining)
                before_bytes=spent
                selected,budget_record=choose_budgeted(ctl,r,reports,payload,limit,remaining if a.sync_budget_reserve else None)
            else:
                selected=ctl.decide(r,reports,available) if not a.sync_posttrain else ()
            if a.sync_counterfactual and r in (counterfactual_schedule if counterfactual_schedule is not None else {0, rpt//2, rpt-1, rpt, 2*rpt-1, a.global_rounds}):
                cf_started=time.monotonic()
                for u in range(t+1):
                    if u not in validation_loaders:
                        validation_loaders[u]=[]
                        for i in range(a.num_clients):
                            verify_shard(a,u,"validation",i)
                            with np.load(dataset_root/f"task_{u}"/"validation"/f"{i}.npz",allow_pickle=True) as z:
                                v=z["data"].item()
                            validation_loaders[u].append(DataLoader(list(zip(torch.from_numpy(v["x"]),torch.from_numpy(v["y"]))),
                                                                    batch_size=a.batch_size,shuffle=False))
                from .counterfactual import analyze
                counterfactual_rows.extend(analyze(model,clients,datasets,t,r,ctl,selected,validation_loaders,
                                                   complete_state,device,a.num_classes))
                counterfactual_seconds=time.monotonic()-cf_started
            starts={i:ctl.local[i].model_version for i in selected}
            trained=[]; outcomes=[]; uploaded=[]; accepted=[]; fallback_ids=[]
            training_roster=list(available if a.sync_posttrain else selected)
            fresh=getattr(a,"sync_fresh_start",False);pre_downloaded=[]
            for i in training_roster:
                c=clients[i]
                starts[i]=ctl.local[i].model_version
                if fresh:
                    ctl.transport.send(r,"pretrain","server",i,"MODEL_DOWNLOAD",{"version":ctl.model_version,"schema":schema["id"]},payload)
                    download(c,model,complete_state)
                    ctl.transport.send(r,"pretrain",i,"server","MODEL_ACK",{"version":ctl.model_version})
                    ctl.local[i].model_version=ctl.model_version;ctl.local[i].local_revision=0
                    starts[i]=ctl.model_version;pre_downloaded.append(i)
                if a.sync_refresh and ctl.local[i].model_version!=ctl.model_version:
                    ctl.transport.send(r,"refresh","server",i,"MODEL_REFRESH",{"version":ctl.model_version,**({"schema":schema["id"]} if a.sync_bandwidth_audit else {})},payload)
                    download(c,model,complete_state)
                    if a.sync_bandwidth_audit:ctl.transport.send(r,"refresh",i,"server","MODEL_ACK",{"version":ctl.model_version})
                    ctl.local[i].model_version=ctl.model_version
                    ctl.local[i].local_revision=0
                    starts[i]=ctl.model_version
                begin=time.monotonic()
                if a.sync_signal in ("update","tier_a_cached_update"):
                    snapshot_started=time.monotonic()
                    update_starts[i]=extractor.snapshot(c)
                    if a.sync_signal=="tier_a_cached_update":probe_cost[i]["update_cache_seconds"]=time.monotonic()-snapshot_started
                c.train_round(r,t,datasets[i])
                ctl.local[i].local_revision+=1
                trained.append(i)
                if a.sync_signal=="tier_a_cached_update":
                    from .tier_a_signals import cache_completed_update
                    cache_started=time.monotonic()
                    cache_completed_update(extractor,c,update_starts.pop(i),t,r,starts[i])
                    probe_cost[i]["update_cache_seconds"]+=time.monotonic()-cache_started
                if a.sync_signal=="update":
                    reports[i],probe_cost[i]=extractor.after(c,update_starts.pop(i),t,r,starts[i])
                    reports[i],probe_cost[i]["quantization_l2"]=quantize_report(reports[i],a.sync_wire_format)
                    if a.sync_update_weighted:
                        reports[i]=replace(reports[i],aggregation_weight=float(aggregation_weights(a,clients,datasets,[i],complete_state)[0]))
                    if a.sync_windowed:
                        reports[i]=replace(reports[i],window=f"task{t}:window{r%rpt}")
                if not a.sync_posttrain:
                    delivered=not fails(r,"upload",i)
                    admitted=delivered and not fails(r,"reject",i)
                    ctl.transport.send(r,"upload",i,"server","MODEL_UPLOAD",
                                       {"start_version":starts[i],"accepted":admitted,**({"schema":schema["id"],"sample_count":len(datasets[i]),**({"body":"delta"} if getattr(a,"sync_delta_aggregation",False) else {})} if a.sync_bandwidth_audit else {})},payload,delivered)
                    if delivered: uploaded.append(i)
                    if admitted: accepted.append(i)
                outcomes.append({"client":i,"optimizer_steps":c.optimizer_steps,
                                 "samples":c.current_samples,"replay_samples":c.replay_samples,"replay_by_task":c.replay_by_task.copy(),"training_class_counts":c.training_counts.copy(),
                                 "seconds":time.monotonic()-begin})
                if a.sync_fallback and i==training_roster[-1] and len(accepted)<len(selected):
                    # Reliable relay has delivered current reports to the server.
                    # Recruit an untrained client maximizing DIRECT coverage of
                    # the currently uncovered clients. All additional work is charged.
                    from .policies import coverage,priority
                    from .advanced import select_cached
                    from .policies import threshold_value
                    tau=threshold_value(a.sync_threshold)
                    candidates=[j for j in available if j not in training_roster]
                    if candidates:
                        view=ctl.relay_view
                        last={j:ctl.local[j].last_sync for j in available}
                        def rank(j):
                            assignment=coverage(available,tuple(accepted)+(j,),view,tau)
                            uncovered=sum(not x["covered"] for x in assignment.values())
                            unknown=view.get(j) is None
                            due=r-last[j]>a.sync_max_skip
                            return (not due,not unknown,uncovered,priority(j,last))
                        recruit=min(candidates,key=rank)
                        ctl.transport.send(r,"fallback","server",recruit,"FALLBACK_ADMISSION",{"failed_slots":len(selected)-len(accepted)})
                        training_roster.append(recruit);fallback_ids.append(recruit)
            if a.sync_posttrain:
                selected=ctl.decide(r,reports,available)
                for i in selected:
                    delivered=not fails(r,"upload",i)
                    admitted=delivered and not fails(r,"reject",i)
                    ctl.transport.send(r,'upload',i,'server','MODEL_UPLOAD',{'start_version':starts[i],'accepted':admitted},payload,delivered)
                    if delivered: uploaded.append(i)
                    if admitted: accepted.append(i)
            accepted=tuple(accepted)
            if r==0 and len(accepted)!=a.num_clients:
                raise RuntimeError("bootstrap incomplete")
            ctl.ingest_uploads(r,reports,accepted,uploaded)
            weights=aggregation_weights(a,clients,datasets,accepted,complete_state)
            representation=ctl.decisions[-1].get("representation",{"transfers":{},"unrepresented":[]})
            if getattr(a,"sync_representative_beta",0.):
                from .representative import weights as represented_weights
                weights=represented_weights(accepted,dict(zip(accepted,weights)),representation)
            if accepted:
                model=aggregate(model,[clients[i] for i in accepted],weights,complete_state)
                ctl.model_version+=1
            downloaded=pre_downloaded if fresh else []
            for i in (() if fresh else accepted):
                delivered=not fails(r,"download",i)
                ctl.transport.send(r,"download","server",i,"MODEL_DOWNLOAD",{"version":ctl.model_version,**({"schema":schema["id"]} if a.sync_bandwidth_audit else {})},payload,delivered)
                if delivered:
                    download(clients[i],model,complete_state)
                    downloaded.append(i)
                    if a.sync_bandwidth_audit:ctl.transport.send(r,"download",i,"server","MODEL_ACK",{"version":ctl.model_version})
            ctl.complete(r,reports,accepted,downloaded,uploaded,ingested=True)
            if utility!="off":
                from .utility import check_budget
                check_budget(ctl,r)
            if budget_record is not None:
                actual_bytes=sum(x["serialized_bytes"] for x in ctl.transport.events)-before_bytes
                budget_record["actual_bytes"]=actual_bytes
                if actual_bytes>budget_record["limit"]:
                    raise RuntimeError("byte-cap invariant violated")
            if r==0 and len(downloaded)!=a.num_clients:
                raise RuntimeError("bootstrap download incomplete")
            if r==0 and a.sync_signal in ("tier_a_features","tier_a_prototypes"):
                extractor.anchor.load_state_dict(model.state_dict());extractor.anchor.eval();extractor.tier_a_anchor_ready=True
            boundary=(r+1)%rpt==0
            if boundary:
                for i in set(range(a.num_clients))-set(trained):
                    clients[i].capture(t,r)
            for u in range(t+1):
                if u not in loaders:
                    if a.sync_eval_split=="validation":
                        loaders[u]=[]
                        for i in range(a.num_clients):
                            verify_shard(a,u,"validation",i)
                            with np.load(dataset_root/f"task_{u}"/"validation"/f"{i}.npz",allow_pickle=True) as z:
                                v=z["data"].item()
                            loaders[u].append(DataLoader(list(zip(torch.from_numpy(v["x"]),torch.from_numpy(v["y"]))),batch_size=a.batch_size,shuffle=False))
                    else:
                        for i in range(a.num_clients):verify_shard(a,u,"test",i)
                        loaders[u]=[DataLoader(read_client_data(os.path.join(a.dataset,f"task_{u}"),i,False),
                                               batch_size=a.batch_size,shuffle=False) for i in range(a.num_clients)]
            metrics.extend(evaluate(model,loaders,t,r,ctl.model_version,device,a.num_classes))
            rounds.append({"round":r,"task":t,"available":list(available),"selected":list(selected),"trained":trained,"fallback":fallback_ids,
                           "uploaded":uploaded,"accepted":list(accepted),"downloaded":downloaded,
                           "start_versions":starts,"aggregation_weights":dict(zip(accepted,weights)),"version":ctl.model_version,"outcomes":outcomes,
                           "no_aggregation":not accepted,"population_histograms":population,"probe_cost":probe_cost,"budget":budget_record,
                           "sampled_reports":{i:None if report is None else {"id":report.id,"values":report.counts,"space":report.space,"window":report.window,**(asdict(report) if a.sync_audit_references else {})} for i,report in reports.items()},
                           "fresh_start":fresh,"delta_aggregation":getattr(a,"sync_delta_aggregation",False),"utility_reports":utility_reports,"representation":representation,
                           "observation_window":r%rpt if a.sync_windowed else None,
                           "counterfactual_seconds":counterfactual_seconds,
                           "pre_memory_class_counts":pre_memory,
                           "supplied_samples":{i:len(datasets[i]) for i in available},
                           "last_complete_sync":{i:s.last_sync for i,s in ctl.local.items()},
                           "local_revisions":{i:s.local_revision for i,s in ctl.local.items()},
                           "memory_missed":[i for i in range(a.num_clients) if boundary and t not in clients[i].task_memories],
                           "memory_sizes":{c.id:{k:len(v) for k,v in c.task_memories.items()} for c in clients},
                           "memory_class_counts":{c.id:{k:np.bincount(np.asarray([int(y)%a.num_classes for _,y in v],dtype=int),minlength=a.num_classes).tolist() for k,v in c.task_memories.items()} for c in clients},
                           "model_ages":{i:ctl.model_version-s.model_version for i,s in ctl.local.items()},
                           "seconds":time.monotonic()-tick})
            write_jsonl(out/"rounds.jsonl",rounds)
            write_jsonl(out/"decisions.jsonl",ctl.decisions)
            write_jsonl(out/"communication_events.jsonl",ctl.transport.events)
            write_jsonl(out/"server_metrics.jsonl",metrics)
            write_jsonl(out/"report_arrivals.jsonl",ctl.arrivals)
            if a.sync_counterfactual:
                write_jsonl(out/'counterfactuals.jsonl',counterfactual_rows)
            if a.sync_checkpoint:
                save_checkpoint(ckpt,r,model,clients,ctl,metrics,rounds,signature,counterfactual_rows,extractor.anchor.state_dict() if a.sync_signal in ("tier_a_features","tier_a_prototypes") else None)
            if getattr(a,"sync_profile",False):
                import resource
                profile={"round":r,"task":t,"seconds":time.monotonic()-tick,
                         "process_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                         "counterfactual_seconds":counterfactual_seconds}
                if device.type=="cuda":
                    torch.cuda.synchronize(device)
                    free,total=torch.cuda.mem_get_info(device)
                    profile.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                   cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                                   cuda_free_bytes_at_end=free,cuda_total_bytes=total)
                with (out/"resource_profile.jsonl").open("a") as handle:handle.write(json.dumps(profile)+"\n")
            if a.sync_stop_after_round is not None and r==a.sync_stop_after_round and r<a.global_rounds:
                atomic_json(out/"status.json",{"status":"paused","last_committed_round":r})
                return None
        final=[x for x in metrics if x["round"]==a.global_rounds]
        forgetting=[max(x["accuracy"] for x in metrics if x["task"]==t)-
                    next(x["accuracy"] for x in final if x["task"]==t) for t in range(a.tasks-1)]
        by_kind={}
        for e in ctl.transport.events:
            by_kind[e["kind"]]=by_kind.get(e["kind"],0)+e["serialized_bytes"]
        summary={"backbone":BACKBONE,"final_accuracy":sum(x["accuracy"] for x in final)/len(final),
                 "forgetting":sum(forgetting)/len(forgetting) if forgetting else 0.,
                 "worst_task_accuracy":min(x["accuracy"] for x in final),
                 "total_bytes":sum(by_kind.values()),"bytes_by_kind":by_kind,
                 "delivered_bytes":sum(x["delivered_bytes"] for x in ctl.transport.events),
                 "complete_syncs":sum(len(x["complete"]) for x in ctl.decisions),
                 "rounds":len(rounds),"evaluation_split":a.sync_eval_split,"accounting":f"logical tensor payload plus {a.sync_wire_format} envelope; common clock/availability free"}
        from .metrics import summarize
        summary.update(summarize(metrics,rounds,ctl.decisions,ctl.transport.events,a.tasks,rpt,a.num_clients))
        if a.sync_controlled_stream:
            from .ground_truth import analyze_run
            summary.update(analyze_run(out,rounds,ctl.decisions,metrics,counterfactual_rows,ctl.transport.events))
        if a.sync_bandwidth_audit:
            from .bandwidth import audit,write_artifacts
            from .decision_audit import verify_decisions
            verify_decisions(config,rounds,ctl.decisions,ctl.transport.events)
            comm,by_round,by_client=audit(ctl.transport.events,rounds,schema,a.num_clients,a.sync_policy)
            write_artifacts(out,comm,by_round,by_client)
            if a.sync_policy in ("own","server","peer") and not a.sync_refresh and not getattr(a,"sync_fresh_start",False):
                from .matched_budget import write_plan
                write_plan(out,rounds,by_round,schema,a.seed,a.sync_max_skip,a.num_clients)
            summary.update({k:comm[k] for k in ("model_bytes","statistics_bytes","control_bytes","upload_bytes","download_bytes","peer_bytes","model_tensor_bytes","statistic_vector_bytes","full_model_payload_bytes","model_uploads","model_downloads","initial_downloads","model_refreshes","skipped_client_rounds","no_aggregation_rounds","mean_participants","participation_fraction")})
            summary["accounting"]="audited dense native tensor bytes plus serialized envelopes; initialization, controls and per-recipient deliveries included; no physical network"
        if a.sync_bandwidth_audit:
            # Only the final trained model is evaluated on the held-out split.
            # No test metric is available to admission or counterfactual selection.
            test_loaders={}
            for u in range(a.tasks):
                for i in range(a.num_clients):verify_shard(a,u,"test",i)
                test_loaders[u]=[DataLoader(read_client_data(os.path.join(a.dataset,f"task_{u}"),i,False),batch_size=a.batch_size,shuffle=False) for i in range(a.num_clients)]
            held=evaluate(model,test_loaders,a.tasks-1,a.global_rounds,ctl.model_version,device,a.num_classes)
            atomic_json(out/"heldout_test_metrics.json",{"evaluation_split":"test","ground_truth_hash":config["ground_truth_hash"],"rows":held})
            summary.update(test_final_accuracy=sum(x["accuracy"] for x in held)/len(held),test_worst_task_accuracy=min(x["accuracy"] for x in held),test_worst_class_accuracy=min(c/n for x in held for c,n in zip(x["class_correct"],x["class_counts"]) if n))
        atomic_json(out/"final_summary.json",summary)
        torch.save(model.state_dict(),out/"server.pt")
        atomic_json(out/"status.json",{"status":"completed"})
        return summary
    except BaseException as error:
        write_jsonl(out/"communication_events.jsonl",ctl.transport.events)
        write_jsonl(out/"decisions.jsonl",ctl.decisions)
        write_jsonl(out/"report_arrivals.jsonl",ctl.arrivals)
        atomic_json(out/"status.json",{"status":"failed","error":repr(error)})
        raise
