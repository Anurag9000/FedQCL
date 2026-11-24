"""Research execution adapter for the native FEDQCL client.
Sequential local workers deliberately avoid the legacy shared-RNG thread race.
"""
import copy
from dataclasses import replace
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
from flcore.clients.clientavg_tl import clientAVG
from flcore.trainmodel.resnet import ResNet18_dpp
from utils.data_utils import read_client_data
from .protocol import Controller

def seed_for(seed,*keys):
    return int.from_bytes(hashlib.sha256(repr((seed,keys)).encode()).digest()[:4],"little")

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ResearchClient(clientAVG):
    def __init__(self,args,**kwargs):
        self.research_seed=args.seed
        self.research_mode=True
        self.research_args=args
        self.replay_state={}
        self.replay_samples=0
        self.optimizer_steps=0
        self.current_samples=0
        self.training_counts=[]
        super().__init__(args,**kwargs)

    def load_train_data(self,batch_size=None,round=0):
        # Client base initializes a list of handles. The actual shard is loaded
        # only at its lawful reveal round.
        return None

    def load_test_data(self,batch_size=None,round=0):
        return None

    def reveal(self,task,r):
        a=self.research_args
        if self.task_trainloaders[task] is None:
            path=os.path.join(self.dataset,f"task_{task}")
            self.task_trainloaders[task]=read_client_data(path,self.id,True,self.few_shot)
        current=self.task_trainloaders[task]
        if isinstance(current,DataLoader):
            current=current.dataset
        if a.sync_windowed:
            path=Path('../dataset')/self.dataset/f"task_{task}"/"train"/f"{self.id}.npz"
            with np.load(path,allow_pickle=True) as z:
                data=z["data"].item()
            if "window" not in data:
                raise ValueError("windowed run requires generated window IDs")
            current=[current[j] for j,w in enumerate(data["window"]) if int(w)==r%self.round_per_task]
        return current

    def prepare_training(self,task,r,dataset):
        a=self.research_args
        g=torch.Generator().manual_seed(seed_for(a.seed,"train",self.id,task,r))
        if a.sync_balanced_sampler:
            labels=np.array([int(y) for _,y in dataset])
            counts=np.bincount(labels)
            weights=[1/counts[y] for y in labels]
            sampler=torch.utils.data.WeightedRandomSampler(weights,len(weights),replacement=True,generator=g)
            loader=DataLoader(dataset,batch_size=self.batch_size,sampler=sampler,drop_last=True,generator=g)
        else:
            loader=DataLoader(dataset,batch_size=self.batch_size,shuffle=True,drop_last=True,generator=g)
        if len(loader)==0:
            raise ValueError("empty training workload")
        self.task_trainloaders[task]=loader
        return loader

    def save_task_memory_from_loader(self,client_id,task_id,trainloader,memory_size):
        if memory_size:
            super().save_task_memory_from_loader(client_id,task_id,trainloader,memory_size)
        else:
            self.task_memories={}
            self.task_memoryloaders={t:None for t in range(self.num_tasks)}
        self.replay_state.clear()
        self._mem_iterators.clear()

    def _get_next_from_memloader(self,k):
        mem=self.task_memories.get(k,[])
        if not mem:
            return None,None
        state=self.replay_state.get(k)
        if state is None or state["cursor"]>=len(state["order"]):
            epoch=0 if state is None else state["epoch"]+1
            g=torch.Generator().manual_seed(seed_for(self.research_seed,"replay",self.id,k,epoch))
            state={"order":torch.randperm(len(mem),generator=g).tolist(),"cursor":0,"epoch":epoch}
            self.replay_state[k]=state
        start=state["cursor"]
        end=min(start+self.memory_batch_size,len(mem))
        state["cursor"]=end
        self.replay_samples+=end-start
        return default_collate([mem[j] for j in state["order"][start:end]])

    def train_round(self,r,task,dataset):
        loader=self.prepare_training(task,r,dataset)
        self.replay_samples=0
        self.optimizer_steps=len(loader)*self.epochs
        self.current_samples=self.optimizer_steps*self.batch_size
        self.training_counts=[0]*self.classes_per_task
        if not self.research_args.sync_er:
            self.train(r)
        else:
            seed_all(seed_for(self.research_seed,"er",self.id,r))
            self.model.train()
            for _ in range(self.epochs):
                for x,y in loader:
                    x,y=x.to(self.device),y.to(self.device)%self.classes_per_task
                    self.record_training_labels(y)
                    self.optimizer.zero_grad()
                    loss=self.loss(self.model(x,task),y)
                    for k in range(task):
                        mx,my=self._get_next_from_memloader(k)
                        if mx is not None:
                            loss=loss+self.loss(self.model(mx.to(self.device),k),my.to(self.device)%self.classes_per_task)
                    loss.backward()
                    self.optimizer.step()
            if r>0 and (r+1)%self.round_per_task==0:
                self.save_task_memory_from_loader(self.id,task,loader,self.memory_size)
        # Static raw shard stays resident; windowed runs re-read raw task data
        # at next reveal rather than interpreting a previous window as the shard.
        self.task_trainloaders[task]=None if self.research_args.sync_windowed else loader

    def record_training_labels(self,y):
        counts=torch.bincount(y.detach().cpu(),minlength=self.classes_per_task).tolist()
        self.training_counts=[a+b for a,b in zip(self.training_counts,counts)]

    def capture(self,task,r,dataset):
        seed_all(seed_for(self.research_seed,"capture",self.id,r))
        loader=DataLoader(dataset,batch_size=self.batch_size)
        self.save_task_memory_from_loader(self.id,task,loader,self.memory_size)

def model_bytes(model,complete):
    tensors=model.state_dict().values() if complete else model.parameters()
    return sum(t.numel()*t.element_size() for t in tensors)

def aggregate(global_model,clients,weights,complete):
    if not clients:
        return global_model
    total=sum(weights)
    if total<=0:
        raise ValueError("zero aggregation weight")
    result=copy.deepcopy(clients[0].model)
    with torch.no_grad():
        if complete:
            target=result.state_dict()
            states=[c.model.state_dict() for c in clients]
            for key,tensor in target.items():
                if tensor.is_floating_point() or tensor.is_complex():
                    tensor.zero_()
                    for state,w in zip(states,weights):
                        tensor.add_(state[key],alpha=w/total)
                else:
                    tensor.copy_(torch.stack([s[key] for s in states]).amax(0))
            result.load_state_dict(target)
        else:
            for target,params in zip(result.parameters(),zip(*(c.model.parameters() for c in clients))):
                target.zero_()
                for param,w in zip(params,weights):
                    target.add_(param,alpha=w/total)
    return result

def download(client,model,complete):
    if complete:
        client.model.load_state_dict(model.state_dict())
    else:
        client.set_parameters(model)

def evaluate(model,loaders,task,r,version,device,classes):
    model.eval()
    rows=[]
    with torch.no_grad():
        for t in range(task+1):
            counts=[0]*classes
            correct=[0]*classes
            loss=0.
            for loader in loaders[t]:
                for x,y in loader:
                    x,y=x.to(device),y.to(device)%classes
                    logits=model(x,t)
                    loss+=torch.nn.functional.cross_entropy(logits,y,reduction="sum").item()
                    n=torch.bincount(y,minlength=classes).cpu().tolist()
                    c=torch.bincount(y[logits.argmax(1)==y],minlength=classes).cpu().tolist()
                    counts=[a+b for a,b in zip(counts,n)]
                    correct=[a+b for a,b in zip(correct,c)]
            if not sum(counts):
                raise ValueError("empty evaluation task")
            rows.append({"round":r,"task":t,"version":version,"accuracy":sum(correct)/sum(counts),
                         "loss":loss/sum(counts),"count":sum(counts),"class_counts":counts,
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
            "Q":c.Q,"Q_history":c.Q_history,"theta_prev":c.theta_prev,
            "memories":c.task_memories,"replay_state":c.replay_state}

def restore_client(c,state):
    c.model.load_state_dict(state["model"])
    c.optimizer.load_state_dict(state["optimizer"])
    c.Q=state["Q"]; c.Q_history=state["Q_history"]; c.theta_prev=state["theta_prev"]
    c.task_memories=state["memories"]; c.replay_state=state["replay_state"]
    c.task_memoryloaders={t:DataLoader(c.task_memories[t],batch_size=c.memory_batch_size,shuffle=True)
                          if c.task_memories.get(t) else None for t in range(c.num_tasks)}

def save_checkpoint(path,r,model,clients,controller,metrics,rounds,signature,counterfactual_rows):
    tmp=path.with_suffix(".tmp")
    torch.save({"round":r,"server":model.state_dict(),"clients":[checkpoint_client(c) for c in clients],
                "controller":controller,"metrics":metrics,"rounds":rounds,"signature":signature,
                "counterfactual_rows":counterfactual_rows,
                "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),
                "torch_rng":torch.get_rng_state(),
                "cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},tmp)
    os.replace(tmp,path)

def run_research(args):
    a=args
    out=Path(a.sync_output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    config={k:v for k,v in vars(a).items() if k not in ("sync_resume","sync_stop_after_round")}
    dataset_root=Path(a.dataset).resolve() if Path(a.dataset).is_absolute() else (Path('../dataset')/a.dataset).resolve()
    manifest_path=dataset_root/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest and (manifest['clients']!=a.num_clients or manifest['tasks']!=a.tasks or (a.sync_windowed and manifest['windows']!=(a.global_rounds+1)//a.tasks)):
        raise ValueError('dataset manifest does not match workload')
    if a.sync_windowed and not manifest:
        raise ValueError('windowed runs require dataset manifest')
    if a.sync_counterfactual and not manifest:
        raise ValueError('counterfactual validation requires controlled dataset manifest')
    config['dataset_manifest_id']=manifest['manifest_id'] if manifest else None
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if not a.sync_resume and (out/"status.json").exists():
        raise FileExistsError("research output already exists; use --sync-resume or a fresh directory")
    atomic_json(out/"config.json",config)
    try:
        commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        commit="unknown"
    atomic_json(out/"provenance.json",{"commit":commit,"torch":torch.__version__,"signature":signature})
    atomic_json(out/"status.json",{"status":"running"})
    seed_all(a.seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    a.encoder=None
    a.model=ResNet18_dpp(num_tasks=a.tasks,num_classes_per_task=a.num_classes)
    device=torch.device(a.device)
    model=copy.deepcopy(a.model).to(device)
    from .signals import SignalExtractor
    extractor=SignalExtractor(a,copy.deepcopy(a.model) if a.sync_signal in ("feature","prototypes","feature_covariance") else None)
    clients=[]
    for i in range(a.num_clients):
        task0=os.path.join(a.dataset,"task_0")
        n=len(read_client_data(task0,i,True))
        clients.append(ResearchClient(a,id=i,train_samples=n,test_samples=0,train_slow=False,send_slow=False))
    ctl=Controller(a,range(a.num_clients))
    ctl.args=copy.copy(a)
    ctl.args.model=None
    ctl.args.encoder=None
    metrics=[]; rounds=[]; start=0
    ckpt=out/"checkpoint.pt"
    if a.sync_resume:
        state=torch.load(ckpt,map_location="cpu",weights_only=False)
        if state["signature"]!=signature:
            raise ValueError("checkpoint configuration mismatch")
        model.load_state_dict(state["server"])
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
    complete_state=a.sync_aggregation=="complete"
    payload=model_bytes(model,complete_state)
    failures=json.loads(Path(a.sync_failures).read_text()) if a.sync_failures else {}
    def fails(r,kind,i):
        return i in failures.get(str(r),{}).get(kind,[])
    try:
        for r in range(start,a.global_rounds+1):
            t=r//rpt
            tick=time.monotonic()
            datasets={c.id:c.reveal(t,r) for c in clients}
            reports={}; population={}; probe_cost={}; update_starts={}
            for c in clients:
                histogram,population[c.id]=ctl.observe(c.id,t,r,datasets[c.id])
                if histogram is not None and a.sync_windowed:
                    histogram=replace(histogram,window=f"task{t}:window{r%rpt}")
                reports[c.id],probe_cost[c.id]=extractor.before(c,datasets[c.id],histogram,t,r,ctl.local[c.id].model_version)
                if reports[c.id] is not None and a.sync_windowed:
                    reports[c.id]=replace(reports[c.id],window=f"task{t}:window{r%rpt}")
            budget_record=None
            if a.sync_byte_cap or a.sync_campaign_byte_cap:
                from .budgets import choose_budgeted
                spent=sum(x["serialized_bytes"] for x in ctl.transport.events)
                remaining=a.sync_campaign_byte_cap-spent if a.sync_campaign_byte_cap else None
                limit=a.sync_byte_cap or remaining
                if remaining is not None:
                    limit=min(limit,remaining)
                before_bytes=spent
                selected,budget_record=choose_budgeted(ctl,r,reports,payload,limit)
            else:
                selected=ctl.decide(r,reports) if not a.sync_posttrain else ()
            if a.sync_counterfactual and r in {0, rpt//2, rpt-1, rpt, 2*rpt-1, a.global_rounds}:
                for u in range(t+1):
                    if u not in validation_loaders:
                        validation_loaders[u]=[]
                        for i in range(a.num_clients):
                            with np.load(dataset_root/f"task_{u}"/"validation"/f"{i}.npz",allow_pickle=True) as z:
                                v=z["data"].item()
                            validation_loaders[u].append(DataLoader(list(zip(torch.from_numpy(v["x"]),torch.from_numpy(v["y"]))),
                                                                    batch_size=a.batch_size,shuffle=False))
                from .counterfactual import analyze
                counterfactual_rows.extend(analyze(model,clients,datasets,t,r,ctl,selected,validation_loaders,
                                                   complete_state,device,a.num_classes))
            starts={i:ctl.local[i].model_version for i in selected}
            trained=[]; outcomes=[]; uploaded=[]; accepted=[]
            for i in (range(a.num_clients) if a.sync_posttrain else selected):
                c=clients[i]
                starts[i]=ctl.local[i].model_version
                if a.sync_refresh and ctl.local[i].model_version!=ctl.model_version:
                    ctl.transport.send(r,"refresh","server",i,"MODEL_REFRESH",{"version":ctl.model_version},payload)
                    download(c,model,complete_state)
                    ctl.local[i].model_version=ctl.model_version
                    ctl.local[i].local_revision=0
                    starts[i]=ctl.model_version
                begin=time.monotonic()
                if a.sync_signal=="update":
                    update_starts[i]=extractor.snapshot(c)
                c.train_round(r,t,datasets[i])
                ctl.local[i].local_revision+=1
                trained.append(i)
                if a.sync_signal=="update":
                    reports[i],probe_cost[i]=extractor.after(c,update_starts.pop(i),t,r,starts[i])
                    if a.sync_windowed:
                        reports[i]=replace(reports[i],window=f"task{t}:window{r%rpt}")
                if not a.sync_posttrain and i in selected:
                    delivered=not fails(r,"upload",i)
                    admitted=delivered and not fails(r,"reject",i)
                    ctl.transport.send(r,"upload",i,"server","MODEL_UPLOAD",
                                       {"start_version":starts[i],"accepted":admitted},payload,delivered)
                    if delivered: uploaded.append(i)
                    if admitted: accepted.append(i)
                outcomes.append({"client":i,"optimizer_steps":c.optimizer_steps,
                                 "samples":c.current_samples,"replay_samples":c.replay_samples,"training_class_counts":c.training_counts.copy(),
                                 "seconds":time.monotonic()-begin})
            if a.sync_posttrain:
                selected=ctl.decide(r,reports)
                for i in selected:
                    delivered=not fails(r,"upload",i)
                    admitted=delivered and not fails(r,"reject",i)
                    ctl.transport.send(r,'upload',i,'server','MODEL_UPLOAD',{'start_version':starts[i],'accepted':admitted},payload,delivered)
                    if delivered: uploaded.append(i)
                    if admitted: accepted.append(i)
            accepted=tuple(accepted)
            if r==0 and len(accepted)!=a.num_clients:
                raise RuntimeError("bootstrap incomplete")
            weights=[clients[i].train_samples if not complete_state else len(datasets[i]) for i in accepted]
            if accepted:
                model=aggregate(model,[clients[i] for i in accepted],weights,complete_state)
                ctl.model_version+=1
            downloaded=[]
            for i in accepted:
                delivered=not fails(r,"download",i)
                ctl.transport.send(r,"download","server",i,"MODEL_DOWNLOAD",{"version":ctl.model_version},payload,delivered)
                if delivered:
                    download(clients[i],model,complete_state)
                    downloaded.append(i)
            ctl.complete(r,reports,accepted,downloaded,uploaded)
            if budget_record is not None:
                actual_bytes=sum(x["serialized_bytes"] for x in ctl.transport.events)-before_bytes
                budget_record["actual_bytes"]=actual_bytes
                if actual_bytes>budget_record["limit"]:
                    raise RuntimeError("byte-cap invariant violated")
            if r==0 and len(downloaded)!=a.num_clients:
                raise RuntimeError("bootstrap download incomplete")
            boundary=(r+1)%rpt==0
            if boundary and a.sync_local_capture:
                for i in set(range(a.num_clients))-set(trained):
                    clients[i].capture(t,r,datasets[i])
            for u in range(t+1):
                if u not in loaders:
                    loaders[u]=[DataLoader(read_client_data(os.path.join(a.dataset,f"task_{u}"),i,False),
                                           batch_size=a.batch_size,shuffle=False)
                                for i in range(a.num_clients)]
            metrics.extend(evaluate(model,loaders,t,r,ctl.model_version,device,a.num_classes))
            rounds.append({"round":r,"task":t,"selected":list(selected),"trained":trained,
                           "uploaded":uploaded,"accepted":list(accepted),"downloaded":downloaded,
                           "start_versions":starts,"version":ctl.model_version,"outcomes":outcomes,
                           "no_aggregation":not accepted,"population_histograms":population,"probe_cost":probe_cost,"budget":budget_record,
                           "sampled_reports":{i:None if report is None else {"id":report.id,"values":report.counts,"space":report.space,"window":report.window} for i,report in reports.items()},
                           "last_complete_sync":{i:s.last_sync for i,s in ctl.local.items()},
                           "local_revisions":{i:s.local_revision for i,s in ctl.local.items()},
                           "memory_missed":[i for i in range(a.num_clients) if boundary and t not in clients[i].task_memories],
                           "reference_missed":[i for i in range(a.num_clients) if r>0 and r%rpt==0 and i not in trained],
                           "queues":{c.id:c.Q.copy() for c in clients},
                           "memory_sizes":{c.id:{k:len(v) for k,v in c.task_memories.items()} for c in clients},
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
                save_checkpoint(ckpt,r,model,clients,ctl,metrics,rounds,signature,counterfactual_rows)
            if a.sync_stop_after_round is not None and r==a.sync_stop_after_round and r<a.global_rounds:
                atomic_json(out/"status.json",{"status":"paused","last_committed_round":r})
                return None
        final=[x for x in metrics if x["round"]==a.global_rounds]
        forgetting=[max(x["accuracy"] for x in metrics if x["task"]==t)-
                    next(x["accuracy"] for x in final if x["task"]==t) for t in range(a.tasks-1)]
        by_kind={}
        for e in ctl.transport.events:
            by_kind[e["kind"]]=by_kind.get(e["kind"],0)+e["serialized_bytes"]
        summary={"final_accuracy":sum(x["accuracy"] for x in final)/len(final),
                 "forgetting":sum(forgetting)/len(forgetting) if forgetting else 0.,
                 "worst_task_accuracy":min(x["accuracy"] for x in final),
                 "total_bytes":sum(by_kind.values()),"bytes_by_kind":by_kind,
                 "delivered_bytes":sum(x["delivered_bytes"] for x in ctl.transport.events),
                 "complete_syncs":sum(len(x["complete"]) for x in ctl.decisions),
                 "rounds":len(rounds),"accounting":f"logical tensor payload plus {a.sync_wire_format} envelope; common clock/availability free"}
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
