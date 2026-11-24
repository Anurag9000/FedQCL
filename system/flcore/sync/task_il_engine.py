"""Schedule-driven conventional Task-IL adapter.

Reuses the existing ResNet, private bounded memory selector, logical Transport
and atomic output helpers. No FEDQCL queues, telemetry selectors or raw-data
exchange are used. Training and communication have independent rosters.
"""
import copy
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .engine import atomic_json, write_jsonl, seed_for, seed_all, model_bytes
from .protocol import Transport
from .replay import select_memory
from .task_il_schedule import digest, make_schedule, data_plan, validate_schedule, ROOT
from ..trainmodel.resnet import ResNet18_dpp

import sys
sys.path.insert(0,str(ROOT/"scripts"))
from task_il_data import verify_manifest, read_verified, episode_path

def local_labels(y, classes):
    lookup=torch.full((100,),-1,dtype=torch.long,device=y.device)
    lookup[torch.tensor(classes,device=y.device)]=torch.arange(len(classes),device=y.device)
    mapped=lookup[y]
    if (mapped<0).any():
        raise ValueError("label outside task")
    return mapped

def clone_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

class PrivateClient:
    """Only reads its own episode shards. Replay contains private source IDs."""
    def __init__(self,i,model,c,s,root,manifest):
        self.id=i;self.model=copy.deepcopy(model);self.base=clone_state(model)
        self.version=0;self.revision=0;self.pending_steps=0
        self.c=c;self.s=s;self.root=root;self.manifest=manifest
        self.memory={};self.cache={};self.seen=set();self.revealed=set()

    def shard(self,eid):
        e=self.s["episodes"][eid]
        if e["client"]!=self.id:
            raise ValueError("cross-client training access")
        if eid not in self.cache:
            x,y,ids=read_verified(self.root,self.manifest,episode_path(e))
            self.cache[eid]=(torch.from_numpy(x),torch.from_numpy(y).long(),ids)
        return self.cache[eid]

    def replay_batch(self,task,eid,local_epoch,step,device):
        pool=self.memory[task]
        rng=random.Random(seed_for(self.c["seed"],"replay",self.id,eid,local_epoch,step,task))
        chosen=rng.sample(pool,min(len(pool),self.c["replay_batch"]))
        xs=[];ys=[]
        for (eid,sid,image),label in chosen:
            e=self.s["episodes"][eid]
            if e["client"]!=self.id or eid not in self.revealed or e["task"]!=task or label not in self.s["tasks"][task]:
                raise ValueError("replay provenance violation")
            xs.append(image);ys.append(label)
        return torch.stack(xs).to(device),torch.tensor(ys,device=device)

    def train(self,e,r,device):
        if not e["start"]<=r<=e["end"] or e["client"]!=self.id:
            raise ValueError("invalid episode reveal")
        c=self.c;task=e["task"];classes=self.s["tasks"]
        x,y,ids=self.shard(e["id"])
        self.seen.add(task);self.revealed.add(e["id"])
        self.model.to(device).train()
        optimizer=torch.optim.SGD(self.model.parameters(),lr=c["lr"])
        steps=0;samples=0;replay=0;loss_sum=0.;class_counts={}
        # Preserve existing ER objective: one mean loss per remembered task.
        # Epoch keys depend on the episode-local epoch, not sync frequency.
        for epoch in range(c["epochs"]):
            local_epoch=(r-e["start"])*c["epochs"]+epoch
            seed_all(seed_for(c["seed"],"task_il",self.id,e["id"],local_epoch))
            gen=torch.Generator().manual_seed(seed_for(c["seed"],"shuffle",self.id,e["id"],local_epoch))
            loader=DataLoader(TensorDataset(x,y),batch_size=c["batch"],shuffle=True,generator=gen,drop_last=False)
            for step,(bx,by) in enumerate(loader):
                bx,by=bx.to(device),by.to(device)
                optimizer.zero_grad()
                current=F.cross_entropy(self.model(bx,task),local_labels(by,classes[task]))
                loss=current
                for k in sorted(self.memory):
                    if not self.memory[k]:continue
                    mx,my=self.replay_batch(k,e["id"],local_epoch,step,device)
                    loss=loss+F.cross_entropy(self.model(mx,k),local_labels(my,classes[k]))
                    replay+=len(my)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite local objective")
                loss.backward();optimizer.step()
                steps+=1;samples+=len(by);loss_sum+=float(current.detach())*len(by)
                for label in by.detach().cpu().tolist():
                    class_counts[label]=class_counts.get(label,0)+1
        self.model.cpu()
        self.revision+=1;self.pending_steps+=steps
        if r==e["end"]:
            # Existing bounded task-balanced uniform selector, retaining earlier
            # samples of recurring tasks rather than overwriting their pool.
            pool={k:list(v) for k,v in self.memory.items()}
            pool.setdefault(task,[]).extend(((e["id"],int(sid),x[j]),int(y[j])) for j,sid in enumerate(ids))
            rng=random.Random(seed_for(c["seed"],"admission",self.id,e["id"]))
            selected=select_memory(pool,c["memory"],"uniform",rng)
            # Clone admitted tensors, not views of the entire source storage.
            self.memory={k:[((eid,sid,image.clone()),label) for (eid,sid,image),label in items]
                         for k,items in selected.items()}
            retained=[key[1] for items in self.memory.values() for key,_ in items]
            if len(retained)!=len(set(retained)) or len(retained)>c["memory"]:
                raise ValueError("invalid replay inventory")
            self.cache.pop(e["id"],None)
        return dict(client=self.id,episode=e["id"],task=task,epochs=c["epochs"],optimizer_steps=steps,
                    samples=samples,replay_samples=replay,training_class_counts=class_counts,
                    current_train_loss=loss_sum/samples,pending_steps=self.pending_steps,
                    shard_size=len(ids),private_memory_count=sum(map(len,self.memory.values())),
                    memory_source_ids=sorted(key[1] for items in self.memory.values() for key,_ in items) if r==e["end"] else None)

    def download(self,model,version):
        self.model.load_state_dict(model.state_dict())
        self.base=clone_state(model)
        self.version=version;self.revision=0;self.pending_steps=0

    def state(self):
        return dict(model=clone_state(self.model),base=self.base,version=self.version,
                    revision=self.revision,pending_steps=self.pending_steps,
                    memory=self.memory,seen=sorted(self.seen),revealed=sorted(self.revealed))

    def restore(self,state):
        self.model.load_state_dict(state["model"]);self.base=state["base"]
        self.version=state["version"];self.revision=state["revision"]
        self.pending_steps=state["pending_steps"];self.memory=state["memory"]
        self.seen=set(state["seen"]);self.revealed=set(state["revealed"]);self.cache.clear()

def aggregate_buffered(server,clients,weights,version):
    """FedBuff-inspired round-buffered update; not vanilla FedAvg.

    Fixed inverse-sqrt version-lag scaling of accumulated parameter deltas.
    Current shard sample weighting, server LR 1. Float buffers are averaged
    directly (not stale deltas), integer buffers use max as in complete-state
    aggregation. All configurations use precisely the same rule.
    """
    if not clients:return server
    if len(clients)!=len(weights) or any(w<=0 for w in weights):
        raise ValueError("invalid aggregation weights")
    total=sum(weights)
    states=[c.model.state_dict() for c in clients]
    old=server.state_dict()
    params=set(dict(server.named_parameters()))
    new={}
    with torch.no_grad():
        for key,value in old.items():
            if key in params:
                tensor=value.clone()
                for c,w,state in zip(clients,weights,states):
                    lag=version-c.version
                    if lag<0:raise ValueError("future client version")
                    tensor.add_(state[key]-c.base[key],alpha=(w/total)/(1+lag)**0.5)
            elif value.is_floating_point() or value.is_complex():
                tensor=torch.zeros_like(value)
                for w,state in zip(weights,states):
                    tensor.add_(state[key],alpha=w/total)
            else:
                tensor=torch.stack([state[key] for state in states]).amax(0)
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise FloatingPointError("nonfinite server model")
            new[key]=tensor
        server.load_state_dict(new)
    return server

class Evaluator:
    def __init__(self,root,manifest,s,c,device):
        x,y,_=read_verified(root,manifest,f'eval/{c["split"]}.npz')
        self.x=torch.from_numpy(x);self.y=torch.from_numpy(y).long()
        self.tasks=s["tasks"];self.device=device;self.batch=max(1,c["batch"])
        self.split=c["split"]
        self.indices=[torch.where(torch.isin(self.y,torch.tensor(classes)))[0] for classes in self.tasks]

    def evaluate(self,model,tasks,r,version):
        rows=[]
        model.to(self.device).eval()
        try:
            with torch.no_grad():
                for task in sorted(set(tasks)):
                    classes=self.tasks[task];idx=self.indices[task];width=len(classes)
                    if len(idx)==0:raise ValueError("empty evaluation task")
                    counts=torch.zeros(width,dtype=torch.long);correct=counts.clone();loss=0.
                    for chunk in idx.split(self.batch):
                        x=self.x[chunk].to(self.device);y=local_labels(self.y[chunk].to(self.device),classes)
                        logits=model(x,task);pred=logits.argmax(1)
                        loss+=float(F.cross_entropy(logits,y,reduction="sum"))
                        counts+=torch.bincount(y,minlength=width).cpu()
                        correct+=torch.bincount(y[pred==y],minlength=width).cpu()
                    n=int(counts.sum());accuracy=int(correct.sum())/n
                    rows.append(dict(round=r,task=task,version=version,evaluation_split=self.split,
                        accuracy=accuracy,loss=loss/n,count=n,class_ids=classes,
                        class_counts=counts.tolist(),class_correct=correct.tolist(),
                        chance_accuracy=1/width,chance_adjusted_accuracy=(accuracy-1/width)/(1-1/width) if width>1 else None))
        finally:model.cpu()
        return rows

def avg(values):
    return sum(values)/len(values) if values else None

def round_summary(rows,current,peaks):
    by_task={x["task"]:x for x in rows}
    old=set(by_task)-set(current)
    past=[by_task[t] for t in sorted(old)]
    now=[by_task[t] for t in sorted(current)]
    for row in rows:
        t=row["task"]
        row["forgetting"]=max(0.,peaks.get(t,row["accuracy"])-row["accuracy"])
        peaks[t]=max(peaks.get(t,row["accuracy"]),row["accuracy"])
    return dict(global_accuracy=avg([x["accuracy"] for x in rows]),
        global_micro_accuracy=sum(x["accuracy"]*x["count"] for x in rows)/sum(x["count"] for x in rows),
        current_task_accuracy=avg([x["accuracy"] for x in now]),current_task_loss=avg([x["loss"] for x in now]),
        previous_task_accuracy=avg([x["accuracy"] for x in past]),
        forgetting=avg([x["forgetting"] for x in past]),current_tasks=sorted(current),
        historical_tasks=sorted(old),chance_adjusted_accuracy=avg([x["chance_adjusted_accuracy"] for x in rows if x["chance_adjusted_accuracy"] is not None]))

def run(c,s,root,out,device,resume=False,stop=None):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    validate_schedule(c,s)
    manifest=verify_manifest(root,data_plan(c,s))
    receipt=json.loads((root/"validation_receipt.json").read_text())
    if not receipt["valid"] or receipt["manifest_id"]!=manifest["manifest_id"]:
        raise ValueError("dataset not validated")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    signature=digest(dict(config=c,schedule=s,manifest=manifest["manifest_id"],commit=commit,device=str(device)))
    if (out/"status.json").exists() and not resume:raise FileExistsError("output already exists")
    atomic_json(out/"config.json",c);atomic_json(out/"schedule.json",s)
    atomic_json(out/"provenance.json",dict(commit=commit,signature=signature,manifest_id=manifest["manifest_id"],torch=str(torch.__version__)))
    atomic_json(out/"status.json",dict(status="running"))
    seed_all(c["seed"]);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    server=ResNet18_dpp(len(s["tasks"]),c["width"],task_widths=[len(t) for t in s["tasks"]])
    clients=[PrivateClient(i,server,c,s,root,manifest) for i in range(c["clients"])]
    transport=Transport("binary");version=0;start=0
    metrics=[];rounds=[];diagnostics=[];peaks={};seen=set()
    payload=model_bytes(server,True)
    ckpt=out/"checkpoint.pt"
    if resume:
        state=torch.load(ckpt,map_location="cpu",weights_only=False)
        if state["signature"]!=signature:raise ValueError("checkpoint signature mismatch")
        server.load_state_dict(state["server"])
        for client,saved in zip(clients,state["clients"]):client.restore(saved)
        version=state["version"];start=state["round"]+1
        metrics=state["metrics"];rounds=state["rounds"];diagnostics=state["diagnostics"]
        peaks=state["peaks"];seen=set(state["seen"]);transport.events=state["events"]
    else:
        for i in range(c["clients"]):
            transport.send(-1,"initialization","server",i,"MODEL_INITIAL",{"version":0},payload)
    evaluator=Evaluator(root,manifest,s,c,device)
    def record(model,tasks,r,ver,phase,client=None,episode=None):
        for row in evaluator.evaluate(model,tasks,r,ver):
            diagnostics.append({**row,"phase":phase,"client":client,"episode":episode})
    try:
        for r in range(start,s["rounds"]):
            active={int(i):s["episodes"][eid] for i,eid in s["active"][r].items()}
            selected=s["selected"][r];current={e["task"] for e in active.values()}
            starting=[e for e in active.values() if e["start"]==r]
            if c["recurrence"] or c["delay"] is not None:
                for e in starting:
                    # Before first or repeated local exposure; server and recipient
                    # use the same held-out task. No training examples are evaluated.
                    if sum(q["task"]==e["task"] for q in s["episodes"])>1:
                        record(server,[e["task"]],r,version,"before_exposure_server",e["client"],e["id"])
                        record(clients[e["client"]].model,[e["task"]],r,clients[e["client"]].version,"before_exposure_client",e["client"],e["id"])
            if c["outage"] and clients[0].seen:
                record(clients[0].model,clients[0].seen,r,clients[0].version,"before_local",0)
            seen.update(current);outcomes=[]
            before_versions={i:client.version for i,client in enumerate(clients)}
            for i,e in sorted(active.items()):
                outcomes.append(clients[i].train(e,r,device))
            if c["outage"]:
                record(clients[0].model,clients[0].seen,r,clients[0].version,"after_local_before_sync",0)
            for e in active.values():
                if (c["recurrence"] or c["delay"] is not None) and e["end"]==r and sum(q["task"]==e["task"] for q in s["episodes"])>1:
                    record(clients[e["client"]].model,[e["task"]],r,clients[e["client"]].version,"after_exposure_before_sync",e["client"],e["id"])
            # Predeclared public schedule: charge directed selection commands and
            # full model messages only to connected participants. Offline clients
            # receive no server traffic. No telemetry or raw sample payloads.
            weights=[]
            outcome_by_client={o["client"]:o for o in outcomes}
            for i in selected:
                e=active[i]
                transport.send(r,"admission","server",i,"SYNC_REQUEST",{"round":r,"version":version})
                transport.send(r,"upload",i,"server","MODEL_UPLOAD",
                    {"base_version":clients[i].version,"pending_steps":clients[i].pending_steps,"episode":e["id"]},payload)
                weights.append(outcome_by_client[i]["shard_size"])
            if selected:
                aggregate_buffered(server,[clients[i] for i in selected],weights,version)
                version+=1
            for i in selected:
                transport.send(r,"download","server",i,"MODEL_DOWNLOAD",{"version":version},payload)
                clients[i].download(server,version)
            if c["outage"]:
                record(clients[0].model,clients[0].seen,r,clients[0].version,"after_sync",0)
            for e in active.values():
                if (c["recurrence"] or c["delay"] is not None) and e["end"]==r and sum(q["task"]==e["task"] for q in s["episodes"])>1:
                    record(server,[e["task"]],r,version,"after_exposure_server",e["client"],e["id"])
                    record(clients[e["client"]].model,[e["task"]],r,clients[e["client"]].version,"after_exposure_client",e["client"],e["id"])
            rows=evaluator.evaluate(server,seen,r,version)
            summary=round_summary(rows,current,peaks);metrics.extend(rows)
            events=[event for event in transport.events if event["round"]==r]
            by_kind={k:sum(e["serialized_bytes"] for e in events if e["kind"]==k) for k in ("MODEL_UPLOAD","MODEL_DOWNLOAD","SYNC_REQUEST")}
            cumulative=lambda key: (rounds[-1].get("cumulative_"+key,0) if rounds else 0)+sum(o[key] for o in outcomes)
            rounds.append(dict(round=r,version=version,active=active,trained=sorted(active),selected=selected,
                cumulative_samples=cumulative("samples"),cumulative_replay_samples=cumulative("replay_samples"),
                cumulative_optimizer_steps=cumulative("optimizer_steps"),cumulative_epochs=cumulative("epochs"),
                unique_seen_classes=len({label for t in seen for label in s["tasks"][t]}),
                uploaded=selected,accepted=selected,downloaded=selected,outcomes=outcomes,
                start_versions=before_versions,upload_lag={i:version-int(bool(selected))-before_versions[i] for i in selected},
                model_ages={i:version-client.version for i,client in enumerate(clients)},
                aggregation_weights=dict(zip(selected,weights)),uploads=len(selected),downloads=len(selected),
                upload_bytes=by_kind["MODEL_UPLOAD"],download_bytes=by_kind["MODEL_DOWNLOAD"],
                metadata_bytes=by_kind["SYNC_REQUEST"],upload_tensor_bytes=payload*len(selected),
                download_tensor_bytes=payload*len(selected),
                model_envelope_bytes=by_kind["MODEL_UPLOAD"]+by_kind["MODEL_DOWNLOAD"]-2*payload*len(selected),
                round_bytes=sum(by_kind.values()),
                cumulative_bytes=sum(e["serialized_bytes"] for e in transport.events),**summary))
            for name,values in (("rounds",rounds),("server_metrics",metrics),("communication_events",transport.events),("client_metrics",diagnostics)):
                write_jsonl(out/(name+".jsonl"),values)
            state=dict(signature=signature,round=r,server=clone_state(server),clients=[client.state() for client in clients],
                       version=version,metrics=metrics,rounds=rounds,diagnostics=diagnostics,events=transport.events,
                       peaks=peaks,seen=sorted(seen))
            tmp=ckpt.with_suffix(".tmp");torch.save(state,tmp);os.replace(tmp,ckpt)
            if stop is not None and r==stop and r<s["rounds"]-1:
                atomic_json(out/"status.json",dict(status="paused",last_committed_round=r));return
        final=rounds[-1];last=metrics[-len(seen):]
        # Final forgetting includes all tasks except those active in final round.
        # BWT uses each task's first completed episode as acquisition reference.
        lookup={(x["round"],x["task"]):x["accuracy"] for x in metrics}
        first_end={t:min(e["end"] for e in s["episodes"] if e["task"]==t) for t in seen}
        bwt=[x["accuracy"]-lookup[first_end[x["task"]],x["task"]] for x in last if x["task"] not in final["current_tasks"]]
        by_kind={}
        for e in transport.events:by_kind[e["kind"]]=by_kind.get(e["kind"],0)+e["serialized_bytes"]
        result=dict(rounds=len(rounds),evaluation_split=c["split"],final_accuracy=final["global_accuracy"],
            final_micro_accuracy=final["global_micro_accuracy"],forgetting=final["forgetting"],backward_transfer=avg(bwt),
            final_current_task_accuracy=final["current_task_accuracy"],final_current_task_loss=final["current_task_loss"],
            final_previous_task_accuracy=final["previous_task_accuracy"],chance_adjusted_accuracy=final["chance_adjusted_accuracy"],
            total_bytes=sum(by_kind.values()),bytes_by_kind=by_kind,uploads=sum(x["uploads"] for x in rounds),
            downloads=sum(x["downloads"] for x in rounds),initial_downloads=c["clients"],model_payload_bytes=payload,
            optimizer_steps=sum(o["optimizer_steps"] for x in rounds for o in x["outcomes"]),
            processed_samples=sum(o["samples"] for x in rounds for o in x["outcomes"]),
            replay_samples=sum(o["replay_samples"] for x in rounds for o in x["outcomes"]),
            terminal_flush=False,aggregation="FedBuff-inspired round-buffered inverse-square-root stale delta; current shard weights",
            accounting="logical tensor payload plus serialized envelopes, predeclared schedule local; no physical network measurement")
        result["final_"+c["split"]+"_accuracy"]=result["final_accuracy"]
        atomic_json(out/"final_summary.json",result);torch.save(server.state_dict(),out/"server.pt")
        atomic_json(out/"status.json",dict(status="completed"))
        return result
    except BaseException as error:
        atomic_json(out/"status.json",dict(status="failed",error=repr(error)))
        raise
